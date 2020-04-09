#!/usr/bin/env python3

import os
import re
import numpy as np
import bottleneck as bn
import pandas as pd
from scipy.special import gamma
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import AgglomerativeClustering


DOT_HEADER = 'digraph G {\n' \
    'node [width=0.75 fillcolor="#a6cee3", style=filled, fontcolor=black, ' \
    'shape=circle, fontsize=20, fontname="arial", fixedsize=True];\n' \

DOT_CELLS = 'node [width=0.5, fillcolor="#e8bdc9", fontcolor=black, ' \
    'style=filled, shape=square, fontsize=8, fontname="arial", fixedsize=True];\n'


# ------------------------------------------------------------------------------
# Mathematical functions
# ------------------------------------------------------------------------------

def check_beta_params(mean, var):
    ''' Check if parameters can be used for a beta function

    Arguments:
        mean (float): Beta function mean
        var (float): Beta function variance

    Returns:
        bool: True if parameters can be used, False otherwise

    '''
    return mean > .5 * (1 - (1 - 4 * var) ** .5)


# ------------------------------------------------------------------------------
# Evaluation functions
# ------------------------------------------------------------------------------

def get_v_measure(pred_clusters, true_clusters, out_file=''):
    score = v_measure_score(true_clusters, pred_clusters)
    if out_file:
        _write_to_file(out_file, score)
    return score


def get_ARI(pred_clusters, true_clusters, out_file=''):
    score = adjusted_rand_score(true_clusters,pred_clusters)
    if out_file:
        _write_to_file(out_file, score)
    return score


def get_hamming_dist(df_pred, df_true):
    np.count_nonzero(df_pred.round() != df_true.T)
    if df_true.shape != df_pred.shape:
        score = np.count_nonzero(df_pred.round() != df_true.T)
    else:
        score = np.count_nonzero(df_pred.round() != df_true)
    return score


def _get_genotype_all(df_in, assign):
    df_out = pd.DataFrame(
        index=np.arange(df_in.shape[0]), columns=np.arange(len(assign))
    )
    if df_in.shape == df_out.shape:
        return df_in

    for cell_id, cl_id in enumerate(assign):
        if isinstance(cl_id, tuple):
            df_out[cell_id] = df_in[list(cl_id)].max(axis=1)
        else:
            df_out[cell_id] = df_in[cl_id]
    return df_out


def get_dist(assignments):
    steps, cells = assignments.shape
    dist = np.zeros(np.arange(cells).sum(), dtype=np.int32)
    # Sum up Hamming distance between cells for each spoterior sample
    for assign in assignments:
        dist += pdist(np.stack([assign, assign]).T, 'hamming').astype(np.int32)
    # Return mean posterior cellwise hamming distance
    return squareform(dist / steps)


def get_MPEAR_assignment(results, single_chains=False):
    assign = {}
    if single_chains:
        for i, result in enumerate(results):
            assignments = result['assignments'][result['burn_in']:]
            assign[i] = _get_MPEAR(assignments)
    else:
        assignments = np.concatenate(
            [i['assignments'][i['burn_in']:] for i in results]
        )
        assign[0] = _get_MPEAR(assignments)
    return assign


def _get_MPEAR(assignments):
    dist = get_dist(assignments)
    n_range = _get_MPEAR_range(assignments)
    return _iterate_MPEAR(dist, n_range)


def _get_MPEAR_range(assign, s=1):
    if len(assign.shape) > 2:
        cl_no = assign[0].shape[1]
    else:
        cl_no = [np.sum(~np.isnan(np.unique(i))) for i in assign]
    n_min = np.round(np.mean(cl_no) - s * np.std(cl_no))
    n_max = np.round(np.mean(cl_no) + s * np.std(cl_no))
    return np.arange(n_min, n_max + 1, dtype=int)


def _iterate_MPEAR(dist, n_range):
    best_MPEAR = -np.inf
    best_assignment = None
    for n in n_range:
        model = AgglomerativeClustering(
            affinity='precomputed', n_clusters=n, linkage='complete'
        ).fit(dist)
        score = _calc_MPEAR(1 - dist, model.labels_)
        if score > best_MPEAR:
            best_assignment = model.labels_
            best_MPEAR = score
    return best_assignment


def _calc_MPEAR(pi, c):
    # Fritsch, A., Ickstadt, K. (2009) - Eq. 13
    n = pi.shape[0]
    ij = np.triu_indices(n, 0)
    norm = np.math.factorial(n) / (2 * np.math.factorial(n - 2))

    I = 1 - squareform(pdist(np.stack([c, c]).T, 'hamming'))

    I_sum = I[ij].sum()
    pi_sum = pi[ij].sum()

    index = (I * pi)[ij].sum()
    expected_index = (I_sum * pi_sum) / norm
    max_index = .5 * (I_sum + pi_sum)

    return (index - expected_index) / (max_index - expected_index)


def get_mean_hierarchy_assignment(assignments, params_full):
    steps = assignments.shape[0]
    cl_no = [np.sum(~np.isnan(np.unique(i))) for i in assignments]
    n = int(np.round(np.mean(cl_no)))

    import pdb; pdb.set_trace()
    dist = get_dist(assignments)
    model = AgglomerativeClustering(
        affinity='precomputed', n_clusters=n, linkage='complete'
    ).fit(dist)
    clusters = np.unique(model.labels_)

    params = np.zeros((clusters.size, params_full[0].shape[1]))
    for i, cluster in enumerate(clusters):
        cells_cl_idx = model.labels_ == cluster
        cells = np.nonzero(cells_cl_idx)[0]
        other = np.nonzero(~cells_cl_idx)[0]
        # Paper - section 2.3: first criteria
        if cells.size == 1:
            same_cluster = np.ones(steps).astype(bool)
        else:
            same_cluster = 0 == bn.nansum(
                bn.move_std(assignments[:, cells], 2, axis=1), axis=1
            )
        # Paper - section 2.3: second criteria
        cl_id = assignments[same_cluster][:,cells[0]]
        other_cl_id = assignments[same_cluster][:,other]
        no_others = [cl_id[i] not in other_cl_id[i] \
            for i in range(same_cluster.sum())]
        # Both criteria fullfilled in at least 1 posterior sample
        if any(no_others):
            cl_ids = assignments[same_cluster][no_others][:,cells[0]]
            params[i] = bn.nanmean(
                params_full[same_cluster][no_others, cl_ids], axis=0
            )
        # If not, take posterior samples where only criteria 1 is fullfilled
        else:
            cl_ids = assignments[same_cluster][:,cells[0]]
            params[i] = bn.nanmean(params_full[same_cluster, cl_ids], axis=0)

    params_df = pd.DataFrame(params).T[model.labels_]

    import pdb; pdb.set_trace()

    return model.labels_, params_df


def get_latents_posterior(results, single_chains=False):
    latents = []
    if single_chains:
        for result in results:
            latents.append(_get_latents_posterior_chain(result))
    else:
        result =_concat_chain_results(results)
        latents.append(_get_latents_posterior_chain(result))
    return latents


def _concat_chain_results(results):
    assign = np.concatenate([i['assignments'][i['burn_in']:] for i in results])
    a = np.concatenate([i['DP_alpha'][i['burn_in']:] for i in results])
    ML = np.concatenate([i['ML'] for i in results])
    MAP = np.concatenate([i['MAP'] for i in results])
    try:
        FN = np.concatenate([i['FN'][i['burn_in']:] for i in results])
        FP = np.concatenate([i['FP'][i['burn_in']:] for i in results])
    except KeyError:
        FN = None
        FP = None
    # Fill clusters not used by all chains with zeros
    params = [i['params'][i['burn_in']:] for i in results]
    cl_max = np.max([i.shape[1] for i in params])
    for i, par_chain in enumerate(params):
        cl_diff = cl_max - par_chain.shape[1]
        params[i] = np.pad(par_chain, [(0, 0), (0, cl_diff), (0, 0)])
    par = np.concatenate(params)

    return {'assignments': assign, 'params': par, 'DP_alpha': a, 'FN': FN,
        'FP': FP, 'burn_in': 0, 'ML': ML, 'MAP': MAP}


def _get_latents_posterior_chain(result):
    burn_in = result['burn_in']
    assign, geno = get_mean_hierarchy_assignment(
        result['assignments'][burn_in:], np.array(result['params'][burn_in:])
    )
    a = _get_posterior_avg(result['DP_alpha'][burn_in:])
    try:
        FN = _get_posterior_avg(result['FN'][burn_in:])
    except (KeyError, TypeError):
        FN = None
    try:
        FP = _get_posterior_avg(result['FP'][burn_in:])
    except (KeyError, TypeError):
        FP = None

    return {'a': a, 'assignment': assign, 'genotypes': geno, 'FN': FN, 'FP': FP}


def _get_posterior_avg(data):
    return np.mean(data), np.std(data)


def get_latents_point(results, est, single_chains=False):
    latents = []
    if single_chains:
        for result in results:
            step = np.argmax(result[est][result['burn_in']:]) \
                + result['burn_in']
            latents.append(_get_latents_point_chain(result, est, step))
    else:
        scores = [np.max(result[est][result['burn_in']:]) for result in results]
        best_chain = results[np.argmax(scores)]
        burn_in = best_chain['burn_in']
        step = np.argmax(best_chain[est][burn_in:]) + burn_in
        latents.append(_get_latents_point_chain(best_chain, est, step))

    return latents


def _get_latents_point_chain(result, estimator, step):
    # DPMM conc. parameter
    if np.unique(result['DP_alpha']).size == 1:
        a = None
    else:
        a = result['DP_alpha'][step]
    # Error rates
    try:
        FP, FN = _get_errors(result, estimator)
    except KeyError:
        FP, FN = (None, None)

    assignment = result['assignments'][step].tolist()
    cl_names = np.unique(assignment)
    geno = pd.DataFrame(result['params'][step][cl_names], index=cl_names) \
        .T[assignment]

    return {'step': step, 'a': a, 'assignment': assignment, 'genotypes': geno,
        'FN': FN, 'FP': FP}


def _get_errors(results, estimator):
    max_assign = np.argmax(results[estimator][results['burn_in']:]) \
        + results['burn_in']
    return results['FP'][max_assign], results['FN'][max_assign]


def _write_to_file(file, content, attach=False):
    if attach and os.path.exists(file):
        open_flag = 'a'
    else:
        open_flag = 'w'

    with open(file, open_flag) as f:
        f.write(str(content))


def newick_to_gv(in_file, out_file=''):
    with open(in_file, 'r') as f:
        tree = f.read().strip().rstrip(';')

    edges, cells = get_edges_from_newick(tree)
    gv_tree = edges_to_gv(edges, cells)

    if out_file:
        _write_to_file(out_file, gv_tree)
    else:
        return gv_tree


def get_edges_from_newick(data):
    cells = sorted(re.findall('\w+cell\d*', data))
    for i, cell in enumerate(cells):
        data = data.replace(cell, 'C{}'.format(i))

    edges = []
    node_no = len(cells)

    while True:
        pairs = re.findall('\((C\d+):(0.\d+),(C\d+):(0.\d+)\)', data)
        if not pairs:
            break
        for i, pair in enumerate(pairs):
            n1, d1, n2, d2 = pair
            edges.append((node_no, int(n1.lstrip('C')), float(d1)))
            edges.append((node_no, int(n2.lstrip('C')), float(d2)))

            data = data.replace(
                '({}:{},{}:{})'.format(*pair), 'C{}'.format(node_no)
            )
            node_no += 1

    return edges, cells


def get_edges_from_gz(data):
        mut_edges = []
        muts = set([])
        cell_edges = []
        cells = []

        for line in data.split(';\n')[1:-1]:
            edge_nodes = re.search('(\d+)\s+->\s+(\d+)', line)
            attachment_nodes = re.search('(\d+)\s+->\s+(s\d+)', line)
            single_node = re.search('(s?\d+)$', line)

            if edge_nodes:
                n_from = int(edge_nodes.group(1))
                n_to = int(edge_nodes.group(2))
                n_from -= 1
                n_to -= 1

                if n_from != -1 and n_to != -1:
                    mut_edges.append((n_from, n_to))
                muts.update([n_from, n_to])
            if attachment_nodes:
                n_from = int(attachment_nodes.group(1))
                n_to = attachment_nodes.group(2)
                n_from -= 1
                cell_edges.append((n_from, n_to))
                cells.append(n_to)

            elif single_node:
                node = single_node.group(1)
                if node.startswith('s'):
                    cells.append(cells)
                else:
                    muts.add(int(node) - 1)

        return mut_edges, muts, cell_edges, cells


def edges_to_gv(edges, cells):
    # GraphViy Header: Node style
    out_str = DOT_HEADER

    e_length = [i[2] for i in edges]
    e_scaled = np.ceil(e_length / np.max(e_length) * (100)).astype(int)

    for i, edge in enumerate(edges):
        try:
            n_to = cells[edge[1]]
        except IndexError:
            n_to = edge[1]
        out_str += '{} -> {} [label="{}"];\n' \
            .format(edge[0], n_to, ' ' * e_scaled[i])

    out_str += '}'
    return out_str


def collapse_cells_on_tree(data_folder, out_file=''):
    tree_file = os.path.join(data_folder, 'tree.gv')

    with open(tree_file, 'r') as f:
        tree_str = f.read()
    mut_edges, muts, cell_edges, cells = get_edges_from_gz(tree_str)

    cell_edges_collapse = {}
    for mut_from, cell_to in cell_edges:
        try:
            cell_edges_collapse[mut_from].append(cell_to)
        except KeyError:
            cell_edges_collapse[mut_from] = [cell_to]

    # GraphViy Header: Node style
    out_str = DOT_HEADER
    for mut_edge in mut_edges:
        out_str += '{} -> {};\n'.format(*mut_edge)

    out_str += DOT_CELLS
    i = 0
    for mut_from, cells_to in cell_edges_collapse.items():
        size = 0.5 + len(cells_to) * 1
        out_str += '{f} -> s{t} [label="{s}", size={s}];\n' \
            .format(f=mut_from, t=i, s=size)
        i += 1
    out_str += '}'

    if not out_file:
        out_file = os.path.join(data_folder, 'tree_collapsed.gv')

    _write_to_file(out_file, out_str)

    try:
        from graphviz import render
        render('dot', 'png', out_file)
    except ImportError:
        pass


def get_lugsail_batch_means_est(data_in, steps=None):
    m = len(data_in)
    T_iL = []
    s_i = []
    n_i = []

    for data_chain, burnin_chain in data_in:
        data = data_chain[burnin_chain:steps]
        if data.size == 1:
            return np.inf
        # [chapter 2.2 in Vats and Knudson, 2018]
        n_ii = data.size
        b = int(n_ii ** (1/2)) # Batch size. Alternative: n ** (1/3)
        n_i.append(n_ii)

        chain_mean = bn.nanmean(data)
        T_iL.append(
            2 * get_tau_lugsail(b, data, chain_mean) \
            - get_tau_lugsail(b // 3, data, chain_mean)
        )
        s_i.append(bn.nanvar(data, ddof=1))

    T_L = np.mean(T_iL)
    s = np.mean(s_i)
    n = np.round(np.mean(n_i))

    sigma_L = ((n - 1) * s + T_L) / n

    # [eq. 5 in Vats and Knudson, 2018]
    R_L = np.sqrt(sigma_L / s)

    return R_L


def get_tau_lugsail(b, data, chain_mean):
    a = data.size // b # Number of batches
    batch_mean = bn.nanmean(np.reshape(data[:a * b], (a, b)), axis=1)
    return (b / (a - 1)) * bn.nansum(np.square(batch_mean - chain_mean))



def get_cutoff_lugsail(e, a=0.05):
    M = (4 * np.pi * chi2.ppf(1 - a, 1)) / (gamma(1/2)**2 * e**2)
    return np.sqrt(1 + 1 / M)


if __name__ == '__main__':
    print('Here be dragons...')