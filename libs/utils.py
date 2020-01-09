#!/usr/bin/env python3

import os
import re
import yaml
import numpy as np
import bottleneck as bn
import pandas as pd
from itertools import cycle
from scipy.special import gamma
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform, hamming
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import adjusted_rand_score, jaccard_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import AgglomerativeClustering

from collections import defaultdict


DOT_HEADER = 'digraph G {\n' \
    'node [width=0.75 fillcolor="#a6cee3", style=filled, fontcolor=black, ' \
    'shape=circle, fontsize=20, fontname="arial", fixedsize=True];\n' \

DOT_CELLS = 'node [width=0.5, fillcolor="#e8bdc9", fontcolor=black, ' \
    'style=filled, shape=square, fontsize=8, fontname="arial", fixedsize=True];\n'

COLORS = [
    '#1F78B4', '#33A02C', '#E31A1C', '#FF7F00', '#6A3D9A', # dark
    '#A6CEE3', '#B2DF8A', '#FB9A99', '#FDBF6F', '#CAB2D6', #light
    '#62A3CB', '#72BF5B', '#EF5A5A', '#FE9F37', '#9A77B8', # medium
    '#FFFF99', '#B15928', #ugly
]


# ------------------------------------------------------------------------------
# Mathematical functions
# ------------------------------------------------------------------------------
def log_beta(a, b):
    # log(B(a, b)) = log(G(a)) + log(G(b)) - log(G(a + b))
    return np.log(gamma(a)) + np.log(gamma(b)) - np.log(gamma(a + b))


def log_beta_pdf(x, a, b):
    # f(x,a,b) = gamma(a+b) / (gamma(a) * gamma(b)) * x^(a-1) * (1-x)^(b-1)
    return np.log(gamma(a + b)) - np.log(gamma(b)) - np.log(gamma(a)) \
        + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)


def log_normal_pdf(x, u, s):
    # f(x, u, s) = 1 / sqrt(2 * pi * s) * e^(-(x - u)^2 / 2*s)
    # log(f(x, u, s)) = - 0.5 * log(sqrt(2 * pi * s)) + ((x - u)^2 / 2*s) * log(e)
    return -(x - u) ** 2 / (2 * s ** 2) - 0.5 * np.log(2 * np.pi * s ** 2)


def check_beta_params(mean, var):
    return mean > .5 * (1 - (1 - 4 * var) ** .5)


# ------------------------------------------------------------------------------
# Evaluation functions
# ------------------------------------------------------------------------------

def get_v_measure(pred_clusters, true_clusters, out_file=''):
    true_clusters_slt = _replace_doublets(true_clusters.copy())
    pred_clusters_slt = _replace_doublets(pred_clusters.copy())

    score = v_measure_score(true_clusters_slt, pred_clusters_slt)

    if out_file:
        _write_to_file(out_file, score)

    return score


def _replace_doublets(arr):
    cl_all = [i for i in arr if not isinstance(i, (tuple, list))]
    dbt = {}
    for i, cl in enumerate(arr):
        if isinstance(cl, (tuple, list)):
            if not tuple(cl) in dbt:
                dbt[tuple(cl)] = np.max(cl_all) + len(dbt) + 1
            arr[i] = dbt[tuple(cl)]

    return arr


def get_ARI(pred_clusters, true_clusters, out_file=''):
    true_clusters_slt = _replace_doublets(true_clusters.copy())
    pred_clusters_slt = _replace_doublets(pred_clusters.copy())

    score = adjusted_rand_score(true_clusters_slt,pred_clusters_slt)

    if out_file:
        _write_to_file(out_file, score)

    return score


def get_hamming_dist(df_pred, df_true):
    if not isinstance(df_true, pd.DataFrame):
        df_true = pd.DataFrame(df_true)

    df_pred.columns = range(df_pred.shape[1])

    if df_true.shape != df_pred.shape:
        score = (df_pred != df_true.T).sum().sum()
    else:
        score = (df_pred != df_true).sum().sum()
    return score


def _get_genotype_all(df_in, assign):
    df_out = pd.DataFrame(
        index=np.arange(df_in.shape[0]),columns=np.arange(len(assign))
    )
    if df_in.shape == df_out.shape:
        return df_in

    for cell_id, cl_id in enumerate(assign):
        if isinstance(cl_id, tuple):
            df_out[cell_id] = df_in[list(cl_id)].max(axis=1)
        else:
            df_out[cell_id] = df_in[cl_id]
    return df_out


def get_dist(results):
    cells = results['assignments'][0].size
    steps = results['ML'].size - results['burn_in']
    dist = np.zeros(np.arange(cells).sum(), dtype=np.int32)
    # Sum up Hamming distance between cells for each spoterior sample
    for assign in results['assignments'][results['burn_in']:]:
        dist += pdist(np.stack([assign, assign]).T, 'hamming').astype(np.int32)
    # Return mean posterior cellwise hamming distance
    return squareform(dist / steps)


def get_MPEAR_assignment(results):
    dist = get_dist(results)

    if 'Z' in results:
        cl_no = results['Z'][0].shape[1]
    else:
        assignments = results['assignments'][results['burn_in']:]
        cl_no = [np.sum(~np.isnan(np.unique(i))) for i in assignments]

    s = 1
    n_min = np.round(np.mean(cl_no) - s * np.std(cl_no))
    n_max = np.round(np.mean(cl_no) + s * np.std(cl_no))
    n_range = np.arange(n_min, n_max + 1, dtype=int)

    assign =  _iterate_MPEAR(dist, n_range)

    return assign


def _iterate_MPEAR(dist, n_range):
    best_MPEAR = -np.inf
    best_assignment = None
    for n in n_range:
        model = AgglomerativeClustering(
            affinity='precomputed', n_clusters=n, linkage='complete'
        ).fit(dist)
        score = _get_MPEAR(1 - dist, model.labels_)
        if score > best_MPEAR:
            best_assignment = model.labels_
            best_MPEAR = score
    return best_assignment


def _get_MPEAR(pi, c):
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


def get_mean_hierarchy_assignment(results, thresh=0.01):
    if 'Z' in results:
        return _get_mean_hierarchy_doublet(results, thresh)
    else:
        return _get_mean_hierarchy_singlet(results, thresh)


def _get_mean_hierarchy_doublet(results, thresh=0.01):
    Z = results['Z'][results['burn_in']:]
    n = Z[0].shape[1]
    steps = Z.shape[0]
    dbt = _get_doublets(results['Z'], results['burn_in'])

    dist = get_dist(results)
    model = AgglomerativeClustering(
        affinity='precomputed', n_clusters=n, linkage='complete'
    ).fit(dist[~dbt][:,~dbt])
    clusters = np.unique(model.labels_)

    assign = np.full(results['Z'][0].shape[0], -1)
    assign[~dbt] = model.labels_
    assign_out = assign.tolist()

    params_full = np.array(results['params'][results['burn_in']:])
    params = np.zeros((clusters.size, params_full[0].shape[1]))

    for i, cluster in enumerate(clusters):
        cells = np.argwhere(assign == cluster).flatten()
        assign_full = np.where(Z[:,cells,:])

        dbt_cells = np.where(
            (np.diff(assign_full[1]) == 0) & (np.diff(assign_full[0]) == 0)
        )[0]
        dbt_cells = np.append(dbt_cells, dbt_cells + 1)

        steps = np.delete(assign_full[0], dbt_cells)
        assign_red = np.delete(assign_full[2], dbt_cells)

        assign_cl = np.split(assign_red, np.where(np.diff(steps))[0] + 1)

        params[i] = np.concatenate(
            [params_full[i, j, :] for i, j in enumerate(assign_cl)], axis=0
        ).mean(axis=0)

    for dbt_idx, dbt_cell in enumerate(np.argwhere(dbt)):
        dbt_cl = _get_closest_cl(assign, dist[dbt_cell])
        assign_out[dbt_cell[0]] = tuple(dbt_cl)
        # Doublet cluster not present as singlet cluster
        if not np.isin(dbt_cl, clusters).all():
            import pdb; pdb.set_trace()
    # Get param of each cell
    params_df = pd.DataFrame(params, index=clusters).T
    return assign_out, params_df


def _get_closest_cl(assign, dist, cl_no=2):
    # Get closest clusters ordered from 0 -> closest to -1: furthest away
    closest_cl = assign[np.argsort(dist, axis=1)][0]
    # Get ranked avg idx of closest clusters
    rkd_cl = {}
    for i, cl in enumerate(closest_cl):
        if cl == -1:
            continue
        try:
            rkd_cl[cl].append(i)
        except KeyError:
            rkd_cl[cl] = [i]
    rkd_cl_norm = {i : np.sum(j) / len(j) for i,j in rkd_cl.items()}
    # Return index of closest cells
    return np.array([
        i[0] for i in sorted(rkd_cl_norm.items(), key=lambda x: x[1])[:cl_no]
    ])


def _get_mean_hierarchy_singlet(results, thresh=0.01):
    assignments = results['assignments'][results['burn_in']:]
    cl_no = [np.sum(~np.isnan(np.unique(i))) for i in assignments]
    n = int(np.round(np.mean(cl_no)))

    dist = get_dist(results)
    model = AgglomerativeClustering(
        affinity='precomputed', n_clusters=n, linkage='complete'
    ).fit(dist)
    clusters = np.unique(model.labels_)

    params_full = np.array(results['params'][results['burn_in']:])
    params = np.zeros((clusters.size, params_full[0].shape[1]))
    for i, cluster in enumerate(clusters):
        cells = np.argwhere(model.labels_ == cluster).flatten()
        other = np.argwhere(model.labels_ != cluster).flatten()
        # Paper - section 2.3: first criteria
        if cells.size == 1:
            same_cluster = np.zeros(assignments.shape[0]).astype(bool)
        else:
            same_cluster = 0 == bn.nansum(
                bn.move_std(assignments[:, cells], 2, axis=1),
                axis=1
            )
        # Paper - section 2.3: second criteria
        no_others = ~np.isin(
            assignments[same_cluster][:,other],
            assignments[same_cluster][:,cells[0]]
        ).any(axis=1)
        # Both criteria fullfilled in at least 1 posterior sample
        if no_others.sum() > 0:
            cl_ids = assignments[same_cluster][no_others][:,cells[0]]
            params[i] = bn.nanmean(
                params_full[same_cluster][no_others, cl_ids], axis=0
            )
        # If not, take posterior samples where only criteria 1 is fullfilled
        else:
            cl_ids = assignments[same_cluster][:,cells[0]]
            params[i] = bn.nanmean(params_full[same_cluster, cl_ids], axis=0)

    params_df = pd.DataFrame(params).T[model.labels_]

    return model.labels_, params_df


def _get_doublets(Z, burn_in, thresh=0.5):
    Z_rel = Z[burn_in:]
    steps = Z_rel.shape[0]
    cl_no = Z_rel.sum(axis=0).sum(axis=1)
    cl_no_rel = (cl_no - steps) / steps
    return cl_no_rel >= thresh


def get_latents_posterior(results):
    assign, geno = get_mean_hierarchy_assignment(results, results['burn_in'])
    a = _get_posterior_avg(results['DP_alpha'][results['burn_in']:])
    try:
        d = _get_posterior_avg(results['delta'][results['burn_in']:])
    except KeyError:
        d = None

    try:
        FN = _get_posterior_avg(results['ad_error'][results['burn_in']:])
    except KeyError:
        FN = None
    try:
        FP = _get_posterior_avg(results['fd_error'][results['burn_in']:])
    except KeyError:
        FP = None

    if 'Z' in results:
        # <TODO: NB> Implement for doublet model
        pass
    else:
        Y = None
        pi = None
        Z = None

    return {'a': a, 'assignment': assign, 'genotypes': geno, 'delta': d,
        'FN': FN, 'FP': FP, 'Z': Z, 'Y': Y, 'pi': pi
    }


def _get_posterior_avg(data):
    return np.mean(data), np.std(data)


def get_latents_point(results, estimator):
    # Best step (after burn in)
    step = np.argmax(results[estimator][results['burn_in']:]) \
        + results['burn_in']
    # DPMM conc. parameter
    if np.unique(results['DP_alpha']).size == 1:
        a = None
    else:
        a = results['DP_alpha'][step]
    # Doublet rate
    try:
        d = results['d'][step]
    except KeyError:
        d = None
    # Error rates
    try:
        FP, FN = _get_errors(results, estimator)
    except KeyError:
        FP, FN = (None, None)

    if 'Z' in results:
        assignment = []
        for cell_assign in results['Z'][step]:
            cl = np.argwhere(cell_assign).flatten()
            if cl.size == 1:
                assignment.append(cl[0])
            else:
                assignment.append(tuple(cl))
        geno = pd.DataFrame(results['params'][step]).T
        Y = results['Z'][step].sum(axis=1) - 1
        pi = results['pi'][step]
        Z = results['Z'][step].astype(int)
    else:
        assignment = results['assignments'][step].tolist()
        cl_names = np.unique(assignment)
        geno = pd.DataFrame(results['params'][step][cl_names], index=cl_names)\
            .T[assignment]
        Y = None
        pi = None
        Z = None

    return {'step': step, 'a': a, 'assignment': assignment, 'genotypes': geno,
        'delta': d, 'FN': FN, 'FP': FP, 'Z': Z, 'Y': Y, 'pi': pi
    }


def _get_errors(results, estimator):
    max_assign = np.argmax(results[estimator][results['burn_in']:]) \
        + results['burn_in']
    return results['fd_error'][max_assign], results['ad_error'][max_assign]


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


def get_lugsail_batch_means_est(data_in, burn_in, steps=None):
    data = data_in[burn_in:steps]
    # [chapter 2.2 in Vats and Knudson, 2018]
    n = data.size
    b = int(n ** (1/2)) # Batch size. Alternative: n ** (1/3)

    chain_mean = bn.nanmean(data)
    T_L = 2 * get_tau_lugsail(b, data, chain_mean) \
        - get_tau_lugsail(b // 3, data, chain_mean)
    s = bn.nanvar(data, ddof=1)

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
