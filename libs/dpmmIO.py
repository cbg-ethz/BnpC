#!/usr/bin/env python3

import os
import re
import warnings
from string import ascii_uppercase
import numpy as np
import pandas as pd

try:
    from graphviz import render
except ImportError:
    pass

try:
    import libs.utils as ut
except ModuleNotFoundError:
    import utils as ut

try:
    import libs.plotting as pl
except ModuleNotFoundError:
    import plotting as pl


# ------------------------------------------------------------------------------
# INPUT - DATA
# ------------------------------------------------------------------------------

def load_data(in_file, transpose=True, get_names=False):
    sep = '\t'
    df = pd.read_csv(in_file, sep=sep, index_col=None, header=None)

    # Wrong seperator
    if df.shape[1] == 1:
        sep = ' '
        df = pd.read_csv(in_file, sep=sep, index_col=None, header=None)

    try:
        index_col = ~df[df.columns[0]].iloc[1:].astype(int).isin([0,1,2,3]).all()
    except ValueError:
        index_col = True

    try:
        header_col = ~df.iloc[0,1:].astype(int).isin([0,1,2,3]).all()
    except ValueError:
        header_col = True

    if index_col and header_col:
        df = pd.read_csv(in_file, sep=sep, index_col=0, header=0)
    elif index_col:
        df = pd.read_csv(in_file, sep=sep, index_col=0)
    elif header_col:
        df = pd.read_csv(in_file, sep=sep, header=0)

    if transpose:
        df = df.T

    data = df.values.astype(float)
    rows = df.index.values
    cols = df.columns.values

    data[data == 3] = np.nan
    # replace homozygos mutations with heterozygos
    data[data == 2] = 1

    if get_names:
        return data, (rows, cols)
    else:
        return data


def load_txt(path):
    with open(path, 'r') as f:
        x = f.read().split(' ')
    l = []
    d = []
    while x:
        i = x.pop()
        if i.endswith(']'):
            i_2 = x.pop()
            l.append((int(i[:-1]), int(i_2[1:].rstrip(','))))
            d.append(1)
        else:
            l.append(int(i))
            d.append(0)
    return l[::-1], np.array(d, dtype=int)[::-1]


# ------------------------------------------------------------------------------
# INPUT - Preprocessing
# ------------------------------------------------------------------------------

def process_sim_folder(args, suffix=''):
    # Input is file: do nothing
    if not os.path.isdir(args.input):
        raw_data_file = os.path.join(os.path.dirname(args.input), 'data_raw.csv')
        if os.path.exists(raw_data_file):
            args.true_data = raw_data_file
        return
    in_dir = args.input

    if re.search('(\d+\.\d+)-(\d+\.\d+)', in_dir):
        data_files = sorted(
            [i for i in os.listdir(in_dir) if 'data' in i]
        )
    args.input = os.path.join(in_dir, 'data{}.csv'.format(suffix))
    if 'transpose' in args and args.transpose:
        args.true_clusters = os.path.join(in_dir, 'attachments.txt')

    raw_data_file = os.path.join(in_dir, 'data_raw.csv')
    if os.path.exists(raw_data_file):
        args.true_data = raw_data_file

    old_error_tree = os.path.join(
        in_dir, 'tree_w_cells_w_errors{}.gv'.format(suffix)
    )
    error_tree = os.path.join(in_dir, 'tree_w_errors{}.gv'.format(suffix))
    old_tree = os.path.join(in_dir, 'tree_w_cells{}.gv'.format(suffix))
    new_tree = os.path.join(in_dir, 'tree{}.gv'.format(suffix))
    if os.path.exists(error_tree):
        args.tree = error_tree
    elif os.path.exists(old_error_tree):
        args.tree = old_error_tree
    elif os.path.exists(old_tree):
        args.tree = old_tree
    elif os.path.exists(new_tree):
        args.tree = new_tree

    args.plot_dir = in_dir


def preprocess_data(data):
    df = pd.DataFrame(data)
    mapping = {}
    if df.duplicated().any():
        print('Data contains duplicated cells.')
        return df.values, mapping
    else:
        return data, mapping


def _get_out_dir(args, timestamp, prefix=''):
    if args.output:
        if '.txt' in args.output or '.gv' in args.output or '.csv' in args.output:
            out_dir = os.path.dirname(args.output)
        else:
            out_dir = args.output
    else:
        out_dir = os.path.join(
            os.path.dirname(args.input), '{:%Y%m%d_%H:%M:%S}{}' \
                .format(timestamp, prefix)
        )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir


# ------------------------------------------------------------------------------
# OUTPUT - PLOTTING
# ------------------------------------------------------------------------------

def save_raw_data_plots(data, geno_data, out_dir):
    for geno, assign, prefix in geno_data:
        geno_ML_fig = os.path.join(out_dir, 'genotypes_{}.png'.format(prefix))
        geno_full = ut._get_genotype_all(geno, assign)
        pl.plot_raw_data(
            geno_full, data, out_file=geno_ML_fig, attachments=assign)


def save_tree_plots(tree, data, out_dir, transpose=True):
    for assign, prefix in data:
        pl.color_tree_nodes(tree, assign, out_dir, transpose, prefix)


def save_basic_plots(args, cell_no, results, out_dir):
    LL_file = os.path.join(out_dir, 'LL_trace.png')
    pl.plot_LL(results, LL_file, results['burn_in'])

    if 'Z' in results:
        dbt_file = os.path.join(out_dir, 'doublet_probs.png')
        pl.plot_doublets(results['Z'][results['burn_in']:], dbt_file)

    if cell_no < 300:
        similarity = (1 - ut.get_dist(results)).T

        similarity_file = os.path.join(out_dir, 'Posterior_similarity.png')
        if args.true_clusters:
            attachments, _ = load_txt(args.true_clusters)
        else:
            attachments = None

        pl.plot_similarity(similarity, similarity_file, attachments)


def save_latents(data, out_file):
    with open(out_file, 'w') as f:
        if data['errors']:
            f.write('FP:\t{}\nFN:\t{}\n'.format(*data['errors']))
        if data['delta']:
            f.write('delta:\t{}\n'.format(data['delta']))
        if isinstance(data['pi'], np.ndarray):
            f.write('pi:\t{}\n'.format(np.array2string(data['pi'], separator=',')))
            pi_norm = data['pi'] / data['pi'].sum()
            f.write('pi norm:\t{}\n'.format(np.array2string(pi_norm, separator=',')))
        if isinstance(data['Y'], np.ndarray):
            f.write('Y:\t{}\n'.format(np.array2string(data['Y'], separator=',')))
        if isinstance(data['Z'], np.ndarray):
            f.write('Z:\t{}\n'.format(np.array2string(data['Z'], separator=',')))


def save_doublet_plot(data, out_file):
    pl.plot_doublets(data, out_file)


def save_geno_plots(geno_data, data, out_dir, names):
    for geno, assign, prefix in geno_data:
        out_file = os.path.join(out_dir, 'genoCluster_{}.png'.format(prefix))
        pl.plot_clusters(data, geno, assign, names, out_file)


def gv_to_png(in_file):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render('dot', 'png', in_file)
    except NameError:
        warnings.warn('Could not load graphviz - no rendering!', UserWarning)
    except subprocess.CalledProcessError:
        warnings.warn('Could not render graphviz - file corrupted!', UserWarning)


# ------------------------------------------------------------------------------
# OUTPUT - DATA
# ------------------------------------------------------------------------------


def show_model_parameters(data, args, fixed_errors_flag):
    print('\nDPMM with:\n\t{} observations (cells)\n\t{} items (mutations)' \
        .format(*data.shape))

    if fixed_errors_flag:
        print('\tfixed errors\n\nInitializing with:\n'
            '\tFixed FN rate: {}\n\tFixed FP rate: {}' \
                .format(args.allelicDropout, args.falseDiscovery)
        )
    else:
        print('\tlearning errors\n\nInitializing with:\n'
            '\tPrior FP:\ttrunc norm({},{})\n\tPrior FN:\ttrunc norm({},{})' \
                .format(
                    args.falseDiscovery_mean, args.falseDiscovery_std,
                    args.allelicDropout_mean, args.allelicDropout_std
                )
        )

    if args.DP_alpha < 1:
        DP_a = np.log(data.shape[0])
    else:
        DP_a = args.DP_alpha
    print('\tPrior params.:\tBeta({},{})\n\tCRP a_0:\tGamma({:.1f},1)\n'
        .format(args.param_alpha, args.param_beta, DP_a) )
    print(
        'Move probabilitites:\n'
        '\tSplit/merge:\t{}\n'
        '\tCRP a_0 update:\t{}\n' \
            .format(args.split_merge_prob, args.conc_update_prob)
    )
    print('Run MCMC:')


def show_MCMC_summary(start_time, end_time, results):
    step_time = (end_time - start_time) / results['ML'].size
    print('\nClustering time:\t{}\t({:.2f} secs. per MCMC step)' \
        .format((end_time - start_time), step_time.total_seconds()))
    PSRF = ut.get_lugsail_batch_means_est(results['ML'], results['burn_in'])
    print('Lugsail PSRF:\t\t{:.5f}\n'.format(PSRF))


def show_estimated_latents(est, latents):
    print('\nInferred latent variables\t--\t{}'.format(est))
    print('\tCRP a_0:\t{}'.format(get_latent_str(latents['a'])))
    for latent_par in ['delta', 'FP', 'FN']:
        if latents[latent_par]:
            if latent_par == 'FP':
                res_str = get_latent_str(latents[latent_par], 1, 'E')
            else:
                res_str = get_latent_str(latents[latent_par], 3)
            print('\t{}:\t\t{}'.format(latent_par, res_str))



def save_config(args, out_dir, out_file='args.txt'):
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)

    if 'allelicDropout' in args_dict and 'falseDiscovery' in args_dict:
        if args_dict['allelicDropout'] >= 0:
            del args_dict['allelicDropout_mean']
            del args_dict['allelicDropout_std']
        else:
            del args_dict['allelicDropout']

        if args_dict['falseDiscovery'] >= 0:
            del args_dict['falseDiscovery_mean']
            del args_dict['falseDiscovery_std']
        else:
            del args_dict['falseDiscovery']

    with open(os.path.join(out_dir, out_file), 'w') as f:
        for key, val in args_dict.items():
            f.write('{}: {}\n'.format(key, val))


def save_assignments(data, out_dir):
    for assign, prefix in data:
        out_file = 'assignment_{}.txt'.format(prefix)
        _save_vector(assign, out_dir, out_file)


def _save_vector(data, out_dir, out_file):
    out_file_abs = os.path.join(out_dir, out_file)
    with open(out_file_abs, 'w') as f:
        f.write(' '.join([str(i) for i in data]))


def save_geno(data, out_dir, names=np.array([])):
    for geno, assign, prefix in data:
        _save_items(geno, assign, out_dir, prefix, names)


def _save_items(data, assign, out_dir, prefix, names=np.array([])):
    try:
        clusters = np.unique(assign)
    except (TypeError, ValueError):
        clusters = set()
        for i in assign:
            if isinstance(i, list):
                clusters.update(i)
            if isinstance(i, tuple):
                clusters.update(list(i))
            else:
                clusters.add(i)
        clusters = list(clusters)

    try:
        params = data[clusters]
    except KeyError:
        data_cp = data.copy()
        data_cp.columns = np.arange(assign.size)
        params = data_cp[clusters]

    if names.size == params.index.size:
        params.index = names

    if (params.round() == params).all().all():
        out_file = os.path.join(out_dir, 'genotypes_{}.tsv'.format(prefix))
        params.to_csv(out_file, sep='\t')
    else:
        out_file = os.path.join(
            out_dir, 'cluster_parameters_{}.tsv'.format(prefix)
        )
        params.round(4).to_csv(out_file, sep='\t')

        out_file_rnd = os.path.join(out_dir, 'genotypes_{}.tsv'.format(prefix))
        params.round().to_csv(out_file_rnd, sep='\t')


def show_MH_acceptance(counter, name, tab_no=2):
    try:
        rate = counter[0] / counter.sum()
    except (ZeroDivisionError, FloatingPointError):
        rate = np.nan
    print('\t\t\t{}:{}{:.2f}'.format(name, '\t' * tab_no, rate))


def show_assignments(data, names=np.array([])):
    for assignment, estimator in data:
        print('{} clusters:'.format(estimator))
        show_assignment(assignment, names)


def show_assignment(assignment, names=np.array([])):
    dbt = {}
    slt = {}
    cl_all = set()
    for i, cl in enumerate(assignment):
        if isinstance(cl, (list, tuple, np.ndarray)):
            cl_all.update(list(cl))
            try:
                dbt[tuple(cl)].append(i)
            except KeyError:
                dbt[tuple(cl)] = [i]
        else:
            cl_all.add(cl)
            try:
                slt[cl].append(i)
            except KeyError:
                slt[cl] = [i]

    for i, cluster in enumerate(cl_all):
        # Skip clusters that are only populated with doublets
        if cluster not in slt:
            continue
        items = slt[cluster]

        if names.size > 0:
            items = names[items]

        if items.size < 30:
            items_str = ', '.join(['{: >4}'.format(i) for i in items])
        else:
            items_str = '{} items'.format(items.size)
        print('\t{}: {}' \
            .format(ascii_uppercase[i % 26] * (i // 26 + 1), items_str)
        )

    for dbt_cl, dbt_cells in dbt.items():
        items = dbt_cells
        if names.size > 0:
            items = names[dbt_cells]
        items_str = ', '.join(['{: >4}'.format(i) for i in items])
        ltr_cl_0 = ascii_uppercase[dbt_cl[0] % 26] * (dbt_cl[0] // 26 + 1)
        ltr_cl_1 = ascii_uppercase[dbt_cl[1] % 26] * (dbt_cl[1] // 26 + 1)
        print('\t{},{}: {}'.format(ltr_cl_0, ltr_cl_1, items_str))


def get_latent_str(latent_var, dec=1, dtype='f'):
    fmt_str = '{:.' + str(int(dec)) + dtype + '}'
    try:
        return (fmt_str + ' +- ' + fmt_str).format(*latent_var)
    except TypeError:
        return fmt_str.format(latent_var)


def save_v_measure(pred_data, true_cl, out_dir):
    for pred_cl, estimator  in pred_data:
        out_file = os.path.join(out_dir, 'V_measure_{}.txt'.format(estimator))
        score = ut.get_v_measure(pred_cl, true_cl)
        _write_to_file(out_file, score)


def save_ARI(pred_data, true_cl, out_dir):
    for pred_cl, estimator  in pred_data:
        out_file = os.path.join(out_dir, 'ARI_{}.txt'.format(estimator))
        score = ut.get_ARI(pred_cl, true_cl)
        _write_to_file(out_file, score)


def save_hamming_dist(pred_data, true_data, out_dir):
    for df_pred, assign, prefix in pred_data:
        out_file = os.path.join(out_dir, 'hammingDist_{}.txt'.format(prefix))
        df_pred_full = ut._get_genotype_all(df_pred, assign)
        score = ut.get_hamming_dist(df_pred_full, true_data)
        _write_to_file(out_file, score)


def _write_to_file(file, content, attach=False):
    if attach and os.path.exists(file):
        open_flag = 'a'
    else:
        open_flag = 'w'

    with open(file, open_flag) as f:
        f.write(str(content))



if __name__ == '__main__':
    print('Here be dragons...')