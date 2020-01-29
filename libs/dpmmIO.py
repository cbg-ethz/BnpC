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
    while x:
        l.append(int(x.pop()))
    return l[::-1]


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
            os.path.dirname(args.input),
            '{:%Y%m%d_%H:%M:%S}{}'.format(timestamp, prefix)
        )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir


# ------------------------------------------------------------------------------
# OUTPUT - PLOTTING
# ------------------------------------------------------------------------------

def save_raw_data_plots(data, data_raw, out_dir):
    for chain, data_chain in geno_data.items():
        for geno, assign, est in data_chain:
            out_file = os.path.join(
                out_dir, 'genotypes_{}_{:0>2}.png'.format(est, chain))
            geno_full = ut._get_genotype_all(geno, assign)
            pl.plot_raw_data(
                geno_full, data, out_file=out_file, attachments=assign)


def save_tree_plots(tree, data, out_dir, transpose=True):
    for chain, data_chain in data.items():
        for assign, est in data_chain:
            prefix = 'colored_{}_{:0>2}'.format(est, chain)
            pl.color_tree_nodes(tree, assign, out_dir, transpose, prefix)


def save_basic_plots(args, results, out_dir):
    pl.plot_LL(results, os.path.join(out_dir, 'LL_trace.png'))

    if results[0]['assignments'][0].size < 300:
        if args.true_clusters:
            attachments = load_txt(args.true_clusters)
        else:
            attachments = None

        if args.single_chains:
            for i, result in enumerate(results):
                assignments = result['assignments'][result['burn_in']:]
                sim = (1 - ut.get_dist(assignments)).T
                sim_file = os.path.join(
                    out_dir, 'Posterior_similarity_{:0>2}.png'.format(i))
                pl.plot_similarity(sim, sim_file, attachments)
        else:
            assignments = np.concatenate(
                [i['assignments'][i['burn_in']:] for i in results]
            )
            sim = (1 - ut.get_dist(assignments)).T
            sim_file = os.path.join(out_dir, 'Posterior_similarity_mean.png')
            pl.plot_similarity(sim, sim_file, attachments)


def save_latents(data, out_file):
    with open(out_file, 'w') as f:
        if data['errors']:
            f.write('FP:\t{}\nFN:\t{}\n'.format(*data['errors']))


def save_doublet_plot(data, out_file):
    pl.plot_doublets(data, out_file)


def save_geno_plots(geno_data, data, out_dir, names):
    for chain, data_chain in geno_data.items():
        for geno, assign, est in data_chain:
            out_file = os.path.join(
                out_dir, 'genoCluster_{}_{:0>2}.png'.format(est, chain))
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

def show_MCMC_summary(args, results):
    total_time = args.time[1] - args.time[0]
    step_time = total_time / results[0]['ML'].size
    print('\nClustering time:\t{}\t({:.2f} secs. per MCMC step)' \
        .format(total_time, step_time.total_seconds()))
    if args.lugsail > 0:
        PSRF = ut.get_lugsail_batch_means_est(
            [(i['ML'], i['burn_in']) for i in results]
        )
        print('Lugsail PSRF:\t\t{:.5f}\n'.format(PSRF))


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


def show_MCMC_summary(args, results):
    total_time = args.time[1] - args.time[0]
    step_time = total_time / results[0]['ML'].size
    print('\nClustering time:\t{}\t({:.2f} secs. per MCMC step)' \
        .format(total_time, step_time.total_seconds()))
    if args.lugsail > 0:
        PSRF = ut.get_lugsail_batch_means_est(
            [(i['ML'], i['burn_in']) for i in results]
        )
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

    args.time = ['{:%Y%m%d_%H:%M:%S}'.format(i) for i in args.time]

    if args_dict['falseNegative'] > 0:
        del args_dict['falseNegative_mean']
        del args_dict['falseNegative_std']
    else:
        del args_dict['falseNegative']

    if args_dict['falsePositive'] > 0:
        del args_dict['falsePositive_mean']
        del args_dict['falsePositive_std']
    else:
        del args_dict['falsePositive']

    with open(os.path.join(out_dir, out_file), 'w') as f:
        for key, val in args_dict.items():
            f.write('{}: {}\n'.format(key, val))


def save_assignments(data, out_dir):
    cols = len(data) * len(data[next(iter(data))])
    df = pd.DataFrame(
        columns=['chain', 'estimator', 'Assignment'], index=range(cols)
    )

    i = 0
    for chain, data_chain in data.items():
        for assign, est in data_chain:
            df.iloc[i] = [chain, est, ' '.join([str(i) for i in assign])]
            i += 1

    df.to_csv(os.path.join(out_dir, 'assignment.txt'), index=False, sep='\t')


def save_geno(data, out_dir, names=np.array([])):
    for i, data_chain in data.items():
        for geno, assign, est in data_chain:
            clusters = np.unique(assign)

            try:
                params = geno[clusters]
            except KeyError:
                geno_cp = geno.copy()
                geno_cp.columns = np.arange(assign.size)
                params = geno_cp[clusters]

            if names.size == params.index.size:
                params.index = names

            if (params.round() == params).all().all():
                out_file = os.path.join(
                    out_dir, 'genotypes_{}_{:0>2}.tsv'.format(est, i))
                params.astype(int).to_csv(out_file, sep='\t')
            else:
                out_file = os.path.join(
                    out_dir, 'genotypes_cont_{}_{:0>2}.tsv'.format(est, i)
                )
                params.round(4).to_csv(out_file, sep='\t')

                out_file_rnd = os.path.join(
                    out_dir, 'genotypes_{}_{:0>2}.tsv'.format(est, i))
                params.round().astype(int).to_csv(out_file_rnd, sep='\t')


def show_MH_acceptance(counter, name, tab_no=2):
    try:
        rate = counter[0] / counter.sum()
    except (ZeroDivisionError, FloatingPointError):
        rate = np.nan
    print('\t\t\t{}:{}{:.2f}'.format(name, '\t' * tab_no, rate))


def show_assignments(data, names=np.array([])):
    for i, data_chain in data.items():
        for est, data_est in data_chain.items():
            print('Chain {:0>2} - {} clusters:'.format(i, est))
            show_assignment(data_est['assignment'], names)


def show_assignment(assignment, names=np.array([])):
    slt = {}
    cl_all = set()
    for i, cl in enumerate(assignment):
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


def show_latents(data):
    for i, data_chain in data.items():
        for est, data_est in data_chain.items():
            print('\nInferred latent variables\t--\tchain {:0>2} - {}'\
                .format(i, est))
            print('\tCRP a_0:\t{}'.format(get_latent_str(data_est['a'])))
            for var in ['FP', 'FN']:
                if data_est[var]:
                    if var == 'FP':
                        res_str = get_latent_str(data_est[var], 1, 'E')
                    else:
                        res_str = get_latent_str(data_est[var], 3)
                    print('\t{}:\t\t{}'.format(var, res_str))


def get_latent_str(latent_var, dec=1, dtype='f'):
    fmt_str = '{:.' + str(int(dec)) + dtype + '}'
    try:
        return (fmt_str + ' +- ' + fmt_str).format(*latent_var)
    except TypeError:
        return fmt_str.format(latent_var)


def save_v_measure(pred_data, true_cl, out_dir):
    Vmes = _dict_to_pd(pred_data, true_cl, 'V-measure', ut.get_v_measure)
    Vmes.to_csv(os.path.join(out_dir, 'V_measure.txt'), index=False, sep='\t')


def save_ARI(pred_data, true_cl, out_dir):
    ARI = _dict_to_pd(pred_data, true_cl, 'ARI', ut.get_ARI)
    ARI.to_csv(os.path.join(out_dir, 'ARI.txt'), index=False, sep='\t')


def _dict_to_pd(data, true_cl, measure, score_fct):
    cols = len(data) * len(data[next(iter(data))])
    df = pd.DataFrame(columns=['chain', 'estimator', measure], index=range(cols))

    i = 0
    for chain, data_chain in data.items():
        for pred_cl, est  in data_chain:
            score = score_fct(pred_cl, true_cl)
            df.iloc[i] = [chain, est, score]
            i += 1
    return df


def save_hamming_dist(data, true_data, out_dir):
    cols = len(data) * len(data[next(iter(data))])
    df = pd.DataFrame(
        columns=['chain', 'estimator', 'Hamming distance'], index=range(cols)
    )

    i = 0
    for chain, data_chain in data.items():
        for df_pred, assign, est in data_chain:
            df_pred_full = ut._get_genotype_all(df_pred, assign)
            score = ut.get_hamming_dist(df_pred_full, true_data)
            df.iloc[i] = [chain, est, score]
            i += 1

    df.to_csv(os.path.join(out_dir, 'hammingDist.txt'), index=False, sep='\t')


def _write_to_file(file, content, attach=False):
    if attach and os.path.exists(file):
        open_flag = 'a'
    else:
        open_flag = 'w'

    with open(file, open_flag) as f:
        f.write(str(content))



if __name__ == '__main__':
    print('Here be dragons...')