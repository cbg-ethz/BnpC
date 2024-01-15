#!/usr/bin/env python3

from datetime import timedelta
import os
import re
from string import ascii_uppercase
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

try:
    import libs.utils as ut
    import libs.plotting as pl
except ModuleNotFoundError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, base_dir)
    import utils as ut
    import plotting as pl


# ------------------------------------------------------------------------------
# INPUT - DATA
# ------------------------------------------------------------------------------

def load_data(in_file, transpose=True, get_names=False):
    lines = []

    # Get first fine lines to determine if col/row names are provided
    with open(in_file, 'r') as f:
        for i in range(5):
            lines.append(f.readline().strip())

    if (lines[0].count('\t') > lines[0].count(' ')) \
            and (lines[0].count('\t') > lines[0].count(',')):
        sep = '\t'
    elif lines[0].count(',') > lines[0].count(' '):
        sep=','
    else:
        sep = ' '

    header_row = False
    header_line = ''
    for el in lines[0].split(sep):
        try:
            el_float = float(el)
        except ValueError:
            if el == ' ':
                continue
            header_row = True
            header_line = lines.pop(0)
            break    
        else:
            if el_float not in [0, 1, 2, 3]:
                header_row = True
                header_line = lines.pop(0)
                break

    index_col = False
    for i, line in enumerate(lines):
        first_el = line.split(sep)[0]
        try:
            first_el_flt = float(first_el)
        except ValueError:
            if first_el == ' ':
                continue
            index_col = True
            break
        else:
            if first_el_flt not in [0, 1, 2, 3]:
                index_col = True
                break

    if index_col and header_row:
        df = pd.read_csv(in_file, sep=sep, index_col=0, header=0, na_values=[3, ' '])
        df = df.astype(float)
    elif index_col:
        col_types = dict([(i, str) if i == 0 else (j, float) \
            for i in range(len(lines[0].split(sep)))])
        df = pd.read_csv(in_file, sep=sep, index_col=0, header=None, dtype=col_types)
    elif header_row:
        df = pd.read_csv(in_file, sep=sep, index_col=None, header=0, dtype=float)
    else:
        df = pd.read_csv(in_file, sep=sep, index_col=None, header=None,
            dtype=float)

    if transpose:
        df = df.T

    df.replace(3, np.nan, inplace=True)
    # replace homozygos mutations with heterozygos
    df.replace(2, 1, inplace=True)

    if get_names:
        return df.values, (df.index.values, df.columns.values)
    else:
        return df.values


def load_txt(path):
    try:
        df = pd.read_csv(path, sep='\t', index_col=False)
        x = df.at[0, 'Assignment'].split(' ')
    except ValueError:
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

    if re.search(r'(\d+\.\d+)-(\d+\.\d+)', in_dir):
        data_files = sorted(
            [i for i in os.listdir(in_dir) if 'data' in i]
        )
    args.input = os.path.join(in_dir, f'data{suffix}.csv')
    if 'transpose' in args and args.transpose:
        args.true_clusters = os.path.join(in_dir, 'attachments.txt')

    raw_data_file = os.path.join(in_dir, 'data_raw.csv')
    if os.path.exists(raw_data_file):
        args.true_data = raw_data_file

    old_error_tree = os.path.join(
        in_dir, f'tree_w_cells_w_errors{suffix}.gv')
    error_tree = os.path.join(in_dir, f'tree_w_errors{suffix}.gv')
    old_tree = os.path.join(in_dir, f'tree_w_cells{suffix}.gv')
    new_tree = os.path.join(in_dir, f'tree{suffix}.gv')
    if os.path.exists(error_tree):
        args.tree = error_tree
    elif os.path.exists(old_error_tree):
        args.tree = old_error_tree
    elif os.path.exists(old_tree):
        args.tree = old_tree
    elif os.path.exists(new_tree):
        args.tree = new_tree

    args.plot_dir = in_dir


def _get_mcmc_termination(args):
    if args.runtime > 0:
        run_var = (args.time[0] + timedelta(minutes=args.runtime),
            args.time[0] + args.burn_in * timedelta(minutes=args.runtime))
        run_str = f'for {args.runtime} mins'
    elif args.lugsail > 0:
        cutoff = args.lugsail
        run_var = (cutoff, 0)
        run_str = f'until PSRF < {cutoff:.4f}'
    else:
        run_var = (args.steps, int(args.steps * args.burn_in))
        run_str = f'for {args.steps} steps'
    return run_var, run_str


def _get_out_dir(args, prefix=''):
    if args.output:
        if any([args.output.endswith(i) for i in ['.txt', '.gv', '.csv']]):
            out_dir = os.path.dirname(args.output)
        else:
            out_dir = args.output
    else:
        res_dir = f'BnpC_{args.time[0]:%Y%m%d_%H:%M:%S}{prefix}'
        out_dir = os.path.join(os.path.dirname(args.input), res_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir


# ------------------------------------------------------------------------------
# OUTPUT - PREPROCESSING
# ------------------------------------------------------------------------------

def _infer_results(args, results, data):
    args.PSRF = ut.get_lugsail_batch_means_est(
        [(i['ML'], i['burn_in']) for i in results]
    )
    args.steps = [i['ML'].size for i in results]

    if args.single_chains:
        inferred = {i: {} for i in range(args.chains)}
    else:
        inferred = {0: {}}

    if isinstance(args.estimator, str):
        args.estimator = [args.estimator]

    for est in args.estimator:
        if est == 'posterior':
            inf_est = ut.get_latents_posterior(results, data, args.single_chains)
        else:
            inf_est = ut.get_latents_point(results, est, data, args.single_chains)

        for i, inf_est_chain in enumerate(inf_est):
            inferred[i][est] = inf_est_chain

    if not args.single_chains:
        inferred['mean'] = inferred.pop(0)

    return inferred


# ------------------------------------------------------------------------------
# OUTPUT - PLOTTING
# ------------------------------------------------------------------------------

def save_tree_plots(tree, data, out_dir, transpose=True):
    for chain, data_chain in data.items():
        for est, data_est in data_chain.items():
            prefix = f'colored_{est}_{chain:0>2}'
            pl.color_tree_nodes(
                tree, data_est['assignment'], out_dir, transpose, prefix
            )


def save_trace_plots(results, out_dir):
    pl.plot_traces(results, os.path.join(out_dir, 'Traces.pdf'))


def save_similarity(args, inferred, results, out_dir):
    if args.true_clusters:
        attachments = load_txt(args.true_clusters)
    else:
        attachments = None

    if args.single_chains:
        for i, result in enumerate(results):
            assignments = result['assignments'][result['burn_in']:]
            if isinstance(attachments, type(None)):
                try:
                    attachments = inferred[i]['posterior']['assignment']
                except KeyError:
                    pass
            sim = squareform(1 - ut.get_dist(assignments))
            sim_file = os.path.join(
                out_dir, f'Posterior_similarity_{i:0>2}.pdf')
            pl.plot_similarity(sim, sim_file, attachments)
    else:
        assignments = np.concatenate(
            [i['assignments'][i['burn_in']:] for i in results]
        )
        if isinstance(attachments, type(None)):
            try:
                attachments = inferred['mean']['posterior']['assignment']
            except KeyError:
                pass
        sim = squareform(1 - ut.get_dist(assignments))
        sim_file = os.path.join(out_dir, 'Posterior_similarity_mean.pdf')
        pl.plot_similarity(sim, sim_file, attachments)


def save_geno_plots(data, data_raw, out_dir, names):
    for chain, data_chain in data.items():
        for est, data_est in data_chain.items():
            out_file = os.path.join(
                out_dir, f'genoCluster_{est}_{chain:0>2}.pdf')

            df_obs = pd.DataFrame(data_raw, index=names[0], columns=names[1]).T
            pl.plot_raw_data(
                data_est['genotypes'], df_obs, assignment=data_est['assignment'],
                out_file=out_file
            )


def gv_to_png(in_file):
    import warnings
    try:
        from graphviz import render
    except ImportError:
        warnings.warn('Could not load graphviz - no rendering!', UserWarning)
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render('dot', 'png', in_file)
    except subprocess.CalledProcessError:
        warnings.warn('Could not render graphviz - file corrupted!', UserWarning)


# ------------------------------------------------------------------------------
# OUTPUT - STDOUT
# ------------------------------------------------------------------------------

def show_MCMC_summary(args, results):
    total_time = args.time[1] - args.time[0]
    step_time = total_time / results[0]['ML'].size
    print(f'\nClustering time:\t{total_time}\t'
        f'({step_time.total_seconds():.2f} secs. per MCMC step)'
        f'\tLugsail PSRF:\t\t{args.PSRF:.5f}\n')

    
def show_model_parameters(data, args, fixed_errors_flag):
    print(f'\nDPMM with:\n\t{data.shape[0]} observations (cells)\n'
        f'\t{data.shape[1]} items (mutations)')

    if fixed_errors_flag:
        print('\tfixed errors\n\nInitializing with:\n'
            f'\tFixed FN rate: {args.allelicDropout}\n'
            f'\tFixed FP rate: {args.falseDiscovery}')
    else:
        print('\tlearning errors\n\nInitializing with:\n'
            '\tPrior FP:\t'
            f'trunc norm({args.falseDiscovery_mean},{args.falseDiscovery_std})\n'
            '\tPrior FN:\t'
            f'trunc norm({args.allelicDropout_mean},{args.allelicDropout_std})')

    if args.DP_alpha < 1:
        DP_a = np.log(data.shape[0])
    else:
        DP_a = args.DP_alpha
    print(f'\tPrior params.:\tBeta({args.param_alpha},{args.param_beta})\n'
        f'\tCRP a_0:\tGamma({DP_a:.1f},1)\n\nMove probabilitites:\n'
        f'\tSplit/merge:\t{args.split_merge_prob}\n'
        f'\tCRP a_0 update:\t{args.conc_update_prob}\nRun MCMC:')


def show_MH_acceptance(counter, name, tab_no=2):
    try:
        rate = counter[0] / counter.sum()
    except (ZeroDivisionError, FloatingPointError):
        rate = np.nan
    print('\t\t\t{}:{}{:.2f}'.format(name, '\t' * tab_no , rate))


def show_assignments(data, names=np.array([])):
    for i, data_chain in data.items():
        for est, data_est in data_chain.items():
            cl_no = np.unique(data_est['assignment']).size
            print(f'Chain {i:0>2} - {est} clusters\t(#{cl_no}):')
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

    print_cells = all([len(i) < 30 for i in slt.values()])
    if not print_cells:
        print(f'\t{len(cl_all)} clusters\n')

    for i, cluster in enumerate(cl_all):
        # Skip clusters that are only populated with doublets
        if cluster not in slt:
            continue
        items = slt[cluster]

        if print_cells:
            if names.size > 0:
                items = names[items]
            items_str = ', '.join([f'{i: >4}' for i in items])
        else:
            items_str = f'{len(items)} items'
        print(f'\t{ascii_uppercase[i % 26] * (i // 26 + 1)}: {items_str}')


def show_latents(data):
    for i, data_chain in data.items():
        for est, data_est in data_chain.items():
            print(f'\nInferred latent variables\t--\tchain {i:0>2} - {est}'
                f'\n\tCRP a_0:\t{get_latent_str(data_est["a"])}')
            for error in ['FP', 'FN']:
                if data_est[error]:
                    geno_error = f'{error}_geno'
                    if error == 'FP':
                        error_model = get_latent_str(data_est[error], 1, 'E')
                        error_geno = get_latent_str(data_est[geno_error], 1, 'E')
                    else:
                        error_model = get_latent_str(data_est[error], 3)
                        error_geno = get_latent_str(data_est[geno_error], 3)
                    print(f'\t{error} (model|genotypes): '
                        f'{error_model}\t|\t{error_geno}')



def get_latent_str(latent_var, dec=1, dtype='f'):
    if latent_var == None:
        return 'not inferred'

    fmt_str = '{:.' + str(int(dec)) + dtype + '}'
    try:
        return (fmt_str + ' ' * (dec - 1) + ' +- ' + fmt_str).format(*latent_var)
    except TypeError:
        return fmt_str.format(latent_var)


# ------------------------------------------------------------------------------
# OUTPUT - DATA
# ------------------------------------------------------------------------------

def save_run(inferred, args, out_dir, names):
    save_config(args, out_dir)
    save_errors(inferred, args, out_dir)
    save_assignments(inferred, args, out_dir)
    save_geno(inferred, out_dir, names[1])


def save_config(args, out_dir, out_file='args.txt'):
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)

    args_dict['time'] = [f'{i:%Y%m%d_%H:%M:%S}' for i in args_dict['time']]

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
            f.write(f'{key}: {val}\n')


def save_errors(data, args, out_dir):
    idx = np.arange(len(args.estimator) * args.chains)
    cols = ['chain', 'estimator', 'FN_model', 'FN_data', 'FP_model', 'FP_data']
    df = pd.DataFrame(index=idx, columns=cols)

    i = 0
    for chain, data_chain in data.items():
        for est, data_est in data_chain.items():
            if est == 'posterior':
                errors = [f'{data_est["FN"][0]:.4f}+-{data_est["FN"][1]:.4f}',
                    data_est['FN_geno'].round(4),
                    f'{data_est["FP"][0]:.8f}+-{data_est["FP"][1]:.8f}',
                    data_est['FP_geno'].round(8)]
            else:
                errors = [data_est['FN'].round(4), data_est['FN_geno'].round(4),
                    data_est['FP'].round(8), data_est['FP_geno'].round(8)]
                    
            df.iloc[i] = [chain, est] + errors
            i += 1

    df.to_csv(os.path.join(out_dir, 'errors.txt'), index=False, sep='\t')


def save_assignments(data, args, out_dir):
    idx = np.arange(len(args.estimator) * args.chains)
    df = pd.DataFrame(columns=['chain', 'estimator', 'Assignment'], index=idx)

    i = 0
    for chain, data_chain in data.items():
        for est, data_est in data_chain.items():
            assign_str = ' '.join([str(i) for i in data_est['assignment']])
            df.iloc[i] = [chain, est, assign_str]
            i += 1

    df.to_csv(os.path.join(out_dir, 'assignment.txt'), index=False, sep='\t')


def save_geno(data, out_dir, names=np.array([])):
    for chain, data_chain in data.items():
        for est, data_est in data_chain.items():
            geno = data_est['genotypes']

            if names.size == geno.index.size:
                geno.index = names

            if (geno.round() == geno).all().all():
                out_file = os.path.join(
                    out_dir, f'genotypes_{est}_{chain:0>2}.tsv')
                geno.astype(int).to_csv(out_file, sep='\t')
            else:
                out_file = os.path.join(
                    out_dir, f'genotypes_cont_{est}_{chain:0>2}.tsv')
                geno.round(4).to_csv(out_file, sep='\t')

                out_file_rnd = os.path.join(
                    out_dir, f'genotypes_{est}_{chain:0>2}.tsv')

                geno.round().astype(int).to_csv(out_file_rnd, sep='\t')


def save_v_measure(data, true_cl, out_dir):
    Vmes = _get_cl_metric_df(data, true_cl, 'V-measure', ut.get_v_measure)
    Vmes.to_csv(os.path.join(out_dir, 'V_measure.txt'), index=False, sep='\t')


def save_ARI(data, true_cl, out_dir):
    ARI = _get_cl_metric_df(data, true_cl, 'ARI', ut.get_ARI)
    ARI.to_csv(os.path.join(out_dir, 'ARI.txt'), index=False, sep='\t')


def _get_cl_metric_df(data, true_cl, measure, score_fct):
    rows = []
    for chain, data_chain in data.items():
        for est, data_est in data_chain.items():
            score = score_fct(data_est['assignment'], true_cl)
            rows.append([chain, est, score])
    return pd.DataFrame(rows, columns=['chain', 'estimator', measure])


def save_hamming_dist(data, true_data, out_dir):
    rows = []
    for chain, data_chain in data.items():
        for est, data_est in data_chain.items():
            score = ut.get_hamming_dist(data_est['genotypes'], true_data)
            rows.append([chain, est, 1 - score / true_data.size])

    col_names = ['chain', 'estimator', '1 - norm Hamming distance']
    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(os.path.join(out_dir, 'hammingDist.txt'), index=False, sep='\t')


if __name__ == '__main__':
    print('Here be dragons...')