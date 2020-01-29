#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta

from libs.MCMC import MCMC as MCMC
import libs.utils as ut
import libs.dpmmIO as io

# ------------------------------------------------------------------------------
# ARGPARSER
# ------------------------------------------------------------------------------
def parse_args():

    def check_ratio(val):
        val = float(val)
        if val <= 0 or val >= 1:
            raise argparse.ArgumentTypeError(
                '{} is an invalid ratio value. Values need to be 0 < x < 1' \
                    .format(val)
            )
        return val

    parser = argparse.ArgumentParser(
          description='*** Clustering of single cell data ' \
          	'based on a Dirichlet process. ***'
    )
    parser.add_argument('--version', action='version', version='0.2')
    parser.add_argument(
        'input', help='Absolute or relative path to input data. ' \
           'Input data is a n x m matrix (n = cells, m = mutations) with 1|0, ' \
           'representing whether a mutation is present in a cell or not. Matrix ' \
           'elements need to be separated by a whitespace or tabulator. Nans can ' \
           'be represented by 3 or empty elements.'
    )
    parser.add_argument(
        '-t', '--transpose', action='store_false',
        help='Transpose the input matrix. Default = True.'
    )
    parser.add_argument(
        '-FN', '--falseNegative', type=float, default=-1,
        help='Fixed error rate for false negatives.'
    )
    parser.add_argument(
        '-FP', '--falsePositive', type=float, default=-1,
        help='Fixed error rate for false positives.'
    )
    parser.add_argument(
        '-FN_m', '--falseNegative_mean', type=check_ratio, default=0.25,
        help='Prior mean of the false negative rate. Default = 0.25.'
    )
    parser.add_argument(
        '-FN_sd', '--falseNegative_std', type=check_ratio, default=0.05,
        help='Prior standard dev. of the false negative rate. Default = 0.05.'
    )
    parser.add_argument(
        '-FP_m', '--falsePositive_mean', type=check_ratio, default=0.001,
        help='Prior mean of the false positive rate. Default = 0.001.'
    )
    parser.add_argument(
        '-FP_sd', '--falsePositive_std', type=check_ratio, default=0.005,
        help='Prior standard dev. of the false positive rate. Default = 0.005.'
    )
    parser.add_argument(
        '-dpa', '--DP_alpha', type=int, default=-1,
        help='Beta(x, 1) prior for the concentration parameter of the CRP. '
            'Default = log(cells).'
    )
    parser.add_argument(
        '-pa', '--param_alpha', type=float, default=1,
        help='Alpha value of the Beta function used as parameter prior. '
            'Default = 1.'
    )
    parser.add_argument(
        '-pb', '--param_beta', type=float, default=1,
        help='Beta value of the Beta function used as parameter prior. '
            'Default = 1.'
    )
    parser.add_argument(
        '-n', '--chains', type=int, default=1,
        help='Number of chains to run in parallel. Maximum possible number is '
            'the number of available cores. Default = 1.'
    )
    parser.add_argument(
        '-s', '--steps', type=int, default=5000,
        help='Number of MCMC steps. Default = 5000.'
    )
    parser.add_argument(
        '-r', '--runtime', type=int, default=-1,
        help='Runtime in minutes. If set, steps argument is overwritten. '
            'Default = -1.'
    )
    parser.add_argument(
        '-ls', '--lugsail', type=check_ratio, default=-1,
        help='Use lugsail batch means estimator as convergence diagnostics '
            '[Vats and Flegal, 2018]. The chain is terminated if the estimator '
            'undercuts a threshold defined by a significance level of 0.05 and '
            'a user defined float between [0,1], comparable to the half-width '
            'of the confidence interval in sample size calculation for a '
            'one sample t-test. Default = -1; Reasonal values = 0.1|0.2|0.3 .'
    )
    parser.add_argument(
        '-b', '--burn_in', type=check_ratio, default=0.33,
        help='Ratio of MCMC steps treated as burn-in. These steps are discarded.'\
            ' Default = 0.33.'
    )
    parser.add_argument(
        '-smp', '--split_merge_prob', type=check_ratio, default=0.25,
        help='Probability to do a split/merge step instead of Gibbs sampling. ' \
            'Default = 0.25.'
    )
    parser.add_argument(
        '-cup', '--conc_update_prob', type=check_ratio, default=0.1,
        help='Probability of updating the CRP concentration parameter. ' \
            'Default = 0.1.'
    )
    parser.add_argument(
        '-e', '--estimator', type=str, default='posterior', nargs='+',
        choices=['posterior', 'ML', 'MAP', 'MPEAR'],
        help='Estimator(s) used for inferrence. Default = posterior. '
            'Options = posterior|ML|MAP|MPEAR.'
    )
    parser.add_argument(
        '-tr', '--tree', type=str, default='',
        help='Absolute or relative path to the tree file (.gv) used for data ' \
            'generation. The cells will be colored accordingly to clusters. ' \
            'Default = "".'
    )
    parser.add_argument(
        '-tc', '--true_clusters', type=str, default='',
        help='Absolute or relative path to the true clusters assignments' \
        'to compare clustering methods. Default = "".'
    )
    parser.add_argument(
        '-td', '--true_data', type=str, default='',
        help='Absolute or relative path to the true/raw data/genotypes. ' \
            'Default = "".'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='',
        help='Path to the output directory. Default = "<DATA_DIR>/<TIMESTAMP>".'
    )
    parser.add_argument(
        '--seed', type=int, default=-1,
        help='Seed used for random number generation. Default = random.'
    )
    parser.add_argument(
        '-si', '--silent', action='store_true', default=False,
        help='Print status massages to stdout. Default = True.'
    )
    parser.add_argument(
        '-np', '--no_plots', action='store_true', default=False,
        help='Generate result plots. Default = False.'
    )
    parser.add_argument(
        '-sc', '--single_chains', action='store_true', default=False,
        help='Infer a result for each chain individually. Default = False.'
    )
    args = parser.parse_args()

    return args


# ------------------------------------------------------------------------------
# INIT AND OUTPUT FUNCTIONS
# ------------------------------------------------------------------------------


def get_CRP_with_errors_fixed(data, args):
    import libs.CRP as CRP
    BnpC = CRP.CRP(
        data, DP_alpha=args.DP_alpha,
        FN_error=args.falseNegative, FP_error=args.falsePositive,
        param_beta_a=args.param_alpha, param_beta_b=args.param_beta
    )
    return BnpC


def get_CRP_with_errors_learning(data, args):
    import libs.CRP_learning_errors as CRP
    BnpC = CRP.CRP_errors_learning(
        data, DP_alpha=args.DP_alpha,
        FP_mean=args.falsePositive_mean, FP_sd=args.falsePositive_std,
        FN_mean=args.falseNegative_mean, FN_sd=args.falseNegative_std,
        param_beta_a=args.param_alpha, param_beta_b=args.param_beta
    )
    return BnpC


def generate_output(args, data, results, names):
    out_dir = io._get_out_dir(args, args.time[1])

    if args.single_chains:
        inferred = {i: {} for i in range(args.chains)}
    else:
        inferred = {0: {}}

    if isinstance(args.estimator, str):
        args.estimator = [args.estimator]

    for est in args.estimator:
        if est == 'MPEAR':
            assign = ut.get_MPEAR_assignment(results, args.single_chains)
            for i, assign_chain in enumerate(assign):
                inferred[i]['MPEAR'] = assign_chain
        else:
            if est == 'posterior':
                inf_est = ut.get_latents_posterior(results, args.single_chains)
            else:
                inf_est = ut.get_latents_point(results, est, args.single_chains)

            for i, inf_est_chain in enumerate(inf_est):
                inferred[i][est] = inf_est_chain

    if not args.single_chains:
        inferred['mean'] = inferred.pop(0)

    if not args.silent:
        io.show_assignments(inferred, names[0])
        io.show_latents(inferred)
        print('\nWriting output to: {}\n'.format(out_dir))

    io.save_config(args, out_dir)
    io.save_assignments(inferred, out_dir)
    if args.true_clusters:
        assign = io.load_txt(args.true_clusters)
        io.save_v_measure(inferred, assign, out_dir)
        io.save_ARI(inferred, assign, out_dir)

    if len(args.estimator) > 0 or args.estimator[0] != 'MPEAR':
        # Save genotyping
        io.save_geno(inferred, out_dir, names[1])

        if args.true_data:
            data_raw = io.load_data(args.true_data, transpose=args.transpose)
            io.save_hamming_dist(inferred, data_raw, out_dir)

        if args.no_plots:
            exit()

        # Generate plots
        io.save_basic_plots(args, data.shape[0], results, out_dir)
        if data.shape[0] < 300:
            if args.tree:
                io.save_tree_plots(
                    args.tree, inferred, out_dir, args.transpose
                )
            if args.true_data:
                io.save_raw_data_plots(inferred, data_raw, out_dir)
            else:
                io.save_geno_plots(inferred, data, out_dir, names)
        else:
            print('Too many cells to plot genotypes/clusters')


def main(args):
    io.process_sim_folder(args, suffix='')
    data_raw, data_names = io.load_data(
        args.input, transpose=args.transpose, get_names=True
    )

    data, _ = io.preprocess_data(data_raw)

    if args.falsePositive > 0 and args.falseNegative > 0:
        crp = get_CRP_with_errors_fixed(data, args)
    else:
        crp = get_CRP_with_errors_learning(data, args)

    start_time = datetime.now()

    if args.runtime > 0:
        run_var = (start_time + timedelta(minutes=args.runtime), args.burn_in)
    elif args.lugsail > 0:
        run_var = (ut.get_cutoff_lugsail(args.lugsail), None)
    else:
        run_var = (args.steps, args.burn_in)

    mcmc = MCMC(crp)

    if not args.silent:
        print(crp)
        print(mcmc)
        print('Run MCMC:')

    mcmc.run(run_var, args.seed, args.chains)

    end_time = datetime.now()
    results = mcmc.get_results()
    args.seed = mcmc.get_seeds()
    args.time = [start_time, end_time]

    if not args.silent:
        io.show_MCMC_summary(args, results)

    generate_output(args, data, results, data_names)


if __name__ == '__main__':
    args = parse_args()
    main(args)
