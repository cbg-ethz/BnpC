#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

from datetime import datetime, timedelta

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
    parser.add_argument('--version', action='version', version='0.1')
    parser.add_argument(
        'input', help='Absolute or relative path to input data. ' \
           'Input data is a n x m matrix (n = cells, m = mutations) with 1|0, ' \
           'representing whether a mutation is present in a cell or not. Matrix ' \
           'elements need to be separated by a whitespace or tabulator. Nans can ' \
           'be represented by 3 or empty elements.'
    )
    parser.add_argument(
        '-t', '--transpose', action='store_false',
        help='Transpose the input matrix. Default = True'
    )
    parser.add_argument(
        '-ad', '--allelicDropout', type=float, default=-1,
        help='Fixed error rate for allelic dropouts (false negatives).'
    )
    parser.add_argument(
        '-fd', '--falseDiscovery', type=float, default=-1,
        help='Fixed error rate for false discoveries (false positives).'
    )
    parser.add_argument(
        '-ad_m', '--allelicDropout_mean', type=check_ratio, default=0.25,
        help='Mean for the prior for the allelic dropout rate (false negatives). ' \
            'Default = 0.25.'
    )
    parser.add_argument(
        '-ad_sd', '--allelicDropout_std', type=check_ratio, default=0.05,
        help='Standard deviation for the prior for the allelic dropout rate ' \
            '(false negatives). Default = 0.05.'
    )
    parser.add_argument(
        '-fd_m', '--falseDiscovery_mean', type=check_ratio, default=0.001,
        help='Mean for the prior for the false discoveries rate (false positives). ' \
            'Default = 0.001.'
    )
    parser.add_argument(
        '-fd_sd', '--falseDiscovery_std', type=check_ratio, default=0.005,
        help='Standard deviation for the prior for the false discoveries rate ' \
            '(false positives). Default = 0.005.'
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
            'one sample t-test. Default = -1; Reasonal values = 0.1|0.2|0.3'
    )
    parser.add_argument(
        '-b', '--burn_in', type=check_ratio, default=0.33,
        help='Ratio of MCMC steps treated as burn-in. These steps are discarded. '\
            'Default = 0.33'
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
            'Default = ""'
    )
    parser.add_argument(
        '-par', '--parameters', action='store_true', default=False,
        help='Whether to plot the cluster parameter traces or not. Default = False'
    )
    parser.add_argument(
        '-tc', '--true_clusters', type=str, default='',
        help='Absolute or relative path to the true clusters assignments' \
        'to compare clustering methods. Default = ""'
    )
    parser.add_argument(
        '-td', '--true_data', type=str, default='',
        help='Absolute or relative path to the true/raw data/genotypes. ' \
            'Default = ""'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='',
        help='Path to the output directory. Default = "<DATA_DIR>/<TIMESTAMP>".'
    )
    parser.add_argument(
        '--seed', type=int, default=-1,
        help='Seed used for random number generation. Default = random'
    )
    parser.add_argument(
        '-si', '--silent', action='store_true', default=False,
        help='Print status massages to stdout. Default = True'
    )
    parser.add_argument(
        '-np', '--no_plots', action='store_true', default=False,
        help='Generate result plots. Default = False'
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
        ad_error=args.allelicDropout, fd_error=args.falseDiscovery,
        param_beta_a=args.param_alpha, param_beta_b=args.param_beta
    )
    return BnpC


def get_CRP_with_errors_learning(data, args):
    import libs.CRP_learning_errors as CRP
    BnpC = CRP.CRP_errors_learning(
        data, DP_alpha=args.DP_alpha,
        fd_mean=args.falseDiscovery_mean, fd_sd=args.falseDiscovery_std,
        ad_mean=args.allelicDropout_mean, ad_sd=args.allelicDropout_std,
        param_beta_a=args.param_alpha, param_beta_b=args.param_beta
    )
    return BnpC


def generate_output(args, data, results, names, out_dir, silent=False):
    latents_all = {}
    assign_data = []
    geno_data = []
    geno_data_approx = []
    if isinstance(args.estimator, str):
        args.estimator = [args.estimator]
    for est in args.estimator:
        if est == 'MPEAR':
            assign = ut.get_MPEAR_assignment(results)
            assign_data.append((assign, 'MPEAR'))
        else:
            if est == 'posterior':
                latents = ut.get_latents_posterior(results)
                name = 'meanHierarchy'
            elif est == 'ML' or est == 'MAP':
                latents = ut.get_latents_point(results, est)
                name = est

            assign_data.append((latents['assignment'], name))
            geno_data.append((latents['genotypes'], latents['assignment'], name))
            geno_data_approx.append(
                (latents['genotypes'].round(), latents['assignment'], name)
            )
            latents_all[est] = latents

    if not args.silent:
        io.show_assignments(assign_data, names[0])

        for latents_est in latents_all.items():
            io.show_estimated_latents(*latents_est)

        print('\nWriting output to: {}\n'.format(out_dir))

    io.save_config(args, out_dir)
    # Save clustering
    io.save_assignments(assign_data, out_dir)
    if args.true_clusters:
        assign, _ = io.load_txt(args.true_clusters)
        io.save_v_measure(assign_data, assign, out_dir)
        io.save_ARI(assign_data, assign, out_dir)

    if geno_data:
        # Save genotyping
        io.save_geno(geno_data, out_dir, names[1])

        if args.true_data:
            data_raw = io.load_data(args.true_data, transpose=args.transpose)
            io.save_hamming_dist(geno_data_approx, data_raw, out_dir)

        if args.no_plots:
            exit()

        # Generate plots
        io.save_basic_plots(args, data.shape[0], results, out_dir)
        if data.shape[0] < 300:
            io.save_geno_plots(geno_data_approx, data, out_dir, names)
        else:
            print('Too many cells to plot genotypes/clusters')

        if args.true_data and data.shape[0] < 300:
            io.save_raw_data_plots(data_raw, geno_data_approx, out_dir)

        if args.tree and data.shape[0] < 300:
            tree_data = [(i[0], 'colored_{}'.format(i[1])) for i in assign_data]
            io.save_tree_plots(args.tree, tree_data, out_dir, args.transpose)


def main(args):
    io.process_sim_folder(args, suffix='')
    data_raw, data_names = io.load_data(
        args.input, transpose=args.transpose, get_names=True
    )

    # Set seed for reproducabilaty
    if args.seed < 0:
        args.seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(args.seed)

    data, item_mapping = io.preprocess_data(data_raw)

    fixed_errors_flag = args.allelicDropout >= 0 and args.falseDiscovery >= 0

    if not args.silent:
        io.show_model_parameters(data, args, fixed_errors_flag)

    if fixed_errors_flag:
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

    results = crp.run(
        run_var, sm_prob=args.split_merge_prob, dpa_prob=args.conc_update_prob,
        silent=args.silent
    )
    end_time = datetime.now()

    if not args.silent:
        io.show_MCMC_summary(start_time, end_time, results)

    out_dir = io._get_out_dir(args, end_time)
    generate_output(args, data, results, data_names, out_dir, args.silent)


if __name__ == '__main__':
    args = parse_args()
    main(args)
