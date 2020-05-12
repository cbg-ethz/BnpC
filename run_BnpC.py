#!/usr/bin/env python3

import argparse
from datetime import datetime

from libs.MCMC import MCMC as MCMC
import libs.dpmmIO as io

# ------------------------------------------------------------------------------
# ARGPARSER
# ------------------------------------------------------------------------------

def parse_args():

    def check_ratio(val):
        val = float(val)
        if val <= 0 or val >= 1:
            raise argparse.ArgumentTypeError(
                f'Invalid value: {val}. Values need to be 0 < x < 1')
        return val

    def check_percent(val):
        val = float(val)
        if val < 0 or val > 1:
            raise argparse.ArgumentTypeError(
                f'Invalid value: {val}. Values need to be 0 <= x <= 1')
        return val

    parser = argparse.ArgumentParser(
        prog='BnpC', usage='python3 run_BnpC.py <DATA> [options]',
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
        '--debug', action='store_true', default=False,
        help='Run single chain in main python thread for debugging with pdb.'
    )

    model = parser.add_argument_group('model')
    model.add_argument(
        '-FN', '--falseNegative', type=float, default=-1,
        help='Fixed error rate for false negatives.'
    )
    model.add_argument(
        '-FP', '--falsePositive', type=float, default=-1,
        help='Fixed error rate for false positives.'
    )
    model.add_argument(
        '-FN_m', '--falseNegative_mean', type=check_ratio, default=0.2,
        help='Prior mean of the false negative rate. Default = 0.2.'
    )
    model.add_argument(
        '-FN_sd', '--falseNegative_std', type=check_ratio, default=0.1,
        help='Prior standard dev. of the false negative rate. Default = 0.1.'
    )
    model.add_argument(
        '-FP_m', '--falsePositive_mean', type=check_ratio, default=0.01,
        help='Prior mean of the false positive rate. Default = 0.001.'
    )
    model.add_argument(
        '-FP_sd', '--falsePositive_std', type=check_ratio, default=0.01,
        help='Prior standard dev. of the false positive rate. Default = 0.01.'
    )
    model.add_argument(
        '-ap', '--DPa_prior', type=float,  nargs=2, default=[-1, -1],
        help='Gamma(a, b) values pf the Gamma function used as prior for the '
        'concentration parameter alpha of the CRP. Default = (#cells, 1).'
    )
    model.add_argument(
        '-pp', '--param_prior', type=float, nargs=2, default=[.1, .1],
        help='Beta(a, b) values of the Beta function used as parameter prior. '
            'Default = [.1, .1].'
    )
    model.add_argument(
        '-fa', '--fixed_assignment', type=str, default='',
        help='Path to file containing a cluster assignment. If set, this '
            'assignment is used and not updated. Default = "".'
    )

    mcmc = parser.add_argument_group('MCMC')
    mcmc.add_argument(
        '-n', '--chains', type=int, default=1,
        help='Number of chains to run in parallel. Maximum possible number is '
            'the number of available cores. Default = 1.'
    )
    mcmc.add_argument(
        '-s', '--steps', type=int, default=5000,
        help='Number of MCMC steps. Default = 5000.'
    )
    mcmc.add_argument(
        '-r', '--runtime', type=int, default=-1,
        help='Runtime in minutes. If set, steps argument is overwritten. '
            'Default = -1.'
    )
    mcmc.add_argument(
        '-ls', '--lugsail', type=check_ratio, default=-1,
        help='Use lugsail batch means estimator as convergence diagnostics '
            '[Vats and Flegal, 2018]. The chain is terminated if the estimator '
            'undercuts a threshold defined by a significance level of 0.05 and '
            'a user defined float between [0,1], comparable to the half-width '
            'of the confidence interval in sample size calculation for a '
            'one sample t-test. Default = -1; Reasonal values = 0.1|0.2|0.3 .'
    )
    mcmc.add_argument(
        '-b', '--burn_in', type=check_percent, default=0.33,
        help='Ratio of MCMC steps treated as burn-in. These steps are discarded.'\
            ' Default = 0.33.'
    )
    mcmc.add_argument(
        '-cup', '--conc_update_prob', type=check_percent, default=0.5,
        help='Probability of updating the CRP concentration parameter. ' \
            'Default = 0.5.'
    )
    mcmc.add_argument(
        '-eup', '--error_update_prob', type=check_percent, default=0.25,
        help='Probability of updating the CRP concentration parameter. ' \
            'Default = 0.25.'
    )
    mcmc.add_argument(
        '-smp', '--split_merge_prob', type=check_percent, default=0.5,
        help='Probability to do a split/merge step instead of Gibbs sampling. ' \
            'Default = 0.5.'
    )
    mcmc.add_argument(
        '-sms', '--split_merge_steps', type=int, default=3,
        help='Number of restricted Gibbs sampling steps during split-merge move.' \
            ' Default = 5.'
    )
    mcmc.add_argument(
        '-smr', '--split_merge_ratios', type=check_percent, nargs=2,
        default=[0.8, 0.2], help='Ratio of splits/merges. Default = 0.75:0.25'
    )

    mcmc.add_argument(
        '-e', '--estimator', type=str, default='posterior', nargs='+',
        choices=['posterior', 'ML', 'MAP'],
        help='Estimator(s) used for inferrence. Default = posterior. '
            'Options = posterior|ML|MAP.'
    )
    mcmc.add_argument(
        '-sc', '--single_chains', action='store_true', default=False,
        help='Infer a result for each chain individually. Default = False.'
    )
    mcmc.add_argument(
        '--seed', type=int, default=-1,
        help='Seed used for random number generation. Default = random.'
    )

    output = parser.add_argument_group('output')
    output.add_argument(
        '-o', '--output', type=str, default='',
        help='Path to the output directory. Default = "<DATA_DIR>/<TIMESTAMP>".'
    )
    output.add_argument(
        '-v', '--verbosity', type=int, default=1, choices=[0, 1, 2],
        help='Print status massages to stdout. Default = 1.'
    )
    output.add_argument(
        '-np', '--no_plots', action='store_true', default=False,
        help='Generate result plots. Default = False.'
    )
    output.add_argument(
        '-tr', '--tree', type=str, default='',
        help='Absolute or relative path to the tree file (.gv) used for data ' \
            'generation. The cells will be colored accordingly to clusters. ' \
            'Default = "".'
    )
    output.add_argument(
        '-tc', '--true_clusters', type=str, default='',
        help='Absolute or relative path to the true clusters assignments' \
        'to compare clustering methods. Default = "".'
    )
    output.add_argument(
        '-td', '--true_data', type=str, default='',
        help='Absolute or relative path to the true/raw data/genotypes. ' \
            'Default = "".'
    )

    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------------
# INIT AND OUTPUT FUNCTIONS
# ------------------------------------------------------------------------------

def generate_output(args, results, data_raw, names):
    out_dir = io._get_out_dir(args)
    inferred = io._infer_results(args, results, data_raw)
    
    if args.verbosity > 0:
        io.show_MCMC_summary(args, results)
        io.show_assignments(inferred, names[0])
        io.show_latents(inferred)
        print(f'\nWriting output to: {out_dir}\n')

    io.save_run(inferred, args, out_dir, names)

    if args.true_clusters:
        true_assign = io.load_txt(args.true_clusters)
        io.save_v_measure(inferred, true_assign, out_dir)
        io.save_ARI(inferred, true_assign, out_dir)

    if args.true_data:
        data_true = io.load_data(args.true_data, transpose=args.transpose)
        io.save_hamming_dist(inferred, data_true, out_dir)

    if args.no_plots:
        exit()

    # Generate plots
    io.save_trace_plots(results, out_dir)
    if data_raw.shape[0] < 300:
        if args.tree:
            io.save_tree_plots(
                args.tree, inferred, out_dir, args.transpose
            )
        io.save_similarity(args, results, out_dir)
        if args.true_data:
            io.save_geno_plots(inferred, data_true, out_dir, names)
        else:
            io.save_geno_plots(inferred, data_raw, out_dir, names)
    else:
        print('Too many cells to plot genotypes/clusters')


def main(args):
    io.process_sim_folder(args, suffix='')
    data, data_names = io.load_data(
        args.input, transpose=args.transpose, get_names=True
    )

    if args.falsePositive > 0 and args.falseNegative > 0:
        args.error_update_prob = 0
        import libs.CRP as CRP
        BnpC = CRP.CRP(
            data, DP_alpha=args.DPa_prior, param_beta=args.param_prior,
            FN_error=args.falseNegative, FP_error=args.falsePositive,
        )
    else:
        import libs.CRP_learning_errors as CRP
        BnpC = CRP.CRP_errors_learning(
            data, DP_alpha=args.DPa_prior, param_beta=args.param_prior,
            FP_mean=args.falsePositive_mean, FP_sd=args.falsePositive_std,
            FN_mean=args.falseNegative_mean, FN_sd=args.falseNegative_std
        )

    args.time = [datetime.now()]
    run_var, run_str = io._get_mcmc_termination(args)

    mcmc = MCMC(
        BnpC, sm_prob=args.split_merge_prob, dpa_prob=args.conc_update_prob,
        error_prob=args.error_update_prob, sm_ratios=args.split_merge_ratios,
        sm_steps=args.split_merge_steps
    )

    if args.verbosity > 0:
        print(BnpC)
        print(mcmc)
        print(f'Run MCMC with ({args.chains} chains {run_str}):')

    if args.debug:
        args.chains = 1

    mcmc.run(
        run_var, args.seed, args.chains, args.verbosity, args.fixed_assignment,
        args.debug
    )

    args.chain_seeds = mcmc.get_seeds()
    args.time.append(datetime.now())
    
    # Obtain results
    results = mcmc.get_results()
    args.steps = [i['ML'].size for i in results]
    generate_output(args, results, data, data_names)


if __name__ == '__main__':
    args = parse_args()
    main(args)