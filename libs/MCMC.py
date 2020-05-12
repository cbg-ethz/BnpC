#!/usr/bin/env python3

from datetime import datetime
from copy import deepcopy
import numpy as np
import multiprocessing as mp

try:
    from libs import utils as ut
    from libs import dpmmIO as io
    # from libs.restricted_gibbs_non_conjugate import *
except ImportError:
    import utils as ut
    import libs.dpmmIO as io

np.seterr(all='raise')

# ------------------------------------------------------------------------------
# MCMC CLASS
# ------------------------------------------------------------------------------

class MCMC:
    def __init__(self, model, sm_prob=0.33, dpa_prob=0.5, error_prob=0.1,
                sm_ratios=[0.75, 0.25], sm_steps=5):
        """
        Arguments
            model (object): Initialized model
            sm_prob (float): Probability of conducting a split merge move
            dpa_prob (float): Probability of updating alpha of the CRP

        """
        # Init model and directory for results
        self.model = model
        self.chains = []
        self.seeds = []
        # Move probabilities
        self.params = {
            'sm_prob': sm_prob,
            'dpa_prob': dpa_prob,
            'error_prob': error_prob,
            # stdevs for paramter update
            'param_proposal_sd': np.array([0.1, 0.25, 0.5]),
            # Split merge variables
            'sm_ratios': sm_ratios,
            'sm_steps': sm_steps
        }


    def __str__(self):
        out_str = 'Move probabilitites:\n' \
            '\tSplit/merge:\t{sm_prob}\n\t\tsplit/merge ratio:\t{sm_ratios}\n' \
            '\t\tintermediate Gibbs:\t{sm_steps}\n' \
            '\tCRP a_0 update:\t{dpa_prob}\n' \
            '\tErrors update:\t{error_prob}\n' \
                .format(**self.params)

        return out_str


    def get_results(self):
        results = []
        for chain in self.chains:
            results.append(chain.get_result())

        if not 'burn_in' in results[0]:
            raise RuntimeError('Error in sampling from MCMC')

        return results


    def get_seeds(self):
        return self.seeds


    def run(self, run_var, seed, n=1, verbosity=1, assign_file='', debug=False):
        cutoff = None
        # Run with steps
        if isinstance(run_var[0], int):
            Chain_type = Chain_steps
        # Run with lugsail batch means estimator
        elif isinstance(run_var[0], float):
            Chain_type = Chain_steps
            cutoff = run_var[0]
            run_var = (int(1 / (cutoff ** 2 - 1)), 0)
            verbosity_ls = verbosity
            verbosity = 0
        # Run with runtime
        else:
            Chain_type = Chain_time

        if assign_file:
            assign = io.load_txt(assign_file)
        else:
            assign = None

        cores = min(n, mp.cpu_count())
        # Seed seed for reproducabilaty
        if seed > 0:
            np.random.seed(seed)
        self.seeds = np.random.randint(0, 2 ** 32 - 1, cores)

        if debug:
            np.random.seed(self.seeds[0])
            print(f'\nSeed set to: {self.seeds[0]}\n')
            run = self.run_chain(Chain_type, run_var, assign, 0, 2)
            self.chains.append(run)
            return

        pool = mp.Pool(cores)
        for i in range(cores):
            pool.apply_async(
                self.run_chain, (Chain_type, run_var, assign, i, verbosity),
                callback=self.chains.append
            )
        pool.close()
        pool.join()

        if cutoff:
            self.run_lugsail_chains(cutoff, cores, verbosity_ls)


    def run_chain(self, Chain_type, run_var, assign, i, verbosity):
        np.random.seed(self.seeds[i])
        model = deepcopy(self.model)
        model.init(assign=assign)
        new_chain = Chain_type(
            model, i + 1, *run_var, self.params, verbosity,
            isinstance(assign, list)
        )
        new_chain.run()
        return new_chain


    def run_lugsail_chains(self, cutoff, cores, verbosity, n=500):
        steps_run = self.chains[0].results['ML'].size

        while True:
            PSRF = ut.get_lugsail_batch_means_est(
                [(i.results['ML'], steps_run // 2) for i in self.chains]
            )
            if verbosity > 1:
                print(f'\tPSRF at {steps_run}:\t{PSRF:.5f}')

            for chain in self.chains:
                try:
                    chain.results['PSRF'].append((steps_run, PSRF))
                except KeyError:
                    chain.results['PSRF'] = [(steps_run, PSRF)]
            if PSRF <= cutoff:
                break

            # new_chain = self.extend_chain(0, n)
            # self.replace_chain(new_chain)

            # Run next n steps
            pool = mp.Pool(cores)
            for i in range(cores):
                pool.apply_async(
                    self.extend_chain, (i, n), callback=self.replace_chain
                )
            try:
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print('Manual termination')
                pool.terminate()
                pool.join()
                break

            steps_run += n

        burn_in = (steps_run // 2) + 1
        for chain in self.chains:
            chain.results['burn_in'] = burn_in
            chain.results['params'] = chain.results['params'][burn_in:]
            chain.results['PSRF_cutoff'] = cutoff


    def extend_chain(self, chain_no, add_steps):
        np.random.seed(self.seeds[chain_no])

        chain = self.chains[chain_no]
        old_steps = chain.get_steps()

        chain._extend_results(add_steps, False)
        chain.set_steps(add_steps)
        chain.run(init_steps=old_steps - 1)
        return chain_no, chain


    def replace_chain(self, new_chain):
        self.chains[new_chain[0]] = new_chain[1]


# ------------------------------------------------------------------------------
# CHAIN BASE CLASS
# ------------------------------------------------------------------------------

class Chain():
    def __init__(self, model, mcmc, no, verbosity=1, fix_assign=False):
        self.model = model
        self.mcmc = mcmc
        self.no = no
        # Model description
        if self.model.__module__ == 'libs.CRP_learning_errors':
            self.learning_errors = True
        else:
            self.learning_errors = False

        self.results = {}
        # MH counter
        self.MH_counter = np.zeros((5, 2))

        self.verbosity = verbosity
        self.fix_assign = fix_assign
        

    def __str__(self):
        return f'Chain: {self.no:0>2d}'


    def get_result(self):
        return self.results


    def run(self, *args):
        pass


    def init_results(self, steps):
        self.results['ML'] = np.zeros(steps)
        self.results['MAP'] = np.zeros(steps)
        self.results['DP_alpha'] = np.zeros(steps)
        self.results['FN'] = np.empty(steps)
        self.results['FP'] = np.empty(steps)
        self.results['assignments'] = np.zeros(
            (steps, self.model.cells_total), dtype=int
        )


    def update_results(self, step, burn_in=True):
        step_diff = self.results['ML'].size - step
        # Extend sample array if run with runtime argument instead of steps
        if step_diff == 0:
            try:
                self._extend_results(burn_in=burn_in)
            except MemoryError:
                step = step % self.results['ML'].size
                self.burn_in = np.nan

        ll = self.model.get_ll_full()
        self.results['ML'][step] = ll
        self.results['MAP'][step] = ll + self.model.get_lprior_full()
        self.results['DP_alpha'][step] = self.model.DP_a
        self.results['FN'][step] = self.model.FN
        self.results['FP'][step] = self.model.FP
        self.results['assignments'][step] = self.model.assignment

        if not burn_in:
            clusters = np.sort(
                np.fromiter(self.model.cells_per_cluster.keys(), dtype=int)
            )
            cluster_ids = np.arange(clusters.size)

            if not 'params' in self.results:
                self.results['params'] = np.zeros(
                    (step_diff, clusters.size, self.model.muts_total),
                    dtype=np.float32
                )
            burn_in_steps = self.results['ML'].size \
                - self.results['params'].shape[0] + 1

            cl_diff = clusters.size - self.results['params'].shape[1]
            if cl_diff > 0:
                self.results['params'] = np.pad(
                    self.results['params'], [(0,0), (0, cl_diff), (0,0)],
                    mode='constant'
                )

            self.results['params'][step - burn_in_steps + 1][cluster_ids] = \
                self.model.parameters[clusters]


    def _extend_results(self, add_size=None, burn_in=True):
        if not add_size:
            add_size = min(200, self.results['ML'].size)
        arr_new = np.zeros(add_size)

        if not burn_in:
            self.results['params'] = np.append(
                self.results['params'],
                np.zeros((add_size, self.results['params'].shape[1],
                    self.model.muts_total)),
                axis=0
            )

        self.results['ML'] = np.append(self.results['ML'], arr_new)
        self.results['MAP'] = np.append(self.results['MAP'], arr_new)
        self.results['DP_alpha'] = np.append(self.results['DP_alpha'], arr_new)
        self.results['FN'] = np.append(self.results['FN'], arr_new)
        self.results['FP'] = np.append(self.results['FP'], arr_new)
        self.results['assignments'] = np.append(self.results['assignments'],
            np.zeros((add_size, self.model.cells_total), int), axis=0
        )


    def stdout_progress(self):
        io.show_MH_acceptance(self.MH_counter[0], 'parameters', 1)
        if not self.fix_assign:
            io.show_MH_acceptance(self.MH_counter[1], 'splits')
            io.show_MH_acceptance(self.MH_counter[2], 'merges')
        if self.learning_errors:
            io.show_MH_acceptance(self.MH_counter[3], 'FP')
            io.show_MH_acceptance(self.MH_counter[4], 'FN')

        self.MH_counter = np.zeros((5, 2))


    def do_step(self):
        if not self.fix_assign:
            if np.random.random() < self.mcmc['sm_prob']:
                sm_declined, sm_move = self.model.update_assignments_split_merge(
                    self.mcmc['sm_ratios'], self.mcmc['sm_steps'])
                if sm_move == 0:
                    self.MH_counter[1] += sm_declined
                else:
                    self.MH_counter[2] += sm_declined
            else:
                self.model.update_assignments_Gibbs()

            if np.random.random() < self.mcmc['dpa_prob']:
                self.model.update_DP_alpha()

        par_declined, par_accepted = self.model.update_parameters()
        self.MH_counter[0][1] += par_declined
        self.MH_counter[0][0] += par_accepted

        if self.learning_errors and np.random.random() < self.mcmc['error_prob']:
            FP_declined, FN_declined = self.model.update_error_rates()
            self.MH_counter[3] += FP_declined
            self.MH_counter[4] += FN_declined


# ------------------------------------------------------------------------------
# RUN WITH STEP NUMBER
# ------------------------------------------------------------------------------

class Chain_steps(Chain):
    def __init__(self, model, no, steps, burn_in, mcmc, verbosity=1,
                fix_assign=False):
        super().__init__(model, mcmc, no, verbosity, fix_assign)

        self.steps = steps + 1
        self.burn_in = burn_in

        self.init_results(steps + 1)
        self.update_results(0, burn_in != 0)


    def set_steps(self, n):
        self.steps = n + 1


    def get_steps(self):
        return self.results['ML'].size


    def stdout_progress(self, step_no, total):
        print(f'\t{self}\tstep:\t{step_no: >3} / {total - 1}\n'
            '\t\tmean MH accept. ratio:')
        super().stdout_progress()


    def run(self, init_steps=0):
        # Run the MCMC - that's where all the work is done
        for step in range(1, self.steps, 1):
            if step % (self.steps // 10) == 0 and self.verbosity > 1:
                self.stdout_progress(step + init_steps, self.steps + init_steps)

            self.do_step()
            try:
                burn_in = step < self.burn_in
            except TypeError:
                burn_in = False
            self.update_results(step + init_steps, burn_in)

        self.results['burn_in'] = self.burn_in


# ------------------------------------------------------------------------------
# RUN WITH RUNTIME
# ------------------------------------------------------------------------------

class Chain_time(Chain):
    def __init__(self, model, no, end_time, burn_in, mcmc, verbosity=1,
                fix_assign=False):
        super().__init__(model, mcmc, no, verbosity, fix_assign)

        self.end_time = end_time
        self.burn_in = burn_in

        self.init_results(500)
        self.update_results(0)


    def stdout_progress(self, step_no, total):
        print(f'\t{self}\tstep:\t{step_no: >3}\t(remaining: {total:.1f} mins.)\n'
            '\t\tmean MH accept. ratio:')
        super().stdout_progress()


    def run(self):
        # Run the MCMC - that's where all the work is done
        step = 0
        while True:
            step_time = datetime.now()
            if step_time > self.end_time:
                break

            if step % 1000 == 0 and self.verbosity > 1:
                remaining = (self.end_time - step_time).seconds / 60
                self.stdout_progress(step, remaining)

            step += 1
            self.do_step()
            try:
                burn_in = step_time < self.burn_in
            except TypeError:
                burn_in = False
            self.update_results(step, burn_in)

        # Truncate empty steps
        zeros = (self.results['ML'] == 0).sum()
        if zeros != 0:
            for key, values in self.results.items():
                self.results[key] = values[:-zeros]

        self.results['burn_in'] = self.results['ML'].size \
            - self.results['params'].shape[0]