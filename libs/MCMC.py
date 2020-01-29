#!/usr/bin/env python3


from datetime import datetime, timedelta
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
            silent (bool); Whether to print progress to stdout or not

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
                .format(**self.params)

        return out_str



    def get_results(self):
        results = []
        for chain in self.chains:
            results.append(chain.get_result())

        if not results:
            raise RuntimeError('Error in sampling from MCMC')

        return results


    def get_seeds(self):
        return self.seeds


    def run(self, run_var, seeds, n=1, silent=False):
        # Run with steps
        if isinstance(run_var[0], int):
            Chain_type = Chain_steps
            chain_vars = run_var
        # Run with lugsail batch means estimator
        elif isinstance(run_var[0], float):
            Chain_type = Chain_steps
            chain_vars = (int(1 / (run_var[0]**2 - 1)), 0)
            silent_ls = silent
            silent = True
        # Run with runtime
        else:
            Chain_type = Chain_time
            chain_vars = run_var

        cores = min(n, mp.cpu_count())
        # Seed seed for reproducabilaty
        if (isinstance(seeds, list) and len(np.unique(seeds)) < cores) \
                or (isinstance(seeds, int) and seeds < 0):
            self.seeds = np.random.randint(0, 2 ** 32 - 1, cores)
        else:
            self.seeds = seeds

        pool = mp.Pool(cores)
        for i in range(cores):
            pool.apply_async(
                self.run_chain, (Chain_type, chain_vars, i, silent),
                callback=self.chains.append
            )
        pool.close()
        pool.join()

        if isinstance(run_var[0], float):
            self.run_lugsail_chains(run_var[0], cores, silent_ls)


    def run_chain(self, Chain_type, chain_vars, i, silent):
        np.random.seed(self.seeds[i])
        model = deepcopy(self.model)
        model.init()
        new_chain = Chain_type(model, i + 1, *chain_vars, self.params, silent)
        new_chain.run()
        return new_chain


    def run_lugsail_chains(self, cutoff, cores, silent, n=500):
        steps_run = self.chains[0].results['ML'].size
        while True:
            PSRF = ut.get_lugsail_batch_means_est(
                [(i.results['ML'], steps_run // 2) for i in self.chains]
            )
            if not silent:
                print('\tPSRF at {}:\t{:.5f}\t(> {:.5f})' \
                    .format(steps_run, PSRF, cutoff))

            if PSRF <= cutoff:
                break

            # Run next n steps
            pool = mp.Pool(cores)
            for i in range(cores):
                pool.apply_async(
                    self.extend_chain, (i, n), callback=self.replace_chain
                )
            pool.close()
            pool.join()

            steps_run += n

        burn_in = steps_run // 2
        for chain in self.chains:
            chain.results['burn_in'] = burn_in


    def extend_chain(self, chain_no, add_steps):
        np.random.seed(self.seeds[chain_no])

        chain = self.chains[chain_no]
        old_steps = chain.get_steps()

        chain._extend_results(add_steps)
        chain.set_steps(add_steps)
        chain.run(init_steps=old_steps - 1)
        return chain, chain_no


    def replace_chain(self, new_chain):
        self.chains[new_chain[1]] = new_chain[0]


# ------------------------------------------------------------------------------
# CHAIN BASE CLASS
# ------------------------------------------------------------------------------

class Chain():
    def __init__(self, model, mcmc, no, silent=False):
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

        self.silent = silent


    def __str__(self):
        return 'Chain: {:0>2d}'.format(self.no)


    def get_result(self):
        return self.results


    def run(self, *args):
        pass


    def init_results(self, steps):
        self.results['ML'] = np.zeros(steps)
        self.results['MAP'] = np.zeros(steps)
        self.results['DP_alpha'] = np.zeros(steps)
        self.results['assignments'] = np.zeros(
            (steps, self.model.cells_total), dtype=int
        )
        self.results['params'] = np.zeros(
            (steps, 1, self.model.muts_total), dtype=np.float32
        )

        if self.learning_errors:
            self.results['FN'] = np.empty(steps)
            self.results['FP'] = np.empty(steps)


    def update_results(self, step):
        ll = self.model.get_ll_full()
        lpost = ll + self.model.get_lprior()

        try:
            self.results['ML'][step] = ll
        except IndexError:
            # Extend sample array if run with runtime argument instead of steps
            try:
                self._extend_results()
            except MemoryError:
                step = step % self.results['ML'].size
            self.results['ML'][step] = ll

        self.results['MAP'][step] = lpost
        self.results['assignments'][step] = self.model.assignment
        self.results['DP_alpha'][step] = self.model.DP_alpha

        clusters = np.fromiter(self.model.cells_per_cluster.keys(), dtype=int)
        try:
            self.results['params'][step][clusters] = \
                self.model.parameters[clusters]
        except IndexError:
            cl_diff = clusters.max() - self.results['params'].shape[1] + 1
            self.results['params'] = np.pad(
                self.results['params'], [(0,0), (0, cl_diff),(0,0)],
                mode='constant'
            )
            self.results['params'][step][clusters] = \
                self.model.parameters[clusters]

        if self.learning_errors:
            self.results['FN'][step] = self.model.beta_error
            self.results['FP'][step] = self.model.alpha_error


    def _extend_results(self, add_size=None):
        if not add_size:
            add_size = min(200, self.results['ML'].size)
        arr_new = np.zeros(add_size)

        self.results['ML'] = np.append(self.results['ML'], arr_new)
        self.results['MAP'] = np.append(self.results['MAP'], arr_new)
        self.results['DP_alpha'] = np.append(self.results['DP_alpha'], arr_new)
        self.results['assignments'] = np.append(self.results['assignments'],
            np.zeros((add_size, self.model.cells_total), int), axis=0
        )
        self.results['params'] = np.append(
            self.results['params'],
            np.zeros((add_size, self.results['params'].shape[1],
                self.model.muts_total)),
            axis=0
        )
        if self.learning_errors:
            self.results['FN'] = np.append(self.results['FN'], arr_new)
            self.results['FP'] = np.append(self.results['FP'], arr_new)


    def _truncate_results(self):
        zeros = (self.results['ML'] == 0).sum()
        if zeros != 0:
            for key, values in self.results.items():
                self.results[key] = values[:-zeros]


    def stdout_progress(self):
        io.show_MH_acceptance(self.MH_counter[0], 'parameters', 1)
        io.show_MH_acceptance(self.MH_counter[1], 'splits')
        io.show_MH_acceptance(self.MH_counter[2], 'merges')
        if self.learning_errors:
            io.show_MH_acceptance(self.MH_counter[3], 'FP')
            io.show_MH_acceptance(self.MH_counter[4], 'FN')

        self.MH_counter = np.zeros((5, 2))


    def do_step(self):
        if np.random.random() < self.mcmc['sm_prob']:
            sm_declined, sm_move = self.model.update_assignments_split_merge(
                self.mcmc['sm_ratios'], self.mcmc['sm_steps'])
            if sm_move == 0:
                self.MH_counter[1] += sm_declined
            else:
                self.MH_counter[2] += sm_declined
        else:
            self.model.update_assignments_Gibbs()

        par_declined = self.model.update_parameters()
        self.MH_counter[0][1] += par_declined
        self.MH_counter[0][0] += self.model.muts_total - par_declined

        if np.random.random() < self.mcmc['dpa_prob']:
            self.model.update_DP_alpha()

        if self.learning_errors and np.random.random() < self.mcmc['error_prob']:
            FP_declined, FN_declined = self.model.update_error_rates()
            self.MH_counter[3] += FP_declined
            self.MH_counter[4] += FN_declined


# ------------------------------------------------------------------------------
# RUN WITH STEP NUMBER
# ------------------------------------------------------------------------------

class Chain_steps(Chain):
    def __init__(self, model, no, steps, burn_in, mcmc, silent=False):
        super().__init__(model, mcmc, no, silent)

        self.steps = steps + 1

        self.results['burn_in'] = int(steps * burn_in)
        self.init_results(steps + 1)
        self.update_results(0)


    def set_steps(self, n):
        self.steps = n + 1


    def get_steps(self):
        return self.results['ML'].size


    def stdout_progress(self, step_no, total):
        print('\t{}\tstep:\t{: >3} / {}\n\t\tmean MH accept. ratio:' \
                .format(self, step_no, total - 1))
        super().stdout_progress()


    def run(self, init_steps=0):
        # Run the MCMC - that's where all the work is done
        for step in range(1, self.steps, 1):
            if step % (self.steps // 10) == 0 and not self.silent:
                self.stdout_progress(step + init_steps, self.steps + init_steps)

            self.do_step()
            self.update_results(step + init_steps)


# ------------------------------------------------------------------------------
# RUN WITH RUNTIME
# ------------------------------------------------------------------------------

class Chain_time(Chain):
    def __init__(self, model, no, end_time, burn_in, mcmc, silent=False):
        super().__init__(model, mcmc, no, silent)

        self.end_time = end_time
        self.burn_in = burn_in

        self.init_results(500)
        self.update_results(0)


    def stdout_progress(self, step_no, total):
        print('\t{}\tstep:\t{: >3}\t(remaining: {:.1f} mins.)\n'
            '\t\tmean MH accept. ratio:'.format(self, step_no, total)
        )
        super().stdout_progress()


    def run(self):
        # Run the MCMC - that's where all the work is done
        step = 0
        while True:
            step += 1
            step_time = datetime.now()

            if step_time > self.end_time:
                break

            if step % 1000 == 0 and not self.silent:
                remaining = (self.end_time - step_time).seconds / 60
                self.stdout_progress(step, remaining)

            self.do_step()
            self.update_results(step)

        results = self._truncate_results()
        if step != self.results['ML'].size: import pdb; pdb.set_trace()
        self.results['burn_in'] = int(step * self.burn_in)