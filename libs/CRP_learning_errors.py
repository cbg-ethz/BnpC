#!/usr/bin/env python3

import numpy as np
from scipy.stats import truncnorm
import bottleneck as bn

try:
    from libs.CRP import CRP
    from libs import dpmmIO as io
except ImportError:
    from CRP import CRP
    import libs.dpmmIO as io


# ------------------------------------------------------------------------------
# LEARNING ERROR RATES - NO NANs
# ------------------------------------------------------------------------------

class CRP_errors_learning(CRP):
    def __init__(self, data, DP_alpha=1, param_beta_a=1, param_beta_b=1, \
                fd_mean=0.001, fd_sd=0.0005, ad_mean=0.25, ad_sd=0.05):
        super().__init__(data, DP_alpha, param_beta_a, param_beta_b)
        # Error rate prior
        FP_trunc_a = (0 - fd_mean) / fd_sd
        FP_trunc_b = (1 - fd_mean) / fd_sd

        self.FP_prior = truncnorm(FP_trunc_a, FP_trunc_b, fd_mean, fd_sd)
        self.FP_sd = np.array([fd_sd * 0.5, fd_sd, fd_sd * 1.5])

        FN_trunc_a = (0 - ad_mean) / ad_sd
        FN_trunc_b = (1 - ad_mean) / ad_sd

        self.FN_prior = truncnorm(FN_trunc_a, FN_trunc_b, ad_mean, ad_sd)
        self.FN_sd = np.array([ad_sd * 0.5, ad_sd, ad_sd * 1.5])

        # Initialize error rates
        self.alpha_error, self.beta_error = fd_mean, ad_mean

        # MH counter for error rate learning
        self.FP_MH_counter = np.zeros(2)
        self.FN_MH_counter = np.zeros(2)


    def _Bernoulli_FN_error(self, cell_data, beta_error):
        return (1 - beta_error) ** cell_data * beta_error ** (1 - cell_data)


    def _Bernoulli_FP_error(self, cell_data, alpha_error):
        return (1 - alpha_error) ** (1 - cell_data) * alpha_error ** cell_data


    def get_lpost_full(self):
        return super().get_lpost_full() \
            + self.FP_prior.logpdf(self.alpha_error) \
            + self.FN_prior.logpdf(self.beta_error)


    def _calc_ll_error(self, cell_id, cluster_params,
                alpha_error, beta_error, flat=False):
        # Bernoulli for FN + Bernoulli for FP
        FN = cluster_params \
            * self._Bernoulli_FN_error(self.data[cell_id], beta_error)
        FP = (1 - cluster_params) \
            * self._Bernoulli_FP_error(self.data[cell_id], alpha_error)
        nan = self.muts_per_cell[2][cell_id] * self._beta_mix_const[2]
        if flat:
            return bn.nansum(np.log(FN + FP), axis=0) + nan
        else:
            return bn.nansum(np.log(FN + FP), axis=1) + nan


    def get_ll_full_error(self, alpha_error, beta_error):
        ll = 0
        for cluster_id in np.unique(self.assignment):
            cells_ids = np.where(self.assignment == cluster_id)
            cluster_params = self.parameters[cluster_id] #.round()
            ll += bn.nansum(
                self._calc_ll_error(cells_ids, cluster_params,
                    alpha_error, beta_error)
            )
        return ll


    def update_error_rates(self):
        self.alpha_error = self.MH_error_rates('alpha')
        self.beta_error = self.MH_error_rates('beta')


    def update_MH_std_model_specific(self, mult=1.5, silent=False):
        # update only after a minimum of 30 steps (~10 x each StDev)
        if self.FN_MH_counter.sum() < 30:
            return

        ratio_FN = (self.FN_MH_counter[0] + 1) / (self.FN_MH_counter.sum() + 1)
        self.FN_MH_counter = np.zeros(2)

        if ratio_FN < 0.45:
            self.FN_sd = np.clip(self.FN_sd / mult, 0, 1)
        elif ratio_FN > 0.55:
            self.FN_sd = np.clip(self.FN_sd * mult, 0, 1)

        ratio_FP = (self.FP_MH_counter[0] + 1) / (self.FP_MH_counter.sum() + 1)
        self.FP_MH_counter = np.zeros(2)

        if ratio_FP < 0.45:
            self.FP_sd = np.clip(self.FP_sd / mult, 0, 1)
        elif ratio_FP > 0.55:
            self.FP_sd = np.clip(self.FP_sd * mult, 0, 1)

        if not silent:
            if not 0.45 < ratio_FP < 0.55:
                print('\tMH acceptance FP: {:.02f}\t'
                    '(StDev: {:.04f}|{:.04f}|{:.04f})'  \
                        .format(ratio_FP, *self.FP_sd)
                )
            if not 0.45 < ratio_FN < 0.55 :
                print('\tMH acceptance FN: {:.02f}\t'
                    '(StDev: {:.03f}|{:.03f}|{:.03f})' \
                        .format(ratio_FN, *self.FN_sd)
                )


    def MH_error_rates(self, error_type):
        # Set error specific values
        if error_type == 'alpha':
            old_error = self.alpha_error
            prior = self.FP_prior
            stdevs = self.FP_sd
            counter = self.FP_MH_counter
        else:
            old_error = self.beta_error
            prior = self.FN_prior
            stdevs = self.FN_sd
            counter = self.FN_MH_counter

        # Get new error from proposal distribution
        std = np.random.choice(stdevs)
        a, b = (0 - old_error) / std, (1 - old_error) / std
        try:
            new_error = truncnorm.rvs(a, b, loc=old_error, scale=std)
        except FloatingPointError:
            new_error = truncnorm.rvs(a, np.inf, loc=old_error, scale=std)

        # Calculate error depending log likelihood
        if error_type == 'alpha':
            new_ll = self.get_ll_full_error(new_error, self.beta_error)
            old_ll = self.get_ll_full_error(old_error, self.beta_error)
        else:
            new_ll = self.get_ll_full_error(self.alpha_error, new_error)
            old_ll = self.get_ll_full_error(self.alpha_error, old_error)

        # Calculate priors
        new_prior = prior.logpdf(new_error)
        old_prior = prior.logpdf(old_error)

        # Calculate probabilitites if not symmetric proposal
        a_rev, b_rev = (0 - new_error) / std, (1 - new_error) / std
        new_p_target = truncnorm \
            .logpdf(new_error, a, b, loc=old_error, scale=std)
        old_p_target = truncnorm \
            .logpdf(new_error, a_rev, b_rev, loc=new_error, scale=std)

        # Calculate decision treshold
        A = new_ll - old_ll + new_prior - old_prior + old_p_target - new_p_target

        if np.log(np.random.random()) < A:
            counter[0] += 1
            return new_error

        counter[1] += 1
        return old_error


    def do_MCMC_step(self, sm_prob=0.33, conc_prob=0.5, error_prob=0.1):
        super().do_MCMC_step(sm_prob, conc_prob)
        if np.random.random() < error_prob:
            self.update_error_rates()


    def run_restricted_gibbs(self, rg_move, obs_i, obs_j, S, params, clusters,
                step_no):
        rg_move.set_errors(self.alpha_error, self.beta_error)
        return rg_move.run(
            i=self.data[obs_i],
            j=self.data[obs_j],
            S=self.data[S],
            cells_S=S,
            params=params,
            clusters=clusters,
            a=self.DP_alpha,
            scan_no=step_no
        )


    def init_MCMC_results_model_specific(self, results, steps):
        results['ad_error'] = np.empty(steps)
        results['fd_error'] = np.empty(steps)


    def _extend_results_array_model_specific(self, results, arr_new):
        results['ad_error'] = np.append(results['ad_error'], arr_new)
        results['fd_error'] = np.append(results['fd_error'], arr_new)


    def stdout_MCMC_progress_model_specific(self):
        io.show_MH_acceptance(self.FN_MH_counter, 'FN')
        self.FN_MH_counter = np.zeros(2)
        io.show_MH_acceptance(self.FP_MH_counter, 'FP')
        self.FP_MH_counter = np.zeros(2)


    def update_MCMC_results_model_specific(self, results, step):
        results['ad_error'][step] = self.beta_error
        results['fd_error'][step] = self.alpha_error


if __name__ == '__main__':
    print('Here be dragons...')