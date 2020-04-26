#!/usr/bin/env python3

import numpy as np
from scipy.stats import truncnorm
import bottleneck as bn

try:
    from libs.CRP import CRP
except ImportError:
    from CRP import CRP


# ------------------------------------------------------------------------------
# LEARNING ERROR RATES - NO NANs
# ------------------------------------------------------------------------------

class CRP_errors_learning(CRP):
    def __init__(self, data, DP_alpha=1, param_beta=[1, 1], \
                FP_mean=0.001, FP_sd=0.0005, FN_mean=0.25, FN_sd=0.05):
        super().__init__(data, DP_alpha, param_beta, FN_mean, FP_mean)
        # Error rate prior
        FP_trunc_a = (0 - FP_mean) / FP_sd
        FP_trunc_b = (1 - FP_mean) / FP_sd

        self.FP_prior = truncnorm(FP_trunc_a, FP_trunc_b, FP_mean, FP_sd)
        self.FP_sd = np.array([FP_sd * 0.5, FP_sd, FP_sd * 1.5])

        FN_trunc_a = (0 - FN_mean) / FN_sd
        FN_trunc_b = (1 - FN_mean) / FN_sd

        self.FN_prior = truncnorm(FN_trunc_a, FN_trunc_b, FN_mean, FN_sd)
        self.FN_sd = np.array([FN_sd * 0.5, FN_sd, FN_sd * 1.5])


    def __str__(self):
        # Fixed values
        out_str = '\nDPMM with:\n' \
            f'\t{self.cells_total} observations (cells)\n' \
            f'\t{self.muts_total} items (mutations)\n\tlearning errors\n' \
            '\n\tPriors:\n' \
            f'\tparams.:\tBeta({self.betaDis_alpha},{self.betaDis_beta})\n' \
            f'\tCRP a_0:\tGamma({self.DP_a_gamma[0]:.1f},1)\n' \
            f'\tFP:\t\ttrunc norm({self.FP_prior.args[2]},{self.FP_prior.args[3]})\n' \
            f'\tFN:\t\ttrunc norm({self.FN_prior.args[2]},{self.FN_prior.args[3]})\n'
        return out_str


    def get_lprior_full(self):
        return super().get_lprior_full() \
            + self.FP_prior.logpdf(self.alpha_error) \
            + self.FN_prior.logpdf(self.beta_error)


    def update_error_rates(self):
        FP_new, FP_count = self.MH_error_rates('FP')
        self.alpha_error = FP_new

        FN_new, FN_count = self.MH_error_rates('FN')
        self.beta_error = FN_new

        return FP_count, FN_count


    def get_ll_full_error(self, alpha, beta):
        par = self.parameters[self.assignment]
        FN = par * (1 - beta) ** self.data * beta ** (1 - self.data) 
        FP = (1 - par) * (1 - alpha) ** (1 - self.data) * alpha ** self.data
        ll = np.log(FN + FP)
        bn.replace(ll, np.nan, self._beta_mix_const[2])
        return bn.nansum(ll)


    def MH_error_rates(self, error_type):
        # Set error specific values
        if error_type == 'FP':
            old_error = self.alpha_error
            prior = self.FP_prior
            stdevs = self.FP_sd
        else:
            old_error = self.beta_error
            prior = self.FN_prior
            stdevs = self.FN_sd

        # Get new error from proposal distribution
        std = np.random.choice(stdevs)
        a, b = (0 - old_error) / std, (1 - old_error) / std
        try:
            new_error = truncnorm.rvs(a, b, loc=old_error, scale=std)
        except FloatingPointError:
            new_error = truncnorm.rvs(a, np.inf, loc=old_error, scale=std)

        # Calculate error depending log likelihood
        if error_type == 'FP':
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
            return new_error, [1, 0]

        return old_error, [0, 1]


if __name__ == '__main__':
    print('Here be dragons...')