#!/usr/bin/env python3

import numpy as np
from scipy.stats import beta, truncnorm
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
        super().__init__(data, DP_alpha, param_beta)
        # Error rate prior
        FP_trunc_a = (0 - FP_mean) / FP_sd
        FP_trunc_b = (1 - FP_mean) / FP_sd
        self.FP_prior = truncnorm(FP_trunc_a, FP_trunc_b, FP_mean, FP_sd)
        self.FP = self.FP_prior.rvs()
        self.FP_sd = np.array([FP_sd * 0.5, FP_sd, FP_sd * 1.5])

        FN_trunc_a = (0 - FN_mean) / FN_sd
        FN_trunc_b = (1 - FN_mean) / FN_sd
        self.FN_prior = truncnorm(FN_trunc_a, FN_trunc_b, FN_mean, FN_sd)
        self.FN = self.FN_prior.rvs()
        self.FN_sd = np.array([FN_sd * 0.5, FN_sd, FN_sd * 1.5])


    def __str__(self):
        out_str = '\nDPMM with:\n' \
            f'\t{self.cells_total} cells\n\t{self.muts_total} mutations\n' \
            f'\tlearning errors\n' \
            '\n\tPriors:\n' \
            f'\tparams.:\tBeta({self.p},{self.q})\n' \
            f'\tCRP a_0:\tGamma({self.DP_a_gamma[0]},{self.DP_a_gamma[1]})\n' \
            f'\tFP:\t\ttrunc norm({self.FP_prior.args[2]},{self.FP_prior.args[3]})\n' \
            f'\tFN:\t\ttrunc norm({self.FN_prior.args[2]},{self.FN_prior.args[3]})\n'
            
        return out_str


    def get_lprior_full(self):
        return super().get_lprior_full() \
            + self.FP_prior.logpdf(self.FP) \
            + self.FN_prior.logpdf(self.FN)


    def update_error_rates(self):
        FP_count = self.MH_error_rates('FP')
        FN_count = self.MH_error_rates('FN')
        return FP_count, FN_count


    def get_ll_full_error(self, FP, FN):
        par = self.parameters[self.assignment]
        FN = par * (1 - FN) ** self.data * FN ** (1 - self.data) 
        FP = (1 - par) * (1 - FP) ** (1 - self.data) * FP ** self.data
        ll = np.log(FN + FP)
        bn.replace(ll, np.nan, self._beta_mix_const[2])
        return bn.nansum(ll)


    def MH_error_rates(self, error_type):
        # Set error specific values
        if error_type == 'FP':
            old_error = self.FP
            prior = self.FP_prior
            stdevs = self.FP_sd
        else:
            old_error = self.FN
            prior = self.FN_prior
            stdevs = self.FN_sd

        # Get new error from proposal distribution
        std = np.random.choice(stdevs)
        a = (0 - old_error) / std
        b = (1 - old_error) / std
        try:
            new_error = truncnorm.rvs(a, b, loc=old_error, scale=std)
        except FloatingPointError:
            new_error = truncnorm.rvs(a, np.inf, loc=old_error, scale=std)

        # Calculate transition probabilitites
        new_p_target = truncnorm \
            .logpdf(new_error, a, b, loc=old_error, scale=std)
        a_rev, b_rev = (0 - new_error) / std, (1 - new_error) / std
        old_p_target = truncnorm \
            .logpdf(old_error, a_rev, b_rev, loc=new_error, scale=std)

        # Calculate likelihood
        if error_type == 'FP':
            new_ll = self.get_ll_full_error(new_error, self.FN)
            old_ll = self.get_ll_full_error(old_error, self.FN)
        else:
            new_ll = self.get_ll_full_error(self.FP, new_error)
            old_ll = self.get_ll_full_error(self.FP, old_error)

        # Calculate priors
        new_prior = prior.logpdf(new_error)
        old_prior = prior.logpdf(old_error)

        # Calculate MH decision treshold
        A = new_ll + new_prior - old_ll - old_prior + old_p_target - new_p_target

        if np.log(np.random.random()) < A:
            old_error = new_error
            return [1, 0]

        return [0, 1]


if __name__ == '__main__':
    print('Here be dragons...')