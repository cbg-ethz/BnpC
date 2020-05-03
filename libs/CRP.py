#!/usr/bin/env python3

import numpy as np
import bottleneck as bn
from scipy.special import gamma, gammaln
from scipy.stats import beta, truncnorm
from scipy.stats import gamma as gamma_fct
from scipy.spatial.distance import pdist, squareform

np.seterr(all='raise')
EPSILON = np.finfo(np.float64).resolution
TMIN = 1e-5
TMAX = 1 - TMIN
log_EPSILON = np.log(EPSILON)

<<<<<<< HEAD
=======
import libs.dpmmIO as io
ct = np.array(io.load_txt('/home/hw/Desktop/BnpC_master/run_04/attachments.txt'))
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3

class CRP:
    """
    Arguments:
        data (np.array): n x m matrix with n cells and m mutations
            containing 0|1|np.nan
<<<<<<< HEAD
        alpha (float): Concentration Parameter for the CRP
=======
        alpha (float): Concentration Parameter for the CRD
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        param_beta ((float, float)): Beta dist parameters used as parameter prior
        FN_error (float): Fixed false negative rate
        FP_error (float): Fixed false positive rate
    """
    def __init__(self, data, DP_alpha=-1, param_beta=[1, 1], FN_error=EPSILON,
                FP_error=EPSILON):
        # Fixed data
        self.data = data
        self.cells_total, self.muts_total = self.data.shape

        # Cluster parameter prior (beta function) parameters
        self.p, self.q = param_beta
        self.param_prior = beta(self.p, self.q)

        mix0 = self.beta_fct(self.p, self.q + 1)
        mix1 = self.beta_fct(self.p + 1, self.q)
        mix0_n = mix0 / (mix0 + mix1)
        mix1_n = mix1 / (mix0 + mix1)
        self._beta_mix_const = np.array(
            [mix0_n, mix1_n, (mix0_n + mix1_n) / 2, np.log((mix0_n + mix1_n) / 2)]
        )

        if self.p == self.q == 1:
            self.beta_prior_uniform = True
        else:
            self.beta_prior_uniform = False

        # Error rates
        self.FP = FP_error
        self.FN = FN_error

<<<<<<< HEAD
        # DP alpha; Prior = Gamma(a, b)
        if DP_alpha[0] < 0 or DP_alpha[1] < 0:
            self.DP_a_gamma = (self.cells_total, 1)
        else:
            self.DP_a_gamma = DP_alpha
=======
        # DP alpha; Prior = Gamma(DP_alpha_a, DP_alpha_b)
        if DP_alpha < 1:
            self.DP_a_gamma = (self.cells_total, 1)
        else:
            self.DP_a_gamma = (DP_alpha, 1)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        self.DP_a_prior = gamma_fct(*self.DP_a_gamma)
        self.DP_a = self.DP_a_prior.rvs()

        # Flexible data - Initialization
        self.CRP_prior = None
        self.assignment = None
        self.parameters = None
        self.cells_per_cluster = None
<<<<<<< HEAD
=======

        # Restricted Gibbs samplers
        self.rg_nc = {}
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3

        # MH proposal stDev's
        self.param_proposal_sd = np.array([0.1, 0.25, 0.5])


    def __str__(self):
        # Fixed values
        out_str = '\nDPMM with:\n' \
<<<<<<< HEAD
            f'\t{self.cells_total} cells\n\t{self.muts_total} mutations\n' \
            f'\tFixed FN rate: {self.FP}\n\tFixed FP rate: {self.FN}\n' \
            '\n\tPriors:\n' \
            f'\tParams.:\tBeta({self.p},{self.q})\n' \
            f'\tCRP a_0:\tGamma({self.DP_a_gamma[0]},{self.DP_a_gamma[1]})\n'
=======
            f'\t{self.cells_total} observations (cells)\n' \
            f'\t{self.muts_total} items (mutations)\n' \
            f'\tFixed FN rate: {self.alpha_error}\n' \
            f'\tFixed FP rate: {self.beta_error}\n' \
            '\n\tPriors:\n' \
            f'\tparams.:\tBeta({self.betaDis_alpha},{self.betaDis_beta})\n' \
            f'\tCRP a_0:\tGamma({self.DP_a_gamma[0]:.1f},1)\n'
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        return out_str


    @staticmethod
    def beta_fct(p, q):
        return gamma(p) * gamma(q) / gamma(p + q)


    @staticmethod
    def log_CRP_prior(n_i, n, a, dtype=np.float64):
        return np.log(n_i, dtype=dtype) - np.log(n - 1 + a, dtype=dtype)


    @staticmethod
    def _normalize_log_probs(probs):
        max_i = bn.nanargmax(probs)
        probs_norm = probs - probs[max_i] - np.log1p(bn.nansum(
            np.exp(probs[np.arange(probs.size) != max_i] - probs[max_i])
        ))
        return np.exp(probs_norm)


    @staticmethod
    def _normalize_log(probs):
        max_i = bn.nanargmax(probs, axis=0)
        try:
            log_probs_norm = probs - probs[max_i] - np.log1p(bn.nansum(
                np.exp(probs[np.arange(probs.size) != max_i] - probs[max_i])
            ))
        except FloatingPointError:
            if probs[0] > probs[1]:
                log_probs_norm = np.array([0, log_EPSILON])
            else:
                log_probs_norm = np.array([log_EPSILON, 0])

        return log_probs_norm


<<<<<<< HEAD
    def init(self, mode='separate', assign=False):
=======
    def init(self, mode='together', assign=False):
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        # All cells in a separate cluster
        if assign:
            self.assignment = np.array(assign)
            self.cells_per_cluster = {}
            cl, cl_size = np.unique(assign, return_counts=True)
            for i in range(cl.size):
                bn.replace(self.assignment, cl[i], i)
                self.cells_per_cluster[i] = cl_size[i]
            self.parameters = self._init_cl_params('assign')
        elif mode == 'separate':
            self.assignment = np.arange(self.cells_total, dtype=int)
            self.cells_per_cluster = {i: 1 for i in range(self.cells_total)}
            self.parameters = self._init_cl_params(mode)
        # All cells in one cluster
        elif mode == 'together':
            self.assignment = np.zeros(self.cells_total, dtype=int)
            self.cells_per_cluster = {0: self.cells_total}
            self.parameters = self._init_cl_params(mode)
        else:
            raise TypeError(f'Unsupported Initialization: {mode}')

        self.init_DP_prior()


<<<<<<< HEAD
    def _init_cl_params(self, mode='together', fkt=1):
        params = np.zeros(self.data.shape)
        if mode == 'separate':
            params = np.random.beta(
                np.nan_to_num(self.p + self.data * fkt, \
                    nan=self._beta_mix_const[0]),
                np.nan_to_num(self.q + (1 - self.data) * fkt, \
                    nan=self._beta_mix_const[1])
=======
    def _init_cl_params(self, mode='together'):
        params = np.zeros(self.data.shape)
        if mode == 'separate':
            params = np.random.beta(
                np.nan_to_num(self.betaDis_alpha + self.data,
                    nan=self.betaDis_alpha),
                np.nan_to_num(self.betaDis_beta + (1 - self.data),
                    nan=self.betaDis_beta)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
            )
        elif mode == 'together':
            params[0] = np.random.beta(
                self.p + bn.nansum(self.data * fkt, axis=0),
                self.q + bn.nansum((1 - self.data) * fkt, axis=0)
            )
        elif mode == 'assign':
            for cl in self.cells_per_cluster:
                cl_data = self.data[np.where(self.assignment == cl)]
                params[cl] = np.random.beta(
                    self.p + bn.nansum(cl_data * fkt, axis=0),
                    self.q + bn.nansum((1 - cl_data)  * fkt, axis=0)
                )
<<<<<<< HEAD
        return np.clip(params, TMIN, TMAX).astype(np.float32)


    def _init_cl_params_new(self, i, fkt=1):
        params = np.random.beta(
            self.p + bn.nansum(self.data[i] * fkt, axis=0),
            self.q + bn.nansum((1 - self.data[i]) * fkt, axis=0)
=======
        return params.astype(np.float32)


    def _init_cl_params_new(self, i, fkt=1):
        return np.random.beta(
            self.betaDis_alpha + bn.nansum(self.data[i] * fkt, axis=0),
            self.betaDis_beta + bn.nansum((1 - self.data[i]) * fkt, axis=0)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        )
        return np.clip(params, TMIN, TMAX).astype(np.float32)


    def init_DP_prior(self):
        cl_vals = np.append(np.arange(1, self.cells_total + 1), self.DP_a)
        CRP_prior = self.log_CRP_prior(cl_vals, self.cells_total, self.DP_a)
        self.CRP_prior = np.append(0, CRP_prior)


    def _calc_ll(self, x, theta, flat=False):
        FN = theta * self._Bernoulli_FN(x)
        FP = (1 - theta) * self._Bernoulli_FP(x)
        ll = np.log(FN + FP)
<<<<<<< HEAD
        bn.replace(ll, np.nan, self._beta_mix_const[3])
=======
        bn.replace(ll, np.nan, self._beta_mix_const[2])
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        if flat:
            return bn.nansum(ll)
        else:
            return bn.nansum(ll, axis=1)


    def _Bernoulli_FN(self, x):
<<<<<<< HEAD
        return (1 - self.FN) ** x * self.FN ** (1 - x)


    def _Bernoulli_FP(self, x):
        return (1 - self.FP) ** (1 - x) * self.FP ** x
=======
        return (1 - self.beta_error) ** x * self.beta_error ** (1 - x)


    def _Bernoulli_FP(self, x):
        return (1 - self.alpha_error) ** (1 - x) * self.alpha_error ** x
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3


    def get_lpost_single(self, cell_id, cl_ids):
        ll = self._calc_ll(self.data[[cell_id]], self.parameters[cl_ids])
        cl_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        lprior = self.CRP_prior[cl_size]
        return ll + lprior


    def get_lpost_single_new_cluster(self):
<<<<<<< HEAD
        FP = self._beta_mix_const[0] * self._Bernoulli_FP(self.data)
        FN = self._beta_mix_const[1] * self._Bernoulli_FN(self.data)
        ll = np.log(FN + FP)
        bn.replace(ll, np.nan, self._beta_mix_const[3])
=======
        FN = self._beta_mix_const[1] * self._Bernoulli_FN(self.data)
        FP = self._beta_mix_const[0] * self._Bernoulli_FP(self.data)
        ll = np.log(FN + FP)
        bn.replace(ll, np.nan, self._beta_mix_const[2])
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        return bn.nansum(ll, axis=1) + self.CRP_prior[-1]


    def get_ll_full(self):
        return self._calc_ll(self.data, self.parameters[self.assignment], True)


    def get_lprior_full(self):
        lprior = self.DP_a_prior.logpdf(self.DP_a) \
            + bn.nansum(self.CRP_prior[
                np.fromiter(self.cells_per_cluster.values(), dtype=int)]
            )
        if not self.beta_prior_uniform:
            cl_ids = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
            # TODO <NB> Is / cl_ids legit?
            lprior += bn.nansum(
                self.param_prior.logpdf(self.parameters[cl_ids])
            ) #/ cl_ids.size

<<<<<<< HEAD
        return lprior
=======
    def get_lpost_full(self):
        return self.get_ll_full() + self.get_lprior_full()
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3


    def update_assignments_Gibbs(self):
        """ Update the assignmen of cells to clusters by Gipps sampling

        """
        new_cl_post = self.get_lpost_single_new_cluster()
        for cell_id in np.random.permutation(self.cells_total):
            # Remove cell from cluster
            old_cluster = self.assignment[cell_id]
            if self.cells_per_cluster[old_cluster] == 1:
                del self.cells_per_cluster[old_cluster]
            else:
                self.cells_per_cluster[old_cluster] -= 1
<<<<<<< HEAD

            cl_ids = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
            # Probability of joining an existing cluster
            post_old = self.get_lpost_single(cell_id, cl_ids)
=======
            self.assignment[cell_id] = -1

            cl_ids = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
            # Probability of joining an existing cluster
            probs = self.get_lpost_single(cell_id, cl_ids)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
            # Probability of starting a new cluster
            post_new = new_cl_post[cell_id]
            # Sample new cluster assignment from posterior
<<<<<<< HEAD
            probs_norm = self._normalize_log_probs(np.append(post_old, post_new))
=======
            probs_norm = self._normalize_log_probs(np.append(probs, new_prob))
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
            cl_ids = np.append(cl_ids, -1)
            new_cluster_id = np.random.choice(cl_ids, p=probs_norm)
            # Start a new cluster
            if new_cluster_id == -1:
                new_cluster_id = self.init_new_cluster(cell_id)
<<<<<<< HEAD
            # Assign to cluster
            self.assignment[cell_id] = new_cluster_id
            try:
                self.cells_per_cluster[new_cluster_id] += 1
            except KeyError:
                self.cells_per_cluster[new_cluster_id] = 1
=======
                self.cells_per_cluster[new_cluster_id] = 0
            # Assign to cluster
            self.assignment[cell_id] = new_cluster_id
            self.cells_per_cluster[new_cluster_id] += 1
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3


    def init_new_cluster(self, cell_id):
        cl_id = self.get_empty_cluster()
<<<<<<< HEAD
=======
        # New parameters based on cell data
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        self.parameters[cl_id] = self._init_cl_params_new([cell_id], 1)
        return cl_id


    def get_empty_cluster(self):
        return next(i for i in range(self.cells_total) 
            if i not in self.cells_per_cluster)


    def update_parameters(self, step_no=None):
        # Iterate over all populated clusters
<<<<<<< HEAD
        declined_t = np.zeros(len(self.cells_per_cluster), dtype= int)
        for i, cl_id in enumerate(self.cells_per_cluster):
            self.parameters[cl_id], _, declined = self.MH_cluster_params(
                self.parameters[cl_id],
                np.argwhere(self.assignment == cl_id).flatten()
            )
            declined_t[i] = declined
        return bn.nansum(declined_t), bn.nansum(self.muts_total - declined_t)
=======
        declined_t = 0
        for cluster_id in self.cells_per_cluster:
            self.parameters[cluster_id], _, declined = self.MH_cluster_params(
                self.parameters[cluster_id],
                self.data[np.where(self.assignment == cluster_id)]
            )
            declined_t += declined
        return declined_t
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3


    def MH_cluster_params(self, old_params, cells, trans_prob=False):
        """ Update cluster parameters

        Arguments:
            old_parameter (float): old val of cluster parameter
            data (np.array): data for cells in the cluster

        Return:
            np.array: New cluster parameter
            float: Sum of MH decision paramters A
            int: Number of declined MH updates
        """

        # Propose new parameter from normal distribution
        std = np.random.choice(self.param_proposal_sd, size=self.muts_total)
        a = (TMIN - old_params) / std 
        b = (TMAX - old_params) / std
        new_params = truncnorm.rvs(a, b, loc=old_params, scale=std) \
            .astype(np.float32)

        A = self._get_log_A(new_params, old_params, cells, a, b, std, trans_prob)
        u = np.log(np.random.random(self.muts_total))

        decline = u >= A
        new_params[decline] = old_params[decline]

        if trans_prob:
            A[decline] = np.log(-1 * np.expm1(A[decline]))
            return new_params, bn.nansum(A), bn.nansum(decline)
        else:
            return new_params, np.nan, bn.nansum(decline)


    def _get_log_A(self, new_params, old_params, cells, a, b, std, clip=False):
        """ Calculate the MH acceptance paramter A
        """
        # Calculate the transition probabilitites
        new_p_target = truncnorm \
            .logpdf(new_params, a, b, loc=old_params, scale=std)

        a_rev, b_rev = (TMIN - new_params) / std, (TMAX - new_params) / std
        old_p_target = truncnorm \
            .logpdf(old_params, a_rev, b_rev, loc=new_params, scale=std)

        # Calculate the log likelihoods
        x = self.data[cells]
        bn.replace(x, np.nan, self._beta_mix_const[2])
        FP = self._Bernoulli_FP(x)
        FN = self._Bernoulli_FN(x)
        new_ll = bn.nansum(
            np.log(new_params * FN + (1 - new_params) * FP), axis=0
        )
        old_ll = bn.nansum(
            np.log(old_params * FN + (1 - old_params) * FP), axis=0
        )
        # Calculate the priors
        if self.beta_prior_uniform:
            new_prior = 0
            old_prior = 0
        else:
            new_prior = self.param_prior.logpdf(new_params)
            old_prior = self.param_prior.logpdf(old_params)

        A = new_ll + new_prior - old_ll - old_prior + old_p_target - new_p_target

        if clip:
            return np.clip(A, a_min=None, a_max=0)
        else:
            return A


    def update_DP_alpha(self, n=None):
        """Escobar, D., West, M. (1995).
        Bayesian Density Estimation and Inference Using Mixtureq.
        Journal of the American Statistical Association, 90, 430.
        Chapter: 6. Learning about a and further illustration
        """
        k = len(self.cells_per_cluster)
        if not n:
            n = self.cells_total

        # Escobar, D., West, M. (1995) - Eq. 14
        eta = np.random.beta(self.DP_a + 1, n)
        w = (self.DP_a_gamma[0] + k - 1) / (n * (self.DP_a_gamma[1] - np.log(eta)))
        pi_eta = w / (1 + w)

        # Escobar, D., West, M. (1995) - Eq. 13
        if np.random.random() < pi_eta:
            new_alpha = np.random.gamma(
                self.DP_a_gamma[0] + k, self.DP_a_gamma[1] - np.log(eta)
            )
        else:
            new_alpha = np.random.gamma(
                self.DP_a_gamma[0] + k - 1, self.DP_a_gamma[1] - np.log(eta)
            )

        self.DP_a = max(new_alpha, 1 + EPSILON)
        self.init_DP_prior()


# ------------------------------------------------------------------------------
# SPLIT MERGE MOVE FOR NON CONJUGATES
# ------------------------------------------------------------------------------

    def update_assignments_split_merge(self, ratios=[.75, .25], step_no=5):
        """ Update the assignmen of cells to clusters by a split-merge move

        """
        cluster_no = len(self.cells_per_cluster)
        if cluster_no == 1:
            return (self.do_split_move(step_no), 0)
        elif cluster_no == self.cells_total:
            return (self.do_merge_move(step_no), 1)
        else:
            move = np.random.choice([0, 1], p=ratios)
            if move == 0:
                return (self.do_split_move(step_no), move)
            else:
                return (self.do_merge_move(step_no), move)


    def do_split_move(self, step_no=5):
        clusters = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
        cluster_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        # Chose larger clusters more often for split move
        cluster_probs = cluster_size / cluster_size.sum()

        # Get cluster with more than one item
        while True:
            clust_i = np.random.choice(clusters, p=cluster_probs)
            cells = np.argwhere(self.assignment == clust_i).flatten()
            if cells.size != 1:
                break

        # Get two random items from the cluster
        obs_i_idx, obs_j_idx = np.random.choice(cells.size, size=2, replace=False)

<<<<<<< HEAD
        cells[0], cells[obs_i_idx] = cells[obs_i_idx], cells[0]
        cells[-1], cells[obs_j_idx] = cells[obs_j_idx], cells[-1]
=======
        S = np.argwhere(self.assignment == clust_i).flatten()
        S = np.delete(S, np.where((S == obs_i) | (S == obs_j)))
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3

        # Eq. 3 in paper, second term
        cluster_idx = np.argwhere(clusters == clust_i).flatten()
        ltrans_prob_size = np.log(cluster_probs[cluster_idx]) \
            - np.log(self.cells_per_cluster[clust_i]) \
            - np.log(self.cells_per_cluster[clust_i] - 1)

        cluster_size_red = np.delete(cluster_size, cluster_idx)
        cluster_size_data = (ltrans_prob_size, cluster_size_red)

        accept, new_assignment, new_params = self.run_rg_nc(
            'split', cells, cluster_size_data, step_no
        )

        if accept:
            clust_new = self.get_empty_cluster()
            # Update parameters
            self.parameters[clust_i] = new_params[0]
            self.parameters[clust_new] = new_params[1]
            # Update assignment
            clust_new_cells = np.append(
                cells[1:-1][np.where(new_assignment == 1)], cells[-1]
            )
            self.assignment[clust_new_cells] = clust_new
            # Update cell-number per cluster
            self.cells_per_cluster[clust_i] -= clust_new_cells.size
            self.cells_per_cluster[clust_new] = clust_new_cells.size

            return [1, 0]
        else:
            return [0, 1]


    def do_merge_move(self, step_no=5):
        clusters = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
        cluster_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        # Chose smaller clusters more often for split move
        cluster_size_inv = 1 / cluster_size
        cluster_probs = cluster_size_inv / cluster_size_inv.sum()
        cl_i, cl_j = np.random.choice(
            clusters, p=cluster_probs, size=2, replace=False
        )

<<<<<<< HEAD
        cells_i = np.argwhere(self.assignment == cl_i).flatten()
        obs_i_idx = np.random.choice(cells_i.size)
        cells_i[0], cells_i[obs_i_idx] = cells_i[obs_i_idx], cells_i[0]

        cells_j = np.argwhere(self.assignment == cl_j).flatten()
        obs_j_idx = np.random.choice(cells_j.size)
        cells_j[-1], cells_j[obs_j_idx] = cells_j[obs_j_idx], cells_j[-1]
=======
        cl_i_cells = np.argwhere(self.assignment == cl_i).flatten()
        i = np.random.choice(cl_i_cells)

        cl_j_cells = np.argwhere(self.assignment == cl_j).flatten()
        j = np.random.choice(cl_j_cells)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3

        cells = np.concatenate((cells_i, cells_j)).flatten()

        # Eq. 6 in paper, second term
        ij_idx = np.argwhere((clusters == cl_j) | (clusters == cl_i)).flatten()
        cluster_size_data = bn.nansum(np.log(cluster_probs[ij_idx])) \
            - bn.nansum(np.log(cluster_size[ij_idx]))

        accept, new_params = self.run_rg_nc(
            'merge', cells, cluster_size_data, step_no
        )

        if accept:
            # Update parameters
            self.parameters[cl_i] = new_params
            # Update Assignment
            self.assignment[cells_j] = cl_i
            # Update cells per cluster
            self.cells_per_cluster[cl_i] += cells_j.size
            del self.cells_per_cluster[cl_j]

            return [1, 0]
        else:
            return [0, 1]


    def run_rg_nc(self, move, cells, size_data, scan_no):   
        # Jain, S., Neal, R. (2007) - Section 4.2: 3,1,1
        self._rg_init_split(cells)
        # Jain, S., Neal, R. (2007) - Section 4.2: 3,2,1
        self.rg_params_merge = self._init_cl_params_new(cells)

        # Jain, S., Neal, R. (2007) - Section 4.2: 3,1,2 / 3,2,2
        # Do restricted Gibbs scans to reach y^{L_{split}} and y^{L_{merge}}
        for scan in range(scan_no):
            self._rg_scan_split(cells)
            self._rg_scan_merge(cells)

        # Jain, S., Neal, R. (2007) - Section 4.2: 4 or 5 (depending on move)
        # Do last scan to reach c_final and calculate Metropolis Hastings prob
        if move == 'split':
            return self._do_rg_split_MH(cells, size_data)
        else:
            return self._do_rg_merge_MH(cells, size_data)


    def _rg_init_split(self, cells, random=False):
        i = cells[0]
        j = cells[-1]
        S = cells[1:-1]
        if random:
            # assign cells to clusters i and j randomly
            self.rg_assignment = np.random.choice([0, 1], size=(S.size))
        else:
<<<<<<< HEAD
            # assign cells to clusters i and j based on likelihood
=======
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
            ll_i = self._calc_ll(self.data[S], np.nan_to_num(self.data[i], nan=0.5))
            ll_j = self._calc_ll(self.data[S], np.nan_to_num(self.data[j], nan=0.5))
            self.rg_assignment = np.where(ll_j > ll_i, 1, 0)
        #initialize cluster parameters
        cells_i = np.append(S[np.argwhere(self.rg_assignment == 0)], i)
        cells_j = np.append(S[np.argwhere(self.rg_assignment == 1)], j)
        cl_i_params = self._init_cl_params_new(cells_i)
        cl_j_params = self._init_cl_params_new(cells_j)
        self.rg_params_split = np.stack([cl_i_params, cl_j_params])


    def _rg_scan_split(self, cells, trans_prob=False):
        prob_cl = self._rg_scan_assign(cells, trans_prob)
        prob_par = self._rg_scan_params(cells, trans_prob)
        
        if trans_prob:
            return prob_cl + prob_par


    def _rg_scan_merge(self, cells, trans_prob=False):
        # Update cluster parameters
        self.rg_params_merge, prob, _ = self.MH_cluster_params(
            self.rg_params_merge, cells, trans_prob
        )
        if trans_prob:
            return prob


    def _rg_scan_params(self, cells, trans_prob=False):
        # Update parameters of cluster i and j
        i = cells[0]
        j = cells[-1]
        S = cells[1:-1]
        prob = np.zeros(2)
        for cl in range(2):
            if cl == 0:
                cl_cells = np.append(S[np.argwhere(self.rg_assignment == 0)], i)
            else:
                cl_cells = np.append(S[np.argwhere(self.rg_assignment == 1)], j)
            self.rg_params_split[cl], prob[cl], _ = self.MH_cluster_params(
                self.rg_params_split[cl], cl_cells, trans_prob
            )

        if trans_prob:
            return prob.sum()


    def _rg_scan_assign(self, cells, trans_prob=False):
        ll = self._rg_get_ll(cells[1:-1], self.rg_params_split)
        n = cells.size
        if trans_prob:
            prob = np.zeros(n - 2)
        
        # Iterate over all obersavtions k
        for cell in np.random.permutation(n - 2):
            self.rg_assignment[cell] = -1
            # Get normalized log probs of assigning an obs. to clusters i or j
<<<<<<< HEAD
            # +1 to compensate obs = -1; +1 for observation j
            n_j = bn.nansum(self.rg_assignment) + 2
            n_i = n - n_j - 1
            log_post = ll[cell] + self.log_CRP_prior([n_i, n_j], n, self.DP_a)
            log_probs = self._normalize_log(log_post)
=======
            # +1 to compensate for obs -1; +1 for observation j
            n_j = np.sum(self.rg_assignment) + 2
            # +1 for observation i
            n_i = S.size - n_j + 1

            log_probs = ll[obs] + self.log_CRP_prior([n_i, n_j], n, self.DP_a)
            log_probs_norm = self._normalize_log(log_probs)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
            # Sample new cluster assignment from posterior
            new_clust = np.random.choice([0, 1], p=np.exp(log_probs))
            
            self.rg_assignment[cell] = new_clust
            if trans_prob:
                prob[cell] = log_probs[new_clust]

        if trans_prob:
            return bn.nansum(prob)


    def _rg_get_ll(self, cells, params):
        ll_i = self._calc_ll(self.data[cells], params[0])
        ll_j = self._calc_ll(self.data[cells], params[1])
        return np.stack([ll_i, ll_j], axis=1)


    def _do_rg_split_MH(self, cells, size_data):
        A = self._get_trans_prob_ratio_split(cells) \
            + self._get_lprior_ratio_split(cells) \
            + self._get_ll_ratio(cells, 'split') \
            + self._get_ltrans_prob_size_ratio_split(*size_data)

        if np.log(np.random.random()) < A:
            return (True, self.rg_assignment, self.rg_params_split)

        return (False, [], [])


    def _do_rg_merge_MH(self, cells, size_data):
        A = self._get_trans_prob_ratio_merge(cells) \
            + self._get_lprior_ratio_merge(cells) \
            + self._get_ll_ratio(cells, 'merge') \
            + self._get_ltrans_prob_size_ratio_merge(size_data)

        if np.log(np.random.random()) < A:
            return (True, self.rg_params_merge)

        return (False, [])


    def _get_trans_prob_ratio_split(self, cells):
        """ [eq. 15 in Jain and Neal, 2007]
        """
        # Do split GS: Launch to proposal state
        GS_split = self._rg_scan_split(cells, trans_prob=True)
        # Do merge GS: Launch to original state
        std = np.random.choice(self.param_proposal_sd, size=self.muts_total)
        a = (TMIN - self.rg_params_merge) / std
        b = (TMAX - self.rg_params_merge) / std

        GS_merge = bn.nansum(
            self._get_log_A(self.parameters[self.assignment[cells[0]]],
                self.rg_params_merge, cells, a, b, std, True)
        )
        return GS_merge - GS_split


    def _get_trans_prob_ratio_merge(self, cells):
        """ [eq. 16 in Jain and Neal, 2007]
        """
        # Do merge GS
        GS_merge = self._rg_scan_merge(cells, trans_prob=True)
        # Do split GS to original merge state
        GS_split = self._rg_get_split_prob(cells)
        return GS_split - GS_merge

<<<<<<< HEAD
=======
        lprior_rate = np.log(self.DP_a) + gammaln(n_i) + gammaln(n_j) \
            - gammaln(S_no + 2)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3

    def _get_lprior_ratio_split(self, cells):
        """ [eq. 7 in Jain and Neal, 2007]
        """
        n = self.rg_assignment.size + 2
        n_j = bn.nansum(self.rg_assignment) + 1
        n_i = n - n_j
        # Cluster assignment prior
        lprior_rate = np.log(self.DP_a) - gammaln(n)
        if n_i > 0:
            lprior_rate += gammaln(n_j)
        if n_j > 0:
            lprior_rate += gammaln(n_i)
        # Cluster parameter prior
        if not self.beta_prior_uniform:
            cl_id = self.assignment[cells[0]]
            # TODO <NB> is the / 2 legit?
            lprior_rate += \
                    (bn.nansum(self.param_prior.logpdf(self.rg_params_split)) / 1 ) \
                - bn.nansum(self.param_prior.logpdf(self.parameters[cl_id]))
        return lprior_rate


    def _get_ll_ratio(self, cells, move):
        """ [eq. 11/eq. 12 in Jain and Neal, 2007]
        """
<<<<<<< HEAD
        i_ids = np.append(
            cells[1:-1][np.argwhere(self.rg_assignment == 0)], cells[0]
        )
        j_ids = np.append(
            cells[1:-1][np.nonzero(self.rg_assignment)], cells[-1]
        )

        ll_i = self._calc_ll(self.data[i_ids], self.rg_params_split[0], True)
        ll_j = self._calc_ll(self.data[j_ids], self.rg_params_split[1], True)
        ll_all = self._calc_ll(self.data[cells], self.rg_params_merge, True)
=======
        i_ids = np.append(S[np.argwhere(self.rg_assignment == 0)], i)
        j_ids = np.append(S[np.argwhere(self.rg_assignment == 1)], j)
        all_ids = np.concatenate([[i], S, [j]])

        ll_i = self._calc_ll(self.data[i_ids], self.rg_params_split[0])
        ll_j = self._calc_ll(self.data[j_ids], self.rg_params_split[1])
        ll_all = self._calc_ll(self.data[all_ids], self.rg_params_merge)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3

        if move == 'split':
            return ll_i + ll_j - ll_all
        else:
            return ll_all - ll_i - ll_j


    def _get_lprior_ratio_merge(self, cells):
        """ [eq. 8 in Jain and Neal, 2007]
        """
<<<<<<< HEAD
        n = cells.size
        n_j = bn.nansum(self.rg_assignment) + 1
        n_i = n - n_j
        # Cluster priors
        lprior_rate = gammaln(n) - np.log(self.DP_a)
        if n_i > 0:
            lprior_rate -= gammaln(n_i)
        if n_j > 0:
            lprior_rate -= gammaln(n_j)
        # Parameter priors
=======
        S_no = assignment.size
        n_j = assignment.sum()
        n_i = S_no - n_j + 2

        lprior_rate = gammaln(S_no + 2) - np.log(self.DP_a) - gammaln(n_i) \
            - gammaln(n_j) \

>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3
        if not self.beta_prior_uniform:
            cl_ids = self.assignment[[cells[0], cells[-1]]]
            # TODO <NB> Is the / 2 legit?
            lprior_rate += \
                    bn.nansum(self.param_prior.logpdf(self.rg_params_merge)) \
                - (bn.nansum(self.param_prior.logpdf(self.parameters[cl_ids])) / 1)
        return lprior_rate


    def _get_ltrans_prob_size_ratio_split(self, ltrans_prob_size, cluster_size):
        n_j = bn.nansum(self.rg_assignment) + 1
        n_i = self.rg_assignment.size + 2 - n_j 

        # Eq. 5 paper, first term
        norm = bn.nansum(1 / np.append(cluster_size, [n_i, n_j]))
        ltrans_prob_rev = np.log(1 / n_i / norm) + np.log(1 / n_j / norm)
        return ltrans_prob_rev - ltrans_prob_size[0]


    def _get_ltrans_prob_size_ratio_merge(self, trans_prob_size):
        # Eq. 6, paper
        try:
            ltrans_prob_rev = -np.log(self.cells_total) \
                - np.log(self.rg_assignment.size - 1)
        except FloatingPointError:
            ltrans_prob_rev = -np.log(self.cells_total)
        return ltrans_prob_rev - trans_prob_size



    def _rg_get_split_prob(self, cells):
        std = np.random.choice(self.param_proposal_sd, size=(2, self.muts_total))
        a = (0 - self.rg_params_split) / std
        b = (1 - self.rg_params_split) / std

        i = cells[0]
        cl_i = self.assignment[i]
        j = cells[-1]
        cl_j = self.assignment[j]
        S = cells[1:-1]
        # Get paramter transition probabilities
        prob_param_i = bn.nansum(self._get_log_A(
            self.parameters[cl_i], self.rg_params_split[0],
            np.append(S[np.argwhere(self.rg_assignment == 0)], i),
            a[0], b[0], std[0], True
        ))
        prob_param_j = bn.nansum(self._get_log_A(
            self.parameters[cl_j], self.rg_params_split[1],
            np.append(S[np.argwhere(self.rg_assignment == 1)], j),
            a[1], b[1], std[1], True
        ))

        # Get assignment transition probabilities
        ll = self._rg_get_ll(
            cells[1:-1], (self.parameters[cl_i], self.parameters[cl_j])
        )
        n = cells.size
        prob_assign = np.zeros(S.size)

        assign = np.where(self.assignment[S] == cl_i, 0, 1)
        # Iterate over all obersavtions k != [i,j]
        for obs in range(S.size):
            self.rg_assignment[obs] = -1
            n_j = bn.nansum(self.rg_assignment) + 2
            n_i = n - n_j - 1

            # Get normalized log probs of assigning an obs. to clusters i or j
<<<<<<< HEAD
            log_post = ll[obs] + self.log_CRP_prior([n_i, n_j], n, self.DP_a)
            log_probs = self._normalize_log(log_post)
=======
            log_probs = ll[obs] + self.log_CRP_prior([n_i, n_j], n, self.DP_a)
            log_probs_norm = self._normalize_log(log_probs)
>>>>>>> e236d01981e9f6d6eaaac588cb734eb088b82aa3

            # assign to original cluster and add probability
            self.rg_assignment[obs] = assign[obs]
            prob_assign[obs] = log_probs[assign[obs]]

        return prob_param_i + prob_param_j + bn.nansum(prob_assign)


if __name__ == '__main__':
    print('Here be dragons....')
