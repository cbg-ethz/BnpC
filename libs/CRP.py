#!/usr/bin/env python3

from datetime import datetime, timedelta
import numpy as np
import numpy.ma as ma
import bottleneck as bn
from scipy.special import gamma, gammaln
from scipy.stats import beta, truncnorm
from scipy.stats import gamma as gamma_fct
from scipy.spatial.distance import pdist, squareform

try:
    from libs import utils as ut
    from libs import dpmmIO as io
except ImportError:
    import utils as ut
    import libs.dpmmIO as io


np.seterr(all='raise')
EPSILON = np.finfo(np.float64).resolution
log_EPSILON = np.log(EPSILON)

class CRP:
    """
    Arguments:
        alpha (int): Concentration Parameter for the CRD
        data (np.array): n x m matrix with n cells and m mutations
            containing 0|1|np.nan
    """
    def __init__(self, data, DP_alpha=-1, param_beta_a=1, param_beta_b=1,
                FN_error=EPSILON, FP_error=EPSILON):
        # Cluster parameter prior (beta function) parameters
        self.betaDis_alpha = param_beta_a
        self.betaDis_beta = param_beta_b
        self.param_prior = beta(self.betaDis_alpha, self.betaDis_beta)

        if self.betaDis_alpha == self.betaDis_beta == 1:
            self.beta_prior_uniform = True
        else:
            self.beta_prior_uniform = False

        bmc0 = self.beta_fct(self.betaDis_alpha, self.betaDis_beta + 1)
        bmc1 = self.beta_fct(self.betaDis_alpha + 1, self.betaDis_beta)
        self._beta_mix_const = np.array([bmc0, bmc1, np.log((bmc0 + bmc1) / 2)])

        # Fixed data
        self.data = data
        self.cells_total, self.muts_total = self.data.shape
        self._clusters = set(range(self.cells_total))
        # 2 x cell number matrix with 0, 1 counts
        self.muts_per_cell = np.stack(
            [np.where(self.data == 0, True, False).sum(axis=1),
            bn.nansum(data, axis=1), np.isnan(self.data).sum(axis=1)]
        ).astype(np.int32)

        # Error rates
        self.alpha_error = FP_error
        self.beta_error = FN_error

        # DP alpha; Prior = Gamma(DP_alpha_a, DP_alpha_b)
        if DP_alpha < 1:
            self.DP_alpha_a = np.log(self.cells_total)
        else:
            self.DP_alpha_a = DP_alpha
        self.DP_alpha_b = 1
        self.DP_alpha_prior = gamma_fct(self.DP_alpha_a, self.DP_alpha_b)
        self.DP_alpha = self.DP_alpha_prior.rvs()

        # Flexible data - Initialization
        self.CRP_prior = None
        self.assignment = None
        self.parameters = None
        self.cells_per_cluster = None

        # Restricted Gibbs samplers
        self.rg_nc = {}
        # self.rg_merge = RestrictedGibbsMerge_nonconjugate(*self)
        # self.rg_split = RestrictedGibbsSplit_nonconjugate(*self)

        # MH proposal stDev's
        self.param_proposal_sd = np.array([0.1, 0.25, 0.5])


    def __str__(self):
        # Fixed values
        out_str = '\nDPMM with:\n' \
            '\t{} observations (cells)\n\t{} items (mutations)\n' \
            '\tFixed FN rate: {}\n\tFixed FP rate: {}\n' \
                .format(self.cells_total, self.muts_total,
                    self.alpha_error,self.beta_error)
        # Prior distributions
        out_str += '\n\tPriors:\n' \
            '\tparams.:\tBeta({},{})\n\tCRP a_0:\tGamma({:.1f},1)\n' \
                .format(self.betaDis_alpha, self.betaDis_beta, self.DP_alpha_a)
        return out_str


    @staticmethod
    def beta_fct(a, b):
        # B(a, b) = G(a) * G(b) / G(a + b)
        return gamma(a) * gamma(b) / gamma(a + b)


    @staticmethod
    def log_CRP_prior(n_i, n, a, dtype=np.float32):
        return np.log(n_i, dtype=dtype) - np.log(n - 1 + a, dtype=dtype)


    @staticmethod
    def _normalize_log_probs(probs):
        """
        Arguments:
            probs (np.array): Probabilities in log space

        Returns:
            np.array: Probabilities in normal space

        """
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


    def init(self, mode='together'):
        # All cells in a seperate cluster
        if mode == 'seperate':
            self.assignment = np.arange(self.cells_total, dtype=int)
            self.cells_per_cluster = {i: 1 for i in range(self.cells_total)}
            self.parameters = self._init_cl_params(self.data)
        # All cells in one cluster
        elif mode == 'together':
            self.assignment = np.zeros(self.cells_total, dtype=int)
            self.cells_per_cluster = {0: self.cells_total}
            self.parameters = np.zeros(self.data.shape, dtype=np.float32)
            self.parameters[0] = self._init_cl_params(self.data, single=True)
        else:
            raise TypeError('Unsupported Initialization: {}'.format(mode))
        self.init_DP_prior()


    def _init_cl_params(self, data, single=False):
        if single:
            return np.random.beta(
                self.betaDis_alpha + bn.nansum(self.data, axis=0),
                self.betaDis_beta + bn.nansum(1 - self.data, axis=0)
            ).astype(np.float32)
        else:
            return np.random.beta(
                np.nan_to_num(self.betaDis_alpha + data, nan=self.betaDis_alpha),
                np.nan_to_num(self.betaDis_beta + (1 - data), nan=self.betaDis_beta)
            ).astype(np.float32)


    def init_DP_prior(self):
        cl_vals = np.append(np.arange(1, self.cells_total + 1), self.DP_alpha)
        CRP_prior = self.log_CRP_prior(cl_vals, self.cells_total, self.DP_alpha)
        self.CRP_prior = np.append(0, CRP_prior)


    def get_lpost_single_new_cluster(self):
        FP = self._beta_mix_const[0] * self._Bernoulli_FP(self.data)
        FN = self._beta_mix_const[1] * self._Bernoulli_FN(self.data)
        nan = self._beta_mix_const[2] * self.muts_per_cell[2]
        return bn.nansum(np.log(FP + FN), axis=1) + nan + self.CRP_prior[-1]


    def get_ll_cl(self, cell_id, cluster_id):
        return  self._calc_ll(self.data[cell_id], self.parameters[cluster_id],
            self.muts_per_cell[2][cell_id])


    def _calc_ll(self, x, theta, nan_no, flat=False):
        # Bernoulli for FP + Bernoulli for FN
        FN = theta * self._Bernoulli_FN(x)
        FP = (1 - theta) * self._Bernoulli_FP(x)
        nan = nan_no * self._beta_mix_const[2]
        if flat:
            return bn.nansum(np.log(FN + FP), axis=0) + nan
        else:
            return bn.nansum(np.log(FN + FP), axis=1) + nan


    def _Bernoulli_FN(self, cell_data):
        return (1 - self.beta_error) ** cell_data \
            * self.beta_error ** (1 - cell_data)


    def _Bernoulli_FP(self, cell_data):
        return (1 - self.alpha_error) ** (1 - cell_data) \
            * self.alpha_error ** cell_data


    def get_lpost_single(self, cell_id):
        cl_ids = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
        cl_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        return self.get_ll_cl([cell_id], cl_ids) + self.CRP_prior[cl_size]


    def get_ll_full(self):
        ll = 0
        for cluster_id in self.cells_per_cluster:
            cell_ids = np.where(self.assignment == cluster_id)
            ll += bn.nansum(self.get_ll_cl(cell_ids, cluster_id))
        return ll


    def get_lprior(self):
        lprior = self.DP_alpha_prior.logpdf(self.DP_alpha) \
            + bn.nansum(self.CRP_prior[
                np.fromiter(self.cells_per_cluster.values(), dtype=int)]
            )
        if not self.beta_prior_uniform:
            lprior += bn.nansum(
                self.param_prior.logpdf(self.parameters[self.assignment])
            )
        return lprior


    def get_lpost_full(self):
        return self.get_ll_full() + self.get_lprior()


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
            self.assignment[cell_id] = -1

            cluster_ids = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
            # Probability of joining an existing cluster
            probs = self.get_lpost_single(cell_id)
            # Probability of starting a new cluster
            new_prob = new_cl_post[cell_id]
            # Sample new cluster assignment from posterior
            probs_norm = self._normalize_log_probs(np.append(probs, new_prob))
            new_cluster_id = np.random.choice(
                np.append(cluster_ids, -1), p=probs_norm
            )
            # Start a new cluster
            if new_cluster_id == -1:
                new_cluster_id = self.init_new_cluster(cell_id)
                self.cells_per_cluster[new_cluster_id] = 0
            # Assign to cluster
            self.assignment[cell_id] = new_cluster_id
            self.cells_per_cluster[new_cluster_id] += 1


    def init_new_cluster(self, cell_id):
        # New cluster id = smallest possible not occupied number
        cluster_id = self.get_empty_cluster()
        # New parameters based on cell data
        self.parameters[cluster_id] = self._init_cl_params(self.data[cell_id])
        return cluster_id


    def get_empty_cluster(self):
        return min(self._clusters.difference(self.assignment))


    def update_parameters(self, step_no=None):
        # Iterate over all populated clusters
        for cluster_id in self.cells_per_cluster:
            self.parameters[cluster_id], _, declined = self.MH_cluster_params(
                self.parameters[cluster_id],
                self.data[np.where(self.assignment == cluster_id)]
            )
        return declined


    def MH_cluster_params(self, old_params, data, trans_prob=False):
        """ Update cluster parameters

        Arguments:
            old_parameter (float): old val of cluster parameter
            data (np.array): data for cells in the cluster

        Return:
            New cluster parameter
        """

        # Propose new parameter from normal distribution
        std = np.random.choice(self.param_proposal_sd, size=self.muts_total)
        a, b = (0 - old_params) / std, (1 - old_params) / std
        new_params = truncnorm.rvs(a, b, loc=old_params, scale=std)\
            .astype(np.float32)

        A = self._get_log_A(new_params, old_params, data, a, b, std, trans_prob)

        u = np.log(np.random.random(self.muts_total))

        decline = u >= A
        new_params[decline] = old_params[decline]

        if trans_prob:
            A[decline] = np.log(-1 * np.expm1(A[decline]))
            return new_params, bn.nansum(A), decline.sum()
        else:
            return new_params, None, decline.sum()


    def _get_log_A(self, new_params, old_params, data, a, b, std, clip=False):
        """ Calculate the MH acceptance paramter A
        """
        # Calculate the transition probabilitites
        new_p_target = truncnorm \
            .logpdf(new_params, a, b, loc=old_params, scale=std)

        a_rev, b_rev = (0 - new_params) / std, (1 - new_params) / std
        old_p_target = truncnorm \
            .logpdf(old_params, a_rev, b_rev, loc=new_params, scale=std)

        # Calculate the log likelihoods
        FP = self._Bernoulli_FP(data)
        FN = self._Bernoulli_FN(data)
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
        eta = np.random.beta(self.DP_alpha + 1, n)
        w = (self.DP_alpha_a + k - 1) / (n * (self.DP_alpha_b - np.log(eta)))
        pi_eta = w / (1 + w)

        # Escobar, D., West, M. (1995) - Eq. 13
        if np.random.random() < pi_eta:
            new_alpha = np.random.gamma(
                self.DP_alpha_a + k, self.DP_alpha_b - np.log(eta)
            )
        else:
            new_alpha = np.random.gamma(
                self.DP_alpha_a + k - 1, self.DP_alpha_b - np.log(eta)
            )

        self.DP_alpha = max(new_alpha, 1 + EPSILON)
        self.init_DP_prior()


# ------------------------------------------------------------------------------
# SPLIT MERGE MOVE FOR NON CONJUGATES
# ------------------------------------------------------------------------------

    def update_assignments_split_merge(self, ratios=[.75, .25], step_no=5):
        """ Update the assignmen of cells to clusters by a split-merge move

        """
        clusters = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
        cluster_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        cluster_no = len(clusters)
        if cluster_no == 1:
            return (self.do_split_move(clusters, cluster_size, step_no), 0)
        elif cluster_no == self.cells_total:
            return (self.do_merge_move(clusters, cluster_size, step_no), 1)
        else:
            move = np.random.choice([0, 1], p=ratios)
            if move == 0:
                return (self.do_split_move(clusters, cluster_size, step_no), move)
            else:
                return (self.do_merge_move(clusters, cluster_size, step_no), move)


    def do_split_move(self, clusters, cluster_size, step_no):
        # Chose larger clusters more often for split move
        cluster_probs = cluster_size / cluster_size.sum()
        clust_i = np.random.choice(clusters, p=cluster_probs)
        clust_i_idx = np.argwhere(self.assignment == clust_i).flatten()

        # Get cluster with more than one item
        while len(clust_i_idx) == 1:
            clust_i = np.random.choice(clusters, p=cluster_probs)
            clust_i_idx = np.argwhere(self.assignment == clust_i).flatten()

        cluster_idx = np.argwhere(clusters == clust_i).flatten()

        # Get two random items from the cluster
        obs_i, obs_j = np.random.choice(clust_i_idx, size=2, replace=False)

        S = np.argwhere(self.assignment == clust_i).flatten()
        S = np.delete(S, np.where((S == obs_i) | (S == obs_j)))

        # Eq. 5 in paper, second term
        ltrans_prob_size = np.log(cluster_probs[cluster_idx]) \
            - np.log(cluster_size[cluster_idx]) \
            - np.log(cluster_size[cluster_idx] -1)

        cluster_size_red = np.delete(cluster_size, cluster_idx)
        cluster_size_data = (ltrans_prob_size, cluster_size_red)

        accept, new_assignment, new_params = self.run_rg_nc(
            'split', obs_i, obs_j, S, self.parameters[clust_i],
            cluster_size_data, step_no
        )

        if accept:
            new_cluster = self.get_empty_cluster()
            # Update parameters
            self.parameters[clust_i] = new_params[0]
            self.parameters[new_cluster] = new_params[1]
            # Update assignment
            new_cluster_cells = np.append(
                S[np.where(new_assignment == 1)], obs_j
            )
            self.assignment[new_cluster_cells] = new_cluster
            # Update cell-number per cluster
            self.cells_per_cluster[new_cluster] = new_cluster_cells.size
            self.cells_per_cluster[clust_i] -= new_cluster_cells.size
            return [1, 0]
        else:
            return [0, 1]


    def do_merge_move(self, clusters, cluster_size, step_no):
        # Chose smaller clusters more often for split move
        cluster_size_inv = 1 / cluster_size
        cluster_probs = cluster_size_inv / cluster_size_inv.sum()
        cl_i, cl_j = np.random.choice(
            clusters, p=cluster_probs, size=2, replace=False
        )

        cl_i_cells = np.argwhere(self.assignment == cl_i).flatten()
        i = np.random.choice(cl_i_cells)

        cl_j_cells = np.argwhere(self.assignment == cl_j).flatten()
        j = np.random.choice(cl_j_cells)

        S = np.concatenate((cl_i_cells, cl_j_cells)).flatten()
        S = np.delete(S, np.where((S == i) | (S == j)))

        params = (self.parameters[cl_i], self.parameters[cl_j])
        assignment = np.append(
            np.zeros(cl_i_cells.size - 1, dtype=int),
            np.ones(cl_j_cells.size - 1, dtype=int)
        )

        ij_idx = np.argwhere((clusters == cl_j) | (clusters == cl_i)).flatten()
        # Eq. 6 in paper, second term
        cluster_size_data = bn.nansum(np.log(cluster_probs[ij_idx])) \
            - bn.nansum(np.log(cluster_size[ij_idx]))

        accept, new_params = self.run_rg_nc(
            'merge', i, j, S, (params, assignment), cluster_size_data, step_no
        )

        if accept:
            # Update parameters
            self.parameters[cl_i] = new_params
            # Update Assignment
            self.assignment[cl_j_cells] = cl_i
            # Update cells per cluster
            self.cells_per_cluster[cl_i] += cl_j_cells.size
            del self.cells_per_cluster[cl_j]
            return [1, 0]
        else:
            return [0, 1]


    def run_rg_nc(self, move, obs_i, obs_j, S, params, size_data, scan_no):
        self.rg_nc['i'] = self.data[obs_i]
        self.rg_nc['j'] = self.data[obs_j]
        self.rg_nc['S'] = self.data[S]
        self.rg_nc['S_no'] = S.size
        iSj = self.data[np.concatenate([[obs_i], S, [obs_j]])]

        # Jain, S., Neal, R. (2007) - Section 4.2: 3,1,1
        self._rg_init_split(S, obs_i, obs_j)
        # Jain, S., Neal, R. (2007) - Section 4.2: 3,1,2
        # Do restricted Gibbs scans to reach y^{L_{split}}
        for scan in range(scan_no):
            self._rg_scan_split(S, obs_i, obs_j)

        # Jain, S., Neal, R. (2007) - Section 4.2: 3,2,1
        self.rg_params_merge = self._init_cl_params(iSj, single=True)
        # Jain, S., Neal, R. (2007) - Section 4.2: 3,2,2
        # Do restricted Gibbs scans to reach y^{L_{merge}}
        for scan in range(scan_no):
            self._rg_scan_merge(iSj)

        # Jain, S., Neal, R. (2007) - Section 4.2: 4 or 5 (depending on move)
        # Do last scan to reach c_final and calculate Metropolis Hastings prob
        return self._do_MH(iSj, S, obs_i, obs_j, move, params, size_data)


    def _rg_init_split(self, S, i, j, random=False):
        # assign cells to clusters i and j
        if random:
            self.rg_assignment = np.random.choice([0, 1], size=(S.size))
        else:
            ll_i = self._calc_ll(self.data[S], self.data[i],
                self.muts_per_cell[2][S])
            ll_j = self._calc_ll(self.data[S], self.data[j],
                self.muts_per_cell[2][S])
            self.rg_assignment = np.where(ll_j > ll_i, 1, 0)

        #initialize cluster parameters
        cl_i_params = self._init_cl_params(
            self.data[np.insert(S[np.argwhere(self.rg_assignment == 0)], 0, i)],
            single=True
        )
        cl_j_params = self._init_cl_params(
            self.data[np.insert(S[np.argwhere(self.rg_assignment == 1)], 0, j)],
            single=True
        )
        self.rg_params_split = np.stack([cl_i_params, cl_j_params])


    def _rg_scan_split(self, S, i, j, trans_prob=False):
        log_trans_prob_p = self._rg_scan_params(S, i, j, trans_prob)
        log_trans_prob_a = self._rg_scan_assign(S, trans_prob)
        if trans_prob:
            return log_trans_prob_a + log_trans_prob_p


    def _rg_scan_merge(self, iSj, trans_prob=False):
        # Update cluster parameters
        new_params = self.MH_cluster_params(self.rg_params_merge, iSj, trans_prob)
        self.rg_params_merge = new_params[0]
        if trans_prob:
            return bn.nansum(new_params[1])


    def _rg_scan_params(self, S, i, j, trans_prob=False):
        # Update parameters of cluster i
        new_params_i = self.MH_cluster_params(
            self.rg_params_split[0],
            self.data[np.append(S[np.argwhere(self.rg_assignment == 0)], i)],
            trans_prob
        )
        self.rg_params_split[0] = new_params_i[0]

        # Update parameters of cluster j
        new_params_j = self.MH_cluster_params(
            self.rg_params_split[1],
            self.data[np.append(S[np.argwhere(self.rg_assignment == 1)], j)],
            trans_prob
        )
        self.rg_params_split[1] = new_params_j[0]

        if trans_prob:
            return new_params_i[1] + new_params_j[1]


    def _rg_scan_assign(self, S, trans_prob=False):
        ll = self._rg_get_ll(S, self.rg_params_split)
        log_trans_prob = 0
        n = S.size + 2
        # Iterate over all obersavtions k
        for obs in np.random.permutation(S.size):
            self.rg_assignment[obs] = -1
            # Get normalized log probs of assigning an obs. to clusters i or j
            # +1 to compensate for obs -1; +1 for observation j
            n_j = np.sum(self.rg_assignment) + 2
            # +1 for observation i
            n_i = S.size - n_j + 1

            log_probs = ll[obs] + self.log_CRP_prior([n_i, n_j], n, self.DP_alpha)
            log_probs_norm = self._normalize_log(log_probs)
            # Sample new cluster assignment from posterior
            new_cluster = np.random.choice([0, 1], p=np.exp(log_probs_norm))
            self.rg_assignment[obs] = new_cluster

            if trans_prob:
                if new_cluster == 0:
                    log_trans_prob += log_probs_norm[0]
                else:
                    log_trans_prob += log_probs_norm[1]

        return log_trans_prob


    def _rg_get_ll(self, cells, params):
        ll_i = self._calc_ll(
            self.data[cells], params[0], self.muts_per_cell[2, cells])
        ll_j = self._calc_ll(
            self.data[cells], params[1], self.muts_per_cell[2, cells])
        return np.stack([ll_i, ll_j], axis=1)


    def _do_MH(self, iSj, S, i, j, move, params, size_data):
        # Jain, S., Neal, R. (2007) - Section 4.2: 5 (b)
        # Do scan and get transition probability: launch state -> final state
        if move == 'split':
            return self._do_rg_split_MH(iSj, S, i, j, params, size_data)
        else:
            return self._do_rg_merge_MH(iSj, S, i, j, params, size_data)


    def _do_rg_split_MH(self, iSj, S, i, j, params, size_data):
        # Do scan and get transition probability: launch state -> final state
        split_prob_dens = self._rg_scan_split(S, i, j, trans_prob=True)
        # Do scan and get transition probability: launch state -> original state
        std = np.random.choice(self.param_proposal_sd, size=self.muts_total)
        a, b = (0 - self.rg_params_merge) / std, (1 - self.rg_params_merge) / std
        corr_prop_dens = bn.nansum(
            self._get_log_A(params, self.rg_params_merge, iSj, a, b, std, True)
        )

        # First term: [eq. 15 in Jain and Neal, 2007]
        A = corr_prop_dens - split_prob_dens \
            + self._get_lprior_ratio_split(self.rg_assignment, params) \
            + self._get_ll_ratio(S, i, j, 'split') \
            + self._get_ltrans_prob_size_ratio_split(*size_data, )

        if np.log(np.random.random()) < A:
            return (True, self.rg_assignment, self.rg_params_split)

        return (False, [], [])


    def _do_rg_merge_MH(self, iSj, S, i, j, params, size_data):
        # Do scan and get transition probability: launch state -> final state
        merge_prop_dens = self._rg_scan_merge(iSj, trans_prob=True)
        # Do scan and get transition probability: launch state -> original state
        corr_prop_dens = self._rg_get_split_prob(S, i, j, *params)

        # First term: [eq. 16 in Jain and Neal, 2007]
        A = corr_prop_dens - merge_prop_dens \
            + self._get_lprior_ratio_merge(params[1], params[0]) \
            + self._get_ll_ratio(S, i, j, 'merge') \
            + self._get_ltrans_prob_size_ratio_merge(S, size_data)

        if np.log(np.random.random()) < A:
            return (True, self.rg_params_merge)

        return (False, [])


    def _get_ltrans_prob_ratio(self, prob_split, prob_merge, move):
        """ [eq. 15/16 in Jain and Neal, 2007]
        """
        if move == 'split':
            return prob_merge - prob_split
        else:
            return prob_split - prob_merge


    def _get_lprior_ratio_split(self, assignment, params):
        """ [eq. 7 in Jain and Neal, 2007]
        """
        S_no = assignment.size
        n_j = assignment.sum() + 1
        n_i = S_no - n_j + 2

        lprior_rate = np.log(self.DP_alpha) + gammaln(n_i) + gammaln(n_j) \
            - gammaln(S_no + 2)

        if not self.beta_prior_uniform:
            lprior_rate += bn.nansum(
                    self.param_prior.logpdf(self.rg_params_split)) \
                - bn.nansum(self.param_prior.logpdf(params))
        return lprior_rate


    def _get_ll_ratio(self, S, i, j, move):
        """ [eq. 11/eq. 12 in Jain and Neal, 2007]
        """
        i_ids = np.append(S[np.argwhere(self.rg_assignment == 0)], i)
        j_ids = np.append(S[np.argwhere(self.rg_assignment == 1)], j)
        all_ids = np.concatenate([[i], S, [j]])

        ll_i = self._calc_ll(self.data[i_ids], self.rg_params_split[0],
            self.muts_per_cell[2, i_ids])

        ll_j = self._calc_ll(self.data[j_ids], self.rg_params_split[1],
            self.muts_per_cell[2, j_ids])

        ll_all = self._calc_ll(self.data[all_ids], self.rg_params_merge,
            self.muts_per_cell[2, all_ids])

        if move == 'split':
            return bn.nansum(ll_i) + bn.nansum(ll_j) - bn.nansum(ll_all)
        else:
            return bn.nansum(ll_all) - bn.nansum(ll_i) - bn.nansum(ll_j)


    def _get_lprior_ratio_merge(self, assignment, params):
        """ [eq. 8 in Jain and Neal, 2007]
        """
        S_no = assignment.size
        n_j = assignment.sum()
        n_i = S_no - n_j + 2

        lprior_rate = gammaln(S_no + 2) - np.log(self.DP_alpha) - gammaln(n_i) \
            - gammaln(n_j) \

        if not self.beta_prior_uniform:
            lprior_rate += bn.nansum(
                self.param_prior.logpdf(self.rg_params_merge) \
                    - self.param_prior.logpdf(params)
            )
        return lprior_rate


    def _get_ltrans_prob_size_ratio_split(self, ltrans_prob_size, cluster_size):
        n_j = self.rg_assignment.sum() + 1
        n_i = self.rg_assignment.size - n_j + 2

        # Eq. 5 paper, first term
        norm = bn.nansum(1 / np.append(cluster_size, [n_i, n_j]))
        ltrans_prob_rev = np.log(1 / n_i / norm) + np.log(1 / n_j / norm)
        return ltrans_prob_rev - ltrans_prob_size[0]


    def _get_ltrans_prob_size_ratio_merge(self, S, trans_prob_size):
        # Eq. 6, paper
        ltrans_prob_rev = np.log(self.cells_total) - np.log(S.size + 1)
        return ltrans_prob_rev - trans_prob_size



    def _rg_get_split_prob(self, S, i, j, params, assign):
        std = np.random.choice(self.param_proposal_sd, size=(2, self.muts_total))
        a, b = (0 - self.rg_params_split) / std, (1 - self.rg_params_split) / std

        log_prob_params = \
            bn.nansum(self._get_log_A(
                params[0][0], self.rg_params_split[0],
                self.data[np.append(S[np.argwhere(self.rg_assignment == 0)], i)],
                a[0], b[0], std[0], True
            )) + bn.nansum(self._get_log_A(
                params[0][1], self.rg_params_split[1],
                self.data[np.append(S[np.argwhere(self.rg_assignment == 1)], j)],
                a[1], b[1], std[1], True
            ))

        ll = self._rg_get_ll(S,  params[0])
        log_prob_assign = 0
        n = S.size + 2
        # Iterate over all obersavtions k
        for obs in range(S.size):
            self.rg_assignment[obs] = -1
            n_j = np.sum(self.rg_assignment) + 2
            n_i = S.size - n_j + 1

            # Get normalized log probs of assigning an obs. to clusters i or j
            log_probs = ll[obs] + self.log_CRP_prior([n_i, n_j], n, self.DP_alpha)
            log_probs_norm = self._normalize_log(log_probs)

            # assign to original cluster and add probability
            self.rg_assignment[obs] = assign[obs]
            log_prob_assign += log_probs_norm[assign[obs]]

        return log_prob_params + log_prob_assign


if __name__ == '__main__':
    print('Here be dragons....')