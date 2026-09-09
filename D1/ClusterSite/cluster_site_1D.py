import numpy as np
import sys

from typing import Sequence, Union
from pathlib import Path
from .Iteration.broyden import iterate_broyden

from .core import get_H_spinon, get_rotor_eig_vec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility.helpers import Helpers
from Utility.iteration_cache import IterationCache

# =====================================================================
#   CLUSTER-SITE APPROXIMATION ON THE 1D HUBBARD MODEL (2-SITE CLUSTER)
# ---------------------------------------------------------------------
#   While the single-site approximation scheme has low time-complexity
#   and is the quickest approximation model we have, it severly suffers
#   from failing to accurately model both short- and long-range charge
#   correlations. The cluster-site approximation scheme is a way to
#   include short-range correlation physics.
# ---------------------------------------------------------------------
#   In this class, we implement a 2-site cluster approximation. We
#   cut up the chain into groups of 2-site clusters. Inside a cluster,
#   we diagonalize the rotor-rotor interactions exactly while
#   interactions to sites outside the cluster are approximated with an
#   order parameter as in the single-site approximation model.
# =====================================================================

class ClusterSite1D:
    _CACHE_VERSION = 1.0
    _CACHE_DIR = Path(__file__).resolve().parent / "_cache"
    _MODEL_TAG = "cluster_site_1D"

    """Names of the converged quantities in iteration methods" return order.
    Parameter guesses looked for in nearest parameter lookup, needed for calling fresh iteration"""
    _ITERATION_NAMES = ("h_vec", "J_eff_mat", "K_vec", "t_eff_mat", "renorm_mat", "phi_vec", "nf_vec", "L_vec", "mu", "count")
    _NEAREST_NAMES = ("h_vec", "J_eff_mat", "K_vec", "t_eff_mat")

    _ALGO_NAMES = ["BROYDEN", "NEWTON_GAUGE"]

    def __init__(self, epsilon_vec, U: float, t: float, l_max: int, beta: Union[float, None], 
                 N_sites: int, N_at: int, PBC: bool=False, iteration_method: int=0, 
                 use_cache: bool=True, cache_dir: str=None, try_nearest_params: bool=True, 
                 nearest_params_max_dist: float=0.3, print_cache_results: bool=False) -> None:
        """
        Construct a 1D HUbbard chain with the cluster-site approximation scheme.

        Args:
            epsilon_vec:             On-site energy at every lattice site, length must be N_sites;
            U:                       On-site Hubbard repulsion potential.
            t:                       Nearest-neighbor hopping amplitude, typically set to 1
            l_max:                   Max rotor angular momentum value.
            beta:                    1/kT where k is Boltzmann constant and T is temperature. beta=None is zero temperature.
            N_sites:                 # of lattice sites.
            N_at:                    # of electron states per site.
            PBC:                     Periodic BCs if True, finite BCs if false
            iteration_method:        0 (Broyden), 1 (Newton w/ gauge fix)
            use_cache:               Use IterationCache system to search for previously solved parameters
            cache_dir:               Give custom caching directory, use default if none provided
            try_nearest_params:      Use nearest solved parameter set as starting point if parameters haven't been solved yet
            nearest_params_max_dist: Max distance a parameter set can be in parameter space to be used for initial guesses
            print_cache_results:     Print cache updates
        """
        
        """Cluster & rotor logistics"""
        if N_sites % 2 != 0:
            raise ValueError(f"{self.__class__.__name__}.__init__ | Error: N_sites must be divisible by 2")
        
        self._dim = 2 * l_max + 1                                                       # Dim. of one rotor
        self._dim_cluster = self._dim**2                                      # Dim. of a cluster
        self._N_cluster = N_sites // 2                                                  # Number of clusters
        self._cluster_id = np.arange(N_sites) // 2                              # _cluster_id[i] = cluster # it belongs to
        self._cluster_sites = np.arange(N_sites).reshape(self._N_cluster, 2)            # _cluster_sites[i] = sites belonging to cluster #
        self._cluster_mask = self._cluster_id[:, None] == self._cluster_id[None, :]  # Bool matrix _is_same_cluster[i,j] = True/False whether site i,j are in same cluster
        
        """Hubbard parameters."""
        self._epsilon_vec = epsilon_vec 
        self._U, self._beta, self._l_max, self._N_sites, self._N_at = U, beta, l_max, N_sites, N_at
        self._N_target = int((N_sites * N_at) / 2.0) # Half-filling

        """Hopping matrix."""
        self._t_vec = np.atleast_1d(t)
        self._t_mat = Helpers.hopping_matrix(t, N_sites, PBC)
         
        """Caching and loading."""
        cache = IterationCache(cache_dir if cache_dir is not None else self._CACHE_DIR,
                               model_tag=self._MODEL_TAG, version=self._CACHE_VERSION)
        params = dict(epsvec=epsilon_vec, U=U, t=t, beta=beta, lmax=l_max, Nsites=N_sites, Nat=N_at, PBC=PBC)
        data = cache.load(params, self._ITERATION_NAMES) if use_cache else None
        
        if data is not None:
            # Successful hit w/ stored data, load into memory
            self._cache_status = "HIT"

            self._h_vec        = np.asarray(data["h_vec"], float).copy()
            self._K_vec        = np.asarray( data["K_vec"], float).copy()

            self._J_eff_mat    = data["J_eff_mat"]
            self._t_eff_mat    = data["t_eff_mat"]

            self._renorm_mat   = data["renorm_mat"]
            self._phi_vec      = np.asarray(data["phi_vec"], float).copy()
            self._nf_vec       = np.asarray(data["nf_vec"], float).copy()
            self._L_vec        = np.asarray(data["L_vec"], float).copy()

            self._mu           = float(data["mu"])
            self._count        = int(data["count"])

            self._evals_spinon, self._evecs_spinon       = np.linalg.eigh(get_H_spinon(self._epsilon_vec, self._t_eff_mat, self._h_vec, self._mu))
            self._evals_rotor_vec, self._evecs_rotor_vec = get_rotor_eig_vec(self._U, self._J_eff_mat, self._K_vec, self._h_vec, self._l_max, self._N_sites)

            if print_cache_results:
                print(f"{self.__class__.__name__}.__init__: Cache hit: {cache.path(params).name}")

        else:
            # Failure to hit; default attempt params before nearest params check
            h_vec_guess     = -self._epsilon_vec
            K_vec_guess     = np.full(self._N_sites, -1.0)

            J_eff_mat_guess = 0.9 * self._t_mat
            t_eff_mat_guess = 0.9 * self._t_mat

            self._cache_status = "MISS"
            self._nearest_start_from = None

            # Search for nearest params 
            if use_cache and try_nearest_params:
                near = cache.nearest(params, self._NEAREST_NAMES, must_match=("Nsites", "Nat", "PBC"))
                if near is not None and near["distance"] <= nearest_params_max_dist:
                    near_data = near["data"]

                    h_vec_guess     = np.asarray(near_data["h_vec"], float).copy()
                    K_vec_guess     = np.asarray(near_data["K_vec"], float).copy()
                    
                    J_eff_mat_guess = near_data["J_eff_mat"]
                    t_eff_mat_guess = near_data["t_eff_mat"]

                    self._cache_status = "NEAREST"

                    if print_cache_results:
                        print(f"{self.__class__.__name__}.__init__: Nearest start from "
                              f"{near['params']} (dist = {near['distance']:.3f})")
                else:
                    print(f"{self.__class__.__name__}.__init__: Nearest start not available. Running {self._ALGO_NAMES[iteration_method]} iteration algorithm.")
            
            if iteration_method == 0: # Iteration w/ Broyden's method
                (
                    self._evals_spinon, self._evecs_spinon, self._evals_rotor_vec, 
                    self._evecs_rotor_vec, self._h_vec, self._t_eff_mat, self._J_eff_mat, 
                    self._K_vec, self._renorm_mat, self._phi_vec, self._nf_vec, self._L_vec, self._mu, self._count
                ) = iterate_broyden(
                    self._epsilon_vec, self._U, self._t_mat, h_vec_guess, t_eff_mat_guess,
                    J_eff_mat_guess, K_vec_guess, self._beta, self._l_max, 
                    self._N_sites, self._N_at, self._N_target
                )
            else:
                raise ValueError(f"{self.__class__.__name__}.__init__ | Error: Iteration argument value")
            
            if use_cache:
                cache.save(params, {name: getattr(self, "_" + name) for name in self._ITERATION_NAMES})

    # ====================================================================
    #   CLASS/STATIC METHODS (PRIMARILY CACHE RELATED)
    # ====================================================================
    @classmethod
    def clear_cache(cls, cache_dir: str=None) -> None:
        """Delete all cached iteration results for the decoupled dimer"""
        IterationCache(cache_dir if cache_dir is not None else cls._CACHE_DIR,
                       model_tag=cls._MODEL_TAG, version=cls._CACHE_VERSION).clear()


    # ====================================================================
    #   PRIVATE METHODS
    # ====================================================================
    def __E0(self) -> None:
        """Calculate MF zero-point energy | Includes intra- and inter-cluster hopping
        E0 = sum_{ij, j in C(i)} J^{eff}_{ij}<cos(theta_i - theta_j)>_{theta} - sum_i K_i phi_i + (N/2)sum_i"""
        J_eff = self._J_eff_mat
        Phi   = self._renorm_mat
        phi_vec   = self._phi_vec
        K_vec = self._K_vec
        h_vec = self._h_vec
        N_sites, N_at = self._N_sites, self._N_at

        site1 = np.arange(0, N_sites, 2)  # 0, 2, 4, ...
        site2 = site1 + 1

        E0_intra = 2 * np.sum(J_eff[site1, site2] * Phi[site1, site2])
        E0_inter = -np.sum(K_vec * phi_vec)
        E0_zero    = (N_at/2.0) * np.sum(h_vec)

        return E0_intra + E0_inter + E0_zero

    # ====================================================================
    #   PUBLIC METHODS
    # ====================================================================
    def get_filling_vec(self) -> np.ndarray:
        """Get spinon/electron occupation at each lattice site."""
        return self._nf_vec

    def get_E_gs(self) -> float:
        """Calculate ground-state energy w/o physical subspace projection."""
        E0 = self.__E0()
        E_gs_spinon, E_gs_rotor_vec = self._evals_spinon[0], self._evals_rotor_vec[:,0]
        E_gs = self._N_at * E_gs_spinon + np.sum(E_gs_rotor_vec) + E0
        return E_gs
    
    def get_Z_vec(self) -> np.ndarray:
        """Get quasiparticle spectral weight."""
        return self._Z_vec