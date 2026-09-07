import numpy as np
import sys

from typing import Sequence, Union
from pathlib import Path
from .Iteration.broyden import iterate_broyden
from .Iteration.newton_gauge import iterate_newton_gauge

from .core import get_H_spinon, get_rotor_eig_vec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility.helpers import Helpers
from Utility.iteration_cache import IterationCache

# ====================================================================
#   SINGLE-SITE APPROXIMATION ON THE 1D HUBBARD CHAIN
# --------------------------------------------------------------------
#   A mean-field approximation is made on the exact slave rotor
#   Hubbard Hamiltoniain, giving a spinon and rotor Hamiltonian.
#   Another approximation is made to decouple all rotors, leading
#   to a tower of rotor states in the |l> basis at each lattice site.
# --------------------------------------------------------------------
#   At each lattice site, there is a tower of rotor basis states |l>
#   A rotor state is a superposition \sum \alpha_l |l> ranging from
#   l \in [-l_max, l_max]. This file features all the functionality to
#   calculate spinon and rotor Hamiltonians, parameters, and some
#   other things such as energies and real-space wavefunction 
#   projections. 
# --------------------------------------------------------------------
#   Specific iteration methods are handled in other files since
#   multiple are used to calculate and compare results. 
# ====================================================================

class SingleSite1D:
    _CACHE_VERSION = 1.0
    _CACHE_DIR = Path(__file__).resolve().parent / '_cache'
    _MODEL_TAG = "single_site_1D"

    """Names of the converged quantities in iteration methods" return order.
    Parameter guesses looked for in nearest parameter lookup, needed for calling fresh iteration"""
    _ITERATION_NAMES = ("h_vec", "Z_vec", "K_vec", "t_eff_mat", "nf_vec", "L_vec", "mu", "count")
    _NEAREST_NAMES = ("h_vec", "Z_vec", "K_vec", "t_eff_mat")

    _ALGO_NAMES = ["BROYDEN", "NEWTON_GAUGE"]

    def __init__(self, epsilon_vec: np.ndarray, U: float, t: Union[float, Sequence[float]], l_max: int, beta: Union[float, None], N_sites: int, N_at: int, PBC: bool=False, 
                 iteration_method: int=0, use_cache: bool=True, cache_dir: str=None, try_nearest_params: bool=True, 
                 nearest_params_max_dist: float=0.3, print_cache_results: bool=False) -> None:
        """
        Construct a 1D HUbbard chain with the single-site approximation scheme.

        Args:
            epsilon_vec:             On-site energy at every lattice site, length must be N_sites;
            U:                       On-site Hubbard repulsion potential.
            t:                       Single value or sequence of tunneling amplitudes. If sequence, order is (NN, NNN, ...)
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
        
        """Hubbard Parameters"""
        self._epsilon_vec = epsilon_vec
        self._U, self._l_max, self._beta, self._N_sites, self._N_at = U, l_max, beta, N_sites, N_at
        self._N_target = int((N_sites * N_at) / 2.0)            # Half-filling
        self._t_mat = Helpers.hopping_matrix(t, N_sites, PBC)

        self._dim = 2 * l_max + 1 # Dimension for one site

        """Cache & Iteration"""
        cache = IterationCache(cache_dir if cache_dir is not None else self._CACHE_DIR,
                               model_tag=self._MODEL_TAG, version=self._CACHE_VERSION)
        params = dict(epsvec=epsilon_vec, U=U, 
                      **{f"t{n}": float(t_n) for n, t_n in enumerate(self._t_vec, start=1)},
                      beta=beta, lmax=l_max, Nsites=N_sites, Nat=N_at, PBC=PBC)
        data = cache.load(params, self._ITERATION_NAMES) if use_cache else None
        
        if data is not None:
            # Successful hit w/ stored data, load into memory
            self._cache_status = "HIT"

            self._h_vec        = np.asarray(data["h_vec"], float).copy()
            self._Z_vec        = np.asarray(data["Z_vec"], float).copy()
            self._K_vec        = np.asarray( data["K_vec"], float).copy()

            self._t_eff_mat    = data["t_eff_mat"]

            self._nf_vec       = np.asarray(data["nf_vec"], float).copy()
            self._L_vec        = np.asarray(data["L_vec"], float).copy()

            self._mu           = float(data["mu"])
            self._count        = int(data["count"])

            self._evals_spinon, self._evecs_spinon       = np.linalg.eigh(get_H_spinon(self._epsilon_vec, self._t_eff_mat, self._h_vec, self._mu))
            self._evals_rotor_vec, self._evecs_rotor_vec = get_rotor_eig_vec(self._U, self._h_vec, self._K_vec, self._l_max, self._N_sites)

            if print_cache_results:
                print(f"{self.__class__.__name__}.__init__: Cache hit: {cache.path(params).name}")

        else:
            # Failure to hit; default attempt params before nearest params check
            h_vec_guess = -self._epsilon_vec
            Z_vec_guess = np.full(self._N_sites, 0.9)
            K_vec_guess = np.full(self._N_sites, -1.0)
            t_eff_mat_guess = 0.9 * self._t_mat

            self._cache_status = "MISS"
            self._nearest_start_from = None

            # Search for nearest params 
            if use_cache and try_nearest_params:
                near = cache.nearest(params, self._NEAREST_NAMES, must_match=("Nsites", "Nat", "PBC"))
                if near is not None and near["distance"] <= nearest_params_max_dist:
                    near_data = near["data"]

                    h_vec_guess     = np.asarray(near_data["h_vec"], float).copy()
                    Z_vec_guess     = np.asarray(near_data["Z_vec"], float).copy()
                    K_vec_guess     = np.asarray(near_data["K_vec"], float).copy()
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
                    self._evecs_rotor_vec, self._h_vec, self._Z_vec, self._K_vec, 
                    self._t_eff_mat, self._nf_vec, self._L_vec, self._mu, self._count
                ) = iterate_broyden(
                    self._epsilon_vec, self._t_mat, self._U, t_eff_mat_guess, 
                    h_vec_guess, Z_vec_guess, K_vec_guess, self._l_max, self._beta, 
                    self._N_sites, self._N_at, self._N_target
                )
            elif iteration_method == 1: # Iteration w/ Newton's method w/ gauge fix
                (
                    self._evals_spinon, self._evecs_spinon, self._evals_rotor_vec,
                    self._evecs_rotor_vec, self._h_vec, 
                    self._Z_vec, self._K_vec, self._t_eff_mat, self._nf_vec, self._L_vec, 
                    self._mu, self._count
                ) = iterate_newton_gauge(
                    self._epsilon_vec, self._t_mat, self._U, t_eff_mat_guess, 
                    h_vec_guess, Z_vec_guess, K_vec_guess, self._l_max, self._beta, 
                    self._N_sites, self._N_at, self._N_target
                )
            else:
                raise ValueError(f"{self.__class__.__name__}.__init__ | Error: Iteration argument")
            
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
        """Calculate MF zero-point energy | E_0 = -\sum_j K_j sqrt(Z_j).
        Also includes extra term to cancel out Lagrange parameter contribution."""
        return -np.sum(self._K_vec * np.sqrt(self._Z_vec)) + self._N_at * np.sum(self._h_vec) / 2.0

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