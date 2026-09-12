import numpy as np
import sys

from pathlib import Path
from typing import Union, Sequence

from .Iteration.newton_gauge import iterate_newton_gauge
from .Iteration.broyden import iterate_broyden
from .Iteration.newton_naive import iterate_newton_naive

from .core import get_H_spinon, get_H_rotor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility.helpers import Helpers
from Utility.iteration_cache import IterationCache

# ====================================================================
#   COUPLED DIMER CLASS 
# --------------------------------------------------------------------
#   A mean-field approximation is made on the exact slave rotor
#   Hubbard Hamiltoniain, giving a spinon and rotor Hamiltonian.
#   In the standard literature for larger crystals, a further
#   approximation is made to decoupled neighboring rotors
#   Otherwise the Hilbert space grows exponentially.
#   
#   A Hubbard dimer is still small enough that we can keep the rotors
#   coupled and solve the problem in reasonable time.
# --------------------------------------------------------------------
#   In this model, rotor states are superpositions of basis states
#   |l1,l2> = |l1>x|l2>. The rotor tunneling element 
#   t<cos(theta1-theta2)> is calculated as is without further 
#   approximation. This code contains the Coupled Dimer class. A user
#   creates a class instance for a set of parameters specifying
#   doping, Hubbard repulsion, temperature, and the rotor Hilbert 
#   space cutoff, then uses class methods here to get desired 
#   quantities like the Green"s function, self-energy, quasiparticle
#   weight, pair occupancy, heat capacity, etc.  
# --------------------------------------------------------------------
#   Methods used to solve the self-consistency condition can be found
#   in Dimer/Coupled/core.py. 
# ====================================================================

class CoupledDimer:
    _CACHE_VERSION = 1.0
    _CACHE_DIR = Path(__file__).resolve().parent / "_cache"
    _MODEL_TAG = "coupled_dimer"

    """Names of the converged quantities in iteration methods" return order.
    Parameter guesses looked for in nearest parameter lookup, needed for calling fresh iteration"""
    _ITERATION_NAMES = ("h_vec", "t_eff", "J_eff", "nf_vec", "L_vec", "mu", "count")
    _NEAREST_NAMES = ("h_vec", "t_eff", "J_eff")

    def __init__(self, epsilon: float, U: float, t: float, l_max: int, beta: Union[None, float]=None, N_target: int=2, iteration_method: int=0,
                 use_cache: bool=True, cache_dir: str=None, try_nearest_params: bool=True, nearest_params_max_dist: float=0.5, 
                 print_cache_results: bool=False) -> None:
        """Takes parameter data and calculates self-consistent parameters and SR eigenspace solutions.
        Checks parameters against stored cache data; if cached, retrieves data. if not cached,
        calls an iteration method to find solutions. Eigenspace solutions are calcualted from
        retrieved parameters to save space."""
        self._epsilon_vec = np.array([-epsilon/2.0, epsilon/2.0])
        self._U, self._t, self._beta, self._l_max, self._N_target = float(U), float(t), beta, int(l_max), int(N_target)
        self._dim = 2*l_max + 1
 
        cache = IterationCache(cache_dir if cache_dir is not None else self._CACHE_DIR,
                               model_tag=self._MODEL_TAG, version=self._CACHE_VERSION)
        params = dict(epsilon=epsilon, U=U, t=t, beta=beta, lmax=l_max, N=N_target)
        data = cache.load(params, self._ITERATION_NAMES) if use_cache else None
        
        if data is not None:
            # Successful hit w/ stored data, load into memory
            self._cache_status = "HIT"

            self._h_vec        = np.asarray(data["h_vec"], float).copy()
            self._nf_vec       = np.asarray(data["nf_vec"], float).copy()
            self._L_vec        = np.asarray(data["L_vec"], float).copy()

            self._t_eff        = float(data["t_eff"])
            self._J_eff        = float(data["J_eff"])
            self._mu           = float(data["mu"])
            self._count        = int(data["count"])

            # Rebuild and solve for eigenspace solutions of spinon & rotor Hamiltonians; saves disk space
            self._evals_spinon, self._evecs_spinon = np.linalg.eigh(get_H_spinon(self._epsilon_vec, self._h_vec, self._t_eff, self._mu))
            self._evals_rotor, self._evecs_rotor   = np.linalg.eigh(get_H_rotor(self._U, self._h_vec, self._J_eff, self._l_max))

            if print_cache_results:
                print(f"{self.__class__.__name__}.__init__: Cache hit: {cache.path(params).name}")
        else:
            # Failure to hit; default attempt params before nearest params check
            h_vec_guess = -self._epsilon_vec
            t_eff_guess = 0.8 * t
            J_eff_guess = 0.8 * t

            self._cache_status = "MISS"

            # Search for nearest params
            if use_cache and try_nearest_params:
                near = cache.nearest(params, self._NEAREST_NAMES, must_match=("N", "epsilon"))
                if near is not None and near["distance"] <= nearest_params_max_dist:
                    near_data = near["data"]

                    h_vec_guess = np.asarray(near_data["h_vec"], float).copy()
                    t_eff_guess = float(near_data["t_eff"])
                    J_eff_guess = float(near_data["J_eff"])

                    self._cache_status = "NEAREST"

                    if print_cache_results:
                        print(f"{self.__class__.__name__}.__init__: Nearest start from "
                              f"{near['params']} (dist = {near['distance']:.3f})")
 
            if iteration_method == 0: # Modified Broyden"s method 
                (
                    self._evals_spinon, self._evecs_spinon, self._evals_rotor,
                    self._evecs_rotor, self._h_vec, self._t_eff, self._J_eff,
                    self._nf_vec, self._L_vec, self._mu, self._count
                ) = iterate_broyden(
                    self._epsilon_vec, self._U, self._t, 
                    h_vec_guess, t_eff_guess, J_eff_guess, 
                    self._beta, self._l_max, self._N_target
                )
            elif iteration_method == 1: # Newton's method w/ gauge fix
                (
                    self._evals_spinon, self._evecs_spinon, self._evals_rotor,
                    self._evecs_rotor, self._h_vec, self._t_eff, self._J_eff,
                    self._nf_vec, self._L_vec, self._mu, self._count
                ) = iterate_newton_gauge(
                    self._epsilon_vec, self._U, self._t, 
                    h_vec_guess, t_eff_guess, J_eff_guess, 
                    self._beta, self._l_max, self._N_target
                )
            elif iteration_method == 2: # Newton's method w/o gauge fix
                (
                    self._evals_spinon, self._evecs_spinon, self._evals_rotor,
                    self._evecs_rotor, self._h_vec, self._t_eff, self._J_eff,
                    self._nf_vec, self._L_vec, self._mu, self._count
                ) = iterate_newton_naive(
                    self._epsilon_vec, self._U, self._t, 
                    h_vec_guess, t_eff_guess, J_eff_guess, 
                    self._beta, self._l_max, self._N_target
                )
            else:
                raise ValueError("Coupled_Dimer.__init__: ERROR: iteration argument")
 
            if use_cache:
                cache.save(params, {name: getattr(self, "_" + name) for name in self._ITERATION_NAMES})
 
        """Add U/2 shift missing from (U/2)L^2 representation of the interaction.
        Note that U/2 is only the correct shift in PHS systems, that is half-filling
        with even-and-opposite doping on either site."""
        self._mu_phys = self._mu + self._U / 2.0
 
    # ====================================================================
    #   CLASS/STATIC METHODS (PRIMARILY CACHE RELATED)
    # ====================================================================
    @classmethod
    def clear_cache(cls, cache_dir: str=None) -> None:
        """Delete all cached iteration results for the coupled dimer"""
        IterationCache(cache_dir if cache_dir is not None else cls._CACHE_DIR,
                       model_tag=cls._MODEL_TAG, version=cls._CACHE_VERSION).clear()

    # ====================================================================
    #   PRIVATE METHODS
    # ====================================================================
    def __shift_op(self, site: int) -> np.ndarray:
        """Matrix of e^{i theta_j} in the |l1,l2> basis"""
        l1 = np.repeat(np.arange(-self._l_max, self._l_max+1), self._dim)
        l2 = np.tile(np.arange(-self._l_max, self._l_max+1), self._dim)

        shift_op = np.zeros((self._dim**2, self._dim**2))
        if site == 0:
            mask = l1 < self._l_max
            bra_index = (l1[mask]+self._l_max+1)*self._dim + (l2[mask]+self._l_max)
        elif site == 1:
            mask = l2 < self._l_max
            bra_index = (l1[mask]+self._l_max)*self._dim + (l2[mask]+self._l_max+1)

        ket_index = np.where(mask)[0]
        shift_op[bra_index, ket_index] = 1.0

        return shift_op

    def __jw_ops(self) -> tuple[np.ndarray, np.ndarray]:
        """Jordan-Wigner electron operators on the 16-dim Fock space.
        Mode ordering: 0:1up 1:1down 2:2up 3:2down."""
        if hasattr(self, "_jw_cache"):
            return self._jw_cache
        
        a_ann = np.array([[0., 1.], [0., 0.]])
        pauli_z = np.diag([1., -1.])
        I = np.eye(2)

        def kron_matrix(*mats):
            out = mats[0]
            for mat in mats[1:]:
                out = np.kron(out, mat)
            return out
        
        c_ann = [kron_matrix(a_ann, I, I, I), 
                kron_matrix(pauli_z, a_ann, I, I),
                kron_matrix(pauli_z, pauli_z, a_ann, I), 
                kron_matrix(pauli_z, pauli_z, pauli_z, a_ann)]
        
        self._jw_cache = (c_ann, [mat.T for mat in c_ann])

        return self._jw_cache

    def __H_exact(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact Hubbard H (bare), number operator N, and the
        eigenspace solutions of K = H - mu_phys*N"""
        if hasattr(self, "_H_exact_cache"):
            return self._H_exact_cache
        
        mode = lambda site, spin: 2 * site + spin
        c_ann, c_cre = self.__jw_ops()

        H, N_op = np.zeros((16, 16)), np.zeros((16, 16))

        for site in range(2):
            n_up = c_cre[mode(site, 0)] @ c_ann[mode(site, 0)] # number operator c^+_i c_i
            n_down = c_cre[mode(site, 1)] @ c_ann[mode(site, 1)]
            
            H += self._epsilon_vec[site] * (n_up + n_down) + self._U * (n_up @ n_down)
            N_op += n_up + n_down

        for spin in range(2):
            tun = c_cre[mode(0, spin)] @ c_ann[mode(1, spin)] # tunneling operator c^+_j c_j
            H += -self._t * (tun + tun.T)

        evals_K, evecs_K = np.linalg.eigh(H - self._mu_phys * N_op)
        self._H_exact_cache = (H, N_op, evals_K, evecs_K)

        return self._H_exact_cache
 
    def __gs_proj(self) -> np.ndarray:
        """Projected MF ground state as a normalized 16-dim Fock vector"""
        if hasattr(self, "_proj_cache"):
            return self._proj_cache
        
        mode = lambda site, spin: 2 * site + spin
        _, c_cre = self.__jw_ops()

        # Projected state in the full 16-dim Fock space
        proj_state = np.zeros(16)

        #Spinon sector
        psi_f = np.zeros(16); psi_f[0] = 1.0 # 16-dim Slater det. spinon state; initial |vac>
        evec_spinon_gs = self._evecs_spinon[:,0]
        for spin in range(2):
            psi_f = (evec_spinon_gs[0] * c_cre[mode(0, spin)] + evec_spinon_gs[1] * c_cre[mode(1, spin)]) @ psi_f
            
        #Rotor sector
        evec_rotor_gs = self._evecs_rotor[:,0]
        for i in range(16):
            if np.abs(psi_f[i]) < 1e-15:
                continue
            
            # Quick bitshift method to recover occupation numbers from computational index 
            n1 = ((i >> 3) & 1) + ((i >> 2) & 1) # n1up -> 3 bits, n1down -> 2 bits
            n2 = ((i >> 1) & 1) + (i & 1)        # n2up -> 1 bits, n2down -> 0 bits

            phys_index = (n1 - 1 + self._l_max) * self._dim + (n2 - 1 + self._l_max) 
            proj_state[i] = psi_f[i] * evec_rotor_gs[phys_index]

        self._proj_cache = proj_state / np.sqrt(float(proj_state @ proj_state))

        return self._proj_cache

    def __E0(self) -> float:
        """Calculate zero-point energy from the MF decoupling | E0 = \sum_{ij} J^{eff}_{ij} <cos(theta_i - theta_j)>"""
        if hasattr(self, "_E0_cache"):
            return self._E0_cache
        
        l1 = np.repeat(np.arange(-self._l_max, self._l_max+1), self._dim)
        l2 = np.tile(np.arange(-self._l_max, self._l_max+1), self._dim)

        # Build shift operator | maps |l1,l2> -> |l1+1,l2-1> + |l1-1,l2+1>
        shift_op = np.zeros((self._dim**2, self._dim**2)) 

        mask = (l1 < self._l_max) & (l2 > -self._l_max) # Filters shifts inside the truncated Hilbert space
        bra_index = (l1[mask] + self._l_max + 1)*self._dim + (l2[mask] + self._l_max - 1)
        ket_index = np.where(mask)[0]

        shift_op[bra_index, ket_index] = 1.0 # |l1,l2> -> |l1+1,l2-1>
        shift_op += shift_op.T               # |l1,l2> -> |l1-1,l2+1>

        bz = Helpers.bz_weights(self._evals_rotor, self._beta)

        # <psi_n|S|psi_n>        
        exp_vals = np.real(np.sum(np.conj(self._evecs_rotor) * (shift_op @ self._evecs_rotor), axis=0))
        E0 = 2 * self._J_eff * np.sum(bz * exp_vals) / 2
        self._E0_cache = E0

        return self._E0_cache
        
    def __gf_spectral_data(self, tol: float=1e-13) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Green"s function data. Poles P_k, residues R_k[i,j] with G_ij(z) = sum_k R_k[i,j]/(z-P_k).
            Cached and saved on first call. Used for quickly calculating values."""
        if hasattr(self, "_gf_cache"):
            return self._gf_cache
        
        # Rotor matrix elements: <m|e^{i\theta_j}|n>
        mel_rotor = np.stack([self._evecs_rotor.T @ self.__shift_op(site) @ self._evecs_rotor for site in range(2)])
        dE_rotor = self._evals_rotor[:, None] - self._evals_rotor[None, :]
        bz_rotor = Helpers.bz_weights(self._evals_rotor, self._beta)     # Rotor boltzmann probabilities
        p_rotor_ket, p_rotor_bra = bz_rotor[None, :], bz_rotor[:, None] 
        
        pole_vec, res_vec = [], []
        for i in range(2):
            fd_spinon_vals = Helpers.fermi(self._evals_spinon[i], self._beta)
            
            sr_mix = fd_spinon_vals * p_rotor_bra + (1.0 - fd_spinon_vals) * p_rotor_ket    # Convolution of spinon/rotor GF"s
            spinon_coeff = self._evecs_spinon[:,i]                                           # spinon state coefficients

            R = np.einsum("i,j,inm,jnm,nm->nmij", spinon_coeff, spinon_coeff, mel_rotor, mel_rotor, sr_mix)
            
            pole_vec.append(self._evals_spinon[i] + dE_rotor.ravel())
            res_vec.append(R.reshape(-1, 2, 2))

        P, R = np.concatenate(pole_vec), np.concatenate(res_vec)
        finite_res_mask = np.abs(R).max(axis=(1, 2)) > tol # Throw away vanishing residues (matrix elements vanish)
        self._gf_cache = (P[finite_res_mask], R[finite_res_mask])
        return self._gf_cache
    
    # ====================================================================
    #   PUBLIC METHODS
    # ====================================================================
    def get_E_gs(self, grand_canonical: bool=False) -> float:
        """Calculate ground-state energy w/o physical subspace projection"""
        E0 = self.__E0()
        E_gs_spinon, E_gs_rotor = self._evals_spinon[0], self._evals_rotor[0]
        E_gs = 2*E_gs_spinon + E_gs_rotor + E0 # Spinon GS counted twice since it corresponds to 2x2 spin-subspace energy
        
        if grand_canonical:
            E_gs -= self._U     # -mu*N

        return E_gs
    
    def get_E_gs_proj(self, grand_canonical: bool=False) -> float:
        """Calculate ground-state energy as the expected value of
        the MF ground state projected onto the physical subspace.
        Effectively a trial wavefunction using the variational 
        principle."""
        H, N_op, _, _ = self.__H_exact()
        proj_state = self.__gs_proj()

        E_proj = float(proj_state @ H @ proj_state)

        if grand_canonical:
            E_proj -= self._mu_phys * float(proj_state @ N_op @ proj_state)
            # E_proj -= self._mu_phys * self._N_target

        return E_proj

    def rotor_gs(self, theta1_vals, theta2_vals, site=0):
        """Calculates rotor ground-state output in the exp(i l theta) basis.
        This is a 2D function for the coupled dimer Psi(theta1, theta2)."""
        evec_rotor_gs = self._evecs_rotor[site][:,0]

        l1 = np.repeat(np.arange(-self._l_max, self._l_max+1), self._dim)
        l2 = np.tile(np.arange(-self._l_max, self._l_max+1), self._dim)

        output = []
        for i in range(len(theta_vals)):
            val = np.abs(evec_rotor_gs @ np.exp(1j * l_vals * theta_vals[i])) / np.sqrt(2*np.pi)
            output.append(val)

        return np.array(output)
    
    def get_filling_vec(self) -> np.ndarray:
        """Get spinon/electron occupation at each lattice site."""
        return self._nf_vec
    
    def get_gf(self, z: np.ndarray, i: int=0, j: int=0):
        """Calculate Green"s function G_ij(z) on complex frequencies z (Matsubara or retarded okay) |
           Can pass scalar or vector values of z. Returns shape (len(z))"""
        z = np.atleast_1d(np.asarray(z, dtype=complex)) # If z is a scalar, make array
        P, R = self.__gf_spectral_data() # Pull residue and pole cache data
        gf_vals = np.sum(R[None, :, i, j] / (z[:, None] - P[None, :]), axis=1) 
        return gf_vals
    
    def get_gf_matrix(self, z: np.ndarray) -> np.ndarray:
        """Calculate the full 2x2 Green"s function matrix at scalar or vector values of z | returns shape (len(z), 2, 2)"""
        z = np.atleast_1d(np.asarray(z, dtype=complex)) # If z is a scalar, make array
        P, R = self.__gf_spectral_data() # Pull poles and residues
        gf_mat = np.sum(R[None, :, :, :]
                      / (z[:, None, None, None] - P[None, :, None, None]), axis=1) # Form matrix
        return gf_mat
    
    def get_self_energy_matrix(self, z: np.ndarray) -> np.ndarray:
        """Self-energy from the Dyson equation Sigma(z) = G0(z)^{-1} - G(z)^{-1}
        Physical chemical potential mu_phys = mu + U/2 must be included to get
        Hartree shift seen in exact diagonalization benchmark"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))

        h0 = np.array([[self._epsilon_vec[0] - self._mu_phys, -self._t],
                       [-self._t, self._epsilon_vec[1] - self._mu_phys]])
        gf_bare = np.linalg.inv(z[:, None, None] * np.eye(2) - h0[None, :, :])
        self_energy_mat = np.linalg.inv(gf_bare) - np.linalg.inv(self.get_gf_matrix(z))
        return self_energy_mat
    
    def get_spectral_vec(self, freqs: np.ndarray, imag_delta: float=1e-4) -> np.ndarray:
        """Calculate site-resolved spectral function output from retarded Green"s function"""
        freqs = np.real(freqs)
        spectral_vec = np.zeros((len(freqs),2))

        for site in range(2):
            spectral_vals = -np.imag(self.get_gf(freqs + 1j * imag_delta, site, site)) / np.pi
            spectral_vec[:,site] = spectral_vals

        return spectral_vec
    
    def get_qp_vec(self, domega: float=1e-5, delta: float=1e-4) -> np.ndarray:
        """Calculate quasiparticle weight Z as a finite difference of the retarded Green"s function"""
        Z_vec = np.zeros(2)
        z = np.array([-domega + 1j*delta, domega + 1j*delta])

        for site in range(2):
            self_energy_vals = self.get_self_energy_matrix(z)[:, site, site]
            Z = 1.0 / (1.0 - (self_energy_vals[1].real - self_energy_vals[0].real) / (2.0 * domega))
            Z_vec[site] = Z

        return Z_vec
    
    def get_exp_shift_vec(self) -> np.ndarray:
        """Thermal average of cos(theta_i) at each site | Assume no flux such that <cos(theta)> = <e^{i*theta}>.
        Gives zero everytime due to symmetry but... why not"""
        Z_vec = np.zeros(2)

        for site in range(2):
            shift_op = self.__shift_op(site) # Create shift operator for respective site
            exp_vals = np.real(np.sum(np.conj(self._evecs_rotor) * (shift_op @ self._evecs_rotor), axis=0))
            bz = Helpers.bz_weights(self._evals_rotor)
            Z_vec[site] = np.sum(bz * exp_vals)

        return Z_vec

    def check_gf_sum(self) -> float:
        """Residues should sum to 1 at each site"""
        _, R = self.__gf_spectral_data()
        sum = np.array([R[:, 0, 0].sum(), R[:, 1, 1].sum()])
        return sum

    def get_proj_fidelity(self) -> float:
        """Calculates projection of exact GS onto MF projected ground state <P Psi|Exact>"""
        _, _, _, evecs_K = self.__H_exact()
        proj_state = self.__gs_proj()
        fidelity = abs(proj_state @ evecs_K[:, 0])
        return fidelity
