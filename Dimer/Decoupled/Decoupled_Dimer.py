import numpy as np
import sys

from typing import Sequence, Union
from pathlib import Path

from .Iteration.broyden import iterate_broyden
from .Iteration.newton_gauge import iterate_newton_gauge
from .Iteration.newton_naive import iterate_newton_naive

from .core import get_H_spinon, get_rotor_eig_vec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility.helpers import Helpers
from Utility.iteration_cache import IterationCache

# ====================================================================
#   DECOUPLED DIMER CLASS
# --------------------------------------------------------------------
#   A mean-field approximation is made on the exact slave rotor
#   Hubbard Hamiltoniain, giving a spinon and rotor Hamiltonian.
#   Another approximation is made to decouple rotors, leading
#   to a tower of rotor states at each lattice site. This
#   approximation is of questionable accuracy, so a big question is
#   how effective it is on a dimer.
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

class DecoupledDimer:
    _CACHE_VERSION = 1.0
    _CACHE_DIR = Path(__file__).resolve().parent / '_cache'
    _MODEL_TAG = "decoupled_dimer"

    """Names of the converged quantities in iteration methods" return order.
    Parameter guesses looked for in nearest parameter lookup, needed for calling fresh iteration"""
    _ITERATION_NAMES = ("h_vec", "Z_vec", "K_vec", "t_eff", "nf_vec", "L_vec", "L2_vec", "mu", "occ", "rho", "count")
    _NEAREST_NAMES = ("h_vec", "Z_vec")

    def __init__(self, epsilon: float, U: float, t: float, l_max: int, beta: Union[None, float]=None, N_target=2, 
                 iteration_method=0, use_cache=True, cache_dir=None, try_nearest_params=True, nearest_params_max_dist=0.5,
                 print_cache_results=False):
        """Takes parameter data and calculates self-consistent parameters and SR eigenspace solutions.
        Checks parameters against stored cache data; if cached, retrieves data. if not cached,
        calls an iteration method to find solutions. Eigenspace solutions are calculted from
        retrieved parameters to save space."""
        self._epsilon_vec = np.array([-epsilon/2.0, epsilon/2.0]) # Even/opposite doping to preserve PHS, restricts chem. potential to U/2
        self._t = t
        self._U, self._beta, self._l_max, self._N_target = U, beta, l_max, N_target
        self._dim = 2 * l_max + 1 # Dimension for one site

        cache = IterationCache(cache_dir if cache_dir is not None else self._CACHE_DIR,
                               model_tag=self._MODEL_TAG, version=self._CACHE_VERSION)
        params = dict(epsilon=epsilon, U=U, t=t, beta=beta, lmax=l_max, N=N_target)
        data = cache.load(params, self._ITERATION_NAMES) if use_cache else None
        
        if data is not None:
            # Successful hit w/ stored data, load into memory
            self._cache_status = "HIT"

            self._h_vec        = np.asarray(data["h_vec"], float).copy()
            self._Z_vec        = np.asarray(data["Z_vec"], float).copy()
            self._K_vec        = np.asarray( data["K_vec"], float).copy()

            self._t_eff        = float(data["t_eff"])

            self._nf_vec       = np.asarray(data["nf_vec"], float).copy()
            self._L_vec        = np.asarray(data["L_vec"], float).copy()
            self._L2_vec       = np.asarray(data["L2_vec"], float).copy()

            self._mu           = float(data["mu"])
            self._occ          = np.asarray(data["occ"], float).copy()
            self._rho          = np.asarray(data["rho"], float).copy()

            self._count        = int(data["count"])

            self._evals_spinon, self._evecs_spinon       = np.linalg.eigh(get_H_spinon(self._epsilon_vec, self._t_eff, self._h_vec))
            self._evals_rotor_vec, self._evecs_rotor_vec = get_rotor_eig_vec(self._U, self._h_vec, self._K_vec, self._l_max)

            if print_cache_results:
                print(f"{self.__class__.__name__}.__init__: Cache hit: {cache.path(params).name}")
        else:
            # Failure to hit; default attempt params before nearest params check
            h_vec_guess = -self._epsilon_vec
            Z_vec_guess = np.full(2, 0.9)

            self._cache_status = "MISS"
            self._nearest_start_from = None

            # Search for nearest params 
            if use_cache and try_nearest_params:
                near = cache.nearest(params, self._NEAREST_NAMES, must_match=("epsilon", "N"))
                if near is not None and near["distance"] <= nearest_params_max_dist:
                    near_data = near["data"]

                    h_vec_guess     = np.asarray(near_data["h_vec"], float).copy()
                    Z_vec_guess     = np.asarray(near_data["Z_vec"], float).copy()
                    self._cache_status = "NEAREST"

                    if print_cache_results:
                        print(f"{self.__class__.__name__}.__init__: Nearest start from "
                              f"{near['params']} (dist = {near['distance']:.3f})")
                else:
                    if print_cache_results:
                        print(f"{self.__class__.__name__}.__init__: Iterating with default params")
            
            if iteration_method == 0: # Iteration w/ modified Broyden's method 
                (
                    self._evals_spinon, self._evecs_spinon, self._evals_rotor_vec, 
                    self._evecs_rotor_vec, self._h_vec, self._Z_vec, self._K_vec, 
                    self._t_eff, self._nf_vec, self._L_vec, self._L2_vec, self._mu, 
                    self._occ, self._rho, self._count
                ) = iterate_broyden(
                    self._epsilon_vec, self._U, self._t, h_vec_guess, Z_vec_guess, 
                    self._beta, self._l_max, self._N_target
                )
            else:
                raise ValueError(f"{self.__class__.__name__}.__init__ | ERROR: ITERATION argument")
            
            if use_cache:
                cache.save(params, {name: getattr(self, "_" + name) for name in self._ITERATION_NAMES})
        
        """Add U/2 shift missing from (U/2)L^2 representation of the interaction.
        Note that U/2 is only the correct shift in PHS systems, that is half-filling
        with even-and-opposite doping on either site."""
        self.mu_phys = self._mu + self._U / 2.0 

    # ====================================================================
    #   CLASS/STATIC METHODS (PRIMARILY CACHE RELATED)
    # ====================================================================
    @classmethod
    def clear_cache(cls, cache_dir=None):
        """Delete all cached iteration results for the decoupled dimer"""
        IterationCache(cache_dir if cache_dir is not None else cls._CACHE_DIR,
                       model_tag=cls._MODEL_TAG, version=cls._CACHE_VERSION).clear()

    # ====================================================================
    #   PRIVATE METHODS
    # ====================================================================
    def __shift_op(self):
        """Shift operator e^{i theta_j} in the |l> basis"""
        shift_op = np.zeros((self._dim, self._dim))
        shift_op[np.arange(1, self._dim), np.arange(0, self._dim - 1)] = 1.0
        return shift_op

    def __jw_ops(self):
        """Jordan-Wigner electron operators on the 16-dim Fock space.
        Mode ordering: 0:1up 1:1down 2:2up 3:2down."""
        if hasattr(self, '_jw_cache'):
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
 
    def __H_exact(self):
        """Exact Hubbard H (bare), number operator N, and the
        eigenspace solutions of K = H - mu_phys*N"""
        if hasattr(self, '_H_exact_cache'):
            return self._H_exact_cache
        
        mode = lambda site, spin: 2 * site + spin
        c_ann, c_cre = self.__jw_ops()

        H, N_op = np.zeros((16, 16)), np.zeros((16, 16))

        for site in range(2):
            n_up = c_cre[mode(site, 0)] @ c_ann[mode(site, 0)] # number op. c^+_1 c_1
            n_down = c_cre[mode(site, 1)] @ c_ann[mode(site, 1)]
            
            H += self._epsilon_vec[site] * (n_up + n_down) + self._U * (n_up @ n_down)
            N_op += n_up + n_down

        for spin in range(2):
            tun = c_cre[mode(0, spin)] @ c_ann[mode(1, spin)] # tunneling op. c^+_1 c_2
            H += -self._t * (tun + tun.T)

        evals_K, evecs_K = np.linalg.eigh(H - self.mu_phys * N_op)
        self._H_exact_cache = (H, N_op, evals_K, evecs_K)

        return self._H_exact_cache
 
    def __gs_proj(self):
        """Projected MF ground state as a normalized 16-dim Fock vector"""
        if hasattr(self, '_proj_cache'):
            return self._proj_cache
        
        mode = lambda site, spin: 2 * site + spin
        _, c_cre = self.__jw_ops()

        # Projected state in the full 16-dim Fock space
        proj_state = np.zeros(16)

        # Spinon sector
        psi_f = np.zeros(16); psi_f[0] = 1.0  # 16-dim Slater det. spinon state; initial |vac>
        evec_spinon_gs = self._evecs_spinon[:,0]
        for spin in range(2):
            psi_f = (evec_spinon_gs[0] * c_cre[mode(0, spin)] 
                    + evec_spinon_gs[1] * c_cre[mode(1, spin)]) @ psi_f
        
        # Rotor sector
        evec_rotor_gs_1 = self._evecs_rotor_vec[0][:,0]
        evec_rotor_gs_2 = self._evecs_rotor_vec[1][:,0]
        for i in range(16):
            if np.abs(psi_f[i]) < 1e-15:
                continue

            # Quick bitshift method to recover occupation numbers from computational index 
            n1 = ((i >> 3) & 1) + ((i >> 2) & 1) # n1up -> 3 bits, n1down -> 2 bits
            n2 = ((i >> 1) & 1) + (i & 1)        # n2up -> 1 bits, n2down -> 0 bits
            
            proj_state[i] = psi_f[i] * evec_rotor_gs_1[n1 - 1 + self._l_max] * evec_rotor_gs_2[n2 - 1 + self._l_max]
        
        self._proj_cache = proj_state / np.sqrt(float(proj_state @ proj_state))
        return self._proj_cache

    def __E0(self):
        """Calculate MF zero-point energy | E_0 = -\sum_j K_j sqrt(Z_j)"""
        return -np.sum(self._K_vec * np.sqrt(self._Z_vec) + self._h_vec)
    
    def __rotor_product(self):
        """Built the rotor product state from two sites |psi_1>|psi_2> with
        energies E_1 + E_2. Returns (E, p, M) where E are eigenenergies,
        p are Boltzmann probabilities, and M = <m|e^{+i theta_i}|n> are
        shift operator matrix elements. This method is intended as a helper
        for the later Green's function analysis."""
                          
        evals_rotor_0, evecs_rotor_0 = self._evals_rotor_vec[0], self._evecs_rotor_vec[0]
        evals_rotor_1, evecs_rotor_1 = self._evals_rotor_vec[1], self._evecs_rotor_vec[1]
        
        energies = (evals_rotor_0[:, None] + evals_rotor_1[None, :]).ravel()    # E_1 + E_2
        bz = Helpers.bz_weights(energies, self._beta) # normalized Boltzmann probabilities
        
        # shift on the full product space, then rotate to the product eigenbasis
        prod_states = np.kron(evecs_rotor_0, evecs_rotor_1)                     

        S, I = self.__shift_op(), np.eye(self._dim)
        mel_vec = [prod_states.T @ np.kron(S, I) @ prod_states,       # e^{+i th_1} = S x I
                   prod_states.T @ np.kron(I, S) @ prod_states]       # e^{+i th_2} = I x S
        
        return energies, bz, mel_vec
 
    def __gf_spectral_data(self, tol=1e-13):
        """Calculate Green's function data. Poles P_k, residues R_k[i,j] with G_ij(z) = sum_k R_k[i,j]/(z-P_k).
            Cached and saved on first call. Used for quickly calculating values."""
        if hasattr(self, '_gf_cache'):
            return self._gf_cache
        
        energies, bz_rotor, mel_vec = self.__rotor_product()
        
        mel_vec = np.stack(mel_vec)                                         # <m|e^{i th_i}|n>
        dE = energies[:, None] - energies[None, :]                          # dE[m,n] = E_m - E_n
        p_rotor_ket, p_rotor_bra = bz_rotor[None, :], bz_rotor[:, None]     # Boltzmann probs. corresponding to matrix element bra/ket states
 
        pole_vec, res_vec = [], []
        for i in range(2):                        # spinon orbitals
            fd_spinon_vals = Helpers.fermi(self._evals_spinon[i], self._beta)
            
            sr_mix = (1.0 - fd_spinon_vals) * p_rotor_ket + fd_spinon_vals * p_rotor_bra          # mix[n,m]
            spinon_coeff = self._evecs_spinon[:, i]

            R = np.einsum('i,j,inm,jnm,nm->nmij', spinon_coeff, spinon_coeff, mel_vec, mel_vec, sr_mix)
            
            pole_vec.append(self._evals_spinon[i] + dE.ravel())
            res_vec.append(R.reshape(-1, 2, 2))

        P, R = np.concatenate(pole_vec), np.concatenate(res_vec)
        finite_res_mask = np.abs(R).max(axis=(1, 2)) > tol
        self._gf_cache = (P[finite_res_mask], R[finite_res_mask])
        return self._gf_cache
    
    # ====================================================================
    #   PUBLIC METHODS
    # ====================================================================
    def get_E_gs(self):
        """Calculate ground-state energy w/o physical subspace projection"""
        E0 = self.__E0()
        E_gs_spinon, E_gs_rotor_vec = self._evals_spinon[0], self._evals_rotor_vec[:,0]
        E_gs = self._N_target * E_gs_spinon + np.sum(E_gs_rotor_vec) + E0
        return E_gs
    
    def get_energy_MF(self):
        """Mean-field energy accounting for difference in particle number:
            E = sum_i eps_i n_i
            - 2 N_AT t rho_12 sqrt(Z_1 Z_2)
            + U/2 sum_i <L_i^2>
            + U/2 sum_i (n_i - 1)"""
        E_site = np.sum(self._epsilon_vec * self._nf_vec)
        E_hop = -2.0 * 2 * self._t * np.real(self._rho[0, 1]) * np.sqrt(self._Z_vec[0] * self._Z_vec[1])
        E_U = 0.5 * self._U * np.sum(self._L2_vec) + 0.5 * self._U * np.sum(self._nf_vec - 1.0)
        return E_site + E_hop + E_U

    def get_E_gs_proj(self, grand_canonical=False):
        """Calculate ground-state energy as the expected value of
        the MF ground state projected onto the physical subspace.
        Effectively a trial wavefunction using the variational 
        principle."""
        H, N_op, _, _ = self.__H_exact()
        proj_state = self.__gs_proj()

        E_proj = float(proj_state @ H @ proj_state)
        if grand_canonical:
            E_proj -= self.mu_phys * float(proj_state @ N_op @ proj_state)
            # E_proj -= self._mu_phys * self._N_target

        return E_proj

    def get_filling_vec(self):
        """Get spinon/electron occupation at each lattice site."""
        return self._nf_vec
    
    def get_double_occ_vec(self):
        """D_i = <n_up n_down>_i = (<L_i^2> + <n_i> - 1) / 2."""
        return 0.5 * (self._L2_vec + self._nf_vec - 1.0)
    
    def rotor_gs(self, theta_vals, site=0):
        """Calculates rotor ground state output in the exp(i l theta) basis"""
        l_vals = np.arange(-self._l_max, self._l_max+1)
        evec_rotor_gs = self._evecs_rotor_vec[site][:,0]
        
        output = []
        for i in range(len(theta_vals)):
            val = np.abs(evec_rotor_gs @ np.exp(1j * l_vals * theta_vals[i])) / np.sqrt(2*np.pi)
            output.append(val)

        return np.array(output)
    
    def get_gf(self, z, i=0, j=0):
        """Calculate Green's function G_ij(z) on complex frequencies z (Matsubara or retarded okay) |
           Can pass scalar or vector values of z. Returns shape (len(z))"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        P, R = self.__gf_spectral_data()
        gf_vals = np.sum(R[None, :, i, j] / (z[:, None] - P[None, :]), axis=1)
        return gf_vals

    def get_gf_matrix(self, z):
        """Full 2x2 GF matrix at each z | shape (len(z), 2, 2)"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        P, R = self.__gf_spectral_data()
        gf_mat = np.sum(R[None, :, :, :]
                        / (z[:, None, None, None] - P[None, :, None, None]), axis=1)
        return gf_mat

    def get_self_energy_matrix(self, z):
        """Self-energy from the Dyson equation Sigma(z) = G0(z)^{-1} - G(z)^{-1}
        Physical chemical potential mu_phys = mu + U/2 must be included to get
        Hartree shift seen in exact diagonalization benchmark"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))

        h0 = np.diag(self._epsilon_vec - self.mu_phys) - self._t
        gf_bare = np.linalg.inv(z[:, None, None] * np.eye(2) - h0[None, :, :])
        self_energy_mat = np.linalg.inv(gf_bare) - np.linalg.inv(self.get_gf_matrix(z))

        return self_energy_mat

    def get_spectral_vec(self, freqs, imag_delta=1e-4):
        """Calculate site-resolved spectral function"""
        freqs = np.real(freqs)
        spectral_vec = np.zeros((len(freqs),2))

        for site in range(2):
            spectral_vals = -np.imag(self.get_gf(freqs + 1j*imag_delta, site, site)) / np.pi
            spectral_vec[:,site] = spectral_vals

        return spectral_vec

    def get_qp_vec(self, domega=1e-5, imag_delta=1e-4):
        """Calculate quasiparticle weight Z as a finite difference of the retarded Green's function"""
        Z_vec = np.zeros(2)
        z = np.array([-domega + 1j * imag_delta, domega + 1j * imag_delta])
        
        for site in range(2):
            self_energy_vals = self.get_self_energy_matrix(z)[:, site, site]
            Z = 1.0 / (1.0 - (self_energy_vals[1].real - self_energy_vals[0].real) / (2.0 * domega))
            Z_vec[site] = Z

        return Z_vec
 
    def get_Z_vec(self):
        return self._Z_vec
    
    def check_gf_sum(self):
        """Residues should sum to 1 at each site."""
        _, R = self.__gf_spectral_data()
        sum = np.array([R[:, 0, 0].sum(), R[:, 1, 1].sum()])
        return sum