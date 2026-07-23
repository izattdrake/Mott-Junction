import numpy as np
import sys

from pathlib import Path
from .Iteration.broyden import iterate_broyden
from .Iteration.newton_gauge import iterate_newton_gauge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility import helpers

# ====================================================================
#   CORE FUNCTIONALITY FOR DECOUPLED ROTOR PROBLEM ON A HUBBARD DIMER
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
class Decoupled_Dimer:
    def __init__(self, epsilon, U, t, beta, l_max, N_target=2, N_at=2, ITERATION=0):
        self.epsilon_vec = np.array([-epsilon/2.0, epsilon/2.0]) # Even/opposite doping to preserve PHS, restricts chem. potential to U/2
        self.t_mat = np.array([[0,t], [t,0]])
        self.U, self.beta, self.l_max, self.N_target, self.N_at = U, beta, l_max, N_target, N_at

        self.dim = 2*l_max + 1 # Dimension for one site

        t_eff_mat_guess = 0.9 * self.t_mat
        h_vec_guess = -self.epsilon_vec
        Z_vec_guess = np.full(2, 0.9)
        K_vec_guess = np.full(2, -1.0)

        if ITERATION == 0: # Iteration w/ Newton's method & gauge fix
            (self.evals_spinon, self.evecs_spinon, self.evals_rotor_vec,self.evecs_rotor_vec, self.h_vec, 
            self.Z_vec, self.K_vec, self.t_eff_mat, self.nf_vec, self.L_vec, self.mu, self.count) = iterate_newton_gauge(self.epsilon_vec, self.t_mat, self.U, t_eff_mat_guess, 
                                                                                                                        h_vec_guess, Z_vec_guess, K_vec_guess, self.beta, self.l_max, 
                                                                                                                        self.N_target, self.N_at)
        elif ITERATION == 1: # Iteration w/ modified Broyden's method
            (self.evals_spinon, self.evecs_spinon, self.evals_rotor_vec,self.evecs_rotor_vec, self.h_vec, 
            self.Z_vec, self.K_vec, self.t_eff_mat, self.nf_vec, self.L_vec, self.mu, self.count) = iterate_broyden(self.epsilon_vec, self.t_mat, self.U, t_eff_mat_guess, 
                                                                                                                    h_vec_guess, Z_vec_guess, K_vec_guess, self.beta, self.l_max, 
                                                                                                                    self.N_target, self.N_at)
        else:
            raise ValueError('Dimer/Decoupled/Decoupled_Dimer.__init__ | ERROR: ITERATION argument')
        
        """Add U/2 shift missing from (U/2)L^2 representation of the interaction.
        Note that U/2 is only the correct shift in PHS systems, that is half-filling
        with even-and-opposite doping on either site."""
        self.mu_phys = self.mu + self.U / 2.0 

    # ====================================================================
    #   PRIVATE METHODS
    # ====================================================================
    def __shift_op(self):
        """Shift operator e^{i theta_j} in the |l> basis"""
        shift_op = np.zeros((self.dim, self.dim))
        shift_op[np.arange(1, self.dim), np.arange(0, self.dim - 1)] = 1.0
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
            
            H += self.epsilon_vec[site] * (n_up + n_down) + self.U * (n_up @ n_down)
            N_op += n_up + n_down

        for spin in range(2):
            tun = c_cre[mode(0, spin)] @ c_ann[mode(1, spin)] # tunneling op. c^+_1 c_2
            H += -self.t_mat[0, 1] * (tun + tun.T)

        evals_K, evecs_K = np.linalg.eigh(H - self.mu_phys * N_op)
        self._H_exact_cache = (H, N_op, evals_K, evecs_K)

        return self._H_exact_cache
 
    def __gs_proj(self):
        """Projected MF ground state as a normalized 16-dim Fock vector"""
        if hasattr(self, '_proj_cache'):
            return self._proj_cache
        
        mode = lambda site, spin: 2 * site + spin
        _, cdag = self.__jw_ops()

        # Projected state in the full 16-dim Fock space
        proj_state = np.zeros(16)

        # Spinon sector
        psi_f = np.zeros(16); psi_f[0] = 1.0  # 16-dim Slater det. spinon state; initial |vac>
        evec_spinon_gs = self.evecs_spinon[:, 0]
        for spin in range(2):
            psi_f = (evec_spinon_gs[0] * cdag[mode(0, spin)] + evec_spinon_gs[1] * cdag[mode(1, spin)]) @ psi_f
        
        # Rotor sector
        evec_rotor_gs_1 = self.evecs_rotor_vec[0][:,0]
        evec_rotor_gs_2 = self.evecs_rotor_vec[1][:,0]
        for i in range(16):
            if np.abs(psi_f[i]) < 1e-15:
                continue

            # Quick bitshift method to recover occupation numbers from computational index 
            n1 = ((i >> 3) & 1) + ((i >> 2) & 1) # n1up -> 3 bits, n1down -> 2 bits
            n2 = ((i >> 1) & 1) + (i & 1)        # n2up -> 1 bits, n2down -> 0 bits
            
            proj_state[i] = psi_f[i] * evec_rotor_gs_1[n1 - 1 + self.l_max] * evec_rotor_gs_2[n2 - 1 + self.l_max]
        
        self._proj_cache = proj_state / np.sqrt(float(proj_state @ proj_state))
        return self._proj_cache

    def __E0(self):
        """Calculate MF zero-point energy | E_0 = -\sum_j K_j sqrt(Z_j)"""
        E0 = -np.sum(self.K_vec * np.sqrt(self.Z_vec))
        return E0
    
    def __rotor_product(self):
        """Built the rotor product state from two sites |psi_1>|psi_2> with
        energies E_1 + E_2. Returns (E, p, M) where E are eigenenergies,
        p are Boltzmann probabilities, and M = <m|e^{+i theta_i}|n> are
        shift operator matrix elements. This method is intended as a helper
        for the later Green's function analysis."""
                          
        evals_rotor_0, evecs_rotor_0 = self.evals_rotor_vec[0], self.evecs_rotor_vec[0]
        evals_rotor_1, evecs_rotor_1 = self.evals_rotor_vec[1], self.evecs_rotor_vec[1]
        
        energies = (evals_rotor_0[:, None] + evals_rotor_1[None, :]).ravel()    # E_1 + E_2
        bz = helpers._bz_weights(energies, self.beta) # normalized Boltzmann probabilities
        
        # shift on the full product space, then rotate to the product eigenbasis
        prod_states = np.kron(evecs_rotor_0, evecs_rotor_1)                     

        S, I = self.__shift_op(), np.eye(self.dim)
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
            fd_spinon_vals = helpers._fermi(self.evals_spinon[i], self.beta)
            
            sr_mix = (1.0 - fd_spinon_vals) * p_rotor_ket + fd_spinon_vals * p_rotor_bra          # mix[n,m]
            spinon_coeff = self.evecs_spinon[:, i]

            R = np.einsum('i,j,inm,jnm,nm->nmij', spinon_coeff, spinon_coeff, mel_vec, mel_vec, sr_mix)
            
            pole_vec.append(self.evals_spinon[i] + dE.ravel())
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
        E_gs_spinon, E_gs_rotor_vec = self.evals_spinon[0], self.evals_rotor_vec[:,0]
        E_gs = 2*E_gs_spinon + np.sum(E_gs_rotor_vec) + E0
        return E_gs
    
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

    def get_occupation_vec(self):
        """Get spinon/electron occupation at each lattice site."""
        return self.nf_vec
    
    def rotor_gs(self, theta_vals, site):
        """Calculates rotor ground state output in the exp(i l theta) basis"""
        l_vals = np.arange(-self.l_max, self.l_max + 1)
        evec_rotor_gs = self.evecs_rotor_vec[site][:,0]
        gs_vals = evec_rotor_gs @ np.exp(1j * l_vals * theta_vals)
        return gs_vals
    
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

        h0 = np.diag(self.epsilon_vec - self.mu_phys) - self.t_mat
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
 
    def check_gf_sum(self):
        """Residues should sum to 1 at each site."""
        _, R = self.__gf_spectral_data()
        sum = np.array([R[:, 0, 0].sum(), R[:, 1, 1].sum()])
        return sum