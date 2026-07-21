import numpy as np
import sys

from pathlib import Path
from .Iteration.newton_gauge import iterate_newton_gauge
from .Iteration.broyden import iterate_broyden

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility import helpers

# ====================================================================
#   COUPLED DIMER CLASS INSTANCE FOR CALCULATING OBSERVABLES
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
#   quantities like the Green's function, self-energy, quasiparticle
#   weight, pair occupancy, heat capacity, etc.  
# --------------------------------------------------------------------
#   Methods used to solve the self-consistency condition can be found
#   in Dimer/Coupled/core.py. 
# ====================================================================

class Coupled_Dimer:
    def __init__(self, epsilon, U, t, beta, l_max, N_target=2, ITERATION=0):
        self.epsilon_vec = np.array([-epsilon/2.0, epsilon/2.0]) # Even/opposite doping to preserve PHS, restricts chem. pot.
        self.U, self.t, self.beta, self.l_max, self.N_target = U, t, beta, l_max, N_target
        
        self.dim_rotor = 2*l_max + 1 # Dimension for one site, full rotor Hilbert space is dim**2

        t_eff_guess = 0.8 * self.t
        J_eff_guess = 0.5 * self.t
        h_vec_guess = -self.epsilon_vec

        if ITERATION == 0: # Iteration w/ Newton's method & gauge fix
            self.evals_spinon, self.evecs_spinon, self.evals_rotor, self.evecs_rotor, self.h_vec, self.t_eff, self.J_eff, self.nf_vec, self.L_vec, self.mu, self.count = iterate_newton_gauge(self.epsilon_vec, self.U, self.t, h_vec_guess, t_eff_guess, 
                                                                                                                                                                                              J_eff_guess, self.beta, self.l_max, self.N_target)
        elif ITERATION == 1: # Iteration w/ modified Broyden's method
            self.evals_spinon, self.evecs_spinon, self.evals_rotor, self.evecs_rotor, self.h_vec, self.t_eff, self.J_eff, self.nf_vec, self.L_vec, self.mu, self.count = iterate_broyden(self.epsilon_vec, self.U, self.t, h_vec_guess, t_eff_guess, 
                                                                                                                                                                                         J_eff_guess, self.beta, self.l_max, self.N_target)
        else:
            raise ValueError('Coupled_Dimer.__init__ | ERROR: ITERATION argument')

        """Add U/2 shift missing from (U/2)L^2 representation of the interaction.
        Note that U/2 is only the correct shift in PHS systems, that is half-filling
        with even-and-opposite doping on either site."""
        self.mu_phys = self.mu + self.U / 2.0 # Account for U/2 shift in SR formalism
    
    # ====================================================================
    #   PRIVATE METHODS
    # ====================================================================
    def __E0(self):
        """Calculate zero-point energy from the MF decoupling | E0 = \sum_{ij} J^{eff}_{ij} <cos(theta_i - theta_j)>"""
        l1 = np.repeat(np.arange(-self.l_max, self.l_max+1), self.dim_rotor)
        l2 = np.tile(np.arange(-self.l_max, self.l_max+1), self.dim_rotor)

        # Build shift operator | maps |l1,l2> -> |l1+1,l2-1> + |l1-1,l2+1>
        shift_op = np.zeros((self.dim_rotor**2, self.dim_rotor**2)) 

        mask = (l1 < self.l_max) & (l2 > -self.l_max) # Filters shifts inside the truncated Hilbert space
        bra_index = (l1[mask] + self.l_max + 1)*self.dim_rotor + (l2[mask] + self.l_max - 1)
        ket_index = np.where(mask)[0]

        shift_op[bra_index, ket_index] = 1.0 # |l1,l2> -> |l1+1,l2-1>
        shift_op += shift_op.T               # |l1,l2> -> |l1-1,l2+1>

        bz = helpers._bz_weights(self.evals_rotor, self.beta)

        # <psi_n|S|psi_n>        
        exp_vals = np.real(np.sum(np.conj(self.evecs_rotor) * (shift_op @ self.evecs_rotor), axis=0))
        E0 = 2 * self.J_eff * np.sum(bz * exp_vals) / 2

        return E0
    
    def __shift_op(self, site):
        """Matrix of e^{i theta_j} in the |l1,l2> basis"""
        l1 = np.repeat(np.arange(-self.l_max, self.l_max+1), self.dim_rotor)
        l2 = np.tile(np.arange(-self.l_max, self.l_max+1), self.dim_rotor)

        shift_op = np.zeros((self.dim_rotor**2, self.dim_rotor**2))
        if site == 0:
            mask = l1 < self.l_max
            bra_index = (l1[mask] + self.l_max + 1)*self.dim_rotor + (l2[self.dim_rotor] + self.l_max)
        elif site == 1:
            mask = l2 < self.l_max
            bra_index = (l1[mask] + self.l_max) * self.dim_rotor + (l2[mask] + self.l_max + 1)

        ket_index = np.where(mask)[0]
        shift_op[bra_index, ket_index] = 1.0
        return shift_op
    
    def __gf_spectral_data(self, tol=1e-12):
        """Calculate Green's function data. Poles P_k, residues R_k[i,j] with G_ij(z) = sum_k R_k[i,j]/(z-P_k).
            Cached and saved on first call. Used for quickly calculating values."""
        if hasattr(self, '_gf_cache'):
            return self._gf_cache
        
        # Rotor matrix elements: <m|e^{i\theta_j}|n>
        mel_rotor = np.stack([self.evecs_rotor.T @ self.__shift_op(site) @ self.evecs_rotor for site in range(2)])

        dE_rotor = self.evals_rotor[:, None] - self.evals_rotor[None, :]

        bz_rotor = helpers._bz_weights(self.evals_rotor, self.beta)     # Rotor boltzmann probabilities
        p_rotor_ket, p_rotor_bra = bz_rotor[None, :], bz_rotor[:, None] 

        pole_vec, res_vec = [], []
        for i in range(2):
            fd_spinon_vals = helpers._fermi(self.evals_spinon[i])
            
            sr_mix = fd_spinon_vals * p_rotor_bra + (1.0 - fd_spinon_vals) * p_rotor_ket # Convolution of spinon/rotor
            spinon_coeff = self.evecs_spinon[:,i] # spinon state coefficients

            R = np.einsum('i,j,inm,jnm,nm->nmij', spinon_coeff, spinon_coeff, mel_rotor, mel_rotor, sr_mix)
            
            pole_vec.append(self.evals_spinon[i] + dE_rotor.ravel())
            res_vec.append(R.reshape(-1, 2, 2))

        P, R = np.concatenate(pole_vec), np.concatenate(res_vec)
        finite_res_mask = np.abs(R).max(axis=(1, 2)) > tol # Throw away vanishing residues (matrix elements vanish)
        self._gf_cache = (P[finite_res_mask], R[finite_res_mask])
        
        return self._gf_cache
    
    # ====================================================================
    #   PUBLIC METHODS
    # ====================================================================
    def get_E_gs(self):
        """Calculate ground-state energy w/o physical subspace projection"""
        E0 = self.__E0()
        E_gs_spinon, E_gs_rotor = self.evals_spinon[0], self.evals_rotor[0]
        E_gs = 2*E_gs_spinon + E_gs_rotor + E0 # Spinon GS counted twice since it corresponds to 2x2 spin-subspace energy
        return E_gs
    
    def get_gf(self, z, i=0, j=0):
        """Calculate Green's function G_ij(z) on complex frequencies z (Matsubara or retarded okay) |
           Can pass scalar or vector values of z. Returns shape (len(z))"""
        z = np.atleast_1d(np.asarray(z, dtype=complex)) # If z is a scalar, make array
        P, R = self.__gf_spectral_data() # Pull residue and pole cache data
        gf_vals = np.sum(R[None, :, i, j] / (z[:, None] - P[None, :]), axis=1) 
        return gf_vals
    
    def get_gf_matrix(self, z):
        """Calculate the full 2x2 Green's function matrix at scalar or vector values of z | returns shape (len(z), 2, 2)"""
        z = np.atleast_1d(np.asarray(z, dtype=complex)) # If z is a scalar, make array
        P, R = self.__gf_spectral_data() # Pull poles and residues
        gf_mat = np.sum(R[None, :, :, :]
                      / (z[:, None, None, None] - P[None, :, None, None]), axis=1) # Form matrix
        return gf_mat
    
    def get_self_energy_matrix(self, z):
        """Self-energy from the Dyson equation Sigma(z) = G0(z)^{-1} - G(z)^{-1}
        Physical chemical potential mu_phys = mu + U/2 must be included to get
        Hartree shift seen in exact diagonalization benchmark"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))

        h0 = np.array([[self.epsilon_vec[0] - self.mu_phys, -self.t],
                       [-self.t, self.epsilon_vec[1] - self.mu_phys]])
        gf_bare = np.linalg.inv(z[:, None, None] * np.eye(2) - h0[None, :, :])

        self_energy_mat = np.linalg.inv(gf_bare) - np.linalg.inv(self.get_gf_matrix(z))

        return self_energy_mat
    
    def get_spectral(self, freq_vec, site=0, delta=1e-4):
        """Calculate site-resolved spectral function output from retarded Green's function"""
        if site in (0,1):
            spectral_vals = -np.imag(self.get_gf(freq_vec + 1j * delta, site, site)) / np.pi
            return spectral_vals
        else:
            raise ValueError(f'Coupled_Dimer.get_spectral | Error: site argument must be 0 or 1')
    
    def get_qp_weight(self, site=0, domega=1e-5, delta=1e-4):
        """Calculate quasiparticle weight Z as a finite difference of the retarded Green's function"""
        z = np.array([-domega + 1j*delta, domega + 1j*delta])
        self_energy_vals = self.get_self_energy_matrix(z)[:, site, site]
        Z = 1.0 / (1.0 - (self_energy_vals[1].real - self_energy_vals[0].real) / (2.0 * domega))
        return Z
    
    def get_Z_vec(self):
        """Thermal average of cos(theta_i) at each site | Assume no flux such that <cos(theta)> = <e^{i*theta}>"""
        Z_vec = np.zeros(2)
        for site in range(2):
            shift_op = self.__shift_op(site) # Create shift operator for respective site

            # <l1,l2|e^{i\theta}|l1,l2>
            exp_vals = np.real(np.sum(np.conj(self.evecs_rotor) * (shift_op @ self.evecs_rotor), axis=0))
            
            bz = helpers._bz_weights(self.evals_rotor)
            Z_vec[site] = np.sum(bz * exp_vals)

        return Z_vec
