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
        
    def __E0(self):
        """Calculate zero-point energy from the MF decoupling | E0 = \sum_{ij} J^{eff}_{ij} <cos(theta_i - theta_j)>"""
        l1 = np.repeat(np.arange(-self.l_max, self.l_max+1), self.dim_rotor)
        l2 = np.tile(np.arange(-self.l_max, self.l_max+1), self.dim_rotor)

        # Build shift operator | maps |l1,l2> -> |l1+1,l2-1> + |l1-1,l2+1>
        S = np.zeros((self.dim_rotor**2, self.dim_rotor**2)) 

        mask = (l1 < self.l_max) & (l2 > -self.l_max) # Filters shifts inside the truncated Hilbert space
        bra_index = np.where(mask)[0]
        ket_index = (l1[mask] + self.l_max + 1)*self.dim_rotor + (l2[mask] + self.l_max - 1)

        S[ket_index, bra_index] = 1.0 # |l1,l2> -> |l1+1,l2-1>
        S[bra_index, ket_index] = 1.0 # |l1,l2> -> |l1-1,l2+1>

        # Shift energies by GS energy to avoid overflow
        energies = self.evals_rotor - self.evals_rotor[0]
        bz = np.exp(-self.beta * energies)

        # <psi_n|S|psi_n>        
        exp_vals = np.real(np.sum(np.conj(self.evecs_rotor) * (S @ self.evecs_rotor), axis=0))
        E0 = 2 * self.J_eff * np.sum(bz * exp_vals) / (2 * np.sum(bz))

        return E0
    
    def get_E_gs(self):
        """Calculate ground-state energy w/o physical subspace projection"""
        E0 = self.__E0()
        E_gs_spinon, E_gs_rotor = self.evals_spinon[0], self.evals_rotor[0]
        E_gs = 2*E_gs_spinon + E_gs_rotor + E0 # Spinon GS counted twice since it corresponds to 2x2 spin-subspace energy
        return E_gs
    
    def get_Z_vec(self):
        """Thermal average of cos(theta_i) at each site | Assume no flux such that <cos(theta)> = <e^{i*theta}>"""
        energies = self.evals_rotor - self.evals_rotor[0]
        bz = np.exp(-self.beta * energies)

        Z_vec = np.zeros(2)

        l1, l2 = np.repeat(np.arange(-self.l_max, self.l_max+1), self.dim_rotor), np.tile(np.arange(-self.l_max, self.l_max+1), self.dim_rotor)
    
        for site in range(2):
            # Create shift operator for respective site
            S = np.zeros((self.dim_rotor**2, self.dim_rotor**2))
            if site == 0:
                mask = l1 < self.l_max
                bra_index = np.where(mask)[0]
                ket_index = (l1[mask] + 1 + self.l_max)*self.dim_rotor + (l2[mask] + self.l_max)
            else:
                mask = l2 < self.l_max
                bra_index = np.where(mask)[0]
                ket_index = (l1[mask] + self.l_max)*self.dim_rotor + (l2[mask] + 1 + self.l_max)

            S[ket_index, bra_index] = 1.0

            # <l1,l2|e^{i\theta}|l1,l2>
            exp_vals = np.real(np.sum(np.conj(self.evecs_rotor) * (S @ self.evecs_rotor), axis=0))
            Z_vec[site] = np.sum(bz * exp_vals) / np.sum(bz)

        return Z_vec