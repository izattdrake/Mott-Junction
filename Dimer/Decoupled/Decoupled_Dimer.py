import numpy as np
import sys

from pathlib import Path
from .broyden import iterate_broyden
from .newton_gauge import iterate_newton_gauge

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

        self.rotor_dim = 2 * l_max + 1 # Dimension for one site

        t_eff_mat_guess = 0.9 * self.t_mat
        h_vec_guess = -self.epsilon_vec
        Z_vec_guess = np.full(2, 0.9)
        K_vec_guess = np.full(2, -1.0)

        if ITERATION == 0: # Iteration w/ Newton's method & gauge fix
            self.evals_spinon, self.evecs_spinon, self.evals_rotor_vec,self.evecs_rotor_vec, self.h_vec, self.Z_vec, self.K_vec, self.t_eff_mat, self.nf_vec, self.L_vec, self.mu, self.count = iterate_newton_gauge(self.epsilon_vec, self.t_mat, self.U, t_eff_mat_guess, 
                                                                                                                                                                                                                     h_vec_guess, Z_vec_guess, K_vec_guess, self.beta, self.l_max, 
                                                                                                                                                                                                                     self.N_target, self.N_at)
        elif ITERATION == 1: # Iteration w/ modified Broyden's method
            self.evals_spinon, self.evecs_spinon, self.evals_rotor_vec,self.evecs_rotor_vec, self.h_vec, self.Z_vec, self.K_vec, self.t_eff_mat, self.nf_vec, self.L_vec, self.mu, self.count = iterate_broyden(self.epsilon_vec, self.t_mat, self.U, t_eff_mat_guess, 
                                                                                                                                                                                                                     h_vec_guess, Z_vec_guess, K_vec_guess, self.beta, self.l_max, 
                                                                                                                                                                                                                     self.N_target, self.N_at)
        else:
            raise ValueError('Dimer/Decoupled/Decoupled_Dimer.__init__ | ERROR: ITERATION argument')
        
    def __E0(self):
        """Calculate MF zero-point energy"""
        return -np.sum(self.K_vec * np.sqrt(self.Z_vec))

    def get_E_gs(self):
        """Calculate ground-state energy w/o physical subspace projection"""
        E0 = self.__E0()
        E_gs_spinon, E_gs_rotor_vec = self.evals_spinon[0], self.evals_rotor_vec[:,0]
        E_gs = 2*E_gs_spinon + np.sum(E_gs_rotor_vec) + E0
        return E_gs
    
    def rotor_gs_vals(self, theta_vals, site):
        """Calculates rotor ground state output in the exp(i l theta) basis"""
        l_vals = np.arange(-self.l_max, self.l_max+1)
        evec_rotor_gs = self.evecs_rotor_vec[site][:,0]
        gs_vals = evec_rotor_gs @ np.exp(1j * l_vals * theta_vals)
        return gs_vals