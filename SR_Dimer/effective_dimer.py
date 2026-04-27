import numpy as np
import scipy
import matplotlib.pyplot as plt
import itertools 
import sys

import utility.helpers as helpers
from utility.plotting import plot_rotor_pair_weight, plot_def_number_spectrum, plot_1D

# l vals corresponding to definite 0,1,2,3,4 particles 
P0_L_VALS = []
P1_L_VALS = [(-1,0), (0,-1)]
P2_L_VALS = [(0,0), (1,-1), (-1,1)]
P3_L_VALS = [(1,0), (0,1)]
P4_L_VALS = [(1,1)]

"""
    Code for the effective dimer
    Spinon and rotor Hamiltonians are coupled by effective spinon and rotor tunneling amplitudes
    Mean-field decouples spinon and rotors, separate site rotors still decoupled  
    Rotors at seperate sites are coupled, leading to an infinity x infinity dimensional rotor Fock space
    Computation is untenable for many-sites, but doable for dimer
    Good problem to bridge the gap between the exact and mean-field problems
"""

def get_eff_rotor_basis(l_max):
    """
        Returns |l_1 l_2> basis elements
    """

    l_vals = np.arange(-l_max, l_max + 1)
    basis = list(itertools.product(
        l_vals, l_vals
    ))

    return np.array(basis)


def get_eff_rotor_index(l1, l2, l_max):
    """
    Returns computation indx i \in [0, 2l_max + 1) for (l1,l2) value
    """
    index = (l1 + l_max)*(2*l_max + 1) + (l2 + l_max)
    return index


def get_H_rotor(U, h, J_eff, l_max):
    """
        Returns effective rotor Hamiltonian matrix in the |l1,l2> basis
    """
    def get_H_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime):
        """
            Returns effective rotor matrix elements <l1,l2|H_{theta}|l1_prime, l2_prime>
        """
        diag = ((U / 2) * (l1**2 + l2**2) + h*(l1 + l2)) * helpers.k_delta(l1,l1_prime) * helpers.k_delta(l2,l2_prime)
        off_diag = -J_eff * (helpers.k_delta(l1, l1_prime + 1) * helpers.k_delta(l2, l2_prime - 1) + helpers.k_delta(l1, l1_prime - 1) * helpers.k_delta(l2, l2_prime + 1))
        
        return diag + off_diag
    
    rotor_basis = get_eff_rotor_basis(l_max)
    dim_basis = len(rotor_basis)

    H_rotor = np.zeros((dim_basis, dim_basis))
    for i in range(dim_basis):

        l1, l2 = rotor_basis[i]

        for j in range(dim_basis):

            l1_prime, l2_prime = rotor_basis[j]
            H_rotor[i,j] = get_H_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime)

    return H_rotor


"""
    Computes t_eff
"""
def get_t_eff(U, t_mat, h_vec, J_eff_vec, beta, l_max):
    rotor_dim = (2*l_max + 1) * (2*l_max + 1)
    rotor_basis = get_eff_rotor_basis(l_max)

    basis_index = {tuple(state): i for i, state in enumerate(rotor_basis)}

    H_rotor = get_H_rotor(U, h_vec, J_eff_vec, l_max)
    rotor_energies, rotor_states = np.linalg.eigh(H_rotor)
    # Shift rotor energies to avoid large exponentials
    # Division in t_eff's formula cancels the shift
    rotor_energies = rotor_energies - rotor_energies[0]

    num_sum, den_sum = 0.0, 0.0

    for i in range(rotor_dim):
        rotor_energy = rotor_energies[i]
        bz = np.exp(-beta * rotor_energy)
        state = rotor_states[:,i]
        
        rotor_sum = 0.0

        for j in range(rotor_dim):
            l1, l2 = rotor_basis[j]

            # Handle l1, l2 calls greater than l_max
            # May give more specific conditions later depending on accuracy
            if (l1 == -l_max or l1 == l_max) and (l2 == -l_max or l2 == l_max):
                rotor_sum += 0

            elif (l1 == -l_max) and (not l2 == -l_max and not l2 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1+1,l2-1]] / 2

            elif (l1 == l_max) and (not l2 == -l_max and not l2 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1-1, l2+1]] / 2

            elif (l2 == -l_max) and (not l1 == -l_max and not l1 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1-1,l2+1]] / 2

            elif (l2 == l_max) and (not l1 == -l_max and not l1 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1+1,l2-1]] / 2

            elif (not l1 == -l_max and not l1 == l_max) and (not l2 == -l_max and not l2 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * (state[basis_index[l1+1,l2-1]] + state[basis_index[l1-1,l2+1]])/2
            
        num_sum += bz * rotor_sum
        den_sum += bz
    
    t_eff_mat = t_mat * (num_sum / den_sum)
    return t_eff_mat
        
"""
    Returns the index corresponding to a 
"""
def get_spinon_index(n1_up, n1_down, n2_up, n2_down):
    index = 8 * n1_up + 4 * n1_down + 2 * n2_up + n2_down
    return index

"""
    Returns the superposition of spinon states coming from gamma excitations, the diagoanlized Hamiltonian states
    I want to use this to check the Fermi-Dirac vs exact picture
    The hand-written calculation is kind of cumbersome though so idk if I'll get around to it
"""
def get_spinon_from_gamma():
    return 

"""
    Return gamma-particle energies
"""
def get_E_plus(epsilon, h, t_eff):
    E_plus = epsilon - h - t_eff
    return E_plus

def get_E_minus(epsilon, h, t_eff):
    E_minus = epsilon - h + t_eff
    return E_minus 


"""
    Returns J_eff by taking a thermal average of the hopping operator over the spinon Hamiltonian
    Traces are evaluated by going to the eigenstate basis
    Can compare with the approach taken in the MF picture where diagonalized states were averaged with the FD distribution
"""
def get_J_eff(epsilon, h, t, t_eff, beta):
    E_plus, E_minus = get_E_plus(epsilon, h, t_eff), get_E_minus(epsilon, h, t_eff)
    J_eff = t * (helpers.fermi_dirac(E_plus, beta) - helpers.fermi_dirac(E_minus, beta))

    return J_eff

"""
    Returns the fermion occupation at a given site
    Since the system is homogenous, <n_i>_f is the same at both sites
"""
def get_nf(epsilon, h, t_eff, beta, mu=0.0):
    E_plus, E_minus = get_E_plus(epsilon, h, t_eff), get_E_minus(epsilon, h, t_eff)
    nf = helpers.fermi_dirac(E_plus, beta, mu) + helpers.fermi_dirac(E_minus, beta, mu)
    
    return nf

"""
    Return derivative of nf with respect to h
"""
def get_d_nf(epsilon, h, dh, t_eff, beta):
    get_t_eff(U, t, h, J_eff)
    nf_p = get_nf(epsilon, h + dh/2.0, t_eff, beta)
    nf_m = get_nf(epsilon, h - dh/2.0, t_eff, beta)

    d_nf = (nf_p - nf_m) / dh

    return d_nf


"""
    Returns the thermal average of L over the rotor Hamiltonian
"""
def get_L(U, h, J_eff, beta, l_max, SITE=1):
    rotor_dim = (2 * l_max + 1)**2
    basis = get_eff_rotor_basis(l_max)

    H_rotor = get_H_rotor(U, h, J_eff, l_max)
    rotor_energies, rotor_states = np.linalg.eigh(H_rotor)

    rotor_energies = rotor_energies - rotor_energies[0]

    num_sum, den_sum = 0.0, 0.0

    for i in range(rotor_dim):
        rotor_energy = rotor_energies[i]
        boltzmann = np.exp(-beta * rotor_energy)
        state = rotor_states[:,i]

        rotor_sum = 0.0

        for j in range(rotor_dim):
            l1, l2 = basis[j]

            if SITE == 0: 
                rotor_sum += l1 * np.abs(state[j])**2
            elif SITE == 1:
                rotor_sum += l2 * np.abs(state[j])**2
            else:
                raise ValueError('SITE must be 1 or 2')
            
        num_sum += boltzmann * rotor_sum
        den_sum += boltzmann 

    L = num_sum / den_sum

    return L


"""
    Return derivative of L with respect to h
"""
def get_d_L(U, h, dh, J_eff, beta, l_max):
    L_plus = get_L(U, h + dh/2, J_eff, beta, l_max)
    L_minus = get_L(U, h - dh/2, J_eff, beta, l_max)

    d_L = (L_plus - L_minus) / dh

    return d_L



def get_Jac(epsilon, U, t_eff, h, dh, mu, K, beta, l_max, sites):
    """
        Return Jacobian used in Newton's method
    """
    def get_Jac_nf():
        """
            Returns spinon occupancy contribution to Jacobian 
        """
        J = np.zeros((2,2))
        for j in range(2):
            h_p, h_m = h.copy(), h.copy()
            h_p[j] += dh / 2.0
            h_m[j] -= dh / 2.0

            eigvals_p, eigvecs_p = np.linalg.eigh(get_)
        return
    
    def get_Jac_L():
        """
            Returns rotor contribution to Jacobian
        """


"""
    Iterate over parameters until convergence
    Code isn't pretty due to having to redundantly define parameters at some point, but it gets the job done
"""
def update_params(epsilon_vec, U, t_mat, h_vec_guess, t_eff_mat_guess, J_eff_vec_guess, beta, l_max):
    ALPHA_H = 0.1
    ALPHA_T_EFF = 0.1
    ALPHA_J_EFF = 0.1

    TOL_H = 1e-8
    TOL_T_EFF = 1e-8
    TOL_J_EFF = 1e-8

    dh = 1e-5
    did_converge = False

    # Initial values to guess for iteration
    # Assume J_eff ~ t i.e. 
    h_vec_update, t_eff_mat_update, J_eff_vec_update = h_vec_guess, t_eff_mat_guess, J_eff_vec_guess

    t_eff_mat_update = get_t_eff(U, t_mat, h_vec_update, J_eff_vec_guess, beta, l_max)
    J_eff_vec_update = get_J_eff(epsilon_vec, h_vec_update, t_mat, t_eff_mat_update, beta)

    while not did_converge:
        h_vec, t_eff_mat, J_eff_vec  = h_vec_update, t_eff_mat_update, J_eff_vec_update

        nf_vec = get_nf(epsilon_vec, h_vec, t_eff_mat, beta)
        L_vec = get_L(U, h_vec, J_eff_vec, beta, l_max)

        d_nf_vec = get_d_nf(epsilon_vec, h_vec, dh, t_eff_mat, beta)
        d_L_vec = get_d_L(U, h_vec, dh, J_eff_vec, beta, l_max)

        f = nf_vec - L_vec - 1.0
        J = get_Jacobian()

        try:
            delta_h = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            sys.exit('Singular Jacobian')

        # Newton's method
        h_vec_update = h_vec - ALPHA_H * delta_h

        # Get updated params
        t_eff_mat_update = (1 - ALPHA_T_EFF) * t_eff_mat + ALPHA_T_EFF * get_t_eff(U, t_mat, h_vec_update, J_eff, beta, l_max)
        J_eff_vec_update = (1 - ALPHA_J_EFF) * J_eff_vec + ALPHA_J_EFF * get_J_eff(epsilon_vec, h_vec_update, t_mat, t_eff_mat_update, beta)

        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_mat_update, t_eff_mat, atol=TOL_T_EFF) and
            np.allclose(J_eff_vec_update, J_eff_vec, atol=TOL_J_EFF)
        )

    nf_vec = get_nf(epsilon_vec, h_vec, t_eff_mat, beta)
    d_L_vec = get_L(U, h_vec, J_eff_vec, beta, l_max)

    return h_vec_update, t_eff_mat_update, J_eff_vec_update, nf_vec


if __name__ == '__main__':
    PBC = True

    N_target = 2 # Half filling

    l_max = 12
    l_vals = np.arange(-l_max, l_max + 1)

    t_val = 1.0;

    t = np.diag(np.full(1, t_val), k=1) + np.diag(np.full(1, t_val), k=-1)
    epsilon = np.zeros(2)

    """
        Initial parameter guesses for iterative Newton's method
    """
    t_eff_guess = 0.8 * t
    h_guess = -epsilon

    U_vals = np.linspace(0,0.01,10)
    beta = 50.0

    count = 0

    for U in U_vals:
        H_exact = exact.get_H_exact(epsilon[0], U, t_nn, exact.get_def_particle_basis(2))
        eigvals_exact, _ = np.linalg.eigh(H_exact)
        

        h, Z, K, t_eff, nf, L, mu, num_cycles = update_params(epsilon)

