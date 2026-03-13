import numpy as np
import scipy
import matplotlib.pyplot as plt
import itertools 
import sys

import utility.helpers as helpers
from utility.plotting import plot_rotor_pair_weight

"""
    Code for the effective dimer
    Spinon and rotor Hamiltonians are coupled by effective spinon and rotor tunneling amplitudes
    No mean-field approximation
    Rotors at seperate sites are coupled, leading to an infinity x infinity dimensional rotor Fock space
    Computation is untenable for many-sites, but doable for dimer
    Good problem to bridge the gap between the exact and mean-field problems
"""

"""
    Returns |l_1 l_2> basis elements
"""
def get_eff_rotor_basis(l_max):
    l_vals = np.arange(-l_max, l_max + 1)
    basis = list(itertools.product(
        l_vals, l_vals
    ))

    return np.array(basis)

"""
    Returns l_1 and l_2 corresponding to a given index
"""
def get_eff_rotor_index(l1, l2, l_max):
    index = (l1 + l_max) * (2 * l_max + 1) + (l2 + l_max)

    return index

"""
    Returns effective rotor matrix elements <l1,l2|H_{theta}|l1_prime, l2_prime>
"""
def get_eff_H_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime):
    diag = ((U / 2) * (l1**2 + l2**2) + h*(l1 + l2)) * helpers.k_delta(l1,l1_prime) * helpers.k_delta(l2,l2_prime)
    off_diag = -J_eff * (helpers.k_delta(l1, l1_prime + 1) * helpers.k_delta(l2, l2_prime - 1) + helpers.k_delta(l1, l1_prime - 1) * helpers.k_delta(l2, l2_prime + 1))
    
    return diag + off_diag

"""
    Returns effective rotor Hamiltonian matrix in the |l1,l2> basis
"""
def get_eff_H_rotor(U, h, J_eff, l_max):
    rotor_basis = get_eff_rotor_basis(l_max)
    dim_basis = len(rotor_basis)

    H_rotor = np.zeros((dim_basis, dim_basis))
    for i in range(dim_basis):

        l1, l2 = rotor_basis[i]

        for j in range(dim_basis):

            l1_prime, l2_prime = rotor_basis[j]
            H_rotor[i,j] = get_eff_H_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime)

    return H_rotor

"""
    Computes t_eff
"""
def get_t_eff(U, t, h, J_eff, beta, l_max):
    rotor_dim = (2 * l_max + 1)**2
    rotor_basis = get_eff_rotor_basis(l_max)

    basis_index = {tuple(state): i for i, state in enumerate(rotor_basis)}

    H_rotor = get_eff_H_rotor(U, h, J_eff, l_max)
    rotor_energies, rotor_states = np.linalg.eigh(H_rotor)
    # Shift rotor energies to avoid large exponentials
    # Division in t_eff's formula cancels the shift
    rotor_energies = rotor_energies - rotor_energies[0]

    num_sum, den_sum = 0.0, 0.0

    for i in range(rotor_dim):
        rotor_energy = rotor_energies[i]
        boltzmann = np.exp(-beta * rotor_energy)
        state = rotor_states[:,i]
        
        rotor_sum = 0.0

        for j in range(rotor_dim):
            l1, l2 = rotor_basis[j]

            # Handle l1, l2 calls greater than l_max
            # May give more specific conditions later depending on accuracy
            if (l1 == -l_max or l1 == l_max) and (l2 == -l_max or l2 == l_max):
                rotor_sum += 0

            elif (l1 == -l_max) and (not l2 == -l_max and not l2 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1+1,l2-1]]/2

            elif (l1 == l_max) and (not l2 == -l_max and not l2 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1-1, l2+1]]/2

            elif (l2 == -l_max) and (not l1 == -l_max and not l1 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1-1,l2+1]] / 2

            elif (l2 == l_max) and (not l1 == -l_max and not l1 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * state[basis_index[l1+1,l2-1]] / 2

            elif (not l1 == -l_max and not l1 == l_max) and (not l2 == -l_max and not l2 == l_max):
                rotor_sum += np.conj(state[basis_index[l1,l2]]) * (state[basis_index[l1+1,l2-1]] + state[basis_index[l1-1,l2+1]])/2
            
        num_sum += boltzmann * rotor_sum
        den_sum += boltzmann
    
    t_eff = t * (num_sum / den_sum)
    return t_eff
        
"""
    Returns the index corresponding to a 
"""
def get_spinon_index(n1_up, n1_down, n2_up, n2_down):
    index = 8 * n1_up + 4 * n1_down + 2 * n2_up + n2_down
    return index

"""
    Returns the superposition of spinon states coming from gamma excitations, the diagoanlized Hamiltonian states
    I want to use this to check the Fermi-Dirac vs 
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
    Returns the thermal average of L over the rotor Hamiltonian
"""
def get_L(U, h, J_eff, beta, l_max, SITE=1):
    rotor_dim = (2 * l_max + 1)**2
    basis = get_eff_rotor_basis(l_max)

    H_rotor = get_eff_H_rotor(U, h, J_eff, l_max)
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
    Return derivative of nf with respect to h
"""
def get_d_nf(epsilon, h, dh, t_eff, beta):
    nf_plus = get_nf(epsilon, h + dh/2, t_eff, beta)
    nf_minus = get_nf(epsilon, h - dh/2, t_eff, beta)

    d_nf = (nf_plus - nf_minus) / dh

    return d_nf


"""
    Return derivative of L with respect to h
"""
def get_d_L(U, h, dh, J_eff, beta, l_max):
    L_plus = get_L(U, h + dh/2, J_eff, beta, l_max)
    L_minus = get_L(U, h - dh/2, J_eff, beta, l_max)

    d_L = (L_plus - L_minus) / dh

    return d_L


"""
    Iterate over parameters until convergence
    Code isn't pretty due to having to redundantly define parameters at some point, but it gets the job done
"""
def update_params(epsilon, U, t, h_guess, J_eff_guess, beta, l_max):
    dh = 1e-4
    did_converge = False

    # Initial values to guess for iteration
    # Assume J_eff ~ t i.e. 
    h_update = h_guess
    t_eff_update = get_t_eff(U, t, h_update, J_eff_guess, beta, l_max)
    J_eff_update = get_J_eff(epsilon, h_update, t, t_eff_update, beta)

    while not did_converge:
        h = h_update
        t_eff = t_eff_update 
        J_eff = J_eff_update

        nf = get_nf(epsilon, h, t_eff, beta)
        L = get_L(U, h, J_eff, beta, l_max)

        d_nf = get_d_nf(epsilon, h, dh, t_eff, beta)
        d_L = get_d_L(U, h, dh, J_eff, beta, l_max)

        # Newton's method
        h_update = h - (nf - L - 1)/(d_nf - d_L)

        # Get updated params
        t_eff_update = get_t_eff(U, t, h_update, J_eff, beta, l_max)
        J_eff_update = get_J_eff(epsilon, h_update, t, t_eff_update, beta)

        if (np.abs(h_update - h) < 1e-4) and (np.abs(t_eff_update - t_eff) < 1e-4) and (np.abs(J_eff_update - J_eff) < 1e-4):
            did_converge = True

    return h_update, t_eff_update, J_eff_update, nf

def spinon_spectrum(omega, beta):
    E_plus = get_E_plus(epsilon, h, t_eff)
    E_minus = get_E_minus(epsilon, h, t_eff)

    energies = [E_plus, E_minus]

    spectrum = 0.0
    for energy in energies:
        spectrum += helpers.d_delta(omega - energy) * (1 - helpers.fermi_dirac(energy, beta) + helpers.d_delta(omega + energy)) * helpers.fermi_dirac(energy, beta)

    return spectrum

"""
    Rotor spectral function
    Calculation is incredibly computationally expensive
    Basically useless
"""
def rotor_spectrum(omega, energies, states, beta, l_max):
    rotor_dim = (2 * l_max + 1)**2

    rotor_basis = get_eff_rotor_basis(l_max)
    basis_index = {tuple(state): i for i, state in enumerate(rotor_basis)}
    energies = energies - energies[0]

    num_sum, den_sum = 0.0, 0.0
    count = 0
    for i in range(rotor_dim):
        state1 = states[:,i]
        energy1 = energies[i]

        boltzmann = np.exp(-beta * energy1)
        rotor_sum = 0.0

        for j in range(rotor_dim):
            state2 = states[:,j]
            energy2 = energies[j]

            for m in range(rotor_dim):
                l1, l2 = rotor_basis[m]

                for n in range(rotor_dim):
                    l1_prime, l2_prime = rotor_basis[n]
                    
                    if (not l1 == -l_max) and (not l1_prime == l_max):
                        rotor_sum += (
                            np.conj(state1[basis_index[l1,l2]]) * state1[basis_index[l1_prime+1,l2_prime]]
                            * np.conj(state2[basis_index[l1_prime,l2_prime]]) * state2[basis_index[l1-1,l2]]
                            * helpers.d_delta(omega - (energy1 - energy2))
                        )

        num_sum += boltzmann * rotor_sum
        den_sum += boltzmann

    spectrum = num_sum / den_sum
    return spectrum

if __name__ == '__main__':
    l_max = 5
    rotor_dim = (2*l_max + 1)**2

    epsilon = 0.5
    t = 1.0
    U = 5.0
    beta = 10    # particle number in (0,2)
    particle_num = 1.0
    h, t_eff, J_eff, nf = update_params(epsilon, U, 0.8 * t, 0.3, 0.8 * t, beta, l_max)

    print(f'h: {h}')
    print(f't_eff: {t_eff}')
    print(f'J_eff: {J_eff}')
    print(f'nf: {nf}')

    H_rotor = get_eff_H_rotor(U, h, J_eff, l_max)
    energies, states = np.linalg.eigh(H_rotor)

    rotor_spectrum(1, energies, states, beta, l_max)
        
        


