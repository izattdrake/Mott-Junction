import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 
import os

from utility.helpers import *

warnings.simplefilter("error", RuntimeWarning)

SPINON_DIM = 4
N_AT = 2
T = 300
BETA = 20

def get_nf(spinon_eigvals, spinon_eigvecs):
    nf = np.zeros(2)

    for i in range(2):
        sum = 0.0
        for j in range(2):
            energy = spinon_eigvals[j]
            sum += (np.abs(spinon_eigvecs[i,j]) ** 2) * fermi_dirac(energy, BETA)

        nf[i] = N_AT * sum

    return nf

"""
Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
"""
def get_L(rotor_site_eigvals, rotor_site_eigvecs, L_MAX):
    ROTOR_DIM = (2 * L_MAX + 1)
    L = np.zeros(2)

    # Sum over lattice sites
    for i in range(2):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        rotor_eigvals = rotor_site_eigvals[i] - rotor_site_eigvals[i,0]
        rotor_eigvecs = rotor_site_eigvecs[i]

        num_sum, den_sum = 0.0, 0.0

        # Sum over rotor energies and states
        for j in range(ROTOR_DIM):
            rotor_energy = rotor_eigvals[j]
            rotor_eigenvec = rotor_eigvecs[:,j]

            boltzmann = np.exp(-BETA * rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range(ROTOR_DIM):
                l = index - L_MAX
                num_sum += boltzmann * (np.abs(rotor_eigenvec[index]) ** 2) * l
        
            den_sum += boltzmann 

        L[i] = num_sum / den_sum

    return L

"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_site_eigvals, rotor_site_eigvecs, L_MAX):
    ROTOR_DIM = (2 * L_MAX + 1)
    Z = np.zeros(2)

    for i in range (2):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        rotor_eigvals = rotor_site_eigvals[i] - rotor_site_eigvals[i,0]        
        rotor_eigvecs = rotor_site_eigvecs[i]

        num_sum = 0.0
        den_sum = 0.0
        
        # Sum over rotor states and energies. There are ROTOR_DIM total eigenstates
        for j in range (ROTOR_DIM):
            rotor_state = rotor_eigvecs[:,j]
            rotor_energy = rotor_eigvals[j]

            boltzmann = np.exp(-BETA*rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range (ROTOR_DIM):
                l = index - L_MAX 
                # Handle truncating the Hilbert space by simply ignoring the L_CUT + 1 state
                if index == 0:
                    num_sum += boltzmann * np.conj(rotor_state[index])*rotor_state[index + 1]
                elif index == ROTOR_DIM - 1:
                    num_sum += boltzmann * np.conj(rotor_state[index])*rotor_state[index - 1]
                else:
                    num_sum += boltzmann * np.conj(rotor_state[index])*(rotor_state[index + 1] + rotor_state[index - 1])
                    
            # Partition function, make function and calculate once?
            den_sum += boltzmann

        num_sum, den_sum = np.real(num_sum), np.real(den_sum)
        Z[i] = (num_sum / (2*den_sum))**2

    return Z

"""
Return effective tunneling amplitude matrix
"""
def get_t_eff(t, Z):
    t_eff = t * np.sqrt(Z[0] * Z[1])
    return t_eff


"""
Return real-space spinon Hamiltonian in the upper-block
The single-particle spinon Hamiltonian is block diagonalized with each block corresponding to a spin
I think we only have to diagonalize a single one of these blocks, but I am not sure
"""
def get_H_spinon(t_eff, mu, h):
    H = np.zeros((2, 2))

    np.fill_diagonal(H, -mu - h)
    H[0,1], H[1,0] = -t_eff, -t_eff

    return H


"""
Return array of rotor Hamiltonians at each site
"""
def get_H_rotors(U, h, K, L_MAX):
    ROTOR_DIM = 2 * L_MAX + 1
    H_sites = np.zeros((2, ROTOR_DIM, ROTOR_DIM))
    l_vals = np.arange(-L_MAX, L_MAX + 1)
    
    for i in range(2):
        H = np.zeros((ROTOR_DIM, ROTOR_DIM))

        diag = ((U / 2.0) * l_vals + h[i]) * l_vals
        np.fill_diagonal(H, diag)

        off_diag = K[i] / 2.0
        H += np.diag(np.full(ROTOR_DIM - 1, off_diag), k=1)
        H += np.diag(np.full(ROTOR_DIM - 1, off_diag), k=-1)

        H_sites[i] = H

    return H_sites

"""
Return K
"""
def get_K(t, Z, spinon_eigvals, spinon_eigvecs):
    K = np.zeros(2)

    sum0 = 0.0
    sum1 = 0.0
    
    for i in range(2):
        energy = spinon_eigvals[i]
        sum0 += t * np.sqrt(Z[1]) * spinon_eigvecs.conj().T[i,0] * spinon_eigvecs[1,i] * fermi_dirac(energy, BETA)
        sum1 += t * np.sqrt(Z[0]) * spinon_eigvecs.conj().T[i,1] * spinon_eigvecs[0,i] * fermi_dirac(energy, BETA)
    
    K[0], K[1] = -2 * N_AT * sum0, -2 * N_AT * sum1

    return K

"""
    Returns the next iteration of the Lagrange parameters using Newton's method
    The minimizing function is f = <n_f(i)> - N/2 - <L(i)>
    Newton's method sets h' = h - f/f'
    Derivatives are calculted with a finite difference method
    TODO: Find a closed form analytic solution using perturbation theory
    NOTE: Analytic solutions may not be possible due to recursion
"""
def update(t, U, mu, h, Z, K, L_MAX=5):
    ROTOR_DIM = 2 * L_MAX + 1

    dh = 1e-4
    h_p, h_m = h + (dh / 2), h - (dh / 2)
    t_eff = get_t_eff(t, Z)
    
    """
    Calculate Hamiltonians needed for parameters
    """
    # H_f(h)
    spinon_H = get_H_spinon(t_eff, mu, h)
    spinon_eigvals, spinon_eigvecs = np.linalg.eigh(spinon_H)

    # H_f(h+dh)
    spinon_H_p = get_H_spinon(t_eff, mu, h_p)
    spinon_eigvals_p, spinon_eigvecs_p = np.linalg.eigh(spinon_H_p)

    # H_f(h-dh)
    spinon_H_m = get_H_spinon(t_eff, mu, h_m)
    spinon_eigvals_m, spinon_eigvecs_m = np.linalg.eigh(spinon_H_m)

    # H_\theta(h)
    rotor_site_eigenvecs = np.zeros((2, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals = np.zeros((2, ROTOR_DIM))
    H_rotors = get_H_rotors(U, h, K, L_MAX)
    for i in range(2):
        rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])

    # H_\theta(h+dh/2)
    rotor_site_eigenvecs_plus_dh = np.zeros((2, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals_plus_dh = np.zeros((2, ROTOR_DIM))
    H_rotors_plus_dh = get_H_rotors(U, h_p, K, L_MAX)
    for i in range(2):
        rotor_site_eigenvals_plus_dh[i], rotor_site_eigenvecs_plus_dh[i] = np.linalg.eigh(H_rotors_plus_dh[i])
    
    # H_\theta(h-dh/2)
    rotor_site_eigenvecs_minus_dh = np.zeros((2, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals_minus_dh = np.zeros((2, ROTOR_DIM))
    H_rotors_minus_dh = get_H_rotors(U, h_m, K, L_MAX)
    for i in range(2):
        rotor_site_eigenvals_minus_dh[i], rotor_site_eigenvecs_minus_dh[i] = np.linalg.eigh(H_rotors_minus_dh[i])
    
    """
    Calculate nf, L, and other parameters for next cycle
    """
    nf = get_nf(spinon_eigvals, spinon_eigvecs)
    nf_plus_dh = get_nf(spinon_eigvals_p, spinon_eigvecs_p)
    nf_minus_dh = get_nf(spinon_eigvals_m, spinon_eigvecs_m)
    L = get_L(rotor_site_eigenvals, rotor_site_eigenvecs, L_MAX)
    L_plus_dh = get_L(rotor_site_eigenvals_plus_dh, rotor_site_eigenvecs_plus_dh, L_MAX)
    L_minus_dh = get_L(rotor_site_eigenvals_minus_dh, rotor_site_eigenvecs_minus_dh, L_MAX)

    h_prime = np.zeros(2)
    Z_prime = get_Z(rotor_site_eigenvals, rotor_site_eigenvecs, L_MAX)
    K_prime = get_K(t, Z_prime, spinon_eigvals, spinon_eigvecs)

    d_nf = (nf_plus_dh - nf_minus_dh)/dh
    d_L = (L_plus_dh - L_minus_dh)/dh
                
    for i in range(2):
        num = nf[i] - L[i] - N_AT / 2.0
        den = d_nf[i] - d_L[i]

        h_prime[i] = h[i] - num/den

    return h_prime, Z_prime, K_prime, nf