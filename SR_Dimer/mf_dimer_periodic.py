import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 
import os

from utility.helpers import *

warnings.simplefilter("error", RuntimeWarning)

N_AT = 2

def get_nf(spinon_eigvals, spinon_eigvecs, beta):
    nf = np.zeros(2)

    for i in range(2):
        sum = 0.0
        for j in range(2):
            energy = spinon_eigvals[j]
            sum += (np.abs(spinon_eigvecs[i,j]) ** 2) * fermi_dirac(energy, beta)

        nf[i] = N_AT * sum

    return nf

"""
    Return derivative of nf with respect to h
"""
def get_d_nf(epsilon, h, dh, t_eff, beta):
    h_p, h_m = h + dh/2, h - dh/2

    eigvals_p, eigvecs_p = np.linalg.eigh(get_H_spinon(epsilon, t_eff, mu, h_p))
    eigvals_m, eigvecs_m = np.linalg.eigh(get_H_spinon(epsilon, t_eff, mu, h_m))

    nf_p, nf_m = get_nf(eigvals_p, eigvecs_p, beta), get_nf(eigvals_m, eigvecs_m, beta)
    d_nf = (nf_p - nf_m) / dh

    return d_nf


"""
    Return derivative of L with respect to h
"""
def get_d_L(U, h, dh, K, l_max):
    dim = 2 * l_max + 1
    h_p, h_m = h + dh/2, h - dh/2

    # H_\theta(h+dh/2)
    site_eigenvecs_p, site_eigenvals_p = np.zeros((2, dim, dim), dtype=complex), np.zeros((2, dim))
    H_rotors_plus_dh = get_H_rotors(U, h_p, K, l_max)
    for i in range(2):
        site_eigenvals_p[i], site_eigenvecs_p[i] = np.linalg.eigh(H_rotors_plus_dh[i])
    
    # H_\theta(h-dh/2)
    site_eigenvecs_m, site_eigenvals_m = np.zeros((2, dim, dim), dtype=complex), np.zeros((2, dim))
    H_rotors_minus_dh = get_H_rotors(U, h_m, K, l_max)
    for i in range(2):
        site_eigenvals_m[i], site_eigenvecs_m[i] = np.linalg.eigh(H_rotors_minus_dh[i])

    L_plus, L_minus = get_L(site_eigenvals_p, site_eigenvecs_p, beta, l_max), get_L(site_eigenvals_m, site_eigenvecs_m, beta, l_max)
    d_L = (L_plus - L_minus)/dh

    return d_L


"""
Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
"""
def get_L(rotor_site_eigvals, rotor_site_eigvecs, beta, l_max):
    ROTOR_DIM = 2 * l_max + 1
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

            boltzmann = np.exp(-beta * rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range(ROTOR_DIM):
                l = index - l_max
                num_sum += boltzmann * (np.abs(rotor_eigenvec[index]) ** 2) * l
        
            den_sum += boltzmann 

        L[i] = num_sum / den_sum

    return L

"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_site_eigvals, rotor_site_eigvecs, beta, l_max):
    dim = (2 * l_max + 1)
    Z = np.zeros(2)

    for i in range (2):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        rotor_eigvals = rotor_site_eigvals[i] - rotor_site_eigvals[i,0]        
        rotor_eigvecs = rotor_site_eigvecs[i]

        num_sum = 0.0
        den_sum = 0.0
        
        # Sum over rotor states and energies. There are ROTOR_DIM total eigenstates
        for j in range (dim):
            rotor_state = rotor_eigvecs[:,j]
            rotor_energy = rotor_eigvals[j]

            boltzmann = np.exp(-beta * rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range (dim):
                l = index - l_max 
                # Handle truncating the Hilbert space by simply ignoring the L_CUT + 1 state
                if index == 0:
                    num_sum += boltzmann * np.conj(rotor_state[index])*rotor_state[index + 1]
                elif index == dim - 1:
                    num_sum += boltzmann * np.conj(rotor_state[index])*rotor_state[index - 1]
                else:
                    num_sum += boltzmann * np.conj(rotor_state[index])*(rotor_state[index + 1] + rotor_state[index - 1])
                    
            # Partition function, make function and calculate once?
            den_sum += boltzmann

        num_sum, den_sum = np.real(num_sum), np.real(den_sum)
        Z[i] = (num_sum/(2 * den_sum))**2

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
def get_H_spinon(epsilon, t_eff, mu, h):
    H = np.zeros((2, 2))

    np.fill_diagonal(H, epsilon - mu - h)
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
def get_K(t, Z, spinon_eigvals, spinon_eigvecs, beta):
    K = np.zeros(2)

    sums = [0.0, 0.0]  

    for i in range(2):
        energy = spinon_eigvals[i]
        sums[0] += t * np.sqrt(Z[1]) * spinon_eigvecs.conj().T[i,0] * spinon_eigvecs[1,i] * fermi_dirac(energy, beta)
        sums[1] += t * np.sqrt(Z[0]) * spinon_eigvecs.conj().T[i,1] * spinon_eigvecs[0,i] * fermi_dirac(energy, beta)
    
    K[0], K[1] = -2 * N_AT * sums[0], -2 * N_AT * sums[1]

    return K

"""
    Returns the next iteration of the Lagrange parameters using Newton's method
    The minimizing function is f = <n_f(i)> - N/2 - <L(i)>
    Newton's method sets h' = h - f/f'
    Derivatives are calculted with a finite difference method
    TODO: Find a closed form analytic solution using perturbation theory
    NOTE: Analytic solutions may not be possible due to recursion
"""
def update_params(epsilon, t, U, mu, h_guess, Z_guess, K_guess, beta, l_max):
    dh = 1e-3
    ALPHA = 0.1
    ROTOR_DIM = 2 * l_max + 1

    h_update, Z_update, K_update = h_guess, Z_guess, K_guess
    t_eff_update = get_t_eff(t, Z_update)

    did_converge = False
    while not did_converge:
        h, Z, K, t_eff = h_update, Z_update, K_update, t_eff_update

        # Used for calculating nf and L derivatives with finite difference methods 
        h_p, h_m = h + dh/2, h - dh/2

        spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon, t_eff, mu, h))

        # H_\theta(h)
        rotor_site_eigenvecs, rotor_site_eigenvals = np.zeros((2, ROTOR_DIM, ROTOR_DIM)), np.zeros((2, ROTOR_DIM))
        H_rotors = get_H_rotors(U, h, K, l_max)
        for i in range(2):
            rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])
        
        """
        Calculate nf, L, and other parameters for next cycle
        """
        nf = get_nf(spinon_eigvals, spinon_eigvecs, beta)
        L = get_L(rotor_site_eigenvals, rotor_site_eigenvecs, beta, l_max)
        
        d_nf = get_d_nf(epsilon, h, dh, t_eff, beta)
        d_L = get_d_L(U, h, dh, K, l_max)

        h_update = np.zeros(2)

        for i in range(2):
            h_update[i] = h[i] - ALPHA*(nf[i] - L[i] - 1.0)/(d_nf[i] - d_L[i])

        Z_update = get_Z(rotor_site_eigenvals, rotor_site_eigenvecs, beta, l_max)
        K_update = get_K(t, Z_update, spinon_eigvals, spinon_eigvecs, beta)
        t_eff_update = get_t_eff(t, Z_update)

        print(f'h_update: {h_update}')
        print(f'h: {h}')
        print(f't_eff_update: {t_eff_update}')
        print(f't_eff: {t_eff}')
        print(f'Z_update: {Z_update}')
        print(f'Z: {Z}')
        print(f'K_update: {K_update}')
        print(f'K: {K}')

        if (np.abs(h_update - h) < 1e-6).all() and (np.abs(t_eff_update - t_eff) < 1e-6) and (np.abs(Z_update - Z) < 1e-6).all() and (np.abs(K_update - K) < 1e-6).all():
            did_converge = True

    return h_update, Z_update, K_update, t_eff, nf, L

if __name__ == '__main__':
    l_max = 5
    rotor_dim = (2 * l_max + 1)**2

    epsilon = 0.5
    t = 1.0
    U = 7.0
    beta = 5.0
    particle_num = 1.0

    J_vals = [3,4,5,6,7,8]
    spinon_energies = []

    mu = np.full(2, 0.0)
    h_guess = np.full(2, -epsilon)
    Z_guess = np.full(2, 0.1)
    K_guess = np.full(2,-0.2)

    h, Z, K, t_eff, nf, L = update_params(epsilon, t, U, mu, h_guess, Z_guess, K_guess, beta, l_max)
    print(f'h: {h}')
    print(f'Z: {Z}')
    print(f'K: {K}')
    print(f't_eff: {t_eff}')
    print(f'nf: {nf}')
    print(f'L: {L}')

