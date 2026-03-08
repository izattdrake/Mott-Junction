import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 
import os

from SR_Dimer.utility.globals import *
from SR_Dimer.utility.helpers import fermi_dirac

warnings.simplefilter("error", RuntimeWarning)

# Constant L_CUT for all sites. Expect to need higher cuts near the depletion layer
L_MAX = 5

# Dimension of the rotor Hilbert space
# Going to make it variable, increasing towards the junction to reduce the computational complexity
ROTOR_DIM = 1 + 2 * L_MAX 
SPINON_DIM = 4
DIM = 4 * 4 * ROTOR_DIM * ROTOR_DIM
N_AT = 2
T = 300
BETA = 10.0

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
def get_L(rotor_site_eigvals, rotor_site_eigvecs):
    L = np.zeros(2)

    # Sum over lattice sites
    for i in range(2):
        # Pick rotor energies and states for each site
        rotor_eigvals = rotor_site_eigvals[i]

        rotor_eigvals = rotor_eigvals - np.min(rotor_eigvals)
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
                num_sum += boltzmann * (np.abs(rotor_eigenvec[index]) **2 ) * l
        
            den_sum += boltzmann 

        L[i] = num_sum / den_sum

    return L

"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_site_eigvals, rotor_site_eigvecs):
    Z = np.zeros(2)

    for i in range (2):
        # Pick out rotor states and energies at site i
        rotor_eigvals = rotor_site_eigvals[i]
        rotor_eigvals = rotor_eigvals - np.min(rotor_eigvals)
        
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

    np.fill_diagonal(H, mu + h)
    H[0,1], H[1,0] = -t_eff, -t_eff

    return H


"""
Return array of rotor Hamiltonians at each site
"""
def get_H_rotors(U, h, K):
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
def update(t, U, mu, h, Z, K):
    dh = 1e-4
    h_plus_dh = h + dh / 2
    h_minus_dh = h - dh / 2

    t_eff = get_t_eff(t, Z)
    
    """
    Calculate Hamiltonians needed for parameters
    """
    # H_f(h)
    spinon_H = get_H_spinon(t_eff, mu, h)
    spinon_eigenvals, spinon_eigenvecs = np.linalg.eigh(spinon_H)

    # H_f(h+dh)
    spinon_H_plus_dh = get_H_spinon(t_eff, mu, h_plus_dh)
    spinon_eigenvals_plus_dh, spinon_eigenvecs_plus_dh = np.linalg.eigh(spinon_H_plus_dh)

    # H_f(h-dh)
    spinon_H_minus_dh = get_H_spinon(t_eff, mu, h_minus_dh)
    spinon_eigenvals_minus_dh, spinon_eigenvecs_minus_dh = np.linalg.eigh(spinon_H_minus_dh)

    # H_\theta(h)
    rotor_site_eigenvecs = np.zeros((2, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals = np.zeros((2, ROTOR_DIM))
    H_rotors = get_H_rotors(U, h, K)
    for i in range(2):
        rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])

    # H_\theta(h+dh/2)
    rotor_site_eigenvecs_plus_dh = np.zeros((2, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals_plus_dh = np.zeros((2, ROTOR_DIM))
    H_rotors_plus_dh = get_H_rotors(U, h_plus_dh, K)
    for i in range(2):
        rotor_site_eigenvals_plus_dh[i], rotor_site_eigenvecs_plus_dh[i] = np.linalg.eigh(H_rotors_plus_dh[i])
    
    # H_\theta(h-dh/2)
    rotor_site_eigenvecs_minus_dh = np.zeros((2, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals_minus_dh = np.zeros((2, ROTOR_DIM))
    H_rotors_minus_dh = get_H_rotors(U, h_minus_dh, K)
    for i in range(2):
        rotor_site_eigenvals_minus_dh[i], rotor_site_eigenvecs_minus_dh[i] = np.linalg.eigh(H_rotors_minus_dh[i])
    
    """
    Calculate nf, L, and other parameters for next cycle
    """
    nf = get_nf(spinon_eigenvals, spinon_eigenvecs)
    nf_plus_dh = get_nf(spinon_eigenvals_plus_dh, spinon_eigenvecs_plus_dh)
    nf_minus_dh = get_nf(spinon_eigenvals_minus_dh, spinon_eigenvecs_minus_dh)
    L = get_L(rotor_site_eigenvals, rotor_site_eigenvecs)
    L_plus_dh = get_L(rotor_site_eigenvals_plus_dh, rotor_site_eigenvecs_plus_dh)
    L_minus_dh = get_L(rotor_site_eigenvals_minus_dh, rotor_site_eigenvecs_minus_dh)

    h_prime = np.zeros(2)
    Z_prime = get_Z(rotor_site_eigenvals, rotor_site_eigenvecs)
    # print(f"Z' - Z: {Z_prime - Z}")
    K_prime = get_K(t, Z, spinon_eigenvals, spinon_eigenvecs)

    J = np.zeros((2, 2))

    nf_d_diag = (nf_plus_dh - nf_minus_dh)/dh
    L_d_diag = (L_plus_dh - L_minus_dh)/dh
    np.fill_diagonal(J, nf_d_diag - L_d_diag)

    delta = nf - L - 1
    dh_vec = np.linalg.solve(J, -delta)

    h_prime = h + dh_vec
                
    """
    for i in range(SITES):
        num = nf[i] - L[i] - N_AT/2
        den = (nf_plus_dh[i] - nf_minus_dh[i])/dh - (L_plus_dh[i] - L_minus_dh[i])/dh

        '''
        print(f'\n SITE: {i}')
        print(f'Z_prime[{i}]: {Z[i]}')
        print(f'K_prime[{i}]: {K_prime[i]}')
        print(f'nf_plus_dh[{i}]: {nf_plus_dh[i]}')
        print(f'nf_minus_dh[{i}]: {nf_minus_dh[i]}')
        print(f'L[{i}]: {L[0]}')
        print(f'L_plus_dh[{i}]: {nf_plus_dh[i]}')
        print(f'L_minus_dh[{i}]: {L_minus_dh[i]}')
        print(f'Numerator: {num}')
        print(f'Denominator: {den}\n\n')
        '''
        h_prime[i] = h[i] - num/den
    """

    return h_prime, Z_prime, K_prime, nf

if __name__ == '__main__':    
    # Define system parameters
    U = 10.0
    t = 1.0

    # Doping setup
    mu = np.zeros(2)

    h = np.zeros(2)
    Z = np.full(2, 0.3)
    K = np.full(2, -0.1)

    did_converge = False
    
    for i in range(50):
        h_prime, Z, K, nf = update(t, U, mu, h, Z, K)

        if np.all(np.abs(h - h_prime) < 1e-40):
            """
            print(f'h: {h}')
            print(f'h_prime: {h_prime}')
            print(f'Z: {Z}')
            print(f'K: {K}')
            print(f'nf: {nf}')
            """

            h = h_prime
            break
        else:
            h = h_prime

    H_rotors = get_H_rotors(U, h, K)
    rotor_eigvals, rotor_eigvecs = np.linalg.eigh(H_rotors[0])
    print(rotor_eigvecs[0])

    
