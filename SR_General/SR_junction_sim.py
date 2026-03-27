import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 

from utility.helpers import *

warnings.simplefilter("error", RuntimeWarning)

SITES = 100
HALF_SITES = int(SITES/2.0 - 0.5)

# Constant L_CUT for all sites. Expect to need higher cuts near the depletion layer
L_CUT = 30

# Dimension of the rotor Hilbert space
# Going to make it variable, increasing towards the junction to reduce the computational complexity
ROTOR_DIM = 1 + 2 * L_CUT 
N_AT = 2
T = 300
BETA = 10.0
PBC = True
FINISH = False 

def get_nf(spinon_eigenvals, spinon_eigenvecs):
    nf = np.zeros(SITES)

    for i in range(SITES):
        sum = 0.0

        for j in range(SITES):
            energy = spinon_eigenvals[j]
            sum += (np.abs(spinon_eigenvecs[i,j])**2) * fermi_dirac(energy, BETA)

        nf[i] = N_AT * sum

    return nf

"""
Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
"""
def get_L(rotor_site_eigenvals, rotor_site_eigenvecs):
    L = np.zeros(SITES)

    # Sum over lattice sites
    for i in range(SITES):
        # Pick rotor energies and states for each site
        rotor_eigenvals = rotor_site_eigenvals[i]
        min_eigenval = np.min(rotor_eigenvals)
        rotor_eigenvals = rotor_eigenvals - np.full(ROTOR_DIM, min_eigenval)

        rotor_eigenvecs = rotor_site_eigenvecs[i]

        num_sum = 0.0
        den_sum = 0.0

        # Sum over rotor energies and states
        for j in range(ROTOR_DIM):
            rotor_energy = rotor_eigenvals[j]

            rotor_eigenvec = rotor_eigenvecs[:,j]

            boltzmann_factor = np.exp(-BETA*rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range(ROTOR_DIM):
                l = index - L_CUT
                num_sum += boltzmann_factor*(np.abs(rotor_eigenvec[index])**2)*l
        
            den_sum += boltzmann_factor 

        L[i] = num_sum/den_sum

    return L

"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_site_eigenvals, rotor_site_eigenvecs):
    Z = np.zeros(SITES)

    for i in range (SITES):
        # Pick out rotor states and energies at site i
        rotor_eigenvals = rotor_site_eigenvals[i]
        min_eigenval = np.min(rotor_eigenvals)
        rotor_eigenvals = rotor_eigenvals - np.full(ROTOR_DIM, min_eigenval)

        rotor_eigenvecs = rotor_site_eigenvecs[i]

        num_sum = 0.0
        den_sum = 0.0
        
        # Sum over rotor states and energies. There are ROTOR_DIM total eigenstates
        for j in range (ROTOR_DIM):
            rotor_state = rotor_eigenvecs[:,j]
            rotor_energy = rotor_eigenvals[j]

            boltzmann_factor = np.exp(-BETA*rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range (ROTOR_DIM):
                l = index - L_CUT 
                # Handle truncating the Hilbert space by simply ignoring the L_CUT + 1 state
                if index == 0:
                    num_sum += boltzmann_factor * np.conj(rotor_state[index])*rotor_state[index + 1]
                elif index == ROTOR_DIM - 1:
                    num_sum += boltzmann_factor * np.conj(rotor_state[index])*rotor_state[index - 1]
                else:
                    num_sum += boltzmann_factor * np.conj(rotor_state[index])*(rotor_state[index + 1] + rotor_state[index - 1])
            # Partition function, make function and calculate once?
            den_sum += boltzmann_factor

        num_sum = np.real(num_sum)
        den_sum = np.real(den_sum)
        Z[i] = (num_sum/(2*den_sum))**2

    return Z

"""
Return effective tunneling amplitude matrix
"""
def get_t_eff(t, Z):
    t_eff = np.zeros((SITES,SITES))

    for i in range (SITES):
        for j in range(SITES):
            t_eff[i,j] = t[np.abs(i-j)]*np.sqrt(Z[i] * Z[j])

    return t_eff

"""
Return real-space spinon Hamiltonian
"""
def get_H_spinon(t_eff, mu, h):
    H = np.zeros((SITES, SITES))

    diag = -(mu + h)
    np.fill_diagonal(H, diag)
    H += -t_eff

    return H

"""
Return array of rotor Hamiltonians at each site
"""
def get_H_rotors(U, h, K):
    H_sites = np.zeros((SITES, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    l_vals = np.arange(-L_CUT, L_CUT + 1)

    for i in range(SITES):
        H = np.zeros((ROTOR_DIM, ROTOR_DIM), dtype=complex)

        diag = ((U / 2.0) * l_vals + h[i]) * l_vals
        off_diag = K[i] / 2.0

        np.fill_diagonal(H, diag)
        H += np.diag(np.full(ROTOR_DIM - 1, off_diag), k=1)
        H += np.diag(np.full(ROTOR_DIM - 1, off_diag), k=-1)

        H_sites[i] = H

    return H_sites

"""
Return K
"""
def get_K(t, Z, spinon_eigenvals, spinon_eigenvecs):
    K = np.zeros(SITES, dtype=complex)

    for i in range(SITES):
        sum = 0.0

        for j in range(SITES):
            for k in range(SITES):
                energy = spinon_eigenvals[k]
                sum += t[np.abs(i-j)] * np.sqrt(Z[j]) * spinon_eigenvecs.conj().T[k,i] * spinon_eigenvecs[j,k] * fermi_dirac(energy, BETA)
        
        K[i] = -(2 * N_AT) * sum

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
    h_plus_dh = h + np.full(SITES, dh/2)
    h_minus_dh = h - np.full(SITES, dh/2)

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
    rotor_site_eigenvecs = np.zeros((SITES, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals = np.zeros((SITES, ROTOR_DIM))
    H_rotors = get_H_rotors(U, h, K)
    for i in range(SITES):
        rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])

    # H_\theta(h+dh/2)
    rotor_site_eigenvecs_plus_dh = np.zeros((SITES, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals_plus_dh = np.zeros((SITES, ROTOR_DIM))
    H_rotors_plus_dh = get_H_rotors(U, h_plus_dh, K)
    for i in range(SITES):
        rotor_site_eigenvals_plus_dh[i], rotor_site_eigenvecs_plus_dh[i] = np.linalg.eigh(H_rotors_plus_dh[i])
    
    # H_\theta(h-dh/2)
    rotor_site_eigenvecs_minus_dh = np.zeros((SITES, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals_minus_dh = np.zeros((SITES, ROTOR_DIM))
    H_rotors_minus_dh = get_H_rotors(U, h_minus_dh, K)
    for i in range(SITES):
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

    h_prime = np.zeros(SITES)
    Z_prime = get_Z(rotor_site_eigenvals, rotor_site_eigenvecs)
    # print(f"Z' - Z: {Z_prime - Z}")
    K_prime = get_K(t, Z, spinon_eigenvals, spinon_eigenvecs)

    J = np.zeros((SITES, SITES))

    nf_d_diag = (nf_plus_dh - nf_minus_dh)/dh
    L_d_diag = (L_plus_dh - L_minus_dh)/dh
    np.fill_diagonal(J, nf_d_diag - L_d_diag)

    for i in range(SITES - 2):
        J[i + 1,i] = -0.05 * J[i,i]
        J[i, i+1] = -0.05 * J[i,i]

    delta = nf - L - np.full(SITES,N_AT/2)
    dh_vec = np.linalg.solve(J, -delta)
    h_prime = h + dh_vec

    return h_prime, Z_prime, K_prime, nf