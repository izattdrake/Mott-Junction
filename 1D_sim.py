import numpy as np
import matplotlib.pyplot as plt
import time

from globals import *
from helpers import *

SITES = 5
HALF_SITES = int(SITES/2.0 - 0.5)

# Constant L_CUT for all sites. Expect to need higher cuts near the depletion layer
L_CUT = 2

# Dimension of the rotor Hilbert space
# Going to make it variable, increasing towards the junction to reduce the computational complexity
ROTOR_DIM = 1 + 2 * L_CUT 
N_AT = 2
T = 300
BETA = KB_EV * T
PBC = True
FINISH = False 

def get_nf(spinon_eigenvals, spinon_eigenvecs):
    nf = np.zeros(SITES)

    for i in range(SITES):
        sum = 0.0

        for j in range(SITES):
            energy = spinon_eigenvals[j]
            sum += (np.abs(spinon_eigenvecs[i,j])**2) * fermi_dirac(energy, T)

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
    Z = np.zeros(SITES, dtype=complex)
    
    for i in range (SITES):
        # Pick out rotor states and energies at site i
        rotor_eigenvals = rotor_site_eigenvals[i]
        rotor_eigenvecs = rotor_site_eigenvecs[i]
        
        num_sum = 0.0
        den_sum = 0.0

        for m in range(ROTOR_DIM):
            print(f'Eigenvector {m}: {rotor_eigenvecs[:,m]}')
            
        # Sum over rotor states and energies. There are ROTOR_DIM total eigenstates
        for j in range (ROTOR_DIM):
            rotor_energy = rotor_eigenvals[j]
            rotor_state = rotor_eigenvecs[:,j]
            #print(f'State {j}: {rotor_state}\n')
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
        
        print(f'Numerator {i}: {num_sum}')
        print(f'Denominator {i}: {den_sum}')
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

    for i in range(SITES):
        for j in range(SITES):
            if i == j:
                H[i,j] = -(mu[i] + h[i])
            else:
                H[i,j] = t_eff[i,j]

    return H

"""
Return array of rotor Hamiltonians at each site
"""
def get_H_rotors(U, h, K):
    
    H_sites = np.zeros((SITES, ROTOR_DIM, ROTOR_DIM))

    for i in range(SITES):
        H = np.zeros((ROTOR_DIM, ROTOR_DIM))

        for index in range(ROTOR_DIM):
            l = index - L_CUT
            H[index, index] = (U / 2) * l * l + h[i] * l

            if index < ROTOR_DIM - 1:
                off_diag = K[i] / 2
                H[index, index + 1] = off_diag 
                H[index + 1, index] = off_diag 

        H_sites[i] = H

    return H_sites

"""
Return K
"""
def get_K(t_eff, Z, spinon_eigenvals, spinon_eigenvecs):
    K = np.zeros(SITES, dtype=complex)

    for i in range(SITES):
        sum = 0.0

        """
        Actual calculation as a sum over \sigma, spin/orbital, but I'm assuming
        energies are independent of spin/orbital such that we can just multiply 
        by N_at at the end
        """
        # Actual calculation has a sum over spin/orbital. Assuming energies are
        for j in range(SITES):
            for k in range(SITES):
                energy = spinon_eigenvals[k]
                sum += t_eff[i,j] * spinon_eigenvecs.conj().T[k,i] * spinon_eigenvecs[j,k] * fermi_dirac(energy, T)
        
        K[i] = -(2 * N_AT / np.sqrt(Z[i])) * sum

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
    dh = 0.01
    h_plus_dh = h + np.full(SITES, dh/2)
    h_minus_dh = h - np.full(SITES, dh/2)

    t_eff = get_t_eff(t, Z)
    print(f't_eff: {t_eff}')
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
    print(f'Rotor Hamiltonian: \n{H_rotors[1]}')
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

    Z_prime = get_Z(rotor_site_eigenvals, rotor_site_eigenvecs)
    K_prime = get_K(t_eff, Z, spinon_eigenvals, spinon_eigenvecs)
    h_prime = np.zeros(SITES)

    for i in range(SITES):
        num = nf[i] - L[i] - N_AT/2
        den = (nf_plus_dh[i] - nf_minus_dh[i])/dh - (L_plus_dh[i] - L_minus_dh[i])/dh
    
        h_prime[i] = h[i] - num/den
        
    return h_prime, Z_prime, K_prime, nf

if __name__ == '__main__':
    h = np.zeros(SITES)
    print(h + np.full(SITES, 0.1))
    
    # Define Hamiltonian parameters
    U = 10.0
    N = 2
    mu_L = -0.5
    mu_R = 0.5

    t0 = 1.0
    t = np.zeros(SITES)
    t[1] = t0

    # Doping setup
    mu = np.zeros(SITES)
    mu[:HALF_SITES] = mu_L; mu[HALF_SITES:] = mu_R 

    h = np.full(SITES, 0.1)
    Z = np.full(SITES, 0.1)
    K = np.full(SITES, 0.1)

    print('INITIAL PARAMS')
    print(f'h: {h}')
    print(f'Z: {Z}')
    print(f'K: {K}')
    while True:
        h_prime, Z, K, nf = update(t, U, mu, h, Z, K)
        if np.all(h - h_prime < 1e-4):
            h = h_prime
            print('DONE')
            break
        else:
            h = h_prime
    print('AFTER PARAMS')
    print(f'h: {h}')
    print(f'Z: {Z}')
    print(f'K: {K}')