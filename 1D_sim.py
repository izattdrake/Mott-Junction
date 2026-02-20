import numpy as np
import matplotlib.pyplot as plt

from globals import *
from helpers import *

SITES = 2 + 1
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

"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_site_eigenvals, rotor_site_eigenvecs):
    Z = np.zeros(SITES)
    
    for i in range (SITES):
        # Pick out rotor states and energies at site i
        rotor_eigenvals = rotor_site_eigenvals[i]
        rotor_eigenvecs = rotor_site_eigenvecs[i]
        
        num_sum = 0.0
        den_sum = 0.0

        # Sum over rotor states and energies. There are ROTOR_DIM total eigenstates
        for j in range (ROTOR_DIM):
            rotor_energy = rotor_eigenvals[j]
            rotor_state = rotor_eigenvecs[:,j]

            boltzmann_factor = np.exp(-BETA*rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range (ROTOR_DIM):
                l = ROTOR_DIM - L_CUT 
                # Handle truncating the Hilbert space by simply ignoring the L_CUT + 1 state
                if index == 0:
                    num_sum += boltzmann_factor * np.conj(rotor_state[index])*rotor_state[index + 1]
                elif index == ROTOR_DIM - 1:
                    num_sum += boltzmann_factor * np.conj(rotor_state[index])*rotor_state[index - 1]
                else:
                    num_sum += boltzmann_factor * np.conj(rotor_state[index])*(rotor_state[index + 1] + rotor_state[index - 1])
        
            # Partition function, make function and calculate once?
            den_sum += boltzmann_factor

        Z[i] = (num_sum/(2*den_sum))**2

    return Z

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
            rotor_eigenvec = rotor_eigenvec[:,j]

            boltzmann_factor = np.exp(-BETA*rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range(ROTOR_DIM):
                l = index - L_CUT
                num_sum += boltzmann_factor*(np.abs(rotor_eigenvec[index])**2)*l
        
            den_sum += boltzmann_factor 

        L[i] = num_sum/den_sum

    return L

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
    
    H_sites = np.zeros((SITES, ROTOR_DIM, ROTOR_DIM), dtype=complex)

    for i in range(SITES):
        H = np.zeros((ROTOR_DIM, ROTOR_DIM), dtype=complex)

        for index in range(ROTOR_DIM):
            l = index - L_CUT
            H[index, index] = (U / 2) * l * l + h[i] * l

            if index < ROTOR_DIM - 1:
                H[index, index + 1] = K[i] / 2
                H[index + 1, index]

        H_sites[i] = H

    return H_sites

"""
Return 
"""
def get_K(t_eff, Z, spinon_eigenvals, spinon_eigenvecs):
    K = np.zeros(SITES, dtype=complex)

    for i in range(SITES):
        sum_i = 0.0

        """
        Actual calculation ahs a sum over \sigma, spin/orbital, but I'm assuming
        energies are independent of spin/orbital such that we can just multiply 
        by N_at at the end
        """
        # Actual calculation has a sum over spin/orbital. Assuming energies are
        for j in range(SITES):
            for k in range(SITES):
                energy = spinon_eigenvals[k]
                sum += t_eff[i,j] * spinon_eigenvecs.adj()[k,i] * spinon_eigenvecs[j,k] * fermi_dirac(energy)
        
        K[i] = -(2 * N_AT / np.sqrt(Z[i])) * sum

        return K
    
def get_nf(spinon_eigenvals, spinon_eigenvecs):
    nf = np.zeros(SITES)

    for i in range(SITES):
        sum = 0.0
        
        for j in range(SITES):
            energy = spinon_eigenvals[j]
            sum += (np.abs(spinon_eigenvecs[i,j])**2) * fermi_dirac(energy)

        nf[i] = sum

    return nf

def main():
    # Hamiltonian parameters (hopping should be a matrix, can still make t_{ij} = t_{i-j} since hopping doesn't depend on doping. Make t_{00} = 0)
    U = 10.0
    N = 2
    mu_L = -0.5
    mu_R = 0.5

    t0 = 1.0
    t = np.zeros(SITES)
    t[1] = t0; t[2] = 0/10.0

    # Doping setup
    mu = np.zeros(SITES)
    mu[:HALF_SITES] = mu_L
    mu[HALF_SITES:] = mu_R 

    h = np.zeros(SITES)
    Z = np.full(SITES, 0.1)
    K = np.zeros(SITES, dtype=complex)

    t_eff = get_t_eff(t, Z)

    H_spinon = get_H_spinon(t_eff, mu, h)
    spinon_eigenvals, spinon_eigenvecs = np.linalg.eigh(H_spinon)

    rotor_site_eigenvecs = np.zeros((SITES, ROTOR_DIM, ROTOR_DIM), dtype=complex)
    rotor_site_eigenvals = np.zeros((SITES, ROTOR_DIM))

    H_rotors = get_H_rotors(U, h, K)
    for i in range(SITES):
        rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])

    print(rotor_site_eigenvals)

if __name__ == '__main__':
    main()