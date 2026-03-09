import numpy as np
import matplotlib.pyplot as plt

from exact_dimer import get_phys_basis, get_exact_H
from mf_dimer_numerical import *
from utility.helpers import *

"""
This is the main file used for processing of exact, effective, and MF dimer solutions, calculating their spectra, and plotting data
Logic behind exact and MF theories in the other files
"""



def plot_rotor_gs_weight(rotor_gs, L_MAX):
    # Construct l values
    l_vals = np.arange(-L_MAX, L_MAX + 1)

    # Plot
    plt.figure()
    plt.plot(l_vals, rotor_gs, marker='o')
    plt.xlabel("l")
    plt.ylabel("Weight")
    plt.title("Rotor Ground State Weights: $\ell_{max}=$" + f'{L_MAX}')
    plt.grid(True)
    plt.show()

"""
Plots energies as a function of J=U/t
Energies is of type list[list,...] where each list inside is a spectrum
"""
def plot_spectrum(energies, J_vals):
    plt.figure()

    for J, E_list in zip(J_vals, energies):
        plt.scatter([J]*len(E_list), E_list, 
                    marker='D',   
                    edgecolors='blue',  
                    facecolors='white',  
                    linewidths=1.0       
                    )

    plt.xlabel("U/t")
    plt.ylabel("Energies")
    plt.title("Energy Spectrum vs U/t")
    plt.grid(True)

    plt.show()

if __name__ == '__main__':    
    # Define system parameters
    U = 10.0
    t = 1.0

    phys_basis = get_phys_basis()
    J_vals = np.arange(0,20)
    phys_Hs = []
    energies = []
    for U in J_vals:
        phys_H = get_exact_H(0, U, t, phys_basis)
        eigenvals, eigenvecs = np.linalg.eigh(phys_H)
        
        # Set to zero
        eigenvals = eigenvals
        energies.append(eigenvals)
        print(eigenvals)

    energies = np.array(energies)
    plot_spectrum(energies, J_vals)
        
    """
    MF iteration code 
    """
    L_MAX = 13
    for L_MAX in range(5, 30):
        # Doping setup
        mu = np.zeros(2)

        h = np.zeros(2)
        Z = np.full(2, 0.2)
        K = np.full(2, -0.3)

        did_converge = False
        for i in range(50):
            h_prime, Z_prime, K_prime, nf = update(t, U, mu, h, Z, K, L_MAX)

            if np.all(np.abs(h - h_prime) < 1e-20) and np.all(np.abs(Z - Z_prime) < 1e-20) and np.all(np.abs(K - K_prime) < 1e-20):
                h, Z, K = h_prime, Z_prime, K_prime
                did_converge = True
                break
            else:
                print('update')
                h, Z, K = h_prime, Z_prime, K_prime

        if did_converge:
            H_rotors = get_H_rotors(U, h, K, L_MAX)
            rotor_eigvals, rotor_eigvecs = np.linalg.eigh(H_rotors[0])
            rotor_gs = rotor_eigvecs[:,0]
            print(Z)
            print(f'K0 = {K[0]}')
            print(f'h0 = {h[0]}')
            print(f'nf0 = {nf[0]}')

            print(check_Z(U, K[0], h[0], nf[0]))
        else:
            print('did not converge')