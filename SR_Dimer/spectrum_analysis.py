import numpy as np
import matplotlib.pyplot as plt

from exact_dimer import get_phys_basis, get_def_particle_basis, get_exact_H
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
Energies is of type list[list,...] where each list inside is the spectrum for a value of U/t
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

"""
    Plots the dimer spectrum across entire Fock space
    Allows for selecting which particle number subspaces to plot 
"""
def plot_def_number_spectrum(epsilon, t, J_vals, PLOT_P0=False, PLOT_P1=False, PLOT_P2=False, PLOT_P3=False, PLOT_P4=False):

    bases = [get_def_particle_basis(0), get_def_particle_basis(1), get_def_particle_basis(2), get_def_particle_basis(3), get_def_particle_basis(4)]
    energies = [[], [], [], [], []]

    for i in range(len(energies)):
        basis = bases[i]
        for J in J_vals:
            H = get_exact_H(epsilon, J, t, basis)

            eigenvals, _ = np.linalg.eigh(H)

            energies[i].append(eigenvals)

    plt.figure()

    shapes = ['D', 'o', 's', '^', 'v']
    colors = ['blue', 'purple', 'green', 'red', 'gold']
    labels = ['N=0', 'N=1', 'N=2', 'N=3', 'N=4']

    datasets = []

    if PLOT_P0:
        datasets.append((energies[0], shapes[0], colors[0], labels[0]))
    if PLOT_P1:
        datasets.append((energies[1], shapes[1], colors[1], labels[1]))
    if PLOT_P2:
        datasets.append((energies[2], shapes[2], colors[2], labels[2]))
    if PLOT_P3:
        datasets.append((energies[3], shapes[3], colors[3], labels[3]))
    if PLOT_P4:
        datasets.append((energies[4], shapes[4], colors[4], labels[4]))

    for E_data, marker, color, label in datasets:
        # Flatten all points for this dataset
        x, y = [], []
        for J, E_list in zip(J_vals, E_data):
            x.extend([J]*len(E_list))
            y.extend(E_list)
        # Plot all points in one call and assign label for legend
        plt.scatter(x, y, marker=marker, edgecolors=color, facecolors='white', linewidths=0.8, label=label, s=60)

    plt.xlabel('U/t')
    plt.ylabel('Energies')
    plt.title('Energy Spectrum vs U/t')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':    
    L_MAX = 13
    # Define system parameters
    U = 10.0; t = 1.0; epsilon=1.0

    J_vals = np.arange(0,20)
    plot_def_number_spectrum(epsilon, t, J_vals, PLOT_P0=True, PLOT_P1=True, PLOT_P2=True, PLOT_P3=True, PLOT_P4=True)
        
    # MF iteration code that does not work
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
