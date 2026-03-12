import numpy as np
import matplotlib.pyplot as plt

from exact_dimer import get_phys_basis, get_def_particle_basis, get_H
from mf_dimer_numerical import *
from utility.helpers import *

"""
This is the main file used for processing of exact, effective, and MF dimer solutions, calculating their spectra, and plotting data
Logic behind exact and MF theories in the other files
"""


"""
    Function that plots rotor ground states
    Used to study the weights around physical values of |l>
    and the dropoff towards l_max 
"""
def plot_rotor_gs_weight(rotor_gs, l_max):
    # Construct l values
    l_vals = np.arange(-l_max, l_max + 1)

    # Plot
    plt.figure()
    plt.plot(l_vals, rotor_gs, marker='o')
    plt.xlabel("l")
    plt.ylabel("Weight")
    plt.title("Rotor Ground State Weights: $\ell_{max}=$" + f'{l_max}')
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
            H = get_H(epsilon, J, t, basis)

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


