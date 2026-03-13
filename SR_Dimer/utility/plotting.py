import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
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
def plot_rotor_weight(rotor_state, l_max):
    # Construct l values
    l_vals = np.arange(-l_max, l_max + 1)

    # Plot
    plt.figure()
    plt.plot(l_vals, rotor_state, marker='o')
    plt.xlabel("l")
    plt.ylabel("Weight")
    plt.title("Rotor Ground State Weights: $\ell_{max}=$" + f'{l_max}')
    plt.grid(True)
    plt.show()

def plot_rotor_pair_weight(rotor_vector, l_max):
    """
    Plot rotor vector coefficients in the |ℓ₁ℓ₂⟩ two-rotor basis.

    Parameters
    ----------
    rotor_vector : 1D array of length (2*l_max+1)**2
        The coefficients ⟨ℓ₁ℓ₂|ψ⟩, indexed by
        index = (ℓ₁ + l_max) * (2*l_max + 1) + (ℓ₂ + l_max).
    l_max : int
        Maximum angular momentum quantum number.
    """
    l_vals = np.arange(-l_max, l_max + 1)
    n = 2 * l_max + 1

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    dx = dy = 0.6
    x, y, dz = [], [], []

    for i, l1 in enumerate(l_vals):
        for j, l2 in enumerate(l_vals):
            idx = i * n + j
            x.append(l1 - dx / 2)
            y.append(l2 - dy / 2)
            dz.append(rotor_vector[idx])

    x, y, dz = np.array(x), np.array(y), np.array(dz)
    z = np.zeros_like(dz)

    colors = plt.cm.viridis((dz - dz.min()) / (dz.max() - dz.min() + 1e-12))

    ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True, alpha=0.85)

    ax.set_xlabel(r'$\ell_1$', fontsize=13)
    ax.set_ylabel(r'$\ell_2$', fontsize=13)
    ax.set_zlabel('Weight', fontsize=13)
    ax.set_title(
        r'Rotor Weights: $|\ell_1 \ell_2\rangle$ basis, $\ell_{\max}=$' + f'{l_max}',
        fontsize=13
    )
    ax.set_xticks(l_vals)
    ax.set_yticks(l_vals)

    plt.tight_layout()
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
def plot_def_number_spectrum(epsilon, t, J_vals, PLOT_P0=False, PLOT_P1=False, PLOT_P2=False, PLOT_P3=False, PLOT_P4=False, PLOT_EFF=False, EFF_ENERGIES=None):

    bases = [get_def_particle_basis(0), get_def_particle_basis(1), get_def_particle_basis(2), get_def_particle_basis(3), get_def_particle_basis(4)]
    energies = [[], [], [], [], []]

    for i in range(len(energies)):
        basis = bases[i]
        for J in J_vals:
            H = get_H(epsilon, J, t, basis)

            eigenvals, _ = np.linalg.eigh(H)

            energies[i].append(eigenvals)

    plt.figure()

    shapes = ['D', 'o', 's', '^', 'v', 'p']
    colors = ['blue', 'purple', 'green', 'red', 'gold', 'black']
    labels = ['N=0', 'N=1', 'N=2', 'N=3', 'N=4', 'EFF']

    datasets = []
    print(f'energies[0]: {energies[0]}')
    print(f'length: {len(energies[0])}')
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
    if PLOT_EFF:
        datasets.append((EFF_ENERGIES, shapes[5], colors[5], labels[5]))

    for E_data, marker, color, label in datasets:
        x, y = [], []
        for J, E_list in zip(J_vals, E_data):
            x.extend([J]*len(E_list))
            y.extend(E_list)
        plt.scatter(x, y, marker=marker, edgecolors=color, facecolors='white', linewidths=0.8, label=label, s=60)

    plt.xlabel('U/t')
    plt.ylabel('Energies')
    plt.title('Energy Spectrum vs U/t')
    plt.legend()
    plt.grid(True)
    plt.show()


