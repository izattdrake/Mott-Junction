import numpy as np
import matplotlib.pyplot as plt 
from exact_dimer import get_phys_basis, get_def_particle_basis, get_H_exact

"""
Plot electron number occupancy across a 1D lattice junction.

Parameters
----------
occupancy : list or array-like
    Electron number occupancy at each lattice site.
    Length defines N_sites.
"""
def plot_electron_occupancy(nf, title_string=None, plot_junction=False) -> None:
    nf = np.array(nf)
    N_sites = len(nf)

    sites = np.arange(N_sites)

    # Junction location (halfway)
    junction = N_sites / 2

    plt.figure(figsize=(8, 5))

    plt.plot(sites, nf, marker='o', linestyle='-', label='Occupancy')

    if plot_junction:
        plt.axvline(x=junction, color='red', linestyle='--', label='Junction')

    plt.xlabel('Lattice Site')
    plt.ylabel('Electron Occupancy')
    plt.xticks(sites[::10])

    if title_string:
        plt.title(title_string)
    else:
        plt.title('Electron Occupancy Across Junction')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
    Function that plots rotor ground states
    Used to study the weights around physical values of |l>
    and the dropoff towards l_max 
"""
def plot_rotor_weight(rotor_state, state_num, epsilon, site, l_max):
    # Construct l values
    l_vals = np.arange(-l_max, l_max + 1)

    # Plot
    plt.figure()
    plt.plot(l_vals, rotor_state, marker='o')
    plt.xlabel("l")
    plt.ylabel("Weight")
    plt.title(f'Weight of Rotor State {state_num}: site {site}, ' + r'$\epsilon_0 =$ ' + f'{epsilon}, ' + r'$\ell_{max}=$' + f'{l_max}')
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
            H = get_H_exact(epsilon, J, t, basis)

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


