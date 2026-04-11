import numpy as np
import matplotlib.pyplot as plt 

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