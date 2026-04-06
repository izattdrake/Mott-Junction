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
def plot_electron_occupancy(nf, title_string=None):
    nf = np.array(nf)
    N_sites = len(nf)

    sites = np.arange(N_sites)

    # Junction location (halfway)
    junction = (N_sites - 1) / 2

    plt.figure(figsize=(8, 5))

    # Plot occupancy
    plt.plot(sites, nf, marker='o', linestyle='-', label='Occupancy')

    # Draw vertical line at junction
    plt.axvline(x=junction, color='red', linestyle='--', label='Junction')

    # Labels and formatting
    plt.xlabel('Lattice Site')
    plt.ylabel('Electron Occupancy')
    plt.xticks(sites[::10])  # label all sites

    if title_string:
        plt.title(title_string)
    else:
        plt.title('Electron Occupancy Across Junction')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()