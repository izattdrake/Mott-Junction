import numpy as np
import matplotlib.pyplot as plt
import time

from SingleSite.single_site_1D import SingleSite1D

if __name__ == '__main__':
    U, t, l_max, beta, N_sites = 1.0, 1.0, 15, None, 100
    
    interface_width = 5
    site_indices = np.arange(N_sites)
    
    """Double doped impurity w/ Gaussian profile"""
    epsilon_vec = -0.25*np.exp(-(site_indices - int(N_sites/3.0))**2 / interface_width)
    epsilon_vec += -0.25*np.exp(-(site_indices - int(2*N_sites/3.0))**2 / interface_width)

    """Junction with hyperbolic tangent profile"""
    #epsilon_vec = -0.25 * np.tanh((site_indices - int(N_sites/2.0)) / interface_width) - 0.25
    
    U_vals = np.linspace(0.01,5.0,50)

    for U in U_vals:
        chain2 = SingleSite1D(
            epsilon_vec, U, t, l_max, beta=beta, N_sites=N_sites, 
            N_at=2, iteration_method=0, print_cache_results=True
        )
        chain4 = SingleSite1D(
            epsilon_vec, U, t, l_max, beta=beta, N_sites=N_sites, 
            N_at=4, iteration_method=0, print_cache_results=True
        )
        filling = chain2.get_filling_vec()

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(N_sites), filling, label='Filling', marker='o', linestyle='-', color='dodgerblue')
    plt.axhline(y=1.0, linestyle='--', color='black', alpha=0.7)
    plt.xlabel(r"Site", fontsize=16, labelpad=20)
    plt.ylabel(r"$n^f$", rotation=0, fontsize=16, labelpad=20)
    plt.title(r"Filling w/ Doping: $\beta=$" + f"{beta}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()