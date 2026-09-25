import numpy as np
import matplotlib.pyplot as plt
import time

from SingleSite.single_site_1D import SingleSite1D
from ClusterSite.cluster_site_1D import ClusterSite1D

if __name__ == '__main__':
    t, l_max, beta = 1.0, 10, None
    N_sites, N_at, N_target = 50, 2, 50
    
    interface_width = 5
    site_indices = np.arange(N_sites)
    
    """Double doped impurity w/ Gaussian profile"""
    #epsilon_vec = -0.25*np.exp(-(site_indices - int(N_sites/3.0))**2 / interface_width)
    #epsilon_vec += -0.25*np.exp(-(site_indices - int(2*N_sites/3.0))**2 / interface_width)

    """Junction with hyperbolic tangent profile"""
    epsilon_vec = -0.25 * np.tanh((site_indices - int(N_sites/2.0)) / interface_width) - 0.25
    
    U = 0.5

    single_chain = SingleSite1D(epsilon_vec, U, t, l_max, beta, 
                                N_sites, N_at, N_target, 
                                print_cache_results=True)
    cluster_chain = ClusterSite1D(epsilon_vec, U, t, l_max, beta,
                                  N_sites, N_at, N_target, 
                                  print_cache_results=True)

    single_filling = single_chain.get_filling_vec()
    cluster_filling = cluster_chain.get_filling_vec()

    site_vals = np.arange(N_sites)

    plt.figure(figsize=(6, 4))
    plt.axhline(y=1.0, linestyle='--', color='black', alpha=0.7)
    plt.grid(True, alpha=0.3)

    plt.plot(np.arange(N_sites), single_filling, label='Single', marker='o', linestyle='-', color='dodgerblue')
    plt.plot(np.arange(N_sites), cluster_filling, label='Cluster', marker='o', linestyle='-', color='tomato')

    plt.xlabel(r"Site", fontsize=16, labelpad=20)
    plt.ylabel(r"$n^f$", rotation=0, fontsize=16, labelpad=20)

    plt.title(r"Filling w/ Doping at $T=0$: $U=$" + f"{U}")

    plt.legend()
    plt.show()