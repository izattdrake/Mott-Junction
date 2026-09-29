import numpy as np
import matplotlib.pyplot as plt

from SingleSite.single_site_1D import SingleSite_1D
from ClusterSite2.cluster_site_2_1D import ClusterSite2_1D
from ClusterSite4.cluster_site_4_1D import ClusterSite4_1D

from Utility.plotting import save_axes_centered

if __name__ == '__main__':
    t, l_max, beta = 1.0, 10, 30
    N_sites, N_at, N_target = 100, 2, 100
    PBC=False
    interface_width = 5
    site_indices = np.arange(N_sites)
    
    """Double doped impurity w/ Gaussian profile"""
    #epsilon_vec = -0.25*np.exp(-(site_indices - int(N_sites/3.0))**2 / interface_width)
    #epsilon_vec += -0.25*np.exp(-(site_indices - int(2*N_sites/3.0))**2 / interface_width)

    """Junction with hyperbolic tangent profile"""
    epsilon_vec = -0.25 * np.tanh((site_indices - int(N_sites/2.0)) / interface_width) - 0.25
    
    U = 1.0

    single_chain = SingleSite_1D(epsilon_vec, U, t, l_max, beta, 
                                N_sites, N_at, N_target, PBC=PBC,
                                print_cache_results=True)
    
    cluster2_chain = ClusterSite2_1D(epsilon_vec, U, t, l_max, beta,
                                  N_sites, N_at, N_target, PBC=PBC,
                                  print_cache_results=True)

    cluster4_chain = ClusterSite4_1D(epsilon_vec, U, t, l_max, beta,
                                  N_sites, N_at, N_target, PBC=PBC,
                                  print_cache_results=True)

    single_filling = single_chain.get_filling_vec()
    cluster2_filling = cluster2_chain.get_filling_vec()
    cluster4_filling = cluster4_chain.get_filling_vec()

    site_vals = np.arange(N_sites)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.axhline(y=1.0, linestyle='--', color='black', alpha=0.7)

    ax.plot(np.arange(N_sites), single_filling, label='Single', marker='o', linestyle='-', color='dodgerblue')
    ax.plot(np.arange(N_sites), cluster2_filling, label='Cluster 2', marker='o', linestyle='-', color='tomato')
    ax.plot(np.arange(N_sites), cluster4_filling, label='Cluster 4', marker='o', linestyle='-', color='slategrey')

    ax.set_xlabel(r"Site", fontsize=16, labelpad=20)
    ax.set_ylabel(r"$n^f$", rotation=0, fontsize=16, labelpad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_axes_centered(fig, ax, f"D1/filling_beta{beta}_lmax{l_max}_Nsites_{N_sites}_Nat_{N_at}_Ntarget_{N_target}.pdf")    

    plt.show()