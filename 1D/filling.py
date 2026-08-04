import numpy as np
import matplotlib.pyplot as plt
import time

from SingleSite.single_site_1D import SingleSite1D

if __name__ == '__main__':
    U, t, l_max, N_sites = 1.0, 1.0, 15, 1000
    
    interface_width = 10
    site_indices = np.arange(N_sites)
    
    """Single doped impurity w/ Gaussian profile"""
    #epsilon_vec = -0.25*np.exp(-(site_indices - int(N_sites/3.0))**2 / interface_width)
    #epsilon_vec += -0.25*np.exp(-(site_indices - int(2*N_sites/3.0))**2 / interface_width)

    """Junction with hyperbolic tangent profile"""
    epsilon_vec = -0.25 * np.tanh((site_indices - int(N_sites/2.0)) / interface_width) - 0.25
    
    start = time.perf_counter_ns()
    chain = SingleSite1D(
        epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
        N_at=2, use_cache=None, iteration_method=0
    )
    end = time.perf_counter_ns()
    print(f"time: {end-start}s")
    filling = chain.get_filling_vec()

    plt.figure(figsize=(6, 4))

    plt.plot(np.arange(N_sites), filling, label='Filling', marker='o', linestyle='-', color='dodgerblue')

    plt.xlabel(r"Site")
    plt.ylabel(r"$n^f$")
    plt.title("Filling w/ Doping")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()