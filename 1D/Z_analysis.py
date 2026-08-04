import numpy as np
import matplotlib.pyplot as plt
import time

from SingleSite.single_site_1D import SingleSite1D

if __name__ == '__main__':
    t, l_max, N_sites = 1.0, 15, 50
    epsilon_vec = np.zeros(N_sites)
    U_vals = np.linspace(0.1, 5.1, 50)

    Z_vals = []
    count = 1
    for U in U_vals:
        chain = SingleSite1D(
            epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
            N_at=2, use_cache=None, iteration_method=0
        )

        Z = chain.get_Z_vec()[0]
        Z_vals.append(Z)
        print(f"Z({U}) = {Z}")

    plt.figure(figsize=(6, 4))

    plt.plot(U_vals, Z_vals, label=r"Single Site", linestyle='-', color='dodgerblue')
    plt.xlabel(r"$U$")
    plt.ylabel(r"$Z$")
    plt.title("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
