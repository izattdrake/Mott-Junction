import numpy as np
import matplotlib.pyplot as plt

from D1.ClusterSite.cluster_site_1D import ClusterSite1D
from Dimer.Coupled.coupled_dimer import CoupledDimer


if __name__ == "__main__":
    t, l_max, N_sites = 1.0, 10, 2

    epsilon_vec = np.zeros(N_sites)
    epsilon = 0.0

    U_vals = np.linspace(0.1, 5.0, 50)

    Egs_dimer, Egs_cluster = [], []
    
    count = 1

    for U in U_vals:
        coupled_dimer = CoupledDimer(epsilon, U, t, l_max, beta=None,
                                     print_cache_results=True
        )
        
        cluster_chain = ClusterSite1D(
            epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
            N_at=2, iteration_method=0, print_cache_results=True
        )

        Egs_dimer.append(coupled_dimer.get_E_gs())
        Egs_cluster.append(cluster_chain.get_E_gs())

        print(f"Run: {count}"); count += 1
    plt.figure(figsize=(6, 4))

    plt.plot(U_vals, Egs_dimer, label=r"Dimer", marker="o", linestyle="-", color="dodgerblue")
    plt.plot(U_vals, Egs_cluster, label=r"Cluster", marker="o", linestyle="-", color="tomato")

    plt.xlabel(r"$U$", rotation=0, fontsize=16, labelpad=12)
    plt.ylabel(r"$E_{gs}$", rotation=0, fontsize=16, labelpad=24)
    plt.title(r"Ground-State Energy as a Function of $U$ at half-filling: $\beta=\infty$, $N_{\text{sites}}=$" + f"{N_sites}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
