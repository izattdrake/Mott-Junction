import numpy as np
import matplotlib.pyplot as plt

from SingleSite.single_site_1D import SingleSite1D
from ClusterSite.cluster_site_1D import ClusterSite1D

N_sites, N_at, N_target = 20, 2, 20

t, l_max, beta = 1.0, 8, None

epsilon_vec = np.zeros(N_sites)
U_vals = np.linspace(0.01, 3.0, 400)

Egs_single, Egs_cluster = [], []

for U in U_vals:
    single = SingleSite1D(
        epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
        N_at=N_at, N_target=N_target, print_cache_results=True
    )
    cluster = ClusterSite1D(
        epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
        N_at=N_at, N_target=N_target, print_cache_results=True
    )

    Egs_single.append(single.get_energy_MF())
    Egs_cluster.append(cluster.get_energy_MF())

plt.figure(figsize=(6, 4))

plt.plot(U_vals, Egs_single, label=r"Single", marker="o", linestyle="-", color="dodgerblue")
plt.plot(U_vals, Egs_cluster, label=r"Cluster", marker="o", linestyle="-", color="tomato")

plt.xlabel(r"$U$", rotation=0, fontsize=16, labelpad=12)
plt.ylabel(r"$E_{g}$", rotation=0, fontsize=16, labelpad=24)
plt.title(r"Ground-State Energy at Half-Filling: $\beta=\infty$, $N_{\text{sites}}=$" + f"{N_sites}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
