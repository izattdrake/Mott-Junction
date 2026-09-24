import numpy as np
import matplotlib.pyplot as plt

from SingleSite.single_site_1D import SingleSite1D
from ClusterSite.cluster_site_1D import ClusterSite1D

N_sites = 50
epsilon_vec = np.zeros(N_sites)
t, l_max = 1.0, 10
U_vals = np.linspace(0.04, 2.0, 10)

Egs_single, Egs_cluster = [], []

count = 1

for U in U_vals:
    sys_single = SingleSite1D(
        epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, PBC=True, 
        N_at=2, N_target=2, iteration_method=0, print_cache_results=True
    )

    Egs_single.append(sys_single.get_energy_MF())
    print('here')

plt.figure(figsize=(6, 4))

plt.plot(U_vals, Egs_single, label=r"Single", marker="o", linestyle="-", color="dodgerblue")
#plt.plot(U_vals, Egs_cluster, label=r"Cluster", marker="o", linestyle="-", color="tomato")

plt.xlabel(r"$U$", rotation=0, fontsize=16, labelpad=12)
plt.ylabel(r"$E_{gs}$", rotation=0, fontsize=16, labelpad=24)
plt.title(r"Ground-State Energy as a Function of $U$ at half-filling: $\beta=\infty$, $N_{\text{sites}}=$" + f"{N_sites}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
