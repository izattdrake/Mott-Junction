import numpy as np
import matplotlib.pyplot as plt

from D1.SingleSite.single_site_1D import SingleSite_1D
from ClusterSite.cluster_site_1D import ClusterSite1D

from scipy.special import j0, j1, expit
 
_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)
 
def lieb_wu_energy_per_site(U, t=1.0, w_max=1e5):
    """Exact E_g/N for the half-filled 1D Hubbard chain."""
    if U < 0:
        raise ValueError("U < 0")
    
    x = U / (2.0 * t)
 
    W = w_max if x == 0 else min(w_max, 60.0 / x)       # e^{-60} ~ 1e-26
    h = np.pi if x == 0 else min(np.pi, 2.0 / x)        # chunk length
    n = int(np.ceil(W / h))
 
    a = h * np.arange(n)[:, None]                       # chunk left edges
    w = a + 0.5 * h * (_GL_X[None, :] + 1.0)            # GL nodes, shape (n, 40)
 
    small = w < 1e-8
    j1_over_w = np.where(small, 0.5, j1(w) / np.where(small, 1.0, w))
    f = j0(w) * j1_over_w * expit(-w * x)
 
    integral = 0.5 * h * np.sum(f * _GL_W[None, :])
    return -4.0 * t * integral
 
 
def lieb_wu_energy(U, N, t=1.0):
    """Total energy E_g = N * (E_g/N).  Thermodynamic-limit value scaled by N."""
    return N * lieb_wu_energy_per_site(U, t)
 
N_sites, N_at, N_target = 100, 2, 100

t, l_max, beta = 1.0, 10, None

epsilon_vec = np.zeros(N_sites)
U_vals = np.linspace(0.01, 5.0, 500)
U_vals = U_vals[U_vals < 2.00]

Egs_single, Egs_cluster, Egs_exact = [], [], []

for U in U_vals:
    single = SingleSite_1D(
        epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
        N_at=N_at, N_target=N_target, PBC=True,
        print_cache_results=True
    )
    cluster = ClusterSite1D(
        epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
        N_at=N_at, N_target=N_target, PBC = True,
        print_cache_results=True
    )

    Egs_single.append(single.get_energy_MF())
    Egs_cluster.append(cluster.get_energy_MF())
    Egs_exact.append(lieb_wu_energy(U, N_sites))

plt.figure(figsize=(6, 4))

plt.plot(U_vals, Egs_single, label=r"Single", linestyle="-", color="dodgerblue")
plt.plot(U_vals, Egs_cluster, label=r"Cluster", linestyle="-", color="tomato")
plt.plot(U_vals, Egs_exact, label=r"Exact", linestyle="-", color="Black")

plt.xlabel(r"$U$", rotation=0, fontsize=16, labelpad=12)
plt.ylabel(r"$E_{g}$", rotation=0, fontsize=16, labelpad=24)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
