import numpy as np
import matplotlib.pyplot as plt

from SingleSite.single_site_1D import SingleSite1D

if __name__ == "__main__":
    t, l_max, N_sites = (1.0,0.1), 15, 100
    epsilon_vec = np.zeros(N_sites)
    U_vals = np.linspace(0.04, 5.0, 50)
    gs_E2, gs_E4 = [], []
    count = 1
    for U in U_vals:
        chain2 = SingleSite1D(
            epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
            N_at=2, iteration_method=0, print_cache_results=True
        )
        chain4 = SingleSite1D(
            epsilon_vec, U, t, l_max, beta=None, N_sites=N_sites, 
            N_at=4, iteration_method=0, print_cache_results=True
        )

        gs_E2.append(chain2.get_E_gs())
        gs_E4.append(chain4.get_E_gs())

        print(count); count += 1

    plt.figure(figsize=(6, 4))

    plt.plot(U_vals, gs_E2, label=r"$N_{at}=2$", marker="o", linestyle="-", color="dodgerblue")
    plt.plot(U_vals, gs_E4, label=r"$N_{at}=4$", marker="o", linestyle="-", color="tomato")

    plt.xlabel(r"$U$", rotation=0, fontsize=16, labelpad=12)
    plt.ylabel(r"$E_{gs}$", rotation=0, fontsize=16, labelpad=24)
    plt.title(r"Ground-State Energy as a Function of $U$ at half-filling: $\beta=\infty$, $N_{\text{sites}}=$" + f"{N_sites}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
