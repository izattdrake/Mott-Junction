import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

from Utility.plotting import save_axes_centered

if __name__ == "__main__":
    epsilon, t, beta, l_max = 0.0, 1.0, None, 15
    U_vals = np.linspace(0.01, 5, 50)

    double_occ_exact, double_occ_c, double_occ_dc  = [], [], []

    count = 1
    for U in U_vals:
        exact_dimer = ExactDimer(epsilon, U, t, mu=U/2)
        coupled_dimer = CoupledDimer(epsilon, U, t, l_max, print_cache_results=True)
        decoupled_dimer = DecoupledDimer(epsilon, U, t, l_max, print_cache_results=True)

        double_occ_exact.append(exact_dimer.get_double_occ_vec()[0])
        double_occ_c.append(coupled_dimer.get_double_occ_vec()[0])
        double_occ_dc.append(decoupled_dimer.get_double_occ_vec()[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
  
    ax.plot(U_vals, double_occ_exact, label="Exact", color="black", zorder=5)
 
    ax.plot(U_vals, double_occ_c, label="Coupled", color="dodgerblue")
    #ax.plot(U_vals, gs_c_proj, label="Coupled (projected)", color="dodgerblue", linestyle="--")
 
    ax.plot(U_vals, double_occ_dc, label="Decoupled", color="tomato")
    #ax.plot(U_vals, gs_dc_proj, label="Decoupled (projected)", color="tomato", linestyle="--")
 
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$D_1$")
    ax.set_xlim(U_vals.min(), U_vals.max())
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.5)
  
    save_axes_centered(fig, ax, f"Dimer/double_occ_epsilon{epsilon}_beta{beta}_lmax{l_max}.pdf")
    plt.show()

