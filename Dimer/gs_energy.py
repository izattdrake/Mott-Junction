import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

from Utility.plotting import save_axes_centered

if __name__ == "__main__":
    epsilon, t, beta, l_max = -0.5, 1.0, None, 15
    U_vals = np.linspace(0.04, 5, 200)

    gs_exact, gs_c, gs_dc  = [], [], []
    count = 1
    for U in U_vals:
        exact_dimer = ExactDimer(epsilon, U, t, mu=U/2)
        coupled_dimer = CoupledDimer(epsilon, U, t, l_max, print_cache_results=True)
        decoupled_dimer = DecoupledDimer(epsilon, U, t, l_max, print_cache_results=True)

        gs_exact.append(exact_dimer.get_E_gs())
        gs_c.append(coupled_dimer.get_energy_MF())
        gs_dc.append(decoupled_dimer.get_energy_MF())

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
  
    ax.plot(U_vals, gs_exact, label="Exact", color="black", zorder=5)
 
    ax.plot(U_vals, gs_c, label="Coupled", color="dodgerblue")
    #ax.plot(U_vals, gs_c_proj, label="Coupled (projected)", color="dodgerblue", linestyle="--")
 
    ax.plot(U_vals, gs_dc, label="Decoupled", color="tomato")
    #ax.plot(U_vals, gs_dc_proj, label="Decoupled (projected)", color="tomato", linestyle="--")
 
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$E_{\mathrm{g}}$")
    ax.set_xlim(U_vals.min(), U_vals.max())
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.5)
  
    save_axes_centered(fig, ax, f"Dimer/gs_energy_epsilon{epsilon}_beta{beta}_lmax{l_max}.pdf")
    plt.show()

