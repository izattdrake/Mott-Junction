import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

from Utility.plotting import save_axes_centered

if __name__ == "__main__":
    epsilon, t, beta, l_max = 0.0, 1.0, None, 15
    U_vals = np.linspace(0.04, 5, 20)

    gs_exact, gs_c, gs_dc, gs_c_proj, gs_dc_proj = [], [], [], [], []
    count = 1
    for U in U_vals:
        exact_dimer = ExactDimer(epsilon, U, t, mu=U/2)
        coupled_dimer = CoupledDimer(epsilon, U, t, l_max, print_cache_results=True)
        decoupled_dimer = DecoupledDimer(epsilon, U, t, l_max, print_cache_results=True)

        gs_exact.append(exact_dimer.get_E_gs())

        gs_c.append(coupled_dimer.get_E_gs())
        gs_c_proj.append(coupled_dimer.get_E_gs_proj())

        gs_dc.append(decoupled_dimer.get_E_gs())
        gs_dc_proj.append(decoupled_dimer.get_E_gs_proj())

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
  
    ax.plot(U_vals, gs_exact, label="Exact", color="black", zorder=5)
 
    ax.plot(U_vals, gs_c, label="Coupled", color="dodgerblue")
    ax.plot(U_vals, gs_c_proj, label="Coupled (projected)", color="dodgerblue", linestyle="--")
 
    ax.plot(U_vals, gs_dc, label="Decoupled", color="tomato")
    ax.plot(U_vals, gs_dc_proj, label="Decoupled (projected)", color="tomato", linestyle="--")
 
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$E_{\mathrm{g}}/t$")
    ax.set_xlim(U_vals.min(), U_vals.max())
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.5)
  
    save_axes_centered(fig, ax, "Dimer/gs_energy.pdf")
    plt.show()

