import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

from Utility.plotting import save_axes_centered

def spectral_by_hand(freqs, U, t, eta=1e-2):
    None
    
if __name__ == "__main__":
    epsilon, t, l_max = 0.5, 1.0, 15
    delta=0.05
    x = np.linspace(-5, 5, 1000, dtype=complex)
    z = x + 1j*delta

    U = 0.04

    exact_dimer = ExactDimer(epsilon, U, t, mu=U/2)
    decoupled_dimer = DecoupledDimer(epsilon, U, t, l_max, print_cache_results=True)
    print("here")
    coupled_dimer = CoupledDimer(epsilon, U, t, l_max, print_cache_results=True, nearest_params_max_dist=False)

    gf_exact = exact_dimer.get_spectral_vec(x, imag_delta=delta)[:,0]
    gf_c = coupled_dimer.get_spectral_vec(x, imag_delta=delta)[:,0]
    gf_dc = decoupled_dimer.get_spectral_vec(x, imag_delta=delta)[:,0]
    
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    
    ax.plot(x, gf_exact, label="Exact", color="black")
    ax.plot(x, gf_c, label="Coupled", color="dodgerblue")
    ax.plot(x, gf_dc, label="Decoupled", color="tomato")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho_{11}(\omega)$")
    ax.set_xlim(-5, 5)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.2)

    save_axes_centered(fig, ax, f"Dimer/spectral_epsilon{epsilon}_U{U}_l{l_max}_delta{delta}.pdf")
    plt.show()
