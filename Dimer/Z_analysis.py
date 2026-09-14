import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

from Utility.plotting import save_axes_centered

def Z_by_hand(U, t):
    Z = 1/(1 + (U/(6*t))**2)
    return Z

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, None, 15

    U_vals = np.linspace(0.04, 5.0, 50)
    
    Z_exact, Z_c, Z_dc = [], [], []

    for U in U_vals:
        exact_dimer = ExactDimer(epsilon, U, t, mu=U/2) 
        c_dimer = CoupledDimer(epsilon, U, t, l_max, print_cache_results=True)
        dc_dimer = DecoupledDimer(epsilon, U, t, l_max, print_cache_results=True)

        Z_exact.append(exact_dimer.get_qp_vec()[0])
        Z_c.append(c_dimer.get_qp_vec()[0])
        Z_dc.append(dc_dimer.get_Z_vec()[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.plot(U_vals, Z_exact, color='black', label='Exact')
    ax.plot(U_vals, Z_c, color='dodgerblue', label='Coupled')
    ax.plot(U_vals, Z_dc, color='tomato', label='Decoupled')

    ax.set_xlabel(r'$U/t$')
    ax.set_ylabel(r'Z')
    ax.set_xlim(0, 5.0)

    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.2)
    save_axes_centered(fig, ax, f"Dimer/Z_analysis_beta{beta}_lmax{l_max}.pdf")    

    plt.show()