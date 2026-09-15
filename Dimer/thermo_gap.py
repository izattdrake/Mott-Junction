import numpy as np
import matplotlib.pyplot as plt

from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

from Utility.plotting import save_axes_centered

# ====================================================================
#   THERMODYNAMIC CHARGE GAP AT HALF-FILLING
# ====================================================================

def exact_charge_gap(U_vals, t):
    U_vals = np.atleast_1d(U_vals)
    return np.sqrt(U_vals**2 + 16*t**2) - 2*t

if __name__ == '__main__':
    epsilon, t, beta, l_max = -1.0, 1.0, None, 15
    U_vals = np.linspace(0.04, 5.0, 50)
    cg_exact = exact_charge_gap(U_vals, t)
    cg_c, cg_dc = [], []

    for U in U_vals:
        dc_N   = DecoupledDimer(epsilon, U, t, l_max, N_target=2, 
                                print_cache_results=True)
        dc_Np1 = DecoupledDimer(epsilon, U, t, l_max, N_target=3, 
                                print_cache_results=True)
        dc_Nm1 = DecoupledDimer(epsilon, U, t, l_max, N_target=1, 
                                print_cache_results=True)

        c_N = CoupledDimer(epsilon, U, t, l_max, N_target=2, 
                           print_cache_results=True)
        c_Np1 = CoupledDimer(epsilon, U, t, l_max, N_target=3, 
                             print_cache_results=True)
        c_Nm1 = CoupledDimer(epsilon, U, t, l_max, N_target=1, 
                             print_cache_results=True)
        
        c_E_N, c_E_Np1, c_E_Nm1 = c_N.get_energy_MF(), c_Np1.get_energy_MF(), c_Nm1.get_energy_MF()
        dc_E_N, dc_E_Np1, dc_E_Nm1 = dc_N.get_energy_MF(), dc_Np1.get_energy_MF(), dc_Nm1.get_energy_MF()
 
        c_charge_gap = c_E_Np1 + c_E_Nm1 - 2*c_E_N
        dc_charge_gap = dc_E_Np1 + dc_E_Nm1 - 2*dc_E_N

        cg_c.append(c_charge_gap)
        cg_dc.append(dc_charge_gap)
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.plot(U_vals, cg_exact, color='black', label='Exact')
    ax.plot(U_vals, cg_c, color='dodgerblue', label='Coupled')
    ax.plot(U_vals, cg_dc, color='tomato', label='Decoupled')

    ax.set_xlabel(r'$U/t$')
    ax.set_ylabel(r'$\Delta_c$')
    ax.set_xlim(0.04, 5.0)

    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.2)
    save_axes_centered(fig, ax, f"Dimer/thermo_charge_gap_beta{beta}_lmax{l_max}.pdf")    

    plt.show()
    """