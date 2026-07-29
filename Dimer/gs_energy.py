import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, None, 10
    U_vals = np.linspace(0, 4, 100)

    gs_exact, gs_c, gs_dc, gs_c_proj, gs_dc_proj = [], [], [], [], []
    count = 1
    for U in U_vals:
        exact_dimer = ExactDimer(epsilon, U, t, mu=U/2)
        coupled_dimer = CoupledDimer(epsilon, U, t, beta, l_max, print_cache_results=True)
        decoupled_dimer = DecoupledDimer(epsilon, U, t, beta, l_max, print_cache_results=True)

        gs_exact.append(exact_dimer.get_E_gs())

        gs_c.append(coupled_dimer.get_E_gs())
        gs_c_proj.append(coupled_dimer.get_E_gs_proj())

        gs_dc.append(decoupled_dimer.get_E_gs())
        gs_dc_proj.append(decoupled_dimer.get_E_gs_proj())


    plt.figure(figsize=(6, 4))

    plt.plot(U_vals, gs_exact, label='Exact', color='black')

    plt.plot(U_vals, gs_c, label='Coupled', color='dodgerblue')
    plt.plot(U_vals, gs_c_proj, label='Coupled (Projected)', color='dodgerblue', linestyle='--')
    
    plt.plot(U_vals, gs_dc, label='Decoupled', color='tomato')
    plt.plot(U_vals, gs_dc_proj, label='Decoupled (Projected)', color='tomato', linestyle='--')
    
    plt.xlabel(r'$U/t$')
    plt.ylabel(r'$E_g$')
    plt.title('Ground-State Energy with Slave-Rotor Approximation Schemes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
