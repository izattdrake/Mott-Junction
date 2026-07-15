import numpy as np
import matplotlib.pyplot as plt

from Exact.Exact_Dimer import Exact_Dimer
from Coupled.Coupled_Dimer import Coupled_Dimer
from Decoupled.Decoupled_Dimer import Decoupled_Dimer

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, 50.0, 15
    U_vals = np.linspace(0, 5, 5)

    gs_exact, gs_c, gs_dc = [], [], []

    for U in U_vals:
        exact_dimer = Exact_Dimer(epsilon, t, U, mu=U/2)
        coupled_dimer = Coupled_Dimer(epsilon, U, t, beta, l_max)
        decoupled_dimer = Decoupled_Dimer(epsilon, U, t, beta, l_max)

        gs_exact.append(exact_dimer.get_E_gs())
        gs_c.append(coupled_dimer.get_E_gs())
        gs_dc.append(decoupled_dimer.get_E_gs())
        
    plt.figure(figsize=(6, 4))

    plt.plot(U_vals, gs_exact, label='Exact')
    plt.plot(U_vals, gs_c, label='Coupled')
    plt.plot(U_vals, gs_dc, label='Decoupled')

    plt.xlabel(r'$U$')
    plt.ylabel(r'$E_g$')
    plt.title('Ground-State Energy w/o Phys-Subspace Projection')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
