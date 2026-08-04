import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

def Z_by_hand(U, t):
    Z = 1/(1 + (U/(6*t))**2)
    return Z

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, None, 15

    U_vals = np.linspace(0.02, 4.0, 20)
    
    exact_Z, c_Z, dc_Z = [], [], []
    count = 1

    for U in U_vals:
        exact_dimer = ExactDimer(epsilon, U, t, mu=U/2) 
        c_dimer = CoupledDimer(epsilon, U, t, l_max, print_cache_results=True)
        dc_dimer = DecoupledDimer(epsilon, U, t, l_max, print_cache_results=True)

        exact_Z.append(exact_dimer.get_qp_vec()[0])
        c_Z.append(c_dimer.get_qp_vec()[0])
        dc_Z.append(dc_dimer.get_Z_vec()[0])

        print(count); count += 1

    plt.figure(figsize=(6, 4))
    plt.plot(U_vals, exact_Z, color='black', label='Exact')
    plt.plot(U_vals, c_Z, color='dodgerblue', label='Coupled')
    plt.plot(U_vals, dc_Z, color='tomato', label='Decoupled')

    plt.xlabel(r'$U$')
    plt.ylabel(r'Quasiparticle spectral weight')
    plt.title('Quasiparticle coherence')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
