import numpy as np
import matplotlib.pyplot as plt

from Exact.Exact_Dimer import Exact_Dimer
from Coupled.Coupled_Dimer import Coupled_Dimer
from Decoupled.Decoupled_Dimer import Decoupled_Dimer

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, 50.0, 15
    U_vals = np.linspace(0.0, 4.0, 20)

    Z_output = {
        'coupled': [],
        'decoupled': []
    }

    for U in U_vals:
        coupled_dimer = Coupled_Dimer(epsilon, U, t, beta, l_max)
        decoupled_dimer = Decoupled_Dimer(epsilon, U, t, beta, l_max)

        Z_output['coupled'].append(coupled_dimer.get_Z_vec()[0])
        Z_output['decoupled'].append(decoupled_dimer.get_Z_vec()[0])
        
    plt.figure(figsize=(6, 4))
    plt.plot(U_vals, Z_output['coupled'], 'o-', lw=1.5, ms=4, color='b', label='Coupled')
    plt.plot(U_vals, Z_output['decoupled'], '^-', lw=1.5, ms=4, color='r', label='Decoupled')

    plt.xlabel(r'$U$')
    plt.ylabel(r'$\langle e^{i\theta}\rangle$')
    plt.title('Quasiparticle coherence')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
