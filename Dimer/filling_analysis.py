import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

if __name__ == '__main__':
    U, t, beta, l_max = 0.5, 1.0, None, 15
    epsilon_vals = np.linspace(0.0, 1.0, 20)

    fill_exact, fill_c, fill_dc = [], [], []

    for epsilon in epsilon_vals:
        exact = ExactDimer(epsilon, U, t, mu=U/2.0)
        coupled = CoupledDimer(epsilon, U, t, l_max, beta=None, print_cache_results=True)
        decoupled = DecoupledDimer(epsilon, U, t, l_max, beta=None, print_cache_results=True)

        fill_exact.append(exact.get_filling_vec(beta=None)[0])
        fill_c.append(coupled.get_filling_vec()[0])
        fill_dc.append(decoupled.get_filling_vec()[0])

    plt.figure(figsize=(6, 4))

    plt.plot(epsilon_vals, fill_exact, label='Exact', color='black')
    plt.plot(epsilon_vals, fill_c, label='Coupled', color='dodgerblue')
    plt.plot(epsilon_vals, fill_dc, label='Decoupled', color='tomato')

    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$n$")
    plt.title("2nd site filling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
