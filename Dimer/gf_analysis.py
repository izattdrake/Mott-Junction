import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

def gf_by_hand(freqs, U, t):
    delta = np.sqrt(U**2 + 16*t**2)

    gf_m = (freqs - 3*t)/((freqs - t)**2 - delta**2/4)
    gf_p = (freqs + 3*t)/((freqs + t)**2 - delta**2/4)
    return gf_p + gf_m


if __name__ == '__main__':
    epsilon, U, t, beta, l_max = 0.0, 1.0, 1.0, None, 10
    freqs = np.linspace(-5.0, 5.0, 1000)
    z = freqs + 1e-4j

    gf_hand, gf_exact, gf_c, gf_dc = [], [], [], []
    count = 1

    exact_dimer = ExactDimer(epsilon, U, t, mu=U/2)
    coupled_dimer = CoupledDimer(epsilon, U, t, l_max, print_cache_results=True)
    decoupled_dimer = DecoupledDimer(epsilon, U, t, l_max, print_cache_results=True)

    gf_exact = exact_dimer.get_gf(z, i=0, j=0, spin=0, beta=None)
    gf_c = coupled_dimer.get_gf(z, i=0, j=0)
    gf_dc = decoupled_dimer.get_gf(z, i=0, j=0)

    plt.figure(figsize=(6, 4))

    plt.plot(freqs, gf_exact, label='Exact', color='black')
    plt.plot(freqs, gf_hand, label='By Hand', color='red')

    #plt.plot(freqs, gf_c, label='Coupled', color='dodgerblue')
    #plt.plot(freqs, gf_dc, label='Decoupled', color='tomato')
    
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$G(\omega)$")
    plt.title("Green's function")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
