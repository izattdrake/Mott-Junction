import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, None, 10
    U_vals = np.linspace(0, 1, 5)
    delta=0.05
    x = np.linspace(-6, 6, 1000, dtype=complex)
    z = x + 1j*delta

    U = 2.5

    exact_dimer = Exact_Dimer(epsilon, U, t, mu=U/2)
    coupled_dimer = Coupled_Dimer(epsilon, U, t, beta, l_max)
    decoupled_dimer = Decoupled_Dimer(epsilon, U, t, beta, l_max)

    gf_exact = exact_dimer.get_spectral(x, site=0, delta=delta, beta=beta)
    gf_c = coupled_dimer.get_spectral(x, delta=delta)
    gf_dc = decoupled_dimer.get_spectral(x, delta=delta)

    print(f'Coupled GF Sum: {coupled_dimer.check_gf_sum()}')
    print(f'Decoupled GF Sum: {decoupled_dimer.check_gf_sum()}')
    
    plt.figure(figsize=(6, 4))

    plt.plot(x, gf_exact, label='Exact', color='black')
    plt.plot(x, gf_c, label='Coupled SR', color='#3776ab')
    plt.plot(x, gf_dc, label='Decoupled SR', color='orange', linestyle='--')

    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\rho_{11}(\omega)$')
    plt.title(r'Spectral Function $\rho_{11}(\omega)$')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
