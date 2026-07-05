import numpy as np
import matplotlib.pyplot as plt

from Coupled.core import get_Z_vec as get_coupled_Z_vec, get_H_rotor as get_coupled_H_rotor
from Coupled.broyden import coupled_broyden_params
from Decoupled.core import get_Z_vec as get_decoupled_Z_vec
from Decoupled.broyden import decoupled_broyden_params

if __name__ == '__main__':
    l_max = 15
    beta = 50.0

    t_nn = 1.0
    t_mat = np.diag(np.full(1, t_nn), k=1) + np.diag(np.full(1, t_nn), k=-1)

    U_vals = np.linspace(0.0, 4.0, 20)
    epsilon_vec = np.zeros(2)

    t_eff_mat_guess = 0.9 * t_mat
    h_vec_guess = -epsilon_vec
    t_eff_guess = t_nn
    J_eff_guess = 0.5 * t_nn
    Z_vec_guess = np.full(2, 0.9)
    K_vec_guess = np.full(2, -1.0)

    Z_output = {
        'coupled': [],
        'decoupled': []
    }

    cycle = 1

    for U in U_vals:
        h_vec_c, t_eff_c, J_eff_c, mu_c, nf_vec_c, L_vec_c, count_c = coupled_broyden_params(epsilon_vec, U, t_nn, h_vec_guess, 
                                                                              t_eff_guess, J_eff_guess, beta, l_max)
        h_vec_dc, Z_vec_dc, K_vec_dc, t_eff_mat_dc, nf_vec_dc, L_vec_dc, count_dc = decoupled_broyden_params(epsilon_vec, t_mat, U, t_eff_mat_guess,
                                                                                                             h_vec_guess, Z_vec_guess, K_vec_guess,
                                                                                                             beta, l_max)
        rotor_evals_c, rotor_evecs_c = np.linalg.eigh(get_coupled_H_rotor(U, h_vec_c, J_eff_c, l_max))
        Z_vec_c = get_coupled_Z_vec(rotor_evals_c, rotor_evecs_c, beta, l_max)

        Z_output['coupled'].append(t_eff_c)
        Z_output['decoupled'].append(t_eff_mat_dc[0,1])

        print(cycle)
        cycle += 1
        
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
