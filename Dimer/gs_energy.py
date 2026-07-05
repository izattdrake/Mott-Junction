import numpy as np
import matplotlib.pyplot as plt

import Exact.core as Exact
import Coupled.core as Coupled
import Decoupled.core as Decoupled

from Coupled.core import get_Z_vec as get_coupled_Z_vec, get_H_rotor as get_coupled_H_rotor
from Coupled.broyden import coupled_broyden_params

import Decoupled.core as Decoupled
from Decoupled.broyden import decoupled_broyden_params

if __name__ == '__main__':
    l_max = 20
    beta = 50.0

    t_nn = 1.0
    t_mat = np.diag(np.full(1, t_nn), k=1) + np.diag(np.full(1, t_nn), k=-1)

    U_vals = np.linspace(0.0, 4, 20)
    epsilon_vec = np.zeros(2)

    t_eff_mat_guess = 0.9 * t_mat
    h_vec_guess = -epsilon_vec
    t_eff_guess = t_nn
    J_eff_guess = 0.5 * t_nn
    Z_vec_guess = np.full(2, 0.9)
    K_vec_guess = np.full(2, -1.0)

    cycle = 1

    gs_exact, gs_c, gs_dc = [], [], []

    for U in U_vals:
        exact_H = Exact.get_H(epsilon_vec[0], U, t_nn, Exact.get_def_particle_basis(2))
        evals_exact, _ = np.linalg.eigh(exact_H)

        h_vec_c, t_eff_c, J_eff_c, nf_vec_c, L_vec_c, mu_c, count_c = coupled_broyden_params(epsilon_vec, U, t_nn, h_vec_guess, 
                                                                                            t_eff_guess, J_eff_guess, beta, l_max)
        
        h_vec_dc, Z_vec_dc, K_vec_dc, t_eff_mat_dc, nf_vec_dc, L_vec_dc, mu_dc, count_dc = decoupled_broyden_params(epsilon_vec, t_mat, U, t_eff_mat_guess,
                                                                                                             h_vec_guess, Z_vec_guess, K_vec_guess,
                                                                                                             beta, l_max)
        
        gs_exact.append(evals_exact[0])
        gs_c.append(Coupled.get_E_gs(epsilon_vec, U, h_vec_c, t_eff_c, J_eff_c, mu_c, beta, l_max))
        gs_dc.append(Decoupled.get_E_gs(epsilon_vec, U, h_vec_dc, t_eff_mat_dc, Z_vec_dc, K_vec_dc, mu_dc, l_max))
        
        print(f'cycle: {cycle}')
        cycle += 1
        
    print(gs_exact)
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
