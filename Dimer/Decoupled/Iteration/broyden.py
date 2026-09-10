import numpy as np
import matplotlib.pyplot as plt
import sys 

from ..core import (
    get_H_spinon,
    get_nf_vec,
    get_rotor_eig_vec,
    get_L_vec,
    get_Z_vec,
    get_K_vec,
    get_t_eff
)

# ====================================================================
#   MODIFIED BROYDEN METHOD FOR DECOUPLED ROTOR ON HUBBARD DIMER 
# --------------------------------------------------------------------
#   In this script, we use an alternate iteration method for solving
#   for the parameters in the decoupled rotor dimer problem. Broyden is
#   a secant method in which we solve for the Jacobian once, then make
#   successive corrections rather than solving for it every iteration.
# --------------------------------------------------------------------
#   Bryoden's method keeps a memory of (dh, df) pairs to build more
#   accurate corrections. Corrections are weighted to keep the algo
#   stable, and the singular gauge subspace never has to be removed.
#   The method agrees with the gauge-fixed Newton's method script 
#   to fantastic accuracy, reassuring myself that results are
#   correct. 
# ====================================================================

MAX_ITER = 1e7

ALPHA_H = 5e-3
ALPHA_Z = 5e-3
ALPHA_K = 5e-3
ALPHA_T_EFF = 5e-3
ALPHA_MU = 5e-3

TOL_H = 1e-10
TOL_T_EFF = 1e-10
TOL_Z = 1e-10
TOL_K = 1e-10
TOL_N = 1e-6
TOL_F = 1e-10

WEIGHT = 1e-2
MEMORY = 8

def modified_broyden_step(f_vec, h_vec, alpha, weight, 
                          dF_hist, u_hist, h_prev, f_prev):
    """
        One limited-memory modified-Broyden step.
        The history lists dF_hist and u_hist accumulate the (normalized) residual 
        differences and the Johnson vectors u = alpha * dFhat + dhhat
    """
    if h_prev is None:
        return alpha * f_vec
 
    dF = f_vec - f_prev
    norm = np.linalg.norm(dF)
    if norm < 1e-64:
        return alpha * f_vec
 
    dF_hat = dF / norm
    dh_hat = (h_vec - h_prev) / norm
    dF_hist.append(dF_hat)
    u_hist.append(alpha * dF_hat + dh_hat)
 
    m = len(dF_hist)
    DF = np.array(dF_hist)                        
    a = DF @ DF.T  # transition matrix <dFhat_k|dFhat_l>
    beta_mat = np.linalg.inv(weight**2 * np.eye(m) + a)
    c = DF @ f_vec                                 # c_k = <dFhat_k|f>
    gamma = beta_mat @ c
 
    return alpha * f_vec - np.array(u_hist).T @ gamma

def iterate_broyden(epsilon_vec, t_mat, U, t_eff_mat_guess, h_vec_guess, Z_vec_guess, K_vec_guess, beta, l_max, N_target=2, N_at=2):
    """Calculate self-consistent parameters w/ modified Broyden's method"""
    mu = 0.0 # Global chemical potential

    dF_hist = []
    u_hist = []
    h_prev = None
    f_prev = None

    did_converge = False
    count = 1

    h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update = h_vec_guess, Z_vec_guess, K_vec_guess, t_eff_mat_guess

    while not did_converge and count < MAX_ITER:
        h_vec, Z_vec, K_vec, t_eff_mat = h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update
        
        evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu))
        evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, h_vec, K_vec, l_max)

        # Calculate nf and L for Jacobian
        nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta, N_at)
        L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

        f_vec = nf_vec - L_vec - N_at / 2.0
        delta_h = modified_broyden_step(f_vec, h_vec, ALPHA_H, WEIGHT, dF_hist, u_hist, h_prev, f_prev)
        
        if len(dF_hist) > MEMORY:
            dF_hist.pop(0)
            u_hist.pop(0)

        h_prev = h_vec.copy()
        f_prev = f_vec.copy()
        h_vec_update = h_vec + delta_h

        # Update parameters
        Z_vec_update = (1.0 - ALPHA_Z) * Z_vec + ALPHA_Z * get_Z_vec(evals_rotor_vec, evecs_rotor_vec, beta)
        K_vec_update = (1.0 - ALPHA_K) * K_vec + ALPHA_K * get_K_vec(t_mat, Z_vec, evals_spinon, evecs_spinon, beta, N_at)
        t_eff_mat_update = (1 - ALPHA_T_EFF) * t_eff_mat + ALPHA_T_EFF * get_t_eff(t_mat, Z_vec)

        # Adjust total particle number
        N = np.sum(nf_vec)
        mu += ALPHA_MU * (N_target - N)

        # Test convergence of all parameters within selected bounds
        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_mat_update, t_eff_mat, atol=TOL_T_EFF) and
            np.allclose(Z_vec_update, Z_vec, atol=TOL_Z) and
            np.allclose(K_vec_update, K_vec, atol=TOL_K) and
            np.allclose(N, N_target, atol=TOL_N) and
            np.linalg.norm(f_vec) < TOL_F
        )

        count += 1

    if not did_converge:
        print(f'ERROR: DID NOT CONVERGE AFTER {MAX_ITER} ITERATIONS')
        sys.exit()

    # Calculate nf, L with final Hamiltonians before return
    evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat_update, h_vec_update, mu))
    evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, h_vec_update, K_vec_update, l_max)

    nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta, N_at)
    L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

    return evals_spinon, evecs_spinon, evals_rotor_vec, evecs_rotor_vec, h_vec_update, Z_vec_update, K_vec_update, t_eff_mat, nf_vec, L_vec, mu, count
