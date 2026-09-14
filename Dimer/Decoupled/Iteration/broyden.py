import numpy as np
import matplotlib.pyplot as plt
import sys 

from ..core import (
    get_H_spinon,
    get_nf_vec,
    get_occupations,
    get_rho,
    get_rotor_eig_vec,
    get_L_vec,
    get_Z_vec,
    get_K_vec,
    get_t_eff,
    get_residual
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

MAX_ITER    = 1e9

ALPHA_H     = 5e-3
ALPHA_Z     = 5e-3
ALPHA_K     = 5e-3
ALPHA_T_EFF = 5e-3
ALPHA_MU    = 1e-4

TOL_H       = 1e-9
TOL_T_EFF   = 1e-9
TOL_Z       = 1e-9
TOL_K       = 1e-9
TOL_N       = 1e-6
TOL_F       = 1e-9

WEIGHT     = 1e-2
NORM_FLOOR = 1e-13

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
    if norm < NORM_FLOOR:
        return alpha * f_vec
 
    dF_hat = dF / norm
    dh_hat = (h_vec - h_prev) / norm
    dF_hist.append(dF_hat)
    u_hist.append(alpha * dF_hat + dh_hat)
 
    m = len(dF_hist)
    DF = np.array(dF_hist)                        
    a = DF @ DF.T                                        # transition matrix <dFhat_k|dFhat_l>
    beta_mat = np.linalg.inv(weight**2 * np.eye(m) + a)
    c = DF @ f_vec                                       # c_k = <dFhat_k|f>
    gamma = beta_mat @ c
 
    return alpha * f_vec - np.array(u_hist).T @ gamma

def iterate_broyden(epsilon_vec, t, U, t_eff_guess, h_vec_guess, Z_vec_guess, K_vec_guess,
                    beta, l_max, N_target=2, memory=8):
    """Calculate self-consistent parameters w/ modified Broyden's method"""
    dF_hist, u_hist = [], []
    h_prev = f_prev = None

    did_converge = False
    count = 1

    h_vec_update, Z_vec_update, K_vec_update, t_eff_update = h_vec_guess, Z_vec_guess, K_vec_guess, t_eff_guess

    while not did_converge and count < MAX_ITER:
        h_vec, Z_vec, K_vec, t_eff = h_vec_update, Z_vec_update, K_vec_update, t_eff_update
        
        # Spinon sector
        evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff, h_vec))
        occ, mu = get_occupations(evals_spinon, N_target, beta)
        nf_vec = get_nf_vec(evecs_spinon, occ)
        rho = get_rho(evecs_spinon, occ)

        # Rotor sector
        evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, h_vec, K_vec, l_max)
        L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

        # Constraint & modified Broyden step
        f_vec = get_residual(nf_vec, L_vec)
        delta_h = modified_broyden_step(f_vec, h_vec, ALPHA_H, WEIGHT, dF_hist, u_hist, h_prev, f_prev)
        
        if len(dF_hist) > memory:
            dF_hist.pop(0)
            u_hist.pop(0)

        h_prev, f_prev = h_vec.copy(), f_vec.copy()
        h_vec_update = h_vec + delta_h

        # Update parameters
        Z_vec_update = (1.0 - ALPHA_Z) * Z_vec + ALPHA_Z * get_Z_vec(evals_rotor_vec, evecs_rotor_vec, beta)
        K_vec_update = (1.0 - ALPHA_K) * K_vec + ALPHA_K * get_K_vec(t, Z_vec, rho)
        t_eff_update = (1.0 - ALPHA_T_EFF) * t_eff + ALPHA_T_EFF * get_t_eff(t, Z_vec)

        # Test convergence of all parameters within selected bounds
        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_update, t_eff, atol=TOL_T_EFF) and
            np.allclose(Z_vec_update, Z_vec, atol=TOL_Z) and
            np.allclose(K_vec_update, K_vec, atol=TOL_K) and
            np.linalg.norm(f_vec) < TOL_F
        )

        count += 1

    if not did_converge:
        raise RuntimeError(f"Broyden did not converge after {MAX_ITER} iterations.")
        sys.exit()

    """Final evaluation of converged parameters"""
    # Spinon sector
    evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_update, h_vec_update))
    occ, mu = get_occupations(evals_spinon, N_target, beta)
    nf_vec = get_nf_vec(evecs_spinon, occ)

    # Rotor sector
    evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, h_vec_update, K_vec_update, l_max)
    L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

    return (evals_spinon, evecs_spinon, evals_rotor_vec, evecs_rotor_vec, 
            h_vec_update, Z_vec_update, K_vec_update, t_eff, 
            nf_vec, L_vec, mu, occ, rho, count)
