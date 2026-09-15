import numpy as np
import matplotlib.pyplot as plt

from ..core import (
    get_H_spinon,
    get_nf_vec,
    get_occupations,
    get_rho,
    get_rotor_eig_vec,
    get_L_vec,
    get_L2_vec,
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

MAX_ITER    = 1e8
ALPHA       = 5e-3
TOL_F       = 1e-9

WEIGHT     = 1e-2
NORM_FLOOR = 1e-10

def modified_broyden_step(f_vec, x_vec, alpha, weight, 
                          dF_hist, u_hist, x_prev, f_prev):
    """One modified Broyden step. dF_hist and u_hist accumulate the past residue
    differences."""
    if x_prev is None:
        return alpha * f_vec

    dF = f_vec - f_prev
    norm = np.linalg.norm(dF)
    if norm < NORM_FLOOR:
        return alpha * f_vec

    dF_hat = dF / norm
    dx_hat = (x_vec - x_prev) / norm
    dF_hist.append(dF_hat)
    u_hist.append(alpha * dF_hat + dx_hat)

    m = len(dF_hist)
    DF = np.array(dF_hist)
    a = DF @ DF.T                                        # transition matrix <dFhat_k|dFhat_l>
    beta_mat = np.linalg.inv(weight**2 * np.eye(m) + a)
    c = DF @ f_vec                                       # c_k = <dFhat_k|f>
    gamma = beta_mat @ c

    return alpha * f_vec - np.array(u_hist).T @ gamma

def iterate_broyden(epsilon_vec, U, t, h_vec_guess, Z_vec_guess,
                    beta, l_max, N_target=2, memory=8):
    """Calculate self-consistent parameters w/ modified Broyden's method"""
    dF_hist, u_hist = [], []
    x_prev = f_prev = None

    did_converge = False
    count = 1

    x = np.concatenate([np.asarray(h_vec_guess, float), np.asarray(Z_vec_guess, float)])

    while not did_converge and count < MAX_ITER:
        h_vec, Z_vec = x[:2], np.clip(x[2:], 1e-12, 1.0)

        # Spinon sector
        t_eff = get_t_eff(t, Z_vec)
        evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff, h_vec))
        occ, mu = get_occupations(evals_spinon, N_target, beta)
        nf_vec = get_nf_vec(evecs_spinon, occ)
        rho = get_rho(evecs_spinon, occ)

        # Rotor sector
        K_vec = get_K_vec(t, Z_vec, rho)
        evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, h_vec, K_vec, l_max)
        L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

        # Constraint & modified Broyden step
        f_vec = np.concatenate([get_residual(nf_vec, L_vec),
                                get_Z_vec(evals_rotor_vec, evecs_rotor_vec, beta) - Z_vec])
        
        did_converge = np.linalg.norm(f_vec) < TOL_F
        if did_converge:
            break

        delta_x = modified_broyden_step(f_vec, x, ALPHA, WEIGHT,
                                        dF_hist, u_hist, x_prev, f_prev)
        
        if len(dF_hist) > memory:
            dF_hist.pop(0)
            u_hist.pop(0)

        x_prev, f_prev = x.copy(), f_vec.copy()
        x = x + delta_x

        count += 1

    if not did_converge:
        raise RuntimeError(f"Broyden did not converge after {MAX_ITER} iterations.")
        
    L2_vec = get_L2_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

    return (evals_spinon, evecs_spinon, evals_rotor_vec, evecs_rotor_vec, 
            h_vec, Z_vec, K_vec, t_eff, 
            nf_vec, L_vec, L2_vec, mu, occ, rho, count)
