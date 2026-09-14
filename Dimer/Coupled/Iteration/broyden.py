import numpy as np
import sys
 
from ..core import (
    get_H_spinon,
    get_t_eff,
    get_nf_vec,
    get_H_rotor,
    get_J_eff,
    get_L_vec,
)

# ====================================================================
#   MODIFIED BROYDEN METHOD FOR SOLVING THE COUPLED ROTOR PROBLEM
# --------------------------------------------------------------------
#   In this script, we use an alternate iteration method for solving
#   for the parameters in the coupled rotor dimer problem. Broyden is
#   a secant method in which we solve for the Jacobian once, then make
#   successive corrections rather than solving for it every iteration.
# --------------------------------------------------------------------
#   Bryoden's method keeps a memory of (dh, df) pairs to build more
#   accurate corrections. Corrections are weighted to keep corrections
#   stable, and the singular gauge subspace never has to be removed.
#   The method agrees with the gauge-fixed Newton's method script 
#   to fantastic accuracy, reassuring myself that results are
#   accurate. 
# ====================================================================
 
MAX_ITER = 1e9

ALPHA_H = 5e-3
ALPHA_T_EFF = 5e-3
ALPHA_J_EFF = 5e-3
ALPHA_MU = 5e-3

TOL_H = 1e-9
TOL_T_EFF = 1e-9
TOL_J_EFF = 1e-9
TOL_N  = 1e-6
TOL_F = 1e-9

# Modified-Broyden control parameters
WEIGHT = 1e-2
MEMORY = 8

def modified_broyden_step(f_vec, h_vec, alpha, weight, dF_hist, u_hist, h_prev, f_prev):
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
 
def iterate_broyden(epsilon_vec, U, t, h_vec_guess, t_eff_guess, J_eff_guess, beta, l_max, N_target=2.0):
    """Calculate self-consistent parameters w/ modified Broyden's method"""
    mu = 0.0 # Global chemical potential 
    
    # Modified-Broyden history for the constraint residual f = n - <L> - 1.
    dF_hist = []
    u_hist = []
    h_prev = None
    f_prev = None
 
    did_converge = False
    count = 1

    h_vec_update, t_eff_update, J_eff_update = h_vec_guess, t_eff_guess, J_eff_guess

    while not did_converge and count < MAX_ITER:
        h_vec, t_eff, J_eff  = h_vec_update, t_eff_update, J_eff_update
 
        evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, h_vec, t_eff, mu))
        evals_rotor, evecs_rotor = np.linalg.eigh(get_H_rotor(U, h_vec, J_eff, l_max))
 
        nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta)
        L_vec = get_L_vec(evals_rotor, evecs_rotor, beta, l_max)
 
        f_vec = nf_vec - L_vec - 1.0
        delta_h = modified_broyden_step(f_vec, h_vec, ALPHA_H, WEIGHT,
                                        dF_hist, u_hist, h_prev, f_prev)
        if len(dF_hist) > MEMORY:
            dF_hist.pop(0)
            u_hist.pop(0)
 
        h_prev = h_vec.copy()
        f_prev = f_vec.copy()
        h_vec_update = h_vec + delta_h
 
        t_eff_update = (1 - ALPHA_T_EFF) * t_eff + ALPHA_T_EFF * get_t_eff(evals_rotor, evecs_rotor, t, beta, l_max)
        J_eff_update = (1 - ALPHA_J_EFF) * J_eff + ALPHA_J_EFF * get_J_eff(evals_spinon, evecs_spinon, t, beta)
 
        N = np.sum(nf_vec)
        mu += ALPHA_MU * (N_target - N)
 
        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_update, t_eff, atol=TOL_T_EFF) and
            np.allclose(J_eff_update, J_eff, atol=TOL_J_EFF) and
            np.allclose(N, N_target, atol=TOL_N) and
            np.linalg.norm(f_vec) < TOL_F
        )
        
        count += 1
 
    if not did_converge:
        print(f'ERROR: Did not converge after {MAX_ITER} iterations')
        sys.exit()
 
    evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, h_vec, t_eff, mu))
    evals_rotor, evecs_rotor = np.linalg.eigh(get_H_rotor(U, h_vec, J_eff, l_max))
 
    nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta)
    L_vec = get_L_vec(evals_rotor, evecs_rotor, beta, l_max)

    return evals_spinon, evecs_spinon, evals_rotor, evecs_rotor, h_vec_update, t_eff_update, J_eff_update, nf_vec, L_vec, mu, count