import numpy as np
import sys

from typing import Union
from ..core import (
    get_H_spinon,
    get_t_eff_mat,
    get_nf_vec,
    get_J_eff_mat,
    get_L_vec,
    get_L2_vec,
    get_rotor_eig_vec,
    get_renorm_mat,
    get_phi_vec,
    get_bond_vec,
    get_K_vec,
    get_occ_vec,
    get_rho_mat,
    get_residual
)

# ====================================================================
#   MODIFIED BROYDEN METHOD FOR CLUSTER-SITE APPROXIMATION 
# --------------------------------------------------------------------
#   In this script, we use an alternate iteration method for solving
#   for the parameters in the cluster-site approximation. Broyden is
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

MAX_ITER   = 1e8
ALPHA      = 5e-3
TOL_F      = 1e-9

WEIGHT     = 1e-2
NORM_FLOOR = 1e-10

def modified_broyden_step(f_vec, x_vec, alpha, weight, 
                          dF_hist, u_hist, x_prev, f_prev):
    """One limited-memory modified-Broyden step. The history lists dF_hist and u_hist accumulate 
    the (normalized) residual differences and the vectors u = alpha * dFhat + dhhat"""
    if x_prev is None:
        return alpha * f_vec
 
    dF = f_vec - f_prev
    norm = np.linalg.norm(dF)
    if norm < NORM_FLOOR:
        return alpha * f_vec
 
    dF_hat = dF / norm
    dh_hat = (x_vec - x_prev) / norm
    dF_hist.append(dF_hat)
    u_hist.append(alpha * dF_hat + dh_hat)
 
    m = len(dF_hist)
    DF = np.array(dF_hist)                        
    a = DF @ DF.T  # transition matrix <dFhat_k|dFhat_l>
    beta_mat = np.linalg.inv(weight**2 * np.eye(m) + a)
    c = DF @ f_vec                                 # c_k = <dFhat_k|f>
    gamma = beta_mat @ c
 
    return alpha * f_vec - np.array(u_hist).T @ gamma
 
def iterate_broyden(epsilon_vec, U: float, t_mat, h_vec_guess, phi_vec_guess, bond_vec_guess, 
                    beta: Union[float, None], l_max: int, N_sites: int, N_at: int, N_target: int, memory=8):
    """Calculate self-consistent parameters w/ modified Broyden's method"""    
    N_cluster = N_sites // 2

    dF_hist, u_hist = [], []
    x_prev = f_prev = None
 
    did_converge = False
    count = 1

    # x is of dimension 2N_{sites} + M where M is the number of clusters (N_{sites}/2)
    x = np.concatenate([np.asarray(h_vec_guess, float), np.asarray(phi_vec_guess, float), np.asarray(bond_vec_guess)])

    while not did_converge and count < MAX_ITER:
        h_vec, phi_vec, bond_vec = x[:N_sites], x[N_sites: 2*N_sites], x[2*N_sites: 2*N_sites+N_cluster]

        renorm_mat = get_renorm_mat(phi_vec, bond_vec, N_sites)

        # Spinon sector
        t_eff_mat = get_t_eff_mat(t_mat, renorm_mat)
        evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec))
        occ_vec, mu = get_occ_vec(evals_spinon, beta, N_sites, N_at, N_target)
        nf_vec = get_nf_vec(evecs_spinon, occ_vec, N_at)
        rho_mat = get_rho_mat(evecs_spinon, occ_vec)

        # Rotor sector
        J_eff_mat = get_J_eff_mat(t_mat, rho_mat, N_at)
        K_vec = get_K_vec(J_eff_mat, phi_vec, N_sites)
        evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, J_eff_mat, K_vec, h_vec, l_max, N_sites)
        L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max, N_sites)

        # Constraint & modified Broyden step
        f_vec = np.concatenate([get_residual(nf_vec, L_vec, N_at),
                                get_phi_vec(evals_rotor_vec, evecs_rotor_vec, l_max, beta, N_sites) - phi_vec,
                                get_bond_vec(evals_rotor_vec, evecs_rotor_vec, l_max, beta, N_sites) - bond_vec])
        
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
    
    L2_vec = get_L2_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max, N_sites)

    return (evals_spinon, evecs_spinon, evals_rotor_vec, evecs_rotor_vec, 
            h_vec, phi_vec, t_eff_mat, J_eff_mat, K_vec, 
            renorm_mat, bond_vec, nf_vec, L_vec, L2_vec, 
            occ_vec, rho_mat, mu, count)