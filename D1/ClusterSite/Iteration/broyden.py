import numpy as np
import sys

from typing import Union
from ..core import (
    get_H_spinon,
    get_t_eff,
    get_nf_vec,
    get_J_eff_mat,
    get_L_vec,
    get_rotor_eig_vec,
    get_renorm_mat,
    get_phi_vec,
    get_bond_vec,
    get_K_vec
)

MAX_ITER    = 1e7

ALPHA_H     = 1e-3
ALPHA_T_EFF = 2e-2
ALPHA_J_EFF = 2e-2
ALPHA_K     = 2e-2
ALPHA_MU    = 2e-2

TOL_H       = 1e-4
TOL_T_EFF   = 1e-4
TOL_J_EFF   = 1e-4
TOL_K       = 1e-4
TOL_N       = 1e-4
TOL_F       = 1e-4

# Modified-Broyden control parameters
WEIGHT      = 1e-2
MEMORY      = 8

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
 
def iterate_broyden(epsilon_vec, U: float, t_mat, h_vec_guess, t_eff_mat_guess, J_eff_mat_guess, K_vec_guess, 
                    beta: Union[float, None], l_max: int, N_sites: int, N_at: int, N_target: int):
    """Calculate self-consistent parameters w/ modified Broyden's method"""
    mu = 0.0 # Global chemical potential 
    
    # Modified-Broyden history for the constraint residual f = n - <L> - 1.
    dF_hist = []
    u_hist = []
    h_prev = None
    f_prev = None
 
    did_converge = False
    count = 1

    h_vec_update, t_eff_mat_update, J_eff_mat_update, K_vec_update = h_vec_guess, t_eff_mat_guess, J_eff_mat_guess, K_vec_guess

    while not did_converge and count < MAX_ITER:
        h_vec, t_eff_mat, J_eff_mat, K_vec  = h_vec_update, t_eff_mat_update, J_eff_mat_update, K_vec_update
 
        evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu))
        evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, J_eff_mat, K_vec, h_vec, l_max, N_sites)
        
        nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta, N_at)
        L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max, N_sites)
 
        f_vec = nf_vec - L_vec -  N_at / 2.0
        delta_h = modified_broyden_step(f_vec, h_vec, ALPHA_H, WEIGHT,
                                        dF_hist, u_hist, h_prev, f_prev)
        if len(dF_hist) > MEMORY:
            dF_hist.pop(0)
            u_hist.pop(0)
 
        h_prev = h_vec.copy()
        f_prev = f_vec.copy()
        h_vec_update = h_vec + delta_h
 
        """Construct correlation/renormalization functions"""
        phi_vec = get_phi_vec(evals_rotor_vec, evecs_rotor_vec, l_max, beta, N_sites)
        bond_vec = get_bond_vec(evals_rotor_vec, evecs_rotor_vec, l_max, beta, N_sites)
        renorm_mat = get_renorm_mat(phi_vec, bond_vec, N_sites)

        """Usual update functions using renormalization functions"""
        t_eff_mat_update = (1 - ALPHA_T_EFF) * t_eff_mat + ALPHA_T_EFF * get_t_eff(t_mat, renorm_mat)
        J_eff_mat_update = (1 - ALPHA_J_EFF) * J_eff_mat + ALPHA_J_EFF * get_J_eff_mat(evals_spinon, evecs_spinon, t_mat, beta, N_at)
        K_vec_update     = (1 - ALPHA_K)     * K_vec     + ALPHA_K     * get_K_vec(J_eff_mat, phi_vec, N_sites)

        N = np.sum(nf_vec)
        mu += ALPHA_MU * (N_target - N)
 
        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_mat_update, t_eff_mat, atol=TOL_T_EFF) and
            np.allclose(J_eff_mat_update, J_eff_mat, atol=TOL_J_EFF) and
            np.allclose(K_vec_update, K_vec, atol=TOL_K) and
            np.allclose(N, N_target, atol=TOL_N) and
            np.linalg.norm(f_vec) < TOL_F
        )
        
        did_converge = (
            np.allclose(t_eff_mat_update, t_eff_mat, atol=TOL_T_EFF) and
            np.allclose(J_eff_mat_update, J_eff_mat, atol=TOL_J_EFF) and
            np.allclose(K_vec_update, K_vec, atol=TOL_K) and
            np.allclose(N, N_target, atol=TOL_N) 
        )

        print(count); count += 1
        print(h_vec_update)
        print(f"h: {np.allclose(h_vec_update, h_vec, atol=TOL_H)}")
        print(f"t_eff: {np.allclose(t_eff_mat_update, t_eff_mat, atol=TOL_T_EFF)}")
        print(f"J_eff: {np.allclose(J_eff_mat_update, J_eff_mat, atol=TOL_J_EFF)}")
        print(f"K_vec: {np.allclose(K_vec_update, K_vec, atol=TOL_K)}")
        print(f"N: {np.allclose(N, N_target, atol=TOL_N)}")
        print(f"f: {np.linalg.norm(f_vec) < TOL_F}\n\n")

    if not did_converge:
        print(f'ERROR: Did not converge after {MAX_ITER} iterations')
        sys.exit()
    
    evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu))
    evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, J_eff_mat, K_vec, h_vec, l_max, N_sites)
 
    phi_vec = get_phi_vec(evals_rotor_vec, evecs_rotor_vec, l_max, beta, N_sites)
    bond_vec = get_bond_vec(evals_rotor_vec, evecs_rotor_vec, l_max, beta, N_sites)
    renorm_mat = get_renorm_mat(phi_vec, bond_vec, N_sites)

    nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta, N_at)
    L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max, N_sites)

    return evals_spinon, evecs_spinon, evals_rotor_vec, evecs_rotor_vec, h_vec_update, t_eff_mat_update, J_eff_mat_update, K_vec_update, renorm_mat, phi_vec, nf_vec, L_vec, mu, count