import numpy as np
import matplotlib.pyplot as plt
import sys 

from ..core import (
    get_H_spinon,
    get_nf_vec,
    get_rotor_eig_vec,
    get_L_vec,
    get_Jacobian,
    get_Z_vec,
    get_K_vec,
    get_t_eff
)

# ====================================================================
#   NAIVE NEWTON'S METHOD FOR DECOUPLED ROTOR ON HUBBARD DIMER
# --------------------------------------------------------------------
#   Newton's method is used to calculate slave rotor parameters in the
#   decoupled rotor dimer model. We calculate the Jacobian, a derivative
#   of the spinon occupancy minus the rotor charge, with the finite
#   difference method. The numpy.linalg library is used to solve the
#   matrix equation h' = h - J^{-1}f where f = n - L is the residual.
# --------------------------------------------------------------------
#   The method struggles at high values of U. When U ~ 2.5 and there's 
#   no doping, the Jacobian becomes singular and is non-invertible, 
#   breaking the matrix equation above. It turns out that problem
#   can be remedied by projecting out the Jacobian's nullspace which
#   is pure gauge. That is done in coupled_gauge.py. 
# --------------------------------------------------------------------
#   In the decoupled rotor problem, we ignore off-diagonal rotor
#   contributions to the Jacobian. The parameters still converge onto
#   the same values as those contributions are expected to vanish.
# ====================================================================

def iterate_newton_naive(epsilon_vec, t_mat, U, t_eff_mat_guess, h_vec_guess, Z_vec_guess, K_vec_guess, beta, l_max, N_target=2, N_at=2):
    """Calculate self-consistnet parameters w/ Newton's method w/o gauge fix"""
    MAX_ITER = 1e6

    ALPHA_H = 1e-1
    ALPHA_Z = 1e-1
    ALPHA_K = 1e-1
    ALPHA_T_EFF = 1e-1
    ALPHA_MU = 1e-1

    TOL_H = 1e-10
    TOL_T_EFF = 1e-10
    TOL_Z = 1e-10
    TOL_K = 1e-10
    TOL_N = 1e-8

    mu = 0.0 # Global chemical potential
    dh = 1e-5 # Finite difference step

    did_converge = False
    count = 1

    h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update = h_vec_guess, Z_vec_guess, K_vec_guess, t_eff_mat_guess

    while not did_converge and count < MAX_ITER:
        h_vec, Z_vec, K_vec, t_eff_mat = h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update
        
        evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu))
        evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, h_vec, K_vec, l_max)

        # Calculate nf and L for Jacobian
        nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta)
        L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

        f = nf_vec - L_vec - N_at / 2.0
        jac = get_Jacobian(epsilon_vec, U, t_eff_mat, h_vec, dh, mu, K_vec, beta, l_max)

        try:
            delta_h = -np.linalg.solve(jac,f)
        except np.linalg.LinAlgError:
            sys.exit('FAILURE: SINGULAR JACOBIAN')

        # Update parameters
        h_vec_update = h_vec + ALPHA_H * delta_h 
        Z_vec_update = (1.0 - ALPHA_Z) * Z_vec + ALPHA_Z * get_Z_vec(evals_rotor_vec, evecs_rotor_vec, beta)
        K_vec_update = (1.0 - ALPHA_K) * K_vec + ALPHA_K * get_K_vec(t_mat, Z_vec, evals_spinon, evecs_spinon, beta)
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
            np.allclose(N, N_target, atol=TOL_N)
        )

        count += 1

    if not did_converge:
        print(f'ERROR: DID NOT CONVERGE AFTER {MAX_ITER} ITERATIONS')
        sys.exit()

    # Calculate nf, L with final Hamiltonians before return
    evals_spinon, evecs_spinon = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat_update, h_vec_update, mu))
    evals_rotor_vec, evecs_rotor_vec = get_rotor_eig_vec(U, h_vec_update, K_vec_update, l_max)

    nf_vec = get_nf_vec(evals_spinon, evecs_spinon, beta)
    L_vec = get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta, l_max)

    return evals_spinon, evecs_spinon, evals_rotor_vec, evecs_rotor_vec, h_vec_update, Z_vec_update, K_vec_update, t_eff_mat, nf_vec, L_vec, mu, count