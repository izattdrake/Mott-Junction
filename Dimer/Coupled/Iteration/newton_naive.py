import numpy as np
import sys 

from ..core import (
    get_H_spinon,
    get_t_eff,
    get_nf_vec,
    get_H_rotor,
    get_J_eff,
    get_L_vec,
    get_Jacobian
)

# ====================================================================
#   NAIVE NEWTON'S METHOD FOR SOLVING THE COUPLED ROTOR PROBLEM
# --------------------------------------------------------------------
#   Newton's method is used to calculate slave rotor parameters in the
#   coupled rotor dimer model. We calculate the Jacobian, a derivative
#   of the spinon occupancy minus the rotor charge, with the finite
#   difference method. The numpy.linalg library is used to solve the
#   matrix equation h' = h - J^{-1}f where f = n - L is the residual.
# --------------------------------------------------------------------
#   The method struggles at high values of U. When U ~ 2.5 and there's 
#   no doping, the Jacobian becomes singular and is non-invertible, 
#   breaking the matrix equation above. It turns out that problem
#   can be remedied by projecting out the Jacobian's nullspace which
#   is pure gauge. That is done in coupled_gauge.py. 
# ====================================================================

def iterate_newton_naive(epsilon_vec, U, t, h_vec_guess, t_eff_guess, J_eff_guess, beta, l_max, N_target=2.0):
    """Calculate self-consistent parameters w/ Newton's method w/o gauge fix"""
    MAX_ITER = 1000

    ALPHA_H = 0.1
    ALPHA_T_EFF = 0.1
    ALPHA_J_EFF = 0.1
    ALPHA_MU = 0.1

    TOL_H = 1e-10
    TOL_T_EFF = 1e-10
    TOL_J_EFF = 1e-10
    TOL_N  = 1e-6

    mu = 0.0 # Global chemical potential for total particle number
    dh = 1e-5 # Finite difference step

    did_converge = False
    count = 1

    # Initial values to guess for iteration
    h_vec_update, t_eff_update, J_eff_update = h_vec_guess, t_eff_guess, J_eff_guess

    while not did_converge and count < MAX_ITER:
        h_vec, t_eff, J_eff  = h_vec_update, t_eff_update, J_eff_update

        spinon_evals, spinon_evecs = np.linalg.eigh(get_H_spinon(epsilon_vec, h_vec, t_eff, mu))
        rotor_evals, rotor_evecs = np.linalg.eigh(get_H_rotor(U, h_vec, J_eff, l_max))
        
        nf_vec = get_nf_vec(spinon_evals, spinon_evecs, beta)
        L_vec = get_L_vec(rotor_evals, rotor_evecs, beta, l_max)
        
        f = nf_vec - L_vec - 1.0
        jacobian = get_Jacobian(U, epsilon_vec, h_vec, dh, t_eff, J_eff, beta, l_max, mu)
        
        try:
            delta_h = -np.linalg.solve(jacobian, f)
        except np.linalg.LinAlgError:
            print('ERROR: SINGULAR JACOBIAN')
            print(f'U: {U}')
            print(f'N: {N_target}')
            print(f'epsilon_vec: {epsilon_vec}')
            sys.exit()

        # Newton's method
        h_vec_update = h_vec + ALPHA_H * delta_h
        t_eff_update = (1 - ALPHA_T_EFF) * t_eff + ALPHA_T_EFF * get_t_eff(rotor_evals, rotor_evecs, t, beta, l_max)
        J_eff_update = (1 - ALPHA_J_EFF) * J_eff + ALPHA_J_EFF * get_J_eff(spinon_evals, spinon_evecs, t, beta)
 
        N = np.sum(nf_vec)
        mu += ALPHA_MU * (N_target - N)

        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_update, t_eff, atol=TOL_T_EFF) and
            np.allclose(J_eff_update, J_eff, atol=TOL_J_EFF) and
            np.allclose(N, N_target, atol=TOL_N)
        )

        count += 1

    if not did_converge:
        print(f'ERROR: DID NOT CONVERGE AFTER {MAX_ITER} ITERATIONS')
        sys.exit()

    spinon_evals, spinon_evecs = np.linalg.eigh(get_H_spinon(epsilon_vec, h_vec, t_eff, mu))
    rotor_evals, rotor_evecs = np.linalg.eigh(get_H_rotor(U, h_vec, J_eff, l_max))

    nf_vec = get_nf_vec(spinon_evals, spinon_evecs, beta)
    L_vec = get_L_vec(rotor_evals, rotor_evecs, beta, l_max)

    return spinon_evals, spinon_evecs, rotor_evals, rotor_evecs, h_vec_update, t_eff_update, J_eff_update, nf_vec, L_vec, mu, count