import numpy as np
import matplotlib.pyplot as plt
import sys 

from core import (
    get_H_spinon,
    get_nf_vec,
    get_rotor_eig_vec,
    get_L_vec,
    get_Jacobian,
    get_Z_vec,
    get_K_vec,
    get_t_eff,
    N_AT
)

# ====================================================================
#   GAUGE-FIXED NEWTON'S METHOD FOR SOLVING THE COUPLED ROTOR PROBLEM
# --------------------------------------------------------------------
#   In this script, we modify Newton's method to handle singular
#   Jacobians. In the naive method, high values of U/t led to the 
#   Jacobian being singular and thus non-invertible, breaking the 
#   matrix equation for delta_h.
# --------------------------------------------------------------------
#   The singularity comes about from the gauge constraint. The total
#   charge is already handled by the global chemical potential, so
#   its information is redundant.
# --------------------------------------------------------------------
#   We can easily implement this method by removing the gauge vector
#   from the vector space at every iteration in Newton's method. We
#   construct a projection operator P = I - u^Tu where
#   u = (1,1,...,1)/sqrt(N), act on the Jacobian and residual, and
#   solve the matrix equation. There is no effect on observables and
#   arbitrarily parameters converge for arbitrarily high values of 
#   U/t. 
# ====================================================================

def gauge_fixed_newton_step(jacobian, f):
    """
    Constructs a Newton step outside the null/gauge subspace
    
    1. Assume the gauge direction is along the total charge vector (1,1,...,1)
    2. Construct a projection operator P = I - u^Tu
    3. Project the null space out of f, J
    4. Solve the matrix equation for delta_h with pseudo inverse
    """
    N = f.size
 
    # Gauge direction along total charge (1,1,...,1) | Projection operator
    u = np.ones(N) / np.sqrt(N)
    P = np.eye(N) - np.outer(u, u)
 
    # Restrict residual and Jacobian to the physical subspace.
    f_phys = P @ f
    J_phys = P @ jacobian @ P
 
    # Null direction is now zero, solve matrix equation with pseudo-inverse
    delta_h = -np.linalg.pinv(J_phys) @ f_phys
    
    return delta_h

def gauge_projected_residual(f):
    """
    Norm of the residual with the gauge component removed.
    This is the quantity that should actually go to zero.
    The uniform part of f is enforced by mu, not h.
    """
    N = f.size
    u = np.ones(N) / np.sqrt(N)

    return np.linalg.norm(f - u * (u @ f))

def decoupled_gauge_params(epsilon_vec, t_mat, U, t_eff_mat_guess, h_vec_guess, Z_vec_guess, K_vec_guess, beta, l_max, N_target):
    """
    Iterate over slave rotor calculations until parameters converge
    """
    MAX_ITER = 1000000

    ALPHA_H = 1e-3
    ALPHA_Z = 1e-3
    ALPHA_K = 1e-3
    ALPHA_T_EFF = 1e-3
    ALPHA_MU = 1e-3

    TOL_H = 1e-30
    TOL_T_EFF = 1e-30
    TOL_Z = 1e-30
    TOL_K = 1e-30
    TOL_N = 1e-30
    TOL_PHYS_F = 1e-30

    mu = 0.0 # Global chemical potential
    dh = 1e-5 # Finite difference step

    did_converge = False
    count = 1

    h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update = h_vec_guess, Z_vec_guess, K_vec_guess, t_eff_mat_guess

    while not did_converge and count < MAX_ITER:
        h_vec, Z_vec, K_vec, t_eff_mat = h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update
        
        spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu))
        rotor_eigvals_vec, rotor_eigvecs_vec = get_rotor_eig_vec(U, h_vec, K_vec, l_max)

        # Calculate nf and L for Jacobian
        nf_vec = get_nf_vec(spinon_eigvals, spinon_eigvecs, beta)
        L_vec = get_L_vec(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max)

        f_vec = nf_vec - L_vec - N_AT / 2.0
        phys_f_norm = gauge_projected_residual(f_vec) # Residual norm that must vanish
        jacobian = get_Jacobian(epsilon_vec, U, t_eff_mat, h_vec, dh, mu, K_vec, beta, l_max)

        try:
            delta_h = gauge_fixed_newton_step(jacobian, f_vec)
        except np.linalg.LinAlgError:
            print('ERROR: SINGULAR JACOBIAN')
            print(f'U: {U}')
            print(f'beta: {beta}')
            print(f'N: {N_target}')
            print(f'epsilon_vec: {epsilon_vec}')
            sys.exit()

        # Update parameters
        h_vec_update = h_vec + ALPHA_H * delta_h
        Z_vec_update = (1.0 - ALPHA_Z) * Z_vec + ALPHA_Z * get_Z_vec(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max)
        K_vec_update = (1.0 - ALPHA_K) * K_vec + ALPHA_K * get_K_vec(t_mat, Z_vec, spinon_eigvals, spinon_eigvecs, beta)
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
            phys_f_norm < TOL_PHYS_F
        )

        count += 1

    if not did_converge:
        print(f'ERROR: DID NOT CONVERGE AFTER {MAX_ITER} ITERATIONS')
        sys.exit()

    # Calculate nf, L with final Hamiltonians before return
    spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat_update, h_vec_update, mu))
    rotor_eigvals_vec, rotor_eigvecs_vec = get_rotor_eig_vec(U, h_vec_update, K_vec_update, l_max)

    nf_vec = get_nf_vec(spinon_eigvals, spinon_eigvecs, beta)
    L_vec = get_L_vec(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max)

    return h_vec_update, Z_vec_update, K_vec_update, t_eff_mat, nf_vec, L_vec, mu, count
