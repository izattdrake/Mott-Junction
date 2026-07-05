import numpy as np
import sys 

from Dimer.Coupled.core import (
    get_H_spinon,
    get_t_eff,
    get_nf_vec,
    get_H_rotor,
    get_J_eff,
    get_L_vec,
    get_Jacobian
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
#   its information is redundant. I.e. if the residual is f = [f1, f2],
#   then f1 + f2 = n1 + n2 + L1 + L2 - N_at = 2N - 2N_{at} is fixed.
#   Only f1 - f2 must be iterated over. 
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
 
def coupled_gauge_params(epsilon_vec, U, t, h_vec_guess, t_eff_guess, J_eff_guess, beta, l_max, N_target=2.0):
    """
        Iterate over parameters until convergence
    """
    MAX_ITER = 1000
 
    ALPHA_H = 0.1
    ALPHA_T_EFF = 0.1
    ALPHA_J_EFF = 0.1
    ALPHA_MU = 0.1
 
    TOL_H = 1e-8
    TOL_T_EFF = 1e-8
    TOL_J_EFF = 1e-8
    TOL_N  = 1e-6
    TOL_PHYS_F = 1e-8
 
    mu = 0.0 # Global chemical potential
    dh = 1e-5 # Finite difference step

    did_converge = False
    count = 1

    h_vec_update, t_eff_update, J_eff_update = h_vec_guess, t_eff_guess, J_eff_guess
 
    while not did_converge and count < MAX_ITER:
        h_vec, t_eff, J_eff  = h_vec_update, t_eff_update, J_eff_update
 
        spinon_evals, spinon_evecs = np.linalg.eigh(get_H_spinon(epsilon_vec, h_vec, t_eff, mu))
        rotor_evals, rotor_evecs = np.linalg.eigh(get_H_rotor(U, h_vec, J_eff, l_max))
 
        nf_vec = get_nf_vec(spinon_evals, spinon_evecs, beta)
        L_vec = get_L_vec(rotor_evals, rotor_evecs, beta, l_max)
 
        f_vec = nf_vec - L_vec - 1.0
        phys_f_norm = gauge_projected_residual(f_vec) # Residual that must vanish | Equals |f1 - f2| for the dimer.
        jacobian = get_Jacobian(U, epsilon_vec, h_vec, dh, t_eff, J_eff, beta, l_max, mu)
 
        # Gauge-fixed Newton's method | h1 + h2 held fixed
        try:
            delta_h = gauge_fixed_newton_step(jacobian, f_vec)
        except np.linalg.LinAlgError:
            print('ERROR: SINGULAR JACOBIAN')
            print(f'U: {U}')
            print(f'beta: {beta}')
            print(f'N: {N_target}')
            print(f'epsilon_vec: {epsilon_vec}')
            sys.exit()

 
        h_vec_update = h_vec + ALPHA_H * delta_h
        t_eff_update = (1 - ALPHA_T_EFF) * t_eff + ALPHA_T_EFF * get_t_eff(rotor_evals, rotor_evecs, t, beta, l_max)
        J_eff_update = (1 - ALPHA_J_EFF) * J_eff + ALPHA_J_EFF * get_J_eff(spinon_evals, spinon_evecs, t, beta)
 
        N = np.sum(nf_vec)
        mu += ALPHA_MU * (N_target - N)
 
        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_update, t_eff, atol=TOL_T_EFF) and
            np.allclose(J_eff_update, J_eff, atol=TOL_J_EFF) and
            np.allclose(N, N_target, atol=TOL_N) and
            phys_f_norm < TOL_PHYS_F
        )
 
        count += 1
 
    if not did_converge:
        print(f'ERROR: DID NOT CONVERGE AFTER {MAX_ITER} ITERATIONS')
        sys.exit()
 
    spinon_evals, spinon_evecs = np.linalg.eigh(get_H_spinon(epsilon_vec, h_vec, t_eff, mu))
    rotor_evals, rotor_evecs = np.linalg.eigh(get_H_rotor(U, h_vec, J_eff, l_max))
 
    nf_vec = get_nf_vec(spinon_evals, spinon_evecs, beta)
    L_vec = get_L_vec(rotor_evals, rotor_evecs, beta, l_max)
 
    return h_vec_update, t_eff_update, J_eff_update, nf_vec, L_vec, mu, count
 
