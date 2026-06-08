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

def update_params(epsilon_vec, U, t, h_vec_guess, t_eff_guess, J_eff_guess, beta, l_max, N_target=2.0):
    """
    Iterate over parameters until convergence
    """
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
    dh = 1e-5

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

    # Calculate nf, L with final Hamiltonians before return
    spinon_evals, spinon_evecs = np.linalg.eigh(get_H_spinon(epsilon_vec, h_vec, t_eff, mu))
    rotor_evals, rotor_evecs = np.linalg.eigh(get_H_rotor(U, h_vec, J_eff, l_max))

    nf_vec = get_nf_vec(spinon_evals, spinon_evecs, beta)
    L_vec = get_L_vec(rotor_evals, rotor_evecs, beta, l_max)

    return h_vec_update, t_eff_update, J_eff_update, mu, nf_vec, L_vec, count

if __name__ == '__main__':
    l_max = 18 # Truncate rotor Hilbert space

    U_vals = [0.0, 0.5, 1.0]
    beta = 50.0 # beta=50.0 is zero temperature
    N_target = 2.0

    t = 1.0
    epsilon_vec = np.array([0.0,0.0])
    h_vec_guess = -epsilon_vec
    t_eff_guess = 0.95 * t
    J_eff_guess = 0.95 * t   

    for U in U_vals:
        h_vec, t_eff, J_eff, mu, nf_vec, L_vec, count = update_params(epsilon_vec, U, t, h_vec_guess, t_eff_guess, J_eff_guess, beta, l_max, N_target)

        print(f'NEWTON (NAIVE)')
        print(f'U: {U}')
        print(f'beta: {beta}')
        print(f'epsilon_vec: {epsilon_vec}')
        print(f'h_vec: {h_vec}')
        print(f't_eff: {t_eff}')
        print(f'J_eff: {J_eff}')
        print(f'nf: {nf_vec}')
        print(f'L: {L_vec}')
        print(f'mu: {mu}')
        print(f'count: {count}\n\n')
