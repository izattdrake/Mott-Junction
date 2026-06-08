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


def decoupled_naive_params(epsilon_vec, t_mat, U, t_eff_mat_guess, h_vec_guess, Z_vec_guess, K_vec_guess, beta, l_max, N_target):
    """
    Iterate over slave rotor calculations until parameters converge
    """
    MAX_ITER = 1000

    ALPHA_H = 0.1
    ALPHA_Z = 0.1
    ALPHA_K = 0.1
    ALPHA_T_EFF = 0.1
    ALPHA_MU = 0.1

    TOL_H = 1e-12
    TOL_T_EFF = 1e-12
    TOL_Z = 1e-12
    TOL_K = 1e-12
    TOL_N = 1e-6

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

        f = nf_vec - L_vec - N_AT / 2.0
        jac = get_Jacobian(epsilon_vec, U, t_eff_mat, h_vec, dh, mu, K_vec, beta, l_max)

        try:
            delta_h = -np.linalg.solve(jac,f)
        except np.linalg.LinAlgError:
            sys.exit('FAILURE: SINGULAR JACOBIAN')

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
            np.allclose(N, N_target, atol=TOL_N)
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

    return h_vec_update, Z_vec_update, K_vec_update, t_eff_mat, nf_vec, L_vec, count

if __name__ == '__main__':
    l_max = 40

    U_vals = np.linspace(0.0, 3.55, 100)
    N_target = 2
    beta = 50.0 

    t_nn = 1.0
    t_mat = np.diag(np.full(1, t_nn), k=1) + np.diag(np.full(1, t_nn), k=-1)

    epsilon_vec = np.zeros(2)
    t_eff_mat_guess = 0.8*t_mat
    h_vec_guess = -epsilon_vec
    Z_vec_guess = np.full(2, 0.5)
    K_vec_guess = np.full(2,-1.0)

    Z_data = []
    for U in U_vals:
        h_vec, Z_vec, K_vec, t_eff_mat, nf_vec, L_vec, count = decoupled_naive_params(
                epsilon_vec, t_mat, U, t_eff_mat_guess, h_vec_guess, 
                Z_vec_guess, K_vec_guess, beta, l_max, N_target=2.0
            )
        
        Z_data.append(Z_vec[0])

        """
        print(f'NEWTON (NAIVE)')
        print(f'U: {U}')
        print(f'beta: {beta}')
        print(f'epsilon: {epsilon_vec}')
        print(f'h: {h_vec}')
        print(f't_eff: {t_eff_mat[0,1]}')
        print(f'Z: {Z_vec}')
        print(f'K: {K_vec}')
        print(f'count: {count}')
        """

    plt.figure(figsize=(6, 4))
    plt.plot(U_vals, Z_data, 'o-', lw=1.5, ms=4)
    plt.xlabel(r"$U$")
    plt.ylabel(r"$Z = m/m^*$")
    plt.title("Quasiparticle weight")
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.show()
