import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys 

import SR_Dimer.exact.core as core
import utility.plotting as SR_plotting
import utility.helpers as helpers
from utility.helpers import *

warnings.simplefilter("error", RuntimeWarning)

N_AT = 2

P2_dimer_states = {
    '1100': (1,1,0,0),
    '1010': (1,0,1,0),
    '1001': (1,0,0,1),
    '0110': (0,1,1,0),
    '0101': (0,1,0,1),
    '0011': (0,0,1,1)
}


def get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu):
    '''
        Return 2x2 Spinon Hamiltonian in a spin subspace
    '''
    H = np.diag(epsilon_vec - h_vec - mu) - t_eff_mat
    return H

def get_H_rotor(U, h_vec, K_vec, l_max, idx):
    '''
        Returns rotor Hamiltonian at site idx
    '''
    dim = 2*l_max + 1
    l_vals = np.arange(-l_max, l_max + 1)

    diag = ((U/2.0)*l_vals + h_vec[idx]) * l_vals 
    off_diag = K_vec[idx] / 2.0
    
    H = np.diag(diag)
    H += np.diag(np.full(dim - 1, off_diag), k=1) + np.diag(np.full(dim - 1, off_diag), k=-1)

    return H

def get_H_rotor_vec(U, h_vec, K_vec, l_max):
    '''
        Returns array of rotor Hamiltonians for each site
    '''
    dim = 2*l_max + 1
    H_vec = np.zeros((2, dim, dim))
    
    for i in range(2):
        H_vec[i] = get_H_rotor(U, h_vec, K_vec, l_max, i)

    return H_vec

def get_rotor_eig_vec(U, h_vec, K_vec, l_max):
    '''
        Return arrays containing the eigenvalues and eigenvectors of H_theta at every site
    '''
    dim = 2*l_max + 1
    eigvals_vec = np.zeros((2, dim))
    eigvecs_vec = np.zeros((2, dim, dim))

    for i in range(2):
        eigvals_vec[i], eigvecs_vec[i] = np.linalg.eigh(get_H_rotor(U, h_vec, K_vec, l_max, i))

    return eigvals_vec, eigvecs_vec

def get_nf_vec(spinon_energies, spinon_states, beta):
    '''
        Returns array of spinon occupancies at each dimer site
    '''
    nf = np.zeros(2)

    for i in range(2):
        sum = 0.0
        for j in range(2):
            energy = spinon_energies[j]
            sum += (np.abs(spinon_states[i,j]) ** 2) * fermi_dirac(energy, beta)

        nf[i] = N_AT * sum

    return nf

def get_L(rotor_eigvals, rotor_eigvecs, beta, l_max):
    '''
        Return thermal average of L
        Arguemnts are eigenvalues and eigenvectors at a site
    '''
    evals = rotor_eigvals - rotor_eigvecs[0]
    evecs = rotor_eigvecs

    bz = np.exp(-beta * evals)

    l_vals = np.arange(-l_max, l_max + 1)
    l_expect_per_state = np.sum((np.abs(evecs)**2) * l_vals[:,np.newaxis], axis=0)

    L = np.dot(bz, l_expect_per_state) / np.sum(bz)
    return L

def get_L_vec(rotor_evals_vec, rotor_evecs_vec, beta, l_max):
    '''
        Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
    '''
    L_vec = np.zeros(2)

    # Sum over lattice sites
    for i in range(2):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        evals = rotor_evals_vec[i] - rotor_evals_vec[i,0]
        eigvecs = rotor_evecs_vec[i]

        L_vec[i] = get_L(evals, eigvecs, beta, l_max)

    return L_vec

def get_Jacobian(epsilon_vec, U, t_eff_mat, h_vec, dh, mu, K_vec, beta, l_max):
    '''
        Returns Jacobian used in Newton's method
    '''
    def get_Jacobian_nf(epsilon_vec, t_eff_mat, h_vec, dh, mu, beta):
        '''
            Returns spinon contribution to Jacobian
        '''
        J = np.zeros((2,2))
        for j in range(2):
            h_p, h_m = h_vec.copy(), h_vec.copy()
            h_p[j] += dh/2.0
            h_m[j] -= dh/2.0

            evals_p, evecs_p = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_p, mu))
            evals_m, evecs_m = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_m, mu))

            nf_p = get_nf_vec(evals_p, evecs_p, beta)
            nf_m = get_nf_vec(evals_m, evecs_m, beta)

            J[:,j] = (nf_p - nf_m) / dh

        return J
    
    def get_Jacobian_L(U, h_vec, dh, K_vec, beta, l_max):
        '''
            Returns rotor contribution to Jacobian
        '''
        J = np.zeros((2,2))

        for j in range(2):
            h_p, h_m = h_vec.copy(), h_vec.copy()
            h_p[j] += dh/2
            h_m[j] -= dh/2
            
            # May need to calculate K_vec_p, K_vec_m
            evals_p, evecs_p = np.linalg.eigh(get_H_rotor(U, h_p, K_vec, l_max, j))
            evals_m, evecs_m = np.linalg.eigh(get_H_rotor(U, h_m, K_vec, l_max, j))
            
            L_p = get_L(evals_p, evecs_p, beta, l_max)
            L_m = get_L(evals_m, evecs_m, beta, l_max)

            J[j,j] = (L_p - L_m) / dh

        return J
    
    # Compute total Jacobian
    J_nf = get_Jacobian_nf(epsilon_vec, t_eff_mat, h_vec, dh, mu, beta)
    J_L = get_Jacobian_L(U, h_vec, dh, K_vec, beta, l_max)

    J = J_nf - J_L

    return J

def get_Z_vec(rotor_evals_vec, rotor_evecs_vec, beta, l_max):
    '''
    Returns array of Z values for each lattice site
    '''
    Z_vec = np.zeros(2)

    for i in range(2):
        evals = rotor_evals_vec[i] - rotor_evals_vec[i,0]
        evecs = rotor_evecs_vec[i]

        bz = np.exp(-beta * evals)
        mel = (evecs[:-1, :].conj() * evecs[1:, :]).sum(axis=0)
        Z_vec[i] = (np.dot(bz, mel) / np.sum(bz))**2

    return Z_vec

def get_t_eff(t_mat, Z_vec):
    '''
        Return effective tunneling amplitude matrix
    '''
    sqrt_Z = np.sqrt(Z_vec)
    t_eff = t_mat * np.outer(sqrt_Z, sqrt_Z)

    return t_eff

def get_K_vec(t_mat, Z_vec, spinon_evals, spinon_evecs, beta):
    '''
        Returns array of K values at each lattice site
        TODO: Might have problem with vectorization, don't think it does
    '''
    fd = np.array([helpers.fermi_dirac(energy, beta) for energy in spinon_evals])
    weighted = spinon_evecs * fd[np.newaxis,:]
    rho = spinon_evecs.conj() @ weighted.T
    sqrt_Z = np.sqrt(Z_vec)
    K_vec = np.real(-2 * N_AT * np.sum(t_mat * sqrt_Z[np.newaxis,:] * rho, axis=1))

    return K_vec

def rotor_state_ps(theta, state, l_max):
    '''
        Projects rotor state onto e^{i*l*theta} basis
    '''
    l_vals = np.arange(-l_max, l_max + 1)
    return state @ np.exp(1j * l_vals * theta)

'''
    Returns the next iteration of the Lagrange parameters using Newton's method
    The minimizing function is f = <n_f(i)> - N/2 - <L(i)>
    Newton's method sets h' = h - f/f'
    Derivatives are calculted with a finite difference method
    TODO: Find a closed form analytic solution using perturbation theory
    NOTE: Analytic solutions may not be possible due to recursion
'''
def update_params(epsilon_vec, t_mat, U, t_eff_mat_guess, h_vec_guess, Z_vec_guess, K_vec_guess, beta, l_max, N_target):
    # Small parameters to slow down updates
    ALPHA_H = 0.05
    ALPHA_Z = 0.05
    ALPHA_K = 0.05
    ALPHA_T_EFF = 0.05
    ALPHA_MU = 0.05

    # Tolerances for when iteration is done
    TOL_H = 1e-12
    TOL_T_EFF = 1e-12
    TOL_Z = 1e-12
    TOL_K = 1e-12
    TOL_N = 1e-6

    mu = 0.0
    dh = 1e-5
    did_converge = False
    num_cycles = 1

    h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update = h_vec_guess, Z_vec_guess, K_vec_guess, t_eff_mat_guess

    while not did_converge:
        h_vec, Z_vec, K_vec, t_eff_mat = h_vec_update, Z_vec_update, K_vec_update, t_eff_mat_update
        
        spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu))
        rotor_eigvals_vec, rotor_eigvecs_vec = get_rotor_eig_vec(U, h_vec, K_vec, l_max)

        # Calculate nf and L for Jacobian
        nf_vec = get_nf_vec(spinon_eigvals, spinon_eigvecs, beta)
        L_vec = get_L_vec(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max)

        f = nf_vec - L_vec - N_AT / 2.0
        jac = get_Jacobian(epsilon_vec, U, t_eff_mat, h_vec, dh, mu, K_vec, beta, l_max)

        try:
            delta_h = np.linalg.solve(jac,f)
        except np.linalg.LinAlgError:
            sys.exit('FAILURE: SINGULAR JACOBIAN')

        # Adjust total particle number
        N = np.sum(nf_vec)
        mu += ALPHA_MU * (N_target - N)

        # Update parameters
        h_vec_update = h_vec - ALPHA_H * delta_h
        Z_vec_update = (1.0 - ALPHA_Z) * Z_vec + ALPHA_Z * get_Z_vec(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max)
        K_vec_update = (1.0 - ALPHA_K) * K_vec + ALPHA_K * get_K_vec(t_mat, Z_vec, spinon_eigvals, spinon_eigvecs, beta)
        t_eff_mat_update = (1 - ALPHA_T_EFF) * t_eff_mat + ALPHA_T_EFF * get_t_eff(t_mat, Z_vec)

        # Test convergence of all parameters within selected bounds
        did_converge = (
            np.allclose(h_vec_update, h_vec, atol=TOL_H) and
            np.allclose(t_eff_mat_update, t_eff_mat, atol=TOL_T_EFF) and
            np.allclose(Z_vec_update, Z_vec, atol=TOL_Z) and
            np.allclose(K_vec_update, K_vec, atol=TOL_K) and
            np.allclose(N, N_target, atol=TOL_N)
        )

        num_cycles += 1

    # Calculate nf, L with final Hamiltonians before return
    spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat_update, h_vec_update, mu))
    rotor_eigvals_vec, rotor_eigvecs_vec = get_rotor_eig_vec(U, h_vec_update, K_vec_update, l_max)

    nf_vec = get_nf_vec(spinon_eigvals, spinon_eigvecs, beta)
    L_vec = get_L_vec(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max)

    return h_vec_update, Z_vec_update, K_vec_update, t_eff_mat, nf_vec, L_vec, num_cycles

def get_E_mel(epsilon_vec, U, t_nn, bra_state, ket_state):
    """
        Calcualte the matrix element <{n_i\sigma}|H|n'_{i\sigma}>
        between two basis kets in the spinon basis
        Rotor nums are constrained
    """
    n1_up, n1_down, n2_up, n2_down = bra_state
    nf1, nf2 = n1_up + n1_down, n2_up + n2_down

    n1_up_prime, n1_down_prime, n2_up_prime, n2_down_prime = ket_state
    nf1_prime, nf2_prime = n1_up_prime + n1_down_prime, n2_up_prime + n2_down_prime

    eps1, eps2 = epsilon_vec[0], epsilon_vec[1]

    diag = (
        (eps1*nf1 + eps2*nf2 + (U/2.0) * ((nf1 - 1)**2 + (nf2 - 1)**2))
          *helpers.k_delta(n1_up, n1_up_prime)*helpers.k_delta(n1_down, n1_down_prime)
          *helpers.k_delta(n2_up, n2_up_prime)*helpers.k_delta(n2_down, n2_down_prime)
    )


    off_diag_12 = -2*t_nn*(
        ((k_delta(0, n1_up_prime)*k_delta(1,n2_up_prime)
        *k_delta(n1_up, n1_up_prime+1)*k_delta(n1_down, n1_down_prime)*k_delta(n2_up,n2_up_prime-1)*k_delta(n2_down,n2_down_prime))
        +(k_delta(0,n1_down_prime)*k_delta(1,n2_up_prime)
        *k_delta(n1_up,n1_up_prime)*k_delta(n1_down,n1_down_prime+1)*k_delta(n2_up,n2_up_prime)*k_delta(n2_down,n2_down_prime-1)))
        *k_delta(nf1,nf1_prime+1)*k_delta(nf2,nf2_prime-1)
    )

    off_diag_21 = -2*t_nn*(
        ((k_delta(0, n2_up_prime)*k_delta(1,n1_up_prime)
        *k_delta(n2_up, n2_up_prime+1)*k_delta(n2_down, n2_down_prime)*k_delta(n1_up,n1_up_prime-1)*k_delta(n1_down,n1_down_prime))
        +(k_delta(0,n2_down_prime)*k_delta(1,n1_up_prime)
        *k_delta(n2_up,n2_up_prime)*k_delta(n2_down,n2_down_prime+1)*k_delta(n1_up,n1_up_prime)*k_delta(n1_down,n1_down_prime-1)))
        *k_delta(nf2,nf2_prime+1)*k_delta(nf1,nf1_prime-1)
    )

    mel = diag + off_diag_12 + off_diag_21
    return mel

def construct_phys_states(spinon, rotor1, rotor2, l_max):
    '''
        Construct physical states with combined spinon/rotor coefficients 
        Satisfy constraint in basis states
        c|n>|l1=n1-1>|l2=n2-1>
    '''

    # Arrays containing weights with spinon x rotor weights, and occupation numbers
    phys_weights = []
    phys_occ = []

    for i in range(len(spinon)):
        spinon_weight, spinon_state = spinon[i]
        spinon_nums = P2_dimer_states[spinon_state]
        
        nf1 = spinon_nums[0] + spinon_nums[1]
        nf2 = spinon_nums[2] + spinon_nums[3]

        l1 = l_max + nf1 - 1
        l2 = l_max + nf2 - 1

        phys_weight = spinon_weight * rotor1[l1] * rotor2[l2]
        phys_weights.append(phys_weight) 
        phys_occ.append(spinon_state)
    
    phys_weights = np.array(phys_weights)
    phys_weights /= np.linalg.norm(phys_weights)

    phys = []

    for i in range(len(spinon)):
        phys.append((phys_weights[i], phys_occ[i]))
    
    return phys

def get_E_dimer_phys(epsilon_vec, U, t_nn, rotor1, rotor2, l_max):
    """
        Returns 4 energies corresponding to the constrained tensor product |\psi_f>|\psi_{theta,1}>|\psi_{theta,2}>
        where |\psi_f> takes on 1 of the 4 calculated eigenkets for the spinon Hamiltonian
        This is highly specific to the homogeneous dimer problem
    """
    E5 = 0.5*(U - np.sqrt(16*t_nn**2 + U**2))
    d = E5/(2*t_nn)
    c1 = 1/np.sqrt(2 + 2 * d**2)
    c2 = d/c1
    #spinon1 = [(c1, '1100'), (c1, '0011'), (c2, '1001'), (c2,'0110')]

    spinon1 = [(0.5,'1100'), (0.5,'0110'), (0.5,'1001'), (0.5,'0011')]
    spinon2 = [(0.5,'1100'), (0.5,'0110'), (-0.5,'1001'), (-0.5,'0011')]
    spinon3 = [(0.5,'1100'), (-0.5,'0110'), (0.5,'1001'), (-0.5,'0011')]
    spinon4 = [(0.5,'1100'), (-0.5,'0110'), (-0.5,'1001'), (0.5,'0011')]

    phys1 = construct_phys_states(spinon1, rotor1, rotor2, l_max)
    phys2 = construct_phys_states(spinon2, rotor1, rotor2, l_max)
    phys3 = construct_phys_states(spinon3, rotor1, rotor2, l_max)
    phys4 = construct_phys_states(spinon4, rotor1, rotor2, l_max)

    phys_states = [phys1, phys2, phys3, phys4]
    phys_energies = [0.0 for _ in range(len(phys_states))]

    for i in range(len(phys_states)):
        phys_state = phys_states[i]
        E = 0.0

        for j in range(len(phys1)):
            bra_weight, bra_state = phys_state[j]
            bra_nums = P2_dimer_states[bra_state]

            for k in range(len(phys1)):
                ket_weight, ket_state = phys_state[k]
                ket_nums = P2_dimer_states[ket_state]

                mel = get_E_mel(epsilon_vec, U, t_nn, bra_nums, ket_nums)
                E += mel * np.conj(bra_weight) * ket_weight

        phys_energies[i] = E + U

    return phys_energies

if __name__ == '__main__':
    l_max = 40
    rotor_dim = 2 * l_max + 1

    N_target = 2
    beta = 50.0

    t_nn = 1.0
    t_mat = np.diag(np.full(1, t_nn), k=1) + np.diag(np.full(1, t_nn), k=-1)

    epsilon_vec = np.zeros(2)
    t_eff_mat_guess = 0.8*t_mat
    h_vec_guess = -epsilon_vec
    Z_vec_guess = np.full(2, 0.5)
    K_vec_guess = np.full(2,-1.0)

    energies_exact = [[] for _ in range(6)]
    energies_mf = [[] for _ in range(4)]

    U_vals = np.linspace(0.0,3.0,100)

    for U in U_vals:
        H_exact = core.get_H(epsilon_vec[0], U, t_nn, core.get_def_particle_basis(2))
        evals_exact, _ = np.linalg.eigh(H_exact)

        for i in range(len(energies_exact)):
            energies_exact[i].append(evals_exact[i])

        h_vec, Z_vec, K_vec, t_eff_mat, nf, L, num_cycles = update_params(epsilon_vec, t_mat, U, t_eff_mat_guess, h_vec_guess, Z_vec_guess, K_vec_guess, beta, l_max, N_target)
        rotor_evals_vec, rotor_evecs_vec = get_rotor_eig_vec(U, h_vec, K_vec, l_max)
        
        rotor1_evecs = rotor_evecs_vec[0]
        rotor2_evecs = rotor_evecs_vec[1]

        rotor1 = rotor1_evecs[:,0]
        rotor2 = rotor2_evecs[:,0]

        energies_mf_vec = get_E_dimer_phys(epsilon_vec, U, 2*t_nn, rotor1, rotor2, l_max)
        for i in range(len(energies_mf_vec)):
            energies_mf[i].append(energies_mf_vec[i])

    plt.figure()

    # Plot exact spectrum for each figure
    for i in range(len(energies_exact)):
        if i==0:
            plt.plot(U_vals, energies_exact[i], color='red', label='Exact')
        else:
            plt.plot(U_vals, energies_exact[i], color='red')

    # Plot MF spectrum
    for i in range(len(energies_mf)):
        if i == 0:
            plt.plot(U_vals, energies_mf[i], color='blue', label='MF')
        else:
            plt.plot(U_vals, energies_mf[i], color='blue')

    plt.xlabel(r'$U/t$')
    plt.ylabel(r'$E$')
    plt.title(r'Exact vs. MF Spectrum: $|\psi_\theta> = |\psi_{\theta,40}>\otimes|\psi_{\theta,40}>$, $\ell_{max}=80$')
    plt.grid(True)
    plt.legend()
    plt.show()

