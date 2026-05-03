import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 
import scipy 
import math

import utility.helpers as helpers
from utility.helpers import k_delta
import utility.plotting as plotting
import exact_dimer as exact

warnings.simplefilter("error", RuntimeWarning)
np.set_printoptions(threshold=sys.maxsize)

"""
    Simulation of a junction with inhomogenous chemical potential with next-nearest-neighbor hopping

"""
# Dimension of the rotor Hilbert space
# Going to make it variable, increasing towards the junction to reduce the computational complexity
N_AT = 2

"""
Return real-space spinon Hamiltonian in the upper-block
The single-particle spinon Hamiltonian is block diagonalized with each block corresponding to a spin
I think we only have to diagonalize a single one of these blocks since the energies are degenerate, but I am not sure
"""
def get_H_spinon(epsilon, t_eff, h, mu):
    H = np.diag(epsilon - h - mu) - t_eff

    return H

"""
    Returns number occupancy at each site
"""
def get_nf_vec(spinon_eigvals, spinon_eigvecs, beta, sites):
    fd = np.array([helpers.fermi_dirac(energy, beta) for energy in spinon_eigvals])
    nf = N_AT * np.sum((np.abs(spinon_eigvecs)**2) * fd[np.newaxis,:], axis=1)

    return nf

"""
    Returns L's thermal expectation value
"""
def get_L(rotor_eigvals, rotor_eigvecs, beta, l_max):
    eigvals = rotor_eigvals - rotor_eigvals[0]
    eigvecs = rotor_eigvecs

    bz = np.exp(-beta * eigvals)

    l_vals = np.arange(-l_max, l_max+1)
    l_expect_per_state = np.sum((np.abs(eigvecs) ** 2) * l_vals[:,np.newaxis], axis=0)

    L = np.dot(bz, l_expect_per_state) / np.sum(bz)

    return L

"""
    Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
"""
def get_L_vec(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max, sites):
    # Dimension of rotor Hilbet space as a function of the cutoff (formally infinite)
    L_vec = np.zeros(sites)

    # Sum over lattice sites
    for i in range(sites):
        rotor_eigvals = rotor_eigvals_vec[i]
        rotor_eigvec = rotor_eigvecs_vec[i]

        L_vec[i] = get_L(rotor_eigvals, rotor_eigvec, beta, l_max)

    return L_vec


"""
    Return full Jacobian used in Newton's method
"""
def get_J(epsilon, U, t_eff, h, dh, mu, K, beta, l_max, sites):
    def get_J_nf(epsilon, t_eff, h, dh, mu, beta, sites):
        """
        Returns matrix contribution dn_i/dh_j to Jacobian in Newton's method
        """
        J = np.zeros((sites, sites))
        for j in range(sites):
            h_p, h_m = h.copy(), h.copy()
            h_p[j] += dh/2.0
            h_m[j] -= dh/2.0

            eigvals_p, eigvecs_p = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h_p, mu))
            eigvals_m, eigvecs_m = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h_m, mu))

            nf_p = get_nf_vec(eigvals_p, eigvecs_p, beta, sites)
            nf_m = get_nf_vec(eigvals_m, eigvecs_m, beta, sites)

            J[:,j] = (nf_p - nf_m) / dh

        return J

    def get_J_L(U, h, dh, K, beta, l_max, sites):
        """
            Return L's contribution to Jacobian
            Off-diagonal derivatives are NOT included
            It is expected that the spinon sector will take care of those
            and their contribution will bleed through to the rotor sector
            for successive iterations
        """
        J = np.zeros((sites, sites))

        for j in range(sites):
            h_p, h_m = h.copy(), h.copy()
            h_p[j] += dh/2
            h_m[j] -= dh/2

            eigvals_p, eigvecs_p = np.linalg.eigh(get_H_rotor(U, h_p, K, l_max, j))
            eigvals_m, eigvecs_m = np.linalg.eigh(get_H_rotor(U, h_m, K, l_max, j))

            L_p = get_L(eigvals_p, eigvecs_p, beta, l_max)
            L_m = get_L(eigvals_m, eigvecs_m, beta, l_max)

            J[j,j] = (L_p - L_m) / dh

        return J

    J_nf = get_J_nf(epsilon, t_eff, h, dh, mu, beta, sites)
    J_L = get_J_L(U, h, dh, K, beta, l_max, sites)

    J = J_nf - J_L

    return J


"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max, sites):
    Z = np.zeros(sites)

    for i in range(sites):
        eigvals = rotor_eigvals_vec[i] - rotor_eigvals_vec[i,0]
        eigvecs = rotor_eigvecs_vec[i]

        bz = np.exp(-beta * eigvals)
        mel = (eigvecs[:-1, :].conj() * eigvecs[1:, :]).sum(axis=0)
        Z[i] = (np.dot(bz, mel) / np.sum(bz))**2

    return Z 

"""
    Return effective tunneling amplitude matrix
    t is treated as a vector 
"""
def get_t_eff(t, Z):
    sqrt_Z = np.sqrt(Z)
    t_eff = t * np.outer(sqrt_Z, sqrt_Z)
    return t_eff

"""
    Returns a single rotor Hamiltonian at a site j
"""
def get_H_rotor(U, h, K, l_max, site_idx):
    rotor_dim = 2 * l_max + 1

    H = np.zeros((rotor_dim, rotor_dim))
    l_vals = np.arange(-l_max, l_max + 1)

    diag = ((U / 2.0) * l_vals + h[site_idx]) * l_vals 
    np.fill_diagonal(H, diag)

    off_diag = K[site_idx] / 2.0
    H += np.diag(np.full(rotor_dim - 1, off_diag), k=1) + np.diag(np.full(rotor_dim - 1, off_diag), k=-1)

    return H

"""
    Return array of rotor Hamiltonians at each site
"""
def get_H_rotor_vec(U, h, K, l_max, sites):
    H_vec = [get_H_rotor(U, h, K, l_max, i) for i in range(sites)]
    return H_vec

"""
    Returns array of rotor Hamiltonian eigenvalues and eigenvectors
"""
def get_rotor_eig_vec(U, h, K, l_max, sites):
    rotor_dim = 2 * l_max + 1
    eigvals_vec = np.zeros((sites, rotor_dim))
    eigvecs_vec = np.zeros((sites, rotor_dim, rotor_dim))

    for i in range(sites):
        eigvals_vec[i], eigvecs_vec[i] = np.linalg.eigh(get_H_rotor(U, h, K, l_max, i))

    return eigvals_vec, eigvecs_vec

"""
Return K
"""
def get_K(t, Z, spinon_eigvals, spinon_eigvecs, beta, sites):
    fd = np.array([helpers.fermi_dirac(energy, beta) for energy in spinon_eigvals])
    weighted = spinon_eigvecs * fd[np.newaxis,:]
    rho = spinon_eigvecs.conj() @ weighted.T
    sqrt_Z = np.sqrt(Z)
    K = np.real(-2 * N_AT * np.sum(t * sqrt_Z[np.newaxis,:] * rho, axis=1))

    return K

def rotor_state_ps(theta, state, l_max):
    rotor_dim = 2 * l_max + 1

    sum = 0.0j
    for i in range(rotor_dim):
        l = i - l_max
        weight = state[i]
        sum += weight * np.exp(l * theta * 1.0j)

    return sum 

"""
    Calculate spinon Green's function for periodic dimer
"""
def spinon_Green(omega: float, k: float, epsilon: float, t_eff: float, h: float, mu: float, eta=1e-4) -> np.complex64:
    energy_k = -2 * t_eff * np.cos(k) + (epsilon - mu - h)
    Green_val = 1/(omega - energy_k + eta*1j)

    return Green_val 

"""
    For homogenous, periodic, translationally invariant systems
"""
def get_spinon_spectrum(omega_vals, epsilon: float, t_eff: float, h: float, mu: float, sites: int):
    def get_spectral_val(omega: float, epsilon: float, t_eff: float, h: float, mu: float, sites: int, a=1.0, eta=1e-4):
        epsilon_val, h_val = epsilon[0], h[0]
        
        # hack this for larger periodic systems 
        t_eff_nn = t_eff[0,1]
                    
        spectral_val = 0.0

        # Assume translational symmetry, sum over k = 0, pi/a
        for n in range(sites):
            k = (2*np.pi * n/(sites * a))
            energy_k = epsilon_val - mu - h_val - 2 * t_eff_nn * np.cos(k * a) 
            spectral_val += helpers.d_delta(omega - energy_k)

        return spectral_val

    spectrum_norm = np.array([get_spectral_val(omega, epsilon, t_eff, h, mu, sites) for omega in omega_vals]) / sites

    return spectrum_norm

def get_rotor_spectrum(omega_vals, U: float, h, K, beta: float, l_max: int, overwrite=False):
    H_rotor = get_H_rotor(U, h, K, l_max, 0)
    eig_vals, eig_vecs = np.linalg.eigh(H_rotor)

    eig_vals = eig_vals - eig_vals[0]

    bz = np.exp(-beta * eig_vals)          
    partition = bz.sum()

    mel = np.conj(eig_vecs[1:, :]).T @ eig_vecs[:-1, :]   
    mel = mel.T                                             

    bz_diff = bz[:, None] - bz[None, :]                  
    weights = bz_diff * np.abs(mel) ** 2                   

    energy_diff = eig_vals[:, None] - eig_vals[None, :]    

    args = omega_vals[:, None, None] + energy_diff[None, :, :] 
    delta_vals = helpers.d_delta(args)                          

    spectrum = np.einsum('ij,kij->k', weights, delta_vals) / partition

    # spectrum /= scipy.integrate.trapezoid(spectrum, omega_vals)
    return spectrum

# Returns the spinon, rotor, and electron spectral functions for all values of omega using Scipy's FFT convolution method
def get_spectra(
        omega_vals: float, epsilon: float, U: float, t_eff: float, h: float, mu: float, K, beta, l_max, sites, N_target,
        plot_spinon_spectrum, plot_rotor_spectrum, plot_electron_spectrum, a=1.0, eta=1e-4
    ):
    domain_dim = len(omega_vals)
    half_domain = int(len(omega_vals)/2.0)
    domega = np.abs(omega_vals[1] - omega_vals[0])
    
    # Super weird with whether I'm passing scalar or vector values for some parameters. Should clean up soon
    spinon_spectrum_vals = get_spinon_spectrum(omega_vals, epsilon, t_eff, h, mu, sites)
    spinon_peaks = helpers.get_peaks_idx(spinon_spectrum_vals)
    # print(f'Spinon spectrum integral: {scipy.integrate.trapezoid(spinon_spectrum_vals, omega_vals)}') # Check spinon norm
    # print(f'spinon peaks: {[omega_vals[peak] for peak in spinon_peaks]}') # Check peaks in spinon
 
    rotor_spectrum_vals =  get_rotor_spectrum(omega_vals, U, h, K, beta, l_max)
    rotor_peaks = helpers.get_peaks_idx(rotor_spectrum_vals)
    # print(f'Rotor spectrum integral: {scipy.integrate.trapezoid(rotor_spectrum_vals, omega_vals)}') # Check rotor norm
    # print(f'Rotor peaks: {[omega_vals[peak] for peak in rotor_peaks]}')
    """
    Manually play with rotors spectrum

    rotor_spectrum_vals = np.zeros(len(omega_vals))
    rotor_spectrum_vals[half_domain] = 500
    rotor_spectrum_vals[half_domain - 1] = 1000
    """

    if plot_spinon_spectrum:
        plt.plot(omega_vals, spinon_spectrum_vals)
        plt.xticks(
            [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10],
            ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8' ,'9', '10'],
        )

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\rho_f(\omega)$')
        plt.title(r'Spinon spectrum for $U= $ ' + f'{U}, ' + r'$\beta =$ ' + f'{beta}, ' + r'$N = $' + f'{N_target}')
        plt.tight_layout()

        plt.show()
    
    if plot_rotor_spectrum:
        plt.plot(omega_vals, rotor_spectrum_vals)

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\rho_f(\omega)$')
        plt.title(r'Rotor spectram for $U= $ ' + f'{U}, ' + r'$\beta =$ ' + f'{beta}, ' + r'$N = $' + f'{N_target}')
        plt.tight_layout()

        plt.show()
    
    fd_vals = np.array([helpers.fermi_dirac(omega, beta) for omega in omega_vals])
    be_vals = np.array([helpers.bose_einstein(omega, beta) for omega in omega_vals])

    spectral_1 = scipy.signal.fftconvolve(
        spinon_spectrum_vals,
        rotor_spectrum_vals,
        mode='same'
    ) * domega

    spectral_fd = scipy.signal.fftconvolve(
        -spinon_spectrum_vals * fd_vals,
        rotor_spectrum_vals,
        mode='same'
    ) * domega

    spectral_be = scipy.signal.fftconvolve(
        spinon_spectrum_vals,
        rotor_spectrum_vals * be_vals,
        mode='same'
    ) * domega

    phys_spectrum_vals_unnorm = (-spectral_1 - spectral_fd - spectral_be) 
    phys_spectrum_vals = phys_spectrum_vals_unnorm / scipy.integrate.trapezoid(phys_spectrum_vals_unnorm, omega_vals)

    if plot_electron_spectrum:
        plt.plot(omega_vals, phys_spectrum_vals)
        plt.xticks(
            [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10],
            ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8' ,'9', '10'],
        )

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\rho_d(\omega)$', rotation='horizontal')
        plt.title(r'Physical spectral function for $U= $ ' + f'{U}, ' + r'$\beta =$ ' + f'{beta}, ' + r'$N = $' + f'{N_target}')

        plt.tight_layout()

        plt.show()

    return spinon_spectrum_vals, rotor_spectrum_vals, phys_spectrum_vals


"""
    Returns the next iteration of the Lagrange parameters using Newton's method
    The minimizing function is f = <n_f(i)> - N/2 - <L(i)>
    Newton's method sets h' = h - f/f'
    Derivatives are calculted with a finite difference method
"""
def update_params(epsilon, t, U, t_eff_guess, h_guess, Z_guess, K_guess, beta, l_max, sites, N_target=2.0, PBC=False):
    # Small parameters to slow down updates
    ALPHA_H = 0.1
    ALPHA_Z = 0.1
    ALPHA_K = 0.1
    ALPHA_T_EFF = 0.1
    ALPHA_MU = 0.1

    # Tolerances for when iteration is done
    TOL_H = 1e-12
    TOL_T_EFF = 1e-12
    TOL_Z = 1e-12
    TOL_K = 1e-12
    TOL_N = 1e-6

    # Global chemical potential adjusts total particle number; epsilon used for doping differentials!
    mu = 0.0
    dh = 1e-5

    h_update, Z_update, K_update, t_eff_update = h_guess, Z_guess, K_guess, t_eff_guess

    did_converge = False
    num_cycles = 1

    while not did_converge:
        h, Z, K, t_eff = h_update, Z_update, K_update, t_eff_update

        spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h, mu))        
        rotor_eigenval_vec, rotor_eigenvec_vec = get_rotor_eig_vec(U, h, K, l_max, sites)
        
        # Calculate nf, L, and other parameters for next cycle
        nf = get_nf_vec(spinon_eigvals, spinon_eigvecs, beta, sites)
        L = get_L_vec(rotor_eigenval_vec, rotor_eigenvec_vec, beta, l_max, sites)

        # Calculate items for Newton's method
        f = nf - L - N_AT / 2.0
        J = get_J(epsilon, U, t_eff, h, dh, mu, K, beta, l_max, sites)

        try:
            delta_h = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            sys.exit('FAILURE')
            delta_h, *_ = np.linalg.lstsq(J, f, rcond=None)

        # Update parameters
        h_update = h - ALPHA_H * delta_h
        Z_update = (1 - ALPHA_Z) * Z + ALPHA_Z * get_Z(rotor_eigenval_vec, rotor_eigenvec_vec, beta, l_max, sites)
        K_update = (1 - ALPHA_K) * K + ALPHA_K * get_K(t, Z_update, spinon_eigvals, spinon_eigvecs, beta, sites)
        t_eff_update = (1 - ALPHA_T_EFF) * t_eff + ALPHA_T_EFF * get_t_eff(t, Z_update)

        N_total = np.sum(nf)
        mu += ALPHA_MU * (N_target - N_total)

        did_converge = (
            np.allclose(h_update, h, atol=TOL_H) and
            np.allclose(t_eff_update, t_eff, atol=TOL_T_EFF) and
            np.allclose(Z_update, Z, atol=TOL_Z) and
            np.allclose(K_update, K, atol=TOL_K) and
            abs(N_target - N_total) < TOL_N
        )

        num_cycles += 1
        helpers.print_params(t_eff, h, mu, Z, K, nf, L, should_print=False)

    # Calculate nf, L with final Hamiltonians
    spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon, t_eff_update, h_update, mu))
    rotor_eigvals, rotor_eigvecs = get_rotor_eig_vec(U, h_update, K_update, l_max, sites)
    
    nf = get_nf_vec(spinon_eigvals, spinon_eigvecs, beta, sites)
    L = get_L_vec(rotor_eigvals, rotor_eigvecs, beta, l_max, sites)
    
    return h_update, Z_update, K_update, t_eff_update, nf, L, mu, num_cycles

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


    off_diag_12 = -t_nn*(
        ((k_delta(0, n1_up_prime)*k_delta(1,n2_up_prime)
        *k_delta(n1_up, n1_up_prime+1)*k_delta(n1_down, n1_down_prime)*k_delta(n2_up,n2_up_prime-1)*k_delta(n2_down,n2_down_prime))
        +(k_delta(0,n1_down_prime)*k_delta(1,n2_up_prime)
        *k_delta(n1_up,n1_up_prime)*k_delta(n1_down,n1_down_prime+1)*k_delta(n2_up,n2_up_prime)*k_delta(n2_down,n2_down_prime-1)))
        *k_delta(nf1-1,nf1_prime)*k_delta(nf2-1,nf2_prime-2)
    )

    off_diag_21 = -t_nn*(
        ((k_delta(0, n2_up_prime)*k_delta(1,n1_up_prime)
        *k_delta(n2_up, n2_up_prime+1)*k_delta(n2_down, n2_down_prime)*k_delta(n1_up,n1_up_prime-1)*k_delta(n1_down,n1_down_prime))
        +(k_delta(0,n2_down_prime)*k_delta(1,n1_up_prime)
        *k_delta(n2_up,n2_up_prime)*k_delta(n2_down,n2_down_prime+1)*k_delta(n1_up,n1_up_prime)*k_delta(n1_down,n1_down_prime-1)))
        *k_delta(nf2-1,nf2_prime)*k_delta(nf1-1,nf1_prime-2)
    )

    mel = diag + off_diag_12 + off_diag_21
    print(f'U: {U}')
    print(f'bra: <{bra_state}|')
    print(f'ket: |{ket_state}>\n')
    print(f'mel: {mel}')
    print(f'diag: {diag}')
    print(f'off_diag_12: {off_diag_12}')
    print(f'off_diag_21: {off_diag_21}\n\n')
    return mel

def get_E_dimer_phys(epsilon_vec, U, t_nn, rotor1, rotor2, l_max):
    """
        Returns 4 energies corresponding to the constrained tensor product |\psi_f>|\psi_{theta,1}>|\psi_{theta,2}>
        where |\psi_f> takes on 1 of the 4 calculated eigenkets for the spinon Hamiltonian
        This is highly specific to the homogeneous dimer problem
    """
    half_filling_states = {
        '1100': (1,1,0,0),
        '1010': (1,0,1,0),
        '1001': (1,0,0,1),
        '0110': (0,1,1,0),
        '0101': (0,1,0,1),
        '0011': (0,0,1,1)
    }

    spinon1 = [(0.5,'1100'), (0.5,'0110'), (0.5,'1001'), (0.5,'0011')]
    spinon2 = [(0.5,'1100'), (0.5,'0110'), (-0.5,'1001'), (-0.5,'0011')]
    spinon3 = [(0.5,'1100'), (-0.5,'0110'), (0.5,'1001'), (-0.5,'0011')]
    spinon4 = [(0.5,'1100'), (-0.5,'0110'), (-0.5,'1001'), (0.5,'0011')]

    E1, E2, E3, E4 = 0.0, 0.0, 0.0, 0.0
    for i in range(4):
        bra_weight, bra_state = spinon1[i]
        bra_nums = half_filling_states[bra_state]

        bra_nf1 = bra_nums[0] + bra_nums[1]
        bra_nf2 = bra_nums[2] + bra_nums[3]

        bra_l1 = l_max + bra_nf1 - 1
        bra_l2 = l_max + bra_nf2 - 1

        for j in range(4):
            ket_weight, ket_state = spinon1[j]
            ket_nums = half_filling_states[ket_state]

            ket_nf1 = ket_nums[0] + ket_nums[1]
            ket_nf2 = ket_nums[2] + ket_nums[3]

            ket_l1 = l_max + ket_nf1 - 1
            ket_l2 = l_max + ket_nf2 - 1

            mel = get_E_mel(epsilon_vec, U, t_nn, bra_nums, ket_nums)
            E_weight = bra_weight * ket_weight * np.conj(rotor1[bra_l1]) * rotor1[ket_l1] * np.conj(rotor2[bra_l2]) * rotor2[ket_l2]

            '''
            print(f'rotor1_bra_l1 weight: {rotor1[bra_l1]}')
            print(f'rotor1_bra_l2 weight: {rotor1[bra_l2]}')
            print(f'rotor1_ket_l1 weight: {rotor1[ket_l1]}')
            print(f'rotor1_ket_l2 weight: {rotor1[ket_l2]}')
            print(f'E1: {E1}')
            '''
            E1 += mel * E_weight
            #print(f'Energy: {E1}')
    return E1

if __name__ == '__main__':
    PBC = True
    plot_spinon_spectral = True
    plot_rotor_spectral = False
    plot_electron_spectral = True 

    sites = 2
    half_sites = int(sites/2.0)
    N_target = int(1.0*sites) #Half filling

    l_max = 20
    l_vals = np.arange(-l_max, l_max + 1)

    # Add nearest-neighbor hopping
    t_nn = 1.0; t_nnn = t_nn/10.0

    t = np.diag(np.full(sites-1, t_nn), k=1) + np.diag(np.full(sites-1, t_nn), k=-1)
    # t += np.diag(np.full(sites-2, t_nnn), k=2) + np.diag(np.full(sites-2, t_nnn), k=-2)
    
    """
    if PBC:
        t[0, -1], t[-1, 0] = t_nn, t_nn
        t[0, -2], t[-2, 0], t[1, -1], t[-1, 1] = t_nnn, t_nnn, t_nnn, t_nnn
    """

    epsilon_vec = np.zeros(sites)

    # parameter to adjust the extent to which the jump varies over
    interface_width = 2.5
    site_indices = np.arange(sites)
    # epsilon_doped = -0.25 * np.tanh((site_indices - half_sites) / interface_width) - 0.25
    epsilon_doped = np.array([0.0, -1.0])

    """
        Initial parameter guesses for iterative Newton's method
    """
    t_eff_guess = 0.8 * t
    h_guess = -epsilon_vec
    h_doped_guess = -epsilon_doped
    Z_guess = np.full(sites, 0.5)
    K_guess = np.full(sites,-1.5)

    omega_vals = np.linspace(-10,10,20000)

    U_vals = np.linspace(0.0,1.0,10)
    beta = 50.0

    energies_exact = [[] for _ in range(6)]
    energies_mf = [[]]

    count = 0
    
    for U in U_vals:
        H_exact = exact.get_H_exact(epsilon_vec[0], U, t_nn, exact.get_def_particle_basis(2))
        eigvals_exact, _ = np.linalg.eigh(H_exact)
        
        for i in range(6):
            energies_exact[i].append(eigvals_exact[i])

        h, Z, K, t_eff, nf, L, mu, num_cycles = update_params(epsilon_vec, t, U, t_eff_guess, h_guess, Z_guess, K_guess, beta, l_max, sites, N_target)

        H_rotor = get_H_rotor(U, h, K, l_max, 0)
        eigvals_rotor, eigvecs_rotor = np.linalg.eigh(H_rotor)
        
        rotor1 = eigvecs_rotor[0]
        rotor2 = eigvecs_rotor[0]

        '''
        for i in range(2*l_max + 1):
            plotting.plot_rotor_weight(eigvecs_rotor[i], 0, epsilon_vec[0], 0, l_max)
        '''
        print(f'rotor1: {rotor1}')
        energy = get_E_dimer_phys(epsilon_vec, U, t_nn, rotor1, rotor2, l_max)
        energies_mf[0].append(energy)
        count += 1
        print(count)


    plt.figure()

    # Plot exact spectrum for each figure
    for i in range(6):
        if i==0:
            plt.plot(U_vals, energies_exact[i], color='red', label='Exact')
        else:
            plt.plot(U_vals, energies_exact[i], color='red')

    # Plot MF spectrum
    for i in range(1):
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
