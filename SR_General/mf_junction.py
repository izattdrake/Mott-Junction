import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 
import scipy 

import utility.helpers as helpers
import utility.plotting as plotting

warnings.simplefilter("error", RuntimeWarning)

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
    Returns matrix contribution dn_i/dh_j to Jacobian in Newton's method
"""
def get_J_nf(epsilon, t_eff, h, dh, mu, beta, sites):
    J = np.zeros((sites, sites))
    for j in range(sites):
        dh_vec = np.zeros(sites)
        dh_vec[j] = dh

        eigvals_p, eigvecs_p = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h + dh_vec/2, mu))
        eigvals_m, eigvecs_m = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h - dh_vec/2, mu))

        nf_p = get_nf_vec(eigvals_p, eigvecs_p, beta, sites)
        nf_m = get_nf_vec(eigvals_m, eigvecs_m, beta, sites)

        J[:,j] = (nf_p - nf_m) / dh

    return J

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
    Return L's contribution to Jacobian
    Off-diagonal derivatives are NOT included
    It is expected that the spinon sector will take care of those
    and their contribution will bleed through to the rotor sector
    for successive iterations
"""
def get_J_L(U, h, dh, K, beta, l_max, sites):
    J = np.zeros((sites, sites))

    for j in range(sites):
        h_p, h_m = h.copy(), h.copy()
        h_p[j] += dh/2
        h_m[j] -= dh/2

        eigvals_p, eigvecs_p = np.linalg.eigh(get_H_rotor(U, h_p, K, l_max, j))
        eigvals_m, eigvecs_m = np.linalg.eigh(get_H_rotor(U, h_m, K, l_max, j))

        J[j,j] = (get_L(eigvals_p, eigvecs_p, beta, l_max) - get_L(eigvals_m, eigvecs_m, beta, l_max)) / dh

    return J

"""
    Return full Jacobian used in Newton's method
"""
def get_J(epsilon, U, t_eff, h, dh, mu, K, beta, l_max, sites):
    J_nf = get_J_nf(epsilon, t_eff, h, dh, mu, beta, sites)
    J_L = get_J_L(U, h, dh, K, beta, l_max, sites)

    J = J_nf - J_L

    return J


"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_eigvals_vec, rotor_eigvecs_vec, beta, l_max, sites):
    # Dimension of rotor Hilbet space as a function of the cutoff (formally infinite)
    rotor_dim = 2 * l_max + 1
    Z = np.zeros(sites)

    for i in range(sites):
        eigvals = rotor_eigvals_vec[i] - rotor_eigvals_vec[i,0]
        eigvecs = rotor_eigvecs_vec[i]

        bz = np.exp(-beta * eigvals)

        mel = (eigvecs[:-1, :].conj() * eigvecs[1:, :]).sum(axis=0)
        Z[i] = (np.dot(bz, np.real(mel)) / np.sum(bz))**2

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
    Calculate spinon spectral density function for periodic dimer (allows usual wavevector Green's function analysis)
    Assume translational symmetry
"""


def spinon_spectrum(omega_vals, epsilon: float, t_eff: float, h: float, mu: float):
    def spinon_spectral_val(omega: float, epsilon: float, t_eff: float, h: float, mu: float, a=1.0, eta=1e-4):
        spectral_val = 0.0

        # Assume translational symmetry, sum over k = 0, pi/a
        N = 2
        for n in range(N):
            k = (2*np.pi/(N*a)) * n
            energy_k = -2 * t_eff * np.cos(k * a) + (epsilon - mu - h)
            spectral_val += helpers.d_delta(omega - energy_k)

        return spectral_val
    
    spectrum = np.array([spinon_spectral_val(omega, epsilon[0], t_eff, h[0], mu) for omega in omega_vals])
    
    norm = scipy.integrate.trapezoid(spectrum, omega_vals)
    print(f'spinon norm: {norm}')
    spectrum_normalized = spectrum / norm

    return spectrum_normalized

def rotor_spectrum(omega_vals, U: float, h, K, beta: float, l_max: int, overwrite=False):
    def get_rotor_spectral_val(omega: float, U: float, h, K, beta: float, l_max):
        rotor_dim = 2 * l_max + 1
        H_rotor = get_H_rotor_vec(U, h, K, l_max)[0]
        eigvals, eigvecs = np.linalg.eigh(H_rotor)
    
        spectral_val = 0.0
        for i in range(rotor_dim):
            energy_1 = eigvals[i]
            state_1 = eigvecs[:,i]
            print(energy_1)
            bz_1 = np.exp(-beta * energy_1)

            for j in range(i, rotor_dim):
                energy_2 = eigvals[j]
                state_2 = eigvecs[:,j]

                bz_2 = np.exp(-beta * energy_2)

                mel = 0.0
                mel = np.dot(np.conj(state_2[1:]), state_1[:-1])

                spectral_val += bz_1 * (np.abs(mel)**2) * helpers.d_delta(omega - (energy_2 - energy_1))

                if not i == j:
                    spectral_val += bz_2 * (np.abs(mel)**2) * helpers.d_delta(omega - (energy_1 - energy_2))

        return spectral_val
    
    """
        Check if rotor spectrum is cached
    """
    spectrum = np.array([get_rotor_spectral_val(omega, U, h, K, beta, l_max) for omega in omega_vals])
    
    norm = scipy.integrate.trapezoid(spectrum, omega_vals)
    print(f'rotor norm: {norm}')
    spectrum_normalized = spectrum / norm

    return spectrum_normalized


"""
    Check condition on rotor spectral function that it must satisfy to be correct
"""
def validate_rotor_spectrum(omega_vals, spectrum_vals, beta: float) -> bool:
    dim = len(omega_vals)

    for omega_index in range(dim):
        omega = omega_vals[omega_index]
        m_omega_index = np.where(omega_vals == -omega)[0][0]

        if not (np.exp(beta * omega) * spectrum_vals[m_omega_index] == spectrum_vals[omega_index]):
            return False
    
    return True 


"""
    Returns the electron spectral function for all values of omega using Scipy's FFT convolution method
"""
def get_electron_spectral_vals(omega_vals: float, epsilon: float, U: float, t_eff: float, h: float, mu: float, beta, l_max, a=1.0, eta=1e-4):
    domega = np.abs(omega_vals[1] - omega_vals[0])

    # Super weird with whether I'm passing scalar or vector values for some parameters. Should clean up soon
    spinon_spectrum_vals = spinon_spectrum(omega_vals, epsilon, t_eff, h, mu)
    rotor_spectrum_vals = rotor_spectrum(omega_vals, U, h, K, beta, l_max)

    fd_vals = np.array([helpers.fermi_dirac(omega, beta) for omega in omega_vals])
    be_vals = np.array([helpers.bose_einstein(omega, beta) for omega in omega_vals])
   
    # See latex docs for definition of f1, f2, g1, g2
    f1_vals = spinon_spectrum_vals * (1 - fd_vals)
    f2_vals = spinon_spectrum_vals 

    g1_vals = rotor_spectrum_vals
    g2_vals = rotor_spectrum_vals * be_vals

    spectral_1 = scipy.signal.fftconvolve(f1_vals, g1_vals, mode='same') * domega
    spectral_2 = scipy.signal.fftconvolve(f2_vals, g2_vals, mode='same') * domega

    phys_spectral = -spectral_1 - spectral_2

    return phys_spectral


"""
    Returns the next iteration of the Lagrange parameters using Newton's method
    The minimizing function is f = <n_f(i)> - N/2 - <L(i)>
    Newton's method sets h' = h - f/f'
    Derivatives are calculted with a finite difference method
    TODO: Find a closed form analytic solution using perturbation theory
    NOTE: Analytic solutions may not be possible due to recursion
"""
def update_params(epsilon, t, U, t_eff_guess, h_guess, Z_guess, K_guess, beta, l_max, sites, N_target=2.0):
    # Small parameters to slow down updates
    ALPHA_H = 0.05
    ALPHA_Z = 0.05
    ALPHA_K = 0.05
    ALPHA_T_EFF = 0.05
    ALPHA_MU = 0.01

    # Tolerances for when iteration is done
    TOL_H = 1e-6
    TOL_T_EFF = 1e-6
    TOL_Z = 1e-6
    TOL_K = 1e-6
    TOL_N = 1e-4

    # Global chemical potential adjusts total particle number; epsilon used for doping differentials!
    mu = 0.0
    dh = 1e-4

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
        delta_h = np.linalg.solve(J,f)

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

if __name__ == '__main__':
    sites = 150
    half_sites = int(sites/2.0)
    l_max = 12

    # Add nearest-neighbor hopping
    t_nn = 1.0; t_nnn = 0.1
    t = np.diag(np.full(sites-1, t_nn), k=1) + np.diag(np.full(sites-1, t_nn), k=-1)
    t += np.diag(np.full(sites-2, t_nnn), k=2) + np.diag(np.full(sites-2, t_nnn), k=-2)
    
    N_target = sites # Half filling

    U = 1.5
    beta = 50.0

    # number of lattice sites chemical potential varies over
    interface_width = 5.0  
    site_indices = np.arange(sites)
    epsilon = -0.25 * np.tanh((site_indices - half_sites) / interface_width) - 0.25

    t_eff_guess = 0.5 * t
    h_guess = -epsilon
    Z_guess = np.full(sites, 0.5)
    K_guess = np.full(sites,-1.5)

    for U in [0.1,1.0,2.0]:
        for beta in [50.0,5.0]:
            h, Z, K, t_eff, nf, L, mu, num_cycles = update_params(epsilon, t, U, t_eff_guess, h_guess, Z_guess, K_guess, beta, l_max, sites, N_target)

            title = f'Electron Occupancy at Half-Filling: ' + r'$U =$ ' + f'{U}, ' + r'$\beta =$ ' + f'{beta}, ' + r'$\mu_L = 0.0$, $\mu_R =$ ' + f'{0.5}'
            plotting.plot_electron_occupancy(nf, title)
