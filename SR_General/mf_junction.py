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

def get_nf(spinon_eigvals, spinon_eigvecs, beta, sites):
    nf = np.zeros(sites)

    for i in range(sites):
        sum = 0.0
        for j in range(sites):
            energy = spinon_eigvals[j]
            sum += (np.abs(spinon_eigvecs[i,j]) ** 2) * helpers.fermi_dirac(energy, beta)

        nf[i] = N_AT * sum

    return nf

"""
    Return derivative of nf with respect to h
"""
def get_d_nf(epsilon, h, dh, t_eff, beta, mu, sites):
    h_p, h_m = h + dh/2, h - dh/2

    eigvals_p, eigvecs_p = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h_p, mu, sites))
    eigvals_m, eigvecs_m = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h_m, mu, sites))

    nf_p, nf_m = get_nf(eigvals_p, eigvecs_p, beta, sites), get_nf(eigvals_m, eigvecs_m, beta, sites)
    d_nf = (nf_p - nf_m) / dh

    return d_nf


"""
    Return derivative of L with respect to h
"""
def get_d_L(U, h, dh, K, l_max, sites):
    rotor_dim = 2 * l_max + 1
    h_p, h_m = h + dh/2, h - dh/2

    # H_\theta(h+dh/2)
    site_eigenvecs_p, site_eigenvals_p = np.zeros((sites, rotor_dim, rotor_dim), dtype=complex), np.zeros((sites, rotor_dim))
    H_rotors_plus_dh = get_H_rotors(U, h_p, K, l_max, sites)
    for i in range(sites):
        site_eigenvals_p[i], site_eigenvecs_p[i] = np.linalg.eigh(H_rotors_plus_dh[i])
    
    # H_\theta(h-dh/2)
    site_eigenvecs_m, site_eigenvals_m = np.zeros((sites, rotor_dim, rotor_dim), dtype=complex), np.zeros((sites, rotor_dim))
    H_rotors_minus_dh = get_H_rotors(U, h_m, K, l_max, sites)
    for i in range(sites):
        site_eigenvals_m[i], site_eigenvecs_m[i] = np.linalg.eigh(H_rotors_minus_dh[i])

    L_plus, L_minus = get_L(site_eigenvals_p, site_eigenvecs_p, beta, l_max, sites), get_L(site_eigenvals_m, site_eigenvecs_m, beta, l_max, sites)
    d_L = (L_plus - L_minus) / dh

    return d_L


"""
    Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
"""
def get_L(rotor_site_eigvals, rotor_site_eigvecs, beta, l_max, sites):
    # Dimension of rotor Hilbet space as a function of the cutoff (formally infinite)
    rotor_dim = 2 * l_max + 1
    L = np.zeros(sites)

    # Sum over lattice sites
    for i in range(sites):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the ground-state energy at that site
        # Allowed because of partition function
        eigvals = rotor_site_eigvals[i] - rotor_site_eigvals[i,0]
        eigvecs = rotor_site_eigvecs[i]

        sum, partition = 0.0, 0.0

        # Sum over rotor energies and states
        for j in range(rotor_dim):
            energy = eigvals[j]
            state = eigvecs[:,j]

            # Boltzmann factor
            bz = np.exp(-beta * energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range(rotor_dim):
                l = index - l_max
                sum += bz * (np.abs(state[index]) ** 2) * l
        
            partition += bz 

        L[i] = sum / partition

    return L

"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_site_eigvals, rotor_site_eigvecs, beta, l_max, sites):
    # Dimension of rotor Hilbet space as a function of the cutoff (formally infinite)
    rotor_dim = 2 * l_max + 1
    Z = np.zeros(sites)

    for i in range (sites):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        eigvals = rotor_site_eigvals[i] - rotor_site_eigvals[i,0]        
        eigvecs = rotor_site_eigvecs[i]

        sum, partition = 0.0, 0.0
        
        # Sum over rotor states and energies. There are ROTOR_DIM total eigenstates
        for j in range (rotor_dim):
            energy = eigvals[j]
            state = eigvecs[:,j]

            bz = np.exp(-beta * energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range (rotor_dim):
                l = index - l_max 

                # Handle truncating the Hilbert space by setting the l_max + 1, -l_max - 1 states to zero
                if index == 0:
                    sum += bz * np.conj(state[index]) * state[index + 1]
                elif index == rotor_dim - 1:
                    sum += bz * np.conj(state[index]) * state[index - 1]
                else:
                    sum += bz * np.conj(state[index]) * (state[index + 1] + state[index - 1])
                    
            partition += bz

        sum, partition = np.real(sum), np.real(partition)
        Z[i] = (sum / (2 * partition))**2

    return Z

"""
    Return effective tunneling amplitude matrix
    t is treated as a vector 
"""
def get_t_eff(t, Z, sites):
    t_eff = np.zeros((sites, sites))

    # Fill the upper triangle block
    for i in range(sites):
        for j in range(i, sites):
            t_eff[i,j] = t[i,j] * np.sqrt(Z[i] * Z[j])

    # Use Hermiticity to quickly fill the lower block
    t_eff = t_eff + t_eff.T.conj() - np.diag(np.diag(t_eff))

    return t_eff


"""
Return real-space spinon Hamiltonian in the upper-block
The single-particle spinon Hamiltonian is block diagonalized with each block corresponding to a spin
I think we only have to diagonalize a single one of these blocks since the energies are degenerate, but I am not sure
"""
def get_H_spinon(epsilon, t_eff, h, mu, sites):
    H = np.zeros((sites, sites))

    np.fill_diagonal(H, epsilon - h - mu)
    for i in range(sites):
        for j in range(sites):
            H[i,j] += -t_eff[i,j]

    return H


"""
Return array of rotor Hamiltonians at each site
"""
def get_H_rotors(U, h, K, l_max, sites):
    # Rotor Hilbert space dimension
    rotor_dim = 2 * l_max + 1

    # Make a rotor Hamiltonian at each lattice sites, data type is list of matrices
    H_sites = np.zeros((sites, rotor_dim, rotor_dim))
    l_vals = np.arange(-l_max, l_max + 1)
    
    for i in range(sites):
        H = np.zeros((rotor_dim, rotor_dim))

        diag = ((U / 2.0) * l_vals + h[i]) * l_vals
        np.fill_diagonal(H, diag)

        off_diag = K[i] / 2.0
        H += np.diag(np.full(rotor_dim - 1, off_diag), k=1)
        H += np.diag(np.full(rotor_dim - 1, off_diag), k=-1)

        H_sites[i] = H

    return H_sites

"""
Return K
"""
def get_K(t, Z, spinon_eigvals, spinon_eigvecs, beta, sites):
    K = np.zeros(sites)

    eigvals = spinon_eigvals
    eigvecs = spinon_eigvecs 
    
    for i in range(sites):
        sum = 0.0

        for j in range(sites):
            t_ij = t[i,j]
            if t_ij != 0:
                for k in range(sites):
                    energy = eigvals[k]
                    sum += t_ij * np.sqrt(Z[j]) * np.conj(eigvecs[i,k]) * eigvecs[j,k] * helpers.fermi_dirac(energy, beta)
    
        K[i] = -2 * N_AT * sum

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
        H_rotor = get_H_rotors(U, h, K, l_max)[0]
        eig_vals, eig_vecs = np.linalg.eigh(H_rotor)
    
        spectral_val = 0.0
        for i in range(rotor_dim):
            energy_1 = eig_vals[i]
            state_1 = eig_vecs[:,i]
            print(energy_1)
            bz_1 = np.exp(-beta * energy_1)

            for j in range(i, rotor_dim):
                energy_2 = eig_vals[j]
                state_2 = eig_vecs[:,j]

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
    ALPHA_H = 0.05
    ALPHA_Z = 0.05
    ALPHA_K = 0.05
    ALPHA_T_EFF = 0.05
    ALPHA_MU = 0.01

    # Global chemical potential adjusts total particle number; epsilon used for doping differentials!
    mu = 0.0

    dh = 1e-4
    rotor_dim = 2 * l_max + 1

    h_update, Z_update, K_update, t_eff_update = h_guess, Z_guess, K_guess, t_eff_guess

    did_converge = False
    num_cycles = 1.0

    while not did_converge:
        h, Z, K, t_eff = h_update, Z_update, K_update, t_eff_update

        spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h, mu, sites))

        # H_\theta(h)
        rotor_site_eigenvecs, rotor_site_eigenvals = np.zeros((sites, rotor_dim, rotor_dim)), np.zeros((sites, rotor_dim))
        H_rotors = get_H_rotors(U, h, K, l_max, sites)
        for i in range(sites):
            rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])
        
        """
        Calculate nf, L, and other parameters for next cycle
        """
        nf = get_nf(spinon_eigvals, spinon_eigvecs, beta, sites)
        L = get_L(rotor_site_eigenvals, rotor_site_eigenvecs, beta, l_max, sites)

        d_nf = get_d_nf(epsilon, h, dh, t_eff, beta, mu, sites)
        d_L = get_d_L(U, h, dh, K, l_max, sites)

        h_update = np.zeros(sites)

        for i in range(sites):
            h_update[i] = h[i] - ALPHA_H * (nf[i] - L[i] - N_AT / 2.0) / (d_nf[i] - d_L[i])

        Z_update = (1 - ALPHA_Z) * Z + ALPHA_Z * get_Z(rotor_site_eigenvals, rotor_site_eigenvecs, beta, l_max, sites)
        K_update = (1 - ALPHA_K) * K + ALPHA_K * get_K(t, Z_update, spinon_eigvals, spinon_eigvecs, beta, sites)
        t_eff_update = (1 - ALPHA_T_EFF) * t_eff + ALPHA_T_EFF * get_t_eff(t, Z_update, sites)

        N_total = np.sum(nf)
        mu += ALPHA_MU * (N_target - N_total)
        
        if (np.abs(h_update - h) < 1e-8).all() and (np.abs(t_eff_update - t_eff) < 1e-6).all() and (np.abs(Z_update - Z) < 1e-6).all() and (np.abs(K_update - K) < 1e-6).all() and (np.abs(N_target - N_total) < 1e-6):
            did_converge = True
        else:
            num_cycles += 1

            """
            print(f'h: {h}')
            print(f'Z: {Z}')
            print(f'K: {K}')
            print(f't_eff: {t_eff}')
            print(f'nf: {nf}')
            print(f'L: {L}')
            print(f'mu: {mu}')
            """

    return h_update, Z_update, K_update, t_eff, nf, L, mu, num_cycles

if __name__ == '__main__':
    sites = 120
    half_sites = int(sites/2.0)
    l_max = 12

    t = np.zeros((sites, sites))

    t_nn = 1.0; t_nnn = t_nn / 10.0

    # Add nearest-neighbor hopping
    t += np.diag(np.full(sites-1, t_nn), k=1) + np.diag(np.full(sites-1, t_nn), k=-1)

    N_target = sites # Half filling

    U_vals = [0.1, 1.0, 2.0, 3.0]
    beta_vals = [50.0,5.0]
    epsilon_R_vals = [0.1]

    U = 1.0
    beta = 50.0

    epsilon = np.concatenate([np.full(half_sites, 0.0), np.full(half_sites, -0.5)])

    t_eff_guess = 0.5 * t
    h_guess = -epsilon
    Z_guess = np.full(sites, 0.5)
    K_guess = np.full(sites,-1.5)

    h, Z, K, t_eff, nf, L, mu, num_cycles = update_params(epsilon, t, U, t_eff_guess, h_guess, Z_guess, K_guess, beta, l_max, sites, N_target)

    print(f'N: {np.sum(nf)}'
            )
    title = f'Electron Occupancy at Half-Filling: ' + r'$U =$ ' + f'{U}, ' + r'$\beta =$ ' + f'{beta}, ' + r'$\mu_L = 0.0$, $\mu_R =$ ' + f'{0.5}'
    plotting.plot_electron_occupancy(nf, title)
