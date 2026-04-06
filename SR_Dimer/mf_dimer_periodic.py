import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 
import os
import scipy

import utility.helpers as helpers
import utility.plotting as SR_plotting

warnings.simplefilter("error", RuntimeWarning)

N_AT = 2

def get_nf(spinon_eigvals, spinon_eigvecs, beta):
    nf = np.zeros(2)

    for i in range(2):
        sum = 0.0
        for j in range(2):
            energy = spinon_eigvals[j]
            sum += (np.abs(spinon_eigvecs[i,j]) ** 2) * helpers.fermi_dirac(energy, beta)

        nf[i] = N_AT * sum

    return nf

"""
    Return derivative of nf with respect to h
"""
def get_d_nf(epsilon, h, dh, t_eff, beta, mu):
    h_p, h_m = h + dh/2, h - dh/2

    eigvals_p, eigvecs_p = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h_p, mu))
    eigvals_m, eigvecs_m = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h_m, mu))

    nf_p, nf_m = get_nf(eigvals_p, eigvecs_p, beta), get_nf(eigvals_m, eigvecs_m, beta)
    d_nf = (nf_p - nf_m) / dh

    return d_nf


"""
    Return derivative of L with respect to h
"""
def get_d_L(U, h, dh, K, l_max):
    dim = 2 * l_max + 1
    h_p, h_m = h + dh/2, h - dh/2

    # H_\theta(h+dh/2)
    site_eigenvecs_p, site_eigenvals_p = np.zeros((2, dim, dim), dtype=complex), np.zeros((2, dim))
    H_rotors_plus_dh = get_H_rotors(U, h_p, K, l_max)
    for i in range(2):
        site_eigenvals_p[i], site_eigenvecs_p[i] = np.linalg.eigh(H_rotors_plus_dh[i])
    
    # H_\theta(h-dh/2)
    site_eigenvecs_m, site_eigenvals_m = np.zeros((2, dim, dim), dtype=complex), np.zeros((2, dim))
    H_rotors_minus_dh = get_H_rotors(U, h_m, K, l_max)
    for i in range(2):
        site_eigenvals_m[i], site_eigenvecs_m[i] = np.linalg.eigh(H_rotors_minus_dh[i])

    L_plus, L_minus = get_L(site_eigenvals_p, site_eigenvecs_p, beta, l_max), get_L(site_eigenvals_m, site_eigenvecs_m, beta, l_max)
    d_L = (L_plus - L_minus)/dh

    return d_L


"""
Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
"""
def get_L(rotor_site_eigvals, rotor_site_eigvecs, beta, l_max):
    ROTOR_DIM = 2 * l_max + 1
    L = np.zeros(2)

    # Sum over lattice sites
    for i in range(2):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        rotor_eigvals = rotor_site_eigvals[i] - rotor_site_eigvals[i,0]
        rotor_eigvecs = rotor_site_eigvecs[i]

        num_sum, den_sum = 0.0, 0.0

        # Sum over rotor energies and states
        for j in range(ROTOR_DIM):
            rotor_energy = rotor_eigvals[j]
            rotor_eigenvec = rotor_eigvecs[:,j]

            boltzmann = np.exp(-beta * rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range(ROTOR_DIM):
                l = index - l_max
                num_sum += boltzmann * (np.abs(rotor_eigenvec[index]) ** 2) * l
        
            den_sum += boltzmann 

        L[i] = num_sum / den_sum

    return L

"""
Returns array of Z values for each lattice site
"""
def get_Z(rotor_site_eigvals, rotor_site_eigvecs, beta, l_max):
    dim = (2 * l_max + 1)
    Z = np.zeros(2)

    for i in range (2):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        rotor_eigvals = rotor_site_eigvals[i] - rotor_site_eigvals[i,0]        
        rotor_eigvecs = rotor_site_eigvecs[i]

        num_sum = 0.0
        den_sum = 0.0
        
        # Sum over rotor states and energies. There are ROTOR_DIM total eigenstates
        for j in range (dim):
            rotor_state = rotor_eigvecs[:,j]
            rotor_energy = rotor_eigvals[j]

            boltzmann = np.exp(-beta * rotor_energy)

            # Sum over rotor Hilbert space in |l> basis forming the rotor state
            for index in range (dim):
                l = index - l_max 
                # Handle truncating the Hilbert space by simply ignoring the L_CUT + 1 state
                if index == 0:
                    num_sum += boltzmann * np.conj(rotor_state[index])*rotor_state[index + 1]
                elif index == dim - 1:
                    num_sum += boltzmann * np.conj(rotor_state[index])*rotor_state[index - 1]
                else:
                    num_sum += boltzmann * np.conj(rotor_state[index])*(rotor_state[index + 1] + rotor_state[index - 1])
                    
            # Partition function, make function and calculate once?
            den_sum += boltzmann

        num_sum, den_sum = np.real(num_sum), np.real(den_sum)
        Z[i] = (num_sum/(2 * den_sum))**2

    return Z

"""
Return effective tunneling amplitude matrix
"""
def get_t_eff(t, Z):
    t_eff = t * np.sqrt(Z[0] * Z[1])
    return t_eff


"""
Return real-space spinon Hamiltonian in the upper-block
The single-particle spinon Hamiltonian is block diagonalized with each block corresponding to a spin
I think we only have to diagonalize a single one of these blocks, but I am not sure
"""
def get_H_spinon(epsilon, t_eff, h, mu):
    H = np.zeros((2, 2))

    np.fill_diagonal(H, epsilon - h - mu)
    H[0,1], H[1,0] = -t_eff, -t_eff

    return H


"""
Return array of rotor Hamiltonians at each site
"""
def get_H_rotors(U, h, K, l_max):
    ROTOR_DIM = 2 * l_max + 1
    H_sites = np.zeros((2, ROTOR_DIM, ROTOR_DIM))
    l_vals = np.arange(-l_max, l_max + 1)
    
    for i in range(2):
        H = np.zeros((ROTOR_DIM, ROTOR_DIM))

        diag = ((U / 2.0) * l_vals + h[i]) * l_vals
        np.fill_diagonal(H, diag)

        off_diag = K[i] / 2.0
        H += np.diag(np.full(ROTOR_DIM - 1, off_diag), k=1)
        H += np.diag(np.full(ROTOR_DIM - 1, off_diag), k=-1)

        H_sites[i] = H

    return H_sites

"""
Return K
"""
def get_K(t, Z, spinon_eigvals, spinon_eigvecs, beta):
    K = np.zeros(2)

    sums = [0.0, 0.0]  

    for i in range(2):
        energy = spinon_eigvals[i]
        sums[0] += t * np.sqrt(Z[1]) * spinon_eigvecs.conj().T[i,0] * spinon_eigvecs[1,i] * helpers.fermi_dirac(energy, beta)
        sums[1] += t * np.sqrt(Z[0]) * spinon_eigvecs.conj().T[i,1] * spinon_eigvecs[0,i] * helpers.fermi_dirac(energy, beta)
    
    K[0], K[1] = -2 * N_AT * sums[0], -2 * N_AT * sums[1]

    return K

"""
    Returns the next iteration of the Lagrange parameters using Newton's method
    The minimizing function is f = <n_f(i)> - N/2 - <L(i)>
    Newton's method sets h' = h - f/f'
    Derivatives are calculted with a finite difference method
    TODO: Find a closed form analytic solution using perturbation theory
    NOTE: Analytic solutions may not be possible due to recursion
"""
def update_params(epsilon, t, U, t_eff_guess, h_guess, Z_guess, K_guess, beta, l_max, N_target=2.0):
    ALPHA_H = 0.05
    ALPHA_Z = 0.05
    ALPHA_K = 0.05
    ALPHA_T_EFF = 0.05
    ALPHA_MU = 0.01

    mu = 0.0

    dh = 1e-4
    ROTOR_DIM = 2 * l_max + 1

    h_update, Z_update, K_update, t_eff_update = h_guess, Z_guess, K_guess, t_eff_guess

    did_converge = False
    num_cycles = 1.0
    while not did_converge:
        h, Z, K, t_eff = h_update, Z_update, K_update, t_eff_update

        spinon_eigvals, spinon_eigvecs = np.linalg.eigh(get_H_spinon(epsilon, t_eff, h, mu))

        # H_\theta(h)
        rotor_site_eigenvecs, rotor_site_eigenvals = np.zeros((2, ROTOR_DIM, ROTOR_DIM)), np.zeros((2, ROTOR_DIM))
        H_rotors = get_H_rotors(U, h, K, l_max)
        for i in range(2):
            rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])
        
        """
        Calculate nf, L, and other parameters for next cycle
        """
        nf = get_nf(spinon_eigvals, spinon_eigvecs, beta)
        L = get_L(rotor_site_eigenvals, rotor_site_eigenvecs, beta, l_max)

        d_nf = get_d_nf(epsilon, h, dh, t_eff, beta, mu)
        d_L = get_d_L(U, h, dh, K, l_max)

        h_update = np.zeros(2)

        for i in range(2):
            h_update[i] = h[i] - ALPHA_H * (nf[i] - L[i] - N_AT/2.0)/(d_nf[i] - d_L[i])

        Z_update = (1 - ALPHA_Z) * Z + ALPHA_Z * get_Z(rotor_site_eigenvals, rotor_site_eigenvecs, beta, l_max)
        K_update = (1 - ALPHA_K) * K + ALPHA_K * get_K(t, Z_update, spinon_eigvals, spinon_eigvecs, beta)
        t_eff_update = (1 - ALPHA_T_EFF) * t_eff + ALPHA_T_EFF * get_t_eff(t, Z_update)

        N_total = np.sum(nf)
        mu += ALPHA_MU * (N_target - N_total)

        if (np.abs(h_update - h) < 1e-6).all() and (np.abs(t_eff_update - t_eff) < 1e-6) and (np.abs(Z_update - Z) < 1e-6).all() and (np.abs(K_update - K) < 1e-6).all() and (np.abs(N_target - N_total) < 1e-3):
            did_converge = True
        else:
            num_cycles += 1

    return h_update, Z_update, K_update, t_eff, nf, L, mu, num_cycles

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
        ROTOR_DIM = 2 * l_max + 1
        H_rotor = get_H_rotors(U, h, K, l_max)[0]
        eig_vals, eig_vecs = np.linalg.eigh(H_rotor)
    
        spectral_val = 0.0
        for i in range(ROTOR_DIM):
            energy_1 = eig_vals[i]
            state_1 = eig_vecs[:,i]
            print(energy_1)
            boltzmann_1 = np.exp(-beta * energy_1)

            for j in range(i, ROTOR_DIM):
                energy_2 = eig_vals[j]
                state_2 = eig_vecs[:,j]

                boltzmann_2 = np.exp(-beta * energy_2)

                mel = 0.0
                mel = np.dot(np.conj(state_2[1:]), state_1[:-1])

                spectral_val += boltzmann_1 * (np.abs(mel)**2) * helpers.d_delta(omega - (energy_2 - energy_1))

                if not i == j:
                    spectral_val += boltzmann_2 * (np.abs(mel)**2) * helpers.d_delta(omega - (energy_1 - energy_2))

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
def validate_rotor_spectrum(omega_vals, spectral_vals, beta: float) -> bool:
    dim = len(omega_vals)

    for omega_index in range(dim):
        omega = omega_vals[omega_index]
        m_omega_index = np.where(omega_vals == -omega)[0][0]

        if not (np.exp(beta * omega) * spectral_vals[m_omega_index] == spectral_vals[omega_index]):
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

    plt.plot(omega_vals, spinon_spectrum_vals)

    plt.xlabel(r'$\omega$')
    plt.title(r'Spinon')

    plt.tight_layout()

    plt.show()

    plt.plot(omega_vals, rotor_spectrum_vals)

    plt.xlabel(r'$\omega$')
    plt.title(r'Rotor')

    plt.tight_layout()

    plt.show()

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

if __name__ == '__main__':
    print_params = True
    print_spinon_spectral = False
    print_rotor_spectral = False
    print_electron_spectral = True
    l_max = 12
    ROTOR_DIM = 2 * l_max + 1

    t = 1.0
    U = -1.0
    beta = 50.0
    N_target = 2.0

    epsilon = np.zeros(2)

    t_eff_guess = 0.8 * t
    h_guess = np.full(2, -epsilon)
    Z_guess = np.full(2, 0.5)
    K_guess = np.full(2,-1.0)

    omega_vals = np.linspace(-10, 10, 20000)

    h, Z, K, t_eff, nf, L, mu, num_cycles = update_params(epsilon, t, U, 0.8 * t, h_guess, Z_guess, K_guess, beta, l_max, N_target)
    
    print(helpers.get_exact_spectrum(U, t, 2))
    if print_params:
        print(f'h: {h}')
        print(f'Z: {Z}')
        print(f'K: {K}')
        print(f't_eff: {t_eff}')
        print(f'nf: {nf}')
        print(f'L: {L}')
        print(f'N: {np.sum(nf)}')
        print(f'num_cycles: {num_cycles}\n')

    if print_spinon_spectral:
        spinon_spectral_vals = spinon_spectrum(omega_vals, epsilon, t_eff, h, mu)

        plt.plot(omega_vals, spinon_spectral_vals)
        plt.xticks(
            [-2, -1, 0, 1, 2],
            ['-2', '-1', '0', '1', '2'],
        )

        plt.yticks(
            [0, 100, 200],
            ['0', '100', '200']
        )

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\rho_f(\omega)$')
        plt.title(r'Spinon spectral density function for $\epsilon_0 = $' + f'{epsilon[0]}, ' + r'$N = $' + f'{N_target}')
        plt.tight_layout()

        plt.show()

    if print_rotor_spectral:
        rotor_spectral_vals = rotor_spectrum(omega_vals, U, h, K, beta, l_max)
        print(f'Good spectral function: {validate_rotor_spectrum(omega_vals, rotor_spectral_vals, beta)}')

        plt.plot(omega_vals, rotor_spectral_vals)

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\rho_f(\omega)$')
        plt.title(r'Rotor spectral function for $\epsilon_0 = $' + f'{epsilon[0]}, ' + r'$N = $' + f'{N_target}')
        plt.tight_layout()

        plt.show()

    if print_electron_spectral:
        electron_spectral = get_electron_spectral_vals(omega_vals, epsilon, U, t_eff, h, mu, beta, l_max)

        plt.plot(omega_vals, electron_spectral)

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\rho_d(\omega)$', rotation='horizontal')
        plt.title(r'Physical spectral function for $\epsilon_0 = $' + f'{epsilon[0]}, ' + r'$N = $' + f'{N_target}')

        plt.tight_layout()

        plt.show()
