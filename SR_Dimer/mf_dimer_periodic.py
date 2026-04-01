import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys 
import os

import utility.plotting as SR_plotting
from utility.helpers import *

warnings.simplefilter("error", RuntimeWarning)

N_AT = 2

def get_nf(spinon_eigvals, spinon_eigvecs, beta):
    nf = np.zeros(2)

    for i in range(2):
        sum = 0.0
        for j in range(2):
            energy = spinon_eigvals[j]
            sum += (np.abs(spinon_eigvecs[i,j]) ** 2) * fermi_dirac(energy, beta)

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
        sums[0] += t * np.sqrt(Z[1]) * spinon_eigvecs.conj().T[i,0] * spinon_eigvecs[1,i] * fermi_dirac(energy, beta)
        sums[1] += t * np.sqrt(Z[0]) * spinon_eigvecs.conj().T[i,1] * spinon_eigvecs[0,i] * fermi_dirac(energy, beta)
    
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

if __name__ == '__main__':
    print_params = True
    plot_gs = False 
    plot_ZU = False 
    plot_eff_m = True
    plot_nfepsilon = False 

    l_max = 12
    ROTOR_DIM = 2 * l_max + 1

    t = 1.0
    beta = 50.0
    N_target = 2.0

    # epsilon used for on-site energy - chemical potential 

    gs_vals = []
    theta_vals = np.linspace(0, 2*np.pi, 200)

    Z_vals = []
    U_vals = [2.0]

    epsilon_single_vals = np.linspace(-3.9, 4.0, 100)  # 0.0, 0.1, ..., 1.0
    epsilon_vals = np.column_stack((epsilon_single_vals,epsilon_single_vals))

    nf_vals = []
    eff_m_vals = []
    for U in U_vals:
        for epsilon in epsilon_vals:
            t_eff_guess = 0.8 * t
            h_guess = np.full(2, -epsilon)
            Z_guess = np.full(2, 0.5)
            K_guess = np.full(2,-1.0)

            h, Z, K, t_eff, nf, L, mu, num_cycles = update_params(epsilon, t, U, 0.8 * t, h_guess, Z_guess, K_guess, beta, l_max, N_target)
            
            if print_params:
                print(f'h: {h}')
                print(f'Z: {Z}')
                print(f'K: {K}')
                print(f't_eff: {t_eff}')
                print(f'nf: {nf}')
                print(f'L: {L}')
                print(f'N: {np.sum(nf)}')
                print(f'num_cycles: {num_cycles}\n')

            if plot_gs:
                rotor_site_eigenvecs, rotor_site_eigenvals = np.zeros((2, ROTOR_DIM, ROTOR_DIM)), np.zeros((2, ROTOR_DIM))
                H_rotors = get_H_rotors(U, h, K, l_max)
                for i in range(2):
                    rotor_site_eigenvals[i], rotor_site_eigenvecs[i] = np.linalg.eigh(H_rotors[i])

                    # Assume eigenvectors at each site are the same (THEY ARE FOR NO CHEMICAL POTENTIAL DIFFERENTIAL)
                    rotor_eigenvecs = rotor_site_eigenvecs[0]
                    rotor_gs = rotor_eigenvecs[:,0]
                    
                    rotor_gs_vals = np.array([np.abs(rotor_state_ps(theta, rotor_gs, l_max)) for theta in theta_vals])
                    gs_vals.append(rotor_gs_vals)

            if plot_ZU:
                Z_vals.append(Z)

            if plot_eff_m:
                eff_m_vals.append(1/Z)
            
            if plot_nfepsilon:
                nf_vals.append(nf[0])

    if plot_gs:
        plt.plot(theta_vals, gs_vals[0], label="U/t=0.01")
        plt.plot(theta_vals, gs_vals[1], label="U/t=0.1")
        plt.plot(theta_vals, gs_vals[2], label="U/t=0.5")
        plt.plot(theta_vals, gs_vals[3], label="U/t=1.0")
        plt.plot(theta_vals, gs_vals[4], label="U/t=2.0")
        plt.plot(theta_vals, gs_vals[5], label="U/t=2.5")
        plt.plot(theta_vals, gs_vals[6], label="U/t=3.0")
        plt.plot(theta_vals, gs_vals[7], label="U/t=3.5")
        plt.plot(theta_vals, gs_vals[8], label="U/t=3.6")

        # Customize x-axis ticks
        plt.xticks(
            [0, np.pi, 2*np.pi],
            [r'$0$', r'$\pi$', r'$2\pi$']
        )

        # Labels and legend
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\Psi_0(\theta)$')
        plt.legend()
        plt.title(r'Rotor ground state at a range of $U/t$ at $\epsilon_0 =$ ' + f'{epsilon[0]}' + r', $\beta = $' + f'{beta}')
        # Optional: improve layout
        plt.tight_layout()

        # Show plot
        plt.show()

    if plot_ZU:
        plt.plot(U_vals, Z_vals)

        plt.xticks(
            [0, 1, 2, 3, 4, 5],
            ['0', '1', '2', '3', '4', '5']
        )

        plt.xlabel(r'$U/t$')
        plt.ylabel(r'$Z$')
        plt.title(r'Quasiparticle weight $Z$ as a function of $U/t$ at $\epsilon = $' + f'{epsilon[0]}, ' + r'$\beta = $' + f'{beta}')
        plt.tight_layout()

        plt.show()

    if plot_eff_m:
        plt.plot(epsilon_single_vals, eff_m_vals)

        plt.xticks(
            [-4, -2, 0, 2, 4],
            ['-4', '-2', '0', '2', '4'],
        )

        plt.xlabel(r'$\epsilon_0$')
        plt.ylabel(r'$m^*/m$')
        plt.title(r'Quasiparticle weight $Z$ as a function of $\epsilon_0$ at $U = 2.0$')
        plt.tight_layout()

        plt.show()

    if plot_nfepsilon:
        plt.plot(epsilon_single_vals, nf_vals)
        plt.xticks(
            [-6, -4, -2, 0, 2, 4, 6],
            ['-6', '-4', '-2', '0', '2', '4', '6'],
        )

        plt.yticks(
            [0, 1, 2, 3, 4],
            ['0', '1', '2', '3', '4']
        )
        plt.xlabel(r'$\epsilon_0$')
        plt.ylabel(r'$n$')
        plt.title(r'Occupancy for N=4')
        plt.tight_layout()

        plt.show()