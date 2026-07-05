import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility import helpers

# ====================================================================
#   CORE FUNCTIONALITY FOR DECOUPLED ROTOR PROBLEM ON A HUBBARD DIMER
# --------------------------------------------------------------------
#   A mean-field approximation is made on the exact slave rotor
#   Hubbard Hamiltoniain, giving a spinon and rotor Hamiltonian.
#   Another approximation is made to decouple rotors, leading
#   to a tower of rotor states at each lattice site. This
#   approximation is of questionable accuracy, so a big question is
#   how effective it is on a dimer.
# --------------------------------------------------------------------
#   At each lattice site, there is a tower of rotor basis states |l>
#   A rotor state is a superposition \sum \alpha_l |l> ranging from
#   l \in [-l_max, l_max]. This file features all the functionality to
#   calculate spinon and rotor Hamiltonians, parameters, and some
#   other things such as energies and real-space wavefunction 
#   projections. 
# --------------------------------------------------------------------
#   Specific iteration methods are handled in other files since
#   multiple are used to calculate and compare results. 
# ====================================================================

N_AT = 2.0

# Used in energy calculations, each dimer_states element is the set of
# states with that particle number
dimer_states = [{
        '0000': (0,0,0,0)
    }, {
        '1000': (1,0,0,0),
        '0100': (0,1,0,0),
        '0010': (0,0,1,0),
        '0001': (0,0,0,1)
    }, {
        '1100': (1,1,0,0),
        '1010': (1,0,1,0),
        '1001': (1,0,0,1),
        '0110': (0,1,1,0),
        '0101': (0,1,0,1),
        '0011': (0,0,1,1)
    }, {
        '1110': (1,1,1,0),
        '1101': (1,1,0,1),
        '1011': (1,0,1,1),
        '0111': (0,1,1,1)
    }, {
        '1111': (1,1,1,1)
    }
]

def get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu):
    """
    Return 2x2 Spinon Hamiltonian in a spin subspace
    """
    H = np.diag(epsilon_vec - h_vec - mu) - t_eff_mat
    return H

def get_H_rotor(U, h_vec, K_vec, l_max, idx):
    """
    Returns rotor Hamiltonian at site idx
    """
    dim = 2*l_max + 1
    l_vals = np.arange(-l_max, l_max + 1)

    diag = ((U/2.0)*l_vals + h_vec[idx]) * l_vals 
    off_diag = K_vec[idx] / 2.0
    
    H = np.diag(diag)
    H += np.diag(np.full(dim - 1, off_diag), k=1) + np.diag(np.full(dim - 1, off_diag), k=-1)

    return H

def get_H_rotor_vec(U, h_vec, K_vec, l_max):
    """
    Returns array of rotor Hamiltonians for each site
    """
    dim = 2*l_max + 1
    H_vec = np.zeros((2, dim, dim))
    
    for i in range(2):
        H_vec[i] = get_H_rotor(U, h_vec, K_vec, l_max, i)

    return H_vec

def get_rotor_eig_vec(U, h_vec, K_vec, l_max):
    """
    Return arrays containing the eigenvalues and eigenvectors of H_theta at every site
    """
    dim = 2*l_max + 1
    eigvals_vec = np.zeros((2, dim))
    eigvecs_vec = np.zeros((2, dim, dim))

    for i in range(2):
        eigvals_vec[i], eigvecs_vec[i] = np.linalg.eigh(get_H_rotor(U, h_vec, K_vec, l_max, i))

    return eigvals_vec, eigvecs_vec

def get_nf_vec(spinon_energies, spinon_states, beta):
    """
    Returns array of spinon occupancies at each dimer site
    """
    nf = np.zeros(2)

    for i in range(2):
        sum = 0.0
        for j in range(2):
            energy = spinon_energies[j]
            sum += (np.abs(spinon_states[i,j]) ** 2) * helpers.fermi_dirac(energy, beta)

        nf[i] = N_AT * sum

    return nf

def get_L(rotor_eigvals, rotor_eigvecs, beta, l_max):
    """
    Return thermal average of L
    Arguemnts are eigenvalues and eigenvectors at a site
    """
    evals = rotor_eigvals - rotor_eigvecs[0]
    evecs = rotor_eigvecs

    bz = np.exp(-beta * evals)

    l_vals = np.arange(-l_max, l_max + 1)
    l_expect_per_state = np.sum((np.abs(evecs)**2) * l_vals[:,np.newaxis], axis=0)

    L = np.dot(bz, l_expect_per_state) / np.sum(bz)
    return L

def get_L_vec(rotor_evals_vec, rotor_evecs_vec, beta, l_max):
    """
    Return thermal expectation value of L with the rotor Hamiltonian at each lattice site
    """
    L_vec = np.zeros(2)

    # Sum over lattice sites
    for i in range(2):
        # Picks the list of eigenvalues at a lattice site [...] and subtract the gs energy at that site
        evals = rotor_evals_vec[i] - rotor_evals_vec[i,0]
        eigvecs = rotor_evecs_vec[i]

        L_vec[i] = get_L(evals, eigvecs, beta, l_max)

    return L_vec

def get_Jacobian(epsilon_vec, U, t_eff_mat, h_vec, dh, mu, K_vec, beta, l_max):
    """
    Returns Jacobian used in Newton's method
    """
    def get_Jacobian_nf(epsilon_vec, t_eff_mat, h_vec, dh, mu, beta):
        """
        Returns spinon contribution to Jacobian
        """
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
        """
        Returns rotor contribution to Jacobian
        """
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
    """
    COMPUTE VECTOR OF Z-VALUES AT EACH SITE

    Z = m/m^* is the ratio of quasiparticle mass to electron mass in this problem
    Z vanishes at the Mott transition as the quasiparticle mass blows up,
    corresponding to vanishing conductivity.
    """
    Z_vec = np.zeros(2)

    for i in range(2):
        evals = rotor_evals_vec[i] - rotor_evals_vec[i,0]
        evecs = rotor_evecs_vec[i]

        bz = np.exp(-beta * evals)
        mel = (evecs[:-1, :].conj() * evecs[1:, :]).sum(axis=0)
        Z_vec[i] = (np.dot(bz, mel) / np.sum(bz))**2

    return Z_vec

def get_t_eff(t_mat, Z_vec):
    """
    COMPUTE EFFECTIVE TUNNELING MATRIX
    """
    sqrt_Z = np.sqrt(Z_vec)
    t_eff = t_mat * np.outer(sqrt_Z, sqrt_Z)

    return t_eff

def get_K_vec(t_mat, Z_vec, spinon_evals, spinon_evecs, beta):
    """
    Returns array of K values at each lattice site
    K is analogous to J_eff in the coupled rotor problem
    Related to rotor tunneling amplitude after the second MF approximation
    """
    fd = np.array([helpers.fermi_dirac(energy, beta) for energy in spinon_evals])
    weighted = spinon_evecs * fd[np.newaxis,:]
    rho = spinon_evecs.conj() @ weighted.T
    sqrt_Z = np.sqrt(Z_vec)
    K_vec = np.real(-2 * N_AT * np.sum(t_mat * sqrt_Z[np.newaxis,:] * rho, axis=1))

    return K_vec

def rotor_state_ps(theta, state, l_max):
    """
    Projects rotor state onto e^{i*l*theta} basis
    """
    l_vals = np.arange(-l_max, l_max + 1)
    return state @ np.exp(1j * l_vals * theta)

def construct_phys_states(spinon, rotor1, rotor2, l_max):
    '''
        Construct physical states with combined spinon/rotor coefficients 
        Satisfy constraint in basis states
        c|n>|l1=n1-1>|l2=n2-1>
    '''

    # Arrays containing weights with spinon x rotor weights, and occupation numbers
    phys_weights = []
    phys_occ = []

    P2_dimer_states = dimer_states[2]

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
        ((helpers.k_delta(0, n1_up_prime)*helpers.k_delta(1,n2_up_prime)
        *helpers.k_delta(n1_up, n1_up_prime+1)*helpers.k_delta(n1_down, n1_down_prime)*helpers.k_delta(n2_up,n2_up_prime-1)*helpers.k_delta(n2_down,n2_down_prime))
        +(helpers.k_delta(0,n1_down_prime)*helpers.k_delta(1,n2_up_prime)
        *helpers.k_delta(n1_up,n1_up_prime)*helpers.k_delta(n1_down,n1_down_prime+1)*helpers.k_delta(n2_up,n2_up_prime)*helpers.k_delta(n2_down,n2_down_prime-1)))
        *helpers.k_delta(nf1,nf1_prime+1)*helpers.k_delta(nf2,nf2_prime-1)
    )

    off_diag_21 = -2*t_nn*(
        ((helpers.k_delta(0, n2_up_prime)*helpers.k_delta(1,n1_up_prime)
        *helpers.k_delta(n2_up, n2_up_prime+1)*helpers.k_delta(n2_down, n2_down_prime)*helpers.k_delta(n1_up,n1_up_prime-1)*helpers.k_delta(n1_down,n1_down_prime))
        +(helpers.k_delta(0,n2_down_prime)*helpers.k_delta(1,n1_up_prime)
        *helpers.k_delta(n2_up,n2_up_prime)*helpers.k_delta(n2_down,n2_down_prime+1)*helpers.k_delta(n1_up,n1_up_prime)*helpers.k_delta(n1_down,n1_down_prime-1)))
        *helpers.k_delta(nf2,nf2_prime+1)*helpers.k_delta(nf1,nf1_prime-1)
    )

    mel = diag + off_diag_12 + off_diag_21
    return mel


def get_E_dimer_phys(epsilon_vec, U, t_nn, rotor1, rotor2, l_max):
    """
        Returns 4 energies corresponding to the constrained tensor product |\psi_f>|\psi_{theta,1}>|\psi_{theta,2}>
        where |\psi_f> takes on 1 of the 4 calculated eigenkets for the spinon Hamiltonian
        This is highly specific to the homogeneous dimer problem
    """
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

    P2_dimer_states = dimer_states[2]

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

def get_E_gs(epsilon_vec, U, h_vec, t_eff_mat, Z_vec, K_vec, mu, l_max):
    """
    Calculate ground-state energy w/o physical subspace projection
    """
    def get_E0(Z_vec, K_vec):
        return -np.sum(K_vec * np.sqrt(Z_vec))
    
    spinon_evals, _ = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu))
    rotor_evals, _ = np.linalg.eigh(get_H_rotor(U, h_vec, K_vec, l_max, 0))

    E0 = get_E0(Z_vec, K_vec)

    E_gs = 2*spinon_evals[0] + 2*rotor_evals[0] + E0
    return E_gs