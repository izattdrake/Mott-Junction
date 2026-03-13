import numpy as np

from utility.helpers import k_delta, get_fermion_basis

L_CUT = 2
ROTOR_DIM = (2 * L_CUT + 1)
PHYS_DIM = 4 * 4
SPINON_DIM = 16
DIM = SPINON_DIM * ROTOR_DIM * ROTOR_DIM

"""
    Returns physical spinon-rotor basis kets
"""
def get_phys_basis():
    basis = get_fermion_basis()
    phys_basis = []

    for q_nums in basis:
        n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
        occ1, occ2 = n1_up + n1_down, n2_up + n2_down

        if (l1 - occ1 + 1 == 0) and (l2 - occ2 + 1 == 0):
            phys_basis.append(q_nums)

    return phys_basis

"""
    Returns physical spinon-rotor basis kets of definite particle number
"""
def get_def_particle_basis(num_particles):
    basis = get_phys_basis()
    def_particle_basis = []

    for q_nums in basis:
        n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
        
        if n1_up + n1_down + n2_up + n2_down == num_particles:
            def_particle_basis.append(q_nums)

    return def_particle_basis

"""
    Takes 6 quantum numbers labeling the slave-rotor state and returns the basis index
"""
def get_index(q_nums):
    n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
    i = ((2*L_CUT + 1) ** 2) * (8*n1_up + 4*n1_down + 2*n2_up + n2_down) + (2*L_CUT + 1) * (l2 + L_CUT) + (l1 + L_CUT)

    return i

"""
    Returns a matrix representation for the constraint operator of a site.
    The constraint operator of site i=0,1 has 0 eigenvalues for physical states on site i
    and non-zero eigenvalues for non-physical states on site i
    Physical states must have zero eigenvalues on both dimer sites
"""
def get_constraint_operator(site):
    mixed_basis = get_fermion_basis()
    operator = np.zeros((DIM, DIM))

    if (not site == 0) and (not site == 1):
        raise ValueError('Site must equal 0 or 1.')
    else:
        for i in range(DIM):
            q_nums = mixed_basis[i]
            n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
            if site == 0:
                operator[i,i] = l1 - (n1_up + n1_down) + 1
            elif site == 1:
                operator[i,i] = l2 - (n2_up + n2_down) + 1

    return operator


"""
    Look at the Latex document to see where I got this.
    An ugly Hamiltonian, it is
"""
def get_mel(epsilon, U, t, q_nums, q_nums_prime):
    n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
    n1_up_prime, n1_down_prime, n2_up_prime, n2_down_prime, l1_prime, l2_prime = q_nums_prime

    mat_el = (
        (
            epsilon*(n1_up + n1_down + n2_up + n2_down) + (U/2)*(l1**2 + l2**2)
        ) * k_delta(n1_up, n1_up_prime) * k_delta(n1_down, n1_down_prime) * k_delta(n2_up, n2_up_prime) * k_delta(n2_down, n2_down_prime) * k_delta(l1, l1_prime) * k_delta(l2, l2_prime)
        - t * (
              ((-1) ** (n1_down_prime)) * k_delta(0, n1_up_prime) * k_delta(1, n2_up_prime) 
            * k_delta(n1_up, n1_up_prime + 1) * k_delta(n1_down, n1_down_prime)
            * k_delta(n2_up, n2_up_prime - 1) * k_delta(n2_down, n2_down_prime)
            * k_delta(l1, l1_prime + 1) * k_delta(l2, l2_prime - 1)

            + ((-1) ** (n1_down_prime)) * k_delta(1, n1_up_prime) * k_delta(0, n2_up_prime) 
            * k_delta(n1_up, n1_up_prime - 1) * k_delta(n1_down, n1_down_prime)
            * k_delta(n2_up, n2_up_prime + 1) * k_delta(n2_down, n2_down_prime)
            * k_delta(l1, l1_prime - 1) * k_delta(l2, l2_prime + 1)

            + ((-1) ** (n2_up_prime)) * k_delta(0, n1_down_prime) * k_delta(1, n2_down_prime)
            * k_delta(n1_up, n1_up_prime) * k_delta(n1_down, n1_down_prime + 1) 
            * k_delta(n2_up, n2_up_prime) * k_delta(n2_down, n2_down_prime - 1)
            * k_delta(l1, l1_prime + 1) * k_delta(l2, l2_prime - 1)

            + ((-1) ** (n2_up_prime)) * k_delta(1, n1_down_prime) * k_delta(0, n2_down_prime) 
            * k_delta(n1_up, n1_up_prime) * k_delta(n1_down, n1_down_prime - 1)
            * k_delta(n2_up, n2_up_prime) * k_delta(n2_down, n2_down_prime + 1)
            * k_delta(l1, l1_prime - 1) * k_delta(l2, l2_prime + 1)
        )
    )

    return mat_el
    

"""
    Returns a Hamiltonian constructed from a basis
    Mixed, physical, or definite particle bases can be input to get the desired Hamiltonian
"""
def get_H(epsilon, U, t, basis):
    dim = len(basis)
    H = np.zeros((dim, dim))

    for i in range(dim):
        q_nums = basis[i]

        for j in range(dim):
            q_nums_prime = basis[j]
            H[i,j] = get_mel(epsilon, U, t, q_nums, q_nums_prime)

    return H

"""
    Takes the exact SR Hamiltonian's eigenvectors and eigenvalues then returns states satisfying the number constraint and their respective eigenvalues
"""
def get_phys_from_mixed(H, mixed_basis):
    mixed_vals, mixed_vecs = np.linalg.eigh(H)
    phys_vals, phys_states = [], []

    for i in range(DIM):
        q_nums = mixed_basis[i]
        n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
    
        occ1 = n1_up + n1_down
        occ2 = n2_up + n2_down

        if (l1 - occ1 + 1 == 0) and (l2 - occ2 + 1 == 0):
            phys_val, phys_state = mixed_vals[i], mixed_vecs[:,i]
            phys_vals.append(phys_val)
            phys_states.append(phys_state)

    phys_vals, phys_states = np.array(phys_vals), np.array(phys_states)

    return phys_vals, phys_states

"""
    Return Hamiltonian truncated over basis of physical states that satisfy the gauge constraint
"""
def truncate_mixed_H(H, phys_states):
    dim_states = len(phys_states)
    H_phys = np.zeros((dim_states, dim_states))

    for i in range(dim_states):
        for j in range(dim_states):
            bra, ket = phys_states[i], phys_states[j]
            H_phys[i,j] = bra.T @ H @ ket
    
    return H_phys