import numpy as np
import matplotlib.pyplot as plt
import itertools

from SR_General.utility.helpers import delta

L_CUT = 2
ROTOR_DIM = (2 * L_CUT + 1)
SPINON_DIM = 16
DIM = SPINON_DIM * ROTOR_DIM * ROTOR_DIM

"""
    Returns the basis of 6 quantum numbers, ordered by their index
"""
def get_basis():
    ell_vals = range(-L_CUT, L_CUT + 1)
    basis = list(itertools.product(
        [0,1], [0,1],
        [0,1], [0,1],
        ell_vals, ell_vals
    ))
    return basis


"""
    Takes 6 quantum numbers labeling the slave-rotor state and returns the basis index
"""
def get_index(q_nums):
    n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
    i = ((2*L_CUT + 1) ** 2) * (8*n1_up + 4*n1_down + 2*n2_up + n2_down) + (2*L_CUT + 1) * (l2 + L_CUT) + (l1 + L_CUT)

    return i


"""
    Look at the Latex document to see where I got this.
    An ugly Hamiltonian, it is
"""
def get_matrix_element(epsilon, U, t, q_nums, q_nums_prime):
    n1_up, n1_down, n2_up, n2_down, l1, l2 = q_nums
    n1_up_prime, n1_down_prime, n2_up_prime, n2_down_prime, l1_prime, l2_prime = q_nums_prime

    mat_el = (
        (
            epsilon*(n1_up + n1_down + n2_up + n2_down) + (U/2)*(l1**2 + l2**2)
        ) * delta(n1_up, n1_up_prime) * delta(n1_down, n1_down_prime) * delta(n2_up, n2_up_prime) * delta(n2_down, n2_down_prime) * delta(l1, l1_prime) * delta(l2, l2_prime)
        - t * (
              delta(0, n1_up_prime) * delta(1, n2_up_prime) 
            * delta(n1_up, n1_up_prime + 1) * delta(n1_down, n1_down_prime)
            * delta(n2_up, n2_up_prime - 1) * delta(n2_down, n2_down_prime)
            * delta(l1, l1_prime + 1) * delta(l2, l2_prime - 1)

            + delta(1, n1_up_prime) * delta(0, n2_up_prime) 
            * delta(n1_up, n1_up_prime - 1) * delta(n1_down, n1_down_prime)
            * delta(n2_up, n2_up_prime + 1) * delta(n2_down, n2_down_prime)
            * delta(l1, l1_prime - 1) * delta(l2, l2_prime + 1)

            + delta(0, n1_down_prime) * delta(1, n2_down_prime)
            * delta(n1_up, n1_up_prime) * delta(n1_down, n1_down_prime + 1) 
            * delta(n2_up, n2_up_prime) * delta(n2_down, n2_down_prime - 1)
            * delta(l1, l1_prime + 1) * delta(l2, l2_prime - 1)

            + delta(1, n1_down_prime) * delta(0, n2_down_prime) 
            * delta(n1_up, n1_up_prime) * delta(n1_down, n1_down_prime - 1)
            * delta(n2_up, n2_up_prime) * delta(n2_down, n2_down_prime + 1)
            * delta(l1, l1_prime - 1) * delta(l2, l2_prime + 1)
        )
    )

    return mat_el
    
def get_H(epsilon, U, t, basis):
    H = np.zeros((DIM, DIM))

    for i in range(DIM):
        q_nums = basis[i]

        for j in range(DIM):
            q_nums_prime = basis[j]
            H[i,j] = get_matrix_element(epsilon, U, t, q_nums, q_nums_prime)

    return H

"""
    Takes the exact SR Hamiltonian's eigenvectors and eigenvalues then returns states satisfying the number constraint and their respective eigenvalues
"""
def get_phys_from_q_nums(H, basis):
    mixed_vals, mixed_vecs = np.linalg.eigh(H)
    phys_vals, phys_states = [], []

    for i in range(DIM):
        q_nums = basis[i]
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
def get_phys_H(H, phys_states):
    dim_states = len(phys_states)
    H_phys = np.zeros((dim_states, dim_states))

    for i in range(dim_states):
        for j in range(dim_states):
            bra, ket = phys_states[i], phys_states[j]
            H_phys[i,j] = bra.T @ H @ ket
    
    return H_phys


if __name__ == '__main__':
    epsilon = 10.0
    U = 10.0
    t = 1.0

    index1 = get_index((0,0,0,0,-1,-1))
    index2 = get_index((1,0,0,0,0,-1))
    index3 = get_index((0,1,0,0,0,-1))
    index4 = get_index((0,0,1,0,-1,0))
    index5 = get_index((0,0,0,1,-1,0))
    index6 = get_index((1,1,0,0,1,-1))
    index7 = get_index((1,0,1,0,0,0))
    index8 = get_index((1,0,0,1,0,0))
    index9 = get_index((0,1,1,0,0,0))
    index10 = get_index((0,1,0,1,0,0))
    index11 = get_index((0,0,1,1,-1,1))
    index12 = get_index((1,1,1,0,1,0))
    index13 = get_index((1,1,0,1,1,0))
    index14 = get_index((1,0,1,1,0,1))
    index15 = get_index((0,1,1,1,0,1))
    index16 = get_index((1,1,1,1,1,1))

    print(f'Index 1: {index1}')
    print(f'Index 2: {index2}')
    print(f'Index 3: {index3}')
    print(f'Index 4: {index4}')
    print(f'Index 5: {index5}')
    print(f'Index 6: {index6}')
    print(f'Index 7: {index7}')
    print(f'Index 8: {index8}')
    print(f'Index 9: {index9}')
    print(f'Index 10: {index10}')
    print(f'Index 11: {index11}')
    print(f'Index 12: {index12}')
    print(f'Index 13: {index13}')
    print(f'Index 14: {index14}')
    print(f'Index 15: {index15}')
    print(f'Index 16: {index16}')


