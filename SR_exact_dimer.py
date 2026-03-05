import numpy as np
import matplotlib.pyplot as plt
import itertools

from helpers import delta

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

    phys_vals, phys_states = np.array(phys_vals), np.array(phys_states).T

    return phys_vals, phys_states

if __name__ == '__main__':
    epsilon = 0.0
    U = 10.0
    t = 1.0

    basis = get_basis()

    H = get_H(epsilon, U, t, basis)
    phys_vals, phys_states = get_phys_from_q_nums(H, basis)
    print(phys_states[:,5])
