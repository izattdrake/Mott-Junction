import numpy as np
import matplotlib.pyplot as plt
import itertools 

from utility.helpers import delta, partition

"""
    Returns |l_1 l_2> basis elements
"""
def get_eff_rotor_basis(l_max):
    l_vals = np.arange(-l_max, l_max + 1)
    basis = list(itertools.product(
        l_vals, l_vals
    ))

    return np.array(basis)

"""
    Returns l_1 and l_2 corresponding to a given index
"""
def get_eff_rotor_index(l1, l2, l_max):
    dim = 2 * l_max + 1
    index = (l1 + l_max) * dim + (l2 + l_max)

    return index


"""
    Returns effective rotor matrix elements <l1,l2|H_{theta}|l1_prime, l2_prime>
"""
def get_eff_H_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime):
    diag = ((U / 2) * (l1**2 + l2**2) + h*(l1 + l2)) * delta(l1,l1_prime) * delta(l2,l2_prime)
    off_diag = -J_eff * (delta(l1, l1_prime + 1) * delta(l2, l2_prime - 1) + delta(l1, l1_prime - 1) * delta(l2, l2_prime + 1))
    
    return diag + off_diag


"""
    Returns effective rotor Hamiltonian matrix in the |l1,l2> basis
"""
def get_eff_H_rotor(U, h, J_eff, l_max):
    rotor_basis = get_eff_rotor_basis(l_max)
    dim_basis = len(rotor_basis)

    H_rotor = np.zeros((dim_basis, dim_basis))
    for i in range(dim_basis):

        l1, l2 = rotor_basis[i]

        for j in range(dim_basis):

            l1_prime, l2_prime = rotor_basis[j]
            H_rotor[i,j] = get_eff_H_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime)

    return H_rotor

"""
    Computes t_eff
"""
def get_t_eff(t, rotor_energies, rotor_states, beta, l_max):
    rotor_dim = len(rotor_energies)
    basis = get_eff_rotor_basis(l_max)

    # Shift rotor energies to avoid divergences, the formula's division cancels out the shift
    rotor_energies = rotor_energies - rotor_energies[0]

    num_sum = 0.0
    den_sum = 0.0

    for n in range(rotor_dim):
        rotor_energy = rotor_energies[n]
        boltzmann = np.exp(-beta * rotor_energy)
        state = rotor_states[n]

        for i in range(rotor_dim):
            rotor_factor = 0.0
            l1, l2 = basis[i]

            # Handle l1, l2 calls greater than l_max
            # May give more specific conditions later depending on accuracy
            if (l1 == -l_max) or (l2 == -l_max) or (l1 == l_max) or (l2 == l_max):
                rotor_factor = np.abs(state[i])**2
                
            else:
                shift_index_1 = get_eff_rotor_index(l1+1,l2-1)
                shift_index_2 = get_eff_rotor_index(l1-1,l2+1)

                rotor_factor = np.conj(state[i])*(state[shift_index_1] + state[shift_index_2])/2
            
            num_sum += boltzmann * rotor_factor
    
    t_eff = num_sum / den_sum

    return t_eff


if __name__ == '__main__':
    basis = get_eff_rotor_basis(2)
    print(basis)