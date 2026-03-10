import numpy as np
import matplotlib.pyplot as plt
import itertools 

from utility.helpers import delta

"""
    Returns |l_1 l_2> basis elements
"""
def get_rotor_basis(l_max):
    l_vals = np.arange(-l_max, l_max + 1)

    basis = list(itertools.product(
        l_vals, l_vals
    ))
  
def get_eff_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime):
    diag = ((U / 2) * (l1**2 + l2**2) + h*(l1 + l2)) * delta(l1,l1_prime) * delta(l2,l2_prime)
    off_diag = -J_eff * (delta(l1, l1_prime + 1) * delta(l2, l2_prime - 1) + delta(l1, l1_prime - 1) * delta(l2, l2_prime + 1))
    
    return diag + off_diag


def get_eff_H_rotor(U, h, J_eff, l_max):
    rotor_basis = get_rotor_basis(l_max)
    dim_basis = len(rotor_basis)

    H_rotor = np.zeros((dim_basis, dim_basis))
    for i in range(dim_basis):
        l1, l2 = rotor_basis[i]
        for j in range(dim_basis):
            l1_prime, l2_prime = rotor_basis[j]

            H_rotor[i,j] = get_eff_rotor_mel(U, h, J_eff, l1, l2, l1_prime, l2_prime)

    return H_rotor