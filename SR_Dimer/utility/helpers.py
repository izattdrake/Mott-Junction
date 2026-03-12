import numpy as np
import itertools

"""
    Constants used for computation
"""
KB_SI = 1.380649 * 10**(-23)
KB_EV = 8.617333262 * 10**(-5)


"""
    Returns spinon basis of the form |0000>, |1000>, |1100>,...
"""
def get_fermion_basis():    
    basis = list(itertools.product(
        [0,1], [0,1], [0,1], [0,1]
    ))
    
    return np.array(basis)


"""
    Kronecker Delta function
"""
def delta(x, y):
    return int(x == y)

"""
    Fermi-Dirac distribution function
"""
def fermi_dirac(E, beta, mu=0):
    x = (E - mu) * beta / 2
    occ = 0.0

    # Assume at large enough beta, the temperature is zero
    # May solve computational singularities at small temperature
    # Need to play around with it and understand natural energy scales better
    if beta > 500:
        occ = np.where(E < mu, 1.0, 0.0)
    else:
        occ = 0.5*(1-np.tanh(x))

    return occ

"""
    Partition function
"""
def partition(energies, beta):
    sum = 0.0

    for energy in energies:
        boltzmann = np.exp(-beta * energy)
        sum += boltzmann

    return sum

"""
Computes Z from U, k, and h with 1st-order perturbation theory
Only accurate for very small Z close to the Mott transition
"""
def check_Z(U, K, h, nf, N_AT=2):
    x = N_AT * (nf - 0.5)
    Z = (2 * U * K/(U * U - 4 * (U * x + h))) ** 2

    return Z