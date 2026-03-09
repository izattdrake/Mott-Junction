import numpy as np

KB_SI = 1.380649 * 10**(-23)
KB_EV = 8.617333262 * 10**(-5)

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
        sum += np

    return sum

"""
Computes Z from U, k, and h with 1st-order perturbation theory
Only accurate for very small Z close to the Mott transition
"""
def check_Z(U, K, h, nf, N_AT=2):
    x = N_AT * (nf - 0.5)
    Z = (2 * U * K/(U * U - 4 * (U * x + h))) ** 2

    return Z