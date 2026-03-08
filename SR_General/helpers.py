import numpy as np
from SR_General.utility.globals import *

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
