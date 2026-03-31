import numpy as np
import itertools

"""
    Constants used for computation
"""

HBAR_SI = 1.054571817 * 1e-34
KB_SI = 1.380649 * 1e-23

HBAR_EV = 6.582119569 * 1e-16
KB_EV = 8.617333262 * 1e-5


"""
    Returns spinon basis of the form |0000>, |1000>, |1100>,...
"""
def get_spinon_basis():    
    basis = list(itertools.product(
        [0,1], [0,1], [0,1], [0,1]
    ))
    
    return np.array(basis)

"""
    Returns mixed basis 
"""
def get_mixed_basis():
    l_vals = [-2,-1,0,1,2]
    basis = list(itertools.product(
        [0,1], [0,1], [0,1], [0,1],
        l_vals, l_vals
    ))

    return basis
"""
    Kronecker Delta function
"""
def k_delta(x, y):
    return int(x == y)


"""
    Dirac Delta function

    x: int; Input value
    eta: float; Small parameter, Lorentzian goes to delta function as \eta -> 0
"""
def d_delta(x, eta=1e-4):
    lorentz = (eta/np.pi)/(x**2 + eta**2)
    return lorentz

"""
    Fermi-Dirac distribution function
"""
def fermi_dirac(E, beta, mu=0):
    x = (E - mu) * beta / 2
    occ = 0.0

    # Assume at large enough beta, the temperature is zero
    # May solve computational singularities at small temperature
    # Need to play around with it and understand natural energy scales better
    if beta > 50:
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

def exact_1P_energies(t):
    return [-t,-t,t,t]

def exact_2P_energies(U,t):
    c = np.sqrt(U**2 + 16 * t**2)
    return [(U - c) / 2, 0, 0, 0, U, (U + c) / 2]

def exact_3P_energies(U,t):
    return [U - t, U - t, U + t, U + t]

def exact_4P_energies(U):
    return 2 * U