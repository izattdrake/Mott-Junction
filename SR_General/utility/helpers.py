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
    x = beta * (E - mu) / 2

    # Assume at large enough beta, the temperature is zero
    # May solve computational singularities at small temperature
    # Need to play around with it and understand natural energy scales better
    if beta > 50:
        n = np.where(E < mu, 1.0, 0.0)
    else:
        n = 0.5*(1-np.tanh(x))

    return n

"""
    Bose-Einstein distribution function
"""
def bose_einstein(E, beta):
    x = beta * E

    if x > 0:
        x_safe = np.maximum(x, 1e-12)
    elif x < 0:
        x_safe = -np.maximum(np.abs(x), 1e-12)

    n = 1 / np.expm1(x_safe)

    return n

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

def get_exact_spectrum(U, t, N):
    if N == 0:
        energies = [0]
    elif N == 1:
        energies = [-t,-t,t,t]
    elif N == 2:
        c = np.sqrt(U**2 + 16 * t**2)
        energies = [(U - c) / 2, 0, 0, 0, U, (U + c) / 2]
    elif N == 3:
        energies = [U - t, U - t, U + t, U + t]
    elif N == 4:
        energies = [2 * U]
    else:
        raise ValueError('N must be 0,1,2,3,4')
    
    return energies

"""
    Returns the hopping distance
    Checks a hopping array and returns the first zero after finite values
    Tells you how far hopping extends
    Used for reducing computation by only looping over finite values
"""
def get_t_dist(t):
    seen_nonzero = False
    hopping_dist = next(
        (i for i, x in enumerate(t)
         if (seen_nonzero := seen_nonzero or x != 0) and x == 0),
         None
    )

    return hopping_dist