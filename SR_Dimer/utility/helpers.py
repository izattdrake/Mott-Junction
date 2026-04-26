import numpy as np
import warnings
import itertools
import scipy

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
    Return vectorized fermi-dirac for all eigenvalues
"""
def fermi_dirac_vec(energies, beta):
    fd = 1.0/(np.exp(beta * energies) + 1.0)
    return fd

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

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        try:
            if x > 0:
                n = 1.0 / (np.exp(x) - 1.0 + 1e-10)
            elif x < 0:
                n = 1.0 / (np.exp(x) - 1.0 - 1e-10)
        except RuntimeWarning as e:
            n = 0.0
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

"""
    Returns physical specturm peaks
    For some reason, whether or not the spectrum is entirely >0 or <0 
    depends on l_max. I'm assuming the peak's locations at \omega are still accurate
    so that it doesn't matter whether everything's down or up
"""
def get_spectrum_peaks(spectrum_vals):
    return 

def print_params(t_eff, h, mu, Z, K, nf, L, should_print=True):
    if should_print:
        print(f'h: {h}')
        print(f'Z: {Z}')
        print(f'K: {K}')
        print(f't_eff: {t_eff}')
        print(f'nf: {nf}')
        print(f'L: {L}')
        print(f'mu: {mu}\n')

"""
    Gets peaks in a function
"""
def get_peaks_idx(spectrum):
    peaks, _ = scipy.signal.find_peaks(spectrum)
    return peaks
