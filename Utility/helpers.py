import numpy as np
import warnings
import itertools
import scipy

#Constants used for computation
HBAR_SI = 1.054571817 * 1e-34
HBAR_EV = 6.582119569 * 1e-16
KB_SI = 1.380649 * 1e-23
KB_EV = 8.617333262 * 1e-5

def _kdelta(x, y):
    """Kronecker delta"""
    d_xy = int(x==y)
    return d_xy


def _ddelta(x, eta=1e-4):
    """Dirac delta function | Calculated with Lorentzian as eta -> 0"""
    lorentz = (eta/np.pi)/(x**2 + eta**2)
    return lorentz

def _bz_weights(energies, beta):
    """Calculates normalized Boltzmann probabilities"""
    if beta is None:                          
        bz = (energies - energies[0] < 1e-10).astype(float) # Lock all weight to the GS
    else:
        bz = np.exp(-beta * (energies - energies[0]))
        
    return bz / bz.sum()

def _fermi(E, beta, mu=0.0):
    """Calculate Fermi-Dirac distribution | T=0 handled by beta=None"""
    if beta is None:
        fd_val = np.where(E < mu, 1.0, np.where(E > mu, 0.0, 0.5)) # E=mu returns 0.5 at zero-temperature
    else:
        fd_val = 0.5 * (1.0 - np.tanh(0.5 * beta * E))
    return fd_val

def _find_peaks(spectrum):
    """Find indices of a spectrum's peaks"""
    peaks, _ = scipy.signal.find_peaks(spectrum)
    return peaks

def _matsubara_freqs(beta, n_max=40):
    """Calculate array of fermionic Matsubara frequencies omega_n = pi(2n+1)/beta"""
    if beta is None:
        raise ValueError(f'_matsubara_freqs | Error: Matsubara freqs only defined at finite temperature')
    
    freqs = np.pi * (2 * np.arange(n_max) + 1) / beta 
    return freqs   