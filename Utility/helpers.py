import numpy as np
import warnings
import itertools
import scipy

class Helpers:
    #Constants
    HBAR_SI = 1.054571817 * 1e-34
    HBAR_EV = 6.582119569 * 1e-16
    KB_SI = 1.380649 * 1e-23
    KB_EV = 8.617333262 * 1e-5

    @staticmethod
    def kdelta(x, y):
        """Kronecker delta"""
        return int(x==y)

    @staticmethod
    def ddelta(x, eta=1e-4):
        """Dirac delta function | Calculated with Lorentzian as eta -> 0"""
        return (eta/np.pi)/(x**2 + eta**2)

    @staticmethod
    def bz_weights(energies, beta):
        """Calculates normalized Boltzmann probabilities"""
        if beta is None:                          
            bz = (energies - energies[0] < 1e-10).astype(float) # Lock all weight to the GS
        else:
            bz = np.exp(-beta * (energies - energies[0]))
        return bz / bz.sum()

    @staticmethod
    def fermi(E, beta, mu=0.0):
        """Calculate Fermi-Dirac distribution | T=0 handled by beta=None"""
        if beta is None:
            fd_val = np.where(E < mu, 1.0, np.where(E > mu, 0.0, 0.5)) # E=mu returns 0.5 at zero-temperature
        else:
            fd_val = 0.5 * (1.0 - np.tanh(0.5 * beta * E))
        return fd_val

    @staticmethod
    def find_peaks(spectrum):
        """Find indices of a spectrum's peaks"""
        peaks, _ = scipy.signal.find_peaks(spectrum)
        return peaks

    @staticmethod
    def matsubara_fermion_freqs(beta, n_max=40):
        """Calculate array of fermionic Matsubara frequencies omega_n = pi(2n+1)/beta"""
        if beta is None:
            raise ValueError(f'_matsubara_freqs | Error: Matsubara freqs only defined at finite temperature')

        return  np.pi*(2 * np.arange(n_max) + 1)/beta    
    
    @staticmethod
    def matsubara_boson_freqs(beta, n_max=40):
        """Calculate array of fermionic Matsubara frequencies omega_n = pi(2n+1)/beta"""
        if beta is None:
            raise ValueError(f'_matsubara_freqs | Error: Matsubara freqs only defined at finite temperature')

        return  2*np.pi*np.arange(n_max)/beta   