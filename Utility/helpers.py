import numpy as np
import scipy
from typing import Sequence

class Helpers:
    """Physical ocnstants in SI and eV."""
    HBAR_SI = 1.054571817 * 1e-34
    HBAR_EV = 6.582119569 * 1e-16
    KB_SI = 1.380649 * 1e-23
    KB_EV = 8.617333262 * 1e-5

    @staticmethod
    def kdelta(x: float, y: float) -> int:
        """Kronecker delta"""
        return int(x==y)

    @staticmethod
    def ddelta(x: float, eta: float=1e-4) -> float:
        """Dirac delta function | Calculated with Lorentzian as eta -> 0"""
        return (eta / np.pi) / (x**2 + eta**2)

    @staticmethod
    def bz_weights(energies: float, beta: float | None) -> np.ndarray:
        """Calculates normalized Boltzmann probabilities."""
        if beta is None:                          
            bz = (energies - energies[0] < 1e-10).astype(float) # Lock all weight to the GS
        else:
            bz = np.exp(-beta * (energies - energies[0]))
        return bz / bz.sum()

    @staticmethod
    def fermi(energy: float, beta: float | None, mu: float=0.0) -> float:
        """Calculate Fermi-Dirac distribution | T=0 handled by beta=None"""
        if beta is None:
            fd_val = np.where(energy < mu, 1.0, np.where(energy > mu, 0.0, 0.5)) # E=mu returns 0.5 at zero-temperature
        else:
            fd_val = 0.5 * (1.0 - np.tanh(0.5 * beta * energy))
        return fd_val

    @staticmethod
    def find_peaks(spectrum: np.ndarray) -> np.ndarray:
        """Find indices of a spectrum's peaks"""
        peaks, _ = scipy.signal.find_peaks(spectrum)
        return peaks

    @staticmethod
    def matsubara_freqs_fermion(beta: float | None | None, n_max: int=40) -> np.ndarray:
        """Calculate array of fermionic Matsubara frequencies omega_n = pi(2n+1)/beta"""
        if beta is None:
            raise ValueError(f'_matsubara_freqs | Error: Matsubara freqs only defined at finite temperature')

        return  np.pi * (2*np.arange(n_max) + 1) / beta    
    
    @staticmethod
    def matsubara_freqs_boson(beta: float | None | None, n_max: int=40) -> np.ndarray:
        """Calculate array of fermionic Matsubara frequencies omega_n = pi(2n+1)/beta"""
        if beta is None:
            raise ValueError(f'_matsubara_freqs | Error: Matsubara freqs only defined at finite temperature')

        return  2 * np.pi * np.arange(n_max) / beta   
    
    @staticmethod
    def hopping_matrix(t: float | Sequence[float], N_sites: int, PBC: bool=False) -> np.ndarray:
        """Constructs a hopping matrix given a sequence of tunneling parameters and size of system.
        Only considers real hopping parameters at the moment."""
        t = np.atleast_1d(t)
        dist = len(t)

        if dist >= N_sites - 1: 
            raise ValueError

        t_mat = np.zeros((N_sites, N_sites))
        
        for i in range(len(t)):
            shift = i + 1
            t_mat += np.diag(np.full(N_sites-shift, t[i]), k=shift) 
            
            if PBC:
                t_mat += np.diag(np.full(shift, t[i]), k=N_sites-shift)

        t_mat += t_mat.conj().T
        return t_mat