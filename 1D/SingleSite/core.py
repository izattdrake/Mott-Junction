import numpy as np
import sys

from typing import Union
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility.helpers import Helpers

# ====================================================================
#   SINGLE-SITE APPROXIMATION CORE FUNCTIONALITY
# --------------------------------------------------------------------
#   These are the core functions for calculating Hamiltonians and 
#   effective parameters needed to converge the slave-rotor rep.
#   on a set of self-consistent parameters. This is "backend" 
#   functionality that will not be seen otherwise. The observables
#   that are actually used in research are all inside the 
#   SingleSite1D class (i.e. Green's functions, sprectral functions,
#   energies, heat capacity.)
# ====================================================================


# ====================================================================
#   SPINON SECTOR
# ====================================================================
def get_H_spinon(epsilon_vec: np.ndarray, t_eff_mat: np.ndarray, h_vec: np.ndarray, mu: float) -> np.ndarray:
    """Calculate Spinon Hamiltonian in a spin-subspace.
    Spin-subspaces are assumed to be equivalent."""
    H = np.diag(epsilon_vec - h_vec - mu) - t_eff_mat
    return H

def get_nf_vec(evals_spinon: np.ndarray, evecs_spinon: np.ndarray, beta: Union[float, None], N_at: int) -> np.ndarray:
    """Calculate spinon occupation at all sites."""
    fd_vals = [Helpers.fermi(energy, beta) for energy in evals_spinon]      
    nf = N_at * (np.abs(evecs_spinon) ** 2 @ fd_vals)     
    return nf

def get_t_eff(t_mat: np.ndarray, Z_vec: np.ndarray) -> np.ndarray:
    """Calculate effective spinon tunneling."""
    sqrt_Z = np.sqrt(Z_vec)
    t_eff_mat = t_mat * np.outer(sqrt_Z, sqrt_Z)
    return t_eff_mat

# ====================================================================
#   ROTOR SECTOR
# ====================================================================
def get_H_rotor(U: float, h_vec: np.ndarray, K_vec: np.ndarray, l_max: int, site: int) -> np.ndarray:
    """Calculate rotor Hamiltonian at a site."""
    dim = 2 * l_max + 1
    l_vals = np.arange(-l_max, l_max+1)

    H_diag = ((U/2.0)*l_vals + h_vec[site]) * l_vals 
    H_off_diag = K_vec[site] / 2.0
    
    H = np.diag(H_diag)
    H += np.diag(np.full(dim-1, H_off_diag), k=1) + np.diag(np.full(dim-1, H_off_diag), k=-1)

    return H

def get_H_rotor_vec(U: float, h_vec: np.ndarray, K_vec: np.ndarray, l_max: int, N_sites: int) -> np.ndarray:
    """Calculate rotor Hamiltonian at all sites."""
    H_vec = np.array([get_H_rotor(U, h_vec, K_vec, l_max, site) for site in range(N_sites)])
    return H_vec

def get_rotor_eig_vec(U: float, h_vec: np.ndarray, K_vec: np.ndarray, l_max: int, N_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate eigenvalues and eigenvectors of the rotor Hamiltonian at every site"""
    dim = 2*l_max + 1

    evals_rotor_vec = np.zeros((N_sites, dim))
    evecs_rotor_vec = np.zeros((N_sites, dim, dim))

    for site in range(N_sites):
        evals_rotor_vec[site], evecs_rotor_vec[site] = np.linalg.eigh(get_H_rotor(U, h_vec, K_vec, l_max, site))

    return (evals_rotor_vec, evecs_rotor_vec)

def get_L(evals_rotor: np.ndarray, evecs_rotor: np.ndarray, l_max: int, beta: Union[float, None]) -> float:
    """Calculate thermal average <L_i> at site | arguemnts are the rotor eigenvalues and eigenvectors at that site"""
    l_vals = np.arange(-l_max, l_max + 1)
    l_expect_per_state = np.sum((np.abs(evecs_rotor)**2) * l_vals[:,np.newaxis], axis=0)

    bz = Helpers.bz_weights(evals_rotor, beta)
    L = np.dot(bz, l_expect_per_state)
    return L

def get_L_vec(evals_rotor_vec: np.ndarray, evecs_rotor_vec: np.ndarray, l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Return thermal expectation value of L at every lattice site."""
    L_vec = np.array([get_L(evals_rotor_vec[site], evecs_rotor_vec[site], l_max, beta) for site in range(N_sites)])
    return L_vec

def get_Z(evals_rotor: np.ndarray, evecs_rotor: np.ndarray, beta: Union[float, None]) -> float:
    bz_weights = Helpers.bz_weights(evals_rotor, beta)
    mel = (evecs_rotor[:-1, :].conj() * evecs_rotor[1:, :]).sum(axis=0)
    Z = np.dot(bz_weights, mel)**2
    return Z

def get_Z_vec(evals_rotor_vec: np.ndarray, evecs_rotor_vec: np.ndarray, beta: Union[float, None], N_sites: int):
    """Calculate Z at every site."""
    Z_vec = np.array([get_Z(evals_rotor_vec[site], evecs_rotor_vec[site], beta) for site in range(N_sites)])
    return Z_vec

def get_K_vec(t_mat: np.ndarray, Z_vec: np.ndarray, evals_spinon: np.ndarray, evecs_spinon: np.ndarray, beta: Union[float, None], N_at: int) -> np.ndarray:
    """Calculate K values at every site"""
    fd_vals = np.array([Helpers.fermi(energy, beta) for energy in evals_spinon])
    evecs_weighted = evecs_spinon * fd_vals[np.newaxis,:]

    sqrt_Z = np.sqrt(Z_vec)
    rho = evecs_spinon.conj() @ evecs_weighted.T

    K_vec = np.real(-2 * N_at * np.sum(t_mat * sqrt_Z[np.newaxis,:] * rho, axis=1))
    return K_vec

# ====================================================================
#   ITERATION METHODS
# ====================================================================
def get_Jacobian_nf(epsilon_vec: np.ndarray, t_eff_mat: np.ndarray, h_vec: np.ndarray, dh: float, 
                    mu: float, beta: Union[float, None], N_sites: int, N_at: int) -> np.ndarray:
    """Calculate Jacobian part from spinon sector"""
    J = np.zeros((N_sites, N_sites))
    for site in range(N_sites):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[site] += dh / 2.0
        h_m[site] -= dh / 2.0

        evals_spinon_p, evecs_spinon_p = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_p, mu))
        evals_spinon_m, evecs_spinon_m = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_m, mu))

        nf_p = get_nf_vec(evals_spinon_p, evecs_spinon_p, beta, N_at)
        nf_m = get_nf_vec(evals_spinon_m, evecs_spinon_m, beta, N_at)

        J[:,site] = (nf_p - nf_m) / dh

    return J

def get_Jacobian_L(U: float, h_vec: np.ndarray, dh: float, K_vec: np.ndarray, 
                   l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Calculate Jacobian part from rotor sector"""
    J = np.zeros((N_sites, N_sites))

    for site in range(N_sites):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[site] += dh/2
        h_m[site] -= dh/2
        
        # May need to calculate K_vec_p, K_vec_m
        evals_rotor_p, evecs_rotor_p = np.linalg.eigh(get_H_rotor(U, h_p, K_vec, l_max, site))
        evals_rotor_m, evecs_rotor_m = np.linalg.eigh(get_H_rotor(U, h_m, K_vec, l_max, site))
        
        L_p = get_L(evals_rotor_p, evecs_rotor_p, l_max, beta)
        L_m = get_L(evals_rotor_m, evecs_rotor_m, l_max, beta)

        J[site, site] = (L_p - L_m) / dh

    return J

def get_Jacobian(epsilon_vec: np.ndarray, U: float, t_eff_mat: np.ndarray, h_vec: np.ndarray, dh: float, mu: float, 
                 K_vec: np.ndarray, l_max: int, beta: Union[float, None], N_sites: int, N_at: int) -> np.ndarray:
    """Calculate Jacobian for Newton's method | J = J_nf - J_L"""
    J_nf = get_Jacobian_nf(epsilon_vec, t_eff_mat, h_vec, dh, mu, beta, N_sites, N_at)
    J_L = get_Jacobian_L(U, h_vec, dh, K_vec, l_max, beta, N_sites)

    J = J_nf - J_L

    return J