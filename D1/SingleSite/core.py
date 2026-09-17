import numpy as np
import sys

from typing import Union, Tuple
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
def get_H_spinon(epsilon_vec, t_eff_mat, h_vec) -> np.ndarray:
    """Calculate Spinon Hamiltonian in a spin-subspace.
    Spin-subspaces are assumed to be equivalent."""
    H = np.diag(epsilon_vec - h_vec) - t_eff_mat
    return H

def get_occ_vec(evals_spinon, beta: Union[float, None], N_sites: int,
                    N_at: int, N_target: int, deg_tol: float=1e-10) -> Tuple[np.ndarray, float]:
    """Calculate orbital occupancy vector and chemical potential to enforce
    particle number. Occupancy is the per-spin occupation of an orbital.
    Length is N_at, not N_sites!."""
    N_max = N_at * N_sites
    if not (0.0 <= N_target <= N_max):
        raise ValueError(f"N_target must lie in [0, {N_max}]")

    if beta is None:
        occ_vec = np.zeros(N_sites)
        remaining = N_target / N_at
        i = 0
        while remaining > 1e-14 and i < N_sites:
            j = i   

            while (j + 1 < N_sites and abs(evals_spinon[j+1] - evals_spinon[i]) < deg_tol):
                j += 1
            g = j - i + 1

            fill = min(remaining, float(g))
            occ_vec[i:j + 1] = fill / g
            remaining -= fill
            i = j + 1

        return occ_vec, 0.0

    width = 50.0 / beta
    lo, hi = evals_spinon.min() - width, evals_spinon.max() + width
    
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        N = N_at * np.sum(Helpers.fermi(evals_spinon, beta, mu))
        if N > N_target:
            hi = mu
        else:
            lo = mu

    mu = 0.5 * (lo + hi)
    occ_vec = np.asarray(Helpers.fermi(evals_spinon, beta, mu), dtype=float)
    
    return occ_vec, mu

def get_nf_vec(evecs_spinon, occ_vec, N_at: int):
    """Calculate spinon occupation at all sites."""
    return N_at * (np.abs(evecs_spinon)**2 @ occ_vec) 

def get_rho_mat(evecs_spinon, occ_vec):
    """Calculate the thermal hopping matrix rho_{ij} = <f^+_i f_j>."""
    return (evecs_spinon * occ_vec[np.newaxis, :]) @ evecs_spinon.conj().T

def get_t_eff_mat(t_mat, Z_vec):
    """Calculate effective spinon tunneling matrix."""
    sqrt_Z = np.sqrt(Z_vec)
    return t_mat * np.outer(sqrt_Z, sqrt_Z)

# ====================================================================
#   ROTOR SECTOR
# ====================================================================
def get_H_rotor(U: float, h: float, K: float, l_max: int) -> np.ndarray:
    """Calculate rotor Hamiltonian at a site."""
    l_vals = np.arange(-l_max, l_max+1, dtype=float)
    off_diag = np.full(2*l_max, K / 2.0)
    
    H = np.diag((U/2.0) * l_vals**2 + h * l_vals)
    H += np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    return H

def get_rotor_eig_vec(U: float, h_vec, K_vec, l_max: int, N_sites: int):
    """Calculate eigenvalues and eigenvectors of the rotor Hamiltonian at every site.
    Returns arrays of shape (N_sites, dim) and (N_sites, dim, dim) where dim=(2l_max + 1)."""
    dim = 2*l_max + 1

    evals_rotor_vec = np.zeros((N_sites, dim))
    evecs_rotor_vec = np.zeros((N_sites, dim, dim))

    for i in range(N_sites):
        evals_rotor_vec[i], evecs_rotor_vec[i] = np.linalg.eigh(get_H_rotor(U, h_vec[i], K_vec[i], l_max))

    return (evals_rotor_vec, evecs_rotor_vec)

def _thermal_diag(evals_rotor, evecs_rotor, beta, diag_op):
    """Thermal average of an operator diagonal in the |l> basis."""
    op_weights = np.sum((np.abs(evecs_rotor)**2) * diag_op[:, np.newaxis], axis=0)
    return np.dot(Helpers.bz_weights(evals_rotor, beta), op_weights)

def _L(evals_rotor, evecs_rotor, l_max: int, beta: Union[float, None]) -> float:
    """Calculate thermal average <L_i> at site | arguemnts are the rotor eigenvalues and eigenvectors at that site"""
    l_vals = np.arange(-l_max, l_max+1, dtype=float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals)

def _L2(evals_rotor, evecs_rotor, l_max: int, beta: Union[float, None]) -> float:
    """Calculate thermal average <L_i> at site | arguemnts are the rotor eigenvalues and eigenvectors at that site"""
    l_vals = np.arange(-l_max, l_max+1, dtype=float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals**2)

def get_L_vec(evals_rotor_vec, evecs_rotor_vec, l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Calculate the thermal expectation value of L at every lattice site."""
    return np.array([_L(evals_rotor_vec[i], evecs_rotor_vec[i], l_max, beta) for i in range(N_sites)])

def get_L2_vec(evals_rotor_vec, evecs_rotor_vec, l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Return thermal expectation value of L^2 at every lattice site."""
    return np.array([_L2(evals_rotor_vec[i], evecs_rotor_vec[i], l_max, beta) for i in range(N_sites)])

def _Z(evals_rotor, evecs_rotor, beta: Union[float, None]) -> float:
    """Calculate Z at a single site."""
    bz_weights = Helpers.bz_weights(evals_rotor, beta)
    mel = np.sum(evecs_rotor[:-1, :].conj() * evecs_rotor[1:, :], axis=0)
    return np.dot(bz_weights, mel)**2

def get_Z_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None], N_sites: int):
    """Calculate Z at every site."""
    return np.array([_Z(evals_rotor_vec[i], evecs_rotor_vec[i], beta) for i in range(N_sites)])

def get_K_vec(t_mat, Z_vec, rho_mat, N_at: int) -> np.ndarray:
    """Calculate K values at every site"""
    return -2.0 * N_at * (t_mat * rho_mat) @ np.sqrt(Z_vec)

# ====================================================================
#   CONSTRAINT METHODS
# ====================================================================
def get_residual(nf_vec, L_vec, N_at: int):
    """Calculate constraint/charge vector f_i = <L_i> + 1 - <n^f_i>"""
    return L_vec - nf_vec + N_at / 2.0

# ====================================================================
#   ITERATION METHODS
# ====================================================================
def _Jnf(epsilon_vec, t_eff_mat, h_vec, dh: float, beta: Union[float, None], 
                    N_sites: int, N_at: int, N_target: int):
    """Calculate Jacobian part from spinon sector"""
    J = np.zeros((N_sites, N_sites))

    for i in range(N_sites):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[i] += dh / 2.0
        h_m[i] -= dh / 2.0

        evals_p, evecs_p = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_p))
        evals_m, evecs_m = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_m))

        occ_p, _ = get_occ_vec(evals_p, beta, N_sites, N_at, N_target)
        occ_m, _ = get_occ_vec(evals_m, beta, N_sites, N_at, N_target)

        J[:,i] = (get_nf_vec(evecs_p, occ_p) - get_nf_vec(evecs_m, occ_m)) / dh

    return J

def _JL(U: float, h_vec: np.ndarray, dh: float, K_vec: np.ndarray, 
                   l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Calculate Jacobian part from rotor sector"""
    J = np.zeros((N_sites, N_sites))

    for i in range(N_sites):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p = h_vec[i] + dh / 2.0
        h_m = h_vec[i] - dh / 2.0
        
        # May need to calculate K_vec_p, K_vec_m
        evals_p, evecs_p = np.linalg.eigh(get_H_rotor(U, h_p, K_vec[i], l_max))
        evals_m, evecs_m = np.linalg.eigh(get_H_rotor(U, h_m, K_vec[i], l_max))
        
        J[i,i] = (_L(evals_p, evecs_p, l_max, beta) - _L(evals_m, evecs_m, l_max, beta)) / dh

    return J

def get_Jacobian(epsilon_vec: np.ndarray, U: float, t_eff_mat: np.ndarray, h_vec: np.ndarray, dh: float, 
                 K_vec: np.ndarray, l_max: int, beta: Union[float, None], N_sites: int, N_at: int, N_target: int):
    """Calculate Jacobian for Newton's method | J = J_nf - J_L"""
    J_nf = _Jnf(epsilon_vec, t_eff_mat, h_vec, dh, beta, N_sites, N_at, N_target)
    J_L = _JL(U, h_vec, dh, K_vec, l_max, beta, N_sites)

    J = J_L - J_nf

    return J