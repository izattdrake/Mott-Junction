import numpy as np
import sys

from typing import Union, Tuple
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility.helpers import Helpers

# ====================================================================
#   SPINON SECTOR
# ====================================================================
def get_H_spinon(epsilon_vec, t_eff, h_vec):
    """Calculate 2x2 Spinon Hamiltonian in a spin-subspace.
    Spin-subspaces are assumed to be equivalent."""
    H = np.array([[epsilon_vec[0] - h_vec[0], -t_eff],
                     [-t_eff, epsilon_vec[1] - h_vec[1]]])
    return H

def get_occupations(evals_spinon, N_target: float, beta: Union[float, None],
                    deg_tol: float=1e-10) -> Tuple[np.ndarray, float]:
    """Calculate orbital occupancy vector and chemical potential to enforce
    particle number. Occupancy is the per-spin occupation of an orbital.
    Length is N_at."""
    if not (0.0 <= N_target <= 4):
        raise ValueError(f"N_target must lie in [0, 4]")
    
    if beta is None:
        N_per_spin = N_target / 2.0
        if abs(evals_spinon[1] - evals_spinon[0]) < deg_tol:
            occ = np.full(2, N_per_spin / 2.0)
        else:
            occ = np.zeros(2)
            occ[0] = min(N_per_spin, 1.0)
            occ[1] = max(N_per_spin - 1.0, 0.0)
        return occ, 0.0
    
    width = 50.0 / beta
    lo, hi = evals_spinon.min() - width, evals_spinon.max() + width
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        N = 2 * (Helpers.fermi(evals_spinon[0], beta, mu) + Helpers.fermi(evals_spinon[1], beta, mu))
        if N > N_target:
            hi = mu
        else:
            lo = mu
    mu = 0.5 * (lo + hi)
    occ = np.array([Helpers.fermi(evals_spinon[0], beta, mu), Helpers.fermi(evals_spinon[1], beta, mu)])
    return occ, mu

def get_nf_vec(evecs_spinon, occ):
    """Calculate spinon occupation at all sites."""
    return 2 * (np.abs(evecs_spinon)**2 @ occ) 

def get_rho(evecs_spinon, occ):
    """Calculate the thermal expectation of the hopping operator rho_{ij} = <f^+_i f_j>."""
    return (evecs_spinon * occ[np.newaxis, :]) @ evecs_spinon.conj().T

def get_t_eff(t, Z_vec):
    """Calculate spinon tunneling"""
    return t * np.sqrt(Z_vec[0] * Z_vec[1])

# ====================================================================
#   ROTOR SECTOR
# ====================================================================
def get_H_rotor(U: float, h: float, K: float, l_max: int):
    """Calculate rotor Hamiltonian at a site"""
    l_vals = np.arange(-l_max, l_max + 1)
    off_diag = np.full(2*l_max, K / 2.0)

    H = np.diag((U / 2.0) * l_vals**2 + h*l_vals)
    H += np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    return H

def get_rotor_eig_vec(U, h_vec, K_vec, l_max: int):
    """Calculate rotor Hamiltonian eigenspace solutions at both sites. 
    Returns arrays of shape (2, dim) and (2, dim, dim) where dim = (2*l_max + 1)."""
    dim = 2*l_max + 1
    evals_rotor_vec = np.zeros((2, dim))
    evecs_rotor_vec = np.zeros((2, dim, dim))

    for i in range(2):
        evals_rotor_vec[i], evecs_rotor_vec[i] = np.linalg.eigh(get_H_rotor(U, h_vec[i], K_vec[i], l_max))

    return (evals_rotor_vec, evecs_rotor_vec)

def _thermal_diag(evals_rotor, evecs_rotor, beta, diag_op):
    """Thermal average of an operator diagonal in the |l> basis."""
    op_weights = np.sum((np.abs(evecs_rotor)**2) * diag_op[:, np.newaxis], axis=0) 
    return np.dot(Helpers.bz_weights(evals_rotor, beta), op_weights)

def _L(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int):
    """Calculate thermal average <L_i> at site | arguemnts are the rotor eigenvalues and eigenvectors at that site"""
    l_vals = np.arange(-l_max, l_max + 1, dtype=float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals)

def _L2(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int):
    l_vals = np.arange(-l_max, l_max+1, dtype=float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals**2)

def get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None], l_max: int):
    """Return thermal expectation value of L with the rotor Hamiltonian at each lattice site"""
    return np.array([_L(evals_rotor_vec[i], evecs_rotor_vec[i], beta, l_max) for i in range(2)])

def get_L2_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None], l_max: int):
    """Return thermal expectation value of L^2 with the rotor Hamiltonian at each lattice site"""
    return np.array([_L2(evals_rotor_vec[i], evecs_rotor_vec[i], beta, l_max) for i in range(2)])

def _Z(evals_rotor, evecs_rotor, beta: Union[float, None]) -> float:
    """Calculate Z at a single site."""
    bz_weights = Helpers.bz_weights(evals_rotor, beta)
    mel = np.sum(evecs_rotor[:-1, :].conj() * evecs_rotor[1:, :], axis=0)
    return np.dot(bz_weights, mel)**2

def get_Z_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None]):
    """Calculate Z at every site."""
    return np.array([_Z(evals_rotor_vec[i], evecs_rotor_vec[i], beta) for i in range(2)])

def get_K_vec(t, Z_vec, rho):
    """Calculate K vector. Definition in Latex notes."""
    r12 = np.real(rho[0, 1])
    r21 = np.real(rho[1,0])
    return np.array([-2.0 * 2 * t * np.sqrt(Z_vec[1]) * r12,
                     -2.0 * 2 * t * np.sqrt(Z_vec[0]) * r21])

# ====================================================================
#   CONSTRAINT METHODS
# ====================================================================
def get_residual(nf_vec, L_vec):
    """Calculate constraint/charge vector f_i = <L_i> + 1 - <n^f_i"""
    return L_vec + 1.0 - nf_vec

# ====================================================================
#   JACOBIAN METHODS
# ====================================================================
def _Jnf(epsilon_vec, t_eff, h_vec, dh: float, beta: Union[float, None], N_target):
    """Calculate Jacobian part from spinon sector"""
    J = np.zeros((2,2))
    for i in range(2):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[i] += dh/2.0
        h_m[i] -= dh/2.0

        evals_p, evecs_p = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff, h_p))
        evals_m, evecs_m = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff, h_m))

        occ_p, _ = get_occupations(evals_p, N_target, beta)
        occ_m, _ = get_occupations(evals_m, N_target, beta)

        J[:,i] = (get_nf_vec(evecs_p, occ_p) - get_nf_vec(evecs_m, occ_m)) / dh

    return J

def _JL(U, h_vec, dh: float, K_vec: np.ndarray, beta: Union[float, None], l_max: int):
    """Calculate Jacobian part from rotor sector"""
    J = np.zeros((2,2))

    for i in range(2):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p = h_vec[i] + dh / 2.0
        h_m = h_vec[i] - dh / 2.0
        
        # May need to calculate K_vec_p, K_vec_m
        evals_p, evecs_p = np.linalg.eigh(get_H_rotor(U, h_p, K_vec[i], l_max))
        evals_m, evecs_m = np.linalg.eigh(get_H_rotor(U, h_m, K_vec[i], l_max))

        J[i,i] = (_L(evals_p, evecs_p, beta, l_max) - _L(evals_m, evecs_m, beta, l_max)) / dh

    return J

def get_Jacobian(epsilon_vec, U, t_eff, h_vec, dh: float, K_vec: np.ndarray, beta: Union[float, None], l_max: int, N_target: int):
    """Calculate Jacobian for Newton's method | J = J_nf - J_L"""
    J_nf = _Jnf(epsilon_vec, t_eff, h_vec, dh, beta, N_target)
    J_L = _JL(U, h_vec, dh, K_vec, beta, l_max)

    # TODO: CHECK SIGN OF JACOBIAN
    J = J_L - J_nf

    return J