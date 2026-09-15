import numpy as np
import itertools
import sys

from typing import Union, Tuple
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility.helpers import Helpers

# ====================================================================
#   CORE FUNCTIONALITY FOR COUPLED ROTOR PROBLEM ON A HUBBARD DIMER
# --------------------------------------------------------------------
#   Core functionality used to solve the self-consistency conditions
#   in the coupled rotor problem on a Hubbard dimer. Functions here
#   are imported into Dimer/Coupled/broyden.py, 
#   Dimer/Coupled/newton_gauge.py, and Dimer/Coupled/newton_naive.py
#   to solve for the eigenspace solutions and parameters. The
#   Dimer/Coupled/Coupled_Dimer.py file contains the class where 
#   additional functionality for calculating observables lie.
# ====================================================================

# ====================================================================
#   COMPUTATIONAL BASIS
# ====================================================================
def get_eff_rotor_basis(l_max: int):
    """Returns |l1,l2> basis elements"""
    l_vals = np.arange(-l_max, l_max + 1)
    basis = list(itertools.product(
        l_vals, l_vals
    ))
 
    return np.array(basis)
 
def get_eff_rotor_index(l1: int, l2: int, l_max: int):
    """Returns computational index for (l1,l2) value"""
    if (l1 > l_max) or (l1 < -l_max) or (l2 > l_max) or (l2 < -l_max):
        raise ValueError
    index = (l1 + l_max)*(2*l_max + 1) + (l2 + l_max)
    return index

# ====================================================================
#   SPINON SECTOR
# ====================================================================
def get_H_spinon(epsilon_vec, h_vec, t_eff):
    """Return spinon Hamiltonian in a spin subspace"""
    H = np.array([[epsilon_vec[0] - h_vec[0], -t_eff],
                     [-t_eff, epsilon_vec[1] - h_vec[1]]])
    return H

def get_occupations(evals_spinon, N_target: float, beta: Union[float, None],
                    deg_tol: float=1e-10) -> Tuple[np.ndarray, float]:
    """Calculate orbital occupancy vector and chemical potential to enforce
    particle number."""
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

def get_t_eff(evals_rotor, evecs_rotor, t: float, beta: Union[float, None], l_max: int):
    """Calculates effective spinon tunneling"""
    dim = 2 * l_max + 1
    l1 = np.repeat(np.arange(-l_max, l_max + 1), dim)
    l2 = np.tile(np.arange(-l_max, l_max + 1), dim)
 
    # Build the shift operator; maps |l1, l2> -> |l1+1,l2-1> + |l1-1,l2+1>
    shift_op = np.zeros((dim**2, dim**2))
 
    mask = (l1 < l_max) & (l2 > -l_max)
    ket_index = np.where(mask)[0]
    bra_index = (l1[mask] + 1 + l_max) * dim + (l2[mask] - 1 + l_max)

    shift_op[bra_index, ket_index] = 1.0  # |l1,l2> -> |l1+1,l2-1>
    shift_op[ket_index, bra_index] = 1.0  # |l1,l2> -> |l1-1,l2+1>
 
    bz_weights = Helpers.bz_weights(evals_rotor, beta)
 
    # <psi_n|S|psi_n> for each eigenstate n
    expectation_vals = np.real(np.sum(np.conj(evecs_rotor) * (shift_op @ evecs_rotor), axis=0))
    t_eff = t * np.sum(bz_weights * expectation_vals) / 2
    return t_eff

def get_nf_vec(evecs_spinon, occ):
    """Calculates spinon occupation at every site"""
    nf_vec = 2 * (np.abs(evecs_spinon)**2 @ occ)
    return nf_vec

def get_rho(evecs_spinon, occ):
    """Calculate the thermal hopping matrix rho_{ij} = <f^+_i f_j>."""
    return (evecs_spinon * occ[np.newaxis, :]) @ evecs_spinon.conj().T

# ====================================================================
#   ROTOR SECTOR
# ====================================================================
def get_H_rotor(U: float, h_vec, J_eff: float, l_max: int):
    """Calculate rotor Hamiltonian in (l1,l2) basis"""
    dim = 2 * l_max + 1
    l1 = np.repeat(np.arange(-l_max, l_max + 1), dim)
    l2 = np.tile(np.arange(-l_max, l_max + 1), dim)

    # Diagonal elements
    H = np.diag(U / 2 * (l1**2 + l2**2) + h_vec[0] * l1 + h_vec[1] * l2)

    mask = (l1 < l_max) & (l2 > -l_max) # Filters valid neighbors for shifting
    ket_index = np.where(mask)[0]
    bra_index = (l1[mask] + l_max + 1) * dim + (l2[mask] + l_max - 1)

    # Fills out off-diagonal elements for bra->ket
    H[bra_index, ket_index] -= J_eff
    H[ket_index, bra_index] -= J_eff

    return H

def get_J_eff(t, rho):
    """Calculates J^{eff}_{12} | <f^+_1 f_2> calculated by diagaonlization then FD distribution"""
    return 2.0 * t * np.real(rho[0, 1])

def _thermal_diag(evals_rotor, evecs_rotor, beta: Union[float, None], diag_op):
    """Thermal average of an operator that is diagonal in the |l1,l2> basis.
    Unlike the decoupled case there is one rotor Hamiltonian for the whole dimer."""
    probs = np.abs(evecs_rotor)**2            
    exp_vals = diag_op @ probs              
    return np.sum(Helpers.bz_weights(evals_rotor, beta) * exp_vals)

def get_L(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int, site: int):
    """Thermal average <L_i> at one site."""
    l_vals = get_eff_rotor_basis(l_max)[:, site].astype(float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals)

def get_L_vec(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int):
    return np.array([get_L(evals_rotor, evecs_rotor, beta, l_max, site) for site in range(2)])

def get_L2(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int, site: int):
    """Thermal average <L_i^2> at one site."""
    l_vals = get_eff_rotor_basis(l_max)[:, site].astype(float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals**2)

def get_L2_vec(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int):
    """Thermal average <L_i^2> at every site."""
    return np.array([get_L2(evals_rotor, evecs_rotor, beta, l_max, site) for site in range(2)])

def get_L1L2(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int):
    """Thermal average <L_1 L_2>. Zero by construction for decoupled rotors;
    nonzero here, and the quantity that distinguishes the cluster solution."""
    basis = get_eff_rotor_basis(l_max).astype(float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, basis[:, 0] * basis[:, 1])

# ====================================================================
#   CONSTRAINT METHODS
# ====================================================================
def get_residual(nf_vec, L_vec):
    """Calculate constraint/charge vector f_i = <L_i> + 1 - <n_f,i"""
    return L_vec + 1.0 - nf_vec

# ====================================================================
#   ITERATION METHODS
# ====================================================================
def _Jacobian_nf(epsilon_vec, t_eff: float, h_vec, dh: float, beta: float, N_target: int):
    """
        Returns spinon occupancy contribution to Jacobian
    """
    J = np.zeros((2,2))
    for j in range(2):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[j] += dh / 2.0
        h_m[j] -= dh / 2.0

        evals_p, evecs_p = np.linalg.eigh(get_H_spinon(epsilon_vec, h_p, t_eff))
        evals_m, evecs_m = np.linalg.eigh(get_H_spinon(epsilon_vec, h_m, t_eff))

        occ_p, _ = get_occupations(evals_p, N_target, beta)
        occ_m, _ = get_occupations(evals_m, N_target, beta)

        J[:,j] = (get_nf_vec(evecs_p, occ_p) - get_nf_vec(evecs_m, occ_m)) / dh

    return J

def _Jacobian_L(U: float, h_vec, dh: float, J_eff: float, l_max: int, beta: Union[float, None]):
    J = np.zeros((2, 2))
    for j in range(2):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[j] += dh / 2.0
        h_m[j] -= dh / 2.0

        evals_p, evecs_p = np.linalg.eigh(get_H_rotor(U, h_p, J_eff, l_max))
        evals_m, evecs_m = np.linalg.eigh(get_H_rotor(U, h_m, J_eff, l_max))

        for i in range(2):
            L_p = get_L(evals_p, evecs_p, beta, l_max, i)
            L_m = get_L(evals_m, evecs_m, beta, l_max, i)

            J[i, j] = (L_p - L_m) / dh

    return J

def get_Jacobian(U: float, epsilon_vec, h_vec, dh: float, t_eff: float, J_eff: float, 
                 beta: Union[float, None], l_max: int, N_target):
    """Calculate the Jacobian used in Newton's method"""
    return _Jacobian_L(U, h_vec, dh, J_eff, l_max, beta) - _Jacobian_nf(epsilon_vec, t_eff, h_vec, dh, beta, N_target)