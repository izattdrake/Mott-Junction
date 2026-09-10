import numpy as np
import sys

from typing import Sequence, Union
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility.helpers import Helpers

# ====================================================================
#   SPINON SECTOR
# ====================================================================
def get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu):
    """Calculate 2x2 Spinon Hamiltonian in a spin-subspace"""
    H = np.diag(epsilon_vec - h_vec - mu) - t_eff_mat
    return H

def get_nf_vec(evals_spinon, evecs_spinon, beta: Union[float, None], N_at):
    """Calculate spinon occupation at all sites."""
    fd_vals = [Helpers.fermi(energy, beta) for energy in evals_spinon]       
    nf = N_at * (np.abs(evecs_spinon) ** 2 @ fd_vals)     
    return nf

def get_t_eff(t_mat, Z_vec):
    """Calculate spinon tunneling"""
    sqrt_Z = np.sqrt(Z_vec)
    t_eff_mat = t_mat * np.outer(sqrt_Z, sqrt_Z)
    return t_eff_mat

# ====================================================================
#   ROTOR SECTOR
# ====================================================================
def get_H_rotor(U, h_vec, K_vec, l_max: int, site: int):
    """Calculate rotor Hamiltonian at a site"""
    dim = 2*l_max + 1
    l_vals = np.arange(-l_max, l_max + 1)

    H_diag = ((U/2.0)*l_vals + h_vec[site]) * l_vals 
    H_off_diag = K_vec[site] / 2.0
    
    H = np.diag(H_diag)
    H += np.diag(np.full(dim - 1, H_off_diag), k=1) + np.diag(np.full(dim - 1, H_off_diag), k=-1)

    return H

def get_H_rotor_vec(U, h_vec, K_vec, l_max: int):
    """Calculate rotor Hamiltonian at all sites"""
    H_vec = np.array([get_H_rotor(U, h_vec, K_vec, l_max, site) for site in range(2)])
    return H_vec

def get_rotor_eig_vec(U, h_vec, K_vec, l_max: int):
    """Calculate eigenvalues and eigenvectors of the rotor Hamiltonian at every site"""
    dim = 2*l_max + 1
    print(l_max)
    evals_rotor_vec = np.zeros((2, dim))
    evecs_rotor_vec = np.zeros((2, dim, dim))

    for i in range(2):
        evals_rotor_vec[i], evecs_rotor_vec[i] = np.linalg.eigh(get_H_rotor(U, h_vec, K_vec, l_max, i))

    return evals_rotor_vec, evecs_rotor_vec

def get_L(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int):
    """Calculate thermal average <L_i> at site | arguemnts are the rotor eigenvalues and eigenvectors at that site"""
    l_vals = np.arange(-l_max, l_max + 1)
    l_expect_per_state = np.sum((np.abs(evecs_rotor)**2) * l_vals[:,np.newaxis], axis=0)

    bz = Helpers.bz_weights(evals_rotor, beta)
    L = np.dot(bz, l_expect_per_state)
    return L

def get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None], l_max: int):
    """Return thermal expectation value of L with the rotor Hamiltonian at each lattice site"""
    L_vec = np.zeros(2)
    
    for i in range(2):
        L_vec[i] = get_L(evals_rotor_vec[i], evecs_rotor_vec[i], beta, l_max)
   
    return L_vec

def get_Z_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None]):
    """Calculate Z at every site"""
    Z_vec = np.zeros(2)
    for i in range(2):
        evals_rotor, evecs_rotor = evals_rotor_vec[i], evecs_rotor_vec[i]
        bz_weights = Helpers.bz_weights(evals_rotor, beta)
        mel = (evecs_rotor[:-1, :].conj() * evecs_rotor[1:, :]).sum(axis=0)

        Z_vec[i] = np.dot(bz_weights, mel)**2

    return Z_vec

def get_K_vec(t_mat, Z_vec, evals_spinon, evecs_spinon, beta: Union[float, None], N_at: int):
    """Calculate K values at every site"""
    fd = np.array([Helpers.fermi(energy, beta) for energy in evals_spinon])
    evecs_weighted = evecs_spinon * fd[np.newaxis,:]

    sqrt_Z = np.sqrt(Z_vec)
    rho = evecs_spinon.conj() @ evecs_weighted.T

    K_vec = np.real(-2 * N_at * np.sum(t_mat * sqrt_Z[np.newaxis,:] * rho, axis=1))
    return K_vec

# ====================================================================
#   ITERATION METHODS
# ====================================================================
def get_Jacobian_nf(epsilon_vec, t_eff_mat, h_vec, dh: float, beta: Union[float, None], mu: float):
    """Calculate Jacobian part from spinon sector"""
    J = np.zeros((2,2))
    for site in range(2):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[site] += dh/2.0
        h_m[site] -= dh/2.0

        evals_spinon_p, evecs_spinon_p = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_p, mu))
        evals_spinon_m, evecs_spinon_m = np.linalg.eigh(get_H_spinon(epsilon_vec, t_eff_mat, h_m, mu))

        nf_p = get_nf_vec(evals_spinon_p, evecs_spinon_p, beta)
        nf_m = get_nf_vec(evals_spinon_m, evecs_spinon_m, beta)

        J[:,site] = (nf_p - nf_m) / dh

    return J

def get_Jacobian_L(U, h_vec, dh: float, K_vec: np.ndarray, beta: Union[float, None], l_max: int):
    """Calculate Jacobian part from rotor sector"""
    J = np.zeros((2,2))

    for site in range(2):
        h_p, h_m = h_vec.copy(), h_vec.copy()
        h_p[site] += dh/2
        h_m[site] -= dh/2
        
        # May need to calculate K_vec_p, K_vec_m
        evals_rotor_p, evecs_rotor_p = np.linalg.eigh(get_H_rotor(U, h_p, K_vec, l_max, site))
        evals_rotor_m, evecs_rotor_m = np.linalg.eigh(get_H_rotor(U, h_m, K_vec, l_max, site))
        
        L_p = get_L(evals_rotor_p, evecs_rotor_p, beta, l_max)
        L_m = get_L(evals_rotor_m, evecs_rotor_m, beta, l_max)

        J[site,site] = (L_p - L_m) / dh

    return J

def get_Jacobian(epsilon_vec, U, t_eff_mat, h_vec, dh: float, K_vec: np.ndarray, beta: Union[float, None], l_max: int, mu: float):
    """Calculate Jacobian for Newton's method | J = J_nf - J_L"""
    J_nf = get_Jacobian_nf(epsilon_vec, t_eff_mat, h_vec, dh, beta, mu)
    J_L = get_Jacobian_L(U, h_vec, dh, K_vec, beta, l_max)

    J = J_nf - J_L

    return J