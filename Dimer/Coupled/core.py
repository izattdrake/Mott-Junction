import numpy as np
import itertools
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility import helpers

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
#   RANDOM FUNCTIONS
# ====================================================================
def get_eff_rotor_basis(l_max):
    """Returns |l1,l2> basis elements"""
    l_vals = np.arange(-l_max, l_max + 1)
    basis = list(itertools.product(
        l_vals, l_vals
    ))
 
    return np.array(basis)
 
def get_eff_rotor_index(l1, l2, l_max):
    """Returns computational index for (l1,l2) value"""
    if (l1 > l_max) or (l1 < -l_max) or (l2 > l_max) or (l2 < -l_max):
        raise ValueError
    index = (l1 + l_max)*(2*l_max + 1) + (l2 + l_max)
    return index

# ====================================================================
#   SPINON SECTOR
# ====================================================================
def get_H_spinon(epsilon_vec, h_vec, t_eff, mu):
    # Return 2x2 Spinon Hamiltonian in a spin subspace
    H = np.diag(epsilon_vec - h_vec - mu) - np.diag(np.array([t_eff]), k=1) - np.diag(np.array([t_eff]), k=-1)
    return H

def get_t_eff(rotor_evals, rotor_evecs, t, beta, l_max):
    """
        Calculates effective spinon tunneling
    """
    dim = 2 * l_max + 1
    l1 = np.repeat(np.arange(-l_max, l_max + 1), dim)
    l2 = np.tile(np.arange(-l_max, l_max + 1), dim)
 
    # Build the shift operator; maps |l1, l2> -> |l1+1,l2-1> + |l1-1,l2+1>
    S = np.zeros((dim**2, dim**2))
 
    mask = (l1 < l_max) & (l2 > -l_max)
    bra_index = np.where(mask)[0]
    ket_index = (l1[mask] + 1 + l_max) * dim + (l2[mask] - 1 + l_max)

    S[ket_index, bra_index] = 1.0  # |l1,l2> -> |l1+1,l2-1>
    S[bra_index, ket_index] = 1.0  # |l1,l2> -> |l1-1,l2+1>
 
    # Shift energies to avoid overflow
    energies = rotor_evals - rotor_evals[0]
    bz = np.exp(-beta * energies)
 
    # <psi_n|S|psi_n> for each eigenstate n
    expectation_vals = np.real(np.sum(np.conj(rotor_evecs) * (S @ rotor_evecs), axis=0))
    Z = np.sum(bz)
 
    t_eff = t * np.sum(bz * expectation_vals) / (2 * Z)
    return t_eff

def get_nf_vec(spinon_evals, spinon_evecs, beta):
    """Calculates spinon occupation at every site"""
    fd_vals = helpers.fermi_dirac(spinon_evals, beta)
    nf_vec = 2 * np.abs(spinon_evecs)**2 @ fd_vals
    return nf_vec

# ====================================================================
#   ROTOR SECTOR
# ====================================================================
def get_H_rotor(U, h_vec, J_eff, l_max):
    """Calculate rotor Hamiltonian in (l1,l2) basis"""
    dim = 2*l_max + 1
    l1 = np.repeat(np.arange(-l_max, l_max + 1), dim)
    l2 = np.tile(np.arange(-l_max, l_max + 1), dim)

    # Diagonal elements
    H = np.diag(U / 2 * (l1**2 + l2**2) + h_vec[0] * l1 + h_vec[1] * l2)

    mask = (l1 < l_max) & (l2 > -l_max) # Filters valid neighbors for shifting
    bra_index = np.where(mask)[0]
    ket_index = (l1[mask] + l_max + 1)*dim + (l2[mask] + l_max - 1)

    # Fills out off-diagonal elements for bra->ket
    H[bra_index, ket_index] -= J_eff
    H[ket_index, bra_index] -= J_eff

    return H

def get_J_eff(spinon_evals, spinon_evecs, t, beta):
    """Calculates J^{eff}_{12} | <f^+_1 f_2> calculated by diagaonlization then FD distribution"""
    J_eff = 2 * t * np.sum(np.conj(spinon_evecs[0,:]) * spinon_evecs[1, :] * helpers.fermi_dirac(spinon_evals, beta))
    return J_eff

def get_L(rotor_evals, rotor_evecs, beta, l_max, site):
    """Calculates thermal average <L_i> at one site"""
    energies = rotor_evals - rotor_evals[0] # Shift energies by GS to prevent overflow
    bz = np.exp(-beta * energies)

    basis = get_eff_rotor_basis(l_max)
    l_vals = basis[:, site].astype(float)
 
    # <psi_n|L|\psi_n>
    exp_vals = l_vals @ np.abs(rotor_evecs)**2
 
    L = np.sum(bz * exp_vals) / np.sum(bz)
    return L
 
def get_L_vec(rotor_evals, rotor_evecs, beta, l_max):
    """Calculate thermal average <L_i> at every site"""
    L_vec = np.zeros(2)
    for site in range(2):
        L_vec[site] = get_L(rotor_evals, rotor_evecs, beta, l_max, site)
    return L_vec

# ====================================================================
#   ITERATION METHODS
# ====================================================================
def get_Jacobian(U, epsilon_vec, h_vec, dh, t_eff, J_eff, beta, l_max, mu):
    """
        Return Jacobian used in Newton's method
    """
    def get_Jacobian_nf(epsilon_vec, t_eff, h_vec, dh, beta):
        """
            Returns spinon occupancy contribution to Jacobian
        """
        J = np.zeros((2,2))
        for j in range(2):
            h_p, h_m = h_vec.copy(), h_vec.copy()
            h_p[j] += dh / 2.0
            h_m[j] -= dh / 2.0
 
            evals_p, evecs_p = np.linalg.eigh(get_H_spinon(epsilon_vec, h_p, t_eff, mu))
            evals_m, evecs_m = np.linalg.eigh(get_H_spinon(epsilon_vec, h_m, t_eff, mu))
 
            nf_p = get_nf_vec(evals_p, evecs_p, beta)
            nf_m = get_nf_vec(evals_m, evecs_m, beta)
 
            J[:,j] = (nf_p - nf_m) / dh
 
        return J
 
    def get_Jacobian_L(U, h_vec, dh, J_eff, l_max):
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
 
    J_nf = get_Jacobian_nf(epsilon_vec, t_eff, h_vec, dh, beta)
    J_L = get_Jacobian_L(U, h_vec, dh, J_eff, l_max)
 
    J = J_nf - J_L
 
    return J
