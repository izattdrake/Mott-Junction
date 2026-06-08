import numpy as np
import itertools
 
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility import helpers

# ====================================================================
#   CORE FUNCTIONALITY FOR COUPLED ROTOR PROBLEM ON A HUBBARD DIMER
# --------------------------------------------------------------------
#   A mean-field approximation is made on the exact slave rotor
#   Hubbard Hamiltoniain, giving a spinon and rotor Hamiltonian.
#   In the standard literature for larger crystals, a further
#   approximation is made to decoupled neighboring rotors
#   Otherwise the Hilbert space grows exponentially.
#   
#   A Hubbard dimer is still small enough that we can keep the rotors
#   coupled and solve the problem in reasonable time.
# --------------------------------------------------------------------
#   In this model, rotor states are superpositions of basis states
#   |l1,l2> = |l1>x|l2>. The rotor tunneling element 
#   t<cos(theta1-theta2)> is calculated as is without further 
#   approximation. This code also calculates spinon tunneling, 
#   average <L>, average spinon occupancy and the like.
# --------------------------------------------------------------------
#   Iteration methods are handled in other files since multiple were
#   used and tested.
# ====================================================================

def get_eff_rotor_basis(l_max):
    """
        Returns |l_1 l_2> basis elements
    """
    l_vals = np.arange(-l_max, l_max + 1)
    basis = list(itertools.product(
        l_vals, l_vals
    ))
 
    return np.array(basis)
 
def get_eff_rotor_index(l1, l2, l_max):
    """
    Returns computation indx i \in [0, 2l_max + 1) for (l1,l2) value
    """
    if (l1 > l_max) or (l1 < -l_max) or (l2 > l_max) or (l2 < -l_max):
        raise ValueError
 
    index = (l1 + l_max)*(2*l_max + 1) + (l2 + l_max)
    return index
 
"""
    REST OF FUNCTIONS
"""
def get_H_spinon(epsilon_vec, h_vec, t_eff, mu):
    """
        Return 2x2 Spinon Hamiltonian in a spin subspace
    """
    H = np.diag(epsilon_vec - h_vec - mu) - np.diag(np.array([t_eff]), k=1) - np.diag(np.array([t_eff]), k=-1)
    return H
 
 
def get_H_rotor(U, h_vec, J_eff, l_max):
    """
        Returns rotor Hamiltonian in the |l1 l2> basis as a 2D Numpy matrix
    """
    # Dimension of 1 index, total Hilbert space of dimension dim^2
    dim = 2 * l_max + 1
 
    # Organizes l values in order of how the index mapping is constructed
    l1 = np.repeat(np.arange(-l_max, l_max + 1), dim)
    l2 = np.tile(np.arange(-l_max, l_max + 1), dim)
 
    # Diagonal elements
    H = np.diag(U / 2 * (l1**2 + l2**2) + h_vec[0] * l1 + h_vec[1] * l2)
 
    # Computes where there are valid neighbors
    mask = (l1 < l_max) & (l2 > -l_max)
 
    # Computes computational indices from l indices for bra and ket states
    bra_index = np.where(mask)[0]
    ket_index = (l1[mask] + 1 + l_max) * dim + (l2[mask] - 1 + l_max)
 
    # Fills out off-diagonal elements for bra->ket
    H[bra_index, ket_index] -= J_eff
 
    # Sets transpose
    H[ket_index, bra_index] -= J_eff
 
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
 
def get_J_eff(spinon_evals, spinon_evecs, t, beta):
    """
        Returns J_eff by taking a thermal average of the hopping operator over the spinon Hamiltonian
        Traces are evaluated by going to the eigenstate basis
        Can compare with the approach taken in the MF picture where diagonalized states were averaged with the FD distribution
    """
    J_eff = 2 * t * np.sum(np.conj(spinon_evecs[0,:]) * spinon_evecs[1, :] * helpers.fermi_dirac(spinon_evals, beta))
    return J_eff
 
def get_nf_vec(spinon_evals, spinon_evecs, beta):
    """
        Returns the fermion occupation at both sites
    """
    fermi_vals = helpers.fermi_dirac(spinon_evals, beta)
    nf_vec = 2 * np.abs(spinon_evecs)**2 @ fermi_vals
 
    return nf_vec
 
def get_L(rotor_evals, rotor_evecs, beta, l_max, site):
    """
        Calculates <L_i> where i is the site | Identical in homogeneous dimer
    """
    # Shift energies for Boltzmann factors
    energies = rotor_evals - rotor_evals[0]
 
    basis = get_eff_rotor_basis(l_max)
    l_vals = basis[:, site].astype(float)
    bz = np.exp(-beta * energies)
 
    # Calculates <psi_n|L|\psi_n>
    expectation = l_vals @ np.abs(rotor_evecs)**2
 
    # Partition function and <L>
    Z = np.sum(bz)
    L = np.sum(bz * expectation) / Z
 
    return L
 
def get_L_vec(rotor_evals, rotor_evecs, beta, l_max):
    # Set GS energy to zero
    L_vec = np.zeros(2)
 
    for site in range(2):
        L_vec[site] = get_L(rotor_evals, rotor_evecs, beta, l_max, site)
 
    return L_vec

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