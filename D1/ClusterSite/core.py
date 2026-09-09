import numpy as np
import sys
import itertools

from typing import Union
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility.helpers import Helpers

# ====================================================================
#   HELPER FUNCTIONS
# ====================================================================
def cluster_to_sites(m: int) -> tuple[int, int]:
    """Sites belonging to 2-site cluster m."""
    return (2 * m, 2 * m + 1)

def cluster_id(N_sites: int) -> np.ndarray:
    """Calculate array of elements whos value indexes the 
    cluster it belongs to."""
    return np.arange(N_sites) // 2

def cluster_mask(N_sites: int) -> np.ndarray:
    """Calculate the matrix cluster_id[i,j] = True if i & j in same cluster.
    False otherwise."""
    cluster_id_mat = cluster_id(N_sites)
    return cluster_id_mat[:, None] == cluster_id_mat[None, :]

def get_shift_op(site: int, l_max: int) -> np.ndarray:
    """Matrix of e^{i theta_i} in the |l1,l2> basis"""
    dim = 2 * l_max + 1

    l1 = np.repeat(np.arange(-l_max, l_max+1), dim)
    l2 = np.tile(np.arange(-l_max, l_max+1), dim)

    shift_op = np.zeros((dim**2, dim**2))
    if site == 0:
        mask = l1 < l_max
        bra_index = (l1[mask]+l_max+1)*dim + (l2[mask]+l_max)
    elif site == 1:
        mask = l2 < l_max
        bra_index = (l1[mask]+l_max)*dim + (l2[mask]+l_max+1)

    ket_index = np.where(mask)[0]
    shift_op[bra_index, ket_index] = 1.0

    return shift_op

# ====================================================================
#   SPINON SECTOR
# ====================================================================
def get_H_spinon(epsilon_vec, t_eff_mat, h_vec, mu: float) -> np.ndarray:
    """Return spinon Hamiltonian in a spin-subspace"""
    return np.diag(epsilon_vec - h_vec - mu) - t_eff_mat

def get_t_eff(t_mat, renorm_mat) -> np.ndarray:
    """Calculate effective spinon tunneling. t^{eff}_{ij} = t_{ij} Phi_{ij}"""
    return t_mat * renorm_mat

def get_nf_vec(evals_spinon, evecs_spinon, beta: Union[float, None], N_at: int) -> np.ndarray:
    fd_vals = Helpers.fermi(evals_spinon, beta)
    nf_vec = N_at * np.abs(evecs_spinon)**2 @ fd_vals
    return nf_vec

# ====================================================================
#   ROTOR SECTOR
# ====================================================================
def get_H_rotor(U: float, J_eff_mat, K_vec, h_vec, l_max: int, m: int) -> np.ndarray:
    """Calculate the m-th cluster's rotor Hamiltonian. Written in the |l1 l2> basis."""
    dim = 2 * l_max + 1
    l1 = np.repeat(np.arange(-l_max, l_max + 1), dim)
    l2 = np.tile(np.arange(-l_max, l_max + 1), dim)

    site1, site2 = cluster_to_sites(m)

    """Construct exact intra-cluster correlation part of Hamiltonian."""
    # Diagonal elements
    H = np.diag(U / 2 * (l1**2 + l2**2) + h_vec[0] * l1 + h_vec[1] * l2)

    mask = (l1 < l_max) & (l2 > -l_max) # Filters valid neighbors for shifting
    ket_index = np.where(mask)[0]
    bra_index = (l1[mask] + l_max + 1) * dim + (l2[mask] + l_max - 1)

    # Fills out off-diagonal elements for bra->ket
    H[bra_index, ket_index] -= J_eff_mat[site1, site2]
    H[ket_index, bra_index] -= J_eff_mat[site2, site1]

    """Construct U(1) symmetry breaking part."""
    # K_1 cos(theta_1): raises/lowers l1 by one, l2 untouched
    mask1 = l1 < l_max

    ket_1 = np.where(mask1)[0]
    bra_1 = (l1[mask1] + l_max + 1) * dim + (l2[mask1] + l_max)

    H[bra_1, ket_1] += K_vec[site1] / 2.0
    H[ket_1, bra_1] += K_vec[site1] / 2.0

    # K_2 cos(theta_2): raises/lowers l2 by one, l1 untouched
    mask2 = l2 < l_max

    ket_2 = np.where(mask2)[0]
    bra_2 = (l1[mask2] + l_max) * dim + (l2[mask2] + l_max + 1)

    H[bra_2, ket_2] += K_vec[site2] / 2.0
    H[ket_2, bra_2] += K_vec[site2] / 2.0

    return H

def get_J_eff_mat(evals_spinon, evecs_spinon, t_mat, beta: Union[float, None], N_at: int) -> np.ndarray:
    """Calculate rotor tunneling J_ij = \sum_{sigma} t_{ij} <f^+_{i sigma}f_{j sigma}>_f.
    
    evals_spinon    : (N,)  Spinon eigenvalues/energies
    evecs_spinon    : (N,N) Diagonalizing matrix of the spinon Hamiltonian. U[i,j] = i-th component of j-th eigenvector
    t_mat           : (N,N) Hopping matrix
    """
    U = evecs_spinon
    fd_vals = Helpers.fermi(evals_spinon, beta)
    return N_at * t_mat * ((U.conj() * fd_vals) @ U.T)  

def get_K_vec(J_eff_mat, phi_vec, N_sites: int) -> np.ndarray:
    """Calculate intra-cluster interaction parameter.
    K_i = -2 \sum_{j notin C(i)}J^{eff}_{ij} phi_J"""
    J_inter = np.where(cluster_mask(N_sites), 0.0, J_eff_mat)
    return -2 * (J_inter @ phi_vec)

def get_phi_vec(evals_rotor_vec, evecs_rotor_vec, l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Calculate <exp(i theta)> at every lattice site. 
    
    evals_rotor_vec     : (M,(2*l_max+1)**2)
    evecs_rotor_vec     : (M,(2*l_max+1)**2,(2*l_max+1)**2)
    """
    N_clusters = int(N_sites/2)
    phi_vec = np.zeros(N_sites)

    for m in range(N_clusters):
        cluster = cluster_to_sites(m)
        
        evals, evecs = evals_rotor_vec[m], evecs_rotor_vec[m]
        bz_weights = Helpers.bz_weights(evals, beta)

        for i in range(2):
            site = cluster[i]
            shift_op = get_shift_op(i, l_max)
            
            exp_vals = np.real(np.sum(evecs.conj() * (shift_op @ evecs), axis=0))

            phi_vec[site] = np.sum(bz_weights * exp_vals)
    
    return phi_vec

def get_bond_vec(evals_rotor_vec, evecs_rotor_vec, l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Calculate <exp(i(theta_1 - theta_2))> on every cluster.

    evals_rotor_vec     : (M,(2*l_max+1)**2)
    evecs_rotor_vec     : (M,(2*l_max+1)**2,(2*l_max+1)**2)
    """
    N_clusters = int(N_sites/2)
    bond_vec = np.zeros(N_clusters)

    # exp(i(theta_1 - theta_2)): raise l1, lower l2. Cluster-independent, so build once.
    bond_op = get_shift_op(0, l_max) @ get_shift_op(1, l_max).conj().T

    for m in range(N_clusters):
        evals, evecs = evals_rotor_vec[m], evecs_rotor_vec[m]
        bz_weights = Helpers.bz_weights(evals, beta)

        exp_vals = np.real(np.sum(evecs.conj() * (bond_op @ evecs), axis=0))

        bond_vec[m] = np.sum(bz_weights * exp_vals)

    return bond_vec

def get_renorm_mat(phi_vec, bond_vec, N_sites: int) -> np.ndarray:
    """Calculate the correlation matrix: Phi_{ij} = <exp[i(theta_i - theta_j)]> if sites i & j in same cluster; 
    Phi_{ij} = <exp(i theta_i)><exp(i theta_j)> otherwise.

    phi_vec      : (N,) Rotor condensate amplitude at every lattice site <exp(i theta)>
    bond_vec     : (M,) Intra-cluster rotor correlation amplitude <exp[i(theta_1 - theta_2)]>
    cluster_id   : (N,) cluster_id[i] = cluster # the i-th site belongs to
    cluster_mask : (N,N) cluster_mask[i,j] = True if sites i & j belong to same cluster; false otherwise
    """
    renorm_mat = np.outer(phi_vec, phi_vec) # Set Phi_ij = phi_i phi_j to start

    # Comb through and set correlated intra-cluster amplitudes
    for i in range(N_sites):
        for j in range(N_sites):
            if i == j:                  # By definition, should be unity on the diagonal
                renorm_mat[i,j] = 1    
            elif cluster_mask(N_sites)[i,j]:     # If in the same cluster, set exact correlation <exp[i(theta_1 - theta_2)]>
                renorm_mat[i,j] = bond_vec[cluster_id(N_sites)[i]]
    
    return renorm_mat

def get_L(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int, site: int) -> float:
    """Calculate thermal average <L_i> at a site.
    
    site    : site in cluster (0 or 1)
    """
    bz_weights = Helpers.bz_weights(evals_rotor, beta)
    l_vals = np.array(list(itertools.product(
            np.arange(-l_max, l_max+1), np.arange(-l_max, l_max+1)
    )))[:, site].astype(float)

    exp_vals = l_vals @ np.abs(evecs_rotor)**2
    L = np.sum(bz_weights * exp_vals)

    return L

def get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None], l_max: int, N_sites: int) -> np.ndarray:
    """Calculate thermal average <L_i> at every site.
    
    evals_rotor_vec     : (M,(2*l_max+1)**2)
    evecs_rotor_vec     : (M,(2*l_max+1)**2,(2*l_max+1)**2)
    """
    N_clusters = int(N_sites / 2)
    L_vec = np.zeros(N_sites)

    for m in range(N_clusters):
        cluster = cluster_to_sites(m)
        evals, evecs = evals_rotor_vec[m], evecs_rotor_vec[m]
        for i in range(2):
            site = cluster[i]
            L_vec[site] = get_L(evals, evecs, beta, l_max, i)

    return L_vec

def get_rotor_eig_vec(U: float, J_eff_mat, K_vec, h_vec, l_max: int, N_sites) -> tuple[np.ndarray, np.ndarray]:
    """Calculate eigenvalues and eigenvectors of the rotor Hamiltonian at every site"""
    dim_cluster = (2*l_max + 1)**2

    N_cluster = int(N_sites / 2)

    evals_rotor_vec = np.zeros((N_cluster, dim_cluster))
    evecs_rotor_vec = np.zeros((N_cluster, dim_cluster, dim_cluster))

    for m in range(N_cluster):
        evals_rotor_vec[m], evecs_rotor_vec[m] = np.linalg.eigh(get_H_rotor(U, J_eff_mat, K_vec, h_vec, l_max, m))

    return evals_rotor_vec, evecs_rotor_vec
