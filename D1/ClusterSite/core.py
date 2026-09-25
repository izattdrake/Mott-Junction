import numpy as np
import sys
import itertools

from typing import Union, Tuple
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utility.helpers import Helpers

# ====================================================================
#   METHODS TO HELP W/ ROTOR BASIS & CLUSTER COUNTING
# ====================================================================
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

def get_eff_rotor_basis(l_max: int):
    """Returns |l1 l2> basis elements."""
    l_vals = np.arange(-l_max, l_max + 1)
    basis = list(itertools.product(
        l_vals, l_vals
    ))
 
    return np.array(basis)
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

# ====================================================================
#   SPINON SECTOR
# ====================================================================
def get_H_spinon(epsilon_vec, t_eff_mat, h_vec) -> np.ndarray:
    """Return spinon Hamiltonian in a spin-subspace"""
    return np.diag(epsilon_vec - h_vec) - t_eff_mat

def get_occ_vec(evals_spinon, beta: Union[float, None], N_sites: int,
                N_at: int, N_target: int, deg_tol: float=1e-10) -> Tuple[np.ndarray, float]:
    """Calculate orbital occupancy vector and chemical potential to enforce
    particle number. Occupancy is the per-spin occupation of an orbital."""
    N_max = N_at * N_sites
    if not (0.0 <= N_target <= N_max):
        raise ValueError(f"N_target must lie in [0, {N_max}]")
    
    # Automatically lock occupation to lowest-energy state stack
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
            occ_vec[i: j+1] = fill / g
            remaining -= fill
            i = j + 1

        return occ_vec, 0.0
    
    # At finite temperature, conduct bisection algo to find chemical potential
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

def get_t_eff_mat(t_mat, renorm_mat) -> np.ndarray:
    """Calculate effective spinon tunneling. t^{eff}_{ij} = t_{ij} Phi_{ij}"""
    return t_mat * renorm_mat

def get_rho_mat(evecs_spinon, occ_vec):
    """Calculate the thermal hopping matrix rho_{ij} = <f^+_j f_i>.
    NOTE: Other implementations have it wrong; need to flip i<->j.
    Fine if everything's real, must be changed if complex."""
    return (evecs_spinon * occ_vec[np.newaxis, :]) @ evecs_spinon.conj().T

def get_nf_vec(evecs_spinon, occ_vec, N_at: int) -> np.ndarray:
    """Calculate spinon occupation at all sites"""
    return N_at * (np.abs(evecs_spinon)**2 @ occ_vec) 

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
    H = np.diag(U / 2 * (l1**2 + l2**2) + h_vec[site1] * l1 + h_vec[site2] * l2)

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

def get_rotor_eig_vec(U: float, J_eff_mat, K_vec, h_vec, l_max: int, N_sites) -> tuple[np.ndarray, np.ndarray]:
    """Calculate eigenvalues and eigenvectors of the rotor Hamiltonian for every cluster."""
    dim_cluster = (2*l_max + 1)**2

    N_cluster = int(N_sites / 2)

    evals_rotor_vec = np.zeros((N_cluster, dim_cluster))
    evecs_rotor_vec = np.zeros((N_cluster, dim_cluster, dim_cluster))

    for m in range(N_cluster):
        evals_rotor_vec[m], evecs_rotor_vec[m] = np.linalg.eigh(get_H_rotor(U, J_eff_mat, K_vec, h_vec, l_max, m))

    return evals_rotor_vec, evecs_rotor_vec

def get_J_eff_mat(t_mat, rho_mat, N_at: int) -> np.ndarray:
    """Calculate rotor tunneling J_ij = \sum_{sigma} t_{ij} <f^+_{i sigma}f_{j sigma}>_f."""
    return N_at * t_mat * rho_mat

def get_K_vec(J_eff_mat, phi_vec, N_sites: int) -> np.ndarray:
    """Calculate intra-cluster interaction parameter.
    K_i = -2 \sum_{j notin C(i)}J^{eff}_{ij} phi_J"""
    J_inter = np.where(cluster_mask(N_sites), 0.0, J_eff_mat)
    return -2 * (J_inter @ phi_vec)

def get_phi_vec(evals_rotor_vec, evecs_rotor_vec, l_max: int, 
                beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Calculate <exp(i theta)> at every lattice site. Not the quasiparticle weight. 
    
    evals_rotor_vec : (M,(2*l_max+1)^2)
    evecs_rotor_vec : (M,(2*l_max+1)^2,(2*l_max+1)^2)"""
    N_clusters = int(N_sites / 2)
    phi_vec = np.zeros(N_sites)

    for m in range(N_clusters):
        cluster = cluster_to_sites(m)
        
        evals, evecs = evals_rotor_vec[m], evecs_rotor_vec[m]
        bz_weights = Helpers.bz_weights(evals, beta)

        # Two site cluster
        for i in range(2):
            site = cluster[i]
            shift_op = get_shift_op(i, l_max)
            
            exp_vals = np.real(np.sum(evecs.conj() * (shift_op @ evecs), axis=0))

            phi_vec[site] = np.sum(bz_weights * exp_vals)
    
    return phi_vec

def get_bond_vec(evals_rotor_vec, evecs_rotor_vec, l_max: int, beta: Union[float, None], N_sites: int) -> np.ndarray:
    """Calculate <exp(i(theta_1 - theta_2))> on every cluster.

    evals_rotor_vec : (M,(2*l_max+1)^2)
    evecs_rotor_vec : (M,(2*l_max+1)^2,(2*l_max+1)^2)
    """
    N_clusters = int(N_sites/2)
    bond_vec = np.zeros(N_clusters)

    # exp(i(theta_1 - theta_2)): raise l1, lower l2. Cluster-independent operator
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
    renorm_mat = np.outer(phi_vec, phi_vec.conj()) # Set Phi_ij = phi_i phi_j to start
    np.fill_diagonal(renorm_mat, 1.0)              # Diagonal is 1.0 by definition

    i = np.arange(0, N_sites, 2);                  # Helper array to select clusters

    renorm_mat[i, i+1] = bond_vec
    renorm_mat[i+1, i] = bond_vec.conj()

    return renorm_mat

def _thermal_diag(evals_rotor, evecs_rotor, beta: Union[float, None], diag_op):
    """Thermal average of an operator that is diagonal in the |l1,l2> basis.
    Unlike the decoupled case there is one rotor Hamiltonian for the whole dimer."""
    probs = np.abs(evecs_rotor)**2            
    exp_vals = diag_op @ probs              
    return np.sum(Helpers.bz_weights(evals_rotor, beta) * exp_vals)

def _L(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int, site: int) -> float:
    """Calculate thermal average <L_i> at a site in a cluster.
    Eigenvalues and eigenvectors must be of the cluster while
    site is 0 or 1, labeling the site we're after in the cluster."""
    l_vals = get_eff_rotor_basis(l_max)[:, site].astype(float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals)

def _L2(evals_rotor, evecs_rotor, beta: Union[float, None], l_max: int, site: int) -> float:
    """Thermal average <L_i^2> at one site."""
    l_vals = get_eff_rotor_basis(l_max)[:, site].astype(float)
    return _thermal_diag(evals_rotor, evecs_rotor, beta, l_vals**2)

def get_L_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None], l_max: int, N_sites: int) -> np.ndarray:
    """Calculate thermal average <L_i> at every site.
    
    evals_rotor_vec     : (M,(2*l_max+1)**2)
    evecs_rotor_vec     : (M,(2*l_max+1)**2,(2*l_max+1)**2)"""
    L_vec = np.zeros(N_sites)

    for m in range(N_sites // 2):
        cluster = cluster_to_sites(m)
        evals, evecs = evals_rotor_vec[m], evecs_rotor_vec[m]
        for i in range(2):
            site = cluster[i]
            L_vec[site] = _L(evals, evecs, beta, l_max, i)

    return L_vec

def get_L2_vec(evals_rotor_vec, evecs_rotor_vec, beta: Union[float, None], l_max: int, N_sites: int) -> np.ndarray:
    """Calculate thermal average <L^2_i> at every site.
    
    evals_rotor_vec : (M,(2*l_max+1)**2)
    evecs_rotor_vec : (M,(2*l_max+1)**2,(2*l_max+1)**2)"""
    L2_vec = np.zeros(N_sites)

    # Loop over clusters
    for m in range(N_sites // 2):
        evals, evecs = evals_rotor_vec[m], evecs_rotor_vec[m]
        for i in range(2):
            L2_vec[cluster_to_sites(m)[i]] = _L2(evals, evecs, beta, l_max, i)

    return L2_vec

# ====================================================================
#   CONSTRAINT METHODS
# ====================================================================
def get_residual(nf_vec, L_vec, N_at: int):
    """Calculate constraint/charge vector f_i = <L_i> + 1 - <n^f_i>"""
    return L_vec - nf_vec + N_at / 2.0
