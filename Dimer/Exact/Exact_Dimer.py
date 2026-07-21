import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility import helpers

# ====================================================================
#   Fermionic operators w/ Jordan-Wigner string construction
# --------------------------------------------------------------------
#   Construction of fermionic operators using the Jordan-Wigner
#   string construction. Fermion occupation is mapped to a spinor
#   and single-mode operators are 2x2 matrices. Operators acting
#   on the 16=2^4-dim Fock space are 16-dim matrices, constructed
#   with the Kronecker tensor product of 4 2x2 matrices.
# --------------------------------------------------------------------
#   This approach was done for the purpose of being explicit, but
#   it is not a good method for a crystal with more than 6 or 7
#   lattice sites. One should use QuSpin or another library for that
#   which uses more advanced methods to store large matrices.
# --------------------------------------------------------------------
#   This class structure differs from the slave-rotor dimers. Because
#   we do not need to iterate this system, we do not need to know
#   the temperature or filling beforehand. A class instance immediately
#   solves for the eigenspace solutions, then each method takes in the
#   temperature, etc, and solves for the observables.
# ====================================================================
a_ann = np.array([[0., 1.], [0., 0.]])     # annihilation on one mode
pauli_z = np.diag([1., -1.])               # (-1)^n string factor
id = np.eye(2)                             # Identity operator

def kronecker_matrix(*mats):
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out

# c[mode], mode = 0:1up 1:1dn 2:2up 3:2dn
c_ann = [kronecker_matrix(a_ann, id, id, id),
         kronecker_matrix(pauli_z, a_ann, id, id),
         kronecker_matrix(pauli_z, pauli_z, a_ann, id),
         kronecker_matrix(pauli_z, pauli_z, pauli_z, a_ann)]

c_cre = [mat.T for mat in c_ann]

# Mode index convention
def mode(site, spin):               
    return 2 * site + spin

# ====================================================================
#   EXACT INHOMOGENEOUS FINITE-TEMPERATURE HUBBARD DIMER CODE 
# --------------------------------------------------------------------
#   Solves the full Fock space of the Hamiltonian
#
#   H = sum_{i,s} (eps_i - mu) n_{i,s} + U sum_i n_{i,up} n_{i,dn}
#        - t sum_s (c^+_{1,s} c_{2,s} + h.c.)
#   
#   Fock space is F = 1+4+6+4+1, parity destroyed by doping so
#   Green's functions and resulting quantities are no longer 
#   diagonal.
# --------------------------------------------------------------------
#   Computes eigenstates, zero- and finite-temperature Green's
#   functions, self-energy w/ matrix Dyson's equation, and
#   quasiparticle weight. Plots above quantities.
#
#   Fermionic operators are given matrix representations, not
#   symbolic, for easy computation. 
#   mu = U/2
# ====================================================================
class Exact_Dimer:
    def __init__(self, epsilon, t, U, mu):
        self.eps_vec = np.array([-epsilon, epsilon])
        self.t, self.U, self.mu = t, U, mu

        H = np.zeros((16, 16))

        for site in (0, 1):
            n_up = c_cre[mode(site, 0)] @ c_ann[mode(site, 0)]
            n_down = c_cre[mode(site, 1)] @ c_ann[mode(site, 1)]
            H += (self.eps_vec[site] - mu) * (n_up + n_down) + U * n_up @ n_down
        for spin in (0, 1):
            tun = c_cre[mode(0, spin)] @ c_ann[mode(1, spin)] # tunneling operators
            H += -t * (tun + tun.T)

        self.evals, self.evecs = np.linalg.eigh(H)

    def get_gf(self, i, j, z, beta=None, spin=0):
        """Calculate Green's functions with the spectral representation | z can be Matsubara or retarded"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))

        bz = helpers._bz_weights(self.evals, beta)

        Mi = self.evecs.T @ c_ann[mode(i, spin)] @ self.evecs    # <m|c_i|n>
        Lj = self.evecs.T @ c_cre[mode(j, spin)] @ self.evecs    # <m|c_j^+|n>

        res = Mi * Lj.T * (bz[:, None] + bz[None, :])            # residues [m,n]
        pol = self.evals[None, :] - self.evals[:, None]          # poles E_n - E_m

        gf_vals = np.sum(res.ravel()[None, :] / (z[:, None] - pol.ravel()[None, :]), axis=1)
        return gf_vals

    def get_gf_mat(self, z, beta=None):
        """Calculate 2x2 Green's function matrix for the entire dimer"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        gf_mat = np.empty((len(z), 2, 2), dtype=complex)

        for i in (0, 1):
            for j in (0, 1):
                gf_mat[:, i, j] = self.get_gf(i, j, z, beta)

        return gf_mat

    def get_self_energy_matrix(self, z, beta=None):
        """Calculate self-energy from Dyson's equation"""
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        h0 = np.array([[self.eps_vec[0] - self.mu, -self.t],    # Single-particle Hamiltonian
                       [-self.t, self.eps_vec[1] - self.mu]])
        
        gf_bare = np.linalg.inv(z[:, None, None] * np.eye(2) - h0)
        self_energy_mat = np.linalg.inv(gf_bare) - np.linalg.inv(self.get_gf_mat(z, beta))
        return self_energy_mat

    def get_qp_weight(self, site=0, beta=None, domega=1e-4, delta=1e-4):
        """Calculate QP weight from retarded GF | Uses finite difference method"""
        self_energy = self.get_self_energy_matrix(np.array([-domega + 1j*delta, domega + 1j*delta]), beta)[:, site, site]
        Z = 1.0 / (1.0 - (self_energy[1].real - self_energy[0].real) / (2 * domega))
        return Z

    def qp_weight_matsubara(self, beta, site=0):
        """Calculates quasiparticle weight with Matsubara self-energy.
        Used to check retarded self-energy approach."""
        if beta is None:
            raise ValueError('Dimer/Exact/Exact_Dimer.qp_weight_matsubara | Error: beta must be finite')
        
        w0 = np.pi / beta                                                                       # Approximation for w0 = 0, not a Matsubara freq
        self_energy = self.get_self_energy_matrix(np.array([1j * w0]), beta)[0, site, site]     # Evaluate self energy at that point
        qp_weight = 1.0 / (1.0 - self_energy.imag / w0)
        return qp_weight

    def get_occupation(self, beta=None):
        """Calculate electron occupation at every site"""
        bz_weights = helpers._bz_weights(self.evals, beta)
        n = np.zeros(2)
        for site in range(2):
            for spin in range(2):
                nop = c_cre[mode(site, spin)] @ c_ann[mode(site, spin)]
                n[site] += bz_weights @ np.einsum('am,ab,bm->m', self.evecs, nop, self.evecs)
        return n
    
    def get_E_gs(self):
        """Return GS energy w/o grand-canonical chemical potential shift"""
        E_gs = self.evals[0] + self.U
        return E_gs
