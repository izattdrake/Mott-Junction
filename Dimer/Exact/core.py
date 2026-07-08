import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# ====================================================================

a_ann = np.array([[0., 1.], [0., 0.]])     # annihilation on one mode
pauli_z = np.diag([1., -1.])               # (-1)^n string factor
ide = np.eye(2)                       # Identity operator

def kronecker_matrix(*mats):
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out

# c[mode], mode = 0:1up 1:1dn 2:2up 3:2dn
c_ann = [kronecker_matrix(a_ann, ide, ide, ide),
         kronecker_matrix(pauli_z, a_ann, ide, ide),
         kronecker_matrix(pauli_z, pauli_z, a_ann, ide),
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
# ====================================================================

class HubbardDimer:
    def __init__(self, eps1, eps2, t, U, mu):
        self.eps_vec = np.array([eps1, eps2])
        self.t, self.U, self.mu = t, U, mu

        H = np.zeros((16, 16))
        for site in (0, 1):
            n_up = c_cre[mode(site, 0)] @ c_ann[mode(site, 0)]
            n_down = c_cre[mode(site, 1)] @ c_ann[mode(site, 1)]
            H += (self.eps_vec[site] - mu) * (n_up + n_down) + U * n_up @ n_down
        for spin in (0, 1):
            hop = c_cre[mode(0, spin)] @ c_ann[mode(1, spin)]
            H += -t * (hop + hop.T)
        self.evals, self.evecs = np.linalg.eigh(H)

    def get_bz_weights(self, beta):
        """
            Boltzmann Statistical Probabilities
            If T=0, beta=None and all weight is locked to the ground state
        """
        if beta is None:                          
            bz = (self.evals - self.evals[0] < 1e-10).astype(float) # Lock all weight to the GS
        else:
            bz = np.exp(-beta * (self.evals - self.evals[0]))

        return bz / bz.sum()

    def get_gf(self, i, j, z, beta=None, spin=0):
        """
        Green's function calculated with the spectral Lehmann representation
        G_ij(z) = sum_{m,n} <m|c_i|n><n|c_j^+|m> (w_m + w_n) / (z - (E_n - E_m))
        where z = i\omega_n, \omega + i\delta^+, etc
        """
        weights = self.get_bz_weights(beta)

        Ci = self.evecs.T @ c_ann[mode(i, spin)] @ self.evecs    # <m|c_i|n>
        Cj = self.evecs.T @ c_cre[mode(j, spin)] @ self.evecs    # <m|c_j^+|n>

        res = Ci * Cj.T * (weights[:, None] + weights[None, :])  # residues [m,n]
        pol = self.evals[None, :] - self.evals[:, None]          # poles E_n - E_m

        z = np.atleast_1d(np.asarray(z, dtype=complex))

        return ((res.ravel()[None, :] / (z[:, None] - pol.ravel()[None, :])).sum(axis=1))

    def get_gf_mat(self, z, beta=None):
        """
        Green's function matrix, corresponding to [[G_{11}, G_{12}], [G_{21}, G_{22}]]
        2x2 GF matrix in site space; shape (len(z), 2, 2)
        """
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        G = np.empty((len(z), 2, 2), dtype=complex)

        for i in (0, 1):
            for j in (0, 1):
                G[:, i, j] = self.get_gf(i, j, z, beta)

        return G

    def matsubara(self, beta, n_max=40):
        """
        Return fermionic Matsubara frequencies
        """
        return 1j * (2 * np.arange(n_max) + 1) * np.pi / beta

    # -- self-energy: Sigma(z) = G0^-1 - G^-1 --------------------------
    def get_self_energy(self, z, beta=None):
        """
        Calculate self-energy from Dyson's equation
        \Sigma_{ij}(z) = G0_{ij}^{-1} - G_{ij}^{-1}
        """
        h0 = np.array([[self.eps_vec[0] - self.mu, -self.t],    # Single-particle matrix
                       [-self.t, self.eps_vec[1] - self.mu]])
        
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        G0 = np.linalg.inv(z[:, None, None] * np.eye(2) - h0)

        self_energy = np.linalg.inv(G0) - np.linalg.inv(self.get_gf_mat(z, beta))
        return self_energy

    def qp_weight(self, site=0, beta=None, d=1e-4):
        """
        Calculate quasiparticle weight from retarded Green's function
        Z = [1 - d Re Sigma_{ii} / d\omega |_{\omega=0}]^{-1}
        Uses finite difference method
        """
        S = self.get_self_energy(np.array([-d + 1e-6j, d + 1e-6j]), beta)[:, site, site]
        return 1.0 / (1.0 - (S[1].real - S[0].real) / (2 * d))

    def qp_weight_matsubara(self, beta, site=0):
        """
        Calculates quasiparticle weight with Matsubara self-energy
        Used to check retarded self-energy approach
        """
        w0 = np.pi / beta
        self_energy = self.get_self_energy(np.array([1j * w0]), beta)[0, site, site]
        return 1.0 / (1.0 - self_energy.imag / w0)

    def occupation(self, beta=None):
        """
        Calculate electron occupation at each site at a given temperature
        """
        bz_weights = self.get_bz_weights(beta)
        n = np.zeros(2)
        for i in (0, 1):
            for s in (0, 1):
                nop = c_cre[mode(i, s)] @ c_ann[mode(i, s)]
                n[i] += bz_weights @ np.einsum('am,ab,bm->m', self.evecs, nop, self.evecs)
        return n
