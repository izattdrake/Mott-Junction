import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
L = 100                 # half system size
N = 2 * L + 1           # total sites
t = 1.0                 # hopping
mu_L = 0.5             # left chemical potential
mu_R = -0.5              # right chemical potential
E_F = 0.0               # Fermi energy (T=0)

# -----------------------
# Lattice coordinates
# -----------------------
x = np.arange(-L, L + 1)
print(x)
# Chemical potential profile
mu = np.where(x < 0, mu_L, mu_R)
print(mu)
# -----------------------
# Hamiltonian
# -----------------------
H = np.zeros((N, N), dtype=float)

# Onsite terms
for i in range(N):
    H[i, i] = mu[i]

# Nearest-neighbor hopping
for i in range(N - 1):
    H[i, i + 1] = -t
    H[i + 1, i] = -t

# -----------------------
# Diagonalize
# -----------------------
E, psi = np.linalg.eigh(H)

# -----------------------
# Fill states (T=0)
# -----------------------
occupied = E < E_F

# -----------------------
# Charge density
# n(x) = sum_{occupied} |psi_n(x)|^2
# -----------------------
density = np.sum(np.abs(psi[:, occupied])**2, axis=1)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(7,4))
plt.plot(x, density, lw=2)
plt.axvline(0, color='k', ls='--', alpha=0.5)
plt.xlabel("x")
plt.ylabel("Charge density ⟨n_x⟩")
plt.title("Charge distribution across chemical potential step")
plt.tight_layout()
plt.show()
