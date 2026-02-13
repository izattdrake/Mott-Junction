import numpy as np
import matplotlib.pyplot as plt

kB = 8.617333262e-5

def fermi_dirac(E, T, mu=0):
    x = (E - mu) / (2 * kB * T)
    occ = 0.0

    if T<0.5:
        occ = np.where(E < mu, 1.0, 0.0)
    else:
        occ = 0.5*(1-np.tanh(x))

    return occ

def H_spinon(mu, h, t, Z, N, sites):
    H = np.zeros((sites,sites))
    
    for i in range(sites-1):
        H[i,i] = N*(mu[i] - h[i])
        H[i,i+1] = -N*t*np.sqrt(Z[i]*Z[i+1])
        H[i+1,i] = -N*t*np.sqrt(Z[i]*Z[i+1])

    H[sites-1,sites-1] = N*(mu[sites-1]-h[sites-1])
    
    return H

def H_MF_rotor(hub_pot, h, K, ang_mom_sites, i):
    H = np.zeros((ang_mom_sites, ang_mom_sites))

    for l in range(ang_mom_sites-1):
        H[l,l] = (hub_pot/2)*l*l+h[i]*l
        H[l,l+1] = K[i]/2
        H[l+1,l] = K[i]/2

    return H

def get_K(U, t, Z, T, spinon_eigvals, sites):
    K = np.zeros(sites)

    for i in range(sites):
        Ki = 0
        for j in range(sites):
            Ki += -4*t*(U.conj()[i,j])*(U[i+1,j]*np.sqrt(Z[i+1]) + U[i-1,j]*np.sqrt(Z[i-1]))*fermi_dirac(spinon_eigvals[j],T)
        K[i] = Ki
    
    return K

def main():
    t = 1
    N = 2
    hub_pot = 1
    T = 300

    REAL_SITES = 51
    REAL_HALF_SITES = int(REAL_SITES/2-0.5)

    ang_mom_sites = 10

    # Initial parameter guess
    h = np.zeros(REAL_SITES)

    mu = np.zeros(REAL_SITES)
    mu[:REAL_HALF_SITES] = -0.5
    mu[REAL_HALF_SITES:] = 0.5

    Z = np.full(101,0.1)

    # Construct Hamiltonians
    Hf = H_spinon(mu,h,t,Z,N,REAL_SITES)

    # Fermion Hamiltonian eigenstates are many-spinon states. 
    # Single-spinon states are NOT eigenstates despite the variable name.
    spinon_eigvals, spinon_eigvecs = np.linalg.eig(Hf)


    K = get_K(spinon_eigvecs, t, Z, T, spinon_eigvals, REAL_SITES)
    Htheta = H_MF_rotor(hub_pot, h, K, ang_mom_sites, 0)

    rotor_eigvals, rotor_eigvecs = np.linalg.eig(Htheta)

    print(rotor_eigvals)


if __name__ == '__main__':
    main()