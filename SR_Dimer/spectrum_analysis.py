import numpy as np
import exact_dimer
import utility.helpers as helpers

import mf_dimer_numerical as mf
from utility.plotting import plot_def_number_spectrum

if __name__ == '__main__':    
    l_max = 13
    # Define system parameters
    U = 0.0; t = 1.0; epsilon=1.0

    J_vals = np.arange(0,20)
    plot_def_number_spectrum(epsilon, t, J_vals, PLOT_P0=True, PLOT_P1=False, PLOT_P2=True, PLOT_P3=False, PLOT_P4=True)
        
    # MF iteration code that does not work
    for l_max in range(5, 30):
        # Doping setup
        mu = np.zeros(2)

        h = np.zeros(2)
        Z = np.full(2, 0.2)
        K = np.full(2, -0.3)

        did_converge = False
        for i in range(50):
            h_prime, Z_prime, K_prime, nf = mf.update(t, U, mu, h, Z, K, l_max)

            if np.all(np.abs(h - h_prime) < 1e-20) and np.all(np.abs(Z - Z_prime) < 1e-20) and np.all(np.abs(K - K_prime) < 1e-20):
                h, Z, K = h_prime, Z_prime, K_prime
                did_converge = True
                break
            else:
                print('update')
                h, Z, K = h_prime, Z_prime, K_prime

        if did_converge:
            H_rotors = mf.get_H_rotors(U, h, K, l_max)
            rotor_eigvals, rotor_eigvecs = np.linalg.eigh(H_rotors[0])
            rotor_gs = rotor_eigvecs[:,0]
            print(Z)
            print(f'K0 = {K[0]}')
            print(f'h0 = {h[0]}')
            print(f'nf0 = {nf[0]}')

            print(helpers.check_Z(U, K[0], h[0], nf[0]))
        else:
            print('did not converge')
