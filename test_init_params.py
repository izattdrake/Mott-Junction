import numpy as np
import matplotlib.pyplot as plt

from gen_sim import *
from globals import *
from helpers import *

def test_init_params():
    N = 2
    U_vals = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    mu_L_vals = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0]
    mu_R_vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    Z_L_vals = [0.05,0.1,0.2,0.3,0.4]
    Z_R_vals = [0.1,0.2,0.4]

    t = np.zeros(SITES)
    t0 = 1.0
    t[1] = t0; t[2] = t0/10.0

    K_vals = np.zeros((10, SITES))

    for i in range (0,10):
        K_vals[i] = np.full(SITES, -1.0/(i+1.0))

    mu = np.zeros(SITES)
    h = np.zeros(SITES)
    Z = np.zeros(SITES)

    success_data = []
    for U_val in U_vals:
        U = U_val
        for Z_left_val in Z_L_vals:
            for Z_right_val in Z_R_vals:
                for K_val in K_vals:
                    for mu_L_val in mu_L_vals:
                        for mu_R_val in mu_R_vals:
                            mu[:HALF_SITES] = mu_L_val; mu[HALF_SITES:] = mu_R_val
                            h[:HALF_SITES] = -mu_L_val; h[HALF_SITES:] = -mu_R_val
                            Z[:HALF_SITES] = Z_left_val; Z[HALF_SITES:] = Z_right_val
                            K = K_val
                            try: 
                                for i in range(15):
                                    h, Z, K, nf = update(t, U, mu, h, Z, K)
                                success_data.append({
                                        f't: {t0}',
                                        f'U: {U_val}',
                                        f'Z_L: {Z_left_val}',
                                        f'Z_R: {Z_right_val}',
                                        f'K: {K_val}',
                                        f'mu_L: {mu_L_val}',
                                        f'mu_R: {mu_R_val}',
                                        f'nf: {nf}'
                                })
                            except Exception as e:
                                break
                # Gives a sense of time for completion
    print(len(success_data))

    return np.array(success_data)

if __name__ == '__main__':
    data = test_init_params()
    print(data)