import numpy as np
import matplotlib.pyplot as plt
import json

from gen_sim import *
from globals import *
from helpers import *

def test_init_params():
    N = 2

    t = np.zeros(SITES)
    t0 = 1.0
    t[1] = t0; t[2] = t0/8.0

    U_vals = [10.0]
    muL_vals = [-0.1]
    muR_vals = [0.2]
    ZL_vals = [0.1]
    ZR_vals = [0.1]
    K_vals = np.zeros((1, SITES))
    for i in range (0,1):
        K_vals[i] = np.full(SITES, -0.3/(i+1.0))

    mu = np.zeros(SITES)
    h = np.zeros(SITES)
    Z = np.zeros(SITES)

    good_params = []
    bad_params = []

    for U_val in U_vals:
        U = U_val
        for ZL_val in ZL_vals:
            for ZR_val in ZR_vals:
                for K_val in K_vals:
                    for muL_val in muL_vals:
                        for muR_val in muR_vals:
                            mu[:HALF_SITES] = muL_val; mu[HALF_SITES:] = muR_val
                            Z[:HALF_SITES] = ZL_val; Z[HALF_SITES:] = ZR_val
                            h = -mu
                            K = K_val
                            try: 
                                for i in range(30):
                                    h_updated, Z, K, nf = update(t, U, mu, h, Z, K)
                                    print(Z)
                                    if np.all(np.abs(h - h_updated) < 1e-4):
                                        good_params.append({
                                                't': t0,
                                                'U': U_val,
                                                'ZL': ZL_val,
                                                'ZR': ZR_val,
                                                'Z': Z,
                                                'K_val': K_val,
                                                'K': K,
                                                'muL': muL_val,
                                                'muR': muR_val,
                                                'h': h,
                                                'nf': nf
                                        })
                                        sites = np.arange(SITES)

                                        fig, ax = plt.subplots(figsize=(8, 5))

                                        ax.plot(sites, nf, 'o-', color='steelblue', markersize=8, linewidth=2, label=r'$\langle n_{f,i} \rangle$')
                                        ax.axvline(x=HALF_SITES - 0.5, color='black', linestyle='--', linewidth=1.5, label='Junction')
                                        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label='Half filling')
                                        ax.set_xlabel('Site $i$', fontsize=14)
                                        ax.set_ylabel(r'$\langle n_{f,i} \rangle$', fontsize=14)
                                        ax.set_title('Spinon Occupancy Across Mott Junction', fontsize=14)
                                        ax.legend(fontsize=12)
                                        ax.set_xlim(-0.5, SITES - 0.5)
                                        ax.set_xticks(sites)
                                        ax.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        plt.show()

                                        break
                                    else:
                                        h = h_updated
                            except Exception as e:
                                print('ERROR')
                                continue

                    # Gives a sense of time for completion
                    print(len(good_params))

    return np.array(good_params)

if __name__ == '__main__':
    data = test_init_params()
    print(data)