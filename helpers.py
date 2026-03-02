import numpy as np
from globals import *

def fermi_dirac(E, beta, mu=0):
    x = (E - mu) * beta / 2
    occ = 0.0

    if beta > 500:
        occ = np.where(E < mu, 1.0, 0.0)
    else:
        occ = 0.5*(1-np.tanh(x))

    return occ