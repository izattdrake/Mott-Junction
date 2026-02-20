import numpy as np
from globals import *

def fermi_dirac(E, T, mu=0):
    x = (E - mu) / (2 * KB_SI * T)
    occ = 0.0

    if T < 0.5:
        occ = np.where(E < mu, 1.0, 0.0)
    else:
        occ = 0.5*(1-np.tanh(x))

    return occ