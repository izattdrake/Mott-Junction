import numpy as np
import matplotlib.pyplot as plt

from Exact.exact_dimer import ExactDimer
from Coupled.coupled_dimer import CoupledDimer
from Decoupled.decoupled_dimer import DecoupledDimer

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, None, 10
