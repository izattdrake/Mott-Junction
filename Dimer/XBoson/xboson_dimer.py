import numpy as np
import sys

from pathlib import Path
from .Iteration.broyden import iterate_broyden
from .Iteration.newton_gauge import iterate_newton_gauge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Utility import helpers
from Utility.iteration_cache import IterationCache

# ====================================================================
#   X-BOSON DIMER CLASS
# --------------------------------------------------------------------
#   An approximation is made to replace rotor operators exp(i theta_i)
#   with complex boson fields X_i with normalization constraint
#   |X_i|^2 treated on average. Solution extends beyond Florens' and
#   Georges' paper to include a finite dimer with doping. Full Green's
#   functions and spectral analysis is included.
# --------------------------------------------------------------------
#   I DON'T KNOW IF I WANT TO PUT THE EFFORT INTO FIGURING THIS MODEL
#   OUT. IT'S KIND OF USELESS FOR THE DOPING PROBLEM OR FINDING
# ====================================================================

class XBosonDimer:
    _CACHE_VERSION = 1.0
    _CACHE_DIR = Path(__file__).resolve().parent / '_cache'
    
    """Names of the converged quantities in iteration methods" return order.
    Parameter guesses looked for in nearest parameter lookup, needed for calling fresh iteration
    
    N: # of electrons
    Q: # of states on a site
    """
    _ITERATION_NAMES = ()
    _NEAREST_NAMES = ()

    def __init__(self, epsilon, U, t, beta=None, N=2, Q=2, iteration_method=0):
        self._epsilon_vec = np.array([-epsilon/2.0, epsilon/2.0]) # Even/opposite doping to preserve PHS
        self._t_mat = np.array([
            [0, t],
            [t, 0]
        ])
        self._U, self._beta, self.N, self.Q = U, beta, N, Q

        if iteration_method == 0: # Iteration w/ Newton's method & gauge fix
            None