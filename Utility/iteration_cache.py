import numpy as np
import os
import tempfile
from pathlib import Path

# ====================================================================
#   ITERATION CACHE FOR READING/WRITING CONVERGED MF DATA 
# --------------------------------------------------------------------
#   Stores/loads named numpy arrays and scalars in an .npz keyed by a
#   parameter dict.
# --------------------------------------------------------------------
#   Input parameters are stored in file and compared with computation
#   input for reading converged data. Interrupted writes can never
#   leave a corrupt cache entry. A read failure returns None so the
#   class simply reiterates and writes again. Multiple MF approxmation
#   schemes are allowed; different model tags are used to distinguish
#   between them.
# ====================================================================

class IterationCache:
    def __init__(self, cache_dir, model_tag, version=1):
        self.cache_dir = Path(cache_dir)
        self.model_tag = model_tag
        self.version = version

    @staticmethod
    def _format(value):
        """Regulates parameter->string conversion (handles int/float/None)"""
        if value is None:
            return 'None'
        elif isinstance(value, (int, np.integer)):
            return f'{value:d}'
        elif isinstance(value, (float, np.floating)):
            return f'{value:.12g}'
        else:
            raise ValueError
        
    def _key(self, params):
        """Creates key string for a set of parameters."""
        key = '_'.join(f'{name}{self._format(val)}' for name, val in params.items())
        return key

    @staticmethod
    def _is_same_value(x, y):
        """Bool check that also treats missing values as matching."""
        check = x == y or (np.isnan(x) and np.isnan(y))
        return check
 
    @staticmethod
    def _distance(wanted, candidate):
        """Return relative Euclidean distance in parameter space of two parameter sets."""
        dist_sq = 0.0
        for name, wanted_val in wanted.items():
            candidate_val = candidate[name]

            if np.isnan(wanted_val) and np.isnan(candidate_val):
                continue # Both values absent, no distance
            
            if np.isnan(wanted_val) or np.isnan(candidate_val):
                return np.inf  # One value absent, infinite distance as set cannot be used
            
            dist_sq += ((wanted_val - candidate_val) / max(1.0, abs(wanted_val), abs(candidate_val))) ** 2

        dist = np.sqrt(dist_sq)
        return dist

    def _read_entry(self, file_path, names):
        """Read one cached file; return None if it is unreadable, written by different version,
        or missing any needed quantity. Purpose is to report on stored parameters and their
        data for computing distance. IterationCache.load is a targeted lookup for a set of parameters."""
        try:
            with np.load(file_path, allow_pickle=False) as stored:
                if int(stored['__version__']) != self.version:
                    return None
                
                # Input parameter dictionary; i.e. U, t, epsilon, beta, l_max
                param_names = [str(name) for name in stored['__param_names__']]
                param_vals = np.asarray(stored['__param_values__'], float)
                params = {name: float(value) for name, value in zip(param_names, param_vals)}
                
                data = {name: stored[name] for name in names}
                return params, data
        except Exception:
            return None      # Cannot be read for some reason

    def path(self, params):
        """Creates file's path string for a set of parameters."""
        file_path = self.cache_dir / f'{self.model_tag}_{self._key(params)}.npz'
        return file_path 

    def load(self, params, names):
        """Returns dictionary {name: value} if data is found; returns None otherwise"""
        path = self.path(params)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as file:
                if int(file['__version__']) != self.version:
                    return None
                if str(file['__key__']) != self._key(params):
                    return None
                return {name: file[name] for name in names}
        except Exception:
            return None      # If failure for some reason, return None and iterate again

    def save(self, params, data):
        """Saves parameter dictionary {name: array-or-scalar} to file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=str(self.cache_dir), suffix='.npz')
        os.close(handle)

        try:
            names = np.array(list(params.keys()))
            values = np.array([np.nan if value is None else float(value)
                               for value in params.values()])
            np.savez_compressed(temp_name,
                                __version__=self.version,
                                __key__=self._key(params),
                                __param_names__=names,
                                __param_values__=values,
                                **data)
            os.replace(temp_name, self.path(params))
        except BaseException:
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise
        
    def nearest(self, params, names, must_match=()):
        """Find the cached data for input params closest in parameter space
        to those chosen. Parameters have equal weight with contribution
        being relative-scale Euclidean distance."""
        if not self.cache_dir.exists():
            return None
 
        # Wanted parameter set, turn type(None) into type(np.nan) for reading/writing
        wanted_params = {name: (np.nan if value is None else float(value)) for name, value in params.items()}
        best_distance, best_entry = np.inf, None

        for file_path in self.cache_dir.glob(f'{self.model_tag}_*.npz'):
            entry = self._read_entry(file_path, names)

            if entry is None:
                continue

            candidate_params, candidate_data = entry
 
            # Verify both files have the same parameter set
            if set(candidate_params) != set(wanted_params):
                continue

            # must agree exactly on the parameters flagged in must_match=()
            if not all(self._is_same_value(candidate_params[name], wanted_params[name]) for name in must_match):
                continue
 
            distance = self._distance(wanted_params, candidate_params)
            if distance < best_distance:
                best_distance = distance
                best_entry = {'params': candidate_params, 'data': candidate_data, 'distance': float(distance)}
                        
        return best_entry

    def clear(self):
        """Delete every cache entry belonging to this model tag"""
        if self.cache_dir.exists():
            for file_path in self.cache_dir.glob('%s_*.npz' % self.model_tag):
                file_path.unlink()
