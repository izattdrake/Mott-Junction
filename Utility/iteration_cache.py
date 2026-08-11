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
    _VECTOR_NAME = "epsvec"          # the site-resolved parameter
    _INDEX_NAME = "epsindex"         # what replaces it in the key
    _SYS_SIZE_NAME = "Nsites"        # parameter holding the system size

    _WEIGHTS = {"U": 2.0}            # Weigh params differently for _distance calculation

    def __init__(self, cache_dir: str, model_tag: str, version: float=1.0, tol: float=1e-12) -> None:
        self._cache_dir = Path(cache_dir)
        self._model_tag = model_tag
        self._version = version
        self._tol = tol  

    # ====================================================================
    #   FORMATTING / COMPARISON METHODS
    # ====================================================================
    @staticmethod
    def _format(value: any) -> str:
        """Regulates parameter->string conversion (handles int/float/None)"""
        if value is None:
            return "None"
        elif isinstance(value, (bool, np.bool_)):
            return f"{int(value):d}"
        elif isinstance(value, (int, np.integer)):
            return f"{value:d}"
        elif isinstance(value, (float, np.floating)):
            return f"{value:.12g}"
        else:
            raise ValueError

    @staticmethod
    def _is_same_value(x: any, y: any) -> bool:
        """Bool check that also treats missing values as matching."""        
        return ((x == y) or (np.isnan(x) and np.isnan(y)))
    
    def _distance(self, wanted: dict, candidate: dict) -> float:
        """Return relative Euclidean distance in parameter space of two parameter sets.
        Each parameter contributes its MEAN squared relative deviation, so an
        epsilon_vec counts as one parameter regardless of system size.
        Mismatched shapes are incomparable"""
        dist_sq = 0.0
        for name, wanted_val in wanted.items():
            candidate_val = candidate[name]

            # Make everying a 1D array like epsilon_vec to be on same footing
            wanted_arr = np.atleast_1d(np.asarray(wanted_val, float))
            candidate_arr = np.atleast_1d(np.asarray(candidate_val, float))

            if wanted_arr.shape != candidate_arr.shape:
                return np.inf       # Different system size

            param_sq = 0.0                         
            for w_val, c_val in zip(wanted_arr, candidate_arr):
                if np.isnan(w_val) and np.isnan(c_val):
                    continue        # Both values absent, no distance

                if np.isnan(w_val) or np.isnan(c_val):
                    return np.inf   # One value absent, infinite distance to disqualify

                param_sq += ((w_val - c_val) / max(1.0, abs(w_val), abs(c_val))) ** 2   # <-- was dist_sq

            dist_sq += self._WEIGHTS.get(name, 1.0) * param_sq / wanted_arr.size   # Average epsilon_vec parameter distance

        return np.sqrt(dist_sq)
    
    # ====================================================================
    #   EPSILON_VEC HANDLING (ONE FILE PER SYSTEM SIZE)
    # ====================================================================
    def _eps_archive_path(self, N_sites: int) -> str:
        """Path of the epsilon_vec file for this system size."""
        return self._cache_dir / f"{self._model_tag}_{self._VECTOR_NAME}_{self._SYS_SIZE_NAME}{int(N_sites)}.npz"
    
    def _load_eps_archive(self, N_sites: int) -> str:
        """Reads all epsilon_vec's stored for this system size, of shape (N_, N_sites) 
        where N is the number of stored entries. Returns an empty (0, N) array if
        no epsilon_vec's have been recorded yet."""
        path = self._eps_archive_path(N_sites)
        empty = np.empty((0, int(N_sites)), dtype=float)

        if not path.exists():
            return empty
        
        try:
            with np.load(path, allow_pickle=False) as file:
                if int(file["__version__"]) != self._version:
                    return empty
                
                vecs = np.atleast_2d(np.asarray(file["epsilon_vecs"], float))

                if vecs.shape[1] != int(N_sites):
                    return empty    # Wrong size, no archive
                
                return vecs
        except Exception:
            return empty    # Cannot be read, return empty
        
    def _write_eps_archive(self, N_sites: int, epsilon_vecs: np.ndarray) -> None:
        """Automatically overwrite the epsilon archive for this system size."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=str(self._cache_dir), suffix=".npz")
        os.close(handle)

        try:
            np.savez_compressed(temp_name,
                                __version__=self._version,
                                __N_sites__=int(N_sites),
                                epsilon_vecs=np.asarray(epsilon_vecs, float))
            os.replace(temp_name, self._eps_archive_path(N_sites))
        except BaseException:
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise

    def _eps_index(self, epsilon_vec: np.ndarray, create: bool=True) -> int:
        """Index of this epsilon_vec in the archive. Searches the archive
        for epsilon_vec within self.tol. If it exists, its index is returned.
        If it doesn't, it is appended with the next inex returned if create=True
        Otherwise, None is returned."""
        N_sites = epsilon_vec.size
        archive = self._load_eps_archive(N_sites)

        if archive.shape[0] > 0:
            matches = np.all(np.abs(archive - epsilon_vec[None, :]) <= self._tol, axis=1)
            if np.any(matches):
                return int(np.flatnonzero(matches)[0])
            
        if not create:
            return None
        
        self._write_eps_archive(N_sites, np.vstack([archive, epsilon_vec[None, :]]))
        return int(archive.shape[0])
            
    def _eps_from_index(self, index: int, N_sites: int) -> np.ndarray:
        """Return epsilon_vec corresponding to index and N_sites. Returns
        None if doesn't exist."""
        archive = self._load_eps_archive(N_sites)
        if index < 0 or index >= archive.shape[0]:
            return None

        return archive[index]

    # ====================================================================
    #   PARAMETER KEY 
    # ====================================================================
    def _resolve(self, params: dict, create: bool=True) -> dict:
        """Replace epsilon_vec by its archive index. Returns the integer
        parameter dict or None if the vector is unknown and create=False."""
        if self._VECTOR_NAME not in params:
            return dict(params)
        
        resolved = {}

        for name, val in params.items():
            if name != self._VECTOR_NAME:
                resolved[name] = val
                continue

            index = self._eps_index(val, create=create)

            if index is None:
                return None
            
            resolved[self._INDEX_NAME] = index

        return resolved

    def _key(self, params: dict) -> str:
        """Creates key string for a set of parameters."""
        return "_".join(f"{name}{self._format(val)}" for name, val in params.items())

    def path(self, params: dict) -> str:
        """Creates file's path string for a set of parameters."""
        resolved = self._resolve(params, create=True)
        return self._cache_dir / f"{self._model_tag}_{self._key(resolved)}.npz"


    # ====================================================================
    #   READ / WRITE METHODS
    # ====================================================================
    def _read_entry(self, file_path: str, names) -> any:
        """Read one cached file; return None if it is unreadable, written by different version,
        or missing any needed quantity. Purpose is to report on stored parameters and their
        data for computing distance. IterationCache.load is a targeted lookup for a set of parameters."""
        try:
            with np.load(file_path, allow_pickle=False) as stored:
                if int(stored["__version__"]) != self._version:
                    return None
                
                # Input parameter dictionary; i.e. U, t, epsilon, beta, l_max
                param_names = [str(name) for name in stored["__param_names__"]]
                param_vals = np.asarray(stored["__param_values__"], float)
                params = {name: float(value) for name, value in zip(param_names, param_vals)}
                
                data = {name: stored[name] for name in names}

                return params, data
        except Exception:
            return None      # Cannot be read for some reason

    def load(self, params: dict, names: np.ndarray) -> any:
        """Returns dictionary {name: value} if data is found; returns None otherwise"""
        resolved = self._resolve(params, create=False)

        if resolved is None:
            return None     # This epsilon_vec has never been run
        
        path = self.path(params)

        if not path.exists():
            return None
        
        try:
            with np.load(path, allow_pickle=False) as file:
                if int(file["__version__"]) != self._version:
                    return None
                if str(file["__key__"]) != self._key(resolved):
                    return None
                return {name: file[name] for name in names}
        except Exception:
            return None      # If failure for some reason, return None and iterate again

    def save(self, params: dict, data) -> None:
        """Saves parameter dictionary {name: array-or-scalar} to file."""
        resolved = self._resolve(params, create=True)
        
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=str(self._cache_dir), suffix=".npz")
        os.close(handle)

        try:
            names = np.array(list(resolved.keys()))
            vals = np.array([np.nan if value is None else float(value)
                               for value in resolved.values()])
            np.savez_compressed(temp_name,
                                __version__=self._version,
                                __key__=self._key(resolved),
                                __param_names__=names,
                                __param_values__=vals,
                                **data)
            os.replace(temp_name, self.path(resolved))
        except BaseException:
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise
        
    # ====================================================================
    #   NEAREST PARAMETER LOOKUP
    # ====================================================================
    def _expand(self, params: dict) -> dict:
        """Swap epsilon_index back for the epsilon_vec it names to measure
        distances in parameter space."""
        if self._INDEX_NAME not in params or self._SYS_SIZE_NAME not in params:
            return dict(params)
        
        vec = self._eps_from_index(int(params[self._INDEX_NAME]),
                                      int(params[self._SYS_SIZE_NAME]))
        
        if vec is None:
            return None     # Archive missing epsilon entry
        
        expanded = {name: val for name, val in params.items() if name != self._INDEX_NAME}
        expanded[self._VECTOR_NAME] = vec

        return expanded
    
    def nearest(self, params: dict, names: np.ndarray, must_match=()) -> dict:
        """Find the cached data for input params closest in parameter space
        to those chosen. Parameters have equal weight with contribution
        being relative-scale Euclidean distance."""
        if not self._cache_dir.exists():
            return None
 
        # Wanted parameter set, turn type(None) into type(np.nan) for reading/writing
        wanted_params = {name: (val if name == self._VECTOR_NAME 
                                else (np.nan if val is None else float(val))) 
                                for name, val in params.items()}
        best_dist, best_entry = np.inf, None

        for file_path in self._cache_dir.glob(f"{self._model_tag}_*.npz"):
            if file_path.name.startswith(f"{self._model_tag}_{self._VECTOR_NAME}_{self._SYS_SIZE_NAME}"):
                continue    # Archive file, not a parameter data file

            entry = self._read_entry(file_path, names)

            if entry is None:
                continue

            candidate_params, candidate_data = entry
            candidate_params = self._expand(candidate_params)

            if candidate_params is None:
                continue    # Didn't recover epsilon_vec

            if set(candidate_params) != set(wanted_params):
                continue    # Files don't have same parameter set

            if not all(self._is_same_value(candidate_params[name], wanted_params[name]) 
                       for name in must_match):
                continue    # Must match parameters (i.e. Nat or N_size) don't match
            
            dist = self._distance(wanted_params, candidate_params)

            if dist < best_dist:
                best_dist = dist
                best_entry = {"params": candidate_params, 
                              "data": candidate_data, 
                              "distance": float(dist)}
                
        return best_entry

    # ====================================================================
    #   INSPECTION / MAINTENANCE
    # ====================================================================
    def summary(self) -> tuple[dict, int]:
        """(Archive size, number of stored data entries) for this model."""
        if not self._cache_dir.exists():
            return {}, 0
        
        archives, entries = {}, 0

        for file_path in sorted(self._cache_dir.glob(f"{self._model_tag}_*.npz")):
            if file_path.name.startswith(f"{self._model_tag}_{self._VECTOR_NAME}_{self._SYS_SIZE_NAME}"):
                N_sites = int(file_path.stem.split(f"_{self._SYS_SIZE_NAME}")[-1])
                archives[N_sites] = self._load_eps_archive(N_sites).shape[0]
            else:
                entries += 1

        return archives, entries
    
    def clear(self) -> None:
        """Delete every cache entry and epsilon_vec archive belonging to this model tag."""
        if self._cache_dir.exists():
            for file_path in self._cache_dir.glob(f"{self._model_tag}_*.npz"):
                file_path.unlink()
        