# data_loaders/energy_matrix_loader.py
import numpy as np
import os
from typing import List, Tuple

def _construct_resource_path(filename: str) -> str:
    """
    Constructs the file path for a given resource file.
    This function navigates up one directory and then down into 'core/resources'.
    """
    path = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "core",
            "resources",
            filename,
        )
    )
    return os.path.normpath(path)

# data_loaders/energy_matrix_loader.py (NUEVA CORRECCIÓN - BASADA EN EL FORMATO REAL DEL TXT)

# data_loaders/energy_matrix_loader.py

import numpy as np
import os
from typing import List, Tuple

# La función _construct_resource_path se mantiene igual
def _construct_resource_path(filename: str) -> str:
    """
    Constructs the file path for a given resource file.
    This function navigates up one directory and then down into 'core/resources'.
    """
    path = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "core",
            "resources",
            filename,
        )
    )
    return os.path.normpath(path)


def _load_energy_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """
    Loads the Miyazawa-Jernigan interaction energy matrix from file.
    The file is read as a full 20x20 matrix where the upper triangle
    and diagonal contain the correct interaction values.
    
    FIXED: Correctly constructs the final symmetric matrix (M=M_diag + M_upper + M_upper.T).
    """
    path = _construct_resource_path("mj_matrix.txt")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # First line contains amino acid symbols
        symbols = lines[0].strip().split()
        n_aa = len(symbols)
        
        # Read the raw 20x20 matrix from file
        data_lines = [line.strip() for line in lines[1:] if line.strip()]
        energy_matrix_raw = np.loadtxt(data_lines, dtype=float)
        
        if energy_matrix_raw.shape != (n_aa, n_aa):
             raise ValueError(f"Matrix shape error. Expected ({n_aa}, {n_aa}), but got {energy_matrix_raw.shape}")
        
        # --- Simetrización Correcta ---
        # 1. Obtener el triángulo superior estricto (sin la diagonal, k=1)
        M_upper = np.triu(energy_matrix_raw, k=1) 
        
        # 2. Obtener la diagonal
        M_diag = np.diag(np.diag(energy_matrix_raw)) 
        
        # 3. Construir la matriz simétrica final: M_diag + M_upper + M_upper.T
        energy_matrix = M_diag + M_upper + M_upper.T

        # -----------------------------
        
        print(f"✅ Loaded MJ matrix: {n_aa}x{n_aa} with symbols: {symbols}")
        print(f"   Diagonal values: C={energy_matrix[0,0]:.2f}, M={energy_matrix[1,1]:.2f}, F={energy_matrix[2,2]:.2f}")
        print(f"   Off-diagonal sample: C-M={energy_matrix[0,1]:.2f}, M-F={energy_matrix[2,1]:.2f}")
        
        return energy_matrix, symbols
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading Miyazawa-Jernigan matrix: {e}")

# ... (El resto de las funciones _load_first_neighbors_matrix_file, etc., se mantiene igual)

def _load_first_neighbors_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """
    Loads the helix propensity matrix from file.
    Expected format:
    - First line: amino acid symbols (tab/space separated)
    - Following lines: full square matrix (n_aa x n_aa values)
    """
    path = _construct_resource_path("helix_pairs_prop.txt")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # First line contains amino acid symbols
        symbols = lines[0].strip().split()
        n_aa = len(symbols)
        
        # Initialize matrix
        helix_matrix = np.zeros((n_aa, n_aa))
        
        # Read full square matrix (starting from line 1)
        for i, line in enumerate(lines[1:]):
            if i >= n_aa:
                break
            values = line.strip().split()
            if not values:  # Skip empty lines
                continue
            
            # Convert strings to floats
            values = [float(v) for v in values]
            
            if len(values) != n_aa:
                raise ValueError(f"Row {i+1} has {len(values)} values, expected {n_aa}")
            
            # Fill row i with all values
            helix_matrix[i, :] = values
        
        print(f"✅ Loaded Helix Propensity matrix: {n_aa}x{n_aa} with symbols: {symbols}")
        print(f"   Diagonal values: A-A={helix_matrix[0,0]:.2f}, R-R={helix_matrix[1,1]:.2f}")
        print(f"   Off-diagonal sample: A-R={helix_matrix[0,1]:.2f}, R-N={helix_matrix[1,2]:.2f}")
        return helix_matrix, symbols
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading helix propensity matrix: {e}")

def _load_third_neighbors_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """Returns the helix i,i+3 interaction matrix from file."""
    path = _construct_resource_path("helix_sd_i_3.txt")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
        # The first line contains the symbols
        symbols = lines[0].strip().split()
        
        # The rest of lines contain the matrix data
        data_lines = [line.strip() for line in lines[1:] if line.strip()]
        helix_matrix = np.loadtxt(data_lines, dtype=float)
        
        print(f"✅ Loaded Helix i,i+3 matrix: {helix_matrix.shape} with symbols: {symbols}")
        return helix_matrix, symbols
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading helix i,i+3 matrix: {e}")
 
def _load_fourth_neighbors_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """Returns the helix i,i+4 interaction matrix from file."""
    path = _construct_resource_path("helix_sd_i_4.txt")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
        # The first line contains the symbols
        symbols = lines[0].strip().split()
        
        # The rest of lines contain the matrix data
        data_lines = [line.strip() for line in lines[1:] if line.strip()]
        helix_matrix = np.loadtxt(data_lines, dtype=float)
        
        print(f"✅ Loaded Helix i,i+4 matrix: {helix_matrix.shape} with symbols: {symbols}")
        return helix_matrix, symbols
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading helix i,i+4 matrix: {e}")