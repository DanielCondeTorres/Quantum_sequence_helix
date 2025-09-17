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

def _parse_energy_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Parses a matrix loaded from the Miyazawa-Jernigan potential file.
    It removes the header and converts the string values to float.
    """
    # Create a new numpy array from the data, converting to float
    energy_matrix = np.zeros((np.shape(matrix)[0] - 1, np.shape(matrix)[1] - 1))
    for row in range(1, np.shape(matrix)[0]):
        for col in range(1, np.shape(matrix)[1]):
            energy_matrix[row-1, col-1] = float(matrix[row, col])
    return energy_matrix


# _load_first_neighbors_matrix_file, _load_third_neighbors_matrix_file, _load_fourth_neighbors_matrix_file

def  _load_energy_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """Loads the Miyazawa-Jernigan interaction energy matrix from file."""
    path = _construct_resource_path("mj_matrix.txt")
    
    try:
        # Load the raw data as strings
        matrix = np.loadtxt(fname=path, dtype=str)
        
        # The symbols are in the first row and first column.
        symbols = list(matrix[0, 1:])
        
        # Parse the matrix data itself
        energy_matrix = _parse_energy_matrix(matrix)
        
        return energy_matrix, symbols
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading Miyazawa-Jernigan matrix: {e}")

def _load_first_neighbors_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """Returns the helix pairs propensity matrix from file."""
    path = _construct_resource_path("helix_pairs_prop.txt")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
            # The first line contains the symbols
            symbols = lines[0].strip().split()
            
            # The rest of lines contain the matrix data
            data_lines = [line.strip() for line in lines[1:]]
            helix_matrix = np.loadtxt(data_lines, dtype=float)
            
        return helix_matrix, symbols
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading helix pairs matrix: {e}")
    
    
    
def _load_third_neighbors_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """Returns the helix pairs propensity matrix from file."""
    path = _construct_resource_path("helix_sd_i_3.txt")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
            # The first line contains the symbols
            symbols = lines[0].strip().split()
            
            # The rest of lines contain the matrix data
            data_lines = [line.strip() for line in lines[1:]]
            helix_matrix = np.loadtxt(data_lines, dtype=float)
            
        return helix_matrix, symbols
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading helix pairs matrix: {e}")
 
 
def _load_fourth_neighbors_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """Returns the helix pairs propensity matrix from file."""
    path = _construct_resource_path("helix_sd_i_4.txt")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
            # The first line contains the symbols
            symbols = lines[0].strip().split()
            
            # The rest of lines contain the matrix data
            data_lines = [line.strip() for line in lines[1:]]
            helix_matrix = np.loadtxt(data_lines, dtype=float)
            
        return helix_matrix, symbols
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {path} was not found. Check the path and filename.")
    except Exception as e:
        raise Exception(f"Error loading helix pairs matrix: {e}")