# utils/general_utils.py
import numpy as np

def get_qubit_index(position: int, bit_idx: int, bits_per_pos: int) -> int:
    """Convert (position, bit_index) to qubit index under binary encoding"""
    return position * bits_per_pos + bit_idx

def decode_solution(bitstring: str, L: int, bits_per_pos: int, amino_acids: list[str]) -> str:
    """
    Decodes a binary bitstring solution into a protein sequence.

    Args:
        bitstring (str): The binary string representing the solution.
        L (int): The length of the protein sequence.
        bits_per_pos (int): The number of bits used to encode each amino acid.
        amino_acids (list[str]): The list of amino acids.

    Returns:
        str: The decoded protein sequence.
    """
    decoded_sequence = []
    n_aa = len(amino_acids)
    for i in range(L):
        code = 0
        for k in range(bits_per_pos):
            q = get_qubit_index(i, k, bits_per_pos)
            if q < len(bitstring) and bitstring[q] == '1':
                code |= (1 << k)
        if code < n_aa:
            decoded_sequence.append(amino_acids[code])
        else:
            decoded_sequence.append('X')  # invalid code
    return ''.join(decoded_sequence)

def compute_energy_from_bitstring(bitstring: str, pauli_terms: list[tuple]) -> float:
    """
    Compute classical energy of a computational-basis bitstring under Z-only Hamiltonian.
    """
    if not pauli_terms:
        return 0.0
    
    # Map bit -> Z eigenvalue (+1 for |0>, -1 for |1>)
    z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
    energy = 0.0
    for coeff, pauli_string in pauli_terms:
        prod = 1.0
        for i, p in enumerate(pauli_string):
            if p == 'Z':
                prod *= z_vals[i]
        energy += coeff * prod
    return float(energy)

def _codes_to_bitstring(codes: list[int], L: int, bits_per_pos: int, n_aa: int) -> str:
    """
    Converts a list of integer codes to a single binary bitstring.
    """
    bits = ['0'] * (L * bits_per_pos)
    for i, code in enumerate(codes):
        code = max(0, min(n_aa - 1, int(code)))
        for k in range(bits_per_pos):
            q = get_qubit_index(i, k, bits_per_pos)
            bits[q] = '1' if ((code >> k) & 1) else '0'
    return ''.join(bits)