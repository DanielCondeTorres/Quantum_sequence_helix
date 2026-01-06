import numpy as np

def decode_solution_logic(bitstring, L, bits_per_pos, amino_acids):
    """Decodifica una cadena binaria a secuencia de aminoácidos."""
    decoded_sequence = ""
    for i in range(L):
        pos_code_str = bitstring[i*bits_per_pos:(i+1)*bits_per_pos]
        try:
            pos_code_int = int(pos_code_str, 2)
            if pos_code_int < len(amino_acids):
                decoded_sequence += amino_acids[pos_code_int]
            else:
                decoded_sequence += 'X'
        except ValueError:
            decoded_sequence += 'X'
    return decoded_sequence

def mask_invalid_probabilities(probs, L, bits_per_pos, n_aa, n_qubits):
    """Filtra estados binarios que no mapean a aminoácidos válidos."""
    if probs.size == 0: return probs
    masked = probs.copy()
    for idx in range(masked.size):
        if masked[idx] < 1e-10: continue
        bitstr = format(idx, f'0{n_qubits}b')
        is_valid = True
        for pos in range(L):
            code = int(bitstr[pos*bits_per_pos : (pos+1)*bits_per_pos], 2)
            if code >= n_aa:
                is_valid = False
                break
        if not is_valid:
            masked[idx] = 0.0
    
    total = np.sum(masked)
    return masked / total if total > 0 else masked