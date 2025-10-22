import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from numba import jit, prange
from functools import lru_cache
import warnings
try:
    from qiskit import QuantumCircuit
    from qiskit_algorithms import QAOA, VQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_aer import AerSimulator
    # Prefer Aer primitives if available (prevents None/default selection issues)
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
        _HAS_AER_PRIMITIVES = True
    except Exception:
        _HAS_AER_PRIMITIVES = False
    # Fallback primitives
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator, Sampler
    # Standard library ansatz circuits
    try:
        from qiskit.circuit.library import RealAmplitudes, EfficientSU2
    except Exception:
        RealAmplitudes = None
        EfficientSU2 = None
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==================== FUNCIONES NUMBA COMPARTIDAS ====================

@jit(nopython=True, cache=True, fastmath=True)
def compute_energy_numba(bitstring_array: np.ndarray, coeffs: np.ndarray, 
                         pauli_indices: np.ndarray, pauli_types: np.ndarray) -> float:
    """Cálculo ultra-rápido de energía con Numba JIT."""
    energy = 0.0
    z_values = 1.0 - 2.0 * bitstring_array
    
    for term_idx in range(len(coeffs)):
        pauli_prod = 1.0
        start_idx = pauli_indices[term_idx]
        end_idx = pauli_indices[term_idx + 1]
        
        for p_idx in range(start_idx, end_idx):
            qubit_idx = pauli_types[p_idx]
            pauli_prod *= z_values[qubit_idx]
        
        energy += coeffs[term_idx] * pauli_prod
    
    return energy

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def get_marginals_numba(probs: np.ndarray, L: int, bits_per_pos: int, 
                        n_aa: int) -> np.ndarray:
    """Cálculo paralelo de marginales."""
    marginals = np.zeros((L, n_aa))
    n_states = len(probs)
    
    for pos in prange(L):
        shift = (L - 1 - pos) * bits_per_pos
        mask = (1 << bits_per_pos) - 1
        
        for state in range(n_states):
            aa_code = (state >> shift) & mask
            if aa_code < n_aa:
                marginals[pos, aa_code] += probs[state]
    
    return marginals

@jit(nopython=True, cache=True, fastmath=True)
def decode_bitstring_numba(bitstring_int: np.uint64, L: np.uint64, bits_per_pos: np.uint64, 
                           n_aa: np.uint64) -> np.ndarray:
    """Decodificación ultra-rápida."""
    sequence_codes = np.zeros(L, dtype=np.int32)
    
    for i in range(L):
        shift = (L - 1 - i) * bits_per_pos
        mask = (1 << bits_per_pos) - 1
        pos_code = (bitstring_int >> shift) & mask
        sequence_codes[i] = pos_code if pos_code < n_aa else -1
    
    return sequence_codes

@jit(nopython=True, cache=True, fastmath=True)
def decode_idx_to_bitstring_int(idx: np.uint64, n_aa: np.uint64, L: np.uint64, bits_per_pos: np.uint64) -> np.uint64:
    """Convierte índice a bitstring integer."""
    bitstring_int = np.uint64(0)
    temp_idx = np.uint64(idx)
    
    for pos in range(L):
        aa_code = temp_idx % n_aa
        temp_idx //= n_aa
        shift = pos * bits_per_pos
        bitstring_int |= (aa_code << shift)
    
    return bitstring_int

@jit(nopython=True, cache=True, fastmath=True)
def bitstring_int_to_array(bitstring_int: np.uint64, n_bits: np.uint64) -> np.ndarray:
    """Convierte entero a array de bits."""
    bitstring_array = np.zeros(n_bits, dtype=np.float64)
    for bit_pos in range(n_bits):
        bitstring_array[bit_pos] = float((bitstring_int >> bit_pos) & 1)
    return bitstring_array

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def brute_force_search_numba(n_aa: np.uint64, L: np.uint64, bits_per_pos: np.uint64,
                              coeffs: np.ndarray, pauli_indices: np.ndarray,
                              pauli_types: np.ndarray) -> Tuple[np.uint64, np.float64]:
    """Búsqueda exhaustiva paralelizada con Numba."""
    total = n_aa ** L
    n_bits = L * bits_per_pos
    
    energies = np.empty(total, dtype=np.float64)
    
    for idx in prange(total):
        bitstring_int = decode_idx_to_bitstring_int(np.uint64(idx), n_aa, L, bits_per_pos)
        bitstring_array = bitstring_int_to_array(bitstring_int, n_bits)
        energies[idx] = compute_energy_numba(bitstring_array, coeffs, pauli_indices, pauli_types)
    
    best_idx = np.uint64(np.argmin(energies))
    best_energy = energies[best_idx]
    
    return best_idx, best_energy

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def search_block_numba(start_idx: np.uint64, block_size: np.uint64, n_aa: np.uint64, L: np.uint64, 
                       bits_per_pos: np.uint64, coeffs: np.ndarray, 
                       pauli_indices: np.ndarray, pauli_types: np.ndarray) -> Tuple[np.uint64, np.float64]:
    """Busca en un bloque específico usando Numba."""
    n_bits = L * bits_per_pos
    energies = np.empty(block_size, dtype=np.float64)
    
    for i in prange(block_size):
        idx = start_idx + i
        bitstring_int = decode_idx_to_bitstring_int(np.uint64(idx), n_aa, L, bits_per_pos)
        bitstring_array = bitstring_int_to_array(bitstring_int, n_bits)
        energies[i] = compute_energy_numba(
            bitstring_array, coeffs, pauli_indices, pauli_types
        )
    
    local_best_idx = np.uint64(np.argmin(energies))
    return start_idx + local_best_idx, energies[local_best_idx]

# ==================== QAOA SOLVER ====================

class QAOASolver:
    """
    QAOA solver ULTRA-OPTIMIZADO con soporte para PennyLane y Qiskit.
    """
    def __init__(self, cost_hamiltonian, n_qubits, pauli_terms, amino_acids, 
                 L, bits_per_pos, layers=1, warm_start=False, shots: int = 1000, backend='pennylane'):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_qubits = n_qubits
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
        self.L = L
        self.bits_per_pos = bits_per_pos
        self.n_aa = len(amino_acids)
        self.layers = layers
        self.warm_start = warm_start
        self.shots = shots
        self.backend = backend
        
        if self.backend == 'pennylane':
            try:
                self.dev = qml.device('lightning.gpu', wires=self.n_qubits, shots=shots)
            except:
                self.dev = qml.device('lightning.qubit', wires=self.n_qubits, shots=shots)
        elif self.backend == 'qiskit' and QISKIT_AVAILABLE:
            # Try GPU first, fallback to CPU
            try:
                self.dev = AerSimulator(method='statevector', device='GPU')
                print("✅ Using Qiskit GPU simulator")
            except Exception as e:
                print(f"⚠️  GPU not available, falling back to CPU: {e}")
                try:
                    self.dev = AerSimulator(method='statevector', device='CPU')
                    print("✅ Using Qiskit CPU simulator")
                except Exception as e2:
                    print(f"⚠️  CPU simulator failed, using default: {e2}")
                    self.dev = AerSimulator(method='statevector')
        else:
            raise ValueError("Qiskit backend requested but Qiskit is not available or invalid backend specified.")
        
        self._precompute_structures()
        
    def _precompute_structures(self):
        """Pre-calcula todas las estructuras necesarias para optimización."""
        self._prepare_numba_structures()
        self.position_masks = []
        self.position_shifts = []
        for i in range(self.L):
            shift = (self.L - 1 - i) * self.bits_per_pos
            mask = ((1 << self.bits_per_pos) - 1) << shift
            self.position_masks.append(mask)
            self.position_shifts.append(shift)
        self.amino_acids_array = np.array(list(self.amino_acids))

    def _prepare_numba_structures(self):
        """Prepara estructuras compatibles con Numba."""
        coeffs = []
        pauli_data = []
        
        for coeff, pauli_string in self.pauli_terms:
            coeffs.append(coeff)
            z_indices = [i for i, p in enumerate(pauli_string) if p == 'Z']
            pauli_data.extend(z_indices)
        
        self.numba_coeffs = np.array(coeffs, dtype=np.float64)
        self.numba_pauli_types = np.array(pauli_data, dtype=np.int32)
        
        pauli_indices = [0]
        current_idx = 0
        for _, pauli_string in self.pauli_terms:
            z_count = sum(1 for p in pauli_string if p == 'Z')
            current_idx += z_count
            pauli_indices.append(current_idx)
        
        self.numba_pauli_indices = np.array(pauli_indices, dtype=np.int32)

    def _qaoa_circuit_pennylane(self, params):
        """Circuito QAOA optimizado con templates eficientes para PennyLane."""
        gammas = params[:self.layers]
        betas = params[self.layers:]
        
        qml.broadcast(qml.Hadamard, wires=range(self.n_qubits), pattern='single')
        
        for p in range(self.layers):
            qml.templates.ApproxTimeEvolution(self.cost_hamiltonian, gammas[p], n=1)
            qml.broadcast(qml.RX, wires=range(self.n_qubits), 
                         pattern='single', parameters=2 * betas[p])

    @lru_cache(maxsize=128)
    def _cost_function_cached_pennylane(self, params_tuple):
        """Función de costo con caching para parámetros repetidos en PennyLane."""
        params = qnp.array(params_tuple)
        
        @qml.qnode(self.dev, diff_method="adjoint", interface="autograd")
        def circuit():
            self._qaoa_circuit_pennylane(params)
            return qml.expval(self.cost_hamiltonian)
        
        return circuit()

    def _cost_function_pennylane(self, params):
        """Wrapper para función de costo cacheada en PennyLane."""
        return self._cost_function_cached_pennylane(tuple(params.tolist()))

    def _initialize_params(self):
        """Inicialización inteligente basada en teoría QAOA."""
        if self.warm_start:
            if self.layers == 1:
                gammas = qnp.array([0.5])
                betas = qnp.array([0.39])
            else:
                gammas = qnp.linspace(0.1, 0.5, self.layers)
                betas = qnp.linspace(0.39, 0.2, self.layers)
        else:
            gammas = qnp.random.uniform(0, 0.5 * np.pi, self.layers)
            betas = qnp.random.uniform(0, 0.5 * np.pi, self.layers)
        
        return qnp.concatenate([gammas, betas])

    def _convert_to_qiskit_hamiltonian(self):
        """Convierte el hamiltoniano de PennyLane a formato Qiskit."""
        if not QISKIT_AVAILABLE:
            raise ValueError("Qiskit is not available.")
        
        pauli_list = []
        for coeff, pauli_string in self.pauli_terms:
            pauli_str = ''.join('I' if p == 'I' else 'Z' for p in pauli_string)
            pauli_list.append((pauli_str, coeff))
        return SparsePauliOp.from_list(pauli_list)

    def solve(self, steps=100, learning_rate=0.1, convergence_threshold=1e-6) -> Dict[str, Any]:
        """
        Optimización ultra-rápida con early stopping adaptativo.
        """
        print(f"\n🚀 Solving with ULTRA-OPTIMIZED QAOA ({self.backend})...")
        print(f"⚙️  Device: {self.dev.__class__.__name__ if self.backend == 'qiskit' else self.dev.name} | Layers: {self.layers} | Shots: {self.shots}")
        
        if self.backend == 'pennylane':
            return self._solve_pennylane(steps, learning_rate, convergence_threshold)
        else:
            return self._solve_qiskit(steps)

    def _solve_pennylane(self, steps, learning_rate, convergence_threshold):
        """Solución QAOA usando PennyLane."""
        params = self._initialize_params()
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)

        min_energy = float('inf')
        best_params = params.copy()
        costs = []
        
        no_improvement_count = 0
        max_no_improvement = min(20, steps // 5)
        patience_threshold = convergence_threshold * 0.1

        for i in range(steps):
            params, cost = optimizer.step_and_cost(self._cost_function_pennylane, params)
            costs.append(cost)
            
            improvement = min_energy - cost
            if improvement > patience_threshold:
                min_energy = cost
                best_params = params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if i % 10 == 0 or i == steps - 1:
                print(f"  Step {i:3d}: Energy = {cost:.6f} | Best = {min_energy:.6f}")
            
            if no_improvement_count >= max_no_improvement:
                if i > 30:
                    print(f"✓ Converged at step {i} (no improvement for {no_improvement_count} steps)")
                    break

        print("📊 Sampling probability distribution...")
        
        @qml.qnode(self.dev, interface="autograd")
        def prob_circuit():
            self._qaoa_circuit_pennylane(best_params)
            return qml.probs(wires=range(self.n_qubits))

        probs = prob_circuit()
        
        if len(probs) == 0:
            print("⚠️  Warning: No probabilities computed.")
            return self._empty_result(costs, best_params)

        print("🔧 Repairing solution with marginal distributions...")
        repaired_bitstring = self._repair_with_marginals_fast(probs)
        best_sequence = self._decode_solution_ultra_fast(repaired_bitstring)
        best_energy = self._compute_energy_ultra_fast(repaired_bitstring)

        print(f"\n✅ QAOA optimization completed!")
        print(f"➡️  Best sequence: {best_sequence}")
        print(f"➡️  Energy: {best_energy:.6f}")
        print(f"➡️  Optimization steps: {len(costs)}")

        try:
            self.plot_prob_with_sequences(probs)
        except Exception as e:
            print(f"⚠️  Plot skipped: {e}")

        return {
            'bitstring': repaired_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': costs,
            'params': best_params,
            'probs': probs
        }

    def _solve_qiskit(self, steps):
        """Solución QAOA usando Qiskit con Estimator y Sampler."""
        if not QISKIT_AVAILABLE:
            raise ValueError("Qiskit is not available.")
        
        hamiltonian = self._convert_to_qiskit_hamiltonian()
        optimizer = COBYLA(maxiter=steps)
        estimator = AerEstimator() if '_HAS_AER_PRIMITIVES' in globals() and _HAS_AER_PRIMITIVES else Estimator()
        sampler = AerSampler() if '_HAS_AER_PRIMITIVES' in globals() and _HAS_AER_PRIMITIVES else Sampler()
        
        # Use standard Qiskit QAOA ansatz (built-in) with provided reps
        qaoa = QAOA(estimator=estimator, optimizer=optimizer, reps=self.layers)
        result = qaoa.compute_minimum_eigenvalue(operator=hamiltonian)
        best_params = result.optimal_parameters
        best_energy = result.eigenvalue.real
        
        # Sample the final state
        final_circuit = qaoa.ansatz.assign_parameters(best_params)
        final_circuit.measure_all()
        job = sampler.run(final_circuit, shots=self.shots)
        counts = job.result().quasi_dists[0]
        
        probs = np.zeros(2**self.n_qubits)
        for key, prob in counts.items():
            if isinstance(key, int):
                idx = key
            else:
                idx = int(key, 2)
            if 0 <= idx < len(probs):
                probs[idx] = float(prob)
        
        if len(probs) == 0 or np.sum(probs) < 1e-10:
            print("⚠️  Warning: No valid probabilities computed.")
            return self._empty_result([], best_params)

        print("🔧 Repairing solution with marginal distributions...")
        repaired_bitstring = self._repair_with_marginals_fast(probs)
        best_sequence = self._decode_solution_ultra_fast(repaired_bitstring)
        best_energy = self._compute_energy_ultra_fast(repaired_bitstring)

        print(f"\n✅ QAOA optimization completed!")
        print(f"➡️  Best sequence: {best_sequence}")
        print(f"➡️  Energy: {best_energy:.6f}")
        print(f"➡️  Optimization steps: {steps}")

        try:
            self.plot_prob_with_sequences(probs)
        except Exception as e:
            print(f"⚠️  Plot skipped: {e}")

        return {
            'bitstring': repaired_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': [],
            'params': best_params,
            'probs': probs
        }

    def _repair_with_marginals_fast(self, probs: np.ndarray) -> str:
        """Reparación ultra-rápida con Numba."""
        marginals = get_marginals_numba(probs, self.L, self.bits_per_pos, self.n_aa)
        repaired_sequence_code = np.argmax(marginals, axis=1)
        return ''.join(format(int(c), f'0{self.bits_per_pos}b') for c in repaired_sequence_code)

    def _decode_solution_ultra_fast(self, bitstring: str) -> str:
        """Decodificación ultra-rápida usando Numba."""
        bitstring_int = int(bitstring, 2)
        sequence_codes = decode_bitstring_numba(
            bitstring_int, self.L, self.bits_per_pos, self.n_aa
        )
        return ''.join(
            self.amino_acids_array[c] if c >= 0 else 'X'
            for c in sequence_codes
        )

    def _compute_energy_ultra_fast(self, bitstring: str) -> float:
        """Cálculo de energía ultra-rápido con Numba."""
        bitstring_array = np.array([int(b) for b in bitstring], dtype=np.float64)
        return compute_energy_numba(
            bitstring_array, 
            self.numba_coeffs,
            self.numba_pauli_indices,
            self.numba_pauli_types
        )

    def _empty_result(self, costs, params):
        """Resultado vacío en caso de error."""
        return {
            'bitstring': '',
            'sequence': '',
            'energy': float('inf'),
            'costs': costs,
            'params': params,
            'probs': np.array([])
        }

    def decode_solution(self, bitstring: str) -> str:
        return self._decode_solution_ultra_fast(bitstring)
    
    def decode_solution_fast(self, bitstring: str) -> str:
        return self._decode_solution_ultra_fast(bitstring)
    
    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        return self._compute_energy_ultra_fast(bitstring)
    
    def compute_energy_from_bitstring_fast(self, bitstring: str) -> float:
        return self._compute_energy_ultra_fast(bitstring)
    
    def _repair_with_marginals(self, probs: np.ndarray) -> str:
        return self._repair_with_marginals_fast(probs)
    
    def _get_marginals(self, probs: np.ndarray) -> np.ndarray:
        return get_marginals_numba(probs, self.L, self.bits_per_pos, self.n_aa)
    
    def _get_marginals_vectorized(self, probs: np.ndarray) -> np.ndarray:
        return get_marginals_numba(probs, self.L, self.bits_per_pos, self.n_aa)

    def plot_prob_with_sequences(self, probs: np.ndarray, top_k: int = 20):
        """Plot optimizado con procesamiento vectorizado."""
        top_k = min(top_k, len(probs))
        
        if top_k < len(probs):
            sorted_indices = np.argpartition(probs, -top_k)[-top_k:]
            sorted_indices = sorted_indices[np.argsort(-probs[sorted_indices])]
        else:
            sorted_indices = np.argsort(-probs)
        
        sorted_probs = probs[sorted_indices]
        sequences = [
            self._decode_solution_ultra_fast(format(idx, f'0{self.n_qubits}b'))
            for idx in sorted_indices
        ]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(sequences)), sorted_probs, 
                       color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
        
        bars[0].set_color('gold')
        bars[0].set_edgecolor('darkorange')
        bars[0].set_linewidth(2)
        
        plt.xlabel('Amino Acid Sequences (Top ' + str(top_k) + ')', fontsize=11, fontweight='bold')
        plt.ylabel('Probability', fontsize=11, fontweight='bold')
        plt.title('Top Probability Distribution from QAOA', fontsize=13, fontweight='bold', pad=15)
        plt.xticks(range(len(sequences)), sequences, rotation=90, fontsize=9)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig('qaoa_probability_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Plot saved as qaoa_probability_plot.png")
    
    def get_top_sequences(self, probs: np.ndarray, top_k: int = 10) -> List[Tuple[str, str, float, float]]:
        """Devuelve las top_k secuencias con sus probabilidades y energías."""
        top_k = min(top_k, len(probs))
        
        if top_k < len(probs):
            sorted_indices = np.argpartition(probs, -top_k)[-top_k:]
            sorted_indices = sorted_indices[np.argsort(-probs[sorted_indices])]
        else:
            sorted_indices = np.argsort(-probs)
        
        results = []
        for idx in sorted_indices:
            bitstring = format(idx, f'0{self.n_qubits}b')
            sequence = self._decode_solution_ultra_fast(bitstring)
            probability = probs[idx]
            energy = self._compute_energy_ultra_fast(bitstring)
            results.append((bitstring, sequence, probability, energy))
        
        return results

# ==================== VQE SOLVER ====================

class VQESolver:
    """
    VQE solver ULTRA-OPTIMIZADO con soporte para PennyLane y Qiskit.
    """
    def __init__(self, cost_hamiltonian, n_qubits, pauli_terms, amino_acids, 
                 L, bits_per_pos, layers=2, shots: int = 1000, backend='pennylane'):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_qubits = n_qubits
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
        self.L = L
        self.bits_per_pos = bits_per_pos
        self.n_aa = len(amino_acids)
        self.layers = layers
        self.shots = shots
        self.backend = backend
        
        if self.backend == 'pennylane':
            try:
                self.dev = qml.device('lightning.gpu', wires=self.n_qubits, shots=shots)
            except:
                self.dev = qml.device('lightning.qubit', wires=self.n_qubits, shots=shots)
        elif self.backend == 'qiskit' and QISKIT_AVAILABLE:
            # Try GPU first, fallback to CPU
            try:
                self.dev = AerSimulator(method='statevector', device='GPU')
                print("✅ Using Qiskit GPU simulator")
            except Exception as e:
                print(f"⚠️  GPU not available, falling back to CPU: {e}")
                try:
                    self.dev = AerSimulator(method='statevector', device='CPU')
                    print("✅ Using Qiskit CPU simulator")
                except Exception as e2:
                    print(f"⚠️  CPU simulator failed, using default: {e2}")
                    self.dev = AerSimulator(method='statevector')
        else:
            raise ValueError("Qiskit backend requested but Qiskit is not available or invalid backend specified.")
        
        self._precompute_structures()

    def _precompute_structures(self):
        """Pre-calcula todas las estructuras necesarias."""
        self._prepare_numba_structures()
        self.amino_acids_array = np.array(list(self.amino_acids))
        self.decode_shifts = np.array([
            (self.L - 1 - i) * self.bits_per_pos for i in range(self.L)
        ])
        self.decode_mask = (1 << self.bits_per_pos) - 1

    def _prepare_numba_structures(self):
        """Prepara estructuras compatibles con Numba."""
        coeffs = []
        pauli_data = []
        
        for coeff, pauli_string in self.pauli_terms:
            coeffs.append(coeff)
            z_indices = [i for i, p in enumerate(pauli_string) if p == 'Z']
            pauli_data.extend(z_indices)
        
        self.numba_coeffs = np.array(coeffs, dtype=np.float64)
        self.numba_pauli_types = np.array(pauli_data, dtype=np.int32)
        
        pauli_indices = [0]
        current_idx = 0
        for _, pauli_string in self.pauli_terms:
            z_count = sum(1 for p in pauli_string if p == 'Z')
            current_idx += z_count
            pauli_indices.append(current_idx)
        
        self.numba_pauli_indices = np.array(pauli_indices, dtype=np.int32)

    def _vqe_ansatz_pennylane(self, params):
        """Ansatz VQE optimizado con StronglyEntanglingLayers para PennyLane."""
        qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_qubits))

    @lru_cache(maxsize=256)
    def _cost_function_cached_pennylane(self, params_tuple):
        """Función de costo cacheada para reutilizar cálculos en PennyLane."""
        params = qnp.array(params_tuple).reshape(self.layers, self.n_qubits, 3)
        
        @qml.qnode(self.dev, diff_method="adjoint", interface="autograd")
        def circuit():
            self._vqe_ansatz_pennylane(params)
            return qml.expval(self.cost_hamiltonian)
        
        return circuit()

    def _cost_function_pennylane(self, params):
        """Wrapper para función de costo cacheada en PennyLane."""
        params_flat = tuple(params.flatten().tolist())
        return self._cost_function_cached_pennylane(params_flat)

    def _initialize_params(self):
        """Inicialización inteligente Xavier/Glorot para VQE."""
        shape = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=self.layers, 
            n_wires=self.n_qubits
        )
        limit = np.sqrt(6.0 / (self.n_qubits + self.layers))
        params = qnp.random.uniform(-limit, limit, shape)
        return params

    def _convert_to_qiskit_hamiltonian(self):
        """Convierte el hamiltoniano de PennyLane a formato Qiskit."""
        if not QISKIT_AVAILABLE:
            raise ValueError("Qiskit is not available.")
        
        pauli_list = []
        for coeff, pauli_string in self.pauli_terms:
            pauli_str = ''.join('I' if p == 'I' else 'Z' for p in pauli_string)
            pauli_list.append((pauli_str, coeff))
        return SparsePauliOp.from_list(pauli_list)

    def solve(self, steps=200, learning_rate=0.1, convergence_threshold=1e-6) -> Dict[str, Any]:
        """
        Optimización VQE ultra-rápida con early stopping adaptativo.
        """
        print(f"\n🔬 Solving with ULTRA-OPTIMIZED VQE ({self.backend})...")
        print(f"⚙️  Device: {self.dev.__class__.__name__ if self.backend == 'qiskit' else self.dev.name} | Layers: {self.layers} | Shots: {self.shots}")
        
        if self.backend == 'pennylane':
            return self._solve_pennylane(steps, learning_rate, convergence_threshold)
        else:
            return self._solve_qiskit(steps)

    def _solve_pennylane(self, steps, learning_rate, convergence_threshold):
        """Solución VQE usando PennyLane."""
        params = self._initialize_params()
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)

        min_energy = float('inf')
        best_params = params.copy()
        costs = []
        
        no_improvement_count = 0
        max_no_improvement = min(30, steps // 5)
        patience_threshold = convergence_threshold * 0.1

        for i in range(steps):
            params, cost = optimizer.step_and_cost(self._cost_function_pennylane, params)
            costs.append(cost)
            
            improvement = min_energy - cost
            if improvement > patience_threshold:
                min_energy = cost
                best_params = params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if i % 20 == 0 or i == steps - 1:
                print(f"  Step {i:3d}: Energy = {cost:.6f} | Best = {min_energy:.6f}")
            
            if no_improvement_count >= max_no_improvement:
                if i > 40:
                    print(f"✓ Converged at step {i} (no improvement for {no_improvement_count} steps)")
                    break

        print("📊 Sampling probability distribution...")
        
        @qml.qnode(self.dev, interface="autograd")
        def prob_circuit():
            self._vqe_ansatz_pennylane(best_params)
            return qml.probs(wires=range(self.n_qubits))

        probs = prob_circuit()
        
        if len(probs) == 0 or np.sum(probs) < 1e-10:
            print("⚠️  Warning: No valid probabilities computed.")
            return self._empty_result(costs, best_params)

        print("🔧 Repairing solution with marginal distributions...")
        repaired_bitstring = self._repair_with_marginals_fast(probs)
        best_sequence = self._decode_solution_ultra_fast(repaired_bitstring)
        best_energy = self._compute_energy_ultra_fast(repaired_bitstring)

        print(f"\n✅ VQE optimization completed!")
        print(f"➡️  Best sequence: {best_sequence}")
        print(f"➡️  Energy: {best_energy:.6f}")
        print(f"➡️  Optimization steps: {len(costs)}")

        try:
            self.plot_prob_with_sequences(probs)
        except Exception as e:
            print(f"⚠️  Plot skipped: {e}")

        return {
            'bitstring': repaired_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': costs,
            'params': best_params,
            'probs': probs
        }

    def _solve_qiskit(self, steps):
        """Solución VQE usando Qiskit con Estimator y Sampler."""
        if not QISKIT_AVAILABLE:
            raise ValueError("Qiskit is not available.")
        
        hamiltonian = self._convert_to_qiskit_hamiltonian()
        optimizer = COBYLA(maxiter=steps)
        estimator = AerEstimator() if '_HAS_AER_PRIMITIVES' in globals() and _HAS_AER_PRIMITIVES else Estimator()
        sampler = AerSampler() if '_HAS_AER_PRIMITIVES' in globals() and _HAS_AER_PRIMITIVES else Sampler()
        
        # Prefer EfficientSU2, fallback to RealAmplitudes, last-resort simple circuit
        if EfficientSU2 is not None:
            ansatz = EfficientSU2(self.n_qubits, reps=self.layers, entanglement='linear')
        elif RealAmplitudes is not None:
            ansatz = RealAmplitudes(self.n_qubits, reps=self.layers, entanglement='linear')
        else:
            ansatz = QuantumCircuit(self.n_qubits)
            for i in range(self.n_qubits):
                ansatz.h(i)
            for _ in range(self.layers):
                for i in range(self.n_qubits - 1):
                    ansatz.cx(i, i + 1)
                for i in range(self.n_qubits):
                    ansatz.ry(0.1, i)
        
        # Initialize VQE with Estimator
        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
        result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        best_params = result.optimal_parameters
        best_energy = result.eigenvalue.real
        
        # Sample the final state
        final_circuit = ansatz.assign_parameters(best_params)
        final_circuit.measure_all()
        job = sampler.run(final_circuit, shots=self.shots)
        counts = job.result().quasi_dists[0]
        
        # Convert counts to probability array
        probs = np.zeros(2**self.n_qubits)
        for key, prob in counts.items():
            if isinstance(key, int):
                idx = key
            else:
                idx = int(key, 2)
            if 0 <= idx < len(probs):
                probs[idx] = float(prob)
        
        # Check for valid probabilities
        if len(probs) == 0 or np.sum(probs) < 1e-10:
            print("⚠️  Warning: No valid probabilities computed.")
            return self._empty_result([], best_params)

        print("🔧 Repairing solution with marginal distributions...")
        repaired_bitstring = self._repair_with_marginals_fast(probs)
        best_sequence = self._decode_solution_ultra_fast(repaired_bitstring)
        best_energy = self._compute_energy_ultra_fast(repaired_bitstring)

        print(f"\n✅ VQE optimization completed!")
        print(f"➡️  Best sequence: {best_sequence}")
        print(f"➡️  Energy: {best_energy:.6f}")
        print(f"➡️  Optimization steps: {steps}")

        try:
            self.plot_prob_with_sequences(probs)
        except Exception as e:
            print(f"⚠️  Plot skipped: {e}")

        return {
            'bitstring': repaired_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': [],
            'params': best_params,
            'probs': probs
        }

    def _repair_with_marginals_fast(self, probs: np.ndarray) -> str:
        """Reparación ultra-rápida con Numba."""
        marginals = get_marginals_numba(probs, self.L, self.bits_per_pos, self.n_aa)
        repaired_sequence_code = np.argmax(marginals, axis=1)
        return ''.join(format(int(c), f'0{self.bits_per_pos}b') for c in repaired_sequence_code)

    def _decode_solution_ultra_fast(self, bitstring: str) -> str:
        """Decodificación ultra-rápida."""
        bitstring_int = int(bitstring, 2)
        sequence_codes = decode_bitstring_numba(
            bitstring_int, self.L, self.bits_per_pos, self.n_aa
        )
        return ''.join(
            self.amino_acids_array[c] if c >= 0 else 'X'
            for c in sequence_codes
        )

    def _compute_energy_ultra_fast(self, bitstring: str) -> float:
        """Cálculo de energía ultra-rápido."""
        bitstring_array = np.array([int(b) for b in bitstring], dtype=np.float64)
        return compute_energy_numba(
            bitstring_array, self.numba_coeffs,
            self.numba_pauli_indices,
            self.numba_pauli_types
        )

    def _empty_result(self, costs, params):
        """Resultado vacío."""
        return {
            'bitstring': '', 'sequence': '', 'energy': float('inf'),
            'costs': costs, 'params': params, 'probs': np.array([])
        }

    def decode_solution(self, bitstring: str) -> str:
        return self._decode_solution_ultra_fast(bitstring)
    
    def decode_solution_fast(self, bitstring: str) -> str:
        return self._decode_solution_ultra_fast(bitstring)
    
    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        return self._compute_energy_ultra_fast(bitstring)
    
    def compute_energy_from_bitstring_fast(self, bitstring: str) -> float:
        return self._compute_energy_ultra_fast(bitstring)
    
    def _repair_with_marginals(self, probs: np.ndarray) -> str:
        return self._repair_with_marginals_fast(probs)
    
    def _get_marginals(self, probs: np.ndarray) -> np.ndarray:
        return get_marginals_numba(probs, self.L, self.bits_per_pos, self.n_aa)
    
    def _get_marginals_vectorized(self, probs: np.ndarray) -> np.ndarray:
        return get_marginals_numba(probs, self.L, self.bits_per_pos, self.n_aa)
    
    def _repair_with_marginals_vectorized(self, probs: np.ndarray) -> str:
        return self._repair_with_marginals_fast(probs)

    def get_top_sequences(self, probs: np.ndarray, top_k: int = 10) -> List[Tuple[str, str, float, float]]:
        """Devuelve las top_k secuencias."""
        top_k = min(top_k, len(probs))
        
        if top_k < len(probs):
            sorted_indices = np.argpartition(probs, -top_k)[-top_k:]
            sorted_indices = sorted_indices[np.argsort(-probs[sorted_indices])]
        else:
            sorted_indices = np.argsort(-probs)
        
        results = []
        for idx in sorted_indices:
            bitstring = format(idx, f'0{self.n_qubits}b')
            sequence = self._decode_solution_ultra_fast(bitstring)
            probability = probs[idx]
            energy = self._compute_energy_ultra_fast(bitstring)
            results.append((bitstring, sequence, probability, energy))
        
        return results

    def plot_prob_with_sequences(self, probs: np.ndarray, top_k: int = 20):
        """Plot optimizado."""
        top_k = min(top_k, len(probs))
        
        if top_k < len(probs):
            sorted_indices = np.argpartition(probs, -top_k)[-top_k:]
            sorted_indices = sorted_indices[np.argsort(-probs[sorted_indices])]
        else:
            sorted_indices = np.argsort(-probs)
        
        sorted_probs = probs[sorted_indices]
        sequences = [
            self._decode_solution_ultra_fast(format(idx, f'0{self.n_qubits}b'))
            for idx in sorted_indices
        ]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(sequences)), sorted_probs, 
                       color='mediumseagreen', alpha=0.8, 
                       edgecolor='darkgreen', linewidth=0.5)
        
        if len(bars) >= 3:
            bars[0].set_color('forestgreen')
            bars[0].set_linewidth(2)
            bars[1].set_color('seagreen')
            bars[2].set_color('mediumseagreen')
        
        plt.xlabel('Amino Acid Sequences (Top ' + str(top_k) + ')', 
                   fontsize=11, fontweight='bold')
        plt.ylabel('Probability', fontsize=11, fontweight='bold')
        plt.title('Top Probability Distribution from VQE', 
                  fontsize=13, fontweight='bold', pad=15)
        plt.xticks(range(len(sequences)), sequences, rotation=90, fontsize=9)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig('vqe_probability_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Plot saved as vqe_probability_plot.png")

# ==================== CLASSICAL SOLVER ====================

class ClassicalSolver:
    """
    Solver clásico ULTRA-OPTIMIZADO con búsqueda paralela Numba y gestión eficiente de memoria.
    """
    def __init__(self, L: int, n_aa: int, bits_per_pos: int, 
                 pauli_terms: List, amino_acids: List[str]):
        self.L = L
        self.n_aa = n_aa
        self.bits_per_pos = bits_per_pos
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
        self.amino_acids_array = np.array(list(amino_acids))
        
        self._prepare_numba_structures()
    
    def _prepare_numba_structures(self):
        """Prepara estructuras para Numba."""
        coeffs = []
        pauli_data = []
        
        for coeff, pauli_string in self.pauli_terms:
            coeffs.append(coeff)
            z_indices = [i for i, p in enumerate(pauli_string) if p == 'Z']
            pauli_data.extend(z_indices)
        
        self.numba_coeffs = np.array(coeffs, dtype=np.float64)
        self.numba_pauli_types = np.array(pauli_data, dtype=np.int32)
        
        pauli_indices = [0]
        current_idx = 0
        for _, pauli_string in self.pauli_terms:
            z_count = sum(1 for p in pauli_string if p == 'Z')
            current_idx += z_count
            pauli_indices.append(current_idx)
        
        self.numba_pauli_indices = np.array(pauli_indices, dtype=np.int32)
    
    def decode_solution(self, bitstring: str) -> str:
        """Decodificación rápida."""
        bitstring_int = int(bitstring, 2)
        sequence_codes = decode_bitstring_numba(
            bitstring_int, self.L, self.bits_per_pos, self.n_aa
        )
        return ''.join(
            self.amino_acids_array[c] if c >= 0 else 'X'
            for c in sequence_codes
        )
    
    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        """Cálculo de energía ultra-rápido."""
        bitstring_array = np.array([int(b) for b in bitstring], dtype=np.float64)
        return compute_energy_numba(
            bitstring_array, self.numba_coeffs,
            self.numba_pauli_indices,
            self.numba_pauli_types
        )
    
    def _idx_to_bitstring(self, idx: int) -> str:
        """Convierte índice a bitstring eficientemente."""
        sequence_codes = []
        temp_idx = idx
        
        for _ in range(self.L):
            aa_code = temp_idx % self.n_aa
            sequence_codes.append(aa_code)
            temp_idx //= self.n_aa
        
        sequence_codes.reverse()
        return ''.join(format(c, f'0{self.bits_per_pos}b') for c in sequence_codes)
    
    def solve(self, use_numba: bool = True, max_memory_mb: int = 1000) -> Dict[str, Any]:
        """
        Búsqueda exhaustiva ultra-optimizada.
        """
        print("\n🏆 Solving with ULTRA-OPTIMIZED Classical Brute-Force...")
        
        total_combinations = self.n_aa ** self.L
        print(f"🔍 Evaluating {total_combinations:,} possible sequences...")
        
        estimated_memory_mb = (total_combinations * 8) / (1024 * 1024)
        
        if use_numba and total_combinations > 1000:
            if estimated_memory_mb > max_memory_mb:
                print(f"⚠️  Large problem ({estimated_memory_mb:.1f} MB > {max_memory_mb} MB)")
                print("⚡ Using block-based parallelized search...")
                best_idx, best_energy = self._block_search_numba(total_combinations, max_memory_mb)
            else:
                print(f"⚡ Using full parallelized Numba search ({estimated_memory_mb:.1f} MB)...")
                best_idx, best_energy = brute_force_search_numba(
                    np.uint64(self.n_aa), np.uint64(self.L), np.uint64(self.bits_per_pos),
                    self.numba_coeffs, self.numba_pauli_indices, self.numba_pauli_types
                )
            
            best_bitstring = self._idx_to_bitstring(best_idx)
            print("🔎 Finding top-5 sequences...")
            top_sequences = self._find_top_k_sequential(5, total_combinations)
            
        else:
            print("🔄 Using sequential search (small problem)...")
            best_bitstring, best_energy, top_sequences = self._sequential_search(total_combinations)
        
        best_sequence = self.decode_solution(best_bitstring)
        
        print("\n🏅 Top 5 sequences by energy:")
        for rank, (seq, energy) in enumerate(top_sequences, 1):
            print(f"  #{rank}: {seq} | Energy: {energy:.6f}")
        
        print(f"\n✅ Classical brute-force completed!")
        print(f"➡️  Best sequence: {best_sequence} | Energy: {best_energy:.6f}")
        
        return {
            'bitstring': best_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': [],
            'top_5': top_sequences
        }
    
    def _block_search_numba(self, total: int, max_memory_mb: int) -> Tuple[np.uint64, float]:
        """Búsqueda por bloques para problemas muy grandes."""
        bytes_per_element = 8
        max_elements = int((max_memory_mb * 1024 * 1024) / bytes_per_element)
        block_size = min(max_elements, total)
        
        best_energy = float('inf')
        best_idx = np.uint64(0)
        
        num_blocks = (total + block_size - 1) // block_size
        print(f"  Processing {num_blocks} blocks of ~{block_size:,} elements each...")
        
        for block_num in range(num_blocks):
            start_idx = block_num * block_size
            end_idx = min(start_idx + block_size, total)
            block_total = end_idx - start_idx
            
            block_best_idx, block_best_energy = search_block_numba(
                np.uint64(start_idx), np.uint64(block_total), np.uint64(self.n_aa), np.uint64(self.L), np.uint64(self.bits_per_pos),
                self.numba_coeffs, self.numba_pauli_indices, self.numba_pauli_types
            )
            
            if block_best_energy < best_energy:
                best_energy = block_best_energy
                best_idx = block_best_idx
            
            if (block_num + 1) % max(1, num_blocks // 10) == 0:
                progress = ((block_num + 1) / num_blocks) * 100
                print(f"  Progress: {progress:.0f}% | Best energy: {best_energy:.6f}")
        
        return best_idx, best_energy
    
    def _find_top_k_sequential(self, k: int, total: int) -> List[Tuple[str, float]]:
        """Encuentra top-k secuencias usando min-heap eficiente."""
        import heapq
        top_heap = []
        
        sample_size = min(total, max(k * 100, 10000))
        step = max(1, total // sample_size)
        
        for idx in range(0, total, step):
            bitstring = self._idx_to_bitstring(idx)
            energy = self.compute_energy_from_bitstring(bitstring)
            sequence = self.decode_solution(bitstring)
            
            if len(top_heap) < k:
                heapq.heappush(top_heap, (-energy, sequence, energy))
            elif energy < -top_heap[0][0]:
                heapq.heapreplace(top_heap, (-energy, sequence, energy))
        
        return sorted([(seq, e) for _, seq, e in top_heap], key=lambda x: x[1])
    
    def _sequential_search(self, total: int) -> Tuple[str, float, List]:
        """Búsqueda secuencial optimizada para problemas pequeños."""
        import itertools
        import heapq
        best_bitstring = None
        best_energy = float('inf')
        top_heap = []
        
        possible_codes = list(range(self.n_aa))
        all_combinations = itertools.product(possible_codes, repeat=self.L)
        
        for idx, sequence_code in enumerate(all_combinations):
            bitstring = ''.join(format(c, f'0{self.bits_per_pos}b') for c in sequence_code)
            energy = self.compute_energy_from_bitstring(bitstring)
            sequence = self.decode_solution(bitstring)
            
            if len(top_heap) < 5:
                heapq.heappush(top_heap, (-energy, sequence, energy))
            elif energy < -top_heap[0][0]:
                heapq.heapreplace(top_heap, (-energy, sequence, energy))
            
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring
            
            if total > 10000 and (idx + 1) % (total // 10) == 0:
                print(f"  Progress: {(idx + 1) / total * 100:.0f}%")
        
        top_sequences = sorted([(seq, e) for _, seq, e in top_heap], key=lambda x: x[1])
        return best_bitstring, best_energy, top_sequences