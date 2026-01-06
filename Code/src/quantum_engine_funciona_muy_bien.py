import os
import numpy as np
import traceback
from typing import Dict, List, Any

# Quantum & Qiskit
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator, AerError

# Optimizadores
try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA
except ImportError:
    from qiskit.algorithms.optimizers import COBYLA, SPSA

# Fallback para Qiskit Primitives V2
try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator, SamplerV2 as AerSampler
    QISKIT_AVAILABLE = True
except ImportError:
    try:
        from qiskit.primitives import EstimatorV2 as AerEstimator, SamplerV2 as AerSampler
        QISKIT_AVAILABLE = True
    except ImportError:
        QISKIT_AVAILABLE = False

# Importaciones locales
from .utils_logic import decode_solution_logic, mask_invalid_probabilities
from visualization.plot_utils import ProteinPlotter
from core.hamiltonian_builder import HamiltonianBuilder
from core.solvers_qiskit import CustomQAOA, CustomVQE, DifferentialEvolutionOptimizer

class QuantumProteinDesign:
    def __init__(self, sequence_length: int, amino_acids: List[str], **kwargs):
        self.L = sequence_length
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = max(1, int(np.ceil(np.log2(self.n_aa))))
        self.n_qubits = self.L * self.bits_per_pos
        self.backend_name = kwargs.get('quantum_backend', 'pennylane')
        self.shots = kwargs.get('shots', 5000)
        self.use_statevector = kwargs.get('use_statevector', False)
        self.output_dir = kwargs.get('output_dir', 'output')
        
        self.plotter = ProteinPlotter(output_dir=self.output_dir)
        
        self.hamiltonian_builder = HamiltonianBuilder(
            L=self.L, amino_acids=self.amino_acids, 
            bits_per_pos=self.bits_per_pos, n_qubits=self.n_qubits, **kwargs
        )
        self.pauli_terms, self.cost_hamiltonian = self.hamiltonian_builder.build_hamiltonian(self.backend_name)
        self._sanitize_hamiltonian()

    def _sanitize_hamiltonian(self):
        self.pauli_terms = [(float(c.real), p) for c, p in self.pauli_terms]
        if self.backend_name == 'pennylane' and hasattr(self.cost_hamiltonian, 'coeffs'):
            self.cost_hamiltonian = qml.Hamiltonian(
                [float(c.real) for c in self.cost_hamiltonian.coeffs],
                self.cost_hamiltonian.ops
            )

    def decode_solution(self, bitstring: str) -> str:
        if not bitstring: return 'X' * self.L
        return decode_solution_logic(bitstring, self.L, self.bits_per_pos, self.amino_acids)

    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        if not bitstring: return float('inf')
        z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
        energy = 0.0
        for coeff, pauli_string in self.pauli_terms:
            prod = 1.0
            for i, p in enumerate(pauli_string):
                if p == 'Z': prod *= z_vals[i]
            energy += coeff * prod
        return float(energy)

    def compute_energy_breakdown(self, bitstring: str) -> Dict[str, float]:
        if not hasattr(self.hamiltonian_builder, 'terms_by_type'):
            return {'Total': self.compute_energy_from_bitstring(bitstring)}
        z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
        breakdown = {}
        for category, terms in self.hamiltonian_builder.terms_by_type.items():
            cat_energy = 0.0
            for coeff, pauli_string in terms:
                prod = 1.0
                for i, p in enumerate(pauli_string):
                    if p == 'Z': prod *= z_vals[i]
                cat_energy += coeff * prod
            breakdown[category] = cat_energy
        return breakdown

    # --- NUEVA FUNCIÓN: SELECTOR DE BACKEND INTELIGENTE ---
    def _get_optimal_backend(self):
        """Intenta usar GPU. Si falla, cae a CPU automáticamente."""
        sim_method = 'statevector' if self.use_statevector else 'automatic'
        
        try:
            # 1. Intentamos forzar GPU
            backend = AerSimulator(method=sim_method, device='GPU', precision='single')
            
            # Prueba de fuego: Ejecutar un circuito dummy para ver si la GPU responde
            dummy_qc = QuantumCircuit(1)
            dummy_qc.h(0)
            dummy_qc.measure_all()
            backend.run(dummy_qc, shots=1).result()
            
            print(f"✅ Backend Activo: GPU A100 (Method={sim_method})")
            return backend
            
        except Exception as e:
            print(f"⚠️ AVISO: Fallo al inicializar GPU ({e}).")
            print(f"🔄 Cambiando a CPU automáticamente (será más lento).")
            # 2. Fallback a CPU
            return AerSimulator(method=sim_method, device='CPU')

    def _get_probabilities(self, circuit, sampler, backend, force_high_shots=False):
        probs = None
        if self.use_statevector:
            try:
                probs = np.abs(Statevector(circuit).data) ** 2
            except: probs = None
        
        if probs is None:
            mc = circuit.copy()
            if mc.num_clbits == 0:
                mc.measure_all()
            
            base_shots = self.shots if not self.use_statevector else 1000
            final_shots = max(base_shots, 50000) if force_high_shots else base_shots
            
            try:
                res = backend.run(mc, shots=final_shots).result()
                if hasattr(res, 'get_counts'): counts = res.get_counts()
                elif isinstance(res, list) and hasattr(res[0], 'data'): counts = res[0].data.meas.get_counts()
                else: counts = {}
                
                probs = np.zeros(2**self.n_qubits)
                for b, c in counts.items():
                    clean_b = b.replace(" ","")
                    if len(clean_b) > self.n_qubits: clean_b = clean_b[:self.n_qubits]
                    try:
                        idx = int(clean_b, 2)
                        if idx < len(probs): probs[idx] = c / final_shots
                    except: continue
            except Exception:
                return np.zeros(2**self.n_qubits)
        
        return mask_invalid_probabilities(probs, self.L, self.bits_per_pos, self.n_aa, self.n_qubits)

    def _smart_select(self, probs):
        best_bs, min_en = None, float('inf')
        if probs is None or np.sum(probs) == 0:
            best_bs = '0' * self.n_qubits
            return best_bs, self.compute_energy_from_bitstring(best_bs)

        indices = np.argsort(probs)[-200:]
        for idx in indices:
            if probs[idx] <= 0: continue
            bs = format(idx, f'0{self.n_qubits}b')
            en = self.compute_energy_from_bitstring(bs)
            if en < min_en:
                min_en, best_bs = en, bs
        
        if best_bs is None:
            idx = np.argmax(probs)
            best_bs = format(idx, f'0{self.n_qubits}b')
            min_en = self.compute_energy_from_bitstring(best_bs)
        return best_bs, min_en

    def solve_classical_brute_force(self):
        print(f"💻 Ejecutando búsqueda clásica exhaustiva...")
        results = []
        limit = 2**self.n_qubits
        indices = range(limit) if limit < 1000000 else np.random.choice(limit, 1000000, replace=False)
        for idx in indices:
            bs = format(idx, f'0{self.n_qubits}b')
            seq = self.decode_solution(bs)
            if 'X' not in seq:
                en = self.compute_energy_from_bitstring(bs)
                results.append((bs, seq, en, 1.0))
        results.sort(key=lambda x: x[2])
        best_bs, best_seq, best_en, _ = results[0]
        return {'bitstring': best_bs, 'repaired_sequence': best_seq, 'repaired_cost': best_en,
                'energy': best_en, 'classical_ranking': results, 'probs': None}

    def solve_qaoa_qiskit(self, p_layers=4, max_iter=800):
        print(f"🔥 Solving with Qiskit QAOA (p={p_layers}, Hybrid Restart Strategy)...")
        ham = SparsePauliOp.from_list([(p, float(c)) for c, p in self.pauli_terms])
        
        # ✅ USAMOS LA SELECCIÓN INTELIGENTE
        backend = self._get_optimal_backend()
        
        gammas_ramp = np.linspace(0.0, 0.6 * np.pi, p_layers)
        betas_ramp = np.linspace(0.6 * np.pi, 0.0, p_layers)
        ramp_point = np.concatenate([gammas_ramp, betas_ramp])
        
        best_result = None
        best_energy = float('inf')
        best_qaoa_instance = None
        
        n_restarts = 40 
        print(f"   🚀 Ejecutando {n_restarts} rondas...")
        
        for i in range(n_restarts):
            if i < 20:
                noise = np.random.uniform(-0.1, 0.1, 2 * p_layers)
                initial_point = ramp_point + noise
            else:
                initial_point = np.random.uniform(0, 2*np.pi, 2 * p_layers)
            
            optimizer = COBYLA(maxiter=max_iter, tol=0.0001)
            
            run_shots = self.shots if not self.use_statevector else None
            
            qaoa = CustomQAOA(
                estimator=AerEstimator(), optimizer=optimizer, reps=p_layers,
                initial_point=initial_point,
                backend=backend, 
                shots=run_shots, 
                alpha=0.15 
            )
            
            try:
                res = qaoa.compute_minimum_eigenvalue(ham)
                if res.optimal_value < best_energy:
                    best_energy = res.optimal_value
                    best_result = res
                    best_qaoa_instance = qaoa
            except Exception:
                continue
        
        print(f"   🏆 Mejor energía encontrada: {best_energy:.6f}")
        
        try:
            if best_result:
                opt_circ = best_qaoa_instance.construct_circuit(ham, best_result.optimal_parameters)
                probs = self._get_probabilities(opt_circ, AerSampler(), backend, force_high_shots=True)
                costs = getattr(best_result, 'all_energies', [])
            else:
                raise ValueError("Fallo total en QAOA")
        except Exception as e:
            print(f"⚠️ QAOA Final Error: {e}")
            probs = np.zeros(2**self.n_qubits)
            best_result = type('obj', (object,), {'optimal_value': 0.0})
            costs = []
        
        bs, en = self._smart_select(probs)
        return {'bitstring': bs, 'energy': best_energy, 'costs': costs, 
                'repaired_sequence': self.decode_solution(bs), 'repaired_cost': en, 'probs': probs}

    def solve_vqe_qiskit(self, layers=2, max_iter=1000):
        print(f"🔥 Solving with Qiskit VQE (layers={layers})...")
        from qiskit.circuit.library import TwoLocal
        ham = SparsePauliOp.from_list([(p, float(c)) for c, p in self.pauli_terms])
        
        ansatz = TwoLocal(self.n_qubits, ['ry'], 'cz', reps=layers, entanglement='full')
        
        # ✅ USAMOS LA SELECCIÓN INTELIGENTE
        backend = self._get_optimal_backend()
        
        optimizer = DifferentialEvolutionOptimizer(max_iterations=max_iter, population_size=50, tol=0.001)
        
        run_shots = self.shots if not self.use_statevector else None

        vqe = CustomVQE(estimator=AerEstimator(), ansatz=ansatz, optimizer=optimizer,
            backend=backend, 
            shots=run_shots,
            alpha=1.0)
        
        try:
            res = vqe.compute_minimum_eigenvalue(ham)
            opt_circ = ansatz.assign_parameters(res.optimal_parameters)
            probs = self._get_probabilities(opt_circ.decompose(), AerSampler(), backend, force_high_shots=True)
        except Exception as e:
            print(f"⚠️ VQE Error: {e}")
            traceback.print_exc()
            probs = np.zeros(2**self.n_qubits)
            res = type('obj', (object,), {'optimal_value': 0.0, 'all_energies': []})
        
        bs, en = self._smart_select(probs)
        return {'bitstring': bs, 'energy': res.optimal_value, 'costs': getattr(res, 'all_energies', []), 
                'repaired_sequence': self.decode_solution(bs), 'repaired_cost': en, 'probs': probs}