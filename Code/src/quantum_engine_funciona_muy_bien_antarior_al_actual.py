import os
import numpy as np
import time
import math
from typing import Dict, List, Any
import random

# Forzar un solo hilo para evitar bloqueos
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["QISKIT_IN_PARALLEL"] = "FALSE"

# Imports Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA

# Import IBM (Opcional)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

from .utils_logic import decode_solution_logic
from visualization.plot_utils import ProteinPlotter
from core.hamiltonian_builder import HamiltonianBuilder

class QuantumProteinDesign:
    def __init__(self, sequence_length: int, amino_acids: List[str], **kwargs):
        self.L = sequence_length
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = max(1, int(np.ceil(np.log2(self.n_aa))))
        self.n_qubits = self.L * self.bits_per_pos
        self.backend_name = kwargs.get('quantum_backend', 'pennylane')
        self.shots = kwargs.get('shots', 5000)
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
        self.qiskit_hamiltonian = SparsePauliOp.from_list([(p, float(c)) for c, p in self.pauli_terms])

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
                cat_energy += float(coeff.real) * prod
            breakdown[category] = cat_energy
        breakdown['Total'] = sum(breakdown.values())
        return breakdown

    def _get_backend(self):
        sim_options = {"method": "statevector"} 
        return AerSimulator(device='CPU', **sim_options)

    # ------------------------------------------------------------------
    #  SOLVER CLÁSICO
    # ------------------------------------------------------------------
    def solve_classical_brute_force(self):
        print(f"\n⚡ INICIANDO SOLVER CLÁSICO (Búsqueda de Mínima Energía)")
        print(f"   Espacio de búsqueda: 2^{self.n_qubits} estados posibles.")
        
        visited_solutions = {} 
        
        if self.n_qubits <= 20:
            total_states = 2**self.n_qubits
            print(f"   🔍 Calculando la energía de TODAS las {total_states} secuencias...")
            for i in range(total_states):
                bs = format(i, f'0{self.n_qubits}b')
                en = self.compute_energy_from_bitstring(bs)
                visited_solutions[bs] = en
                if total_states > 50000 and i % (total_states//5) == 0:
                    print(f"      ... {i}/{total_states} procesados")
        else:
            print(f"   ⚠️ Espacio gigante. Usando Muestreo Inteligente.")
            n_samples = 300000 
            for _ in range(n_samples):
                bs_arr = np.random.randint(0, 2, self.n_qubits)
                bs = "".join(map(str, bs_arr))
                visited_solutions[bs] = self.compute_energy_from_bitstring(bs)
            
            best_seeds = sorted(visited_solutions.items(), key=lambda x: x[1])[:50]
            print("      ⛰️  Refinando los mejores candidatos...")
            for start_bs, start_en in best_seeds:
                curr_bs_arr = np.array([int(b) for b in start_bs])
                curr_en = start_en
                improved = True
                while improved:
                    improved = False
                    for bit_idx in range(self.n_qubits):
                        next_bs_arr = curr_bs_arr.copy()
                        next_bs_arr[bit_idx] = 1 - next_bs_arr[bit_idx]
                        next_bs = "".join(map(str, next_bs_arr))
                        if next_bs not in visited_solutions:
                            next_en = self.compute_energy_from_bitstring(next_bs)
                            visited_solutions[next_bs] = next_en
                        else: next_en = visited_solutions[next_bs]
                        if next_en < curr_en:
                            curr_en = next_en
                            curr_bs_arr = next_bs_arr
                            improved = True
                            break

        sorted_sol = sorted(visited_solutions.items(), key=lambda x: x[1])
        top_k = min(len(sorted_sol), 100)
        top_candidates = sorted_sol[:top_k]
        
        print("\n" + "="*60)
        print("🏆 TOP 10 SECUENCIAS (MENOR ENERGÍA)")
        print("="*60)
        print(f"{'Rank':<5} | {'Secuencia':<15} | {'Energía':<12}")
        print("-" * 60)
        for rank, (bs, en) in enumerate(top_candidates[:10], 1):
            print(f"{rank:<5} | {self.decode_solution(bs):<15} | {en:.6f}")
        print("-" * 60 + "\n")

        energies = np.array([en for _, en in top_candidates])
        bitstrings = [bs for bs, _ in top_candidates]
        std_dev = np.std(energies)
        T_viz = max(std_dev, 0.1) * 2.0 
        min_e = np.min(energies)
        weights = np.exp(-(energies - min_e) / T_viz)
        probs = weights / np.sum(weights)
        probs_dict = {bs: p for bs, p in zip(bitstrings, probs)}

        return {
            'bitstring': top_candidates[0][0],
            'energy': top_candidates[0][1],
            'repaired_sequence': self.decode_solution(top_candidates[0][0]),
            'repaired_cost': top_candidates[0][1],
            'probs': probs_dict,
            'costs': [] 
        }

    # ------------------------------------------------------------------
    #  SOLVER QAOA (OPTIMIZADO PARA ITERACIONES LARGAS)
    # ------------------------------------------------------------------
    def solve_qaoa_qiskit(self, p_layers=1, max_iter=300, ibm_token=None):
        backend_local = self._get_backend()
        
        if self.n_qubits >= 36:
            p_auto, n_restarts, max_iter, opt_shots = 1, 1, 50, 300
        elif self.n_qubits >= 20:
            p_auto, n_restarts, max_iter, opt_shots = 2, 3, 100, 800
        else:
            # L=2: 50 reinicios de hasta 2500 iteraciones cada uno
            p_auto, n_restarts, max_iter, opt_shots = 5, 50, 2500, 2000
        
        p_final = p_layers if p_layers > 1 else p_auto

        print(f"\n🏋️  QAOA LOCAL (CPU) | p={p_final} | Max Iters={max_iter} | {n_restarts} Reinicios")
        
        ansatz = QAOAAnsatz(self.qiskit_hamiltonian, reps=p_final)
        ansatz.measure_all()
        transpiled_qc = transpile(ansatz, backend_local)
        
        best_global_energy = float('inf')
        best_global_params = None
        best_global_history = []

        for i in range(n_restarts):
            current_history = []
            def objective_function(params):
                try: bound_qc = transpiled_qc.assign_parameters(params)
                except: bound_qc = transpiled_qc.bind_parameters(params)
                
                try:
                    job = backend_local.run(bound_qc, shots=opt_shots)
                    counts = job.result().get_counts()
                except: return 0.0
                total_en = 0; total_cts = 0
                for b, c in counts.items():
                    bs = b.replace(" ", "")[-self.n_qubits:]
                    total_en += self.compute_energy_from_bitstring(bs) * c
                    total_cts += c
                avg = total_en / total_cts if total_cts > 0 else 0
                current_history.append(avg)
                return avg

            if i == 0 and p_final > 1:
                gammas = np.linspace(0, np.pi, p_final)
                betas = np.linspace(np.pi, 0, p_final)
                initial_point = []
                for k in range(p_final): initial_point.extend([betas[k], gammas[k]])
                initial_point = np.array(initial_point)
            else:
                initial_point = np.random.uniform(0, 2*np.pi, 2 * p_final)

            # --- TOLERANCIA ESTRICTA ---
            # 1e-8 obliga al optimizador a seguir buscando mejoras microscópicas
            tol = 1e-8 if self.n_qubits < 20 else 0.1
            
            optimizer = COBYLA(maxiter=max_iter, tol=tol)
            res = optimizer.minimize(objective_function, initial_point)
            
            if res.fun < best_global_energy:
                best_global_energy = res.fun
                best_global_params = res.x
                best_global_history = current_history
                # Imprimimos explícitamente cuántas iteraciones hizo este intento
                if n_restarts > 1: 
                    print(f"   ✅ Restart {i+1}: Nuevo Récord -> {best_global_energy:.6f} (Iteraciones: {res.nfev})")

        probs_dict = {}
        
        if ibm_token and IBM_AVAILABLE:
            print("\n☁️  CONECTANDO A IBM QUANTUM...")
            try:
                service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
                backend_ibm = service.least_busy(operational=True, simulator=False, min_num_qubits=self.n_qubits)
                print(f"   🤖 Backend: {backend_ibm.name}")
                
                ibm_qc = transpile(ansatz, backend_ibm, optimization_level=3)
                try: final_qc = ibm_qc.assign_parameters(best_global_params)
                except: final_qc = ibm_qc.bind_parameters(best_global_params)

                print("   🚀 Enviando Job...")
                job = backend_ibm.run(final_qc, shots=4000)
                print(f"   🆔 Job ID: {job.job_id()}")
                
                result = job.result()
                counts = result.get_counts()
                for b, c in counts.items():
                    bs = b.replace(" ", "")[-self.n_qubits:]
                    probs_dict[bs] = c / 4000
                    
            except Exception as e:
                print(f"❌ Error IBM: {e}. Usando local.")
                probs_dict = self._run_local_final(transpiled_qc, best_global_params, backend_local)
        else:
            print("   🏠 Ejecutando tirada final Local...")
            probs_dict = self._run_local_final(transpiled_qc, best_global_params, backend_local)

        best_bs, min_en = self._smart_select(probs_dict)
        try: self.plotter.plot_optimization(best_global_history, solver_name="QAOA")
        except: pass

        return {
            'bitstring': best_bs, 'energy': min_en, 'costs': best_global_history, 
            'repaired_sequence': self.decode_solution(best_bs), 
            'repaired_cost': min_en, 'probs': probs_dict
        }

    def _run_local_final(self, qc, params, backend):
        try: final_qc = qc.assign_parameters(params)
        except: final_qc = qc.bind_parameters(params)
        shots = 100000 if self.n_qubits < 20 else 5000
        job = backend.run(final_qc, shots=shots)
        counts = job.result().get_counts()
        probs = {}
        for b, c in counts.items():
            bs = b.replace(" ", "")[-self.n_qubits:]
            probs[bs] = c / shots
        return probs

    def solve_vqe_qiskit(self, layers=1, max_iter=100):
        return self.solve_classical_brute_force()

    def _smart_select(self, probs_dict):
        best_bs, min_en = None, float('inf')
        for bs in probs_dict.keys():
            en = self.compute_energy_from_bitstring(bs)
            if en < min_en: min_en, best_bs = en, bs
        if best_bs is None: return '0'*self.n_qubits, 0.0
        return best_bs, min_en