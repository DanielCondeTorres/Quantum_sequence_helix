import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_algorithms.optimizers import OptimizerResult
from scipy.optimize import differential_evolution
import traceback

# ==============================================================================
#  HELPER: CVaR Calculation
# ==============================================================================
def compute_cvar_fast(energies, probabilities, alpha=0.25):
    if alpha >= 1.0 - 1e-6:
        return np.dot(energies, probabilities)
    sorted_idx = np.argsort(energies)
    sorted_energies = energies[sorted_idx]
    sorted_probs = probabilities[sorted_idx]
    cum_probs = np.cumsum(sorted_probs)
    cutoff_index = np.searchsorted(cum_probs, alpha)
    cvar_sum = 0.0
    if cutoff_index > 0:
        cvar_sum = np.dot(sorted_energies[:cutoff_index], sorted_probs[:cutoff_index])
    prob_accumulated = cum_probs[cutoff_index-1] if cutoff_index > 0 else 0.0
    remainder = alpha - prob_accumulated
    if cutoff_index < len(sorted_energies):
        cvar_sum += remainder * sorted_energies[cutoff_index]
    return cvar_sum / alpha

# ==============================================================================
#  CLASE BASE: GESTIÓN DE ENERGÍA
# ==============================================================================
class FastQuantumSolver:
    def __init__(self):
        self.diagonal_energies = None
        self.hamiltonian_sparse = None

    def precompute_diagonal(self, hamiltonian, n_qubits):
        try:
            if n_qubits <= 22:
                if hasattr(hamiltonian, 'to_matrix'):
                     self.diagonal_energies = np.real(hamiltonian.to_matrix().diagonal())
                else:
                     self.diagonal_energies = np.real(hamiltonian.to_operator().data.diagonal())
            else:
                self.diagonal_energies = None
                self.hamiltonian_sparse = hamiltonian
        except Exception:
            self.diagonal_energies = None
            self.hamiltonian_sparse = hamiltonian

    def get_energy_fast(self, counts, alpha=1.0):
        bitstrings = list(counts.keys())
        raw_counts = np.array(list(counts.values()), dtype=float)
        probs = raw_counts / np.sum(raw_counts)
        
        if self.diagonal_energies is not None:
            indices = [int(b.replace(" ", ""), 2) for b in bitstrings]
            energies = self.diagonal_energies[indices]
        else:
            energies = []
            for b in bitstrings:
                clean_b = b.replace(" ", "")
                z_vals = np.array([1 if bit == '0' else -1 for bit in clean_b])
                e = 0.0
                for op_str, coeff in zip(self.hamiltonian_sparse.paulis.to_labels(), self.hamiltonian_sparse.coeffs):
                    term = float(coeff.real)
                    for i, char in enumerate(reversed(op_str)):
                        if char == 'Z': term *= z_vals[i]
                    e += term
                energies.append(e)
            energies = np.array(energies)
        return compute_cvar_fast(energies, probs, alpha)

# ==============================================================================
#  OPTIMIZADOR (CON HISTORY TRACKING)
# ==============================================================================
class DifferentialEvolutionOptimizer:
    def __init__(self, max_iterations=1000, population_size=20, tol=0.01, atol=0):
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.tol = tol
        self.atol = atol
        
    def minimize(self, fun, x0, jac=None, bounds=None):
        n_params = len(x0)
        if bounds is None: bounds = [(0, 2*np.pi)] * n_params
        
        print(f"  🌍 DE Optimizer: Pop={self.population_size}, MaxIter={self.max_iterations}")

        result = differential_evolution(
            func=fun,
            bounds=bounds,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            polish=True, # Polish activado para refinar el resultado final
            workers=1,          
            strategy='best1bin', 
            atol=self.atol,          
            tol=self.tol,          
            mutation=(0.5, 1.5),  
            recombination=0.7,    
            disp=True
        )
        
        res = OptimizerResult()
        res.x = result.x
        res.fun = result.fun
        res.nfev = result.nfev
        return res

# ==============================================================================
#  VQE & QAOA (FIXED PARAMETER BINDING)
# ==============================================================================
class CustomVQE(FastQuantumSolver):
    def __init__(self, estimator, ansatz, optimizer, initial_point=None, 
                 callback=None, backend=None, shots=None, alpha=1.0):
        super().__init__()
        self.temp_backend = backend
        self.temp_shots = shots
        self.ansatz = ansatz
        self.optimizer = optimizer
        self._initial_point = initial_point
        self.alpha = alpha 
        self._cached_circuit = None
        self.history = [] 
        print(f"   🎯 VQE configurado con α={alpha:.2f}")

    def compute_minimum_eigenvalue(self, operator):
        self.precompute_diagonal(operator, self.ansatz.num_qubits)
        
        # Descomponer y transpilar
        raw_circuit = self.ansatz.copy()
        raw_circuit.measure_all()
        self._cached_circuit = transpile(raw_circuit, self.temp_backend)
        
        self.history = [] 
        
        def objective_function(params):
            try:
                # Asignación de parámetros segura
                bound_circ = self._cached_circuit.assign_parameters(params)
                counts = {}
                if self.temp_backend:
                    job = self.temp_backend.run(bound_circ, shots=self.temp_shots)
                    res = job.result()
                    if hasattr(res, 'get_counts'): counts = res.get_counts()
                    elif isinstance(res, list): counts = res[0].data.meas.get_counts()
                    else: counts = getattr(res, 'get_counts', lambda: {})()
                
                cost = self.get_energy_fast(counts, self.alpha)
                self.history.append(cost)
                return cost
            except Exception as e:
                if len(self.history) == 0: print(f"❌ CRASH en VQE Objective: {e}")
                return 1000.0

        x0 = self._initial_point if self._initial_point is not None else np.random.rand(self.ansatz.num_parameters)
        res = self.optimizer.minimize(objective_function, x0)
        
        class Result:
            def __init__(self, x, fun, hist): 
                self.optimal_parameters = x
                self.optimal_value = fun
                self.all_energies = hist
        
        return Result(res.x, res.fun, self.history)

class CustomQAOA(FastQuantumSolver):
    def __init__(self, estimator, optimizer, reps=3, initial_point=None, 
                 callback=None, backend=None, shots=None, alpha=1.0):
        super().__init__()
        self.reps = reps
        self.optimizer = optimizer
        self._initial_point = initial_point
        self.backend = backend
        self.shots = shots
        self.alpha = alpha 
        self._cached_circuit = None
        self.history = []
        print(f"   🎯 QAOA configurado con p={reps}, α={alpha:.2f}")

    def _build_qaoa_circuit(self, operator):
        n = operator.num_qubits
        qc = QuantumCircuit(n)
        # Usamos ParameterVector para gestión limpia
        theta = ParameterVector('Theta', 2 * self.reps)
        qc.h(range(n))
        idx = 0
        for layer in range(self.reps):
            gamma, beta = theta[idx], theta[idx+1]
            idx += 2
            for pauli, coeff in operator.label_iter(): 
                if abs(coeff.real) < 1e-5: continue
                active = [i for i, c in enumerate(reversed(pauli)) if c == 'Z']
                if len(active) == 1: qc.rz(2 * coeff.real * gamma, active[0])
                elif len(active) == 2: qc.rzz(2 * coeff.real * gamma, active[0], active[1])
            qc.rx(2 * beta, range(n))
        qc.measure_all()
        return qc

    def compute_minimum_eigenvalue(self, operator):
        self.precompute_diagonal(operator, operator.num_qubits)
        
        # Construimos y transpilamos UNA VEZ
        raw_circuit = self._build_qaoa_circuit(operator)
        self._cached_circuit = transpile(raw_circuit, self.backend)
        self.history = []

        def cost_func(params):
            try:
                # Asignación segura
                bound_circ = self._cached_circuit.assign_parameters(params)
                counts = {}
                if self.backend:
                    job = self.backend.run(bound_circ, shots=self.shots)
                    res = job.result()
                    if hasattr(res, 'get_counts'): counts = res.get_counts()
                    elif isinstance(res, list): counts = res[0].data.meas.get_counts()
                    else: counts = getattr(res, 'get_counts', lambda: {})()
                
                cost = self.get_energy_fast(counts, self.alpha)
                self.history.append(cost)
                return cost
            except Exception as e:
                if len(self.history) == 0: print(f"❌ CRASH en QAOA Objective: {e}")
                return 1000.0

        if self._initial_point is None:
            self._initial_point = np.concatenate([np.random.uniform(0, 0.2, self.reps), np.random.uniform(0.6, 1.0, self.reps)])
            
        res = self.optimizer.minimize(cost_func, self._initial_point)
        
        class Result:
            def __init__(self, x, fun, hist): 
                self.optimal_parameters = x
                self.optimal_value = fun
                self.all_energies = hist
        return Result(res.x, res.fun, self.history)
    
    def construct_circuit(self, operator, parameters):
        """
        Reconstruye el circuito final asegurando que los parámetros se aplican correctamente.
        """
        # Si tenemos el circuito en caché (transpilado y listo), lo usamos
        if self._cached_circuit is not None:
            return self._cached_circuit.assign_parameters(parameters)
        
        # Si no, lo reconstruimos (caso raro)
        print("⚠️ Warning: Rebuilding QAOA circuit from scratch (cache miss)")
        raw_circ = self._build_qaoa_circuit(operator)
        transpiled = transpile(raw_circ, self.backend)
        return transpiled.assign_parameters(parameters)