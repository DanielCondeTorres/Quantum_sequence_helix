import numpy as np
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import traceback

class CustomVQE:
    """Custom VQE class with robust error handling and statevector support"""
    def __init__(self, estimator, ansatz, optimizer, initial_point=None, callback=None):
        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer
        self._initial_point = initial_point
        self._callback = callback
        self._evaluation_count = 0
        self._operator = None
        self._optimal_parameters = None
        self._optimal_value = None
    
    def evaluate_energy(self, parameters):
        try:
            # Ensure parameters are valid
            parameters = np.array(parameters, dtype=float)
            if not np.all(np.isfinite(parameters)):
                print(f"⚠️ Invalid parameters: {parameters}")
                return 1e6
            
            # Validate Hamiltonian
            for pauli, coeff in zip(self._operator.paulis, self._operator.coeffs):
                if not np.isfinite(coeff):
                    print(f"⚠️ Invalid Hamiltonian coefficient: {coeff} for {pauli}")
                    return 1e6
            
            # Use statevector calculation (preferred for accuracy and avoiding primitive issues)
            try:
                from qiskit.quantum_info import Statevector
                circuit = self.ansatz.assign_parameters(parameters)
                statevector = Statevector(circuit)
                energy = 0.0
                for pauli, coeff in zip(self._operator.paulis, self._operator.coeffs):
                    coeff = float(coeff.real)
                    if abs(coeff) > 1e-10:
                        expectation = statevector.expectation_value(pauli)
                        energy += coeff * float(np.real(expectation))
                energy = float(energy)
            except Exception as e:
                print(f"⚠️ Statevector calculation failed: {e}")
                return 1e6
            
            if not np.isfinite(energy):
                print(f"⚠️ Non-finite energy computed: {energy}")
                return 1e6
            
            # Call callback if provided
            if self._callback is not None:
                try:
                    self._callback(
                        nfev=self._evaluation_count,
                        parameters=parameters,
                        energy=energy
                    )
                except Exception as e:
                    print(f"⚠️ Callback failed: {e}")
            
            self._evaluation_count += 1
            return energy
        except Exception as exc:
            print(f"⚠️ SPSA evaluation failed: '{str(exc)}'")
            return 1e6  # Return large penalty
    
    def compute_minimum_eigenvalue(self, operator):
        """Compute minimum eigenvalue using the optimizer"""
        from qiskit_algorithms.optimizers import OptimizerResult
        
        self._operator = operator
        
        # Use the optimizer to find minimum
        result = self.optimizer.minimize(
            fun=self.evaluate_energy,
            x0=self._initial_point,
            operator=operator
        )
        
        self._optimal_parameters = result.x
        self._optimal_value = result.fun
        
        # Create result object
        class VQEResult:
            def __init__(self, optimal_parameters, optimal_value):
                self.optimal_parameters = optimal_parameters
                self.optimal_value = optimal_value
        
        return VQEResult(self._optimal_parameters, self._optimal_value)

class CustomQAOA:
    """Custom QAOA class to handle Qiskit QAOA without ansatz setter issues"""
    def __init__(self, sampler, optimizer, reps=1, initial_point=None, callback=None):
        self.sampler = sampler
        self.optimizer = optimizer
        self.reps = reps
        self._initial_point = initial_point if initial_point is not None else np.random.uniform(0, np.pi, 2 * reps)
        self.callback = callback
        self._optimal_parameters = None
        self._optimal_value = None
        self._costs = []
        self._operator = None

    def construct_circuit(self, operator, parameters):
        """Construct QAOA circuit with proper problem and mixer Hamiltonians"""
        num_qubits = operator.num_qubits
        circuit = QuantumCircuit(num_qubits)
        p = self.reps
        gammas = parameters[:p]
        betas = parameters[p:]
        
        # Initial state: uniform superposition
        circuit.h(range(num_qubits))
        
        # Apply p layers of QAOA
        for layer in range(p):
            # Problem Hamiltonian: exp(-i * gamma * H_P)
            # Apply rotations for each Pauli term in the operator
            for pauli, coeff in zip(operator.paulis, operator.coeffs):
                coeff = float(coeff.real)
                if abs(coeff) < 1e-10:
                    continue
                    
                pauli_str = str(pauli)
                if pauli_str == 'I' * num_qubits:
                    # Identity term doesn't affect the circuit (global phase)
                    continue
                
                # For multi-qubit Pauli terms (e.g., ZZ, ZZZ), we need to handle them properly
                # Collect qubit indices and corresponding Pauli operators
                active_qubits = []
                pauli_types = []
                for idx in range(num_qubits):
                    gate = pauli_str[idx]
                    if gate != 'I':
                        active_qubits.append(idx)
                        pauli_types.append(gate)
                
                if len(active_qubits) == 0:
                    continue
                
                # Convert X and Y to Z basis if needed
                for idx, gate_type in zip(active_qubits, pauli_types):
                    if gate_type == 'X':
                        circuit.h(idx)
                    elif gate_type == 'Y':
                        circuit.sdg(idx)
                        circuit.h(idx)
                
                # Apply CNOTs to entangle qubits for multi-qubit terms
                if len(active_qubits) > 1:
                    for i in range(len(active_qubits) - 1):
                        circuit.cx(active_qubits[i], active_qubits[i+1])
                
                # Apply rotation on the last qubit
                angle = 2 * coeff * gammas[layer]
                circuit.rz(angle, active_qubits[-1])
                
                # Uncompute CNOTs
                if len(active_qubits) > 1:
                    for i in range(len(active_qubits) - 2, -1, -1):
                        circuit.cx(active_qubits[i], active_qubits[i+1])
                
                # Convert back from Z basis
                for idx, gate_type in zip(active_qubits, pauli_types):
                    if gate_type == 'X':
                        circuit.h(idx)
                    elif gate_type == 'Y':
                        circuit.h(idx)
                        circuit.s(idx)
            
            # Mixer Hamiltonian: exp(-i * beta * H_M) where H_M = sum_i X_i
            for qubit in range(num_qubits):
                circuit.rx(2 * betas[layer], qubit)
        
        return circuit

    def compute_minimum_eigenvalue(self, operator):
        """Run QAOA optimization"""
        from qiskit_algorithms.optimizers import OptimizerResult
        
        self._operator = operator
        result = OptimizerResult()
        
        def objective_function(params):
            circuit = self.construct_circuit(operator, params)
            try:
                psi = Statevector.from_instruction(circuit)
                energy = 0.0
                
                # Calculate expectation value properly
                for pauli, coeff in zip(operator.paulis, operator.coeffs):
                    coeff = float(coeff.real)
                    if abs(coeff) > 1e-10:
                        expectation = psi.expectation_value(pauli)
                        energy += coeff * float(np.real(expectation))
                
                if not np.isfinite(energy):
                    print(f"⚠️ Non-finite energy computed: {energy}")
                    return 1e6
                    
                energy = float(energy)
                
            except Exception as e:
                print(f"⚠️ Error calculating energy: {e}")
                traceback.print_exc()
                return 1e6
                
            if self.callback:
                self.callback(nfev=len(self._costs), parameters=params, energy=energy)
            self._costs.append(energy)
            return energy

        try:
            # Pass operator to optimizer for classical fallback if needed
            opt_result = self.optimizer.minimize(
                fun=objective_function,
                x0=self._initial_point,
                operator=operator
            )
            self._optimal_parameters = opt_result.x
            self._optimal_value = opt_result.fun
            result.optimal_parameters = self._optimal_parameters
            result.optimal_value = self._optimal_value
            return result
        except Exception as e:
            print(f"❌ Error during QAOA optimization: {e}")
            traceback.print_exc()
            raise

class HybridMultiStartOptimizer:
    """Optimizer using Differential Evolution for better convergence"""
    def __init__(self, max_iterations=2000, n_starts=10, patience=50, restart_threshold=1e-6):
        self.max_iterations = max_iterations
        self.n_starts = n_starts
        self.patience = patience
        self.restart_threshold = restart_threshold
        # Optional external callback to report (nfev, parameters, energy)
        self.external_callback = None
        
    def get_diverse_initializations(self, n_params, rng):
        """Generate diverse initializations"""
        strategies = [
            np.zeros(n_params),
            rng.uniform(0, 0.1 * np.pi, n_params),
            rng.uniform(0.5 * np.pi, np.pi, n_params),
            rng.uniform(np.pi, 2 * np.pi, n_params),
            np.linspace(0, np.pi, n_params),
            np.sin(np.linspace(0, 4 * np.pi, n_params)) * np.pi/2 + np.pi/2,
            rng.normal(np.pi, np.pi/4, n_params).clip(0, 2*np.pi),
            rng.uniform(0, 2 * np.pi, n_params)
        ]
        return strategies[:self.n_starts]
    
    def detect_local_minimum(self, history, window=50):
        """Robust local minimum detection"""
        if len(history) < window:
            return False
        recent = history[-window:]
        variance = np.var(recent)
        mean_gradient = abs(recent[-1] - recent[0]) / window
        return variance < self.restart_threshold and mean_gradient < self.restart_threshold / 10
    
    def classical_fallback(self, operator):
        """Compute classical minimum energy for small systems"""
        print("🧮 Running classical fallback...")
        num_qubits = operator.num_qubits
        min_energy = float('inf')
        min_bitstring = None
        for idx in range(2**num_qubits):
            bitstring = format(idx, f'0{num_qubits}b')
            state = np.zeros(2**num_qubits)
            state[idx] = 1.0
            energy = 0.0
            for pauli, coeff in zip(operator.paulis, operator.coeffs):
                coeff = float(coeff.real)
                if abs(coeff) > 1e-10:
                    pauli_str = str(pauli)
                    if pauli_str == 'I' * num_qubits:
                        energy += coeff
                    else:
                        z_values = [1 if bit == '0' else -1 for bit in bitstring]
                        expectation = 1.0
                        for i, gate in enumerate(pauli_str):
                            if gate == 'Z':
                                expectation *= z_values[i]
                        energy += coeff * expectation
            if energy < min_energy:
                min_energy = energy
                min_bitstring = bitstring
        print(f"Debug: Classical fallback energy: {min_energy:.6f}, bitstring: {min_bitstring}")
        return min_energy, min_bitstring
    
    def minimize(self, fun, x0, jac=None, bounds=None, operator=None):
        """Fast L-BFGS-B optimization with multiple starts"""
        from qiskit_algorithms.optimizers import OptimizerResult
        from scipy.optimize import minimize
        import time
        
        n_params = len(x0)
        rng = np.random.default_rng(42)
        all_energies = []
        
        # Set up bounds
        if bounds is None:
            bounds = [(0, 2*np.pi) for _ in range(n_params)]
        
        def tracking_objective(params):
            try:
                energy = fun(params)
                if not np.isfinite(energy):
                    print(f"⚠️ Non-finite energy: {energy}")
                    energy = 1e6
            except Exception as e:
                print(f"⚠️ Energy evaluation failed: {e}")
                energy = 1e6
            
            all_energies.append(energy)
            
            # Forward to external callback if provided
            if self.external_callback is not None:
                try:
                    self.external_callback(nfev=len(all_energies), parameters=params, energy=energy)
                except Exception as cb_e:
                    print(f"⚠️ External callback failed: {cb_e}")
            
            return energy
        
        print(f"\n  ⚡ Fast L-BFGS-B optimization...")
        print(f"  📊 Parameters: {n_params}, Multiple starts: {min(10, self.n_starts)}")
        
        try:
            # Generate diverse starting points
            start_points = [
                x0,  # Original point
                rng.uniform(0, 2*np.pi, n_params),  # Random
                np.zeros(n_params),  # Zero
                np.full(n_params, np.pi),  # Pi
            ]
            
            best_result = None
            best_energy = float('inf')
            
            for i, start_point in enumerate(start_points[:min(10, self.n_starts)]):
                print(f"  🎯 Start {i+1}/{min(10, self.n_starts)}")
                
                result = minimize(
                    tracking_objective,
                    start_point,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': self.max_iterations // min(10, self.n_starts),
                        'ftol': 1e-6,
                        'gtol': 1e-6,
                        'maxfun': 1000  # Limit function evaluations
                    }
                )
                
                print(f"    📊 Energy: {result.fun:.6f}")
                
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_result = result
                    print(f"    🏆 New best!")
            
            if best_result is None:
                raise Exception("All L-BFGS-B runs failed")
            
            print(f"\n  ✅ Final best energy: {best_energy:.6f}")
            print(f"  📊 Total evaluations: {len(all_energies)}")
            if all_energies:
                print(f"  📊 Average energy: {np.mean(all_energies):.6f}")
                print(f"  📊 Minimum energy: {np.min(all_energies):.6f}")
            
            result = OptimizerResult()
            result.x = best_result.x
            result.fun = best_energy
            result.nfev = len(all_energies)
            result.history = all_energies
            result.all_energies = all_energies
            return result
            
        except Exception as e:
            print(f"❌ L-BFGS-B optimization failed: {e}")
            
            # Classical fallback for small systems
            if operator is not None:
                print("⚠️ Using classical fallback...")
                classical_energy, classical_bitstring = self.classical_fallback(operator)
                all_energies.append(classical_energy)
                
                result = OptimizerResult()
                result.x = x0
                result.fun = classical_energy
                result.nfev = len(all_energies)
                result.history = all_energies
                result.all_energies = all_energies
                result.classical_bitstring = classical_bitstring
                return result
            else:
                raise