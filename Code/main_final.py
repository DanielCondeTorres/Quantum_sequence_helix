# main.py
import argparse
import sys
import numpy as np
import pennylane as qml
from typing import Optional, List, Tuple, Dict, Any

try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.algorithms import QAOA, VQE
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.opflow import PauliSumOp
    from qiskit.providers.aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not available. Using PennyLane only.")

from core.protein_designer import QuantumProteinDesign

def describe_qaoa(n_qubits: int, p_layers: int):
    print("🔥 QAOA CIRCUIT STRUCTURE:")
    print(f"   • Qubits: {n_qubits}")
    print(f"   • p-layers: {p_layers}")
    print(f"   • Gate sequence:")
    print(f"     1. Hadamard on all qubits → superposition")
    print(f"     2. For each p-layer:")
    print(f"        - Cost Hamiltonian evolution (problem-specific)")
    print(f"        - Mixer Hamiltonian evolution (RX rotations)")

def run_quantum_protein_design(sequence_length: int = 3, amino_acids: Optional[List[str]] = None,
                               membrane_span: Optional[Tuple[int,int]] = None,
                               membrane_charge: str = 'neu',
                               lambda_charge: float = 0.0,
                               lambda_env: float = 0.0,
                               lambda_mu: float = 0.0,
                               membrane_positions: Optional[List[int]] = None,
                               membrane_mode: str = 'span',
                               wheel_phase_deg: float = 0.0,
                               wheel_halfwidth_deg: float = 40.0,
                               classical: bool = False):
    
    print("🚀 QUANTUM PROTEIN SEQUENCE DESIGN 🚀")
    print("="*50)
    
    designer = QuantumProteinDesign(
        sequence_length=sequence_length,
        amino_acids=amino_acids if amino_acids is not None else ['A', 'L', 'E', 'K', 'W'],
        quantum_backend='pennylane',
        membrane_span=membrane_span,
        membrane_charge=membrane_charge,
        lambda_charge=lambda_charge,
        lambda_env=lambda_env,
        lambda_mu=lambda_mu,
        membrane_positions=membrane_positions,
        membrane_mode=membrane_mode,
        wheel_phase_deg=wheel_phase_deg,
        wheel_halfwidth_deg=wheel_halfwidth_deg,
        classical=classical,
    )
    
    print(f"\nTotal qubits required: {designer.n_qubits}")
    print(f"Sequence length (L): {designer.L} | Amino acids (|Σ|): {designer.n_aa} | bits/pos: {designer.bits_per_pos}")
    
    print("\nQubit mapping (qubit -> position:bit):")
    for i in range(designer.L):
        row = []
        for b in range(designer.bits_per_pos):
            q = i * designer.bits_per_pos + b
            row.append(f"{q}->{i}:b{b}")
        print("  " + "  ".join(row))

    describe_qaoa(n_qubits=designer.n_qubits, p_layers=2)
    
    if not classical:
        designer.plot_qaoa_circuit(p_layers=1)
    
    if classical:
        print("\n🧮 Solving classically (QUBO heuristic)...")
        qaoa_result = designer.solve_classical_qubo()
        sequence_qaoa, violations_qaoa = designer.analyze_solution(qaoa_result)
    else:
        qaoa_result = designer.solve_qaoa_pennylane(p_layers=2, max_iterations=200)
        sequence_qaoa, violations_qaoa = designer.analyze_solution(qaoa_result)
        designer.plot_alpha_helix_wheel(qaoa_result['repaired_sequence'])
    
    print(f"\n📊 RESULTADOS 📊")
    print(f"Solución QAOA: {sequence_qaoa} (violaciones: {violations_qaoa})")
    print(f"Secuencia reparada: {qaoa_result['repaired_sequence']} | E(clásica): {qaoa_result['repaired_cost']:.6f}")
    print(f"Energía (expval): {qaoa_result['energy']:.6f}")
    
    if not classical and qaoa_result.get('costs'):
        designer.plot_optimization(qaoa_result['costs'])
    
    return designer, qaoa_result

def demonstrate_scaling():
    print("\n📏 QUBIT SCALING ANALYSIS 📏")
    print("="*40)
    lengths = [2, 3, 4, 5, 10]
    n_amino_acids = [4, 8, 20]
    print("Sequence | AA Types | Total Qubits | Feasible?")
    print("-" * 45)
    for L in lengths:
        for n_aa in n_amino_acids:
            total_qubits = L * (1 << int(np.ceil(np.log2(n_aa))))
            feasible = "✅" if total_qubits <= 20 else "❌" if total_qubits <= 50 else "🚫"
            print(f"{L:8d} | {n_aa:8d} | {total_qubits:11d} | {feasible}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum peptide design with QAOA")
    parser.add_argument("--length", "-L", type=int, default=6, help="Peptide length (number of residues)")
    parser.add_argument("--residues", "-R", type=str, default='VLT', help="Allowed residues, e.g. 'A,L,E,K' or 'ALEK'")
    parser.add_argument("--membrane", "-M", type=str, default=None, help="Membrane span as start:end (0-based, inclusive), e.g. '1:4'")
    parser.add_argument("--membrane_positions", type=str, default=None, help="Explicit membrane-facing positions, e.g. '0,2,5' (0-based)")
    parser.add_argument("--membrane_mode", type=str, default='span', choices=['span','set','wheel'], help="Membrane selection mode")
    parser.add_argument("--wheel_phase_deg", type=float, default=0.0, help="Helical wheel phase in degrees (membrane-facing direction)")
    parser.add_argument("--wheel_halfwidth_deg", type=float, default=40.0, help="Half-width (deg) around membrane-facing direction")
    parser.add_argument("--membrane_charge", type=str, default='neu', choices=['neg','pos','neu'], help="Membrane charge: neg/pos/neu")
    parser.add_argument("--lambda_charge", type=float, default=0.0, help="Weight for membrane charge term")
    parser.add_argument("--lambda_env", type=float, default=0.0, help="Weight for environment hydrophobicity term")
    parser.add_argument("--lambda_mu", type=float, default=0.0, help="Weight for hydrophobic moment alignment term")
    parser.add_argument("--classical", action='store_true', help="Solve classically (no QAOA)")
    args = parser.parse_args()

    if args.residues is None:
        aa_list = None
    else:
        s = args.residues.strip().upper()
        if "," in s:
            aa_list = [t.strip() for t in s.split(",") if t.strip()]
        else:
            aa_list = [c for c in s if c.strip()]

    mem_span = None
    if args.membrane:
        try:
            a, b = args.membrane.split(":")
            mem_span = (int(a), int(b))
        except Exception:
            print("Invalid --membrane format. Use start:end, e.g. 1:4")
            sys.exit(1)

    mem_positions = None
    if args.membrane_positions:
        try:
            mem_positions = [int(t) for t in args.membrane_positions.split(',') if t.strip()]
        except Exception:
            print("Invalid --membrane_positions. Use comma-separated indices, e.g. 0,2,5")
            sys.exit(1)

    designer, qaoa_result = run_quantum_protein_design(
        sequence_length=args.length,
        amino_acids=aa_list,
        membrane_span=mem_span,
        membrane_charge=args.membrane_charge,
        lambda_charge=args.lambda_charge,
        lambda_env=args.lambda_env,
        lambda_mu=args.lambda_mu,
        membrane_positions=mem_positions,
        membrane_mode=args.membrane_mode,
        wheel_phase_deg=args.wheel_phase_deg,
        wheel_halfwidth_deg=args.wheel_halfwidth_deg,
        classical=args.classical,
    )
    
    demonstrate_scaling()