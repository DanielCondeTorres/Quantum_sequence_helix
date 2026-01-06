import numpy as np
import os

def _get_top_candidates(designer, result, top_n=50):
    """
    Función auxiliar para extraer y ordenar los candidatos.
    Se usa tanto para imprimir en pantalla como para guardar en archivo.
    """
    probs = result.get('probs', None)
    top_candidates = []

    if probs is None:
        return []

    # CASO A: Diccionario {bitstring: probabilidad} (MPS/Quantum)
    if isinstance(probs, dict):
        # Filtrar ruido y ordenar
        sorted_items = sorted(
            [(k, v) for k, v in probs.items() if v > 1e-8], 
            key=lambda x: x[1], 
            reverse=True
        )
        top_candidates = sorted_items[:top_n]

    # CASO B: Numpy Array (Statevector/Clásico)
    else:
        if len(probs) > top_n:
            indices = np.argpartition(probs, -top_n)[-top_n:]
            indices = indices[np.argsort(probs[indices])[::-1]]
        else:
            indices = np.argsort(probs)[::-1]

        for idx in indices:
            p = probs[idx]
            if p < 1e-8: continue
            bs = format(idx, f'0{designer.n_qubits}b')
            top_candidates.append((bs, p))
            
    return top_candidates

def save_energy_results(designer, result, solver_name, output_dir):
    """
    Guarda el REPORTE COMPLETO en energy.txt (Resumen + Tabla Top 50 + Desglose).
    """
    filepath = os.path.join(output_dir, "energy.txt")
    
    try:
        with open(filepath, "w") as f:
            # 1. CABECERA Y RESUMEN
            final_solver = solver_name if solver_name else result.get('solver', 'qaoa')
            seq = result.get('repaired_sequence', 'N/A')
            energy = result.get('repaired_cost', 0.0)
            
            f.write("="*80 + "\n")
            f.write(f"=== ENERGY RESULTS - {final_solver.upper()} SOLVER ===\n")
            f.write("="*80 + "\n")
            f.write(f"Solver: {final_solver}\n")
            f.write(f"Lowest Energy Sequence found: {seq} (Energy: {energy:.6f})\n")
            f.write("-" * 80 + "\n\n")

            # 2. GENERAR TABLA TOP 50
            f.write(f"TOP SEQUENCES DETAILED REPORT\n")
            f.write("-" * 80 + "\n")
            
            top_candidates = _get_top_candidates(designer, result, top_n=50)
            
            if not top_candidates:
                f.write("No probability distribution available to generate ranking.\n")
            
            for rank, (bs, p) in enumerate(top_candidates, 1):
                # Decodificar y calcular energías
                decoded_seq = designer.decode_solution(bs)
                current_energy = designer.compute_energy_from_bitstring(bs)
                breakdown = designer.compute_energy_breakdown(bs)

                # Escribir línea principal
                f.write(f"Rank {rank:2}: {decoded_seq} | Energy: {current_energy:.6f} | Probability: {p:.6f} | Bitstring: {bs}\n")
                
                # Escribir desglose detallado
                if breakdown and isinstance(breakdown, dict):
                    f.write("      Energy Breakdown:\n")
                    for key, value in breakdown.items():
                        if key != 'Total':
                            f.write(f"        {key:<30}: {value:.6f}\n")
                    
                    # Offset constante
                    if 'Total' in breakdown and abs(breakdown['Total'] - current_energy) > 1e-5:
                        offset = current_energy - breakdown['Total']
                        f.write(f"        {'Constant Offset':<30}: {offset:.6f}\n")
                
                f.write("-" * 80 + "\n")

        print(f"✅ Full detailed report saved to {filepath}")

    except Exception as e:
        print(f"⚠️ Error saving energy.txt: {e}")

def print_top_sequences_table(designer, result):
    """
    Imprime la tabla en la consola (STDOUT) para verla en el log de Slurm.
    """
    probs = result.get('probs', None)
    
    print("\n" + "="*80)
    print(f"TOP SEQUENCES REPORT (Solver: {result.get('solver', 'QAOA/VQE')})")
    print("-" * 80)

    top_candidates = _get_top_candidates(designer, result, top_n=50)

    if not top_candidates:
        print("⚠️ No probability distribution available.")
        return

    for rank, (bs, p) in enumerate(top_candidates, 1):
        seq = designer.decode_solution(bs)
        energy = designer.compute_energy_from_bitstring(bs)
        breakdown = designer.compute_energy_breakdown(bs)

        print(f"Rank {rank:2}: {seq} | Energy: {energy:.6f} | Probability: {p:.6f} | Bitstring: {bs}")
        
        if breakdown and isinstance(breakdown, dict):
            print("      Energy Breakdown:")
            for key, value in breakdown.items():
                if key != 'Total':
                    print(f"        {key:<30}: {value:.6f}")
            if 'Total' in breakdown and abs(breakdown['Total'] - energy) > 1e-5:
                 print(f"        {'Constant Offset':<30}: {energy - breakdown['Total']:.6f}")
        
        print("-" * 80)
