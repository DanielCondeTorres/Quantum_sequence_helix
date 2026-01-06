import os
import numpy as np

def save_energy_results(designer, result, solver_name, output_dir):
    """Guarda resultados detallados con Probabilidad y Desglose."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        energy_file = os.path.join(output_dir, "energy.txt")
        
        best_seq = result.get('repaired_sequence', result.get('sequence', 'N/A'))
        best_energy = result.get('repaired_cost', result.get('energy', 0.0))
        
        with open(energy_file, 'w') as f:
            f.write(f"=== ENERGY RESULTS - {solver_name.upper()} SOLVER ===\n")
            f.write(f"Solver: {solver_name}\n")
            f.write(f"Lowest Energy Sequence found: {best_seq} (Energy: {float(best_energy):.6f})\n")
            f.write("="*80 + "\n\n")
            
            # PREPARAR LISTA DE CANDIDATOS (Tuple: bs, seq, en, prob)
            candidates = []
            
            if 'classical_ranking' in result:
                # El clásico "dummy" probability es 1.0 para el primero, o N/A
                for bs, s, e, p in result['classical_ranking'][:50]:
                    candidates.append((bs, s, e, "N/A"))
            
            elif 'probs' in result and result['probs'] is not None:
                probs = result['probs']
                # Ordenamos por probabilidad (el VQE debe maximizar la prob del estado de menor energía)
                top_idx = np.argsort(probs)[-50:][::-1]
                for idx in top_idx:
                    p = probs[idx]
                    if p > 1e-5:
                        bs = format(idx, f'0{designer.n_qubits}b')
                        s = designer.decode_solution(bs)
                        try: e = designer.compute_energy_from_bitstring(bs)
                        except: e = 0.0
                        candidates.append((bs, s, e, f"{p:.6f}"))
            
            f.write(f"TOP {len(candidates)} SEQUENCES (Sorted by Probability/Rank):\n")
            f.write("-" * 80 + "\n")
            
            for rank, (bs, seq, en, prob) in enumerate(candidates, 1):
                f.write(f"Rank {rank:2d}: {seq} | Energy: {en:.6f} | Probability: {prob} | Bitstring: {bs}\n")
                
                try:
                    breakdown = designer.compute_energy_breakdown(bs)
                    f.write(f"         Energy Breakdown:\n")
                    for term, val in breakdown.items():
                        f.write(f"           {term}: {val:.6f}\n")
                except:
                    pass
                f.write("\n")
                
        print(f"✅ Reporte guardado con probabilidades: {energy_file}")
        
    except Exception as e:
        print(f"⚠️ Error al escribir energy.txt: {e}")

def print_top_sequences_table(designer, result):
    """Imprime tabla en consola."""
    print("\n" + "="*60)
    print(f"{'Rank':>4} | {'Secuencia':<10} | {'Energía':>10} | {'Probabilidad':>12}")
    print("-" * 60)
    
    if 'classical_ranking' in result:
        for i, (bs, s, e, p) in enumerate(result['classical_ranking'][:10], 1):
            print(f"{i:4d} | {s:<10} | {e:10.6f} | {'N/A':>12}")
            
    elif 'probs' in result and result['probs'] is not None:
        probs = result['probs']
        top_idx = np.argsort(probs)[-10:][::-1]
        for i, idx in enumerate(top_idx, 1):
            if probs[idx] < 1e-5: continue
            bs = format(idx, f'0{designer.n_qubits}b')
            s = designer.decode_solution(bs)
            e = designer.compute_energy_from_bitstring(bs)
            print(f"{i:4d} | {s:<10} | {e:10.6f} | {probs[idx]:12.6f}")
            
    print("="*60)