import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_results(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 1. Parsear la secuencia y la Energía Total reportada en la línea de Rank
        rank_match = re.match(r'Rank\s+\d+:\s*([A-Z]+)\s*\|\s*Energy:\s*([-.\d]+)', line) 
        
        if rank_match:
            seq = rank_match.group(1)
            total_energy_reported = float(rank_match.group(2))
            
            # Inicializamos contribuciones y un valor para el offset
            contrib = {'seq': seq, 'total_energy_reported': total_energy_reported, 'offset': 0.0}
            
            i += 1 
            # 2. Parsear las contribuciones de términos usando REGEX flexible
            while i < len(lines) and not re.match(r'Rank\s+\d+:', lines[i].strip()):
                bline = lines[i].strip()
                
                # Expresión regular flexible para capturar cualquier término (incluido Constant Offset)
                tmatch = re.match(r'^\s*(.+?)\s*:\s*([-.\d]+)\s*$', bline)
                
                if tmatch:
                    key = tmatch.group(1).strip()
                    val = float(tmatch.group(2))
                    
                    if key == 'Constant Offset': 
                        contrib['offset'] = val
                    elif key != 'Total Energy': 
                        # Solo almacenamos términos variables para las barras
                        contrib[key] = val
                i += 1
            
            # 3. Calcular la Energía Total AJUSTADA para el plot (Suma de términos variables)
            contrib['total_energy_adjusted'] = total_energy_reported - contrib['offset']
            
            data.append(contrib)
        else:
            i += 1
            
    return data

# Detectar términos dinámicamente
def get_all_terms(data):
    terms_set = set()
    for d in data:
        for key in d.keys():
            # Excluimos todas las claves de control, dejando solo los términos variables
            if key not in ['seq', 'total_energy_reported', 'total_energy_adjusted', 'offset']:
                terms_set.add(key)
    return sorted(terms_set)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot top N lowest energy sequences from classical solver output.')
    parser.add_argument('-f', '--file', required=True, help='Path to the results.txt file')
    parser.add_argument('--top', type=int, default=100, help='Number of top sequences to plot (default: 100)')
    args = parser.parse_args()

    print(f"Parsing {args.file}...")
    data = parse_results(args.file)
    print(f"Parsed {len(data)} sequences.") 

    # Sort by total_energy_adjusted (lowest first)
    data_sorted = sorted(data, key=lambda x: x['total_energy_adjusted'])
    top_n = min(args.top, len(data_sorted)) 
    top_data = data_sorted[:top_n]

    # Detectar términos presentes
    all_terms = get_all_terms(top_data)
    print(f"Términos detectados (variables): {all_terms}")
    print(f"Plotting top {top_n} lowest energy sequences.")

    n = len(top_data)
    y_pos = np.arange(n)
    
    # Estilo 'ggplot' para compatibilidad y estética
    #plt.style.use('ggplot') 
    fig, ax = plt.subplots(figsize=(18, max(10, n * 0.25)))  

    # Colores distintos para cada término
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_terms)))

    # ESTRATEGIA: apilar desde 0 en ambas direcciones (bilateral)
    left_stack = np.zeros(n)   
    right_stack = np.zeros(n)  

    for j, term in enumerate(all_terms):
        values = np.array([d.get(term, 0.0) for d in top_data])
        
        # Separar valores negativos y positivos
        neg_values = np.minimum(values, 0)
        pos_values = np.maximum(values, 0)
        
        # Graficar negativos (hacia la izquierda desde left_stack)
        if np.any(neg_values < 0):
            ax.barh(y_pos, neg_values, left=left_stack, 
                   color=colors[j], label=term, 
                   height=0.8, alpha=0.8, edgecolor='none')
            
            left_stack += neg_values
        
        # Graficar positivos (hacia la derecha desde right_stack)
        if np.any(pos_values > 0):
            label = None if np.any(neg_values < 0) else term
            
            ax.barh(y_pos, pos_values, left=right_stack, 
                   color=colors[j], label=label,
                   height=0.8, alpha=0.8, edgecolor='none')
            
            right_stack += pos_values

    # Verificación de la Energía Ajustada (Suma de las barras)
    totals_adjusted = np.array([d['total_energy_adjusted'] for d in top_data])
    totals_reported = np.array([d['total_energy_reported'] for d in top_data])
    offsets = np.array([d['offset'] for d in top_data])
    final_sum = left_stack + right_stack
    
    if not np.allclose(final_sum, totals_adjusted, atol=1e-4): 
        print(f"\n⚠️  WARNING: Mismatch detectado en la energía ajustada! (Suma de Barras != Total Ajustado)")
        for i in range(min(5, n)):
            print(f"  {top_data[i]['seq']}: Suma de Barras={final_sum[i]:.6f}, Total Ajustado={totals_adjusted[i]:.6f}, Diff={final_sum[i] - totals_adjusted[i]:+.6f}")
    else:
        print("\n✅ Apilamiento verificado correctamente! (Suma de Barras $\\approx$ Total Ajustado)")

    # Línea vertical en x=0 (eje central)
    ax.axvline(x=0, color='gray', linewidth=1, linestyle='-', zorder=1)

    # El punto rojo es la energía AJUSTADA
    ax.scatter(totals_adjusted, y_pos, color='darkred', s=120, zorder=6, marker='D',
               edgecolors='black', linewidths=0.5, label='Adjusted Total Energy (Variable Terms Sum)')

    # Labels del eje Y
    seq_labels = [f"{d['seq']}" for d in top_data]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(seq_labels, rotation=0, va='center', fontsize=18, fontweight='bold') 
    
    # 1. Aumentar el tamaño de las etiquetas del eje X
    ax.tick_params(axis='x', labelsize=18)
    
    # 2. APLICACIÓN DE NEGRITA A LOS X-TICKS 
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    # Títulos y Ejes
    ax.set_xlabel('Energy Contributions', 
                  fontsize=20, fontweight='bold', labelpad=15)
                  
    # TÍTULO (sin Mathtext)
    ax.set_title(f'Top {top_n} Lowest Energy Sequences', 
                 fontsize=16, fontweight='bold', pad=20)

    # Leyenda sin duplicados Y FILTRADO DE 'Electrostatic terms'
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Filtramos la etiqueta que queremos quitar
    filtered_labels = []
    filtered_handles = []
    
    for label, handle in zip(by_label.keys(), by_label.values()):
        if label != 'Electrostatic terms':
            filtered_labels.append(label)
            filtered_handles.append(handle)

    #ax.legend(filtered_handles, filtered_labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Grid (solo en X)
    ax.grid(axis='x', alpha=0.5, linestyle=':', zorder=0)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)

    # Anotaciones de energía total (AJUSTADA)
    for idx in range(min(20, n)):
        # 1. Definir el texto solo como el valor Ajustado
        text_label = f"{totals_adjusted[idx]:.2f}"
        
        # 2. Posicionamiento: A LA MISMA ALTURA (sin offset_y)
        if totals_adjusted[idx] < 0:
            offset_x = -0.05  # A la izquierda del punto rojo
            ha = 'right'
        else:
            offset_x = 0.05   # A la derecha del punto rojo
            ha = 'left'
        
        x_position = totals_adjusted[idx] + offset_x
        y_position = y_pos[idx] 
        
        ax.text(x_position, y_position, text_label, 
                va='center', fontsize=14, fontweight='bold', ha=ha,
                color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    output_png = f"top{top_n}_final_energy_{args.file.split('.')[0]}.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_png}")
    plt.show()

    # --- Salida de Verificación ---
    print("\n🏆 Top 10 (Sorted by Adjusted Energy):")
    for i, d in enumerate(top_data[:10], 1):
        print(f"{i}. {d['seq']}: Adjusted={d['total_energy_adjusted']:.6f}, Reported={d['total_energy_reported']:.6f}, Offset={d['offset']:.6f}")
    
    print("\n🔍 Detailed Verification (first 3 sequences):")
    for i in range(min(3, n)):
        d = top_data[i]
        terms_sum = 0.0
        print(f"\n{d['seq']} (Total Ajustado (Bar End): {d['total_energy_adjusted']:.6f}, Reported: {d['total_energy_reported']:.6f}):")
        print(f"{'Contribution Term':35s}: {'Value':>8s}")
        print(f"{'-'*35}: {'-'*8}")
        
        for term in all_terms:
            val = d.get(term, 0.0)
            terms_sum += val
            print(f"  {term:33s}: {val:8.4f}")
            
        print(f"{'='*35}: {'='*8}")
        print(f"{'Total Sum of Variable Terms':33s}: {terms_sum:8.6f}")

        diff = d['total_energy_adjusted'] - terms_sum
        if abs(diff) > 1e-4:
            print(f"⚠️  Mismatch: Diff (Adjusted - Sum) = {diff:+.6f}")
        else:
            print(f"✅ Match: Diff (Adjusted - Sum) = {diff:+.6f}")