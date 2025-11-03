import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# Datos: propensión a hélice (tabla 1)
helix_propensity = {
    "Ala": 0.00, "Leu": 0.21, "Arg": 0.21, "Met": 0.24, "Lys": 0.26, "Gln": 0.39,
    "Glu": 0.40, "Ile": 0.41, "Trp": 0.49, "Ser": 0.50, "Tyr": 0.53, "Phe": 0.54,
    "Val": 0.61, "His": 0.61, "Asn": 0.65, "Thr": 0.66, "Cys": 0.68, "Asp": 0.69,
    "Gly": 1.00, "Pro": 3.15
}

# Datos: escala de hidrofobicidad (tabla 2)
hydrophobicity = {
    "Asp": -0.77, "Glu": -0.64, "Lys": -0.99, "Arg": -1.01, "His": 0.13, "Gly": 0.00,
    "Ala": 0.31, "Val": 1.22, "Leu": 1.70, "Ile": 1.80, "Pro": 0.72, "Met": 1.23,
    "Phe": 1.79, "Trp": 2.25, "Tyr": 0.96, "Thr": -0.04, "Ser": 0.26, "Cys": 1.54,
    "Asn": -0.60, "Gln": -0.22
}

# Lista de aminoácidos comunes a ambas tablas
amino_acids = list(helix_propensity.keys())

# Extraer valores de x (hidrofobicidad) y y (propensión a hélice)
x_vals = np.array([hydrophobicity[aa] for aa in amino_acids])
y_vals = np.array([helix_propensity[aa] for aa in amino_acids])

# --- Normalización Z-score ---
x_z = (x_vals - np.mean(x_vals)) / np.std(x_vals)
y_z = (y_vals - np.mean(y_vals)) / np.std(y_vals)

# Coordenadas en 2D (z-score)
coords = np.vstack([x_z, y_z]).T

# Calcular matriz de distancias euclídeas
dist_matrix = cdist(coords, coords, metric="euclidean")

# --- a) Graficar el mapa 2D ---
plt.figure(figsize=(8, 6))
plt.scatter(x_z, y_z, color="green")
for i, aa in enumerate(amino_acids):
    plt.text(x_z[i] + 0.05, y_z[i], aa, fontsize=9)
plt.axhline(0, color="grey", linewidth=0.8, linestyle="--")
plt.axvline(0, color="grey", linewidth=0.8, linestyle="--")
plt.xlabel("Hydrophobicity (Z-score)")
plt.ylabel("Helix propensity (Z-score)")
plt.title("Amino Acids in Z-score Space")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("amino_acids_zscore_map.png", dpi=300)
plt.show()

# --- b) Graficar la matriz de distancias con números más grandes ---
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(dist_matrix, cmap="viridis", interpolation="nearest")

# Colocar números encima de cada celda (fuente más grande)
for i in range(len(amino_acids)):
    for j in range(len(amino_acids)):
        ax.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", 
                color="w", fontsize=8, fontweight="bold")

# Ajustes de ejes y colorbar
plt.colorbar(im, ax=ax, label="Euclidean distance (Z-score space)")
ax.set_xticks(range(len(amino_acids)))
ax.set_xticklabels(amino_acids, rotation=90)
ax.set_yticks(range(len(amino_acids)))
ax.set_yticklabels(amino_acids)
ax.set_title("Pairwise Distance Matrix with Values")
plt.tight_layout()
plt.savefig("amino_acids_distance_matrix.png", dpi=300)
plt.show()

# --- c) Imprimir los 8 pares más cercanos ---
pairs = []
for i in range(len(amino_acids)):
    for j in range(i+1, len(amino_acids)):
        pairs.append(((amino_acids[i], amino_acids[j]), dist_matrix[i, j]))

# Ordenar por distancia
pairs_sorted = sorted(pairs, key=lambda x: x[1])

print("Top 8 pares de aminoácidos más cercanos en el espacio z-score:")
for k in range(8):
    (aa1, aa2), dist = pairs_sorted[k]
    print(f"{aa1} - {aa2}: {dist:.3f}")
