import matplotlib.pyplot as plt
import numpy as np

# Parameters
steps = 100
true_energy = 0.5  # target convergence energy
noise_level = 0.1

# Generate noisy descending energy values
np.random.seed(42)  # for reproducibility
energy = np.zeros(steps)
energy[0] = 2  # starting energy

for i in range(1, steps):
    # Step descends towards true_energy + some noise
    energy[i] = energy[i-1] - (energy[i-1] - true_energy) * 0.05 + np.random.randn() * noise_level * 0.05

# Plot
plt.figure(figsize=(10,6))
plt.plot(range(steps), energy, color='royalblue', linewidth=2)
plt.xlabel('Step', fontsize=16, fontweight='bold')
plt.ylabel('Energy', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 2.2)

plt.show()
