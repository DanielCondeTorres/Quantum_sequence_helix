import matplotlib.pyplot as plt
import random

# Generate 6 random bitstrings of length 4
bitstrings = [''.join(random.choice(['0', '1']) for _ in range(4)) for _ in range(6)]

# Probabilities in a staircase pattern, normalized so the first is 1
probabilities = [0.7, 0.2, 0.08, 0.005, 0.0003, 0.0001]

# Plotting
plt.figure(figsize=(10,6))
bars = plt.bar(bitstrings, probabilities, color='cornflowerblue', edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for bar, prob in zip(bars, probabilities):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{prob:.2f}', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Styling
plt.xlabel('Bitstrings', fontsize=16, fontweight='bold')
plt.ylabel('Probability', fontsize=16, fontweight='bold')
plt.title('Histogram with Staircase Probabilities', fontsize=18, fontweight='bold')
plt.ylim(0, 1.2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
