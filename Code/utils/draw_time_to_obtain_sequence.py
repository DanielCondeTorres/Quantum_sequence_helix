import matplotlib.pyplot as plt

# Provided data
sequences_length = [2, 4, 6, 8]
time = [2.1, 532, 1523, 9390]

# --- Aesthetic Configuration ---
# Using 'ggplot' style for a clean, white background
plt.style.use('ggplot') 
fig, ax = plt.subplots(figsize=(10, 6))

# Ensure the figure area outside the plot is also white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# --- Main Plot ---
# Scatter plot with distinctive markers and connecting lines
ax.plot(sequences_length, time, 
        marker='D',          # Diamond marker for distinction
        markersize=10,       
        markerfacecolor='tab:red', # Red marker fill
        markeredgecolor='black', # Black border
        markeredgewidth=1.0, 
        linestyle='--',      # Dashed line
        linewidth=2,         
        color='tab:red',     
        label='Execution Time'
       )

# --- Labels and Titles ---
ax.set_xlabel('Sequence Length', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Execution Time vs. Sequence Length', fontsize=16, fontweight='bold', pad=20)

# --- Ticks and Grid ---
ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
ax.grid(True, linestyle=':', alpha=0.6) # Lighter grid

# --- Annotate values on points (for clarity) ---
for i, txt in enumerate(time):
    ax.annotate(f'{txt:.1f}s', 
                (sequences_length[i], time[i]), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=10, 
                fontweight='bold',
                # White box around text for visibility
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.9))

# --- Legend ---
ax.legend(fontsize=12, frameon=True, shadow=True, fancybox=True, loc='upper left')

# --- Adjust layout and save ---
plt.tight_layout()
output_filename = 'time_vs_sequence_white_bg.png'
plt.savefig(output_filename, dpi=300, facecolor='white') # Explicitly save with a white background
plt.show() 
print(f"Plot saved as '{output_filename}'")