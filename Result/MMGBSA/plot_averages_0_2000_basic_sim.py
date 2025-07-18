import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

def pad_to_length(arrays):
    max_len = max(len(a) for a in arrays)
    return np.array([np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrays])

# MMGBSA values are often the same as the simulation does not always find a better ligand after every 200 steps


bromo_1 = [-11.27, -30.71, -31.26, -30.04, -23.89, -29.96, -32.14, -50.16, -34.00, -41.88, -27.74]
bromo_2 = [-11.27, -19.85, -29.32, -36.34, -25.83, -25.83, -25.83, -25.83, -25.83, -25.83, -29.79]
bromo_3 = [-11.27, -19.19, -27.27, -22.46, -22.46, -22.46, -23.12, -32.01, -32.01, -32.01, -19.19]
p38_1 = [-11.42, -26.21, -22.99, -22.87, -30.66, -25.71, -37.07, -38.80, -36.90, -28.35, -40.20]
p38_2 = [-11.42, -21.44, -31.24, -29.82, -41.27, -37.75, -37.75, -37.75, -37.75, -38.56, -41.73]
p38_3 = [-11.42, -39.72, -35.22, -32.02, -34.15, -34.15, -34.15, -34.15, -34.15, -34.15, -34.01]
serine_1 = [-8.74, -27.31, -26.03, -26.03, -38.71, -38.71, -29.313, -34.23, -34.23, -34.23, -34.23]
serine_2 = [-8.74, -33.32, -40.85, -40.85, -35.22, -33.95, -33.95, -25.90, -41.12, -41.12, -32.54]
serine_3 = [-8.74, -33.74, -37.30, -23.11, -31.12, -30.03, -30.03, -27.18, -27.18, -27.18, -31.55]

bromo_avg = np.nanmean(pad_to_length([bromo_1, bromo_2, bromo_3]), axis=0)
p38_avg = np.nanmean(pad_to_length([p38_1, p38_2, p38_3]), axis=0)
serine_avg = np.nanmean(pad_to_length([serine_1, serine_2, serine_3]), axis=0)
steps = np.arange(len(bromo_avg)) * 200
plt.figure(figsize=(10, 6))
plt.plot(steps, bromo_avg, label='Bromo', marker='o')
plt.plot(steps, p38_avg, label='P38', marker='o')
plt.plot(steps, serine_avg, label='Serine', marker='o')

plt.xlabel("Step")
plt.ylabel("MMPBSA score (kcal/mol)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
