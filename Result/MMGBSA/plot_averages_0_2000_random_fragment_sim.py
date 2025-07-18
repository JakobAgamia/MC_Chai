import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})


def pad_to_length(arrays):
    max_len = max(len(a) for a in arrays)
    return np.array([np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrays])

# MMGBSA values are often the same as the simulation does not always find a better ligand after every 200 steps


bromo_1 = [-10.34, -23.50, -23.50, -23.50, -22.13, -22.13, -19.37, -19.37, -19.37, -19.37, -28.13]
bromo_2 = [-10.51, -23.32, -23.32, -26.08, -26.08, -26.08, -26.08, -28.81, -28.81, -28.81, -29.15]
bromo_3 = [-10.54, -27.03, -28.74, -28.59, -28.59, -38.30, -38.30, -38.30, -38.30, -38.30, -28.20]
p38_1 = [-10.26, -24.48, -24.48, -24.48, -24.48, -24.48, -24.48, -24.48, -24.48, -24.48, -28.55]
p38_2 = [-20.37, -23.01, -23.01, -23.01, -22.50, -22.50, -19.25, -20.00, -15.90, -15.90, -27.71]
p38_3 = [-12.27, -17.46, -22.33, -22.33, -22.33, -22.33, -18.17, -18.17, -18.17, -18.17, -28.40]
serine_1 = [-1.27, -27.31, -26.03, -26.03, -38.71, -38.71, -29.313, -34.23, -34.23, -34.23, -30.68]
serine_2 = [-9.36, -33.32, -40.85, -40.85, -35.22, -33.95, -33.95, -25.90, -41.12, -41.12, -30.08]
serine_3 = [-15.24, -33.74, -37.30, -23.11, -31.12, -30.03, -30.03, -27.18, -27.18, -27.18, -32.27]

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
