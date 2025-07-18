import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})


def pad_to_length(arrays):
    max_len = max(len(a) for a in arrays)
    return np.array([np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrays])

# MMGBSA values are often the same as the simulation does not always find a better ligand after every 200 steps, one value could not
# be calculated because antechamber failed to create force field parameters for the ligand, the value with the average of the other two

bromo_1 = [-12.24, -34.21, -34.21, -31.59, -30.67, -30.67, -30.67, -30.67, -29.28, -29.28, -29.70]
bromo_2 = [-15.37, -35.82, -36.87, -36.87, -36.87, -39.20, -39.20, -39.20, -39.20, -39.20, -28.22]
bromo_3 = [-7.67, -22.01, -22.01, -32.75, -32.75, -32.75, -30.07, -30.07, -30.07, -30.07, -28.01]
p38_1 = [-4.22, -24.10, -26.51, -39.25, -39.25, -39.25, -39.25, -41.38, -41.38, -41.38, -29.93]
p38_2 = [-12.23, -46.05, -38.35, -52.60, -52.60, -29.89, -31.01, -31.01, -31.01, -31.01, -41.96]
p38_3 = [-7.80, -(46.05 + 24.10)/2, -22.33, -46.87, -46.87, -46.87, -46.87, -46.87, -46.87, -46.87, -38.67]
serine_1 = [-14.06, -40.37, -40.37, -40.37, -40.37, -40.37, -40.37, -40.37, -40.37, -40.37, -44.70]
serine_2 = [-1.03, -33.32, -27.87, -27.87, -27.87, -27.87, -27.87, -27.87, -27.87, -27.87, -29.11]
serine_3 = [-14.71, -19.87, -19.87, -19.87, -20.74, -22.25, -22.25, -22.25, -22.25, -22.25, -21.06]

print(len(serine_1), len(serine_2), len(serine_3))
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
