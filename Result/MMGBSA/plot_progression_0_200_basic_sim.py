import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})
# MM/PBSA values
bromo = [-11.28, -25.53, -30.36, -35.75, -25.30, -33.32, -16.41, -26.65, -36.06, -32.06, -30.71]
p38 = [-11.42, -24.66, -19.71, -37.14, -40.64, -33.90, -19.72, -24.23, -31.59, -26.72, -26.21]
serine = [-8.74, -25.92, -31.00, -30.93, -27.11, -25.99, -21.76, -29.73, -26.72, -26.52, -27.31]

# X-axis: steps (every 20 from 0 to 200)
steps = list(range(0, 20 * len(bromo), 20))

plt.figure(figsize=(12, 6))
plt.plot(steps, bromo, label="Bromo", linestyle='-', marker='o')
plt.plot(steps, p38, label="P38", linestyle='-', marker='o')
plt.plot(steps, serine, label="Serine", linestyle='-', marker='o')

# Invert Y-axis (lower energy = better)
plt.gca().invert_yaxis()

# Labels and title
plt.xlabel("Step")
plt.ylabel("MMPBSA score (kcal/mol)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

