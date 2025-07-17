import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 17})
# File path (update with your actual file path)
file_path = "chai_score_check.txt"

# Lists to store extracted values
x_values = []
y_values = []

# Read the file and extract numbers
with open(file_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")  # Split by commas
        if len(parts) >= 3:  # Ensure valid line
            y = float(parts[-2])  # Second last number as y
            x = float(parts[-1])  # Last number as x
            y_values.append(y)
            x_values.append(-x)

# Define y intervals
y_intervals = np.arange(0.6, 1, 0.05)

# Compute averages and standard deviations
avg_x_per_interval = []
std_x_per_interval = []
midpoints = []

for i in range(len(y_intervals) - 1):
    lower, upper = y_intervals[i], y_intervals[i + 1]
    x_in_bin = [x for x, y in zip(x_values, y_values) if lower <= y < upper]
    if x_in_bin:
        avg = np.mean(x_in_bin)
        std = np.std(x_in_bin)
        avg_x_per_interval.append(avg)
        std_x_per_interval.append(std)
        midpoints.append((lower + upper) / 2)
    else:
        avg_x_per_interval.append(None)
        std_x_per_interval.append(None)
        midpoints.append((lower + upper) / 2)

# Filter out None values for fitting
avg_x_per_interval1, midpoints1, std_x_per_interval1 = zip(*[
    (x, y, z) for x, y, z in zip(avg_x_per_interval, midpoints, std_x_per_interval) if x is not None
])

# Convert back to lists
avg_x_per_interval = list(avg_x_per_interval1)
midpoints = list(midpoints1)
std_x_per_interval = list(std_x_per_interval1)


for i in range(len(avg_x_per_interval)):
    if avg_x_per_interval[i]:
        avg_x_per_interval[i] = float(avg_x_per_interval[i])
midpoints_np = np.array(midpoints)
avg_np = np.array([val if val is not None else np.nan for val in avg_x_per_interval])
print(avg_np, type(avg_np))
valid = ~np.isnan(avg_np)
fit_coeffs = np.polyfit(midpoints_np[valid], avg_np[valid], 1)
fit_line = np.poly1d(fit_coeffs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Left plot: Raw data
ax1.scatter(y_values, x_values, color="blue", alpha=0.7)
ax1.set_xlabel("Chai-1 confidence score")
ax1.set_ylabel("Binding affinities (kcal·mol$^{-1}$)")
ax1.grid(True)
ax1.invert_yaxis()

# Right plot: Averages and linear fit
ax2.errorbar(midpoints_np, avg_x_per_interval, yerr=std_x_per_interval,
             fmt='o', color='blue', ecolor='gray', capsize=5, label='Average ± SD')
x_fit = np.linspace(min(midpoints_np), max(midpoints_np), 100)
ax2.plot(x_fit, fit_line(x_fit), color='red', linestyle='-', label=f'Fit: y = {fit_coeffs[0]:.2f}x + {fit_coeffs[1]:.2f}')
ax2.set_xlabel("Chai-1 confidence score")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
