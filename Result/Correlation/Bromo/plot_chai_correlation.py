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
        parts = line.strip().split(",")
        if len(parts) >= 3:
            y = float(parts[-2])
            x = float(parts[-1])
            y_values.append(y)
            x_values.append(-x)

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
    else:
        avg, std = np.nan, np.nan
    avg_x_per_interval.append(avg)
    std_x_per_interval.append(std)
    midpoints.append((lower + upper) / 2)

# Filter out NaNs for fitting
midpoints_np = np.array(midpoints)
avg_np = np.array(avg_x_per_interval)
valid = ~np.isnan(avg_np)
fit_coeffs = np.polyfit(midpoints_np[valid], avg_np[valid], 1)
fit_line = np.poly1d(fit_coeffs)

# Plot: Side-by-side subplots with shared y-axis
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
