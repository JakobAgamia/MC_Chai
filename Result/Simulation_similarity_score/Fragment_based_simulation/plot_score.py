import matplotlib.pyplot as plt
import glob
plt.rcParams.update({'font.size': 16})
# Get all text files (modify pattern if needed)
file_list = ['./Results/6fnx_prob_add_remove_5_log.txt', './Results/4lzs_prob_add_remove_5_log.txt']  # Assumes all text files are in the same directory

plt.figure(figsize=(10, 6))  # Set figure size
labels = ['4lzs', '6fnx']
# Loop through each file and plot the data
i = 0
for file in file_list:
    scores = []

    with open(file, "r") as f:
        for line in f:
            if "Score:" in line:
                score = float(line.split("Score:")[1].strip())  # Extract Score value
                scores.append(score)

    plt.plot(range(1, 1001), scores[:1000], label=labels[i])
    i += 1

# Customize plot
plt.xlabel("Step")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid(True)

# Show plot
plt.show()
