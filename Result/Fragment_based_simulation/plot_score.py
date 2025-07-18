import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
# File paths and labels
file_paths = [
  "./Random_fragments/Serine/random_a1_log.txt",
    "./Random_fragments/Bromo/random_1_log.txt",
    "./Random_fragments/P38/random_1_log.txt",
]
labels = ['serine', 'bromodomain', 'p38']

# Containers for raw and averaged data
raw_data = []
avg_data = []

for file_path in file_paths:
    line_numbers = []
    scores = []

    with open(file_path, "r") as file:
        for i, line in enumerate(file, start=1):
            try:
                score_str = line.split("tensor([")[1].split("])")[0]
                score = float(score_str)
                line_numbers.append(i)
                scores.append(score)
            except (IndexError, ValueError):
                continue

    raw_data.append((line_numbers, scores))

    # Compute averages every 50 steps
    avg_x, avg_y = [0], [scores[0]]
    for i in range(1, len(scores), 50):
        chunk = scores[i-50:i+50]
        print(len(chunk))
        if chunk:
            avg = sum(chunk) / len(chunk)
            avg_x.append(line_numbers[i])  # center of chunk
            avg_y.append(avg)
    avg_data.append((avg_x, avg_y))

print(raw_data)
print(avg_data)
# Plot 1: Raw scores
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(raw_data):
    plt.plot(x, y, label=labels[i])
plt.xlabel("Step")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

# Plot 2: Averaged scores
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(avg_data):
    plt.plot(x, y, label=f"{labels[i]}", linewidth=2, linestyle='-', marker='o')
plt.xlabel("Step")
plt.ylabel("Average Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
