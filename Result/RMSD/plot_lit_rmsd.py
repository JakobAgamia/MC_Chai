import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

def read_values(file_path):
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    return values


file1 = "./rmsd_to_3u5l.txt"
file2 = "./rmsd_to_1di9.txt"
file3 = "./rmsd_to_5dia.txt"


data1 = read_values(file1)
data2 = read_values(file2)
data3 = read_values(file3)

# Generate line numbers
x1 = list(range(1, len(data1) + 1))
x2 = list(range(1, len(data2) + 1))
x3 = list(range(1, len(data3) + 1))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x1, data1, label='bromo', marker='o')
plt.plot(x2, data2, label='p38', marker='o')
plt.plot(x3, data3, label='serine', marker='o')

plt.xlabel('Different ligands')
plt.ylabel('RMSD (Å)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
