import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

df = pd.read_csv("rmsd_data.csv")

plt.plot(df["Step"], df["P38"], label="P38")
plt.plot(df["Step"], df["Bromo"], label="Bromo")
plt.plot(df["Step"], df["Serine"], label="Serine")
plt.xlabel("Step")
plt.ylabel("RMSD (Å)")
plt.legend()
plt.grid(True)
plt.show()
