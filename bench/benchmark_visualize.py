import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("results/jaro_winkler.csv")

df *= 1000 * 1000
df["length"] /= 1000 * 1000


ax=df.plot(x="length")

plt.xticks(list(range(0, 513, 64)))

plt.title("Performance comparision of the \nJaro-Winkler similarity in different libraries")
plt.xlabel("string length [in characters]")
plt.ylabel("runtime [μs]")
ax.set_xlim(xmin=0)
ax.set_ylim(bottom=0)
plt.grid()
plt.show()


