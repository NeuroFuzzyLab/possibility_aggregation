import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np
from matplotlib import pyplot as plt
from possibilities import PossibiltyDistribution as PD


pi_1 = PD([1., 1., 0.5])
pi_2 = PD([1., 0.5, 1.])

sup = pi_1 | pi_2
inf = pi_1 ^ pi_2

params = {"x": np.array([1, 2, 3]), 
		"align":"center",
		"tick_label": [f"$\omega_{i}$" for i in range(1, 4)],
		}
xlabel = "$\Omega$"
ylabels = ["$\pi_1$", "$\pi_2$", "$\pi_1 \land \pi_2$", "$\pi_1 \lor \pi_2$"]
heights = [pi.possibilities for pi in [pi_1, pi_2, inf, sup]]
colors = ["blue", "orange", "red", "green"]

fig, axs = plt.subplots(2, 2)

for i in range(2):
	for j in range(2):
		n = 2 * i + j
		axs[i, j].bar(height=heights[n], color=colors[n], **params)
		axs[i, j].set_xlabel(xlabel)
		axs[i, j].set_ylabel(ylabels[n])
		axs[i, j].set_yticks(np.array([0., 0.5, 1.]))


plt.tight_layout()
plt.show()
