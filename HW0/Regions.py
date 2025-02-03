import numpy as np
import matplotlib.pyplot as plt
import os


# Making a folder for the plots to go to.
os.mkdir('./RegionPlots')

# Region 1
theta = np.linspace(0, 2 * np.pi, 200)
x = np.cos(theta)
y = np.sin(theta)
plt.figure()
plt.plot(x, y, label=r'$\|x\|_2 = 1$ boundary')
plt.fill(x, y, alpha=0.2)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(r'Region for $\|x\|_2 \leq 1$ (Unit Disk)')
plt.legend()
plt.savefig('./RegionPlots/Region1.pdf')
plt.show()

# Region 2
X = np.array([1, 0, -1, 0, 1])
Y = np.array([0, 1, 0, -1, 0])
plt.figure()
plt.plot(X, Y, label=r'$\|x\|_1 = 1$ boundary')
plt.fill(X, Y, alpha=0.2)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(r'Region for $\|x\|_1 \leq 1$ (Diamond)')
plt.legend()
plt.savefig('./RegionPlots/Region2.pdf')
plt.show()

# Region 3
X = np.array([1, 1, -1, -1, 1])
Y = np.array([1, -1, -1, 1, 1])
plt.figure()
plt.plot(X, Y, label=r'$\|x\|_\infty = 1$ boundary')
plt.fill(X, Y, alpha=0.2)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(r'Region for $\|x\|_1 \leq 1$ (Square)')
plt.legend()
plt.savefig('./RegionPlots/Region3.pdf')
plt.show()