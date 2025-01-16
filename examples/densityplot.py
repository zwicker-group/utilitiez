import matplotlib.pyplot as plt
import numpy as np

from utilitiez import densityplot

x = np.geomspace(1, 1e2, 6)
y = np.linspace(0, 9, 4)
data = np.random.uniform(size=(len(x), len(y)))

densityplot(data, x, y, vmin=0, vmax=1)
plt.colorbar()
plt.show()
