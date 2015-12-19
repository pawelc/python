from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.umath import exp

# fun = lambda x,y: np.maximum(x,y)
# fun = lambda x,y: (2*x-1)*(2*y-1)
fun = lambda x,y: (x+1)*(exp(y)-1)/(x+2*exp(y)-1)

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = fun(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()