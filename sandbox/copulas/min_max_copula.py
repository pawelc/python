from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.umath import exp

fun1 = lambda x,y: np.minimum(x,y)
fun2 = lambda x,y: np.maximum(x+y-1,0)

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z1 = fun1(X,Y)
Z2 = fun2(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z1, rstride=10, cstride=10)
ax.plot_wireframe(X, Y, Z2, rstride=10, cstride=10, color="red")

plt.show()