#dependant variables with covariance 0
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import mdentropy as md

x = np.random.normal(0,1,1000)
y = x**2
print np.corrcoef(x,y)
print spearmanr(x,y)

plt.plot(x,y,'bs')
plt.show()

