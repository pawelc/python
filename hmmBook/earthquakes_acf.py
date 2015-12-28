import numpy as np
import statsmodels.tsa.stattools as st
import matplotlib.pyplot as plt

with open("../data/earthquakes.txt", "r") as infile:
    data = np.array([int(line.split()[1]) for line in infile], dtype='Float64')

data_acf = st.acf(data)

plt.bar(range(len(data_acf)),data_acf,width=0.08,edgecolor='None',color='k',align='center')
# plt.acorr(data)
plt.show()