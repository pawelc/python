import numpy as np

np.random.seed(2) # for reproducibility
e = np.random.normal(30, 3, 50)
F = np.random.normal(1000, e)

w = 1. / e ** 2
F_hat = np.sum(w * F) / np.sum(w)
sigma_F = w.sum() ** -0.5
print sigma_F