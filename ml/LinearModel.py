import numpy as np
np.random.seed(42) # for repeatability
theta_true = (25, 0.5)
xdata = 100 * np.random.random(20)
ydata = theta_true[0] + theta_true[1] * xdata
ydata = np.random.normal(ydata, 10) # add error

X = np.vstack([np.ones_like(xdata), xdata]).T
theta_hat = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, ydata))
y_hat = np.dot(X, theta_hat)
sigma_hat = np.std(ydata - y_hat)
Sigma = sigma_hat ** 2 * np.linalg.inv(np.dot(X.T, X))

import statsmodels.api as sm
X = sm.add_constant(xdata)
result = sm.OLS(ydata, X).fit()
sigma_hat = result.params
Sigma = result.cov_params()
print(result.summary2())