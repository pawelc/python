import autograd.numpy as np
from autograd.util import quick_grad_check

x = np.array([4., 5.])

def my_fun(a):
    return a[0] * np.power(x[0], 3) + a[1] * np.power(x[1], 2)

def my_fun2(a):
    outter = np.dot(a.reshape(2,1),x.reshape(1,2))
    np.fill_diagonal(outter,1)
    return np.sum(outter)

quick_grad_check(my_fun2, np.array([1., 2.]))