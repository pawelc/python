import autograd.numpy as np
from autograd.core import primitive

@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def make_grad_logsumexp(ans, x):
    def gradient_product(g):
        return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))
    return gradient_product

logsumexp.defgrad(make_grad_logsumexp)

from autograd import grad

def example_func(y):
    z = y**2
    lse = logsumexp(z)
    return np.sum(lse)

grad_of_example = grad(example_func)
print "Gradient: ", grad_of_example(np.array([1.5, 6.7, 1e-10]))