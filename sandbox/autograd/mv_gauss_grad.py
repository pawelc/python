import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
import autograd.scipy.stats.multivariate_normal as mvn


inputs = np.array([[1,1],[1.1,2],[2,3],[1,3],[4,1]])


def training_loss(params):
    # Training loss is the negative log-likelihood of the training labels.
    return np.sum(mvn.logpdf(inputs, params[0:2], np.reshape(params[2:], (2,2))))

# plot logistic
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x=np.linspace(-2, 2, 100);
# ax.plot(x, sigmoid(x))
# plt.show()




# Define a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.array([0.0, 0, 1,0,0,1])
print "Initial LL:", training_loss(weights)
for i in xrange(1000000):
    weights += training_gradient_fun(weights) * 0.01
    print "LL:", training_loss(weights)

print  "Trained LL:", training_loss(weights)