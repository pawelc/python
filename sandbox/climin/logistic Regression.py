import numpy as np
import climin.initialize
import gzip
import cPickle

# load data
datafile = 'mnist.pkl.gz'
# Load data.
with gzip.open(datafile,'rb') as f:
    train_set, val_set, test_set = cPickle.load(f)

X, Z = train_set
VX, VZ = val_set
TX, TZ = test_set

def one_hot(arr):
    result = np.zeros((arr.shape[0], 10))
    result[xrange(arr.shape[0]), arr] = 1.
    return result

Z = one_hot(Z)
VZ = one_hot(VZ)
TZ = one_hot(TZ)


n_inpt=784
n_output=10

def unpack_parameters(pars):
    w = pars[:n_inpt * n_output].reshape((n_inpt, n_output))
    b = pars[n_inpt * n_output:].reshape((1, n_output))
    return w, b

def predict(parameters, inpt):
    w, b = unpack_parameters(parameters)
    before_softmax = np.dot(inpt, w) + b
    softmaxed = np.exp(before_softmax - before_softmax.max(axis=1)[:, np.newaxis])
    return softmaxed / softmaxed.sum(axis=1)[:, np.newaxis]

def loss(parameters, inpt, targets):
    predictions = predict(parameters, inpt)
    loss = -np.log(predictions) * targets
    return loss.sum(axis=1).mean()

def d_loss_wrt_pars(parameters, inpt, targets):
    p = predict(parameters, inpt)
    g_w = np.dot(inpt.T, p - targets) / inpt.shape[0]
    g_b = (p - targets).mean(axis=0)
    return np.concatenate([g_w.flatten(), g_b])


wrt = np.empty(7850)
climin.initialize.randomize_normal(wrt, 0, 1)

import itertools
args = itertools.repeat(([X, Z], {}))

import climin
opt = climin.GradientDescent(wrt, d_loss_wrt_pars, step_rate=0.1, momentum=.95, args=args)

print loss(wrt, VX, VZ)   # prints something like 2.49771627484
for info in opt:
    if info['n_iter'] >= 100:
        break
print loss(wrt, VX, VZ)   # prints something like 0.324243334583