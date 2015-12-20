#mixture of gaussian reparametrization
import autograd.numpy as np
from scipy.misc import factorial
from scipy.stats import poisson
from autograd import grad
from autograd.core import primitive

COMPONENTS = 3

lambdaParams = lambda eta:np.exp(eta)
mixingParams2_n = lambda tau:np.exp(tau)/(1+np.sum(np.exp(tau)))
mixingParams = lambda tau: np.concatenate((np.array([1-sum(mixingParams2_n(tau))]) , mixingParams2_n(tau)))

#poison not in autograd so have to define derivative
@primitive
def poissonPmfWithDeriv(x,mu):
    return poisson.pmf(x,mu)

def make_grad_poisson(ans, x, mu):
    def gradient_product(g):
        return g * (- ans +  x*np.exp(-mu)*np.power(mu,x-1)/factorial(x))
    return gradient_product

poissonPmfWithDeriv.defgrad(make_grad_poisson,argnum=1)

# have to vectorize this function to be able computer outer of inputs
poissonPdfVec = np.vectorize(poissonPmfWithDeriv);

#autograd log is not vectorized
logVec = np.vectorize(np.log);

with open("../data/earthquakes.txt", "r") as infile:
    data = np.array([int(line.split()[1]) for line in infile], dtype='Float64')

def mixPoissonLogLikelihood(lambdaParams,mixingParams):
    return np.sum(logVec(np.inner(poissonPdfVec(data[:,None],lambdaParams),mixingParams)))

def mixPoissonLogLikelihoodUnconstraint(params):
    eta=params[0:COMPONENTS]
    tau=params[COMPONENTS:]
    return mixPoissonLogLikelihood(lambdaParams(eta),mixingParams(tau))

params = np.concatenate((np.array([2,2,2],dtype='Float64'),np.array([0,0],dtype='Float64')))

training_gradient_fun = grad(mixPoissonLogLikelihoodUnconstraint)

print "Initial loglikelihood:", mixPoissonLogLikelihoodUnconstraint(params)
for i in xrange(1000000):
    dparams = training_gradient_fun(params)
    params+=dparams * 0.001
    print "loglikelihood:", mixPoissonLogLikelihoodUnconstraint(params)