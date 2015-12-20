#mixture of gaussian reparametrization
import autograd.numpy as np
from scipy.misc import factorial
from scipy.stats import poisson
from autograd import grad
from autograd.core import primitive
import climin

COMPONENTS = 4

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

with open("../data/earthquakes.txt", "r") as infile:
    data = np.array([int(line.split()[1]) for line in infile], dtype='Float64')

def mixPoissonNegLogLikelihood(lambdaParams,mixingParams):
    logLikelihood = 0;
    for dataI in xrange(len(data)):
        dataSum = 0;
        for paramI in xrange(len(lambdaParams)):
            dataSum = dataSum + mixingParams[paramI] * poissonPmfWithDeriv(data[dataI],lambdaParams[paramI]);
        logLikelihood = logLikelihood + np.log(dataSum)

    return -logLikelihood

def mixPoissonNegLogLikelihoodUnconstraint(params):
    eta=params[0:COMPONENTS]
    tau=params[COMPONENTS:]
    return mixPoissonNegLogLikelihood(lambdaParams(eta),mixingParams(tau))

# params = np.random.randn(2*COMPONENTS-1)
params = 0.1*np.random.randn(2*COMPONENTS-1)
# params = np.array([0]*(2*COMPONENTS-1),dtype="Float64")

training_gradient_fun = grad(mixPoissonNegLogLikelihoodUnconstraint)

print "Initial loglikelihood:", mixPoissonNegLogLikelihoodUnconstraint(params)

# Use climin
opt = climin.Rprop(params, training_gradient_fun)
# opt = climin.RmsProp(params, training_gradient_fun, step_rate=1e-4, decay=0.9)
# opt = climin.Lbfgs(params,
#                    mixPoissonNegLogLikelihoodUnconstraint,
#                    training_gradient_fun)
# opt = climin.GradientDescent(params, training_gradient_fun, step_rate=0.001, momentum=.95)
# opt = climin.NonlinearConjugateGradient(params, mixPoissonNegLogLikelihoodUnconstraint, training_gradient_fun)

for info in opt:
    lambdaParameters = lambdaParams(params[0:COMPONENTS])
    mixingParameters = mixingParams(params[COMPONENTS:])
    print "loglikelihood:", mixPoissonNegLogLikelihoodUnconstraint(params) , ", lambdaParameters: ", lambdaParameters ,", mixingParameters: ", mixingParameters

# Use GD with constant step
# for i in xrange(1000000):
#     dparams = training_gradient_fun(params)
#     params-=dparams * 0.001
#     lambdaParameters = lambdaParams(params[0:COMPONENTS])
#     mixingParameters = mixingParams(params[COMPONENTS:])
#     print "loglikelihood:", mixPoissonNegLogLikelihoodUnconstraint(params) , ", lambdaParameters: ", lambdaParameters ,", mixingParameters: ", mixingParameters