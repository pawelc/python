#mixture of gaussian reparametrization
import numpy as np
from scipy.stats import poisson
from autograd import grad

lambdaParams = lambda eta:np.exp(eta)
mixingParams2_n = lambda tau:np.exp(tau)/(1+np.sum(np.exp(tau)))
mixingParams = lambda tau: [1-sum(mixingParams2_n(tau))] + mixingParams2_n(tau);

with open("../data/earthquakes.txt", "r") as infile:
    data = [int(line.split()[1]) for line in infile]

def mixPoissonLogLikelihood(lambdaParams,mixingParams):
    np.sum(np.log(np.inner(poisson.pmf(np.outer(data,lambdaParams)),mixingParams)))

def mixPoissonLogLikelihoodUnconstraint(eta,tau):
    mixPoissonLogLikelihood(lambdaParams(eta),mixingParams(tau))

eta=np.array([0,0])
tau=np.array([0,0])

training_gradient_fun = grad(mixPoissonLogLikelihoodUnconstraint)

print "Initial loglikelihood:", mixPoissonLogLikelihoodUnconstraint(eta,tau)
for i in xrange(1000000):
    deta,dtau = training_gradient_fun(eta,tau)
    eta+=deta * 0.01
    tau+=dtau * 0.01
    print "loglikelihood:", mixPoissonLogLikelihoodUnconstraint(eta,tau)