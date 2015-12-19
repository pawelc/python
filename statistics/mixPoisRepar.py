#mixture of gaussian reparametrization
import numpy as np

def mixPoissonLogLikelihood(x,lambdaParams,dpois,mixingParams):
    np.sum(np.log(np.outer(x,lambdaParams,dpois)*mixingParams))


with open("../data/earthquakes.txt", "r") as infile:
    data = [int(line.split()[1]) for line in infile]


lambdaParams = lambda eta:np.exp(eta)
mixingParams2_n = lambda tau:np.exp(tau)/(1+np.sum(np.exp(tau)))
mixingParams = lambda tau: [1-sum(mixingParams2_n(tau))] + mixingParams2_n(tau);


print mixPoissonLogLikelihood(data, lambdaParams, mixingParams)