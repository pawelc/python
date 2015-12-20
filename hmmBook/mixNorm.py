#mixture of gaussian reparametrization so we can run unconstrained optimization

import autograd.numpy as np
from autograd import grad
from autograd.scipy.stats import norm
import climin

COMPONENTS = 4

varianceFromUnc = lambda varUnconstr:np.exp(varUnconstr)
mixingParams2_n = lambda tau:np.exp(tau)/(1+np.sum(np.exp(tau)))
mixingParams = lambda tau: np.concatenate((np.array([1-sum(mixingParams2_n(tau))]) , mixingParams2_n(tau)))

data = np.array([-0.39, 0.12 ,0.94 ,1.67 ,1.76 ,2.44 ,3.72 ,4.28 ,4.92 ,5.53, 0.06, 0.48 ,1.01 ,1.68 ,1.80 ,3.25 ,4.12 ,4.60 ,5.28 ,6.22],dtype="Float64");

def mixNormNegLogLikelihood(meanParams,varianceParams,mixingParams):
    logLikelihood = 0;
    for dataI in xrange(len(data)):
        dataSum = 0;
        for paramI in xrange(len(meanParams)):
            dataSum = dataSum + mixingParams[paramI] * norm.pdf(data[dataI], meanParams[paramI], varianceParams[paramI]);
        logLikelihood = logLikelihood + np.log(dataSum)

    return -logLikelihood

def mixNormNegLogLikelihoodUnconstraint(params):
    meanParams = params[0:COMPONENTS]
    varUnconstr = params[COMPONENTS:2*COMPONENTS]
    tau=params[2*COMPONENTS:]
    return mixNormNegLogLikelihood(meanParams, varianceFromUnc(varUnconstr),mixingParams(tau))

# params = np.random.randn(2*COMPONENTS-1)
params = 0.1*np.random.randn(3*COMPONENTS-1)
# params = np.array([0]*(2*COMPONENTS-1),dtype="Float64")

training_gradient_fun = grad(mixNormNegLogLikelihoodUnconstraint)

print "Initial loglikelihood:", mixNormNegLogLikelihoodUnconstraint(params)

# Use climin
opt = climin.Rprop(params, training_gradient_fun)
# opt = climin.RmsProp(params, training_gradient_fun, step_rate=1e-4, decay=0.9)
# opt = climin.Lbfgs(params,
#                    mixPoissonNegLogLikelihoodUnconstraint,
#                    training_gradient_fun)
# opt = climin.GradientDescent(params, training_gradient_fun, step_rate=0.001, momentum=.95)
# opt = climin.NonlinearConjugateGradient(params, mixPoissonNegLogLikelihoodUnconstraint, training_gradient_fun)

for info in opt:
    meanParams = params[0:COMPONENTS]
    var = varianceFromUnc(params[COMPONENTS:2*COMPONENTS])
    mixingParameters=mixingParams(params[2*COMPONENTS:])
    print "loglikelihood:", mixNormNegLogLikelihoodUnconstraint(params) , ", meanParams: ", meanParams ,", var: ",var,", mixingParameters: ", mixingParameters

# Use GD with constant step
# for i in xrange(1000000):
#     dparams = training_gradient_fun(params)
#     params-=dparams * 0.001
#     lambdaParameters = lambdaParams(params[0:COMPONENTS])
#     mixingParameters = mixingParams(params[COMPONENTS:])
#     print "loglikelihood:", mixPoissonNegLogLikelihoodUnconstraint(params) , ", lambdaParameters: ", lambdaParameters ,", mixingParameters: ", mixingParameters