# This file is a slightly edited version of this: 
# https://docs.gpytorch.ai/en/v1.6.0/examples/01_Exact_GPs/Simple_GP_Regression.html
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pickle
import os
import math

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

regions = ['Boise_river',  'Clearwater', 'Clearwater_Canyon_Ranger', 
              'Flathead', 'North_Santiam',
              'South_Fork_clearwater', 'Yakima']
region = regions[6]
path_suffix = os.getcwd()
with open(path_suffix+'/latent_gp/latent_'+ region +'.pickle', 'rb') as handle:
    gp_results = pickle.load(handle)
train_x = gp_results['latent_train'] 
test_x = gp_results['latent_test'] 
train_y = gp_results['y_train']
test_y = gp_results['y_test']

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # "Loss" for GPs - the marginal log likelihood
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

training_iter = 50
model.train()
likelihood.train()
for i in range(training_iter):
    optimizer.zero_grad() # Zero gradients from previous iteration
    output = model(train_x) # Output from model
    loss = -mll(output, train_y) # Calc loss and backprop gradients
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
    
model.eval()
likelihood.eval()
# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    y_pred = observed_pred.mean
    y_lower, y_upper = observed_pred.confidence_region()
s = 2
