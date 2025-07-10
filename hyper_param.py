#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:45:49 2025

@author: toby
"""

# smoke_test (this makes sure this example notebook gets tested)

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))
    

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
    
    
    # We will use the simplest form of GP model, exact inference
class FancyGPWithPriors(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(FancyGPWithPriors, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
        outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
            ),
            outputscale_prior=outputscale_prior
        )

        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
        self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')
    
raw_outputscale = model.covar_module.raw_outputscale
print('raw_outputscale, ', raw_outputscale)

# Three ways of accessing the raw outputscale constraint
print('\nraw_outputscale_constraint1', model.covar_module.raw_outputscale_constraint)

printmd('\n\n**Printing all model constraints...**\n')
for constraint_name, constraint in model.named_constraints():
    print(f'Constraint name: {constraint_name:55} constraint = {constraint}')

printmd('\n**Getting raw outputscale constraint from model...**')
print(model.constraint_for_parameter_name("covar_module.raw_outputscale"))


printmd('\n**Getting raw outputscale constraint from model.covar_module...**')
print(model.covar_module.constraint_for_parameter_name("raw_outputscale"))



raw_outputscale = model.covar_module.raw_outputscale
constraint = model.covar_module.raw_outputscale_constraint

print('Transformed outputscale', constraint.transform(raw_outputscale))
print(constraint.inverse_transform(constraint.transform(raw_outputscale)))
print(torch.equal(constraint.inverse_transform(constraint.transform(raw_outputscale)), raw_outputscale))

print('Transform a bunch of negative tensors: ', constraint.transform(torch.tensor([-1., -2., -3.])))




# Recreate model to reset outputscale
model = ExactGPModel(train_x, train_y, likelihood)

# Inconvenient way of getting true outputscale
raw_outputscale = model.covar_module.raw_outputscale
constraint = model.covar_module.raw_outputscale_constraint
outputscale = constraint.transform(raw_outputscale)
print(f'Actual outputscale: {outputscale.item()}')

# Inconvenient way of setting true outputscale
model.covar_module.raw_outputscale.data.fill_(constraint.inverse_transform(torch.tensor(2.)))
raw_outputscale = model.covar_module.raw_outputscale
outputscale = constraint.transform(raw_outputscale)
print(f'Actual outputscale after setting: {outputscale.item()}')


# Recreate model to reset outputscale
model = ExactGPModel(train_x, train_y, likelihood)

# Convenient way of getting true outputscale
print(f'Actual outputscale: {model.covar_module.outputscale}')

# Convenient way of setting true outputscale
model.covar_module.outputscale = 2.
print(f'Actual outputscale after setting: {model.covar_module.outputscale}')



print(f'Actual noise value: {likelihood.noise}')
print(f'Noise constraint: {likelihood.noise_covar.raw_noise_constraint}')


# Registers a prior on the sqrt of the noise parameter
# (e.g., a prior for the noise standard deviation instead of variance)
likelihood.noise_covar.register_prior(
    "noise_std_prior",
    gpytorch.priors.NormalPrior(0, 1),
    lambda module: module.noise.sqrt()
)

# Create a GaussianLikelihood with a normal prior for the noise
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    
    noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
    noise_prior=gpytorch.priors.NormalPrior(0, 1)
)
    


likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-2),
)

model = FancyGPWithPriors(train_x, train_y, likelihood)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(1.),
    'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
    'covar_module.outputscale': torch.tensor(2.),
}

model.initialize(**hypers)
print(
    model.likelihood.noise_covar.noise.item(),
    model.covar_module.base_kernel.lengthscale.item(),
    model.covar_module.outputscale.item()
)

