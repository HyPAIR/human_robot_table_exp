#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 22:12:11 2025

@author: toby
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# 1. Generate some sample data
rng = np.random.RandomState(42)
X = np.sort(5 * rng.rand(15, 1), axis=0)   # training inputs
y = np.sin(X).ravel() + rng.normal(0, 0.2, X.shape[0])  # noisy observations

# 2. Define kernel: RBF + WhiteKernel for noise
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.2)

# 3. Fit Gaussian Process
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp.fit(X, y)

# 4. Predict on a grid of points
X_pred = np.linspace(0, 5, 1000).reshape(-1, 1)
y_mean, y_std = gp.predict(X_pred, return_std=True)

# 5. Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c="k", label="Observations")
plt.plot(X_pred, y_mean, "b", label="Mean prediction")
plt.fill_between(
    X_pred.ravel(),
    y_mean - 1.96 * y_std,
    y_mean + 1.96 * y_std,
    alpha=0.3,
    color="blue",
    label="95% confidence interval"
)
plt.title("Gaussian Process Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()