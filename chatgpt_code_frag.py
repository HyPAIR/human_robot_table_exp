#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:44:21 2025

@author: toby
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --- 1. Generate sample training data ---
# Function: y = sin(x)
X_train = np.linspace(0, 10, 10).reshape(-1, 1)
y_train = np.sin(X_train).ravel()

# --- 2. Define kernel and Gaussian Process model ---
# C(1.0) * RBF(length_scale=1.0) is a common default choice
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# --- 3. Fit the model to the training data ---
gp.fit(X_train, y_train)

# --- 4. Define new test points where we want predictions ---
X_test = np.linspace(0, 10, 100).reshape(-1, 1)

# --- 5. Get predictions from the trained GP ---
# return_std=True gives you uncertainty estimates
y_mean, y_std = gp.predict(X_test, return_std=True)

# --- 6. Access learned kernel parameters ---
print("Optimized kernel:", gp.kernel_)

# --- 7. Example: print first few predicted values ---
for i in range(5):
    print(f"x={X_test[i,0]:.2f}, mean={y_mean[i]:.3f}, std={y_std[i]:.3f}")