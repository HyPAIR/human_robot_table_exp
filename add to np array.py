#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:07:13 2025

@author: toby
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# --- Step 1: initial training data ---
X = np.array([[0.5], [1.5], [3.0]])   # shape (n_samples, 1)
y = np.array([0.0, 1.0, -0.5])        # shape (n_samples,)

# --- Step 2: add more data ---
X_new = np.array([[4.0], [2.0]])
y_new = np.array([0.5, 0.2])

# Concatenate to existing arrays
X = np.vstack([X, X_new])  # add new rows to X
y = np.hstack([y, y_new])  # add new values to y

# --- Step 3: Fit Gaussian Process ---
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.2)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp.fit(X, y)

# --- Step 4: Predict on a grid ---
X_pred = np.linspace(0, 5, 500).reshape(-1, 1)
y_mean, y_std = gp.predict(X_pred, return_std=True)

# --- Step 5: Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c="k", s=50, zorder=10, label="Training data")
plt.plot(X_pred, y_mean, "b", lw=2, label="GP mean prediction")
plt.fill_between(
    X_pred.ravel(),
    y_mean - 1.96 * y_std,
    y_mean + 1.96 * y_std,
    alpha=0.3,
    color="blue",
    label="95% confidence interval"
)
plt.title("Gaussian Process Regression (with added data)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
