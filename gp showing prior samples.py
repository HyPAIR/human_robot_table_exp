#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 22:17:59 2025

@author: toby
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Set random seed for reproducibility
rng = np.random.RandomState(42)

# Kernel (without WhiteKernel, since prior has no noise)
kernel = RBF(length_scale=1.0)

# Prediction points
X_pred = np.linspace(0, 5, 500).reshape(-1, 1)

# --- PRIOR ---
gp_prior = GaussianProcessRegressor(kernel=kernel)
y_prior_samples = gp_prior.sample_y(X_pred, n_samples=5, random_state=rng)

# --- TRAINING DATA ---
X = np.sort(5 * rng.rand(15, 1), axis=0)
y = np.sin(X).ravel() + rng.normal(0, 0.2, X.shape[0])

# --- POSTERIOR ---
# Use RBF + WhiteKernel for noise modeling
kernel_post = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.2)
gp_post = GaussianProcessRegressor(kernel=kernel_post, random_state=42)
gp_post.fit(X, y)

y_mean, y_std = gp_post.predict(X_pred, return_std=True)
y_post_samples = gp_post.sample_y(X_pred, n_samples=5, random_state=rng)

# --- PLOTS ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot PRIOR
axes[0].plot(X_pred, y_prior_samples, lw=1)
axes[0].set_title("GP Prior Samples (before data)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# Plot POSTERIOR
axes[1].scatter(X, y, c="k", label="Observations")
axes[1].plot(X_pred, y_mean, "b", lw=2, label="Mean prediction")
axes[1].fill_between(
    X_pred.ravel(),
    y_mean - 1.96 * y_std,
    y_mean + 1.96 * y_std,
    alpha=0.3,
    color="blue",
    label="95% confidence interval"
)
for sample in y_post_samples.T:
    axes[1].plot(X_pred, sample, "--", lw=1)

axes[1].set_title("GP Posterior (after data)")
axes[1].set_xlabel("x")
axes[1].legend()

plt.tight_layout()
plt.show()
