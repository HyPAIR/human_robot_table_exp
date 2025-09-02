#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 22:22:39 2025

@author: toby
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Reproducibility
rng = np.random.RandomState(42)

# Training data
X = np.sort(5 * rng.rand(15, 1), axis=0)
y = np.sin(X).ravel() + rng.normal(0, 0.2, X.shape[0])

# Prediction points
X_pred = np.linspace(0, 5, 500).reshape(-1, 1)

# Try different length scales
length_scales = [0.2, 1.0, 3.0]

fig, axes = plt.subplots(len(length_scales), 2, figsize=(12, 9))

for i, l in enumerate(length_scales):
    # --- PRIOR ---
    kernel_prior = RBF(length_scale=l)
    gp_prior = GaussianProcessRegressor(kernel=kernel_prior)
    y_prior_samples = gp_prior.sample_y(X_pred, n_samples=3, random_state=rng)

    # --- POSTERIOR ---
    kernel_post = RBF(length_scale=l) + WhiteKernel(noise_level=0.2)
    gp_post = GaussianProcessRegressor(kernel=kernel_post, random_state=42)
    gp_post.fit(X, y)
    y_mean, y_std = gp_post.predict(X_pred, return_std=True)
    y_post_samples = gp_post.sample_y(X_pred, n_samples=3, random_state=rng)

    # Plot prior samples
    axes[i, 0].plot(X_pred, y_prior_samples, lw=1)
    axes[i, 0].set_title(f"Prior (length scale={l})")
    axes[i, 0].set_ylim(-3, 3)

    # Plot posterior mean + samples
    axes[i, 1].scatter(X, y, c="k", label="Observations")
    axes[i, 1].plot(X_pred, y_mean, "b", lw=2, label="Mean prediction")
    axes[i, 1].fill_between(
        X_pred.ravel(),
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        alpha=0.3,
        color="blue",
        label="95% CI"
    )
    for sample in y_post_samples.T:
        axes[i, 1].plot(X_pred, sample, "--", lw=1)
    axes[i, 1].set_title(f"Posterior (length scale={l})")
    axes[i, 1].set_ylim(-3, 3)

plt.tight_layout()
plt.show()
