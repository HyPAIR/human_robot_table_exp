#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:20:38 2025

@author: toby
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# --- Step 1: Generate some 2D training data ---
rng = np.random.RandomState(42)
X = rng.uniform(0, 5, (20, 2))  # 20 points in 2D (x1, x2)
# true function: sin(x1) * cos(x2), plus noise
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + rng.normal(0, 0.1, X.shape[0])

# --- Step 2: Fit Gaussian Process ---
kernel = RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp.fit(X, y)

# --- Step 3: Create a prediction grid ---
x1 = np.linspace(0, 5, 50)
x2 = np.linspace(0, 5, 50)
X1, X2 = np.meshgrid(x1, x2)
X_pred = np.column_stack([X1.ravel(), X2.ravel()])

y_mean, y_std = gp.predict(X_pred, return_std=True)
Y_mean = y_mean.reshape(X1.shape)
Y_std = y_std.reshape(X1.shape)

# --- Step 4: Plot mean prediction surface ---
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X1, X2, Y_mean, cmap="viridis", alpha=0.9)
ax1.scatter(X[:, 0], X[:, 1], y, c="r", s=50, label="Training data")
ax1.set_title("GP mean prediction")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("y")

# --- Step 5: Plot uncertainty (std deviation as a surface) ---
ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(X1, X2, Y_std, cmap="magma", alpha=0.9)
ax2.scatter(X[:, 0], X[:, 1], 0, c="r", s=50, label="Training data")
ax2.set_title("GP predictive uncertainty (std)")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("σ")

plt.tight_layout()
plt.show()

