#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:26:25 2025

@author: toby
"""


import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()
mu, sigma = 3., 1. # mean and standard deviation
s = rng.lognormal(mu, sigma, 1000)

print (s)

count, bins, _ = plt.hist(s, 100, density=True, align='mid')

x = np.linspace(min(bins), max(bins), 10000)

pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))

plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()





# Parameters for the log-normal distribution
#mean = 0    # Mean of the underlying normal distribution
#std_dev = 1  # Standard deviation of the underlying normal distribution
#size = 1000  # Number of samples to generate

# Generate samples from a log-normal distribution
#samples = np.random.lognormal(mean, std_dev, size)

# Plot the histogram of the samples
#plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')
#plt.title('Log-Normal Distribution Samples')
#plt.xlabel('Value')
#plt.ylabel('Density')
#plt.show()