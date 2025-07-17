#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 21:04:48 2025

@author: toby
"""

# Imports
import os
from itertools import islice
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm.notebook import tqdm
import bokeh
import bokeh.io
import bokeh.plotting
import bokeh.models
from IPython.display import display, HTML

import matplotlib.pyplot as plt



bokeh.io.output_notebook(hide_banner=True)

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

np.random.seed(42)
tf.random.set_seed(42)
#

# Load the data
# Load the data from the Scripps CO2 program website. 
co2_df = pd.read_csv(
    # Source: https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv
    './monthly_in_situ_co2_mlo.csv', 
    header=0, # Data starts here
    #skiprows=[4, 5], # Headers consist of multiple rows
    usecols=[2, 4], # Only keep the 'Date' and 'CO2' columns
    na_values='-99.99',  # NaNs are denoted as '-99.99'
    #dtype=np.float64
)


# Drop missing values
co2_df.dropna(inplace=True)
# Remove whitespace from column names
co2_df.rename(columns=lambda x: x.strip(), inplace=True)
#

# convert the date column to datetime format
co2_df.Date = pd.to_datetime(co2_df.Date)

print(co2_df.Date)
print(co2_df.CO2)


# Plot data
fig = bokeh.plotting.figure(
    width=600, height=300, 
    x_range=(1958, 2021), y_range=(310, 420))
fig.xaxis.axis_label = 'Date'
fig.yaxis.axis_label = 'CO₂ (ppm)'
fig.add_layout(bokeh.models.Title(
    text='In situ air measurements at Mauna Loa, Observatory, Hawaii',
    text_font_style="italic"), 'above')
fig.add_layout(bokeh.models.Title(
    text='Atmospheric CO₂ concentrations', 
    text_font_size="14pt"), 'above')
fig.line(
    co2_df.Date, co2_df.CO2, legend_label='All data',
    line_width=2, line_color='midnightblue')
fig.legend.location = 'top_left'
fig.toolbar.autohide = True
bokeh.plotting.show(fig)
#


plt.figure(figsize=(6, 4))
# Make the plots
plt.plot(co2_df.Date, co2_df.CO2)
plt.title('CO2 conc. ppm')
plt.xlabel('date', fontsize=13)
plt.ylabel('concentration', fontsize=13)
plt.xlim([1958, 2024])
plt.tight_layout()
plt.show()
#

print("I have made it to here")



# Split the data into observed and to predict
date_split_predict = 2010
date_split_predict = pd.to_datetime(date_split_predict)

df_observed = co2_df[co2_df.Date < date_split_predict]
print('{} measurements in the observed set'.format(len(df_observed)))
df_predict = co2_df[co2_df.Date >= date_split_predict]
print('{} measurements in the test set'.format(len(df_predict)))
#

# Define mean function which is the means of observations
observations_mean = tf.constant(
    [np.mean(df_observed.CO2.values)], dtype=tf.float64)
mean_fn = lambda _: observations_mean
#


# Define the kernel with trainable parameters. 
# Note we transform some of the trainable variables to ensure
#  they stay positive.

# Use float64 because this means that the kernel matrix will have 
#  less numerical issues when computing the Cholesky decomposition

# Constrain to make sure certain parameters are strictly positive
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

# Smooth kernel hyperparameters
smooth_amplitude = tfp.util.TransformedVariable(
    initial_value=10., bijector=constrain_positive, dtype=np.float64,
    name='smooth_amplitude')
smooth_length_scale = tfp.util.TransformedVariable(
    initial_value=10., bijector=constrain_positive, dtype=np.float64,
    name='smooth_length_scale')
# Smooth kernel
smooth_kernel = tfk.ExponentiatedQuadratic(
    amplitude=smooth_amplitude, 
    length_scale=smooth_length_scale)

# Local periodic kernel hyperparameters
periodic_amplitude = tfp.util.TransformedVariable(
    initial_value=5.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_amplitude')
periodic_length_scale = tfp.util.TransformedVariable(
    initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_length_scale')
periodic_period = tfp.util.TransformedVariable(
    initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_period')
periodic_local_length_scale = tfp.util.TransformedVariable(
    initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_local_length_scale')
# Local periodic kernel
local_periodic_kernel = (
    tfk.ExpSinSquared(
        amplitude=periodic_amplitude, 
        length_scale=periodic_length_scale,
        period=periodic_period) * 
    tfk.ExponentiatedQuadratic(
        length_scale=periodic_local_length_scale))

# Short-medium term irregularities kernel hyperparameters
irregular_amplitude = tfp.util.TransformedVariable(
    initial_value=1., bijector=constrain_positive, dtype=np.float64,
    name='irregular_amplitude')
irregular_length_scale = tfp.util.TransformedVariable(
    initial_value=1., bijector=constrain_positive, dtype=np.float64,
    name='irregular_length_scale')
irregular_scale_mixture = tfp.util.TransformedVariable(
    initial_value=1., bijector=constrain_positive, dtype=np.float64,
    name='irregular_scale_mixture')
# Short-medium term irregularities kernel
irregular_kernel = tfk.RationalQuadratic(
    amplitude=irregular_amplitude,
    length_scale=irregular_length_scale,
    scale_mixture_rate=irregular_scale_mixture)

# Noise variance of observations
# Start out with a medium-to high noise
observation_noise_variance = tfp.util.TransformedVariable(
    initial_value=1, bijector=constrain_positive, dtype=np.float64,
    name='observation_noise_variance')

trainable_variables = [v.variables[0] for v in [
    smooth_amplitude,
    smooth_length_scale,
    periodic_amplitude,
    periodic_length_scale,
    periodic_period,
    periodic_local_length_scale,
    irregular_amplitude,
    irregular_length_scale,
    irregular_scale_mixture,
    observation_noise_variance
]]

#


# Sum all kernels to single kernel containing all characteristics
kernel = (smooth_kernel + local_periodic_kernel + irregular_kernel)

# Define mini-batch data iterator
batch_size = 128

batched_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (df_observed.Date.values.reshape(-1, 1), df_observed.CO2.values))
    .shuffle(buffer_size=len(df_observed))
    .repeat(count=None)
    .batch(batch_size)
)
#
