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
    dtype=np.float64
)



# Drop missing values
co2_df.dropna(inplace=True)
# Remove whitespace from column names
co2_df.rename(columns=lambda x: x.strip(), inplace=True)
#

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

# Split the data into observed and to predict
date_split_predict = 2010
df_observed = co2_df[co2_df.Date < date_split_predict]
print('{} measurements in the observed set'.format(len(df_observed)))
df_predict = co2_df[co2_df.Date >= date_split_predict]
print('{} measurements in the test set'.format(len(df_predict)))
#