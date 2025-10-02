#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:21:31 2025

@author: toby
"""

import numpy as np


rng = np.random.default_rng() 

time_human_predict_mean = 0.


for time_human_predict_mean in range (0,100):

    print (time_human_predict_mean, "  ", (rng.normal(time_human_predict_mean, 2, 1)), " ", (time_human_predict_mean + rng.normal(0,2,1)))
    
  