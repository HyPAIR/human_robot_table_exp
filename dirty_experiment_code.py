#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:12:50 2025

@author: toby
"""

import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split



################################################################
#################### define experiment param ###################
################################################################


speed_robot = 0.3
speed_human = 1.0

rng = np.random.default_rng(10) 

GP = np.array([0,0])  ##### NEED TO FIX THIS - ZEROS WILL KILL THE AVERAGE


################################################################
#################### calculate positon  ########################
################################################################

 
x_position = rng.integers(0, 20, endpoint = False)
y_position = rng.integers(-3, 3, endpoint = True)

print("x position: ", x_position)
print("y poistion: ", y_position)


################################################################
#################### sample time  ##############################
################################################################

print ("robot speed: ", speed_robot)
print ("human speed: ", speed_human)
        
### Euclidean distance of counter_x_pos, counter_y_pos
distance_robot = (x_position**2 + y_position**2) ** 0.5  
distance_human = ((20-x_position)**2 + y_position**2) ** 0.5
        
### time = distance / speed
        
## NEED TO LOOK AT LOG NORMAL DISTRIBUTION - USED THIS TO 
## CALCULATE THE TIME - IE DIXTRIBUTION GIVES THE VARIABILITY ON TIJME
        
##def decide ()
       
### time = distance / speed
        
time_robot = distance_robot / speed_robot
time_human = distance_human / speed_human
       
               
print ("robot time: ", time_robot)
print ("human time: ", time_human) 
       
        
       
mu  = 1. # mean and standard deviation
sampled_human = rng.lognormal(time_human, mu, 1)
sampled_robot = rng.lognormal(time_robot, mu, 1)
        
print (sampled_human)
print (sampled_robot)


