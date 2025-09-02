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
#from sklearn.model_selection import train_test_split
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C




################################################################
#################### define experiment param ###################
################################################################


speed_robot = 0.3
speed_human = 1.0

rng = np.random.default_rng() 

#grid_position = np.empty((0, 2))   # 0 rows, 2 columns
#GP_time = np.empty((0,))     # empty targets

grid_position = np.array ([[1,1],[1,1]])
GP_time = np.array ([1,1])


################################################################
#################### create GP's  ##############################
################################################################

kernel = RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.1)
gp_human = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp_human.fit(grid_position, GP_time)

kernel = RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.1)
gp_robot = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp_robot.fit(grid_position, GP_time)


################################################################
#################### calculate positon  ########################
################################################################

 
x_position = rng.integers(0, 20, endpoint = True)
y_position = rng.integers(0, 5, endpoint = True)

print("x position: ", x_position)
print("y poistion: ", y_position)

################################################################
####################  Predict from GP  #########################
################################################################


X_pred = np.array([[x_position, y_position]])
time_robot_predict_mean, time_robot_predict_std = gp_robot.predict(X_pred, return_std=True)
print ("time_robot_predict_mean: ", time_robot_predict_mean)

time_human_predict_mean, time_human_predict_std = gp_human.predict(X_pred, return_std=True)
print ("time_human_predict_mean: ", time_human_predict_mean)


################################################################
####################  Decide H or R  ###########################
################################################################

if time_human_predict_mean <= time_robot_predict_mean:
    
    print("human")

else:
    
    print("Robot")

################################################################
#################### calculate time  ###########################
################################################################





sys.exit()



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


################################################################
#################### Update GP  ################################
################################################################



