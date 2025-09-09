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
#################### generate positon  #########################
################################################################

 
x_position = rng.integers(0, 20, endpoint = True)
y_position = rng.integers(0, 5, endpoint = True)

print("x position: ", x_position)
print("y poistion: ", y_position)

temp_position = np.array ([[x_position,y_position]])

print ("Grid positon: ",grid_position)

grid_position = np.append(grid_position, temp_position, axis = 0)

print ("Grid positon: ",grid_position)




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


## Decide Robot or human based upon output of GP and calc. Euclidian distance and work out time

mu  = 0. # mean
sigma = 0.5  #standard deviation


if time_human_predict_mean <= time_robot_predict_mean:
    
    print("Human")
    distance_human = ((20-x_position)**2 + y_position**2) ** 0.5
    time_human = distance_human / speed_human
    noise_human = time_human + rng.normal(mu, sigma, 1)
    print (distance_human, time_human, noise_human)
    GP_time = np.append(GP_time, noise_human, axis = 0)
    
    

    
    print ("GP time: ", GP_time)

    

else:
    
    print("Robot")
    distance_robot = (x_position**2 + y_position**2) ** 0.5  
    time_robot = distance_robot / speed_robot
    noise_robot = time_robot + rng.normal(mu, sigma, 1)
    print (distance_robot, time_robot, noise_robot)
    GP_time = np.append(GP_time, noise_robot, axis = 0)

    

   






sys.exit()





