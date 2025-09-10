#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:12:50 2025

@author: toby
"""

import numpy as np
import sys
import time

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
count = 20 #number of iterations
sleep_time = 0.1 #delay in seconds on each loop

rng = np.random.default_rng() 

grid_position_human = np.array ([[1,1],[1,1]])
grid_position_robot = np.array ([[1,1],[1,1]])
#GP_time = np.array ([[1],[1]])
GP_time_robot = np.random.rand(2,1)
GP_time_human = np.random.rand(2,1)

#print (GP_time)

#print (np.shape(grid_position))

#print (np.shape(GP_time))

#print (np.shape(GP_time_temp))



#sys.exit()

################################################################
#################### create GP's  ##############################
################################################################

kernel = RBF(length_scale=[1.0, 1.0])
gp_human = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp_human.fit(grid_position_human, GP_time_human)

kernel = RBF(length_scale=[1.0, 1.0])
gp_robot = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp_robot.fit(grid_position_robot, GP_time_robot)


for i in range (0,count):

    print ("Count: ",i)

################################################################
#################### generate positon  #########################
################################################################

 
    x_position = rng.integers(0, 20, endpoint = True)
    y_position = rng.integers(0, 5, endpoint = True)

    print("x position: ", x_position)
    print("y poistion: ", y_position)

    temp_position = np.array ([[x_position,y_position]])

    #print ("Grid positon: ",grid_position)



    #print ("Grid positon: ",grid_position)




################################################################
####################  Predict from GP  #########################
################################################################


    #X_pred = np.array([[x_position, y_position]])
    time_robot_predict_mean, time_robot_predict_std = gp_robot.predict(temp_position, return_std=True)
    print ("time_robot_predict_mean: ", time_robot_predict_mean)

    time_human_predict_mean, time_human_predict_std = gp_human.predict(temp_position, return_std=True)
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
        display = np.float64(noise_human[0])
        txt = f"{distance_human:.2f}, {time_human:.2f}, {display:.2f}"
        print(txt)
        noise_human = noise_human.reshape (1,1)
    
        GP_time_human = np.append(GP_time_human, noise_human, axis = 0)
        
        grid_position_human = np.append(grid_position_human, temp_position, axis = 0)
        
        gp_human.fit(grid_position_human, GP_time_human)
        
       
    

    else:
    
        print("Robot")
        distance_robot = (x_position**2 + y_position**2) ** 0.5  
        time_robot = distance_robot / speed_robot
        noise_robot = time_robot + rng.normal(mu, sigma, 1)
        display = np.float64(noise_robot[0])
        txt = f"{distance_robot:.2f}, {time_robot:.2f}, {display:.2f}"
        print(txt)
        
        noise_robot = noise_robot.reshape (1,1)
        
        GP_time_robot = np.append(GP_time_robot, noise_robot, axis = 0)
    
        
        grid_position_robot = np.append(grid_position_robot, temp_position, axis = 0)
        
        
        gp_robot.fit(grid_position_robot, GP_time_robot)

    print ("\n")

    #time.sleep(sleep_time) #add a little pause?


result_robot = np.concatenate((grid_position_robot, GP_time_robot), axis = 1)
                        
print ("ROBOT: \n",result_robot)    


result_human = np.concatenate((grid_position_human, GP_time_human), axis = 1)

print ("HUMAN: \n",result_human)




sys.exit()





