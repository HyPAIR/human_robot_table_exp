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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
#from sklearn.model_selection import train_test_split
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

f = open('workfile', 'w', encoding="utf-8")

################################################################
#################### define experiment param ###################
################################################################


speed_robot = 0.3
speed_human = 1.0
count =300 #number of iterations
sleep_time = 0.1 #delay in seconds on each loop
robot_count = 0
human_count = 0

rng = np.random.default_rng() 

grid_position_human = np.array ([[1,1],[1,1]])
grid_position_robot = np.array ([[1,1],[1,1]])
#GP_time = np.array ([[1],[1]])
#GP_time_robot = np.random.rand(2,1)
GP_time_robot = np.array ([[60.],[60.]])
GP_time_human = np.random.rand(2,1)

time_human_predict = np.random.rand(2,1)
time_robot_predict = np.random.rand(2,1)


#print (GP_time)

#print (np.shape(grid_position))

#print (np.shape(GP_time))

#print (np.shape(GP_time_temp))



#sys.exit()

################################################################
#################### create GP's  ##############################
################################################################

kernel = RBF(length_scale=[1.0, 2.0],length_scale_bounds=(1e-10, 1e5))
gp_human = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp_human.fit(grid_position_human, GP_time_human)

kernel = RBF(length_scale=[1.0, 2.0],length_scale_bounds=(1e-10, 1e5))
gp_robot = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp_robot.fit(grid_position_robot, GP_time_robot)


for i in range (0,count):

    print ("Count: ",i)
    output = str(i)
    output = "\n\n" + output + "\n"
    f.write (output)

################################################################
#################### generate positon  #########################
################################################################

 
    x_position = rng.integers(0, 20, endpoint = True)
    y_position = rng.integers(0, 5, endpoint = True)

    print("x position: ", x_position)
    print("y poistion: ", y_position)
    
    output = str (x_position)
    output = "x position: " + output + "\n"
    f.write (output)
    
    output = str (y_position)
    output = "y position: " + output + "\n"
    f.write (output)

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


    output = str (time_robot_predict_mean)
    output = "time_robot_predict_mean: " + output + "\n"
    f.write (output)
    
    output = str (time_human_predict_mean)
    output = "time_human_predict_mean: " + output + "\n"
    f.write (output)

################################################################
####################  Decide H or R  ###########################
################################################################


## Decide Robot or human based upon output of GP and calc. Euclidian distance and work out time

    mu  = 0. # mean
    sigma = 0.5  #standard deviation


    if time_human_predict_mean <= time_robot_predict_mean:
    
        print("Human")
        
        human_count += 1
        
        output = "Human \n"
        f.write(output)
        
        distance_human = ((20-x_position)**2 + y_position**2) ** 0.5
        time_human = distance_human / speed_human
        noise_human = time_human + rng.normal(mu, sigma, 1)
        
        
        if noise_human < 0: 
            print (noise_human)
            output = str (noise_human)
            f.write (output)
            noise_human = np.float64([0.1])
            print ("altered: ", noise_human)
            f.write ("altered \n")
            
            
        display = np.float64(noise_human[0])
        txt = f"{distance_human:.2f}, {time_human:.2f}, {display:.2f}"
        print(txt)
        
        f.write (txt)
        
        
        noise_human = noise_human.reshape (1,1)
    
        GP_time_human = np.append(GP_time_human, noise_human, axis = 0)
        
        grid_position_human = np.append(grid_position_human, temp_position, axis = 0)
        
        gp_human.fit(grid_position_human, GP_time_human)
        
        time_human_predict_mean = time_human_predict_mean.reshape(1,1)
                
        time_human_predict = np.append(time_human_predict, time_human_predict_mean, axis = 0)
        
        
    else:
    
        print("Robot")
        
        robot_count += 1
        
        output = "Robot \n"
        f.write(output)
        
        distance_robot = (x_position**2 + y_position**2) ** 0.5  
        time_robot = distance_robot / speed_robot
        noise_robot = time_robot + rng.normal(mu, sigma, 1)
        
        if noise_robot < 0: 
            print(noise_robot)
            output = str (noise_robot)
            f.write (output)
            noise_robot = np.float64([0.1])
            print ("altered: ", noise_robot)
            f.write ("altered \n")
            
        display = np.float64(noise_robot[0])
        txt = f"{distance_robot:.2f}, {time_robot:.2f}, {display:.2f}"
        print(txt)
        
        f.write(txt)
        
        noise_robot = noise_robot.reshape (1,1)
        
        GP_time_robot = np.append(GP_time_robot, noise_robot, axis = 0)
        
        grid_position_robot = np.append(grid_position_robot, temp_position, axis = 0)
        
        gp_robot.fit(grid_position_robot, GP_time_robot)
        
        time_robot_predict_mean = time_robot_predict_mean.reshape(1,1)
        
        time_robot_predict = np.append(time_robot_predict, time_robot_predict_mean, axis = 0)

        if noise_robot < 0:
            print ("died")
            sys.exit()

    print ("\n")

    #time.sleep(sleep_time) #add a little pause?

delta = GP_time_robot - time_robot_predict

result_robot = np.concatenate((grid_position_robot, GP_time_robot), axis = 1)
result_robot = np.concatenate((result_robot, time_robot_predict), axis = 1)
result_robot = np.concatenate((result_robot, delta), axis = 1)

                        
np.set_printoptions(suppress=True,precision=2)
print ("ROBOT: \n",result_robot)    

delta = GP_time_human - time_human_predict

result_human = np.concatenate((grid_position_human, GP_time_human), axis = 1)
result_human = np.concatenate((result_human, time_human_predict), axis = 1)
result_human = np.concatenate((result_human, delta), axis = 1)

np.set_printoptions(suppress=True,precision=2)
print ("HUMAN: \n",result_human)



np.savetxt("robot.csv", result_robot, delimiter=",", fmt="%.2f")  # 2 decimal places
np.savetxt("human.csv", result_human, delimiter=",", fmt ="%.2f") 


output = "robot count: " + str(robot_count)
f.write (output)
output = "human count: " + str(human_count)
f.write (output)          

print ("robot count: ", str(robot_count))
print ("human count: ", str(human_count))

f.close()  ##close the results file

################################################################
####################  plot some graphs  ########################
################################################################


column = result_robot[:, 3]
plt.plot(column , marker='o')
plt.title("Predited time - Robot")
plt.xlabel("Index")
plt.ylabel("Time (s)")
plt.grid(True)
plt.show()


column = result_robot[:, 4]
plt.plot(column , marker='o')
plt.title("Error - Robot")
plt.xlabel("Index")
plt.ylabel("Time (s)")
plt.grid(True)
plt.show()

column = result_human[:, 3]
plt.plot(column , marker='o')
plt.title("Predited time - Human")
plt.xlabel("Index")
plt.ylabel("Time (s)")
plt.grid(True)
plt.show()


column = result_human[:, 4]
plt.plot(column , marker='o')
plt.title("Error - Human")
plt.xlabel("Index")
plt.ylabel("Time (s)")
plt.grid(True)
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(16, 4))

################################################################
####################  Plot surfaces  ###########################
################################################################

## set up grid for plot - surface
x_grid = np.linspace(0, 20, 20)
y_grid = np.linspace(0, 5, 5)
x_grid, y_grid = np.meshgrid(x_grid, y_grid)
XY_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])

# Predict mean from robot GP
z_mean_robot, z_std_robot = gp_robot.predict(XY_grid, return_std=True)
z_mean_robot = z_mean_robot.reshape(x_grid.shape)

# Predict mean from human GP
z_mean_human, z_std_human = gp_human.predict(XY_grid, return_std=True)
z_mean_human = z_mean_human.reshape(x_grid.shape)



# Create figure with 1 row and 2 columns
fig = plt.figure(figsize=(12, 5))

# First 3D subplot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(x_grid, y_grid, z_mean_human, cmap='viridis')
ax1.set_title("Human")

# Second 3D subplot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(x_grid, y_grid, z_mean_robot, cmap='plasma')
ax2.set_title("Robot")

plt.tight_layout()
plt.show()



sys.exit()











