#!/usr/bin/env python_311
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:45:07 2025

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
####### class to generate random position ######################
################################################################


class Position:
  def __init__(self, token_x, token_y):
      
    
      rng = np.random.default_rng()  
      self.token_x = rng.integers(0, 20, endpoint = False)
      self.token_y = rng.integers(-3, 3, endpoint = True)
      

################################################################
#################### class to manage GP ########################
################################################################



class Gaussian_P:
    def __init__(self, x_pos, y_pos):
        
        self.counter_x_pos = x_pos
        self.counter_y_pos = y_pos
        
        ### define speed of robot and speed of human
        speed_robot = 0.3
        speed_human = 1.0
        
        print ("robot speed: ", speed_robot)
        print ("human speed: ", speed_human)
        
        ### Euclidean distance of counter_x_pos, counter_y_pos
        self.distance_robot = (self.counter_x_pos**2 + self.counter_y_pos**2) ** 0.5  
        self.distance_human = ((20-self.counter_x_pos)**2 + self.counter_y_pos**2) ** 0.5
        
        ### time = distance / speed
        
        
        
        ##def decide ()
       
        
        ### take sample from lognormal
#        rng = np.random.default_rng()
#        mu, sigma = 3., 1. # mean and standard deviation
#        s = rng.lognormal(mu, sigma, 1000)

        ##print (s)
        
        
        ##print(self.x_pos)
        ##print(self.y_pos)
        
        ##print (distance)




################################################################
############################ Main ##############################
################################################################


trial = Position(0,0)


print (trial.token_x, trial.token_y)




distance = Gaussian_P(trial.token_x,trial.token_y)

print ("distance robot: ", distance.distance_robot)
print ("distance human: ", distance.distance_human)



sys.exit()

#print (trial.token_x)
#print (trial.token_y)







################################################################
##################### Set up data ##############################
################################################################


#np.random.seed(17)  # So you can reproduce these results at home

robot_speed = 0.3
human_speed = 1.

rng = np.random.default_rng()
token_x = rng.integers(0, 20, endpoint = False)

print ("token_x: ", token_x)
print ("token_y: ", token_y)

sys.exit()


sigma = 1.
rng = np.random.default_rng()
# nv = 20  # Number of points in each direction
xv = np.linspace(-10.,10.,20)  # x-vector
yv = np.linspace(-3.,3.,6)  # y-vector
x, y = np.meshgrid(xv, yv)  # x and y are (nv, nv) matrices
# z = (x-3.)**2. + 2.*x*y + (2.*y+3.)**2. - 3.

#z = rng.lognormal(x, sigma, 1) #((x**2.)+(y**2.)**0.5)
#z = ((x**2.)+(y**2.)**0.5)

z = y**2

print (z)

#for i in range (0,x):
    
#    for j in range (0,y):
        
        

#noise_level = 5.
#z_noisy = z + np.random.normal(size=x.shape) * noise_level


X = np.column_stack((x.reshape(-1), y.reshape(-1)))
Z = z.reshape(-1,1)

#print(Z)

sys.exit()


################################################################
##################### Set up kernel ############################
################################################################



guess_l = (2., 1.)  # In general, x and y have different scales
bounds_l = ((1e-1,100.),) * 2  # Same bounds for x and y
guess_n = 1.  # Amount of noise
bounds_n = (1e-20, 10.) # Bounds for noise
kernel = (  # Kernel objects can simply be summed using +
    RBF(length_scale=guess_l, length_scale_bounds=bounds_l)
    + WhiteKernel(noise_level=guess_n, noise_level_bounds=bounds_n)
)


################################################################
##################### Split data - train and test set ##########
################################################################



X_train, X_test, Y_train, Y_test = train_test_split(X, Z, test_size=0.95)


################################################################
##################### Generate GP ##############################
################################################################


gpr = GaussianProcessRegressor(kernel, normalize_y=True )
gpr.fit(X_train, Y_train)


print(gpr.kernel_)
#RBF(length_scale=[7.86, 3.71]) + WhiteKernel(noise_level=0.00678)


Zfit, Zstd = gpr.predict(X, return_std=True)
zstd = Zstd.reshape(x.shape)
zfit = Zfit.reshape(x.shape)


################################################################
##################### Plot   ###################################
################################################################


# Set up figure
fig, ax = plt.subplots(figsize=(6., 4.))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((-5,5))
ax.set_ylim((-5,5))
ax.set_aspect('equal')
ax.set_xticks((-5,0,5))
ax.set_yticks((-5,0,5))
ax.grid(False)

# Do the plotting
lev = np.linspace(0.,250.,6)
ax.contour(x,y,z,lev,colors='k')  # Truth
ax.plot(*X_train.T,'o',color="C1")  # Training samples
ax.contour(x, y,zfit, lev, colors='C0', linestyles='dashed')  # Posterior mean

# Legend
truth_line = mlines.Line2D([], [], color='black', label='True $z(x,y)$')
sample_line = mlines.Line2D([], [], color='C1', marker="o", linestyle="none", label='Train samples')
mean_line = mlines.Line2D([], [], color='C0', linestyle="--", label='Posterior mean')
ax.legend(handles=[truth_line, sample_line, mean_line],bbox_to_anchor=(1.05, 1), loc="upper left")

# Write out
plt.tight_layout()
plt.savefig('gpr_posterior_mean.svg')


# Set up figure
fig, ax = plt.subplots(figsize=(6., 4.))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((-5,5))
ax.set_ylim((-5,5))
ax.set_aspect('equal')
ax.set_xticks((-5,0,5))
ax.set_yticks((-5,0,5))
ax.grid(False)

# Do the plotting
ax.plot(*X_train.T,'o',color="C1")  # Training samples
lev = np.linspace(6.,16.,11)
hc = ax.contourf(x, y, zstd, lev)  # Posterior std
#for hci in hc.collections:
#  hci.set_edgecolor("face")

# Colorbar
hcb = plt.colorbar(hc)
hcb.ax.grid(False)
hcb.set_label('Posterior standard deviation')

# Write out
plt.tight_layout()
plt.savefig('gpr_posterior_std.svg')