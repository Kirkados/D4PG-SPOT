""" This script loads the deep guidance data logged from an experiment (specifically, logged by use_deep_guidance.py)
renders a few plots, and animates the motion.

It should be run from the folder where use_deep_guidance.py was run from for the given experiment.
 """
import numpy as np
import glob
import os
 
from settings import Settings
environment_file = __import__('environment_' + Settings.ENVIRONMENT) # importing the environment


#####################################
### Load in the experimental data ###
#####################################
log_filename = glob.glob('*.txt')[0]
data = np.load(log_filename)

t = data[:,0]


# Process the data. Need to make the raw total state log
# [relative_x, relative_y, relative_vx, relative_vy, 
#relative_angle, relative_angular_velocity, chaser_x, chaser_y, chaser_theta, 
#target_x, target_y, target_theta, chaser_vx, chaser_vy, chaser_omega, 
#target_vx, target_vy, target_omega] *# Relative pose expressed in the chaser's body frame; everythign else in Inertial frame #*

raw_total_state_log = []
for i in range(len(data)):
    Pi_time, Pi_red_x, Pi_red_y, Pi_red_theta, \
    Pi_red_Vx, Pi_red_Vy, Pi_red_omega,        \
    Pi_black_x, Pi_black_y, Pi_black_theta,    \
    Pi_black_Vx, Pi_black_Vy, Pi_black_omega,  \
    SPOTNet_relative_x, SPOTNet_relative_y, SPOTNet_relative_angle, SPOTNet_sees_target = data[i,:]
    
    if SPOTNet_sees_target:
        rel_vx_body = 0
        rel_vy_body = 0
        raw_total_state_log.append([SPOTNet_relative_x, SPOTNet_relative_y, rel_vx_body, rel_vy_body, SPOTNet_relative_angle, Pi_black_omega - Pi_red_omega, Pi_red_x, Pi_red_y, Pi_red_theta, Pi_black_x, Pi_black_y, Pi_black_theta, Pi_red_Vx, Pi_red_Vy, Pi_red_omega, Pi_black_Vx, Pi_black_Vy, Pi_black_omega])
    else:
        rel_x_body = 0
        rel_y_body = 0
        rel_vx_body = 0
        rel_vy_body = 0        
        raw_total_state_log.append([rel_x_body, rel_y_body, rel_vx_body, rel_vy_body, Pi_black_theta - Pi_red_theta, Pi_black_omega - Pi_red_omega,          Pi_red_x, Pi_red_y, Pi_red_theta, Pi_black_x, Pi_black_y, Pi_black_theta, Pi_red_Vx, Pi_red_Vy, Pi_red_omega, Pi_black_Vx, Pi_black_Vy, Pi_black_omega])

# Render the episode
environment_file.render(np.asarray(raw_total_state_log), 0, 0, 0, 0, 0, 0, 0, 0, 1, 'ExperimentAnimation', os.getcwd())
