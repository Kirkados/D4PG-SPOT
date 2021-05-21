""" This script loads the deep guidance data logged from an experiment (specifically, logged by use_deep_guidance.py)
renders a few plots, and animates the motion.

It should be run from the folder where use_deep_guidance.py was run from for the given experiment.
 """
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display # for rendering

# import code # for debugging
#code.interact(local=dict(globals(), **locals())) # Ctrl+D or Ctrl+Z to continue execution
 
from settings import Settings
environment_file = __import__('environment_' + Settings.ENVIRONMENT) # importing the environment

def make_C_bI(angle):        
    C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
    return C_bI


# Check if we're using the right environment
if Settings.ENVIRONMENT != 'SPOT':
    print("You must use the SPOT environment in Settings... quitting")
    raise SystemExit

# Generate a virtual display for plotting
display = Display(visible = False, size = (1400,900))
display.start()

#####################################
### Load in the experimental data ###
#####################################
log_filename = glob.glob('*.txt')[0]
data = np.load(log_filename)
print("Data file %s is loaded" %log_filename)
os.makedirs(log_filename.split('.')[0], exist_ok=True)

########################
### Plot some things ###
########################
time_log = data[:,0]
deep_guidances_ax = data[:,1]
deep_guidances_ay = data[:,2]
deep_guidances_alpha = data[:,3]


plt.figure()
plt.plot(time_log, deep_guidances_ax)
plt.plot(time_log, deep_guidances_ay)
plt.savefig(log_filename.split('.')[0] + "/Acceleration Commands.png")
print("Saved acceleration commands figure")

plt.figure()
plt.plot(time_log, deep_guidances_alpha)
plt.savefig(log_filename.split('.')[0] + "/Angular Acceleration commands.png")
print("Saved angular acceleration commands figure")



##########################
### Animate the motion ###
##########################
# Generate an Environment to use for reward logging
environment = environment_file.Environment()
environment.reset(False,False)

# Process the data. Need to make the raw total state log
# [relative_x, relative_y, relative_vx, relative_vy, 
#relative_angle, relative_angular_velocity, chaser_x, chaser_y, chaser_theta, 
#target_x, target_y, target_theta, chaser_vx, chaser_vy, chaser_omega, 
#target_vx, target_vy, target_omega] *# Relative pose expressed in the chaser's body frame; everythign else in Inertial frame #*

print("Rendering animation...", end='')
raw_total_state_log = []
cumulative_reward_log = []
SPOTNet_sees_target_log = []
cumulative_rewards = 0
SPOTNet_previous_relative_x = 0.0
for i in range(len(data)):
    Pi_time, deep_guidance_ax, deep_guidance_ay, deep_guidance_alpha, \
    Pi_red_x, Pi_red_y, Pi_red_theta, \
    Pi_red_Vx, Pi_red_Vy, Pi_red_omega,        \
    Pi_black_x, Pi_black_y, Pi_black_theta,    \
    Pi_black_Vx, Pi_black_Vy, Pi_black_omega,  \
    SPOTNet_relative_x, SPOTNet_relative_y, SPOTNet_relative_angle, SPOTNet_sees_target, \
    SPOTNet_target_x_inertial, SPOTNet_target_y_inertial, SPOTNet_target_angle_inertial= data[i,:]
    
    if SPOTNet_sees_target:
        rel_vx_body = 0
        rel_vy_body = 0
        
        # Log the data, holding the target inertially fixed between SPOTNet updates        
        raw_total_state_log.append([SPOTNet_relative_x, SPOTNet_relative_y, rel_vx_body, rel_vy_body, SPOTNet_relative_angle, Pi_black_omega - Pi_red_omega, Pi_red_x, Pi_red_y, Pi_red_theta, SPOTNet_target_x_inertial, SPOTNet_target_y_inertial, SPOTNet_target_angle_inertial, Pi_red_Vx, Pi_red_Vy, Pi_red_omega, Pi_black_Vx, Pi_black_Vy, Pi_black_omega])
        SPOTNet_sees_target_log.append(True)
    else:
        rel_x_body = 0
        rel_y_body = 0
        rel_vx_body = 0
        rel_vy_body = 0        
        raw_total_state_log.append([rel_x_body, rel_y_body, rel_vx_body, rel_vy_body, Pi_black_theta - Pi_red_theta, Pi_black_omega - Pi_red_omega,          Pi_red_x, Pi_red_y, Pi_red_theta, Pi_black_x, Pi_black_y, Pi_black_theta, Pi_red_Vx, Pi_red_Vy, Pi_red_omega, Pi_black_Vx, Pi_black_Vy, Pi_black_omega])
        SPOTNet_sees_target_log.append(False)
    
    
    # Check the reward function based off this state
    environment.chaser_position = np.array([Pi_red_x, Pi_red_y, Pi_red_theta])
    environment.chaser_velocity = np.array([Pi_red_Vx, Pi_red_Vy, Pi_red_omega])
    environment.target_position = np.array([Pi_black_x, Pi_black_y, Pi_black_theta])
    environment.target_velocity = np.array([Pi_black_Vx, Pi_black_Vy, Pi_black_omega])
    
    # Get environment to check for collisions
    environment.update_docking_locations()
    environment.check_collisions()
    rewards_this_timestep = environment.reward_function(0)
    cumulative_rewards += rewards_this_timestep
    cumulative_reward_log.append(cumulative_rewards)
    

# Render the episode
environment_file.render(np.asarray(raw_total_state_log), 0, 0, np.asarray(cumulative_reward_log), 0, 0, 0, 0, 0, 1, log_filename.split('.')[0], '', time_log, np.asarray(SPOTNet_sees_target_log))
print("Done!")
# Close the display
del environment
plt.close()
display.stop()