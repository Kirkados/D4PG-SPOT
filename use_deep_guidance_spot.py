"""
This script loads in a trained policy neural network and uses it for inference.

Typically this script will be executed on the Nvidia Jetson TX2 board during an
experiment in the Spacecraft Robotics and Control Laboratory at Carleton
University.

Script created: June 12, 2019
@author: Kirk (khovell@gmail.com)
"""

import tensorflow as tf
import numpy as np
import socket
import time
import threading
from collections import deque

from settings import Settings
from build_neural_networks import BuildActorNetwork

"""
*# Relative pose expressed in the chaser's body frame; everything else in Inertial frame #*
Deep guidance output in x and y are in the chaser body frame
"""

def make_C_bI(angle):        
    C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
    return C_bI



class MessageParser:
    
    def __init__(self, testing, client_socket, messages_to_deep_guidance, stop_run_flag):
        
        print("Initializing Message Parser!")
        self.client_socket = client_socket
        self.messages_to_deep_guidance = messages_to_deep_guidance
        self.stop_run_flag = stop_run_flag
        self.testing = testing
        
        # Target detection threshold for SPOTNet
        self.SPOTNet_detection_threshold = 0.8        
        
        # Initializing all variables that will be passed to the Deep Guidance
        # Items from SPOTNet
        self.SPOTNet_relative_x = 0
        self.SPOTNet_relative_y = 0
        self.SPOTNet_relative_angle = 0
        self.SPOTNet_sees_target = False
        
        # Items from the Pi
        self.Pi_time = 0
        self.Pi_red_x = 0
        self.Pi_red_y = 0
        self.Pi_red_theta = 0
        self.Pi_red_Vx = 0
        self.Pi_red_Vy = 0
        self.Pi_red_omega = 0
        self.Pi_black_x = 0
        self.Pi_black_y = 0
        self.Pi_black_theta = 0
        self.Pi_black_Vx = 0
        self.Pi_black_Vy = 0
        self.Pi_black_omega = 0
        print("Done initializing parser!")
        
        
    def run(self):
        
        print("Running Message Parser!")
        
        # Run until we want to stop
        while not self.stop_run_flag.is_set():
            
            if self.testing:
                # Assign test values
                # Items from SPOTNet
                self.SPOTNet_relative_x = 2
                self.SPOTNet_relative_y = 0
                self.SPOTNet_relative_angle = 0
                self.SPOTNet_sees_target = True
                
                # Items from the Pi
                self.Pi_time = 15
                self.Pi_red_x = 3
                self.Pi_red_y = 1
                self.Pi_red_theta = 0
                self.Pi_red_Vx = 0
                self.Pi_red_Vy = 0
                self.Pi_red_omega = 0
                self.Pi_black_x = 1
                self.Pi_black_y = 1
                self.Pi_black_theta = 0
                self.Pi_black_Vx = 0
                self.Pi_black_Vy = 0
                self.Pi_black_omega = 0
            else:
                # It's real
                try:
                    data = self.client_socket.recv(4096) # Read the next value
                except socket.timeout:
                    print("Socket timeout")
                    continue
                data_packet = np.array(data.decode("utf-8").splitlines())
                #print('Got message: ' + str(data.decode("utf-8")))
    
                # Check if it's a SpotNet packet or a Pi packet
                if data_packet[0] == "SPOTNet":
                    # We received a SPOTNet packet, update those variables accordingly           
                    self.SPOTNet_relative_x     = float(data_packet[1].astype(np.float32))
                    self.SPOTNet_relative_y     = float(data_packet[2].astype(np.float32))
                    self.SPOTNet_relative_angle = float(data_packet[3].astype(np.float32))
                    self.SPOTNet_sees_target    = float(data_packet[4]) > self.SPOTNet_detection_threshold
                    print("SPOTNet Packet. See target?: %s" %(self.SPOTNet_sees_target))
                    
                else:
                    # We received a packet from the Pi
                    # input_data_array is: [red_x, red_y, red_theta, red_vx, red_vy, red_omega, black_x, black_y, black_theta, black_vx, black_vy, black_omega]  
                    self.Pi_time, self.Pi_red_x, self.Pi_red_y, self.Pi_red_theta, self.Pi_red_Vx, self.Pi_red_Vy, self.Pi_red_omega, self.Pi_black_x, self.Pi_black_y, self.Pi_black_theta, self.Pi_black_Vx, self.Pi_black_Vy, self.Pi_black_omega = data_packet.astype(np.float32)
                    print("Pi Packet! Time: %.1f" %self.Pi_time)
                
            # Write the data to the queue for DeepGuidanceModelRunner to use!
            """ This queue is thread-safe. If I append multiple times without popping, the data in the queue is overwritten. Perfect! """
            #(self.Pi_time, self.Pi_red_x, self.Pi_red_y, self.Pi_red_theta, self.Pi_red_Vx, self.Pi_red_Vy, self.Pi_red_omega, self.Pi_black_x, self.Pi_black_y, self.Pi_black_theta, self.Pi_black_Vx, self.Pi_black_Vy, self.Pi_black_omega, self.SPOTNet_relative_x, self.SPOTNet_relative_y, self.SPOTNet_relative_angle, self.SPOTNet_sees_target)
            self.messages_to_deep_guidance.append((self.Pi_time, self.Pi_red_x, self.Pi_red_y, self.Pi_red_theta, self.Pi_red_Vx, self.Pi_red_Vy, self.Pi_red_omega, self.Pi_black_x, self.Pi_black_y, self.Pi_black_theta, self.Pi_black_Vx, self.Pi_black_Vy, self.Pi_black_omega, self.SPOTNet_relative_x, self.SPOTNet_relative_y, self.SPOTNet_relative_angle, self.SPOTNet_sees_target))
        
        print("Message handler gently stopped")
 

class DeepGuidanceModelRunner:
    
    def __init__(self, testing, client_socket, messages_to_deep_guidance, stop_run_flag):
        
        print("Initializing deep guidance model runner")
        self.client_socket = client_socket
        self.messages_to_deep_guidance = messages_to_deep_guidance
        self.stop_run_flag = stop_run_flag
        self.testing = testing
        
        ###############################
        ### User-defined parameters ###
        ###############################
        self.offset_x = 0 # Docking offset in the body frame
        self.offset_y = 0 # Docking offset in the body frame
        self.offset_angle = 0

        # Uncomment this on TF2.0
        # tf.compat.v1.disable_eager_execution()
        
        # Clear any old graph
        tf.reset_default_graph()
        
        # Initialize Tensorflow, and load in policy
        self.sess = tf.Session()
        # Building the policy network
        self.state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, Settings.OBSERVATION_SIZE], name = "state_placeholder")
        self.actor = BuildActorNetwork(self.state_placeholder, scope='learner_actor_main')
    
        # Loading in trained network weights
        print("Attempting to load in previously-trained model\n")
        saver = tf.train.Saver() # initialize the tensorflow Saver()
    
        # Try to load in policy network parameters
        try:
            ckpt = tf.train.get_checkpoint_state('../')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("\nModel successfully loaded!\n")
    
        except (ValueError, AttributeError):
            print("No model found... quitting :(")
            raise SystemExit
        
        print("Done initializing model!")

    def run(self):
        
        print("Running Deep Guidance!")
        
        counter = 1
        # Parameters for normalizing the input
        relevant_state_mean = np.delete(Settings.STATE_MEAN, Settings.IRRELEVANT_STATES)
        relevant_half_range = np.delete(Settings.STATE_HALF_RANGE, Settings.IRRELEVANT_STATES)
        
        # To log data
        data_log = []
        
        # Run until we want to stop
        while not stop_run_flag.is_set():            
                       
            # Total state is [relative_x, relative_y, relative_vx, relative_vy, relative_angle, relative_angular_velocity, chaser_x, chaser_y, chaser_theta, target_x, target_y, target_theta, chaser_vx, chaser_vy, chaser_omega, target_vx, target_vy, target_omega] *# Relative pose expressed in the chaser's body frame; everything else in Inertial frame #*
            # Network input: [relative_x, relative_y, relative_angle, chaser_theta, chaser_vx, chaser_vy, chaser_omega, target_omega] ** Normalize it first **
            
            # Get data from Message Parser
            try:
                Pi_time, Pi_red_x, Pi_red_y, Pi_red_theta, \
                Pi_red_Vx, Pi_red_Vy, Pi_red_omega,        \
                Pi_black_x, Pi_black_y, Pi_black_theta,    \
                Pi_black_Vx, Pi_black_Vy, Pi_black_omega,  \
                SPOTNet_relative_x, SPOTNet_relative_y, SPOTNet_relative_angle, SPOTNet_sees_target = self.messages_to_deep_guidance.pop()
            except IndexError:
                # Queue was empty, try agian
                #print("Queue was empty!")
                continue
                        
            #################################
            ### Building the Policy Input ###
            ################################# 
            if SPOTNet_sees_target:
                policy_input = np.array([SPOTNet_relative_x - self.offset_x, SPOTNet_relative_y - self.offset_y, SPOTNet_relative_angle - self.offset_angle, Pi_red_theta, Pi_red_Vx, Pi_red_Vy, Pi_red_omega, Pi_black_omega])
            else:
                # Calculating the relative X and Y in the chaser's body frame
                relative_pose_inertial = np.array([Pi_black_x - Pi_red_x, Pi_black_y - Pi_red_y])
                relative_pose_body = np.matmul(make_C_bI(Pi_red_theta), relative_pose_inertial)
                policy_input = np.array([relative_pose_body[0] - self.offset_x, relative_pose_body[1] - self.offset_y, (Pi_black_theta - Pi_red_theta - self.offset_angle)%(2*np.pi), Pi_red_theta, Pi_red_Vx, Pi_red_Vy, Pi_red_omega, Pi_black_omega])
    
            # Normalizing            
            if Settings.NORMALIZE_STATE:
                normalized_policy_input = (policy_input - relevant_state_mean)/relevant_half_range
            else:
                normalized_policy_input = policy_input
                
            # Reshaping the input
            normalized_policy_input = normalized_policy_input.reshape([-1, Settings.OBSERVATION_SIZE])
    
            # Run processed state through the policy
            deep_guidance = self.sess.run(self.actor.action_scaled, feed_dict={self.state_placeholder:normalized_policy_input})[0] # [accel_x, accel_y, alpha]
            
            # Rotating the command into the inertial frame
            deep_guidance[:-1] = np.matmul(make_C_bI(Pi_red_theta).T,deep_guidance[:-1])
     
            # Commanding constant values in the inertial frame for testing purposes
            #deep_guidance[0] = -0.02 # [m/s^2]
            #deep_guidance[1] = 0.02 # [m/s^2]
            #deep_guidance[2] = -np.pi/60 # [rad/s^2]														  
            #################################################################
            ### Cap output if we are exceeding the max allowable velocity ###
            #################################################################
            # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
    			# this is in the inertial frame							   
            current_velocity = np.array([Pi_red_Vx, Pi_red_Vy, Pi_red_omega])
            deep_guidance[(np.abs(current_velocity) > Settings.VELOCITY_LIMIT) & (np.sign(deep_guidance) == np.sign(current_velocity))] = 0  
    
            # Return commanded action to the Raspberry Pi 3
            if self.testing:
                print(deep_guidance)
                pass
            
            else:
                deep_guidance_acceleration_signal_to_pi = str(deep_guidance[0]) + "\n" + str(deep_guidance[1]) + "\n" + str(deep_guidance[2]) + "\n"
                self.client_socket.send(deep_guidance_acceleration_signal_to_pi.encode())
            
            if counter % 2000 == 0:
                print("Output to Pi: ", deep_guidance, " In table inertial frame")
                print(normalized_policy_input)
            # Incrementing the counter
            counter = counter + 1
            
            # Log this timestep's data
            data_log.append([Pi_time, deep_guidance[0], deep_guidance[1], deep_guidance[2], \
                             Pi_red_x, Pi_red_y, Pi_red_theta, \
                             Pi_red_Vx, Pi_red_Vy, Pi_red_omega,        \
                             Pi_black_x, Pi_black_y, Pi_black_theta,    \
                             Pi_black_Vx, Pi_black_Vy, Pi_black_omega,  \
                             SPOTNet_relative_x, SPOTNet_relative_y, SPOTNet_relative_angle, SPOTNet_sees_target])
        
        print("Model gently stopped.")

        print("Saving data to file...",end='')
        with open('deep_guidance_data_' + time.strftime('%Y-%m-%d-%H_%M-%S', time.localtime()) + '.txt', 'wb') as f:
                np.save(f, np.asarray(data_log))
        print("Done!")
        # Close tensorflow session
        self.sess.close()




# Are we testing?
testing = False

##################################################
#### Start communication with JetsonRepeater #####
##################################################
if testing:
    client_socket = 0
else:
    # Looping forever until we are connected
    while True:
        try: # Try to connect
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect("/tmp/jetsonRepeater") # Connecting...
            client_socket.settimeout(2) # Setting the socket timeout to 2 seconds
            print("Connected to JetsonRepeater!")
            break
        except: # If connection attempt failed
            print("Connection to JetsonRepeater FAILED. Trying to re-connect in 1 second")
            time.sleep(1)
    # WE ARE CONNECTED 

# Generate Queues
messages_to_deep_guidance = deque(maxlen = 1)

#####################
### START THREADS ###
#####################
all_threads = []
stop_run_flag = threading.Event() # Flag to stop all threads 
# Initialize Message Parser
message_parser = MessageParser(testing, client_socket, messages_to_deep_guidance, stop_run_flag)
# Initialize Deep Guidance Model
deep_guidance_model = DeepGuidanceModelRunner(testing, client_socket, messages_to_deep_guidance, stop_run_flag)
       
all_threads.append(threading.Thread(target = message_parser.run))
all_threads.append(threading.Thread(target = deep_guidance_model.run))

#############################################
##### STARTING EXECUTION OF ALL THREADS #####
#############################################
#                                           #
#                                           #
for each_thread in all_threads:             #
#                                           #
    each_thread.start()                     #
#                                           #
#                                           #
#############################################
############## THREADS STARTED ##############
#############################################
counter_2 = 1   
try:       
    while True:
        time.sleep(0.5)
        if counter_2 % 200 == 0:
            print("100 seconds in, trying to stop gracefully")
            stop_run_flag.set()
            for each_thread in all_threads:
                each_thread.join()
            break
except KeyboardInterrupt:
    print("Interrupted by user. Ending gently")
    stop_run_flag.set()
    for each_thread in all_threads:
            each_thread.join()

        

print('Done :)')
