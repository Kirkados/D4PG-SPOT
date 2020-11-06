
"""
This script provides the environment for a docking experiment in the SPOT
facility at Carleton University.

All dynamic environments I create will have a standardized architecture. The
reason for this is I have one learning algorithm and many environments. All
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - seeding the dynamics (via the seed method)
    - animating the motion (via the render method):
        - Rendering is done all in one shot by passing the completed states
          from a trial to the render() method.

Outputs:
    Reward must be of shape ()
    State must be of shape (OBSERVATION_SIZE,)
    Done must be a bool

Inputs:
    Action input is of shape (ACTION_SIZE,)

Communication with agent:
    The agent communicates to the environment through two queues:
        agent_to_env: the agent passes actions or reset signals to the environment
        env_to_agent: the environment returns information to the agent

Reward system:
        - Zero reward at all timesteps except when docking is achieved
        - A large reward when docking occurs. The episode also terminates when docking occurs
        - A variety of penalties to help with docking, such as:
            - penalty for end-effector angle (so it goes into the docking cone properly)
            - penalty for relative velocity during the docking (so the end-effector doesn't jab the docking cone)
        - A penalty for colliding with the target

State clarity:
    - self.dynamic_state contains the chaser states propagated in the dynamics
    - self.observation is passed to the agent and is a combination of the dynamic
                       state and the target position
    - self.OBSERVATION_SIZE 


Started November 5, 2020
@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import os
import signal
import multiprocessing
import queue
from scipy.integrate import odeint # Numerical integrator
import faulthandler # Added July 11 2020 to debug "segmentation faule (core dumped)" error

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

class Environment:

    def __init__(self):
        ##################################
        ##### Environment Properties #####
        ##################################
        self.TOTAL_STATE_SIZE         = 18 # [relative_x, relative_y, relative_vx, relative_vy, relative_angle, relative_angular_velocity, chaser_x, chaser_y, chaser_theta, target_x, target_y, target_theta, chaser_vx, chaser_vy, chaser_omega, target_vx, target_vy, target_omega] *# All expressed in the chaser's body frame #*
        ### Note: TOTAL_STATE contains all relevant information describing the problem, and all the information needed to animate the motion
        #         TOTAL_STATE is returned from the environment to the agent.
        #         A subset of the TOTAL_STATE, called the 'observation', is passed to the policy network to calculate acitons. This takes place in the agent
        #         The TOTAL_STATE is passed to the animator below to animate the motion.
        #         The chaser and target state are contained in the environment. They are packaged up before being returned to the agent.
        #         The total state information returned must be as commented beside self.TOTAL_STATE_SIZE.
        self.IRRELEVANT_STATES                = [6,7,8,9,10,11,12,13,14,15,16,17] # indices of states who are irrelevant to the policy network
        self.OBSERVATION_SIZE                 = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy
        self.ACTION_SIZE                      = 3 # [x_dot_dot, y_dot_dot, theta_dot_dot] in the BODY frame
        self.VELOCITY_LIMIT                   = 0.5 # [m/s] maximum allowable velocity, a hard cap is enforced if this velocity is exceeded
        self.LOWER_ACTION_BOUND               = np.array([-0.025, -0.025, -0.001]) # [m/s^2, m/s^2, rad/s^2]
        self.UPPER_ACTION_BOUND               = np.array([ 0.025,  0.025,  0.001]) # [m/s^2, m/s^2, rad/s^2]
        self.LOWER_STATE_BOUND                = np.array([-3., -3., -self.VELOCITY_LIMIT, -self.VELOCITY_LIMIT, -2*np.pi, -np.pi/6, -3., -3., -2*np.pi, -3., -3., -2*np.pi, -self.VELOCITY_LIMIT, -self.VELOCITY_LIMIT, -np.pi/6, -self.VELOCITY_LIMIT, -self.VELOCITY_LIMIT, -np.pi/6]) # [m, m, m/s, m/s, rad, rad/s, m, m, rad, m, m, rad, m/s, m/s, rad/s, m/s, m/s, rad/s] // lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND                = np.array([ 3.,  3.,  self.VELOCITY_LIMIT,  self.VELOCITY_LIMIT,  2*np.pi,  np.pi/6,  3.,  3.,  2*np.pi,  3.,  3.,  2*np.pi,  self.VELOCITY_LIMIT,  self.VELOCITY_LIMIT,  np.pi/6,  self.VELOCITY_LIMIT,  self.VELOCITY_LIMIT,  np.pi/6]) # [m, m, m,s, m,s, rad, rad/s, m, m, rad, m, m, rad, m/s, m/s, rad/s, m/s, m/s, rad/s] // upper bound for each element of TOTAL_STATE
        self.INITIAL_CHASER_POSITION          = np.array([3.0, 1.2, 0.0]) # [m, m, rad]
        self.INITIAL_CHASER_VELOCITY          = np.array([0.0, 0.0, 0.0]) # [m/s, m/s, rad/s]
        self.INITIAL_TARGET_POSITION          = np.array([2.0, 1.0, 0.0]) # [m, m, rad]
        self.INITIAL_TARGET_VELOCITY          = np.array([0.0, 0.0, 0.0]) # [m/s, m/s, rad/s]
        self.NORMALIZE_STATE                  = True # Normalize state on each timestep to avoid vanishing gradients
        self.RANDOMIZE                        = True # whether or not to RANDOMIZE the state & target location
        self.RANDOMIZATION_LENGTH             = 1 # [m] standard deviation of position randomization
        self.RANDOMIZATION_ANGLE              = np.pi/2 # [rad] standard deviation of angular randomization
        self.RANDOMIZATION_TARGET_VELOCITY    = 0.0 # [m/s] standard deviation of the target's velocity randomization
        self.RANDOMIZATION_TARGET_OMEGA       = 0.0 # [rad/s] standard deviation of the target's angular velocity randomization
        self.MIN_V                            = -100.
        self.MAX_V                            =  100.
        self.N_STEP_RETURN                    =   5
        self.DISCOUNT_FACTOR                  =   0.95**(1/self.N_STEP_RETURN)
        self.TIMESTEP                         =   0.2 # [s]
        self.DYNAMICS_DELAY                   =   0 # [timesteps of delay] how many timesteps between when an action is commanded and when it is realized
        self.AUGMENT_STATE_WITH_ACTION_LENGTH =   0 # [timesteps] how many timesteps of previous actions should be included in the state. This helps with making good decisions among delayed dynamics.
        self.MAX_NUMBER_OF_TIMESTEPS          = 200 # per episode
        self.ADDITIONAL_VALUE_INFO            = False # whether or not to include additional reward and value distribution information on the animations
        self.SKIP_FAILED_ANIMATIONS           = True # Error the program or skip when animations fail?

        # Reward function properties
        self.DOCKING_REWARD            = 100 # A lump-sum given to the chaser when it docks
        self.DOCKING_ANGLE_PENALTY     = 1 # A penalty given to the chaser, upon docking, for having an angle when docking
        self.DOCKING_VELOCITY_PENALTY  = 1 # A penalty given to the chaser, upon docking, for having residual velocity and therefore bumping the target
        self.TARGET_COLLISION_PENALTY  = 15           # [rewards/second] penalty given for colliding with target  
        self.TARGET_COLLISION_DISTANCE = np.sqrt(2)*self.LENGTH # [m] how close chaser and target need to be before a penalty is applied
        self.END_ON_FALL               = False # end episode on a fall off the table        
        self.FALL_OFF_TABLE_PENALTY    =  100.

        # Physical properties
        self.LENGTH                = 0.3  # [m] side length
        self.MASS                  = 10.0   # [kg] for chaser
        self.INERTIA               = 1/12*self.MASS*(self.LENGTH**2 + self.LENGTH**2) # 0.15 [kg m^2]
        self.END_EFFECTOR_POSITION = [self.LENGTH, self.LENGTH] # position of the optimally-deployed end-effector on the chaser in the body frame
        self.DOCKING_PORT_POSITION = [-self.LENGTH/2, 0] # position of the docking cone on the target in irs body frame
        
        # Test time properties
        self.TEST_ON_DYNAMICS            = True # Whether or not to use full dynamics along with a PD controller at test time
        self.KINEMATIC_NOISE             = False # Whether or not to apply noise to the kinematics in order to simulate a poor controller
        self.KINEMATIC_POSITION_NOISE_SD = [0.2, 0.2, 0.2] # The standard deviation of the noise that is to be applied to each position element in the state
        self.KINEMATIC_VELOCITY_NOISE_SD = [0.1, 0.1, 0.1] # The standard deviation of the noise that is to be applied to each velocity element in the state
        self.FORCE_NOISE_AT_TEST_TIME    = False # [Default -> False] Whether or not to force kinematic noise to be present at test time
        self.KI                          = 0.5 # Integral gain for the integral-linear acceleration controller
        

        # Some calculations that don't need to be changed
        self.LOWER_STATE_BOUND        = np.concatenate([self.LOWER_STATE_BOUND, np.tile(self.LOWER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND        = np.concatenate([self.UPPER_STATE_BOUND, np.tile(self.UPPER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # upper bound for each element of TOTAL_STATE        
        self.OBSERVATION_SIZE         = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy

    ###################################
    ##### Seeding the environment #####
    ###################################
    def seed(self, seed):
        np.random.seed(seed)

    ######################################
    ##### Resettings the Environment #####
    ######################################
    def reset(self, use_dynamics, test_time):
        # This method resets the state and returns it
        """ NOTES:
               - if use_dynamics = True -> use dynamics
               - if test_time = True -> do not add "controller noise" to the kinematics
        """
        # Setting the default to be kinematics
        self.dynamics_flag = False

        # Logging whether it is test time for this episode
        self.test_time = test_time

        # If we are randomizing the initial conditions and state
        if self.RANDOMIZE:
            # Randomizing initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION + np.random.randn(3)*[self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_ANGLE]
            # Randomizing target state in Inertial frame
            self.target_location = self.INITIAL_TARGET_POSITION + np.random.randn(3)*[self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_ANGLE]
            # Randomizing target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY + np.random.randn(3)*[self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_OMEGA]
            

        else:
            # Constant initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION
            # Constant target location in Inertial frame
            self.target_location = self.INITIAL_TARGET_POSITION
            # Constant target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY
        
        # Resetting the chaser's initial velocity
        self.chaser_velocity = self.INITIAL_CHASER_VELOCITY
        
        # End effector position in the Inertial frame
        #TODO: calculate end-effector position
        self.end_effector_position = self.chaser_location
        
        # Docking port location in Inertial frame
        #TODO: calculate docking port location
        self.docking_port = self.target_location[:-1] + np.array([np.cos(self.target_location[3])*0.5, np.sin(self.target_location[3])*0.5, 0.])
            
        # Initializing the previous velocity and control effort for the integral-acceleration controller
        self.previous_velocity = np.zeros(len(self.INITIAL_CHASER_VELOCITY))
        self.previous_control_effort = np.zeros(len(self.ACTION_SIZE))
                
        if use_dynamics:            
            self.dynamics_flag = True # for this episode, dynamics will be used

        # Resetting the time
        self.time = 0.

        # Resetting the action delay queue
        if self.DYNAMICS_DELAY > 0:
            self.action_delay_queue = queue.Queue(maxsize = self.DYNAMICS_DELAY + 1)
            for i in range(self.DYNAMICS_DELAY):
                self.action_delay_queue.put(np.zeros(self.ACTION_SIZE + 1), False)

    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):

        # Integrating forward one time step using the calculated action.
        # Oeint returns initial condition on first row then next TIMESTEP on the next row
        #########################################
        ##### PROPAGATE KINEMATICS/DYNAMICS #####
        #########################################
        if self.dynamics_flag:
            ############################
            #### PROPAGATE DYNAMICS ####
            ############################

            # First, calculate the control effort
            control_effort = self.controller(action)

            # Anything that needs to be sent to the dynamics integrator
            dynamics_parameters = [control_effort, self.MASS, self.INERTIA]

            # Propagate the dynamics forward one timestep
            next_states = odeint(dynamics_equations_of_motion, np.concatenate([self.chaser_position, self.chaser_velocity]), [self.time, self.time + self.TIMESTEP], args = (dynamics_parameters,), full_output = 0)

            # Saving the new state
            self.chaser_position = next_states[1,:len(self.INITIAL_CHASER_POSITION)] # extract position
            self.chaser_velocity = next_states[1,len(self.INITIAL_CHASER_POSITION):] # extract velocity

        else:

            # Parameters to be passed to the kinematics integrator
            kinematics_parameters = [action, len(self.INITIAL_CHASER_POSITION)]

            ###############################
            #### PROPAGATE KINEMATICS #####
            ###############################
            next_states = odeint(kinematics_equations_of_motion, np.concatenate([self.chaser_position, self.chaser_velocity]), [self.time, self.time + self.TIMESTEP], args = (kinematics_parameters,), full_output = 0)

            # Saving the new state
            self.chaser_position = next_states[1,:len(self.INITIAL_CHASER_POSITION)] # extract position
            self.chaser_velocity = next_states[1,len(self.INITIAL_CHASER_POSITION):] # extract velocity  
            
            # Optionally, add noise to the kinematics to simulate "controller noise"
            if self.KINEMATIC_NOISE and (not self.test_time or self.FORCE_NOISE_AT_TEST_TIME):
                 # Add some noise to the position part of the state
                 self.chaser_position += np.random.randn(len(self.chaser_position)) * self.KINEMATIC_POSITION_NOISE_SD
                 self.chaser_velocity += np.random.randn(len(self.chaser_velocity)) * self.KINEMATIC_VELOCITY_NOISE_SD
            
            # Ensuring the velocity is within the bounds
            self.chaser_velocity = np.clip(self.chaser_velocity, -self.VELOCITY_LIMIT, self.VELOCITY_LIMIT)

start here
        # Step target's attitude ahead one timestep
        self.target_location[3] += self.TARGET_ANGULAR_VELOCITY*self.TIMESTEP

        # Update the docking port target state
        self.docking_port = self.target_location[:-1] + np.array([np.cos(self.target_location[3])*0.5, np.sin(self.target_location[3])*0.5, 0.])

        # Update the hold point location
        self.hold_point   = self.target_location[:-1] + np.array([np.cos(self.target_location[3])*self.HOLD_POINT_DISTANCE, np.sin(self.target_location[3])*self.HOLD_POINT_DISTANCE, 0.])


        # Increment the timestep
        self.time += self.TIMESTEP

        # Calculating the reward for this state-action pair
        reward = self.reward_function(action)

        # Check if this episode is done
        done = self.is_done()

        # Return the (reward, done)
        return reward, done

    def controller(self, action):
        # This function calculates the control effort based on the state and the
        # desired acceleration (action)
        
        ###########################################################
        ### Position control (integral-acceleration controller) ###
        ###########################################################
        desired_linear_acceleration = action
        
        current_velocity = self.chaser_velocity # [v_x, v_y, v_z]
        current_linear_acceleration = (current_velocity - self.previous_velocity)/self.TIMESTEP # Approximating the current acceleration [a_x, a_y, a_z]
        
        # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
        desired_linear_acceleration[(np.abs(current_velocity) > self.VELOCITY_LIMIT) & (np.sign(desired_linear_acceleration) == np.sign(current_velocity))] = 0        
        
        # Calculating acceleration error
        linear_acceleration_error = desired_linear_acceleration - current_linear_acceleration
        
        # Integral-acceleration control
        linear_control_effort = self.previous_linear_control_effort + self.KI * linear_acceleration_error

        # Saving the current velocity for the next timetsep
        self.previous_velocity = current_velocity
        
        # Saving the current control effort for the next timestep
        self.previous_linear_control_effort = linear_control_effort

        return linear_control_effort

    def pose_error(self):
        """
        This method returns the pose error of the current state.
        Instead of returning [state, desired_state] as the state, I'll return
        [state, error]. The error will be more helpful to the policy I believe.
        """
        if self.phase_number == 0:
            return self.hold_point - self.chaser_position
        elif self.phase_number == 1:
            return self.docking_port - self.chaser_position


    def reward_function(self, action):
        # Returns the reward for this TIMESTEP as a function of the state and action

        # Sets the current location that we are trying to move to
        desired_location = self.hold_point

        current_position_reward = np.zeros(1)

        # Calculates a reward map
        if self.REWARD_TYPE:
            # Linear reward
            current_position_reward = -np.abs((desired_location - self.chaser_position)*self.REWARD_WEIGHTING)* self.TARGET_REWARD
        else:
            # Exponential reward
            current_position_reward = np.exp(-np.sum(np.absolute(desired_location - self.chaser_position)*self.REWARD_WEIGHTING)) * self.TARGET_REWARD

        reward = np.zeros(1)

        # If it's not the first timestep, calculate the differential reward
        if np.all([self.previous_position_reward[i] is not None for i in range(len(self.previous_position_reward))]):
            reward = (current_position_reward - self.previous_position_reward)*self.REWARD_MULTIPLIER
            for i in range(len(reward)):
                if reward[i] < 0:
                    reward[i]*= self.NEGATIVE_PENALTY_FACTOR

        self.previous_position_reward = current_position_reward

        # Collapsing to a scalar
        reward = np.sum(reward)

        # Giving a massive penalty for falling off the table
        if self.chaser_position[0] > self.UPPER_STATE_BOUND[0] or self.chaser_position[0] < self.LOWER_STATE_BOUND[0] or self.chaser_position[1] > self.UPPER_STATE_BOUND[1] or self.chaser_position[1] < self.LOWER_STATE_BOUND[1]:
            reward -= self.FALL_OFF_TABLE_PENALTY/self.TIMESTEP

        # Giving a large reward for completing the task
        if np.sum(np.absolute(self.chaser_position - desired_location)) < 0.01:
            reward += self.GOAL_REWARD
            
        # Giving a large penalty for colliding with the obstacle
        if np.linalg.norm(self.chaser_position - self.obstacle_location) <= self.OBSTABLE_DISTANCE and self.USE_OBSTACLE:
            reward -= self.OBSTABLE_PENALTY
            
        # Giving a penalty for colliding with the target
        if np.linalg.norm(self.chaser_position[:-1] - self.target_location[:-2]) <= self.TARGET_COLLISION_DISTANCE:
            reward -= self.TARGET_COLLISION_PENALTY
            
        # Giving a penalty for high velocities near the target location
        if self.PENALIZE_VELOCITY:
            radius = np.linalg.norm(desired_location- self.target_location[:2]) # vector from the target to the desired location
            reference_velocity = self.TARGET_ANGULAR_VELOCITY*np.array([-radius*np.sin(self.target_location[2]), radius*np.cos(self.target_location[2]), 0, 1])
            reward -= np.sum(np.abs(action - reference_velocity)/(self.pose_error()**2+0.01)*self.VELOCITY_PENALTY)
        
        # Giving a penalty for exceeding the recommended maximum velocity (decided by Murat)
        if self.PENALIZE_MAX_VELOCITY:
            reward -= self.MAX_VELOCITY_PENALTY*np.sum(np.abs(self.chaser_velocity) > self.VELOCITY_LIMIT)
            
        
        # Penalizing acceleration commands (to encourage fuel efficiency)
        reward -= np.sum(self.ACCELERATION_PENALTY*np.abs(action))

        # Multiplying the reward by the TIMESTEP to give the rewards on a per-second basis
        return (reward*self.TIMESTEP).squeeze()

    def is_done(self):
        # Checks if this episode is done or not
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """

        # If we've fallen off the table, end the episode
        if self.chaser_position[0] > self.UPPER_STATE_BOUND[0] or self.chaser_position[0] < self.LOWER_STATE_BOUND[0] or self.chaser_position[1] > self.UPPER_STATE_BOUND[1] or self.chaser_position[1] < self.LOWER_STATE_BOUND[1] or self.chaser_position[2] > self.UPPER_STATE_BOUND[2] or self.chaser_position[2] < self.LOWER_STATE_BOUND[2]:
            done = self.END_ON_FALL
        else:
            done = False

        # If we've run out of timesteps
        if round(self.time/self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            done = True

        return done


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_env, self.env_to_agent
    
    def obstable_relative_location(self):
        # Returns the position of the obstacle with respect to the chaser
        relative_position = self.obstacle_location - self.chaser_position[:len(self.chaser_position)-1]
        
        return relative_position

    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        """
        # Instructing this process to treat Ctrl+C events (called SIGINT) by going SIG_IGN (ignore).
        # This permits the process to continue upon a Ctrl+C event to allow for graceful quitting.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        
        self.counter = 0 # INITIALIZING
        
        # Loop until the process is terminated
        while True:
            # Blocks until the agent passes us an action
            action, *test_time = self.agent_to_env.get()        

            if type(action) == bool:
                # The signal to reset the environment was received
                self.reset(action, test_time[0])
                
                # Return the TOTAL_STATE
                total_state = np.concatenate([self.chaser_position, np.concatenate([self.target_location, self.chaser_velocity]) ])
                self.env_to_agent.put(total_state)

            else:
                # Delay the action by DYNAMICS_DELAY timesteps. The environment accumulates the action delay--the agent still thinks the sent action was used.
                if self.DYNAMICS_DELAY > 0:
                    self.action_delay_queue.put(action,False) # puts the current action to the bottom of the stack
                    action = self.action_delay_queue.get(False) # grabs the delayed action and treats it as truth.                

                ################################
                ##### Step the environment #####
                ################################                
                reward, done = self.step(action)

                if (self.counter % 1000) == 0:
                    pass
                    #print("Environment.py PID %i is using %2.3f GB of RAM" %(os.getpid(), self.process.memory_info().rss/1000000000.0))
                self.counter += 1

                # Return (TOTAL_STATE, reward, done, guidance_position)
                self.env_to_agent.put((np.concatenate([self.chaser_position, np.concatenate([self.target_location, self.chaser_velocity]) ]), reward, done))

###################################################################
##### Generating kinematics equations representing the motion #####
###################################################################
def kinematics_equations_of_motion(state, t, parameters):
    # From the state, it returns the first derivative of the state
    
    # Unpacking the action from the parameters
    action = parameters[0]
    position_length = parameters[1]
    
    # state is [position, velocity]
    # its derivative is [velocity, acceleration]
    #position = state[:position_length] # [x, y, z]
    velocity = state[position_length:] # [x_dot, y_dot, z_dot]
    
    acceleration = action # [x_dot_dot, y_dot_dot, z_dot_dot]

    # Building the derivative matrix.
    derivatives = np.concatenate([velocity, acceleration])

    return derivatives


#####################################################################
##### Generating the dynamics equations representing the motion #####
#####################################################################
def dynamics_equations_of_motion(state, t, parameters):
    # state = [chaser_x, chaser_y, chaser_z, chaser_theta, chaser_Vx, chaser_Vy, chaser_Vz]

    # Unpacking the state
    x, y, z, xdot, ydot, zdot = state
    control_effort, mass, inertia = parameters # unpacking parameters

    derivatives = np.array((xdot, ydot, zdot, control_effort[0]/mass, control_effort[1]/mass, control_effort[2]/mass)).squeeze()
    
    #print("The achieved acceleration is ", control_effort[0]/mass, control_effort[1]/mass, control_effort[2]/mass, control_effort[3]/inertia)

    return derivatives


##########################################
##### Function to animate the motion #####
##########################################
def render(states, actions, instantaneous_reward_log, cumulative_reward_log, critic_distributions, target_critic_distributions, projected_target_distribution, bins, loss_log, guidance_position_log, episode_number, filename, save_directory):

    faulthandler.enable() # to get more information about segmentation fault (core dumped) error 
    
    # Load in a temporary environment, used to grab the physical parameters
    temp_env = Environment()

    # Checking if we want the additional reward and value distribution information
    extra_information = temp_env.ADDITIONAL_VALUE_INFO

    # Unpacking state
    chaser_x, chaser_y, chaser_z = states[:,0], states[:,1], states[:,2]
    
    # Assigning the chaser a fixed attitude
    chaser_theta = np.zeros(len(chaser_x))
    
    target_x, target_y, target_z, target_theta = states[:,3], states[:,4], states[:,5], states[:,6]

    # Extracting physical properties
    length = temp_env.LENGTH

    ### Calculating spacecraft corner locations through time ###
    
    # Corner locations in body frame    
    chaser_body_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                                [[-1],[-1],[1]],
                                                [[-1],[-1],[-1]],
                                                [[1],[-1],[-1]],
                                                [[-1],[-1],[-1]],
                                                [[-1],[1],[-1]],
                                                [[1],[1],[-1]],
                                                [[-1],[1],[-1]],
                                                [[-1],[1],[1]],
                                                [[1],[1],[1]],
                                                [[-1],[1],[1]],
                                                [[-1],[-1],[1]]]).squeeze().T
    
    chaser_front_face_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                                       [[1],[1],[1]],
                                                       [[1],[1],[-1]],
                                                       [[1],[-1],[-1]],
                                                       [[1],[-1],[1]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib = np.moveaxis(np.array([[np.cos(chaser_theta),       -np.sin(chaser_theta),        np.zeros(len(chaser_theta))],
                                 [np.sin(chaser_theta),        np.cos(chaser_theta),        np.zeros(len(chaser_theta))],
                                 [np.zeros(len(chaser_theta)), np.zeros(len(chaser_theta)), np.ones(len(chaser_theta))]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 3, 3]
    
    # Rotating body frame coordinates to inertial frame
    chaser_body_inertial       = np.matmul(C_Ib, chaser_body_body_frame)       + np.array([chaser_x, chaser_y, chaser_z]).T.reshape([-1,3,1])
    chaser_front_face_inertial = np.matmul(C_Ib, chaser_front_face_body_frame) + np.array([chaser_x, chaser_y, chaser_z]).T.reshape([-1,3,1])

    ### Calculating target spacecraft corner locations through time ###
    
    # Corner locations in body frame    
    target_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                           [[-1],[-1],[1]],
                                           [[-1],[-1],[-1]],
                                           [[1],[-1],[-1]],
                                           [[-1],[-1],[-1]],
                                           [[-1],[1],[-1]],
                                           [[1],[1],[-1]],
                                           [[-1],[1],[-1]],
                                           [[-1],[1],[1]],
                                           [[1],[1],[1]],
                                           [[-1],[1],[1]],
                                           [[-1],[-1],[1]]]).squeeze().T
        
    target_front_face_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                                       [[1],[1],[1]],
                                                       [[1],[1],[-1]],
                                                       [[1],[-1],[-1]],
                                                       [[1],[-1],[1]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib = np.moveaxis(np.array([[np.cos(target_theta),       -np.sin(target_theta),        np.zeros(len(target_theta))],
                                 [np.sin(target_theta),        np.cos(target_theta),        np.zeros(len(target_theta))],
                                 [np.zeros(len(target_theta)), np.zeros(len(target_theta)), np.ones(len(target_theta))]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 3, 3]
    target_body_inertial = np.matmul(C_Ib, target_body_frame)+ np.array([target_x, target_y, target_z]).T.reshape([-1,3,1])
    target_front_face_inertial = np.matmul(C_Ib, target_front_face_body_frame) + np.array([target_x, target_y, target_z]).T.reshape([-1,3,1])

    # Generating figure window
    figure = plt.figure(constrained_layout = True)
    figure.set_size_inches(5, 4, True)

    if extra_information:
        grid_spec = gridspec.GridSpec(nrows = 2, ncols = 3, figure = figure)
        subfig1 = figure.add_subplot(grid_spec[0,0], projection = '3d', aspect = 'equal', autoscale_on = False, xlim3d = (-5, 5), ylim3d = (-5, 5), zlim3d = (0, 10), xlabel = 'X (m)', ylabel = 'Y (m)', zlabel = 'Z (m)')
        subfig2 = figure.add_subplot(grid_spec[0,1], xlim = (np.min([np.min(instantaneous_reward_log), 0]) - (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02, np.max([np.max(instantaneous_reward_log), 0]) + (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02), ylim = (-0.5, 0.5))
        subfig3 = figure.add_subplot(grid_spec[0,2], xlim = (np.min(loss_log)-0.01, np.max(loss_log)+0.01), ylim = (-0.5, 0.5))
        subfig4 = figure.add_subplot(grid_spec[1,0], ylim = (0, 1.02))
        subfig5 = figure.add_subplot(grid_spec[1,1], ylim = (0, 1.02))
        subfig6 = figure.add_subplot(grid_spec[1,2], ylim = (0, 1.02))

        # Setting titles
        subfig1.set_xlabel("X (m)",    fontdict = {'fontsize': 8})
        subfig1.set_ylabel("Y (m)",    fontdict = {'fontsize': 8})
        subfig2.set_title("Timestep Reward",    fontdict = {'fontsize': 8})
        subfig3.set_title("Current loss",       fontdict = {'fontsize': 8})
        subfig4.set_title("Q-dist",             fontdict = {'fontsize': 8})
        subfig5.set_title("Target Q-dist",      fontdict = {'fontsize': 8})
        subfig6.set_title("Bellman projection", fontdict = {'fontsize': 8})

        # Changing around the axes
        subfig1.tick_params(labelsize = 8)
        subfig2.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig3.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig4.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig5.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig6.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = True, labelsize = 8)

        # Adding the grid
        subfig4.grid(True)
        subfig5.grid(True)
        subfig6.grid(True)

        # Setting appropriate axes ticks
        subfig2.set_xticks([np.min(instantaneous_reward_log), 0, np.max(instantaneous_reward_log)] if np.sign(np.min(instantaneous_reward_log)) != np.sign(np.max(instantaneous_reward_log)) else [np.min(instantaneous_reward_log), np.max(instantaneous_reward_log)])
        subfig3.set_xticks([np.min(loss_log), np.max(loss_log)])
        subfig4.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig4.tick_params(axis = 'x', labelrotation = -90)
        subfig4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig5.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig5.tick_params(axis = 'x', labelrotation = -90)
        subfig5.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig6.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig6.tick_params(axis = 'x', labelrotation = -90)
        subfig6.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])

    else:
        subfig1 = figure.add_subplot(1, 1, 1, projection = '3d', aspect = 'equal', autoscale_on = False, xlim3d = (-5, 5), ylim3d = (-5, 5), zlim3d = (0, 10), xlabel = 'X (m)', ylabel = 'Y (m)', zlabel = 'Z (m)')
    
    # Setting the proper view
    if temp_env.TOP_DOWN_VIEW:
        subfig1.view_init(-90,0)
    else:
        subfig1.view_init(25, 190)        

    # Defining plotting objects that change each frame
    chaser_body,       = subfig1.plot([], [], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed
    chaser_front_face, = subfig1.plot([], [], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed
    target_body,       = subfig1.plot([], [], [], color = 'g', linestyle = '-', linewidth = 2)
    target_front_face, = subfig1.plot([], [], [], color = 'k', linestyle = '-', linewidth = 2)
    chaser_body_dot    = subfig1.scatter(0., 0., 0., color = 'r', s = 0.1)

    if extra_information:
        reward_bar           = subfig2.barh(y = 0, height = 0.2, width = 0)
        loss_bar             = subfig3.barh(y = 0, height = 0.2, width = 0)
        q_dist_bar           = subfig4.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        target_q_dist_bar    = subfig5.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        projected_q_dist_bar = subfig6.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        time_text            = subfig1.text2D(x = 0.2, y = 0.91, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text          = subfig1.text2D(x = 0.0,  y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)
    else:        
        time_text    = subfig1.text2D(x = 0.1, y = 0.9, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text  = subfig1.text2D(x = 0.62, y = 0.9, s = '', fontsize = 8, transform=subfig1.transAxes)
        episode_text = subfig1.text2D(x = 0.4, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        episode_text.set_text('Episode ' + str(episode_number))

    # Function called repeatedly to draw each frame
    def render_one_frame(frame, *fargs):
        temp_env = fargs[0] # Extract environment from passed args

        # Draw the chaser body
        chaser_body.set_data(chaser_body_inertial[frame,0,:], chaser_body_inertial[frame,1,:])
        chaser_body.set_3d_properties(chaser_body_inertial[frame,2,:])

        # Draw the front face of the chaser body in a different colour
        chaser_front_face.set_data(chaser_front_face_inertial[frame,0,:], chaser_front_face_inertial[frame,1,:])
        chaser_front_face.set_3d_properties(chaser_front_face_inertial[frame,2,:])

        # Draw the target body
        target_body.set_data(target_body_inertial[frame,0,:], target_body_inertial[frame,1,:])
        target_body.set_3d_properties(target_body_inertial[frame,2,:])

        # Draw the front face of the target body in a different colour
        target_front_face.set_data(target_front_face_inertial[frame,0,:], target_front_face_inertial[frame,1,:])
        target_front_face.set_3d_properties(target_front_face_inertial[frame,2,:])

        # Drawing a dot in the centre of the chaser
        chaser_body_dot._offsets3d = ([chaser_x[frame]],[chaser_y[frame]],[chaser_z[frame]])

        # Update the time text
        time_text.set_text('Time = %.1f s' %(frame*temp_env.TIMESTEP))

        # Update the reward text
        reward_text.set_text('Total reward = %.1f' %cumulative_reward_log[frame])
        
        if extra_information:
            # Updating the instantaneous reward bar graph
            reward_bar[0].set_width(instantaneous_reward_log[frame])
            # And colouring it appropriately
            if instantaneous_reward_log[frame] < 0:
                reward_bar[0].set_color('r')
            else:
                reward_bar[0].set_color('g')

            # Updating the loss bar graph
            loss_bar[0].set_width(loss_log[frame])

            # Updating the q-distribution plot
            for this_bar, new_value in zip(q_dist_bar, critic_distributions[frame,:]):
                this_bar.set_height(new_value)

            # Updating the target q-distribution plot
            for this_bar, new_value in zip(target_q_dist_bar, target_critic_distributions[frame, :]):
                this_bar.set_height(new_value)

            # Updating the projected target q-distribution plot
            for this_bar, new_value in zip(projected_q_dist_bar, projected_target_distribution[frame, :]):
                this_bar.set_height(new_value)
#
        # Since blit = True, must return everything that has changed at this frame
        return chaser_body_dot, time_text, chaser_body, chaser_front_face, target_body, target_front_face 

    # Generate the animation!
    fargs = [temp_env] # bundling additional arguments
    animator = animation.FuncAnimation(figure, render_one_frame, frames = np.linspace(0, len(states)-1, len(states)).astype(int),
                                       blit = False, fargs = fargs)

    """
    frames = the int that is passed to render_one_frame. I use it to selectively plot certain data
    fargs = additional arguments for render_one_frame
    interval = delay between frames in ms
    """

    # Save the animation!
    if temp_env.SKIP_FAILED_ANIMATIONS:
        try:
            # Save it to the working directory [have to], then move it to the proper folder
            animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
            # Make directory if it doesn't already exist
            os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
            # Move animation to the proper directory
            os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
        except:
            print("Skipping animation for episode %i due to an error" %episode_number)
            # Try to delete the partially completed video file
            try:
                os.remove(filename + '_episode_' + str(episode_number) + '.mp4')
            except:
                pass
    else:
        # Save it to the working directory [have to], then move it to the proper folder
        animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
        # Make directory if it doesn't already exist
        os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
        # Move animation to the proper directory
        os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')

    del temp_env
    plt.close(figure)