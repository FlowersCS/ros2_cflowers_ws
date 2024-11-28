#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from time import time
from datetime import datetime
from math import degrees
import matplotlib.pyplot as plt
import numpy as np

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from simulations.Qlearning import QLearning
from simulations.Lidar import Lidar
from simulations.Control import Control

# Episode parameters
MAX_EPISODES = 400
MAX_STEPS_PER_EPISODE = 500
MIN_TIME_BETWEEN_ACTIONS = 0.0

# Learning parameters
ALPHA = 0.5
GAMMA = 0.9

T_INIT = 25
T_GRAD = 0.95
T_MIN = 0.001

EPSILON_INIT = 0.9
EPSILON_GRAD = 0.96
EPSILON_MIN = 0.05

# 1 - Softmax , 2 - Epsilon greedy
EXPLORATION_FUNCTION = 1

# Initial position
X_INIT = -0.4
Y_INIT = -0.4
THETA_INIT = 45.0

RANDOM_INIT_POS = False

# Log file directory
DATA_PATH = '/home/flowers/ros2_cflowers_ws/src/simulations/data'
LOG_FILE_DIR = DATA_PATH + '/Log_learning'

# Q table source file
Q_SOURCE_DIR = ''


class LearningNode(Node):

    def __init__(self):
        super().__init__('learning_node')
        self.qlearning = QLearning()
        self.lidar = Lidar()
        self.control = Control()
        
        self.set_params()
        self.init_learning()
        
        # Set up publishers
        self.setPosPub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create a subscriber to the LaserScan topic
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        self.timer = self.create_timer(0.1, self.main_loop)  # Timer for main loop, 10Hz
        self.t_step = self.get_clock().now()
        
        # Initialize state variables
        self.robot_in_pos = False
        self.first_action_taken = False
        self.ep_steps = 0
        self.ep_reward = 0
        self.episode = 1
        self.crash = 0
        self.reward_min_per_episode = np.array([])
        self.reward_max_per_episode = np.array([])
        self.reward_avg_per_episode = np.array([])
        self.ep_reward_arr = np.array([])

    def lidar_callback(self, msgScan):
        """Callback for Lidar messages."""
        # Store the received LaserScan message for later use in the main loop
        self.msgScan = msgScan

    def set_params(self):
        """Initialize parameters for learning."""
        self.T = T_INIT
        self.EPSILON = EPSILON_INIT
        self.alpha = ALPHA
        self.gamma = GAMMA

        self.log_sim_info = open(LOG_FILE_DIR + '/LogInfo.txt', 'w+')
        self.log_sim_params = open(LOG_FILE_DIR + '/LogParams.txt', 'w+')

        now_start = datetime.now()
        dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")
        self.log_sim_info.write(f'SIMULATION START ==> {dt_string_start}\n')

        # Log simulation parameters (example)
        self.log_sim_params.write(f'INITIAL POSITION = {"RANDOM" if RANDOM_INIT_POS else f"({X_INIT}, {Y_INIT}, {THETA_INIT})"}\n')

    def init_learning(self):
        """Initialize the learning parameters."""
        self.actions = self.qlearning.createActions()
        self.state_space = self.qlearning.createStateSpace()
        if Q_SOURCE_DIR != '':
            self.Q_table = self.qlearning.readQTable(Q_SOURCE_DIR + '/Qtable.csv')
        else:
            self.Q_table = self.qlearning.createQTable(len(self.state_space), len(self.actions))
        self.get_logger().info(f'Initial Q-table:\n{self.Q_table}')

    def main_loop(self):
        """Main loop for simulation and learning."""
        if self.robot_in_pos:
            step_time = (self.get_clock().now() - self.t_step).seconds
            if step_time > MIN_TIME_BETWEEN_ACTIONS:
                self.t_step = self.get_clock().now()
                if self.episode > MAX_EPISODES:
                    self.end_learning()
                else:
                    self.run_episode()

    def end_learning(self):
        """End the learning process and log data."""
        sim_time = (self.get_clock().now() - self.t_step).seconds
        self.log_sim_info.write(f'Simulation finished at {sim_time} seconds\n')

        # Save Q-table and other data
        self.qlearning.saveQTable(LOG_FILE_DIR + '/Qtable.csv', self.Q_table)

        self.log_sim_info.close()
        self.log_sim_params.close()
        self.get_logger().info('Learning finished, shutting down.')
        rclpy.shutdown()

    def run_episode(self):
        """Run an individual learning episode."""
        self.ep_steps += 1
        if self.crash or self.ep_steps >= MAX_STEPS_PER_EPISODE:
            self.finish_episode()
        else:
            self.take_action()

    def finish_episode(self):
        """Finish the current episode."""
        self.robotStop()
        # Log episode results
        text = f'Episode {self.episode} finished at step {self.ep_steps}\n'
        self.log_sim_info.write(text)

        self.episode += 1
        self.ep_steps = 0
        self.ep_reward = 0

    def take_action(self):
        """Perform a learning action (Q-learning)."""
        lidar, angles = self.lidar.lidarScan(self.msgScan)
        state_ind, *_ = self.lidar.scanDiscretization(self.state_space, lidar)

        if EXPLORATION_FUNCTION == 1:
            action, _ = self.qlearning.softMaxSelection(self.Q_table, state_ind, self.actions, self.T)
        else:
            action, _ = self.qlearning.epsiloGreedyExploration(self.Q_table, state_ind, self.actions, self.T)

        self.control.robotDoAction(self.velPub, action)
        self.ep_reward += self.get_reward(action, lidar)
        self.update_q_table(state_ind, action)

    def get_reward(self, action, lidar):
        """Calculate the reward."""
        # Placeholder reward function
        return 1

    def update_q_table(self, state_ind, action):
        """Update the Q-table."""
        # Placeholder update function
        pass

    def robotStop(self):
        """Stop the robot by sending zero velocities."""
        stop_msg = Twist()
        self.velPub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LearningNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

