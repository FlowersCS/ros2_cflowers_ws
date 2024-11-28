#! /usr/bin/env python

import rclpy
from rclpy.node import Node
from time import time, sleep
from datetime import datetime
import matplotlib.pyplot as plt
import sys
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from math import radians, degrees

DATA_PATH = '/home/flowers/ros2_cflowers_ws/src/simulations/data'

from simulations.Qlearning import QLearning
from simulations.Lidar import Lidar
from simulations.Control import Control

# Real robot flag
REAL_ROBOT = False

# Action parameter
MIN_TIME_BETWEEN_ACTIONS = 0.0

# Initial and goal positions
INIT_POSITIONS_X = [-0.7, -0.7, -0.5, -1, -2]
INIT_POSITIONS_Y = [-0.7, 0.7, 1, -2, 1]
INIT_POSITIONS_THETA = [45, -45, -120, -90, 150]
GOAL_POSITIONS_X = [2.0, 2.0, 0.5, 1, 2]
GOAL_POSITIONS_Y = [1.0, -1.0, -1.9, 2, -1]
GOAL_POSITIONS_THETA = [25.0, -40.0, -40, 60, -30]

PATH_IND = 4

# Initial & Goal position
if REAL_ROBOT:
    X_INIT = 0.0
    Y_INIT = 0.0
    THETA_INIT = 0.0
    X_GOAL = 1.7
    Y_GOAL = 1.1
    THETA_GOAL = 90
else:
    RANDOM_INIT_POS = False

    X_INIT = INIT_POSITIONS_X[PATH_IND]
    Y_INIT = INIT_POSITIONS_Y[PATH_IND]
    THETA_INIT = INIT_POSITIONS_THETA[PATH_IND]

    X_GOAL = GOAL_POSITIONS_X[PATH_IND]
    Y_GOAL = GOAL_POSITIONS_Y[PATH_IND]
    THETA_GOAL = GOAL_POSITIONS_THETA[PATH_IND]

# Log file directory - Q table source
Q_TABLE_SOURCE = DATA_PATH + '/Log_learning_FINAL'

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.qlearning = QLearning()
        self.lidar = Lidar()
        self.control = Control()

        # Publishers
        self.setPosPub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Actions and state space
        self.actions = self.qlearning.createActions()
        self.state_space = self.qlearning.createStateSpace()
        self.Q_table = self.qlearning.readQTable(Q_TABLE_SOURCE + '/Qtable.csv')
        self.get_logger().info('Initial Q-table: ' + str(self.Q_table))

        # Timers
        self.t_0 = self.get_clock().now()
        self.t_start = self.get_clock().now()

        # Waiting for the clock
        while (self.t_start.nanoseconds - self.t_0.nanoseconds) < 0:
            self.t_start = self.get_clock().now()

        self.t_step = self.t_start
        self.count = 0

        # Robot position initialization
        self.robot_in_pos = False

        # Sleep to stabilize
        sleep(1)

    def run(self):
        try:
            # Main loop
            while rclpy.ok():
                msgScan = self.wait_for_message('/scan', LaserScan)
                odomMsg = self.wait_for_message('/odom', Odometry)

                # Secure the minimum time interval between 2 actions
                step_time = (self.get_clock().now() - self.t_step).nanoseconds / 1e9

                if step_time > MIN_TIME_BETWEEN_ACTIONS:
                    self.t_step = self.get_clock().now()

                    if not self.robot_in_pos:
                        self.robotStop()
                        # Initialize position
                        if REAL_ROBOT:
                            (x_init, y_init, theta_init) = (0, 0, 0)
                            odomMsg = self.wait_for_message('/odom', Odometry)
                            (x, y) = self.control.getPosition(odomMsg)
                            theta = degrees(self.control.getRotation(odomMsg))
                            self.robot_in_pos = True
                            self.get_logger().info(f'Initial position:\n x = {x:.2f} [m]\n y = {y:.2f} [m]\n theta = {theta:.2f} [degrees]')
                        else:
                            if RANDOM_INIT_POS:
                                (x_init, y_init, theta_init) = self.control.robotSetRandomPos(self.setPosPub)
                            else:
                                (x_init, y_init, theta_init) = self.control.robotSetPos(self.setPosPub, X_INIT, Y_INIT, THETA_INIT)

                            odomMsg = self.wait_for_message('/odom', Odometry)
                            (x, y) = self.control.getPosition(odomMsg)
                            theta = degrees(self.control.getRotation(odomMsg))

                            if abs(x - x_init) < 0.05 and abs(y - y_init) < 0.05 and abs(theta - theta_init) < 2:
                                self.robot_in_pos = True
                                self.get_logger().info(f'Initial position:\n x = {x:.2f} [m]\n y = {y:.2f} [m]\n theta = {theta:.2f} [degrees]')
                                sleep(1)
                            else:
                                self.robot_in_pos = False
                    else:
                        self.count += 1
                        text = f'\r\nStep {self.count} , Step time {step_time:.2f} s'

                        # Get robot position and orientation
                        (x, y) = self.control.getPosition(odomMsg)
                        theta = self.control.getRotation(odomMsg)

                        # Get lidar scan
                        (lidar, angles) = self.lidar.lidarScan(msgScan)
                        (state_ind, x1, x2, x3, x4) = self.lidar.scanDiscretization(self.state_space, lidar)

                        # Check for objects nearby
                        crash = self.lidar.checkCrash(lidar)
                        object_nearby = self.lidar.checkObjectNearby(lidar)
                        goal_near = self.lidar.checkGoalNear(x, y, X_GOAL, Y_GOAL)
                        enable_feedback_control = True

                        # Stop the simulation
                        if crash:
                            self.robotStop()
                            self.get_logger().info('End of testing! ==> Crash!')
                            rclpy.shutdown()

                        # Feedback control algorithm
                        elif enable_feedback_control and (not object_nearby or goal_near):
                            status = self.control.robotFeedbackControl(self.velPub, x, y, theta, X_GOAL, Y_GOAL, radians(THETA_GOAL))
                            text += ' ==> Feedback control algorithm '
                            if goal_near:
                                text += '(goal near)'

                        # Q-learning algorithm
                        else:
                            (action, status) = self.qlearning.getBestAction(self.Q_table, state_ind, self.actions)
                            if status != 'getBestAction => OK':
                                self.get_logger().warn(status)

                            status = self.control.robotDoAction(self.velPub, action)
                            if status != 'robotDoAction => OK':
                                self.get_logger().warn(status)
                            text += ' ==> Q-learning algorithm'

                        text += f'\r\nx: {x:.2f} -> {X_GOAL:.2f} [m]'
                        text += f'\r\ny: {y:.2f} -> {Y_GOAL:.2f} [m]'
                        text += f'\r\ntheta: {degrees(theta):.2f} -> {THETA_GOAL:.2f} [degrees]'

                        if status == 'Goal position reached!':
                            self.robotStop()
                            self.get_logger().info('Goal position reached! End of simulation!')
                            rclpy.shutdown()

                        self.get_logger().info(text)

        except rclpy.shutdown:
            self.robotStop()
            self.get_logger().info('Simulation terminated!')

    def robotStop(self):
        stop_msg = Twist()
        self.velPub.publish(stop_msg)

    def wait_for_message(self, topic, msg_type):
        msg = self.create_subscription(msg_type, topic, lambda msg: msg, 10)
        rclpy.spin_once(self)
        return msg

def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode()
    control_node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()