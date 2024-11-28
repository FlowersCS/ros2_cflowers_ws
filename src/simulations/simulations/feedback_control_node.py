#! /usr/bin/env python

import rclpy
from rclpy.node import Node
from time import sleep
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelState

from simulations.Control import Control
from .Control import *

# Constants
X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0
X_GOAL = 3
Y_GOAL = 2
THETA_GOAL = 15

# Initialize trajectory arrays
X_traj = np.array([])
Y_traj = np.array([])
THETA_traj = np.array([])
X_goal = np.array([])
Y_goal = np.array([])
THETA_goal = np.array([])

# Log directory
DATA_PATH = '/home/flowers/ros2_cflowers_ws/src/simulations/data'
LOG_DIR = DATA_PATH + '/Log_feedback'

class FeedbackControlNode(Node):
    def __init__(self):
        super().__init__('feedback_control_node')

        self.control = Control()
        # Initialize publishers
        self.set_pos_pub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Log simulation parameters
        self.log_simulation_params()

        # Create a timer to run the control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Initialize odometry subscriber
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Set up initial state
        self.x = X_INIT
        self.y = Y_INIT
        self.theta = THETA_INIT
        self.status = 'Simulation running'

    def log_simulation_params(self):
        with open(LOG_DIR + '/LogSimParams.txt', 'w') as log_file:
            log_file.write(f"Simulation parameters: \n")
            log_file.write(f"k_rho = {K_RO:.3f} \n")
            log_file.write(f"k_alpha = {K_ALPHA:.3f} \n")
            log_file.write(f"k_beta = {K_BETA:.3f} \n")
            log_file.write(f"v_const = {V_CONST:.3f} \n")

        self.get_logger().info("Simulation parameters logged.")

        # Check stability
        stab_dict = {True: 'Satisfied!', False: 'Not Satisfied!'}
        stability = stab_dict[self.control.check_stability(K_RO, K_ALPHA, K_BETA)]
        self.get_logger().info(f"Stability Condition: {stability}")
        strong_stability = stab_dict[self.control.check_strong_stability(K_RO, K_ALPHA, K_BETA)]
        self.get_logger().info(f"Strong Stability Condition: {strong_stability}")

    def odom_callback(self, msg):
        self.x, self.y, self.theta = self.get_position(msg)
        self.update_trajectory(self.x, self.y, self.theta)

    def get_position(self, msg):
        # Get robot position and orientation from odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        return position.x, position.y, theta

    def update_trajectory(self, x, y, theta):
        # Update the trajectory arrays
        global X_traj, Y_traj, THETA_traj, X_goal, Y_goal, THETA_goal
        X_traj = np.append(X_traj, x)
        Y_traj = np.append(Y_traj, y)
        THETA_traj = np.append(THETA_traj, np.degrees(theta))
        X_goal = np.append(X_goal, X_GOAL)
        Y_goal = np.append(Y_goal, Y_GOAL)
        THETA_goal = np.append(THETA_goal, THETA_GOAL)

    def control_loop(self):
        # Main control loop
        if self.status == 'Simulation running':
            # Perform feedback control
            status = self.robot_feedback_control(self.x, self.y, self.theta, X_GOAL, Y_GOAL, np.radians(THETA_GOAL))
            
            # Print current status
            self.get_logger().info(f"x: {self.x:.2f} -> {X_GOAL:.2f} [m]")
            self.get_logger().info(f"y: {self.y:.2f} -> {Y_GOAL:.2f} [m]")
            self.get_logger().info(f"theta: {np.degrees(self.theta):.2f} -> {THETA_GOAL:.2f} [degrees]")

            if status == 'Goal position reached!':
                self.robot_stop()

                # Save the trajectory data
                np.savetxt(LOG_DIR + '/X_traj.csv', X_traj, delimiter=' , ')
                np.savetxt(LOG_DIR + '/Y_traj.csv', Y_traj, delimiter=' , ')
                np.savetxt(LOG_DIR + '/THETA_traj.csv', THETA_traj, delimiter=' , ')
                np.savetxt(LOG_DIR + '/X_goal.csv', X_goal, delimiter=' , ')
                np.savetxt(LOG_DIR + '/Y_goal.csv', Y_goal, delimiter=' , ')
                np.savetxt(LOG_DIR + '/THETA_goal.csv', THETA_goal, delimiter=' , ')

                self.status = 'Goal position reached!'
                self.get_logger().info('Goal position reached! End of simulation!')

                # Shutdown the node after reaching the goal
                rclpy.shutdown()

    def robot_feedback_control(self, x, y, theta, x_goal, y_goal, theta_goal):
        # Call your control function here (implement `robotFeedbackControl` in ROS 2)
        twist_msg = Twist()

        # Implement feedback control logic
        # For example, simple proportional control:
        distance_error = np.sqrt((x_goal - x)**2 + (y_goal - y)**2)
        angle_error = np.arctan2(y_goal - y, x_goal - x) - theta

        # Proportional control for velocity
        linear_velocity = K_RO * distance_error
        angular_velocity = K_ALPHA * angle_error

        # Ensure that the robot doesn't exceed a maximum velocity
        twist_msg.linear.x = min(linear_velocity, V_CONST)
        twist_msg.angular.z = angular_velocity

        # Publish the velocity command
        self.vel_pub.publish(twist_msg)

        # Check if the goal is reached
        if distance_error < 0.1 and abs(angle_error) < np.radians(5):
            return 'Goal position reached!'
        return 'Moving'

    def robot_stop(self):
        # Stop the robot by sending zero velocities
        twist_msg = Twist()
        self.vel_pub.publish(twist_msg)
        self.get_logger().info("Robot stopped.")

def main(args=None):
    rclpy.init(args=args)

    feedback_control_node = FeedbackControlNode()

    rclpy.spin(feedback_control_node)

    feedback_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()