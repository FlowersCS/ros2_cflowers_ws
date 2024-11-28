#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from simulations.Qlearning import QLearning
from simulations.Lidar import Lidar
from simulations.Control import Control


ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75

MIN_TIME_BETWEEN_SCANS = 0
MAX_SIMULATION_TIME = float('inf')

class ScanNode(Node):
    def __init__(self):
        super().__init__('scan_node')
        self.qlearning = QLearning()
        self.lidar = Lidar()
        self.control = Control()
        
        self.state_space = self.qlearning.createStateSpace()
        
        # Inicializar variables
        self.scan_time = 0
        self.count = 0
        self.t_0 = self.get_clock().now()
        self.t_start = self.get_clock().now()
        
        self.get_logger().info('SCAN NODE START ==> {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        # Init figure for real-time plotting
        plt.style.use('seaborn-ticks')
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('distance [m]')
        self.ax.set_ylabel('distance [m]')
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-0.2, 1.2)
        self.ax.set_title('Lidar horizon')
        self.ax.axis('equal')
        
        self.timer = self.create_timer(0.1, self.update)  # Timer to simulate rate at 10 Hz
    
    def lidar_callback(self, msgScan):
        self.scan_time = (self.get_clock().now() - self.t_0).nanoseconds / 1e9
        sim_time = (self.get_clock().now() - self.t_start).nanoseconds / 1e9
        
        self.count += 1
        
        if self.scan_time > MIN_TIME_BETWEEN_SCANS:
            self.get_logger().info(f'\nScan cycle: {self.count}')
            self.get_logger().info(f'Scan time: {self.scan_time:.2f} s')
            self.get_logger().info(f'Simulation time: {sim_time:.2f} s')
        
            self.t_0 = self.get_clock().now()
            
            lidar, angles = self.lidar.lidarScan(msgScan)
            state_ind, x1, x2, x3, x4 = self.lidar.scanDiscretization(self.state_space, lidar)
        
            crash = self.lidar.checkCrash(lidar)
            object_nearby = self.lidar.checkObjectNearby(lidar)
            
            self.get_logger().info(f'State index: {state_ind}')
            self.get_logger().info(f'x1 x2 x3 x4: {x1} {x2} {x3} {x4}')

            if crash:
                self.get_logger().info('CRASH!')
            if object_nearby:
                self.get_logger().info('OBJECT NEARBY!')
        
            # Horizon lidar data
            lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1], lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
            angles_horizon = np.linspace(90 + HORIZON_WIDTH, 90 - HORIZON_WIDTH, 150)

            # Horizon in x-y plane
            x_horizon = np.array([])
            y_horizon = np.array([])

            for i in range(len(lidar_horizon)):
                x_horizon = np.append(x_horizon, lidar_horizon[i] * np.cos(np.radians(angles_horizon[i])))
                y_horizon = np.append(y_horizon, lidar_horizon[i] * np.sin(np.radians(angles_horizon[i])))

            self.ax.clear()
            self.ax.plot(x_horizon, y_horizon, 'b.', markersize=8, label='obstacles')
            self.ax.plot(0, 0, 'r*', markersize=20, label='robot')
            self.ax.legend(loc='lower right', shadow=True)
            self.ax.set_xlabel('distance [m]')
            self.ax.set_ylabel('distance [m]')
            self.ax.set_xlim((-1.0, 1.0))
            self.ax.set_ylim((-0.2, 1.2))
            self.ax.set_title('Lidar horizon')
            self.ax.axis('equal')
            plt.draw()
            plt.pause(0.0001)
            
        if sim_time > MAX_SIMULATION_TIME:
            self.get_logger().info(f'End of simulation at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            rclpy.shutdown()
        
    def update(self):
        """ Function to continuously update the state of the simulation """
        # Placeholder for any other updates if necessary.
        pass
    

def main(args=None):
    rclpy.init(args=args)
    scan_node = ScanNode()
    
    try:
        rclpy.spin(scan_node)
    except KeyboardInterrupt:
        pass
    finally:
        now = datetime.now()
        dt_string_stop = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'SCAN NODE STOP ==> {dt_string_stop}')
        scan_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()