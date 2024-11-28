#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from math import *
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = 1.0
COLLISION_DISTANCE = 0.14 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45

ZONE_0_LENGTH = 0.4
ZONE_1_LENGTH = 0.7

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75

class Lidar(Node):
    def __init__(self):
        super().__init__('lidar_node')

        # Suscripción al tópico de LIDAR (usualmente /scan)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.subscription
        # Almacenar los datos del LIDAR para su procesamiento
        #self.lidar_data = np.array([])

    def lidar_callback(self, msgScan):
        # Procesar el mensaje LaserScan y convertirlo a un array
        #self.lidar_data, _ = self.lidarScan(msgScan)
        #self.get_logger().info(f"LIDAR data received: {self.lidar_data[:5]}...")  # Imprimir solo los primeros 5 valores para depuración
        lidar_data, angles = self.lidarScan(msgScan)
        self.get_logger().info('Lidar scan data received')
        
        if self.checkCrash(lidar_data):
            self.get_logger().info('Crash detected!')
        
        if self.checkObjectNearby(lidar_data):
            self.get_logger().info('Object nearby detected!')

    def lidarScan(self, msgScan):
        distances = np.array([])
        angles = np.array([])

        for i in range(len(msgScan.ranges)):
            angle = degrees(i * msgScan.angle_increment)
            if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
                distance = MAX_LIDAR_DISTANCE
            elif ( msgScan.ranges[i] < msgScan.range_min ):
                distance = msgScan.range_min
                # Para robots reales - protección
                if msgScan.ranges[i] < 0.01:
                    distance = MAX_LIDAR_DISTANCE
            else:
                distance = msgScan.ranges[i]

            distances = np.append(distances, distance)
            angles = np.append(angles, angle)

        # Distancias en [m], ángulos en [grados]
        return distances, angles

    def scanDiscretization(self, state_space, lidar):
        # Lógica de discretización del escaneo LIDAR
        x1 = 2 # Zona izquierda (sin obstáculos detectados)
        x2 = 2 # Zona derecha (sin obstáculos detectados)
        x3 = 3 # Sector izquierdo (sin obstáculos detectados)
        x4 = 3 # Sector derecho (sin obstáculos detectados)

        # Encontrar los valores del LIDAR a la izquierda del vehículo
        lidar_left = min(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH)])
        if ZONE_1_LENGTH > lidar_left > ZONE_0_LENGTH:
            x1 = 1 # Zona 1
        elif lidar_left <= ZONE_0_LENGTH:
            x1 = 0 # Zona 0

        # Encontrar los valores del LIDAR a la derecha del vehículo
        lidar_right = min(lidar[(ANGLE_MAX - HORIZON_WIDTH):(ANGLE_MAX)])
        if ZONE_1_LENGTH > lidar_right > ZONE_0_LENGTH:
            x2 = 1 # Zona 1
        elif lidar_right <= ZONE_0_LENGTH:
            x2 = 0 # Zona 0

        # Detección de objeto al frente del robot
        if ( min(lidar[(ANGLE_MAX - HORIZON_WIDTH // 3):(ANGLE_MAX)]) < 1.0 ) or ( min(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH // 3)]) < 1.0 ):
            object_front = True
        else:
            object_front = False

        # Detección de objeto a la izquierda del robot
        if min(lidar[(ANGLE_MIN):(ANGLE_MIN + 2 * HORIZON_WIDTH // 3)]) < 1.0:
            object_left = True
        else:
            object_left = False

        # Detección de objeto a la derecha del robot
        if min(lidar[(ANGLE_MAX - 2 * HORIZON_WIDTH // 3):(ANGLE_MAX)]) < 1.0:
            object_right = True
        else:
            object_right = False

        # Detección de objeto en el extremo izquierdo del robot
        if min(lidar[(ANGLE_MIN + HORIZON_WIDTH // 3):(ANGLE_MIN + HORIZON_WIDTH)]) < 1.0:
            object_far_left = True
        else:
            object_far_left = False

        # Detección de objeto en el extremo derecho del robot
        if min(lidar[(ANGLE_MAX - HORIZON_WIDTH):(ANGLE_MAX - HORIZON_WIDTH // 3)]) < 1.0:
            object_far_right = True
        else:
            object_far_right = False

        # El sector izquierdo del vehículo
        if ( object_front and object_left ) and ( not object_far_left ):
            x3 = 0 # Sector 0
        elif ( object_left and object_far_left ) and ( not object_front ):
            x3 = 1 # Sector 1
        elif object_front and object_left and object_far_left:
            x3 = 2 # Sector 2

        if ( object_front and object_right ) and ( not object_far_right ):
            x4 = 0 # Sector 0
        elif ( object_right and object_far_right ) and ( not object_front ):
            x4 = 1 # Sector 1
        elif object_front and object_right and object_far_right:
            x4 = 2 # Sector 2

        # Encontrar el índice del estado (x1, x2, x3, x4) en la tabla Q
        ss = np.where(np.all(state_space == np.array([x1,x2,x3,x4]), axis = 1))
        state_ind = int(ss[0])

        return state_ind, x1, x2, x3 , x4

    def checkCrash(self, lidar):
        lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
        W = np.linspace(1.2, 1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1, 1.2, len(lidar_horizon) // 2))
        if np.min( W * lidar_horizon ) < COLLISION_DISTANCE:
            return True
        else:
            return False

    def checkObjectNearby(self, lidar):
        lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
        W = np.linspace(1.4, 1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1, 1.4, len(lidar_horizon) // 2))
        if np.min( W * lidar_horizon ) < NEARBY_DISTANCE:
            return True
        else:
            return False

    def checkGoalNear(self, x, y, x_goal, y_goal):
        ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
        if ro < 0.3:
            return True
        else:
            return False


def main():
    rclpy.init()
    lidar_node = Lidar()

    rclpy.spin(lidar_node)
    
    lidar_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()