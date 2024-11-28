#! /usr/bin/env python

#import rclpy
#from rclpy.node import Node
#import numpy as np
#from math import *
#from sensor_msgs.msg import LaserScan
#from itertools import product
from geometry_msgs.msg import Twist
import numpy as np
from math import *
from itertools import product
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# Parámetros del estado y la acción
STATE_SPACE_IND_MAX = 144 - 1
STATE_SPACE_IND_MIN = 1 - 1
ACTIONS_IND_MAX = 2
ACTIONS_IND_MIN = 0

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75
T_MIN = 0.001

class QLearning(Node):
    def __init__(self):
        super().__init__('qlearning_node')

        self.create_subscription(LaserScan, '/scan', self.lidar_callback, QoSProfile(depth=10))
        self.create_timer(1.0, self.timer_callback)
        ## Parámetros iniciales para Q-learning
        #self.actions = self.createActions()
        #self.state_space = self.createStateSpace()
        #self.Q_table = self.createQTable(len(self.state_space), len(self.actions))
#
        ## Suscripción a LIDAR para obtener datos
        #self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
#
        ## Variables para almacenar los datos del LIDAR y estado anterior
        #self.lidar_data = np.array([])
#
        ## Publicación de los comandos de movimiento del robot
        #self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)

    def lidar_callback(self, msgScan):
        # Procesar el mensaje LaserScan y convertirlo a un array
        lidar = np.array(msgScan.ranges)
        self.get_logger().info(str(lidar))
        #self.lidar_data, _ = self.lidarScan(msgScan)
        #self.get_logger().info(f"LIDAR data received: {self.lidar_data[:5]}...")  # Imprimir solo los primeros 5 valores para depuración

    def timer_callback(self):
        pass

    # Crear acciones
    def createActions(self):
        actions = np.array([0, 1, 2])
        return actions

    # Crear el espacio de estados
    def createStateSpace(self):
        x1 = set((0, 1, 2))
        x2 = set((0, 1, 2))
        x3 = set((0, 1, 2, 3))
        x4 = set((0, 1, 2, 3))
        state_space = set(product(x1, x2, x3, x4))
        return np.array(list(state_space))

    # Crear la tabla Q
    def createQTable(self, n_states, n_actions):
        Q_table = np.zeros((n_states, n_actions))
        return Q_table
    
    def readQTable(self, path):
        Q_table = np.genfromtxt(path, delimiter=' , ')
        return Q_table
    
    def saveQTable(path, Q_table):
        np.savetxt(path, Q_table, delimiter=' , ')

    # Seleccionar la mejor acción en un estado
    def getBestAction(self, Q_table, state_ind, actions):
        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            status = 'getBestAction => OK'
            a_ind = np.argmax(Q_table[state_ind, :])
            a = actions[a_ind]
        else:
            status = 'getBestAction => INVALID STATE INDEX'
            a = self.getRandomAction(actions)

        return (a, status)

    # Acción aleatoria
    def getRandomAction(self, actions):
        n_actions = len(actions)
        a_ind = np.random.randint(n_actions)
        return actions[a_ind]

    # Exploración epsilon-greedy
    def epsiloGreedyExploration(self, Q_table, state_ind, actions, epsilon):
        if np.random.uniform() > epsilon and STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            status = 'epsiloGreedyExploration => OK'
            (a, status_gba) = self.getBestAction(Q_table, state_ind, actions)
            if status_gba == 'getBestAction => INVALID STATE INDEX':
                status = 'epsiloGreedyExploration => INVALID STATE INDEX'
        else:
            status = 'epsiloGreedyExploration => OK'
            a = self.getRandomAction(actions)

        return (a, status)

    def softMaxSelection(self, Q_table, state_ind, actions, T):
        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            status = 'softMaxSelection => OK'
            n_actions = len(actions)
            P = np.zeros(n_actions)
            
            # Boltzman distribution
            P = np.exp(Q_table[state_ind, :] / T) / np.sum(np.exp(Q_table[state_ind, :] / T))

            if T < T_MIN or np.any(np.isnan(P)):
                (a, status_gba) = self.getBestAction(Q_table,state_ind, actions)
                if status_gba == 'getBestAction => INVALID STATE INDEX':
                    status = 'softMaxSelection => INVALID STATE INDEX'
            else:
                rnd = np.random.uniform()
                status = 'softMaxSelection => OK'
                if P[0] > rnd:
                    a = 0
                elif P[0] <= rnd and (P[0] + P[1]) > rnd:
                    a = 1
                elif (P[0] + P[1]) <= rnd:
                    a = 2
                else:
                    status = 'softMaxSelection => Boltzman distribution error => getBestAction '
                    status = status + '\r\nP = (%f , %f , %f) , rnd = %f' % (P[0], P[1], P[2], rnd)
                    status = status + '\r\nQ(%d,:) = ( %f, %f, %f) ' % (state_ind, Q_table[state_ind, 0], Q_table[state_ind, 1], Q_table[state_ind, 2])
                    (a, status_gba) = self.getBestAction(Q_table, state_ind, actions)
                    if status_gba == 'getBestAction => INVALID STATE INDEX':
                        status = 'softMaxSelection => INVALID STATE INDEX'
        else:
            status = 'softMaxSelection => INVALID STATE INDEX'
            a = self.getRandomAction(actions)
            
        return (a, status)
            
    # Función de recompensa
    def getReward(self, action, prev_action, lidar, prev_lidar, crash):
        if crash:
            terminal_state = True
            reward = -100
        else:
            lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1], lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
            prev_lidar_horizon = np.concatenate((prev_lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1], prev_lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
            terminal_state = False
            if action == 0:
                r_action = +0.2
            else:
                r_action = -0.1
            W = np.linspace(0.9, 1.1, len(lidar_horizon) // 2)
            W = np.append(W, np.linspace(1.1, 0.9, len(lidar_horizon) // 2))
            if np.sum(W * (lidar_horizon - prev_lidar_horizon)) >= 0:
                r_obstacle = +0.2
            else:
                r_obstacle = -0.2
            if (prev_action == 1 and action == 2) or (prev_action == 2 and action == 1):
                r_change = -0.8
            else:
                r_change = 0.0

            reward = r_action + r_obstacle + r_change

        return (reward, terminal_state)

    # Actualizar la tabla Q
    def updateQTable(self, Q_table, state_ind, action, reward, next_state_ind, alpha, gamma):
        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX and STATE_SPACE_IND_MIN <= next_state_ind <= STATE_SPACE_IND_MAX:
            status = 'updateQTable => OK'
            Q_table[state_ind, action] = (1 - alpha) * Q_table[state_ind, action] + alpha * (reward + gamma * max(Q_table[next_state_ind, :]))
        else:
            status = 'updateQTable => INVALID STATE INDEX'
        return (Q_table, status)


def main():
    rclpy.init()
    qlearning_node = QLearning()

    rclpy.spin(qlearning_node)
    
    qlearning_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
