#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from tf_transformations import euler_from_quaternion, quaternion_from_euler 
import numpy as np
from math import *

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Q-learning speed parameters
CONST_LINEAR_SPEED_FORWARD = 0.08
CONST_ANGULAR_SPEED_FORWARD = 0.0
CONST_LINEAR_SPEED_TURN = 0.06
CONST_ANGULAR_SPEED_TURN = 0.4

# Feedback control parameters
K_RO = 2
K_ALPHA = 15
K_BETA = -3
V_CONST = 0.1 # [m/s]

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.1 # [m]
GOAL_ANGLE_THRESHOLD = 15 # [degrees]

class Control(Node):
    def __init__(self):
        super().__init__('control_node')
        
        # cmd_vel publisher
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # ModelState publisher
        self.setPosPub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        
        # Odometry suscriber
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
    def odom_callback(self, msg):
        # process odometry dates if necessaty
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Obtener la orientación del robot (quaternion) desde el mensaje de odometría
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        # Convertir el quaternion a ángulo de yaw (theta) en radianes
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        # Guardar los valores de la posición y la orientación para su uso posterior
        self.current_x = x
        self.current_y = y
        self.current_theta = yaw

        # Obtener las velocidades (lineales y angulares) del robot
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z

        # Guardar las velocidades para su uso posterior
        self.current_linear_velocity = linear_velocity
        self.current_angular_velocity = angular_velocity

        # Imprimir los datos de la odometría (opcional, solo para depuración)
        self.get_logger().info(f"Posición (x, y): ({x:.2f}, {y:.2f}), Orientation (theta): {yaw:.2f} rad")
        self.get_logger().info(f"Velocidades - Lineal: {linear_velocity:.2f} m/s, Angular: {angular_velocity:.2f} rad/s")

    def publish_goal_marker(self, x_goal, y_goal):
        # Crear un marcador de tipo esfera
        marker = Marker()
        marker.header.frame_id = "map"  # Usa "map" o el frame correspondiente a tu mundo
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE  # Usamos una esfera para el marcador
        marker.action = Marker.ADD
        marker.pose.position.x = x_goal
        marker.pose.position.y = y_goal
        marker.pose.position.z = 0.0  # Altura de la esfera
        marker.scale.x = 0.2  # Tamaño de la esfera
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0  # Opacidad
        marker.color.r = 1.0  # Color rojo
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Publicar el marcador
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Publishing marker at: ({x_goal}, {y_goal})")  # Agregar log para depuración

    # Función para obtener la rotación (theta) en [radianes] desde el mensaje de odometría
    def getRotation(self, odomMsg):
        orientation_q = odomMsg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        return yaw

    # Función para obtener las coordenadas (x, y) en [m] desde el mensaje de odometría
    def getPosition(self, odomMsg):
        x = odomMsg.pose.pose.position.x
        y = odomMsg.pose.pose.position.y
        return (x, y)

    # Función para obtener la velocidad lineal en [m/s] desde el mensaje de odometría
    def getLinVel(self, odomMsg):
        return odomMsg.twist.twist.linear.x

    # function for generate the velocity message
    def createVelMsg(self, v, w):
        v = float(v)
        w = float(w)
        
        velMsg = Twist()
        velMsg.linear.x = v
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0
        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        velMsg.angular.z = w
        return velMsg
    
    # control function
    def robotGoForward(self):
        velMsg = self.createVelMsg(CONST_LINEAR_SPEED_FORWARD,CONST_ANGULAR_SPEED_FORWARD)
        self.velPub.publish(velMsg)
        
    def robotTurnLeft(self):
        velMsg = self.createVelMsg(CONST_LINEAR_SPEED_TURN,+CONST_ANGULAR_SPEED_TURN)
        self.velPub.publish(velMsg)
    
    def robotTurnRight(self):
        velMsg = self.createVelMsg(CONST_LINEAR_SPEED_TURN, -CONST_ANGULAR_SPEED_TURN)
        self.velPub.publish(velMsg)
    
    def robotStop(self):
        velMsg = self.createVelMsg(0.0, 0.0)
        self.velPub.publish(velMsg)
    
    def robotSetPos(self, x, y, theta):
        checkpoint = ModelState()
        checkpoint.model_name = 'turtlebot3_burger'

        checkpoint.pose.position.x = x
        checkpoint.pose.position.y = y
        checkpoint.pose.position.z = 0.0

        [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, radians(theta))

        checkpoint.pose.orientation.x = x_q
        checkpoint.pose.orientation.y = y_q
        checkpoint.pose.orientation.z = z_q
        checkpoint.pose.orientation.w = w_q

        self.setPosPub.publish(checkpoint)
        return (x, y, theta)

    # Función para generar una posición aleatoria inicial para el robot
    def robotSetRandomPos(self, setPosPub):
        x_range = np.array([-0.4, 0.6, 0.6, -1.4, -1.4, 2.0, 2.0, -2.5, 1.0, -1.0])
        y_range = np.array([-0.4, 0.6, -1.4, 0.6, -1.4, 1.0, -1.0, 0.0, 2.0, 2.0])
        theta_range = np.arange(0, 360, 15)

        ind = np.random.randint(0, len(x_range))
        ind_theta = np.random.randint(0, len(theta_range))

        x = x_range[ind]
        y = y_range[ind]
        theta = theta_range[ind_theta]

        checkpoint = ModelState()
        checkpoint.model_name = 'turtlebot3_burger'

        checkpoint.pose.position.x = x
        checkpoint.pose.position.y = y
        checkpoint.pose.position.z = 0.0

        [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, radians(theta))

        checkpoint.pose.orientation.x = x_q
        checkpoint.pose.orientation.y = y_q
        checkpoint.pose.orientation.z = z_q
        checkpoint.pose.orientation.w = w_q

        self.setPosPub.publish(checkpoint)
        return (x, y, theta)

    # Función para ejecutar una acción
    def robotDoAction(self, velPub, action):
        status = 'robotDoAction => OK'
        if action == 0:
            self.robotGoForward()
        elif action == 1:
            self.robotTurnLeft()
        elif action == 2:
            self.robotTurnRight()
        else:
            status = 'robotDoAction => INVALID ACTION'
            self.robotGoForward()

        return status

    # Función para controlar el robot hacia una meta (feedback control)
    def robotFeedbackControl(self, velPub, x, y, theta, x_goal, y_goal, theta_goal):
        # Normalización del ángulo objetivo
        if theta_goal >= pi:
            theta_goal_norm = theta_goal - 2 * pi
        else:
            theta_goal_norm = theta_goal

        ro = sqrt(pow((x_goal - x), 2) + pow((y_goal - y), 2))
        lamda = atan2(y_goal - y, x_goal - x)

        alpha = (lamda - theta + pi) % (2 * pi) - pi
        beta = (theta_goal - lamda + pi) % (2 * pi) - pi

        if ro < GOAL_DIST_THRESHOLD and degrees(abs(theta - theta_goal_norm)) < GOAL_ANGLE_THRESHOLD:
            status = 'Goal position reached!'
            v = 0
            w = 0
            v_scal = 0
            w_scal = 0
        else:
            status = 'Goal position not reached!'
            v = K_RO * ro
            w = K_ALPHA * alpha + K_BETA * beta
            v_scal = v / abs(v) * V_CONST
            w_scal = w / abs(v) * V_CONST

        velMsg = self.createVelMsg(v_scal, w_scal)
        velPub.publish(velMsg)

        return status
    
    # Condición de estabilidad
    def check_stability(self, k_rho, k_alpha, k_beta):
        return k_rho > 0 and k_beta < 0 and k_alpha > k_rho

    # Condición de estabilidad fuerte
    def check_strong_stability(self, k_rho, k_alpha, k_beta):
        return k_rho > 0 and k_beta < 0 and k_alpha + 5 * k_beta / 3 - 2 * k_rho / np.pi > 0


def main():
    rclpy.init()
    control = Control()
    rclpy.spin(control)
    control.destroy_node()
    rclpy.shutdown()

#def main():
#    rclpy.init()
#    node = Control()
#
#    # Establecer una meta (x_goal, y_goal, theta_goal)
#    x_goal = 2.0  # Meta en x
#    y_goal = 2.0  # Meta en y
#    theta_goal = 0.0  # Meta en orientación (en radianes)
#    # Ejecutar un ciclo de control de retroalimentación
#    # Primero, obtener la odometría (esto debería estar siendo publicado en el callback)
#    while rclpy.ok():
#        node.publish_goal_marker(x_goal, y_goal)
#        # Este es el lugar donde el robot debería estar recibiendo odometría y aplicando el control
#        # Si el nodo está recibiendo la odometría, puedes obtener los valores de x, y, y theta del robot
#        # (en este caso los obtendrás en el callback de `odom_callback`)
#        
#        # Llamar al control de retroalimentación para mover el robot hacia la meta
#        # Asegúrate de que el robot esté en movimiento hacia la meta
#        node.robotFeedbackControl(node.velPub, 0.0, 0.0, 0.0, x_goal, y_goal, theta_goal)  # Aquí usamos 0.0, 0.0, 0.0 como valores de posición inicial temporales
#        
#        # Aquí puedes poner un `sleep` si no quieres que el loop sea constante, lo que ayudará a reducir la carga del procesador
#        rclpy.spin_once(node)
#
#    rclpy.shutdown()


if __name__ == '__main__':
    main()