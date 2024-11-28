#! /usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import matplotlib.pyplot as plt
import os

class DataPlotterNode(Node):
    def __init__(self):
        super().__init__('data_plotter')
        
        # Parámetros de directorios, estos podrían ser cargados desde el servidor de parámetros
        self.declare_parameter('log_gamma_1', '/path/to/log_gamma_1')
        self.declare_parameter('log_gamma_2', '/path/to/log_gamma_2')
        self.declare_parameter('log_gamma_3', '/path/to/log_gamma_3')
        self.declare_parameter('log_alpha_1', '/path/to/log_alpha_1')
        self.declare_parameter('log_alpha_2', '/path/to/log_alpha_2')
        self.declare_parameter('log_alpha_3', '/path/to/log_alpha_3')

        # Obtener los valores de los parámetros
        self.log_gamma_1 = self.get_parameter('log_gamma_1').get_parameter_value().string_value
        self.log_gamma_2 = self.get_parameter('log_gamma_2').get_parameter_value().string_value
        self.log_gamma_3 = self.get_parameter('log_gamma_3').get_parameter_value().string_value
        self.log_alpha_1 = self.get_parameter('log_alpha_1').get_parameter_value().string_value
        self.log_alpha_2 = self.get_parameter('log_alpha_2').get_parameter_value().string_value
        self.log_alpha_3 = self.get_parameter('log_alpha_3').get_parameter_value().string_value

        self.get_logger().info('DataPlotterNode initialized.')

        # Llamar a las funciones de graficado
        self.plot_gamma(self.log_gamma_1, self.log_gamma_2, self.log_gamma_3)
        self.plot_alpha(self.log_alpha_1, self.log_alpha_2, self.log_alpha_3)

    def plot_gamma(self, log_gamma_1, log_gamma_2, log_gamma_3):
        reward_per_episode_1 = np.genfromtxt(log_gamma_1 + '/reward_per_episode.csv', delimiter=',')
        reward_per_episode_2 = np.genfromtxt(log_gamma_2 + '/reward_per_episode.csv', delimiter=',')
        reward_per_episode_3 = np.genfromtxt(log_gamma_3 + '/reward_per_episode.csv', delimiter=',')
        
        steps_per_episode_1 = np.genfromtxt(log_gamma_1 + '/steps_per_episode.csv', delimiter=',')
        steps_per_episode_2 = np.genfromtxt(log_gamma_2 + '/steps_per_episode.csv', delimiter=',')
        steps_per_episode_3 = np.genfromtxt(log_gamma_3 + '/steps_per_episode.csv', delimiter=',')
        
        accumulated_reward_1 = np.array([])
        accumulated_reward_2 = np.array([])
        accumulated_reward_3 = np.array([])

        av_steps_per_10_episodes_1 = np.array([])
        av_steps_per_10_episodes_2 = np.array([])
        av_steps_per_10_episodes_3 = np.array([])

        episodes_10 = np.arange(10, len(reward_per_episode_1) + 10, 10)

        # Accumulated rewards and average steps
        for i in range(len(episodes_10)):
            accumulated_reward_1 = np.append(accumulated_reward_1, np.sum(reward_per_episode_1[0:10 * (i + 1)]))
            accumulated_reward_2 = np.append(accumulated_reward_2, np.sum(reward_per_episode_2[0:10 * (i + 1)]))
            accumulated_reward_3 = np.append(accumulated_reward_3, np.sum(reward_per_episode_3[0:10 * (i + 1)]))
            av_steps_per_10_episodes_1 = np.append(av_steps_per_10_episodes_1, np.sum(steps_per_episode_1[10 * i:10 * (i + 1)]) / 10)
            av_steps_per_10_episodes_2 = np.append(av_steps_per_10_episodes_2, np.sum(steps_per_episode_2[10 * i:10 * (i + 1)]) / 10)
            av_steps_per_10_episodes_3 = np.append(av_steps_per_10_episodes_3, np.sum(steps_per_episode_3[10 * i:10 * (i + 1)]) / 10)

        plt.style.use('seaborn-ticks')

        # Plot accumulated rewards
        plt.figure(1)
        plt.plot(episodes_10, accumulated_reward_1, label=r'$\gamma = 0.9$')
        plt.plot(episodes_10, accumulated_reward_2, label=r'$\gamma = 0.7$')
        plt.plot(episodes_10, accumulated_reward_3, label=r'$\gamma = 0.5$')
        plt.xlabel('Episode')
        plt.ylabel('Accumulated reward')
        plt.title('Accumulated reward per 10 episodes')
        plt.ylim(np.min(accumulated_reward_3) - 500, np.max(accumulated_reward_3) + 500)
        plt.xlim(np.min(episodes_10), np.max(episodes_10))
        plt.legend()
        plt.grid()

        # Plot average steps
        plt.figure(2)
        plt.plot(episodes_10, av_steps_per_10_episodes_1, label=r'$\gamma = 0.9$')
        plt.plot(episodes_10, av_steps_per_10_episodes_2, label=r'$\gamma = 0.7$')
        plt.plot(episodes_10, av_steps_per_10_episodes_3, label=r'$\gamma = 0.5$')
        plt.xlabel('Episode')
        plt.ylabel('Average steps')
        plt.title('Average steps per 10 episode')
        plt.ylim(np.min(av_steps_per_10_episodes_1) - 10, np.max(av_steps_per_10_episodes_1) + 10)
        plt.xlim(np.min(episodes_10), np.max(episodes_10))
        plt.legend(loc=4)
        plt.grid()

        plt.show()

    def plot_alpha(self, log_alpha_1, log_alpha_2, log_alpha_3):
        reward_per_episode_1 = np.genfromtxt(log_alpha_1 + '/reward_per_episode.csv', delimiter=',')
        reward_per_episode_2 = np.genfromtxt(log_alpha_2 + '/reward_per_episode.csv', delimiter=',')
        reward_per_episode_3 = np.genfromtxt(log_alpha_3 + '/reward_per_episode.csv', delimiter=',')
        
        steps_per_episode_1 = np.genfromtxt(log_alpha_1 + '/steps_per_episode.csv', delimiter=',')
        steps_per_episode_2 = np.genfromtxt(log_alpha_2 + '/steps_per_episode.csv', delimiter=',')
        steps_per_episode_3 = np.genfromtxt(log_alpha_3 + '/steps_per_episode.csv', delimiter=',')
        
        accumulated_reward_1 = np.array([])
        accumulated_reward_2 = np.array([])
        accumulated_reward_3 = np.array([])

        av_steps_per_10_episodes_1 = np.array([])
        av_steps_per_10_episodes_2 = np.array([])
        av_steps_per_10_episodes_3 = np.array([])

        episodes_10 = np.arange(10, len(reward_per_episode_1) + 10, 10)

        # Accumulated rewards and average steps
        for i in range(len(episodes_10)):
            accumulated_reward_1 = np.append(accumulated_reward_1, np.sum(reward_per_episode_1[0:10 * (i + 1)]))
            accumulated_reward_2 = np.append(accumulated_reward_2, np.sum(reward_per_episode_2[0:10 * (i + 1)]))
            accumulated_reward_3 = np.append(accumulated_reward_3, np.sum(reward_per_episode_3[0:10 * (i + 1)]))
            av_steps_per_10_episodes_1 = np.append(av_steps_per_10_episodes_1, np.sum(steps_per_episode_1[10 * i:10 * (i + 1)]) / 10)
            av_steps_per_10_episodes_2 = np.append(av_steps_per_10_episodes_2, np.sum(steps_per_episode_2[10 * i:10 * (i + 1)]) / 10)
            av_steps_per_10_episodes_3 = np.append(av_steps_per_10_episodes_3, np.sum(steps_per_episode_3[10 * i:10 * (i + 1)]) / 10)

        plt.style.use('seaborn-ticks')

        # Plot accumulated rewards
        plt.figure(3)
        plt.plot(episodes_10, accumulated_reward_1, label=r'$\alpha = 0.9$')
        plt.plot(episodes_10, accumulated_reward_2, label=r'$\alpha = 0.7$')
        plt.plot(episodes_10, accumulated_reward_3, label=r'$\alpha = 0.5$')
        plt.xlabel('Episode')
        plt.ylabel('Accumulated reward')
        plt.title('Accumulated reward per 10 episodes')
        plt.ylim(np.min(accumulated_reward_3) - 500, np.max(accumulated_reward_3) + 500)
        plt.xlim(np.min(episodes_10), np.max(episodes_10))
        plt.legend()
        plt.grid()

        # Plot average steps
        plt.figure(4)
        plt.plot(episodes_10, av_steps_per_10_episodes_1, label=r'$\alpha = 0.9$')
        plt.plot(episodes_10, av_steps_per_10_episodes_2, label=r'$\alpha = 0.7$')
        plt.plot(episodes_10, av_steps_per_10_episodes_3, label=r'$\alpha = 0.5$')
        plt.xlabel('Episode')
        plt.ylabel('Average steps')
        plt.title('Average steps per 10 episode')
        plt.ylim(np.min(av_steps_per_10_episodes_1) - 10, np.max(av_steps_per_10_episodes_1) + 10)
        plt.xlim(np.min(episodes_10), np.max(episodes_10))
        plt.legend(loc=4)
        plt.grid()

        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = DataPlotterNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()