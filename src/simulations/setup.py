from setuptools import find_packages, setup

package_name = 'simulations'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='flowers',
    maintainer_email='carlos.flores.p@utec.edu.pe',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar = simulations.Lidar:main',
            'control = simulations.Control:main',
            'qlearning = simulations.Qlearning:main',
            'scan_node = simulations.scan_node:main',
            'learning_node = simulations.learning_node:main',
            'feedback_control_node = simulations.feedback_control_node:main',
            'control_node = simulations.control_node:main',
        ],
    },
)
