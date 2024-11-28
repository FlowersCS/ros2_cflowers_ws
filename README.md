# Instalación
## Requisitos previos
Tener python3 y las siguientes dependencias:
- `ROS 2 Humble`
- `setuptools`
- `colcon`

## Instalación de paquetes
```bash
    git clone https://github.com/FlowersCS/ros2_cflowers_ws.git
    cd ~/ros2_cflowers_ws
    colcon build
```
Para mayor facilidad y no configurar tantas veces la terminal
```bash
    # Configuración de ROS2
    source /opt/ros/humble/setup.bash
    # Configuración de Gazebo
    source /usr/share/gazebo/setup.sh
    # Configuración del modelo de TurtleBot3
    export TURTLEBOT3_MODEL=burger
    source ~/.bashrc

    # esto si hacerlo manualmente o cada rebuildeas
    source ~/ros2_cflowers_ws/install/setup.bash
```

# Ejecución de nodos
```bash
# Configurable en src/simulations/setup.py
ros2 run simulations control_node

ros2 run simulations feedback_control_node

ros2 run simulations learning_node

ros2 run simulations scan_node

ros2 run simulations Qlearning
```