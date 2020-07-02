# Volumetric Grasping Network

- [Setup](#setup)
- [Data Generation](#data-generation)
- [Training](#training)
- [Simulated Grasping](#simulated-grasping)
- [Robot Grasping](#robot-grasping)

## Setup

The following instructions were tested with Python 2.7 and ROS Melodic on Ubuntu 18.04.

First, install some additonal system dependencies. OpenMPI is optionally used to distribute the data generation.

```
sudo apt install libopenmpi-dev
```

Clone the repository into the `src` folder of a catkin workspace.

```
git clone https://github.com/ethz-asl/vgn
```

Build and source the catkin workspace.

```
catkin build
source /path/to/catkin_ws/devel/setup.zsh
```

Create and activate a new virtual environment.

```
virtualenv -p python2 --system-site-packages .venv
source .venv/bin/activate
```

Install the Python dependencies within the activated virtual environment.

```
pip install -r requirements.txt
```

## Data Generation

## Training

## Simulated Grasping

## Robot Grasping

```
sudo apt install ros-melodic-apriltag-ros
```
