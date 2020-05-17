# Volumetric Grasping Network

- [vgn](vgn/README.md): The core modules and scripts to train and evaluate VGN.
- [vgn_ros](vgn_ros/README.md): ROS interface to pre-trained VGN models.

## Quick Start

The following instructions were tested with ROS Melodic on Ubuntu 18.04 and Python 2.7.
OpenMPI is optionally used to distribute the data generation.

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
