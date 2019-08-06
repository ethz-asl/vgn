# Volumetric Grasping Network

## Prerequisites

The package was tested on Ubuntu 18.04 with ROS Melodic and Python 2.7.
While ROS is used for visualizations and real-robot experiments, the data generation and network training are ROS independent.

It is recommended to use virtual environments so that packages from different projects do not interfere with each other.

OpenMPI is used to distribute the data generation.

```console
sudo apt install python-dev python-pip libopenmpi-dev
sudo pip install virtualenv
```

## Installation

Clone the Git repository.
In case of a ROS installation, this should be done within the `src` folder of a catkin workspace.

```console
git clone https://github.com/ethz-asl/grasp_playground
```

Create and activate a new virtual environment.

```console
cd grasp_playground
virtualenv -p python2 --system-site-packages venv
source venv/bin/activate
```

Install the Python dependencies within the activated virtual environment.

```console
pip install -r vgn/requirements.txt
pip install -r vgn_ros/requirements.txt
```

Build the packages with `catkin`.

```console
catkin build vgn vgn_ros
```

Lastly, don't forget to source the catkin workspace.

## Data Generation

The data generation can be distributed over multiple processors using MPI.

```console
mpiexec -n <n-procs> python scripts/generate_data.py <path-to-dataset>
```
