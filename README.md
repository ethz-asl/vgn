# Volumetric Grasping Network

VGN is a 3D convolutional neural network for real-time 6 DOF grasp pose detection. The network accepts a Truncated Signed Distance Function (TSDF) representation of the scene and outputs a volume of the same spatial resolution, where each cell contains the predicted quality, orientation, and width of a grasp executed at the center of the voxel. The network is trained on a synthetic grasping dataset generated with physics simulation.

![overview](docs/overview.png)

This repository contains the implementation of the following publication:

* M. Breyer, J. J. Chung, L. Ott, R. Siegwart, and J. Nieto. Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter. _Conference on Robot Learning (CoRL 2020)_, 2020. [[pdf](http://arxiv.org/abs/2101.01132)][[video](https://youtu.be/FXjvFDcV6E0)]

If you use this work in your research, please [cite](#citing) accordingly.

The next sections provide instructions for getting started with VGN.

* [Installation](#installation)
* [Dataset Generation](#data-generation)
* [Network Training](#network-training)
* [Simulated Grasping](#simulated-grasping)
* [Robot Grasping](#robot-grasping)

## Installation

The following instructions were tested with `python3.8` on Ubuntu 20.04. A ROS installation is only required for visualizations and interfacing hardware. Simulations and network training should work just fine without. The [Robot Grasping](#robot-grasping) section describes the setup for robotic experiments in more details.

OpenMPI is optionally used to distribute the data generation over multiple cores/machines.

```
sudo apt install libopenmpi-dev
```

Clone the repository into the `src` folder of a catkin workspace.

```
git clone https://github.com/ethz-asl/vgn
```

Create and activate a new virtual environment.

```
cd /path/to/vgn
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install the Python dependencies within the activated virtual environment.

```
pip install -r requirements.txt
```

Build and source the catkin workspace,

```
catkin build vgn
source /path/to/catkin_ws/devel/setup.zsh
```

or alternatively install the project locally in "editable" mode using `pip`.

```
pip install -e .
```

Finally, download the data folder [here](https://drive.google.com/file/d/1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ/view?usp=sharing), then unzip and place it in the repo's root.

## Data Generation

Generate raw synthetic grasping trials using the [pybullet](https://github.com/bulletphysics/bullet3) physics simulator.

```
python scripts/generate_data.py data/raw/foo --scene pile --object-set blocks [--num-grasps=...] [--sim-gui]
```

* `python scripts/generate_data.py -h` prints a list with all the options.
* `mpirun -np <num-workers> python ...` will run multiple simulations in parallel.

The script will create the following file structure within `data/raw/foo`:

* `grasps.csv` contains the configuration, label, and associated scene for each grasp,
* `scenes/<scene_id>.npz` contains the synthetic sensor data of each scene.

Clean the generated grasp configurations using the `data.ipynb` notebook.

Finally, generate the voxel grids/grasp targets required to train VGN.

```
python scripts/construct_dataset.py data/raw/foo data/datasets/foo
```

* Samples of the dataset can be visualized with the `vis_sample.py` script and `vgn.rviz` configuration. The script includes the option to apply a random affine transform to the input/target pair to check the data augmentation procedure.

## Network Training

```
python scripts/train_vgn.py --dataset data/datasets/foo [--augment]
```

Training and validation metrics are logged to TensorBoard and can be accessed with

```
tensorboard --logdir data/runs
```

## Simulated Grasping

Run simulated clutter removal experiments.

```
python scripts/sim_grasp.py --model data/models/vgn_conv.pth [--sim-gui] [--rviz]
```

* `python scripts/sim_grasp.py -h` prints a complete list of optional arguments.
* To detect grasps using GPD, you first need to install and launch the [`gpd_ros`](https://github.com/atenpas/gpd_ros) node (`roslaunch vgn gpd.launch`).

Use the `clutter_removal.ipynb` notebook to compute metrics and visualize failure cases of an experiment.

## Robot Grasping

This package contains an example of open-loop grasp execution with a Franka Emika Panda and a wrist-mounted Intel Realsense D435. Since the robot drivers are not officially supported on ROS noetic yet, we used the following workaround:

- Launch the roscore and hardware drivers on a NUC with [`libfranka`](https://frankaemika.github.io/docs/installation_linux.html) installed.
- Run MoveIt and the VGN scripts on a second computer with a ROS noetic installation connected to the same roscore following these [instructions](http://wiki.ros.org/ROS/Tutorials/MultipleMachines). This requires the latest version of [`panda_moveit_config`](https://github.com/ros-planning/panda_moveit_config).

First, on the NUC, start a roscore and launch the robot and sensor drivers: 

```
roscore &
roslaunch vgn panda_grasp.launch
```

Then, on the 20.04 computer, run

```
roslaunch panda_moveit_config move_group.launch
python scripts/panda_grasp.py --model data/models/vgn_conv.pth
```

## Citing

```
@inproceedings{breyer2020volumetric,
 title={Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter},
 author={Breyer, Michel and Chung, Jen Jen and Ott, Lionel and Roland, Siegwart and Juan, Nieto},
 booktitle={Conference on Robot Learning},
 year={2020},
}
```
