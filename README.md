# Volumetric Grasping Network

VGN is a 3D convolutional neural network for real-time 6 DOF grasp detection. The network accepts a Truncated Signed Distance Function (TSDF) representation of the scene and outputs a volume of the same spatial resolution, where each cell contains the predicted quality, orientation, and width of a grasp executed at the center of the voxel. The network is trained on synthetic grasping dataset generated with a physics simulator.

<!-- TODO insert citation -->

![](docs/overview.png)

The next sections provide instructions for getting started with VGN.

- [Installation](#installation)
- [Dataset Generation](#data-generation)
- [Network Training](#network-training)
- [Simulated Grasping](#simulated-grasping)
- [Robot Grasping](#robot-grasping)

## Installation

The following instructions were tested with `python2.7` and ROS Melodic.

(**Note**: A ROS installation is only required for visualizations and interfacing the robot and sensors. Simulations and network training should work just fine without.)

OpenMPI is used to distribute the data generation over multiple cores/machines.

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
virtualenv -p python2 --system-site-packages .venv
source .venv/bin/activate
```

Install the Python dependencies within the activated virtual environment.

```
pip install -r requirements.txt
```

Build and source the catkin workspace.

```
catkin build vgn
source /path/to/catkin_ws/devel/setup.zsh
```

<!-- TODO data download -->

## Data Generation

Generate raw synthetic grasping trials using the [pybullet](https://github.com/bulletphysics/bullet3) physics simulator.

First, a scene with randomly placed objects is generated. Next, multiple depth images are rendered to reconstruct a point cloud of the scene. A geometric heuristic then samples grasp candidates which are evaluated using dynamic simulation. These steps are repeated until the desired number of grasps has been generated.

```
python scripts/generate_data.py data/raw/foo --scene pile --object-set blocks [--num-grasps=...] [--sim-gui]
```

* The options for `scene` are `pile` and `packed`.
* See `data/urdfs` for available object sets.
* Use the `--sim-gui` option to show the simulation.
* `mpirun -np <num-workers> python ...` will run multiple simulations in parallel.

The script will create the following file structure within `path/to/foo`:

* `grasps.csv` contains the configuration, label, and associated scene for each grasp,
* `scenes/<scene_id>.npz` contains the synthetic sensor data of each scene.

The `data.ipynb` notebook is useful to clean, balance and visualize the generated data.

Finally, use `construct_dataset.py` to generate the voxel grids/grasp targets required to train VGN.

```
python scripts/construct_dataset.py data/raw/foo data/datasets/foo
```

* Samples of the dataset can be visualized with the `vis_sample.py` script. The script includes the option to apply a random affine transform to the input/target pair to check the data augmentation procedure.

## Network Training

```
python scripts/train_vgn.py --dataset data/datasets/foo [--augment]
```

Training and validation metrics are logged to TensorBoard and can be accessed with

```
tensorboard --logdir data/runs
```

## Simulated Grasping

The following command runs a simulated clutter removal experiment to evaluate VGN.

```
python scripts/sim_grasp.py --model data/models/vgn_conv.pth
```

* Use [`clutter_removal.ipynb`](notebooks/clutter_removal.ipynb) to compute metrics and visualize failure cases of an experiment.
* To detect grasps using GPD, you first need to install and launch the [`gpd_ros`](https://github.com/atenpas/gpd_ros) node, `mon launch vgn gpd.launch`, then run `python scripts/sim_grasp.py --model gpd`.

## Robot Grasping

This package contains an example of open-loop grasp execution on a Franka Emika Panda with a wrist-mounted Intel Realsense D435 depth sensor.

Frist, launch the robot and sensor drivers

```
mon launch vgn panda_grasp.launch
```

Then in a second terminal, run

```
ptyhon scripts/panda_grasp.py --model data/models/vgn_conv.pth.
```
