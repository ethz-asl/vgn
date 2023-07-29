
# Volumetric Grasping Network

VGN是一种三维卷积神经网络，专门用于实时检测6自由度的抓取姿态。这种网络采用的是场景的截断符号距离函数（TSDF）作为输入，然后输出一个与输入空间分辨率相同的体积。在这个输出体积中，每个单元格都包含对应于体素中心执行的抓取的预测质量、方向和宽度。VGN网络的训练基于一个由物理模拟生成的合成抓取数据集。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34306602/1690443124599-85925743-3fb0-4c42-be15-4b2e30f2ff16.png#averageHue=%23b2ab9f&clientId=u5c5199ba-ad60-4&from=paste&height=518&id=ub72a3d62&originHeight=648&originWidth=2100&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=960477&status=done&style=none&taskId=uc43415f4-4374-4a9f-83fb-d30511a2c38&title=&width=1680)`<br/>`This repository contains the implementation of the following publication:

- M. Breyer, J. J. Chung, L. Ott, R. Siegwart, and J. Nieto. Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter. _Conference on Robot Learning (CoRL 2020)_, 2020. [[pdf](http://arxiv.org/abs/2101.01132)][[video](https://youtu.be/FXjvFDcV6E0)]

If you use this work in your research, please [cite](https://github.com/ethz-asl/vgn/tree/corl2020#citing) accordingly.

本仓库作用：`<br/>`VGN模拟实验`<br/>`VGN实机实验`<br/>`GPD模拟实验

实验设置进行100轮实验，堆积场景，每个场景5个物体，无高亮和透明物体`<br/>`物品全部被清理`<br/>`检测到无法进行抓取`<br/>`连续两次抓取失败

模型验证指：标抓取次数、成功率、全部清除率、平均规划时间`<br/>`![image.png](https://cdn.nlark.com/yuque/0/2023/png/34306602/1690441971160-31a3deb7-9151-4888-9d71-51a6c72d06cc.png#averageHue=%232b2b2b&clientId=uc65d52b9-9580-4&from=paste&height=96&id=u9dac0abf&originHeight=191&originWidth=776&originalType=binary&ratio=2&rotation=0&showTitle=false&size=41895&status=done&style=none&taskId=u11f286c5-2278-4cde-97c2-8e06e623315&title=&width=388)

<aname="D93vF">`</a>`

## 实验环境

Ubuntu 20.04  ROS noetic`<br/>`注：`<br/>`没有安装ros，也可进行模拟实验，请将notebook/data.py和clutter_removal.py中有关ros的注释掉`<br/>`如果仅使用预训练模型复现结果，可跳过Data Generation和Network Training

<aname="cMZlj">`</a>`

## Installation

[https://github.com/ethz-asl/vgn](https://github.com/ethz-asl/vgn)`<br/>`安装用于并行生成数据的软件包

```

sudo apt install libopenmpi-dev

```

创建ros工作空间

```

mkdir -p ~/catkin_ws/src

cd ~/catkin_ws/

catkin_make

```

克隆代码仓库作为ros功能包

```

cd src

git clone https://github.com/ZixuanWang1210/vgn.git

cd vgn

python3 -m venv --system-site-packages .venv

source .venv/bin/activate

```

配置python虚拟环境并安装功能包

```

pip install -r requirements.txt

pip install -e .

```

编译ros工作空间

```

cd ~/catkin_ws/

catkin_make

```

下载预训练模型和URDF [here](https://drive.google.com/file/d/1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ/view?usp=sharing)，将data文件夹放在vgn功能包的根目录![](https://cdn.nlark.com/yuque/0/2023/png/34306602/1690301599160-10d7d4a4-2740-4d50-bb66-aa36f319cc61.png#averageHue=%23c6c5c5&from=url&id=zJ8T3&originHeight=594&originWidth=1578&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)

<aname="oUb9b">`</a>`

## Data Generation

以下操作均基于  (.venv) wzx@wzx:~/catkin_ws/src/vgn$ 目录进行`<br/>`并行生成数据，[mpirun -np `<num-workers>` ]python scripts/generate_data.py data/raw/foo --scene pile --object-set blocks [--num-grasps=...] [--sim-gui]，例如：

```

mpirun -np 7 python scripts/generate_data.py data/raw/foo --scene pile --object-set blocks

```

新开一个终端，运行

```

roscore

```

修改vgn/notebooks/data.py中的第18行，将地址改为vgn/data/row/foo的绝对路径，例如：

```python

root = Path("/home/wzx/catkin_ws/src/vgn/data/raw/foo")

```

保存文件后在终端运行

```

python notebooks/data.py 

```

输出 “cleanup completed!”，完成清理`<br/>`生成数据集

```

python scripts/construct_dataset.py data/raw/foo data/datasets/foo

```

<aname="Vy8kC">`</a>`

## Network Training

使用默认参数训练模型，使用-h查看可自定义参数

```

python scripts/train_vgn.py --dataset data/datasets/foo

```

使用tensorborad监看

```python

tensorboard --logdir data/runs

```

<aname="XFL76">`</a>`

## Simulated Grasping

<aname="DfSGK">`</a>`

### VGN

将模型从data/runs/.../vgn_con_01.ph重命名并移动到data/models/vgn_con.pth`<br/>`可选参数[--sim-gui] [--rviz] 可视化`<br/>`注意：vgn_conv.pth是作者提供的预训练模型

```python

python scripts/sim_grasp.py --model data/models/vgn_conv.pth 

```

pybullet和rviz可视化

```python

python scripts/sim_grasp.py --model data/models/vgn_conv.pth --sim-gui --rviz

rviz config/sim.rviz

```

评估模型`<br/>`修改vgn/notebooks/clutter_removel.py中的第18行，将地址改为vgn/experiments/...的绝对路径，例如：

```python

logdir = Path("/home/wzx/catkin_ws/src/vgn/data/experiments/23-07-26-01-07-11")

```

显示评估结果，并使用rviz可视化抓取失败案例

```python

python notebooks/clutter_removel.py

rviz config/sim.rviz

```

<aname="VHw0j">`</a>`

### GPD

在非VGN的工作空间中安装GPD （[https://github.com/atenpas/gpd#install](https://github.com/atenpas/gpd#install)）

```python

git clone https://github.com/atenpas/gpd

```

修改CmakeList（[https://github.com/atenpas/gpd_ros/issues/12](https://github.com/atenpas/gpd_ros/issues/12)）

```python

#set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17")

```

build

```python

cd gpd

mkdir build && cd build

cmake ..

make -j

sudo make install

```

安装GPD_ros([https://github.com/atenpas/gpd_ros](https://github.com/atenpas/gpd_ros))`<br/>`在catkin_ws/src目录下

```python

cd ~/catkin_ws/src

git clone https://github.com/atenpas/gpd_ros

```

更正bug，catkin_ws/gpd_ros/src/gpd_ros/grasp_detection_node.cpp: Line:149

```cpp

// grasp_detection_node.cpp Line:149

void GraspDetectionNode::cloud_callback(const sensor_msgs::PointCloud2& msg)

{

if (!has_cloud_)

{

if(!cloud_camera_)

    delete cloud_camera_;

cloud_camera_ = NULL;

// ...

```

修改vgn功能包中gpd的配置文件 `src/vgn/config/gpd/ros_eigen_params.cfg`（[https://github.com/atenpas/gpd_ros/issues/12](https://github.com/atenpas/gpd_ros/issues/12)）

```python

# ros_eigen_params.cfg Line 32

# (OpenVINO) Path to directory that contains neural network parameters

weights_file = /home/wzx/Document/gpd/models/lenet/15channels/params/

# 修改这个为第一步安装GPD的参数文件地址

# 如果是中文系统，“Document”要替换为“文档”

```

在工作空间的根目录catkin_make构建，如果报错就在重复执行一下指令

```python

cd ~/catkin_ws

catkin_make

```

使用GPD进行模拟抓取

```

python scripts/sim_grasp.py --model gpd

```

<aname="PIcsS">`</a>`

## Robot Grasping

This package contains an example of open-loop grasp execution with a Franka Emika Panda and a wrist-mounted Intel Realsense D435. Since the robot drivers are not officially supported on ROS noetic yet, we used the following workaround:

- Launch the roscore and hardware drivers on a NUC with [libfranka](https://frankaemika.github.io/docs/installation_linux.html) installed.
- Run MoveIt and the VGN scripts on a second computer with a ROS noetic installation connected to the same roscore following these [instructions](http://wiki.ros.org/ROS/Tutorials/MultipleMachines). This requires the latest version of [panda_moveit_config](https://github.com/ros-planning/panda_moveit_config).

在机器人NUC上运行

```

roscore &

roslaunch vgn panda_grasp.launch

```

在本机运行

```

roslaunch panda_moveit_config move_group.launch

python scripts/panda_grasp.py --model data/models/vgn_conv.pth

```

如果运行时报错

```

RLException: [abc.launch] is neither a launch file in package [abc] nor is [abc] a launch file name


在/catkin_ws中执行:

source devel/setup.bash

```

<aname="XRpKt">`</a>`

## 附录A

**理论不会出现但可能出现的报错**`<br/>`**Q1：AttributeError: module ‘numpy‘ has no attribute ‘long‘**`<br/>`A1：在当前python环境中卸载numpy，并安装pip install numpy==1.23.0`<br/>`**Q2：RLException: [abc.launch] is neither a launch file in package [abc] nor is [abc] a launch file name**`<br/>`A2：在/catkin_ws中执行：source devel/setup.bash`<br/>`**Q3：运行python代码时，import 时提示缺少包**`<br/>`A3：在vgn目录下运行：source .venv/bin/activate`<br/>`**Q4：进行数据清洗时，运行clutter_removal.py，get_grid()这行报错**`<br/>`A4：去代码中注释这一行，这一行在清洗数据中没有起到作用`<br/>`**Q5：进行数据清洗时，运行clutter_removal.py，无法找到grasp.csv**`<br/>`A5：使用绝对路径替换，可使用pwd命令`<br/>`**Q6：进行数据清洗时，运行clutter_removal.py，没有看到finish提示**`<br/>`A6：关闭启动的可视化窗口`<br/>`**Q7：进行数据清洗时，运行clutter_removal.py，提示Could not contact ROS master at **[http://localhost:11311],retrying...`<br/>`A7：新建终端，运行roscore`<br/>`**Q8：进行数据清洗时，运行clutter_removal.py/data.py时，没有安装ros**`<br/>`A8：注释所有和ros有关的代码，ros仅用来数据可视化，没有对清洗产生影响`<br/>`**Q9：训练过程提示数组越界（一般在Epoch=1时出现）**`<br/>`A9：删除data/dateset文件夹，并且重新生成数据并进行数据清理！！！注意运行顺序！问题源头在get_grid()函数，可以打印相关日志检查

<aname="DdGOD">`</a>`

## 附录B

<aname="eP86T">`</a>`

### 配置vscode开发环境

设置python虚拟环境：ctrl+shift+P ->  "select interpreter" -> vgn/.venv/bin/python3 `<br/>`![](https://cdn.nlark.com/yuque/0/2023/png/34306602/1690309223950-38eb01ef-be43-404e-b74b-929079e2ad7d.png#averageHue=%235d5d5d&from=url&id=hnEBu&originHeight=524&originWidth=1494&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)`<br/>`设置python调试器：

```

{

    "version": "0.2.0",

    "configurations": [

        {

            "name": "Python: 当前文件",

            "type": "python",

            "request": "launch",

            "program": "${file}",

            "console": "integratedTerminal",

            "justMyCode": true,

            "args": ["arg1","arg2","arg3"]


        }

    ]

}

```
