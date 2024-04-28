# RBE3002 D24 - Code and Documentation

Written by Christian Piper

This repo contains code for the RBE3002 course at WPI. The code is written in Python (**3.8**) and is designed to run on the ROS Noetic framework. The code is designed to run on the Turtlebot 3, but can be modified to run on other robots.

## DO NOT COPY THIS CODE - I AM NOT RESPONSIBLE FOR ANY ACADEMIC DISHONESTY THAT MAY OCCUR FROM COPYING THIS CODE

Academic dishonesty is a serious offense and can result in expulsion from WPI. It also hurts your learning and will lead you to struggle later. The only one who suffers from academic dishonesty is **YOU**. Please use this code as a reference only.

## Setup

Clone this folder into your catkin workspace. Then, run `catkin_make` in the root of your workspace. Finally, source the workspace with `source devel/setup.bash`.

You'll also need to install the following Python packages:

- numpy
- scipy
- opencv-python
- numba

You can install these packages with `pip3 install numpy scipy opencv-python numba`.

If you run into issues with lab3 not installing from `catkin_make`, you can install the package manually with `pip3 install .` from inside the lab3 folder.

If you run into issues, feel free to leave an issue on this repo

## Running the Code

All code can be run from a singular launch file within the package. Launch files with a `launch_sel` prefix allow the use of a "use_sim" argument to run the code in simulation.
For example, to run the lab2 code in sim, run `roslaunch lab2 launch_sel.launch use_sim:=true`.
Conversely, to run the code on the real robot, run `roslaunch lab2 launch_sel.launch`.

Best of luck with your labs!

P.S. We love you Kirby! Thanks for being an awesome Turtlebot!
