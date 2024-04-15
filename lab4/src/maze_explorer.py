#!/usr/bin/env python3

# Required if using Python 3.8 or below
from __future__ import annotations

import subprocess

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped
from lab2.srv import GoToPoseStamped
from lab3.path_planner import PathPlanner
from nav_msgs.msg import GridCells, OccupancyGrid, Odometry
from nav_msgs.srv import GetPlan
from std_msgs.msg import Float32


class MazeExplorer:
    def __init__(self):
        # Initialize node, name it 'maze_explorer'
        rospy.init_node("maze_explorer")

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber("/odom", Odometry, self.update_odometry)

        # Subscribe to the frontier POI topic
        rospy.Subscriber("/frontier_finder/poi", Point, self.update_poi)

        # Attributes to keep track of current position
        self.curr_pose = PoseStamped()

        # Point of interest for the robot to explore
        self.poi = None
        self.poi_updated = False

    def update_odometry(self, msg: Odometry):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        self.curr_pose = PoseStamped(pose=msg.pose.pose)

    def update_poi(self, msg: OccupancyGrid):
        """
        Updates the internal point of interest for the robot to explore.
        """
        self.poi = msg
        self.poi_updated = True

    def run(self):
        # Wait until the first point of interest is found
        while not self.poi:
            rospy.sleep(0.25)

        # Loop until there are no more points of interest
        while self.poi:
            # If the point of interest has been updated, print the new point
            if self.poi_updated:
                rospy.loginfo(f"New point of interest: {self.poi}")

            # Call the path planning algorithm
            rospy.wait_for_service("/plan_path")
            rospy.wait_for_service("/go_to_pose_stamped")
            try:
                path_planner = rospy.ServiceProxy("/plan_path", GetPlan)
                path = path_planner(
                    self.curr_pose,
                    PoseStamped(pose=Pose(position=self.poi)),
                    0.0,
                ).plan

                # rospy.loginfo(f"Path: {path}")

                # Follow the path
                go_to_pose = rospy.ServiceProxy("/go_to_pose_stamped", GoToPoseStamped)
                for pose in path.poses[1:]:
                    rospy.loginfo(
                        f"Moving to {pose.pose.position.x}, {pose.pose.position.y}"
                    )
                    # Move the robot to the next point
                    reached = go_to_pose(pose, Float32(data=0.2), Float32(data=0.5))

                    # If the robot didn't reach the goal, return False
                    if not reached:
                        return False

                    # Check if the point of interest has changed
                    if self.poi_updated:
                        break

            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                return False

        # Save the current map
        subprocess.run(
            "rosrun map_server map_saver -f maze", shell=True, executable="/bin/bash"
        )


if __name__ == "__main__":
    MazeExplorer().run()
