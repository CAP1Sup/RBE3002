#!/usr/bin/env python3

# Required if using Python 3.8 or below
from __future__ import annotations

import os

import subprocess

import rospy
import tf
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan


class MazeExplorer:
    def __init__(self):
        # Initialize node, name it 'maze_explorer'
        rospy.init_node("maze_explorer")

        # Rate limit the node
        self.rate = rospy.Rate(10)

        # Create a TF listener
        self.tf_listener = tf.TransformListener()

        # Variable to disable the POI callback
        self.disable_poi = False

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber("/odom", Odometry, self.update_odometry)

        # Subscribe to the frontier POI topic
        rospy.Subscriber("/frontier_finder/poi", Point, self.update_poi)

        # Subscribe to the 2D Nav Goal topic
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.end_exploration)

        # Publish the goal for the robot
        self.goal_pub = rospy.Publisher("/go_to_point/goal", Point, queue_size=10)

        # Attributes to keep track of current position
        self.curr_pose = PoseStamped()

        # Save the first odom message
        # This will be used to bring the robot back to the starting point
        self.starting_pose = None

        # Point of interest for the robot to explore
        self.poi = None
        self.poi_updated = False

    def end_exploration(self, msg: PoseStamped):
        """
        Ends the exploration.
        """
        # Deregister the update POI callback
        self.disable_poi = True

        # Publish a Point with None values to stop the exploration
        self.poi = Point(x=None)
        self.poi_updated = True

    def update_odometry(self, msg: Odometry):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        trans = [0, 0]
        rot = [0, 0, 0, 0]
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                "/map", "/base_footprint", rospy.Time(0)
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            print("Error running tf transform")

        # Create a PoseStamped message
        quat = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        pose = Pose(position=Point(x=trans[0], y=trans[1]), orientation=quat)
        self.curr_pose = PoseStamped(pose=pose)

        # Save the starting pose
        if not self.starting_pose:
            self.starting_pose = self.curr_pose

    def update_poi(self, msg: Point):
        """
        Updates the internal point of interest for the robot to explore.
        """
        if not self.disable_poi:
            self.poi = msg
            self.poi_updated = True

    def run(self):
        # Wait until the first point of interest is found
        while not self.poi:
            rospy.sleep(0.25)

        # Loop until there are no more points of interest
        # When there are no more points of interest, a final Point with None values will be published
        while self.poi.x:

            # Reset the point of interest flag
            self.poi_updated = False

            # Check if the robot is already at the point of interest
            # This would mean the map is fully explored
            if (
                abs(self.curr_pose.pose.position.x - self.poi.x) < 0.05
                and abs(self.curr_pose.pose.position.y - self.poi.y) < 0.05
            ):
                break

            # Call the path planning algorithm
            rospy.wait_for_service("/plan_path")
            try:
                path_planner = rospy.ServiceProxy("/plan_path", GetPlan)
                path = path_planner(
                    self.curr_pose,
                    PoseStamped(pose=Pose(position=self.poi)),
                    0.5,
                ).plan

                # Follow the path
                if len(path.poses) <= 1:
                    continue
                for pose in path.poses:
                    rospy.loginfo(
                        f"Moving to {pose.pose.position.x:.4f}m, {pose.pose.position.y:.4f}m"
                    )
                    # Tell the robot to move to the next point
                    self.goal_pub.publish(pose.pose.position)

                    # Loop until the robot reaches the goal
                    while True:
                        # Check if the point of interest has changed
                        if self.poi_updated:
                            break

                        # Check if the robot has reached the goal
                        if (
                            abs(self.curr_pose.pose.position.x - pose.pose.position.x)
                            < 0.025
                            and abs(
                                self.curr_pose.pose.position.y - pose.pose.position.y
                            )
                            < 0.025
                        ):
                            break

                        self.rate.sleep()

                    # Check if the point of interest has changed
                    if self.poi_updated:
                        break

            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                return False

        # Get the path to this file
        path = subprocess.run(
            "rospack find lab4",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Check if the maps directory exists
        # If it doesn't, create it
        if not os.path.exists(f"{path}/maps"):
            os.makedirs(f"{path}/maps")

        # Save the current map
        subprocess.run(
            f"rosrun map_server map_saver -f {path}/maps/map",
            shell=True,
            executable="/bin/bash",
        )

        # Print that the map has been saved and the robot is returning to the starting point
        rospy.loginfo("Map saved. Returning to starting point.")

        # Bring the robot back to the starting point
        rospy.wait_for_service("/plan_path")
        path_planner = rospy.ServiceProxy("/plan_path", GetPlan)
        path = path_planner(
            self.curr_pose,
            self.starting_pose,
            0.5,
        ).plan

        # Follow the path
        for pose in path.poses:
            rospy.loginfo(f"Moving to {pose.pose.position.x}, {pose.pose.position.y}")
            # Tell the robot to move to the next point
            self.goal_pub.publish(pose.pose.position)

            # Loop until the robot reaches the goal
            while True:
                # Check if the robot has reached the goal
                if (
                    abs(self.curr_pose.pose.position.x - pose.pose.position.x) < 0.05
                    and abs(self.curr_pose.pose.position.y - pose.pose.position.y)
                    < 0.05
                ):
                    break

                self.rate.sleep()

        # Shutdown the node
        rospy.signal_shutdown("Exploration complete")


if __name__ == "__main__":
    MazeExplorer().run()
