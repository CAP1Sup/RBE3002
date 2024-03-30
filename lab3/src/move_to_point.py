#!/usr/bin/env python3

import rospy
from lab2.srv import GoToPoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, Point
from path_planner import PathPlanner
import time


class MoveToPoint:

    def __init__(self):
        """
        Class constructor
        """
        # Initialize node, name it 'path_creator'
        rospy.init_node("path_creator")

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber("/odom", Odometry, self.update_odometry)

        # Tell ROS that this node subscribes to PoseStamped messages on the '/move_base_simple/goal' topic
        # When a message is received, call self.go_to
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.move_to_point)

        # Attributes to keep track of current position
        self.curr_pose = PoseStamped()

    def update_odometry(self, msg: Odometry):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        self.curr_pose = PoseStamped(pose=msg.pose.pose)

    def move_to_point(self, goal: PoseStamped):
        """
        Moves the robot to a given point.
        :param goal [PoseStamped] The goal.
        """
        # Print the current position
        rospy.loginfo(
            f"Current position: {self.curr_pose.pose.position.x}, {self.curr_pose.pose.position.y}"
        )

        # Print the goal position
        rospy.loginfo(f"Goal position: {goal.pose.position.x}, {goal.pose.position.y}")

        # Call the path planning algorithm
        rospy.wait_for_service("/plan_path")
        rospy.wait_for_service("/go_to_pose_stamped")
        try:
            path_planner = rospy.ServiceProxy("/plan_path", GetPlan)
            path = path_planner(
                self.curr_pose,
                goal,
                0.0,
            ).plan
            rospy.loginfo(f"Path: {path}")

            # Follow the path
            go_to_pose = rospy.ServiceProxy("/go_to_pose_stamped", GoToPoseStamped)
            for pose in path.poses:
                rospy.loginfo(
                    f"Moving to {pose.pose.position.x}, {pose.pose.position.y}"
                )
                # Move the robot to the next point
                go_to_pose(pose)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def run(self):
        start = (16, 16)
        end = (27, 15)
        map = PathPlanner.request_map()
        start_time = time.time()
        rospy.loginfo(f"Calculated path: {PathPlanner.a_star(map, start, end)}")
        rospy.loginfo(f"Time taken: {time.time() - start_time:.4f}s")
        rospy.spin()


if __name__ == "__main__":
    MoveToPoint().run()
