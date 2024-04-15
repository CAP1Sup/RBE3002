#!/usr/bin/env python3

import rospy
import math
import argparse
from lab2.srv import GoToPoseStamped
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped
from lab2.srv import GoToPoseStamped


class MoveToPoint:

    def __init__(self, is_service=False):
        """
        Class constructor
        """
        # Initialize node, name it 'path_creator'
        rospy.init_node("path_creator")

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber("/odom", Odometry, self.update_odometry)

        if is_service:
            # Create a service called '/move_to_point' with the 'GoToPoseStamped' service type
            rospy.Service("/move_to_point", GoToPoseStamped, self.move_to_point_service)
        else:
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

    def move_to_point_service(self, req: GoToPoseStamped):
        """
        Moves the robot to a given point.
        :param req [GoToPoseStamped] The request containing the goal and the speeds.
        :return [Bool] True if the robot reached the goal.
        """
        result = self.move_to_point(req.goal, req.linear_speed, req.angular_speed)
        return Bool(data=result)

    def move_to_point(
        self,
        goal: PoseStamped,
        linear_speed=Float32(data=0.2),
        angular_speed=Float32(data=0.5),
    ):
        """
        Moves the robot to a given point.
        :param goal [PoseStamped] The goal.
        :param linear_speed [Float32] The linear speed.
        :param angular_speed [Float32] The angular speed.
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

            # Print the length of the path
            length = 0
            for i in range(1, len(path.poses)):
                length += math.dist(
                    [path.poses[i].pose.position.x, path.poses[i].pose.position.y],
                    [
                        path.poses[i - 1].pose.position.x,
                        path.poses[i - 1].pose.position.y,
                    ],
                )
            rospy.loginfo(f"Path length: {length:.4f}m")

            # Follow the path
            go_to_pose = rospy.ServiceProxy("/go_to_pose_stamped", GoToPoseStamped)
            for pose in path.poses:
                rospy.loginfo(
                    f"Moving to {pose.pose.position.x}, {pose.pose.position.y}"
                )
                # Move the robot to the next point
                reached = go_to_pose(pose, linear_speed, angular_speed)

                # If the robot didn't reach the goal, return False
                if not reached:
                    return False

            # Path was followed successfully, return True
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move to Point")
    parser.add_argument(
        "-s",
        "--service",
        action="store_true",
    )
    args, unknown = parser.parse_known_args()
    MoveToPoint(args.service).run()
