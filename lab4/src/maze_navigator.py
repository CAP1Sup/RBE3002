#!/usr/bin/env python3

import math

import numpy as np

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseWithCovarianceStamped, Twist
from lab2.srv import GoToPoseStamped
from nav_msgs.srv import GetPlan
from std_msgs.msg import Float32
from std_srvs.srv import Empty


class MazeNavigator:

    def __init__(self):
        """
        Class constructor
        """
        # Initialize node, name it 'maze_navigator'
        rospy.init_node("maze_navigator")

        # Rate limit the node
        self.rate = rospy.Rate(10)

        # Attributes to keep track of current position
        self.curr_pose = None
        self.localized = False

        # Publish the goal for the robot
        self.goal_pub = rospy.Publisher("/go_to_point/goal", Point, queue_size=10)

        # Publish to the cmd_vel topic
        # Used to move the robot around while attempting to localize
        self.speed_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Tell ROS that this node subscribes to Odometry messages on the '/amcl_pose' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.update_pos_est)

        # Create a publisher for the current position
        self.curr_pose_pub = rospy.Publisher("/current_pose", Pose, queue_size=10)

        # Tell ROS that this node subscribes to PoseStamped messages on the '/move_base_simple/goal' topic
        # When a message is received, call self.go_to
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.move_to_point)

        # Ignore the first couple of messages
        self.skip_msgs = 5

        # Wait for AMCL to come up
        # The global localization service is provided by AMCL
        rospy.wait_for_service("/global_localization")

        # Call the global localization service
        try:
            global_localization = rospy.ServiceProxy("/global_localization", Empty)
            global_localization()
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def update_pos_est(self, msg: PoseWithCovarianceStamped):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        # Check if the first few messages should be ignored
        if self.skip_msgs > 0:
            self.skip_msgs -= 1
            return

        # Check if the robot has been localized
        if self.localized:
            self.curr_pose = PoseStamped(pose=msg.pose.pose)
            self.curr_pose_pub.publish(self.curr_pose.pose)
            return

        # The robot has not been localized yet, check the covariance
        # Set the localization threshold
        x_localized_threshold = 0.005
        y_localized_threshold = 0.01
        yaw_localized_threshold = 0.04

        # Name the covariance indices
        x = 0
        y = 7
        yaw = 35

        # Get the X, Y, and yaw covariances of the pose
        cov = msg.pose.covariance

        # Print the covariance of the pose
        # Helpful for debugging
        rospy.loginfo(
            f"AMCL Cov: X: {cov[x]:.4f}, Y: {cov[y]:.4f}, Yaw: {cov[yaw]:.4f}"
        )

        # Check if the X, Y, and yaw covariances are below the threshold
        if (
            abs(cov[x]) < x_localized_threshold
            and abs(cov[y]) < y_localized_threshold
            and abs(cov[yaw]) < yaw_localized_threshold
        ):
            # Set the current pose
            self.curr_pose = PoseStamped(pose=msg.pose.pose)
            # Set the robot as localized
            self.localized = True
            rospy.loginfo("Robot localized.")

    def send_speed(self, linear_speed: float, angular_speed: float):
        """
        Sends the speeds to the motors.
        :param linear_speed  [float] [m/s]   The forward linear speed.
        :param angular_speed [float] [rad/s] The angular speed for rotating around the body center.
        """
        # Make a new Twist message
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed

        # Publish the message
        self.speed_pub.publish(twist)

    def move_to_point(
        self,
        goal: PoseStamped,
        linear_speed=Float32(data=0.2),
        angular_speed=Float32(data=2),
    ):
        """
        Moves the robot to a given point.
        :param goal [PoseStamped] The goal.
        :param linear_speed [Float32] The linear speed.
        :param angular_speed [Float32] The angular speed.
        """
        # Check if the robot has been localized
        if not self.localized:
            rospy.logerr("Robot has not been localized yet. Waiting...")
            while not self.localized:
                rospy.sleep(0.1)

        # Print the current position
        rospy.loginfo(
            f"Current position: {self.curr_pose.pose.position.x}, {self.curr_pose.pose.position.y}"
        )

        # Print the goal position
        rospy.loginfo(f"Goal position: {goal.pose.position.x}, {goal.pose.position.y}")

        # Call the path planning algorithm
        rospy.wait_for_service("/plan_path")
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
                    f"Moving to {pose.pose.position.x:.4f}m, {pose.pose.position.y:.4f}m"
                )
                # Tell the robot to move to the next point
                self.goal_pub.publish(pose.pose.position)

                # Loop until the robot reaches the goal
                while True:
                    # Check if the robot has reached the goal
                    if (
                        abs(self.curr_pose.pose.position.x - pose.pose.position.x)
                        < 0.05
                        and abs(self.curr_pose.pose.position.y - pose.pose.position.y)
                        < 0.05
                    ):
                        break

                    self.rate.sleep()

            # Path was followed successfully, return True
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def run(self):
        # Attempt to localize the robot
        while not self.localized:
            self.send_speed(0.01, 0.5)
            rospy.loginfo("Waiting for robot to localize...")
            rospy.sleep(0.1)

        # The robot is localized, stop spinning
        rospy.loginfo("Robot localized. Ready to navigate.")
        self.send_speed(0.0, 0.0)

        # Spin to keep the node alive
        # Any path planning and navigation will be done through callbacks
        rospy.spin()


if __name__ == "__main__":
    MazeNavigator().run()
