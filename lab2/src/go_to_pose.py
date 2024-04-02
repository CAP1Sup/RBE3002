#!/usr/bin/env python3

import rospy
import math
import numpy as np
import argparse
from lab2.srv import GoToPoseStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_quaternion


# Helper function to extract the sign of a number
# Any value below 0 is considered negative
def sign(value):
    if value < 0:
        return -1
    else:
        return 1


# Helper function to wrap an angle to the range [-pi, pi]
def wrap(angle):
    while abs(angle) > math.pi:
        angle = angle - 2 * math.pi * sign(angle)
    return angle


class Lab2:

    def __init__(self, is_service=False):
        """
        Class constructor
        """
        # Initialize node, name it 'lab2'
        rospy.init_node("lab2")

        # Set the rate
        self.rate = rospy.Rate(10)

        # Tell ROS that this node publishes Twist messages on the '/cmd_vel' topic
        self.speed_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber("/odom", Odometry, self.update_odometry)

        # Decide whether to use the service or the subscriber
        if is_service:
            # Tell ROS that this node provides a service on the '/go_to_goal' topic
            # When a request is received, call self.go_to_service
            rospy.Service("/go_to_pose_stamped", GoToPoseStamped, self.go_to_service)
        else:
            # Tell ROS that this node subscribes to PoseStamped messages on the '/move_base_simple/goal' topic
            # When a message is received, call self.go_to
            rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.go_to)

        # Attributes to keep track of current position
        self.px = 0.0
        self.py = 0.0
        self.dir = 0.0

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

    def drive(self, distance: float, linear_speed: float):
        """
        Drives the robot in a straight line. If the speed is negative, the robot moves backwards. Negative distances will be ignored
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The forward linear speed.
        """
        # Note the starting position
        start_x = self.px
        start_y = self.py

        # Publish the movement to the '/cmd_vel' topic
        # Continuously check if we have covered the distance
        while (self.px - start_x) ** 2 + (self.py - start_y) ** 2 < distance**2:
            self.send_speed(linear_speed, 0.0)
            self.rate.sleep()

        # Stop the robot
        self.send_speed(0.0, 0.0)

    def rotate(self, angle: float, aspeed: float):
        """
        Rotates the robot around the body center by the given angle.
        :param angle         [float] [rad]   The distance to cover.
        :param aspeed [float] [rad/s] The angular speed.
        """
        # Wrap the angle to the range [-pi, pi]
        angle = wrap(angle)

        # Invert the move speed if needed
        aspeed = aspeed if angle > 0 else -aspeed

        # Publish the movement to the '/cmd_vel' topic
        # Continuously check if we have covered the angle
        start_dir = self.dir
        while abs(wrap(self.dir - start_dir) - angle) > 0.1:
            self.send_speed(0.0, aspeed)
            self.rate.sleep()

        # Stop the robot
        self.send_speed(0.0, 0.0)

    def go_to_service(self, req: GoToPoseStamped):
        self.go_to(req.goal)
        return Bool(data=True)

    def go_to(self, msg: PoseStamped):
        """
        Calls rotate(), drive(), and rotate() to attain a given pose.
        This method is a callback bound to a Subscriber.
        :param msg [PoseStamped] The target pose.
        """
        # Calculate the angle to the target point
        initial_angle = (
            math.atan2(msg.pose.position.y - self.py, msg.pose.position.x - self.px)
            - self.dir
        )

        # Wrap the angle to the range [-pi, pi]
        initial_angle = wrap(initial_angle)

        # Attempt to optimize the direction of the robot
        # Instead of turning 180 degrees, we can turn 180 - angle degrees and drive backwards
        drive_dir = 1
        if abs(initial_angle) > math.pi / 2:
            initial_angle += math.pi
            drive_dir = -1

        # Execute the robot movements to reach the target pose
        self.rotate(initial_angle, 0.5)
        self.smooth_drive(
            msg.pose.position.x,
            msg.pose.position.y,
            0.2 * drive_dir,
        )

        # Convert the quaternion to Euler angles
        quat_list = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        (roll, pitch, target_yaw) = euler_from_quaternion(quat_list)
        self.rotate(target_yaw - self.dir, 0.5)

    def update_odometry(self, msg: Odometry):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        quat_orig = msg.pose.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll, pitch, self.dir) = euler_from_quaternion(quat_list)

    def smooth_drive(self, goal_x: float, goal_y: float, linear_speed: float):
        """
        Drives the robot in a straight line by changing the actual speed smoothly.
        :param goal_x       [float] [m]   The target x-coordinate.
        :param goal_y       [float] [m]   The target y-coordinate.
        :param linear_speed [float] [m/s] The maximum forward linear speed.
        """
        # Note the starting position
        start_x = self.px
        start_y = self.py

        # Calculate the distance
        distance = math.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2)

        # Publish the movement to the '/cmd_vel' topic
        # Adjust the speed smoothly over time
        # Continuously check if we have covered the distance
        current_speed = 0.0
        dist_tol = 0.01  # m
        while (
            abs(
                distance
                - math.sqrt((self.px - start_x) ** 2 + (self.py - start_y) ** 2)
            )
            > dist_tol
        ):
            # Calculate the remaining distance
            remaining_dist = distance - math.sqrt(
                (self.px - start_x) ** 2 + (self.py - start_y) ** 2
            )

            # Forward project the robot's position
            # By making this point follow the line between the start and goal points,
            # the robot will follow the line like a trailer follows a truck
            # The larger the projection distance, the smoother the movement, but the slower the correction
            proj_dist = 0.1 * sign(linear_speed)  # m
            forward_x = self.px + proj_dist * math.cos(self.dir)
            forward_y = self.py + proj_dist * math.sin(self.dir)
            forward_pt = np.array([forward_x, forward_y])

            # Find the distance from the forward point to the line
            # This distance is the error that we want to correct
            start_pt = np.array([start_x, start_y])
            goal_pt = np.array([goal_x, goal_y])
            heading_error = np.cross(
                goal_pt - start_pt, forward_pt - start_pt
            ) / np.linalg.norm(goal_pt - start_pt)

            # Calculate the heading correction by scaling the error
            headingP = -2.5 * math.pi  # rad/s/m
            heading_correction = headingP * heading_error

            # Initial acceleration
            # Make sure that the robot hasn't made it halfway to the target
            # Otherwise we could continue accelerating and overshoot the target
            if abs(current_speed) < abs(linear_speed) and remaining_dist > distance / 2:
                # Increase the speed by a constant acceleration
                # Pull the sign from the linear speed
                accel = 0.1  # m/s^2
                current_speed += (
                    accel * self.rate.sleep_dur.to_sec() * sign(linear_speed)
                )
                self.send_speed(current_speed, heading_correction)
                self.rate.sleep()

            else:
                # Simple P controller to adjust the speed
                # Note that we have to integrate the sign of linear speed into the controller
                kP = 0.75  # 1/m
                desired_speed = kP * remaining_dist * sign(linear_speed)

                # Cap the desired speed to the maximum speed
                # Extract the sign of the desired speed to preserve the direction
                if abs(desired_speed) > abs(linear_speed):
                    desired_speed = abs(linear_speed) * sign(desired_speed)

                # Send over the speed
                # If the desired speed is higher than the maximum speed, send the maximum speed
                self.send_speed(desired_speed, heading_correction)
                self.rate.sleep()

        # Stop the robot
        self.send_speed(0.0, 0.0)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 2")
    parser.add_argument(
        "-s",
        "--service",
        action="store_true",
    )
    args, unknown = parser.parse_known_args()
    Lab2(args.service).run()
