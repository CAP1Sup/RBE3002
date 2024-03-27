#!/usr/bin/env python3

import rospy
import math
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

    def __init__(self):
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
            math.sqrt(
                (msg.pose.position.y - self.py) ** 2
                + (msg.pose.position.x - self.px) ** 2
            ),
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

    def smooth_drive(self, distance: float, linear_speed: float):
        """
        Drives the robot in a straight line by changing the actual speed smoothly.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The maximum forward linear speed.
        """
        # Note the starting position
        start_x = self.px
        start_y = self.py

        # Ignore the sign of the distance
        distance = abs(distance)

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
                self.send_speed(current_speed, 0.0)
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
                self.send_speed(desired_speed, 0.0)
                self.rate.sleep()

        # Stop the robot
        self.send_speed(0.0, 0.0)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    Lab2().run()
