#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class Lab2:

    def __init__(self):
        """
        Class constructor
        """
        # Initialize node, name it 'lab2'
        rospy.init_node('lab2')

        # Set the rate
        self.rate = rospy.Rate(10)

        # Tell ROS that this node publishes Twist messages on the '/cmd_vel' topic
        self.speed_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber('/odom', Odometry, self.update_odometry)

        # Tell ROS that this node subscribes to PoseStamped messages on the '/move_base_simple/goal' topic
        # When a message is received, call self.go_to
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.go_to)

        #attributes to keep track of current position
        self.px = 0.0
        self.py = 0.0
        self.dir = 0.0




    def send_speed(self, linear_speed: float, angular_speed: float):
        """
        Sends the speeds to the motors.
        :param linear_speed  [float] [m/s]   The forward linear speed.
        :param angular_speed [float] [rad/s] The angular speed for rotating around the body center.
        """
        ### Make a new Twist message
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed

        ### Publish the message
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
        while (self.px - start_x)**2 + (self.py - start_y)**2 < distance**2:
            self.send_speed(linear_speed, 0.0)
            self.rate.sleep()

        # Stop the robot
        self.send_speed(0.0, 0.0)



    def rotate(self, angle: float, aspeed: float):
        """
        Rotates the robot around the body center by the given angle.
        :param angle         [float] [rad]   The distance to cover.
        :param angular_speed [float] [rad/s] The angular speed.
        """
        # Wrap the angle to the range [-pi, pi]
        while abs(angle) > math.pi:
            angle = angle - 2 * math.pi * (angle / abs(angle))

        # Invert the move speed if needed
        aspeed = aspeed if angle > 0 else -aspeed

        rospy.loginfo("Rotating by %f radians at %f rad/s", angle, aspeed)

        # Publish the movement to the '/cmd_vel' topic
        # Continuously check if we have covered the angle
        start_dir = self.dir
        while abs((self.dir - start_dir) - angle) > 0.1:
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
        # Execute the robot movements to reach the target pose
        self.rotate(math.atan2(msg.pose.position.y - self.py, msg.pose.position.x - self.px) - self.dir, 0.5)
        self.drive(math.sqrt((msg.pose.position.y - self.py)**2 + (msg.pose.position.x - self.px)**2), 0.2)
        self.rotate(msg.pose.orientation.z - self.dir, 0.5)



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
        (roll , pitch , yaw) = euler_from_quaternion(quat_list)
        self.dir = yaw



    def smooth_drive(self, distance: float, linear_speed: float):
        """
        Drives the robot in a straight line by changing the actual speed smoothly.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The maximum forward linear speed.
        """
        ### EXTRA CREDIT
        # TODO
        pass # delete this when you implement your code



    def run(self):
        rospy.spin()

if __name__ == '__main__':
    Lab2().run()
