#!/usr/bin/env python3

# Required if using Python 3.8 or below
from __future__ import annotations

import time

import cv2

import numpy as np

import rospy
from geometry_msgs.msg import Point, PoseStamped
from lab3.path_planner import PathPlanner
from lab3.srv import GetPathLen
from nav_msgs.msg import GridCells, OccupancyGrid, Odometry
from numba import njit


class FrontierFinder:
    def __init__(self):
        # Initialize node, name it 'frontier_finder'
        rospy.init_node("frontier_finder")

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber("/odom", Odometry, self.update_odometry)

        # Subscribe to the map topic
        rospy.Subscriber("/map", OccupancyGrid, self.update_poi)

        # Create a point of interest publisher
        self.poi_pub = rospy.Publisher("/frontier_finder/poi", Point, queue_size=10)

        # Create a point of interest visualization publisher
        self.poi_vis_pub = rospy.Publisher(
            "/frontier_finder/vis_poi", GridCells, queue_size=10
        )

        # Create a frontier visualization publisher
        self.frontier_vis_pub = rospy.Publisher(
            "/frontier_finder/vis_frontier", GridCells, queue_size=10
        )
        # Attributes to keep track of current position
        self.curr_pose = PoseStamped()

        # Point of interest for the robot to explore
        self.poi = None
        self.poi_updated = False

        # Point of interest exclusion set
        self.poi_exclusions = set()

        # Skip the first few messages
        # This is necessary because the map is not well defined at the start
        self.skip = 1

    def update_odometry(self, msg: Odometry):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        self.curr_pose = PoseStamped(pose=msg.pose.pose)

    def update_poi(self, msg: OccupancyGrid):
        """
        Updates the point of interest for the robot to explore.
        Runs when the map is updated.
        """
        # Skip the first few messages
        if self.skip > 0:
            self.skip -= 1
            return

        # Get the current time
        start_time = time.time()

        # Get the centroids of the frontier groups
        centroids = self.get_frontier_centroids(msg)

        # If there are no frontier groups, set the point of interest to None
        if not centroids:

            # There are no frontiers, so the robot should stop searching
            self.poi_pub.publish(None, None, None)

            # Kill the node
            rospy.signal_shutdown("No frontiers remaining, shutting down node")
            return

        # Attempt to filter out the excluded points of interest
        filtered_centroids = []
        for centroid in centroids:
            if centroid not in self.poi_exclusions:
                filtered_centroids.append(centroid)

        # If the filtered list is empty, reset the exclusions
        if not filtered_centroids:
            self.poi_exclusions.clear()
            filtered_centroids = centroids

        # Calculate the current coordinates of the robot
        curr_coords = PathPlanner.world_to_grid(msg, self.curr_pose.pose.position)

        # Create a service proxy to get the path length
        rospy.wait_for_service("/get_path_len")
        path_len_srv = rospy.ServiceProxy("/get_path_len", GetPathLen)

        # Sort the list by the path len to the robot
        closest_centroid = None
        closest_dist = float("inf")
        for centroid in centroids:
            dist = path_len_srv(
                PathPlanner.grid_to_pose_stamped(msg, curr_coords),
                PathPlanner.grid_to_pose_stamped(msg, centroid),
                0.25,
            ).length.data

            # If the distance is 0, there was an error
            # Exclude the point and continue
            if dist == 0:
                self.poi_exclusions.add(centroid)

            # Otherwise check if the distance is the closest
            elif dist < closest_dist:
                closest_dist = dist
                closest_centroid = centroid

        # If the closest centroid is None, there are no valid points remaining
        if not closest_centroid:
            if self.poi_exclusions:
                self.poi_exclusions.clear()
                rospy.logwarn("Resetting point of interest exclusions")
                self.update_poi(msg)
                return
            else:
                self.poi_pub.publish(None, None, None)
                rospy.signal_shutdown(
                    "No valid points of interest remaining, shutting down node"
                )
                return

        # Calculate the world coordinates of the centroid
        centroid_world_point = PathPlanner.grid_to_world(msg, closest_centroid)

        # Check if the point of interest has changed
        if self.poi:
            if (
                abs(self.poi.x - centroid_world_point.x) < 0.025
                and abs(self.poi.y - centroid_world_point.y) < 0.025
            ):
                self.poi_updated = False
                rospy.loginfo("Point of interest has not changed")
                return

        # The point of interest is different, update it
        self.poi_pub.publish(centroid_world_point)

        # Print the new point of interest
        rospy.loginfo(
            f"New point of interest: {centroid_world_point.x:.4f}, {centroid_world_point.y:.4f}",
        )

        # Publish the point of interest visualization
        gc_poi = GridCells()

        # Copy the data from the POI
        gc_poi.header.frame_id = msg.header.frame_id
        gc_poi.cell_width = msg.info.resolution
        gc_poi.cell_height = msg.info.resolution
        gc_poi.cells = [centroid_world_point]
        self.poi_vis_pub.publish(gc_poi)

        # Note the new point of interest
        self.poi = centroid_world_point
        self.poi_updated = True

        # Print the time taken to find the point of interest
        rospy.loginfo(f"POI found in: {time.time() - start_time:.4f}s")

    def get_frontier_centroids(self, mapdata: OccupancyGrid) -> list[tuple[int, int]]:
        """
        Returns the centroids of the frontier groups of the current map.
        :return [list] The centroids of the frontier groups.
        """
        # Note the start time
        start_time = time.time()

        # Get the frontier cells
        frontier_cells = self.get_frontier_cells(mapdata)

        # If there are no frontier cells, return an empty list
        if not frontier_cells:
            return []

        # Calculate the blobs of the frontier cells
        blobs, min_x, min_y = self.get_frontier_blobs(frontier_cells)

        # Get the centroids of the frontier blobs
        centroids = []
        for i in range(1, blobs[0]):
            # Get the cells of the blob
            blob = np.argwhere(blobs[1] == i)

            # If the blob is too small, skip it
            if len(blob) < 5:
                continue

            # Calculate the centroid of the blob
            x_sum = 0
            y_sum = 0
            for cell in blob:
                x_sum += cell[1]
                y_sum += cell[0]
            centroids.append((x_sum // len(blob) + min_x, y_sum // len(blob) + min_y))

        # Print the time taken to find the frontier centroids
        rospy.loginfo(
            f"Total time for frontier centroids: {time.time() - start_time:.4f}s"
        )
        return centroids

    def get_frontier_blobs(
        self, ungrouped_cells: list[tuple[int, int]]
    ) -> tuple[tuple[int, cv2.MatLike], int, int]:
        """
        Returns the frontier groups of the current map.
        :return [list] The frontier groups.
        """
        # Note the start time
        start_time = time.time()

        # Get the bounds of the frontier cells
        min_x = min(ungrouped_cells, key=lambda x: x[0])[0]
        max_x = max(ungrouped_cells, key=lambda x: x[0])[0]
        min_y = min(ungrouped_cells, key=lambda x: x[1])[1]
        max_y = max(ungrouped_cells, key=lambda x: x[1])[1]

        # Create an array of zeros the same size as the working area of the frontier cells
        frontier_array = np.zeros(
            (max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8
        )

        # Add the frontier cells to the array at their positions
        for cell in ungrouped_cells:
            frontier_array[cell[1] - min_y, cell[0] - min_x] = 255

        # Calculate the blobs
        blobs = cv2.connectedComponents(frontier_array)

        # Print the time taken to find the frontier groups
        rospy.loginfo(f"Frontier groups found in: {time.time() - start_time:.4f}s")
        return [blobs, min_x, min_y]

    @staticmethod
    @njit
    def is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
        """
        Returns True if the two cells are adjacent.
        :return [bool] True if the two cells are adjacent.
        """
        if abs(a[0] - b[0]) > 1:
            return False
        return abs(a[1] - b[1]) <= 1

    def get_frontier_cells(self, mapdata: OccupancyGrid) -> list[tuple[int, int]]:
        """
        Returns the frontier cells of the current map.
        :return [list] The frontier cells.
        """
        # Note the start time
        start_time = time.time()

        # Initialize the frontier list, visited set, and queue set
        frontier = set()
        queue = set()
        visited = set()

        # Seed the queue with the current position
        curr_coords = PathPlanner.world_to_grid(mapdata, self.curr_pose.pose.position)
        queue.add(curr_coords)

        # Process until the queue is empty
        while len(queue) > 0:
            # Get the current cell
            curr_cell = queue.pop()

            # Check if the cell is a frontier cell
            if FrontierFinder.is_on_frontier(mapdata, curr_cell):
                frontier.add(curr_cell)
            else:
                # Mark the cell as visited
                visited.add(curr_cell)

            # Add the neighbors to the queue
            for neighbor in PathPlanner.neighbors_of_4(
                mapdata, curr_cell, exclude_unknown=True
            ):
                if neighbor not in visited:
                    if neighbor not in frontier:
                        queue.add(neighbor)

        # Publish the point of interest visualization
        gc_frontier = GridCells()

        # Copy the data from the frontier cells
        gc_frontier.header.frame_id = mapdata.header.frame_id
        gc_frontier.cell_width = mapdata.info.resolution
        gc_frontier.cell_height = mapdata.info.resolution
        gc_frontier.cells = [
            PathPlanner.grid_to_world(mapdata, cell) for cell in frontier
        ]
        self.frontier_vis_pub.publish(gc_frontier)

        # Print the time taken to find the frontier cells
        rospy.loginfo(f"Frontier cells found in: {time.time() - start_time:.4f}s")

        return list(frontier)

    @staticmethod
    def is_on_frontier(mapdata: OccupancyGrid, cell: tuple[int, int]) -> bool:
        """
        Returns True if the cell is a frontier cell.
        :return [bool] True if the cell is a frontier cell.
        """
        # Cell must be walkable
        if not PathPlanner.is_cell_walkable(mapdata, cell, exclude_unknown=True):
            return False

        # Cell must have at least one unknown neighbor
        if PathPlanner.is_cell_unknown(mapdata, (cell[0] + 1, cell[1])):
            return True
        if PathPlanner.is_cell_unknown(mapdata, (cell[0] - 1, cell[1])):
            return True
        if PathPlanner.is_cell_unknown(mapdata, (cell[0], cell[1] + 1)):
            return True
        if PathPlanner.is_cell_unknown(mapdata, (cell[0], cell[1] - 1)):
            return True

        # If we got here, the cell is a not frontier cell
        # There are no unknown neighbors
        return False


if __name__ == "__main__":
    # Create the FrontierFinder node
    FrontierFinder()

    # All of the logic is handled by callbacks, so we just need to keep the node alive
    rospy.spin()
