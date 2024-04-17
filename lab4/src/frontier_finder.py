#!/usr/bin/env python3

# Required if using Python 3.8 or below
from __future__ import annotations

import time

import rospy
from geometry_msgs.msg import Point, PoseStamped
from lab3.path_planner import PathPlanner
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

        # Set of visited cells
        self.visited = set()

        # Cells of the last frontier
        self.last_frontier = set()

        # Point of interest for the robot to explore
        self.poi = None
        self.poi_updated = False

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

        # Calculate the current coordinates of the robot
        curr_coords = PathPlanner.world_to_grid(msg, self.curr_pose.pose.position)

        # Sort the list by the Manhattan distance to the robot
        closest_centroid = None
        closest_dist = float("inf")
        for centroid in centroids:
            dist = PathPlanner.manhattan_dist(
                centroid, (curr_coords[0], curr_coords[1])
            )
            if dist < closest_dist:
                closest_dist = dist
                closest_centroid = centroid

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
            f"New point of interest: {centroid_world_point.x}, {centroid_world_point.y}",
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
        # Get the frontier cells
        frontier_cells = self.get_frontier_cells(mapdata)

        # Get the frontier groups
        groups = self.get_frontier_groups(frontier_cells)

        # Remove groups with less than 5 cells
        groups = [group for group in groups if len(group) >= 3]

        # Calculate the centroids of the groups
        centroids = []
        for group in groups:
            x_sum = 0
            y_sum = 0
            for cell in group:
                x_sum += cell[0]
                y_sum += cell[1]
            centroids.append((x_sum // len(group), y_sum // len(group)))
        return centroids

    def get_frontier_groups(
        self, ungrouped_cells: list[tuple[int, int]]
    ) -> list[list[tuple[int, int]]]:
        """
        Returns the frontier groups of the current map.
        :return [list] The frontier groups.
        """
        # Note the start time
        start_time = time.time()

        # Initialize the groups list
        groups = []

        # Sort the cells by the x coordinate
        ungrouped_cells = sorted(ungrouped_cells, key=lambda x: x[0])

        # Loop until the ungrouped cells are empty
        while ungrouped_cells:
            # Initialize the group
            group = [ungrouped_cells.pop(0)]

            # Check if there are any cells left
            if not ungrouped_cells:
                groups.append(group)
                break

            # Get the maximum x coordinate of the group
            max_x = group[-1][0] + 1

            skipped_cells = 0
            while ungrouped_cells[skipped_cells][0] <= max_x:
                try:
                    for member in group:
                        if FrontierFinder.is_adjacent(
                            ungrouped_cells[skipped_cells], member
                        ):
                            group.append(ungrouped_cells.pop(skipped_cells))
                            max_x = group[-1][0] + 1
                            # Throw an exception to break out of the loop
                            # Prevents the skipped_cells index from being incremented
                            raise StopIteration

                    # No adjacent cells found, increment the skipped_cells index
                    skipped_cells += 1

                    # Check if we have reached the end of the list
                    if skipped_cells >= len(ungrouped_cells):
                        break

                except StopIteration:
                    if skipped_cells >= len(ungrouped_cells):
                        break
                    else:
                        continue

            # Add the group to the groups list
            groups.append(group)

        # Print the time taken to find the frontier groups
        rospy.loginfo(f"Frontier groups found in: {time.time() - start_time:.4f}s")
        return groups

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

        if not self.last_frontier:
            # Seed the queue with the current position
            curr_coords = PathPlanner.world_to_grid(
                mapdata, self.curr_pose.pose.position
            )
            queue.add(curr_coords)
        else:
            for cell in self.last_frontier:
                queue.add(cell)

        # Process until the queue is empty
        while len(queue) > 0:
            # Get the current cell
            curr_cell = queue.pop()

            # Check if the cell is a frontier cell
            if FrontierFinder.is_on_frontier(mapdata, curr_cell):
                frontier.add(curr_cell)
            else:
                # Mark the cell as visited
                self.visited.add(curr_cell)

            # Add the neighbors to the queue
            for neighbor in PathPlanner.neighbors_of_4(
                mapdata, curr_cell, exclude_unknown=True
            ):
                if neighbor not in self.visited:
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

        # Store the last frontier
        self.last_frontier = frontier

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
