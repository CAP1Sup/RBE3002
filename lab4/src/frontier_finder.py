#!/usr/bin/env python3

# Required if using Python 3.8 or below
from __future__ import annotations
import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid, GridCells
from lab3.path_planner import PathPlanner


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
        Updates the point of interest for the robot to explore.
        Runs when the map is updated.
        """
        # Get the centroids of the frontier groups
        centroids = FrontierFinder.get_frontier_centroids(msg)

        # If there are no frontier groups, set the point of interest to None
        if not centroids:
            if self.poi is not None:
                self.poi = None
                self.poi_updated = True
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
        rospy.loginfo(f"New point of interest: {centroid_world_point}")

        # Publish the point of interest visualization
        gc_poi = GridCells()

        # Copy the data from the POI message
        gc_poi.header.frame_id = msg.header.frame_id
        gc_poi.cell_width = msg.info.resolution
        gc_poi.cell_height = msg.info.resolution
        gc_poi.cells = [centroid_world_point]
        self.poi_vis_pub.publish(gc_poi)

        # Note the new point of interest
        self.poi = centroid_world_point
        self.poi_updated = True

    @staticmethod
    def get_frontier_centroids(mapdata: OccupancyGrid) -> list[tuple[int, int]]:
        """
        Returns the centroids of the frontier groups of the current map.
        :return [list] The centroids of the frontier groups.
        """
        # Get the frontier groups
        groups = FrontierFinder.get_frontier_groups(mapdata)

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

    @staticmethod
    def get_frontier_groups(mapdata: OccupancyGrid) -> list[list[tuple[int, int]]]:
        """
        Returns the frontier groups of the current map.
        :return [list] The frontier groups.
        """
        # Get the frontier cells
        ungrouped_cells = FrontierFinder.get_frontier_cells(mapdata)

        # Loop until all cells are grouped
        groups = []
        while ungrouped_cells:
            # Create a new group with the first ungrouped cell
            group = [ungrouped_cells.pop(0)]

            repeat_loop = True
            while repeat_loop:
                # Assume the loop will not repeat
                repeat_loop = False

                # Use a try loop so we can break out of both loops at once
                try:
                    # Loop through the ungrouped cells, adding them to the group if they are adjacent
                    for cell in ungrouped_cells:
                        for group_cell in group:
                            if FrontierFinder.is_adjacent(cell, group_cell):
                                # Add the cell to the group
                                group.append(cell)
                                ungrouped_cells.remove(cell)

                                # Repeat the loop, the new cell may be adjacent to other ungrouped cells
                                raise StopIteration
                except StopIteration:
                    repeat_loop = True

            # Add the new group to the list of groups
            groups.append(group)
        return groups

    @staticmethod
    def is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
        """
        Returns True if the two cells are adjacent.
        :return [bool] True if the two cells are adjacent.
        """
        return (abs(a[0] - b[0]) <= 1) and (abs(a[1] - b[1]) <= 1)

    @staticmethod
    def get_frontier_cells(mapdata: OccupancyGrid) -> list[tuple[int, int]]:
        """
        Returns the frontier cells of the current map.
        :return [list] The frontier cells.
        """
        frontier = []
        for x in range(mapdata.info.width):
            for y in range(mapdata.info.height):
                if FrontierFinder.is_on_frontier(mapdata, (x, y)):
                    frontier.append((x, y))
        return frontier

    @staticmethod
    def is_on_frontier(mapdata: OccupancyGrid, cell: tuple[int, int]) -> bool:
        """
        Returns True if the cell is a frontier cell.
        :return [bool] True if the cell is a frontier cell.
        """
        return (
            mapdata.data[PathPlanner.grid_to_index(mapdata, cell)] == -1
            and len(PathPlanner.neighbors_of_4(mapdata, cell, exclude_unknown=True)) > 0
        )


if __name__ == "__main__":
    # Create the FrontierFinder node
    FrontierFinder()

    # All of the logic is handled by callbacks, so we just need to keep the node alive
    rospy.spin()
