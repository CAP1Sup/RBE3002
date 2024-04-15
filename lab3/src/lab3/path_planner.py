#!/usr/bin/env python3

# Required if using Python 3.8 or below
from __future__ import annotations
import copy
import math
import rospy
import cv2
import argparse
import numpy as np
from nav_msgs.srv import GetPlan, GetMap
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Pose, PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from lab3.priority_queue import PriorityQueue
import time


class PathPlanner:

    def __init__(self, dynamic_map=False, visualize=True):
        """
        Class constructor
        """
        # Initialize the node and call it "path_planner"
        rospy.init_node("path_planner")

        # Create a new service called "plan_path" that accepts messages of
        # type GetPlan and calls self.plan_path() when a message is received
        rospy.Service("plan_path", GetPlan, self.plan_path)

        # If we're providing visuals, create the publishers
        if visualize:
            # Create a publisher for the C-space (the enlarged occupancy grid)
            # The topic is "/path_planner/cspace", the message type is GridCells
            self.cspace_pub = rospy.Publisher(
                "/path_planner/cspace", GridCells, queue_size=10
            )

            # Create publishers for A* (expanded cells, frontier, ...)
            # Choose the topic names, the message type is GridCells
            self.expanded_pub = rospy.Publisher(
                "/path_planner/expanded", GridCells, queue_size=10
            )
            self.frontier_pub = rospy.Publisher(
                "/path_planner/frontier", GridCells, queue_size=10
            )
            self.path_pub = rospy.Publisher("/path_planner/path", Path, queue_size=10)

        # Initialize the request counter
        self.request_counter = 0

        # Store the flags
        self.dynamic_map = dynamic_map
        self.visualize = visualize

        # Sleep to allow roscore to do some housekeeping
        rospy.sleep(1.0)
        rospy.loginfo("Path planner node ready")

    @staticmethod
    def grid_to_index(mapdata: OccupancyGrid, p: tuple[int, int]) -> int:
        """
        Returns the index corresponding to the given (x,y) coordinates in the occupancy grid.
        :param p [(int, int)] The cell coordinate.
        :return  [int] The index.
        """
        # Wrap the 2D index into a 1D index
        return p[0] + p[1] * mapdata.info.width

    @staticmethod
    def manhattan_dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """
        Calculates the Manhattan distance between two points.
        :param p1 [(float, float)] first point.
        :param p2 [(float, float)] second point.
        :return   [float]          heuristic distance.
        """
        # Much faster than Euclidean distance
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def grid_to_world(mapdata: OccupancyGrid, p: tuple[int, int]) -> Point:
        """
        Transforms a cell coordinate in the occupancy grid into a world coordinate.
        :param mapdata [OccupancyGrid] The map information.
        :param p [(int, int)] The cell coordinate.
        :return        [Point]         The position in the world.
        """
        # Create a new point
        point = Point()

        # Calculate the world coordinates
        # Cells have to be offset by half a cell to be in the center of the cell
        point.x = (
            p[0] + 0.5
        ) * mapdata.info.resolution + mapdata.info.origin.position.x
        point.y = (
            p[1] + 0.5
        ) * mapdata.info.resolution + mapdata.info.origin.position.y
        return point

    @staticmethod
    def world_to_grid(mapdata: OccupancyGrid, wp: Point) -> tuple[int, int]:
        """
        Transforms a world coordinate into a cell coordinate in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param wp      [Point]         The world coordinate.
        :return        [(int,int)]     The cell position as a tuple.
        """
        # Get the rotation around the z-axis
        (roll, pitch, yaw) = euler_from_quaternion(
            [
                mapdata.info.origin.orientation.x,
                mapdata.info.origin.orientation.y,
                mapdata.info.origin.orientation.z,
                mapdata.info.origin.orientation.w,
            ]
        )

        # Convert the point in space to an index in the 2D occupancy grid
        ix = int(
            (
                wp.x * math.cos(yaw)
                - wp.y * math.sin(yaw)
                - mapdata.info.origin.position.x
            )
            / mapdata.info.resolution
        )
        iy = int(
            (
                wp.x * math.sin(yaw)
                + wp.y * math.cos(yaw)
                - mapdata.info.origin.position.y
            )
            / mapdata.info.resolution
        )

        # Return the grid coordinates
        return (ix, iy)

    @staticmethod
    def path_to_poses(
        mapdata: OccupancyGrid, path: list[tuple[int, int]]
    ) -> list[PoseStamped]:
        """
        Converts the given path into a list of PoseStamped.
        :param mapdata [OccupancyGrid] The map information.
        :param  path   [[(int,int)]]   The path as a list of tuples (cell coordinates).
        :return        [[PoseStamped]] The path as a list of PoseStamped (world coordinates).
        """
        # Loop through the path, appending the poses to the list
        poses = []
        for i in range(len(path)):
            # Create a new pose and set the frame id to the same as the map
            pose = PoseStamped()
            pose.header.frame_id = mapdata.header.frame_id

            # Calculate the world coordinates of the cell
            pose.pose.position = PathPlanner.grid_to_world(mapdata, path[i])

            # Check if there is a next cell in the path
            if i > 0:
                # We should face towards the previous cell in the path
                orientation = math.atan2(
                    path[i][1] - path[i - 1][1], path[i][0] - path[i - 1][0]
                )
            else:
                # If we're at the start, we should face towards the next cell
                orientation = math.atan2(
                    path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0]
                )

            # Convert the orientation to a quaternion
            quaternion = quaternion_from_euler(0, 0, orientation)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]

            # Pose is ready, append it to the list
            poses.append(pose)
        return poses

    @staticmethod
    def is_cell_walkable(
        mapdata: OccupancyGrid, p: tuple[int, int], exclude_unknown=False
    ) -> bool:
        """
        A cell is walkable if all of these conditions are true:
        1. It is within the boundaries of the grid;
        2. It is free (not unknown, not occupied by an obstacle)
        :param mapdata [OccupancyGrid] The map information.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [bool]          True if the cell is walkable, False otherwise
        """
        # Check if the cell is within the boundaries of the grid
        if p[0] < 0 or p[0] >= mapdata.info.width:
            return False
        if p[1] < 0 or p[1] >= mapdata.info.height:
            return False

        # Check if the cell is free
        index = PathPlanner.grid_to_index(mapdata, p)
        return mapdata.data[index] <= 25 and (
            mapdata.data[index] != -1 or not exclude_unknown
        )

    @staticmethod
    def neighbors_of_4(
        mapdata: OccupancyGrid, p: tuple[int, int], exclude_unknown=False
    ) -> list[tuple[int, int]]:
        """
        Returns the walkable 4-neighbors cells of (x,y) in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable 4-neighbors.
        """
        neighbors = []
        if PathPlanner.is_cell_walkable(mapdata, (p[0] - 1, p[1]), exclude_unknown):
            neighbors.append((p[0] - 1, p[1]))
        if PathPlanner.is_cell_walkable(mapdata, (p[0] + 1, p[1]), exclude_unknown):
            neighbors.append((p[0] + 1, p[1]))
        if PathPlanner.is_cell_walkable(mapdata, (p[0], p[1] - 1), exclude_unknown):
            neighbors.append((p[0], p[1] - 1))
        if PathPlanner.is_cell_walkable(mapdata, (p[0], p[1] + 1), exclude_unknown):
            neighbors.append((p[0], p[1] + 1))
        return neighbors

    @staticmethod
    def neighbors_of_8(
        mapdata: OccupancyGrid, p: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Returns the walkable 8-neighbors cells of (x,y) in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable 8-neighbors.
        """
        neighbors = PathPlanner.neighbors_of_4(mapdata, p)
        if PathPlanner.is_cell_walkable(mapdata, (p[0] - 1, p[1] - 1)):
            neighbors.append((p[0] - 1, p[1] - 1))
        if PathPlanner.is_cell_walkable(mapdata, (p[0] + 1, p[1] - 1)):
            neighbors.append((p[0] + 1, p[1] - 1))
        if PathPlanner.is_cell_walkable(mapdata, (p[0] - 1, p[1] + 1)):
            neighbors.append((p[0] - 1, p[1] + 1))
        if PathPlanner.is_cell_walkable(mapdata, (p[0] + 1, p[1] + 1)):
            neighbors.append((p[0] + 1, p[1] + 1))
        return neighbors

    def request_map(self) -> OccupancyGrid:
        """
        Requests the map from the map server.
        :return [OccupancyGrid] The grid if the service call was successful,
                                None in case of error.
        """
        try:
            # Decide which service to call based on the dynamic_map flag
            if self.dynamic_map:
                map_service_name = "dynamic_map"
            else:
                map_service_name = "static_map"

            # Wait for the service to become available
            rospy.wait_for_service(map_service_name, timeout=5.0)

            # Create a proxy for the map service
            get_map = rospy.ServiceProxy(map_service_name, GetMap)

            # Call the service
            return get_map().map
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return None

    def calc_cspace(self, mapdata: OccupancyGrid, padding: int) -> OccupancyGrid:
        """
        Calculates the C-Space, i.e., makes the obstacles in the map thicker.
        Publishes the list of cells that were added to the original map.
        :param mapdata [OccupancyGrid] The map data.
        :param padding [int]           The number of cells around the obstacles.
        :return        [OccupancyGrid] The C-Space.
        """
        rospy.loginfo("Calculating C-Space")

        # Copy the original so we can modify the C-space without affecting the original map
        cspace = copy.deepcopy(mapdata)

        # Convert the data to a numpy array
        mapdata_np = np.array(mapdata.data).reshape(
            mapdata.info.height, mapdata.info.width
        )

        # Convert the numpy array to uint8
        mapdata_np = mapdata_np.astype(np.uint8)

        # Create a element for the dilation
        element = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * padding + 1, 2 * padding + 1)
        )

        # Dilate the obstacles
        cspace_np = cv2.dilate(mapdata_np, element, iterations=1)

        # Convert the numpy array back to int8
        # This will preserve the -1 values for unknown cells
        cspace_np = cspace_np.astype(np.int8)

        # Convert the numpy array back to a list
        cspace.data = cspace_np.flatten().tolist()

        # If we're providing visuals, publish the C-space
        if self.visualize:
            # Create a GridCells message
            cspace_grid = GridCells()

            # Copy the data from the cspace map
            cspace_grid.header.frame_id = cspace.header.frame_id
            cspace_grid.cell_width = cspace.info.resolution
            cspace_grid.cell_height = cspace.info.resolution

            # Populate the GridCells message with the C-space data
            for x in range(cspace.info.width):
                for y in range(cspace.info.height):
                    # Check if the cell is occupied
                    index = PathPlanner.grid_to_index(cspace, (x, y))
                    if cspace.data[index] >= 25:
                        cspace_grid.cells.append(
                            PathPlanner.grid_to_world(cspace, (x, y))
                        )

            # Publish GridCells message
            self.cspace_pub.publish(cspace_grid)

        # Return the C-space
        return cspace

    def a_star(
        self, mapdata: OccupancyGrid, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Executes the A* algorithm to find the optimal path from the start to the goal.
        :param mapdata [OccupancyGrid] The map data.
        :param start [(int, int)] The starting point coordinates.
        :param goal [(int, int)] The goal point coordinates.
        :return [list[(int, int)]] The optimal path as a list of coordinates.
        """

        rospy.loginfo(
            "Executing A* from (%d,%d) to (%d,%d)"
            % (start[0], start[1], goal[0], goal[1])
        )

        # Note the time
        start_time = time.time()

        # Create a priority queue, with the start node as the first element
        frontier = PriorityQueue()
        frontier.put(start, 0)

        # Create dictionaries to store the cost and the parent of each node
        cost_so_far = {start: 0}
        came_from = {start: None}

        # If we're providing visuals, initialize the GridCells messages
        if self.visualize:
            expanded_cells = GridCells()
            expanded_cells.header.frame_id = mapdata.header.frame_id
            expanded_cells.cell_width = mapdata.info.resolution
            expanded_cells.cell_height = mapdata.info.resolution

        # Process the frontier until the goal is reached
        path_found = False
        while not frontier.empty() and not path_found:
            # Get the current node
            current = frontier.get()

            # Check if the goal has been reached
            if current == goal:
                rospy.loginfo(
                    "Goal reached in %.6f seconds" % (time.time() - start_time)
                )
                path_found = True
                path = PathPlanner.reconstruct_path(came_from, start, goal)

            # Loop through the neighbors of the current node
            else:
                neighbors = PathPlanner.neighbors_of_4(mapdata, current)
                for neighbor in neighbors:
                    # Calculate the cost to move to the neighbor
                    new_cost = cost_so_far[current] + 1

                    # Check if the neighbor is not in the cost map or the new cost is lower
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        # Update the cost and the parent of the neighbor
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + PathPlanner.manhattan_dist(neighbor, goal)
                        frontier.put(neighbor, priority)
                        came_from[neighbor] = current

            # Visualize the frontier and the expanded cells
            if self.visualize:
                # Publish the expanded cells
                expanded_cells.cells = [
                    PathPlanner.grid_to_world(mapdata, node)
                    for node in cost_so_far.keys()
                ]
                self.expanded_pub.publish(expanded_cells)

                # Publish the frontier cells
                # Reuse the expanded_cells variable
                expanded_cells.cells = [
                    PathPlanner.grid_to_world(mapdata, node[1])
                    for node in frontier.get_queue()
                ]
                self.frontier_pub.publish(expanded_cells)

        # Return the found path or an empty list
        if path_found:
            return path
        else:
            # The frontier is empty, the goal is unreachable
            rospy.logwarn("Goal is unreachable")

            # Time taken
            rospy.loginfo(f"Time taken: {time.time() - start_time}")
            return []

    @staticmethod
    def reconstruct_path(
        came_from: dict[tuple[int, int], tuple[int, int]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Reconstructs the path from the start to the goal using the came_from dictionary.
        :param came_from [dict] The dictionary containing the parent of each node.
        :param start     [(int,int)] The start node.
        :param goal      [(int,int)] The goal node.
        :return          [[(int,int)]] The path as a list of tuples.
        """
        path = [goal]
        current = goal
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def optimize_path(
        mapdata: OccupancyGrid, path: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        Optimizes the path, removing unnecessary intermediate nodes.
        :param path [[(x,y)]] The path as a list of tuples (grid coordinates)
        :return     [[(x,y)]] The optimized path as a list of tuples (grid coordinates)
        """
        rospy.loginfo("Optimizing path")

        # Loop through the path
        i = 0
        while i < len(path) - 2:
            # Check if the path from the current node to the next node is clear
            if PathPlanner.is_clear_path(mapdata, path[i], path[i + 2]):
                # Remove the intermediate node
                path.pop(i + 1)
            else:
                i += 1
        return path

    @staticmethod
    def is_clear_path(
        mapdata: OccupancyGrid, point1: tuple[int, int], point2: tuple[int, int]
    ) -> bool:
        """
        Checks if the path between two points is clear.
        :param point1 [(int,int)] The first point.
        :param point2 [(int,int)] The second point.
        :return  [bool]       True if the path is clear, False otherwise.
        """
        # If the X or Y coordinates are the same, the path is clear
        if point1[0] == point2[0] or point1[1] == point2[1]:
            return True

        # Calculate the angle between the two points
        angle = math.atan2(point2[1] - point1[1], point2[0] - point1[0])

        # If the angle is a multiple of 90 degrees, the path is clear
        if (
            abs(angle) % (math.pi / 2) < 0.01
            or abs(angle) % (math.pi / 2) > math.pi / 2 - 0.01
        ):
            return True

        # The path is not clear
        return False
        """
        # Convert the grid coordinates to world coordinates
        p1 = PathPlanner.grid_to_world(mapdata, point1)

        # Calculate the world distance between the two points
        distance = (
            math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
            * mapdata.info.resolution
        )

        # Check if all cells touched by the path are clear
        search_precision = 0.1
        for dist in np.arange(0, distance, search_precision):
            x = p1.x + dist * math.cos(angle)
            y = p1.y + dist * math.sin(angle)
            grid_point = PathPlanner.world_to_grid(mapdata, Point(x, y, 0))
            if not PathPlanner.is_cell_walkable(mapdata, grid_point):
                return False
        return True
        """

    def path_to_message(
        self, mapdata: OccupancyGrid, point_path: list[tuple[int, int]]
    ) -> Path:
        """
        Takes a path on the grid and returns a Path message.
        :param point_path [[(int,int)]] The path on the grid (a list of tuples)
        :return     [Path]        A Path message (the coordinates are expressed in the world)
        """
        rospy.loginfo("Converting coord list to a Path message")

        # Create a Path message
        path = Path()

        # Copy the frame id from the map data
        path.header.frame_id = mapdata.header.frame_id

        # Populate the Path message with the path
        path.poses = PathPlanner.path_to_poses(mapdata, point_path)
        return path

    def plan_path(self, msg: GetPlan) -> Path:
        """
        Plans a path between the start and goal locations in the requested.
        Internally uses A* to plan the optimal path.
        :param msg [GetPlan] The path planning request.
        """
        # Request the map
        # In case of error, return an empty path
        mapdata = self.request_map()
        if mapdata is None:
            return Path()

        # Calculate the C-space and publish it
        cspacedata = self.calc_cspace(mapdata, 1)

        # Execute A*
        start = PathPlanner.world_to_grid(cspacedata, msg.start.pose.position)
        goal = PathPlanner.world_to_grid(cspacedata, msg.goal.pose.position)
        path = self.a_star(cspacedata, start, goal)

        # Optimize waypoints
        waypoints = PathPlanner.optimize_path(cspacedata, path)

        # Generate a Path message from the found path
        path_msg = self.path_to_message(cspacedata, waypoints)

        # Publish the path
        if self.visualize:
            self.path_pub.publish(path_msg)

        # Return the Path message
        return path_msg

    def run(self):
        """
        Runs the node until Ctrl-C is pressed.
        """
        rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path Planner")
    parser.add_argument(
        "-d",
        "--use_dynamic_map",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--no_visualization",
        action="store_true",
    )
    args, unknown = parser.parse_known_args()
    PathPlanner(args.use_dynamic_map, not args.no_visualization).run()
