#!/usr/bin/env python3

# Required if using Python 3.8 or below
from __future__ import annotations
import copy
import math
import rospy
from nav_msgs.srv import GetPlan, GetMap
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Pose, PoseStamped
from tf.transformations import euler_from_quaternion
import priority_queue


class PathPlanner:

    def __init__(self):
        """
        Class constructor
        """
        # Initialize the node and call it "path_planner"
        rospy.init_node("path_planner")

        # Create a new service called "plan_path" that accepts messages of
        # type GetPlan and calls self.plan_path() when a message is received
        rospy.Service("plan_path", GetPlan, self.plan_path)

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
    def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """
        Calculates the Euclidean distance between two points.
        :param p1 [(float, float)] first point.
        :param p2 [(float, float)] second point.
        :return   [float]          distance.
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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
        poses = []
        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = mapdata.header.frame_id
            pose.pose = Pose(position=PathPlanner.grid_to_world(mapdata, point))
            poses.append(pose)
        return poses

    @staticmethod
    def is_cell_walkable(mapdata: OccupancyGrid, p: tuple[int, int]) -> bool:
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
        return mapdata.data[index] == 0

    @staticmethod
    def neighbors_of_4(
        mapdata: OccupancyGrid, p: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Returns the walkable 4-neighbors cells of (x,y) in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable 4-neighbors.
        """
        neighbors = []
        if PathPlanner.is_cell_walkable(mapdata, (p[0] - 1, p[1])):
            neighbors.append((p[0] - 1, p[1]))
        if PathPlanner.is_cell_walkable(mapdata, (p[0] + 1, p[1])):
            neighbors.append((p[0] + 1, p[1]))
        if PathPlanner.is_cell_walkable(mapdata, (p[0], p[1] - 1)):
            neighbors.append((p[0], p[1] - 1))
        if PathPlanner.is_cell_walkable(mapdata, (p[0], p[1] + 1)):
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

    @staticmethod
    def request_map() -> OccupancyGrid:
        """
        Requests the map from the map server.
        :return [OccupancyGrid] The grid if the service call was successful,
                                None in case of error.
        """
        try:
            # Wait for the service to become available
            rospy.wait_for_service("static_map", timeout=5.0)

            # Create a proxy for the map service
            get_map = rospy.ServiceProxy("static_map", GetMap)

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

        # Change the data of the map to a list
        # This is necessary because the data is originally a tuple and tuples are immutable
        mapdata.data = list(mapdata.data)

        # Copy the original so we can modify the C-space without affecting the original map
        cspace = copy.deepcopy(mapdata)

        # Clear the data of the C-space
        cspace.data = [0] * len(mapdata.data)

        # Go through each cell in the occupancy grid
        for x in range(mapdata.info.width):
            for y in range(mapdata.info.height):
                # Check if the cell is occupied
                index = PathPlanner.grid_to_index(mapdata, (x, y))
                if mapdata.data[index] != 0:
                    # Inflate the obstacles
                    for i in range(-padding, padding + 1):
                        for j in range(-padding, padding + 1):
                            # Check if the cell is within the boundaries of the grid
                            if (
                                x + i >= 0
                                and x + i < mapdata.info.width
                                and y + j >= 0
                                and y + j < mapdata.info.height
                            ):
                                # Mark the cell as occupied
                                cspace.data[
                                    PathPlanner.grid_to_index(mapdata, (x + i, y + j))
                                ] = 100

        # Create a GridCells message to publish the C-space
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
                if cspace.data[index] != 0:
                    cspace_grid.cells.append(PathPlanner.grid_to_world(cspace, (x, y)))

        # Publish GridCells message
        self.cspace_pub.publish(cspace_grid)

        # Return the C-space
        return cspace

    @staticmethod
    def a_star(
        mapdata: OccupancyGrid, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        rospy.loginfo(
            "Executing A* from (%d,%d) to (%d,%d)"
            % (start[0], start[1], goal[0], goal[1])
        )

        # Create a priority queue, with the start node as the first element
        # Each element is a list [previous, position, cost]
        frontier = priority_queue.PriorityQueue()
        frontier.put(
            [start, list([start]), 0], PathPlanner.euclidean_distance(start, goal)
        )

        # Process the frontier until the goal is reached
        while True:
            # Check if the frontier is empty
            if frontier.empty():
                rospy.logwarn("No path found")
                return []

            # Get the next node from the frontier
            current = frontier.get()
            # rospy.loginfo(f"Current node: {current}")

            # Check if the goal has been reached
            if current[0] == goal:
                rospy.loginfo("Goal reached")
                return current[1]

            # Get all of the nodes of the frontier
            frontier_nodes = frontier.get_queue()

            # Strip the paths and costs from the nodes, leaving only the positions
            frontier_positions = [node[1][0] for node in frontier_nodes]

            # Get the neighbors of the current node
            for neighbor in PathPlanner.neighbors_of_4(mapdata, current[0]):
                # Add the neighbor to the frontier if it is not already there
                if neighbor in frontier_positions:
                    # Check if the new path is shorter
                    index = frontier_positions.index(neighbor)
                    if (
                        current[2] + 1 + PathPlanner.euclidean_distance(neighbor, goal)
                        < frontier_nodes[index][0]
                    ):
                        frontier.replace(
                            [neighbor, current[1] + [neighbor], current[2] + 1],
                            current[2]
                            + 1
                            + PathPlanner.euclidean_distance(neighbor, goal),
                        )
                else:
                    frontier.put(
                        [neighbor, current[1] + [neighbor], current[2] + 1],
                        current[2] + 1 + PathPlanner.euclidean_distance(neighbor, goal),
                    )

            # TODO: Publish the frontier

    @staticmethod
    def optimize_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Optimizes the path, removing unnecessary intermediate nodes.
        :param path [[(x,y)]] The path as a list of tuples (grid coordinates)
        :return     [[(x,y)]] The optimized path as a list of tuples (grid coordinates)
        """
        rospy.loginfo("Optimizing path")
        # TODO Implement path optimization
        return path

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

    def plan_path(self, msg):
        """
        Plans a path between the start and goal locations in the requested.
        Internally uses A* to plan the optimal path.
        :param msg [GetPlan] The path planning request.
        """
        # Request the map
        # In case of error, return an empty path
        mapdata = PathPlanner.request_map()
        if mapdata is None:
            return Path()

        # Calculate the C-space and publish it
        cspacedata = self.calc_cspace(mapdata, 1)

        # Execute A*
        start = PathPlanner.world_to_grid(cspacedata, msg.start.pose.position)
        goal = PathPlanner.world_to_grid(cspacedata, msg.goal.pose.position)
        path = PathPlanner.a_star(cspacedata, start, goal)

        # Optimize waypoints
        waypoints = PathPlanner.optimize_path(path)

        # Publish the path
        self.path_pub.publish(self.path_to_message(mapdata, waypoints))

        # Return a Path message
        return self.path_to_message(mapdata, waypoints)

    def run(self):
        """
        Runs the node until Ctrl-C is pressed.
        """
        rospy.spin()


if __name__ == "__main__":
    PathPlanner().run()
