#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

class Plane:
    """ This class defines a plane object in the blocks task environment. It is defined as a 2D plane with an infinite
        length and width. Its location is defined by the normal axis that the plane is set to.
    """
    def __init__(self, axis='z', position=0.0, stiffness=2000):
        self.axis = axis
        self.position = position
        self.stiffness = stiffness
        self.is_colliding = False

    def over_plane(self, point):
        """ This method checks if a point is over the plane. """
        if self.axis == 'x':
            return point[0] > self.position
        elif self.axis == 'y':
            return point[1] > self.position
        elif self.axis == 'z':
            return point[2] > self.position
        else:
            raise ValueError('Invalid axis. Please use "x", "y", or "z".')

    def through_plane(self, point, radius):
        """ Checks if the point is under the plane and returns the distance between the point and the plane.
        Parameters:
        ------------
        point: list
            The x, y, z coordinates of the center of the sphere.
        radius: float
            The radius of the sphere.

        Returns:
        ------------
        is_colliding: bool
            A boolean value indicating if the sphere is colliding with the plane.
        distance: list
            The distance between the sphere's center and the plane in the x, y, or z direction, normal to the plane.
        """
        if self.axis == 'x':
            distance = [point[0] - self.position, 0.0, 0.0]
            self.is_colliding = point[0] - self.position <= radius
        elif self.axis == 'y':
            distance = [0.0, point[1] - self.position, 0.0]
            self.is_colliding = point[1] - self.position <= radius
        elif self.axis == 'z':
            distance = [0.0, 0.0, point[2] - self.position]
            self.is_colliding = point[2] - self.position <= radius
        else:
            raise ValueError('Invalid axis. Please use "x", "y", or "z".')
        return self.is_colliding, distance

class Table:
    """ This class defines a table object in the blocks task environment. It is defined as a 2D plane with a finite
        length, width, and height. Its location is defined by a user-defined origin point ir (0,0,0) by default.
    """
    def __init__(self, length, width, height, origin=[0.0, 0.0, 0.0], stiffness=2000):
        self.length = length
        self.width = width
        self.height = height
        self.origin = origin
        self.stiffness = stiffness
        self.is_colliding = False

    def over_table(self, point):
        """ This method checks if a point is over and within the size boundaries of the table. """
        x, y, z = point
        x0, y0, z0 = self.origin
        half_height = self.height / 2
        half_length = self.length / 2
        half_width = self.width / 2
        min_x = x0 - half_length
        max_x = x0 + half_length
        min_y = y0 - half_width
        max_y = y0 + half_width
        max_z = z0 + self.height
        return (min_x <= x <= max_x) and (min_y <= y <= max_y) and (z > max_z)

    def under_table(self, point):
        """ This method checks if a point is under and within the size boundaries of the table. """
        x, y, z = point
        x0, y0, z0 = self.origin
        half_height = self.height / 2
        half_length = self.length / 2
        half_width = self.width / 2
        min_x = x0 - half_length
        max_x = x0 + half_length
        min_y = y0 - half_width
        max_y = y0 + half_width
        min_z = z0
        return (min_x <= x <= max_x) and (min_y <= y <= max_y) and (z < min_z)

    def is_colliding_with_table(self, sphere_center, sphere_radius):
        """ Checks to see if a sphere is colliding with the table geometry.
        Parameters:
        ------------
        point: list
            The x, y, z coordinates of the center of the sphere.
        radius: float
            The radius of the sphere.
        """
        # Compute the half-lengths of the cube in each dimension
        half_length = self.length / 2
        half_width = self.width / 2
        half_height = self.height / 2

        # Compute the closest point on the cube to the sphere's center
        x0, y0, z0 = self.origin
        closest_point = np.array([
            np.clip(sphere_center[0], x0 - half_length, x0 + half_length),
            np.clip(sphere_center[1], y0 - half_width, y0 + half_width),
            np.clip(sphere_center[2], z0 - half_height, z0 + half_height)
        ])

        # Compute the distance between the closest point and the sphere's center
        distance = np.linalg.norm(sphere_center - closest_point)

        # Check if the distance is less than or equal to the sphere's radius
        return distance <= sphere_radius, distance

class Cube:
    """ This class defines a cube object in the blocks task environment. It is defined as a 3D object with a finite
        length, width, and height. Its location is defined by a user-defined xyz origin point [0,0,0] and quaternion
        orientation [0, 0, 0 ,1] by default.

        Parameters:
        ------------
        name: str
            The name of the cube object.
        length: float
            The length of the cube.
        width: float
            The width of the cube.
        height: float
            The height of the cube.
        pose: list
            The pose of the cube in the environment. It is a list of 7 elements [x, y, z, qx, qy, qz, qw].
            The first 3 elements represent the position of the cube in 3D space, and the last 4 elements represent
            the orientation of the cube in quaternion form.
    """
    def __init__(self, name='DefaultCube', length=1, width=1, height=1, pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], logger=None):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        self.stiffness = 1000
        self.pose = pose  # Can be used to store the pose of the cube in the environment
        self.logger = logger
        self.is_colliding = False

    def cube_corners(self, position=None, orientation=None):
        """ This method returns the 8 corners of the cube in 3D space. Given the cube's length, width, height, and
            volumetric position and orientation, the 8 corners are extracted

            Parameters:
            ------------
            position: list (optional)
                The position of the cube in 3D space. It is a list of 3 elements [x, y, z].
            orientation: list (optional)
                The orientation of the cube in quaternion form. It is a list of 4 elements [qx, qy, qz, qw].

            Returns:
            ------------
            translated_corners: numpy array
                The 8 corners of the cube in 3D space. The corners are returned as a numpy array of shape (8, 3).

            Example:
                cube3D = parent_cube.get_cube_corners()
        """

        if position is None:
            position = self.pose[0:3]

        if orientation is None:
            orientation = self.pose[3:7]

        # Define the local coordinates of the corners relative to the center
        local_corners = np.array([
               # [-1, -1, -1],  # Corner 1
               # [-1, -1, 1],  # Corner 2
               # [-1, 1, -1],  # Corner 3
               # [-1, 1, 1],  # Corner 4
               # [1, -1, -1],  # Corner 5
               # [1, -1, 1],  # Corner 6
               # [1, 1, -1],  # Corner 7
               # [1, 1, 1]  # Corner 8
            [-1, -1, 1],  # Corner 1

        ])

        # Apply rotation to the local coordinates
        r = Rotation.from_quat(orientation)
        rotated_corners = r.apply(local_corners)

        # Scale the rotated corners by the size of the cube
        scaled_corners = rotated_corners * np.array([self.length, self.width, self.height])

        # Translate the scaled corners to the position of the cube's center
        translated_corners = scaled_corners + position

        return translated_corners

    def is_inside_cube(self, points):
        """
        Parameters:
        ------------
        points = array of points with shape (N, 3).
        cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.

        Returns:
        --------------
        the indices of the points array which are outside the cube and the distances to the cube.
        """
        cube3d = self.cube_corners()
        b1, b2, b3, b4, t1, t2, t3, t4 = cube3d
        self.logger("corners: {}".format(cube3d))

        dir1 = (t1 - b1)
        size1 = np.linalg.norm(dir1)
        dir1 = dir1 / size1

        dir2 = (b2 - b1)
        size2 = np.linalg.norm(dir2)
        dir2 = dir2 / size2

        dir3 = (b4 - b1)
        size3 = np.linalg.norm(dir3)
        dir3 = dir3 / size3

        cube3d_center = (b1 + t3) / 2.0
        dir_vec = points - cube3d_center

        dist1 = (np.absolute(np.dot(dir_vec, dir1)) * 2) - size1
        dist2 = (np.absolute(np.dot(dir_vec, dir2)) * 2) - size2
        dist3 = (np.absolute(np.dot(dir_vec, dir3)) * 2) - size3

        res1 = np.where(dist1 > 0, dist1, 0)
        res2 = np.where(dist2 > 0, dist2, 0)
        res3 = np.where(dist3 > 0, dist3, 0)

        distances = np.stack((res1, res2, res3), axis=0)
        distances2 = [dist1, dist2, dist3]
        indices = np.where(np.sum(distances, axis=0) < 0)[0]

        is_inside = False
        #if all(distances2==0):
        #    is_inside = True

        return is_inside, distances2