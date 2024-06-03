#!/usr/bin/env python

# Import ROS2 libraries
import rclpy
import rosidl_runtime_py
from rclpy.node import Node
from rclpy.qos import qos_profile_default
import rosidl_runtime_py
#from rclpy_message_converter import message_converter

# Importing ROS2 message types and services
from geometry_msgs.msg import Pose, Point, Twist, Wrench
from haptic_device_interfaces.msg import Sigma7
from ros2_utilities_py import ros2serviceclient as r2sc

import math, sys, select, termios, tty, threading, yaml, datetime
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

settings = termios.tcgetattr(sys.stdin)

main_msg = """

Collects the range of motion of the user's wrist and approximating the rotational axes.
---------------------------
The steps are as follows:
  1. With the user's wrist in a neutral position, determine the position of the handle that is most comfortable to the user for grip and save the position as front
  2. Perform a wrist flexion rotation at 90 degrees and record the position and orientation of the handle, saving the flex-plane. Determine where the flex-plane and x-z-front plane intersect, and save this position as the flexion/extension axis.
  3. Perform a wrist radial deviation rotation at 90 degrees and record the position and orientation of the handle, saving the radial-plane. Determine where the radial-plane and x-y-front plane intersect, and save this position as the radial/ulnar deviation axis.
  4. Apply the "wrist_manipulandum_restriction" function to restrict robot movement to the wrist manipulandum workspace using the calculated axes of rotation.
  5. Ask the user if the axes are correct and feel okay to use. If not, repeat the process.

CTRL-C to quit

Are you ready? (y/n)
"""

begin_msg = """
Are you ready to begin? (y/n)
"""

msg_1 = """
Please relax your wrist and grab the handle. Press 'y' to continue.
"""

msg_2 = """
Please rotate your wrist in a flexion/extension movement. Press 'y' to continue.
"""

msg_3 = """
Please rotate your wrist in a radial/ulnar deviation movement. Press 'y' to continue.
"""
range_msg = """
Move your wrist to set the max and min angle values"""

confirm_msg = """
Is this position okay? (y/n)\n
"""
msg_4 = """
Applying wrist manipulandum restriction...
"""

axis_confirm_msg = """
Are the axes correct? (y/n)
"""

msg_5 = """
All done! Press 'y' to save and exit.
"""

flex_adjust_msg = """
Adjust the position of the rotation axis.
-----------------------------------------
Default increment: 0.005m
Movement:
           W| (+x: ↑)
  A|(-y: ←)          D| (+y: →)
           S| (-x: ↓)  
-----------------------------------------
Press 'y' to confirm\n
"""
flexMoveBindings = {
    'w': (1.0, 0.0, 0.0),
    'a': (0.0, -1.0, 0.0),
    's': (-1.0, 0.0, 0.0),
    'd': (0.0, 1.0, 0.0),
}

radial_adjust_msg = """
Adjust the position of the rotation axis..
-----------------------------------------
Default increment: 0.005m
Movement:
           W| (+z: ↑)
  A|(-x: ←)          D| (+x: →)
           S| (-z: ↓)
-----------------------------------------
Press 'y' to confirm\n
"""
radialMoveBindings = {
    'w': (0.0, 0.0, 1.0),
    'a': (-1.0, 0.0, 0.0),
    's': (0.0, 0.0, -1.0),
    'd': (1.0, 0.0, 0.0),
}
moveStep = 0.005

# Function to calculate the rotation matrix from a quaternion using scipy
def quaternion_to_rotation_matrix(quaternion):
    # Convert the quaternion to a scipy Rotation object
    rotation = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    # Get the rotation matrix from the Rotation object
    return rotation.as_matrix()

# Function to calculate the quaternion from Euler angles
def euler_to_quaternion(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    q[0] = cj * sc - sj * cs
    q[1] = cj * ss + sj * cc
    q[2] = cj * cs - sj * sc
    q[3] = cj * cc + sj * ss

    return q

# Function to calculate the Euler angles from a quaternion
# https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
def quaternion_to_euler(x, y, z, w):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2 > +1.0, +1.0, t2)
    # t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2 < -1.0, -1.0, t2)
    # t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z

# Function to calculate the normal vector from a rotation matrix
def calculate_normal_vector(rotation_matrix):
    # Extract the third column of the rotation matrix as the normal vector
    return rotation_matrix[:, 2]

# Function to find the line of intersection given two normal vectors and positions
def find_line_of_intersection(normal1, pos1, normal2, pos2):
    # Calculate the cross product of the normal vectors to get the direction of the line of intersection
    line_direction = np.cross(normal1, normal2)

    # Set up and solve the linear equation system to find a point on the line of intersection
    A = np.vstack((normal1, normal2, line_direction))
    B = np.array([np.dot(normal1, pos1), np.dot(normal2, pos2), 0])
    point_of_intersection = np.linalg.solve(A, B)

    return point_of_intersection, line_direction

class KeyboardThread(threading.Thread):

    def __init__(self, input_cbk = None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        self.last_key = None
        self.stop = False
        super(KeyboardThread, self).__init__(name=name)
        self.start()

    def run(self):
        while self.stop == False:
            self.input_cbk(input()) #waits to get input + Return

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class ROM(Node):
    def __init__(self, namespace='robot'):
        """
        Initialize the ROM class.
        """
        super().__init__('rom_node', namespace=namespace)
        self.__version__ = '0.1.0'
        self.initialize_parameters()
        self.initialize_publishers()
        self.initialize_subscribers()

        # Start the keyboard thread
        self.keyboard_thread = KeyboardThread(input_cbk=self.keyboard_callback)
        self.last_key = None

        # Create a service client connection to the force_dimension node
        self.force_dimension_client = r2sc.ROS2ServiceClient(parent_node=self,
                                                             external_node_name='robot/sigma7',
                                                             tag='sigma7_client')

        # Also want to change this node's parameters
        self.rom_node_client = r2sc.ROS2ServiceClient(self, 'robot/rom_node', 'rom_node_client')

        self.initialize_forces()
        self.f_e = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Handles for the different poses and data
        self.axis_type = 'front'
        self.pose_data = {
            'front': {'pose': Pose(), 'pos_data': [], 'center': [0.0, 0.0, 0.0], 'radius': 0.0},
            'flexion': {'pose': Pose(), 'pos_data': [], 'center': [0.0, 0.0, 0.0], 'radius': 0.0, 'min_angle': 0.0, 'max_angle': 0.0},
            'radial': {'pose': Pose(), 'pos_data': [], 'center': [0.0, 0.0, 0.0], 'radius': 0.0, 'min_angle': 0.0, 'max_angle': 0.0},}

        # Handles for the current robot pose and twist
        self.robot_pose = None
        self.robot_vel = None

        # Flags
        self.state = 'user_input'
        self.user_input = False
        self.new_msg = True

        # Timers
        self.position_display_timer = None
        self.angular_range_display_timer = None
        self.force_timer = None
        self.pos_timer = None
        self.begin_timer = self.create_timer(0.1, self.begin)
        self.logger("begin!")

        # TO-DO: automatic scaling of appropriate "effector_mass_kg" param value
        #self.initialize_forces() # For now, using the default of 1.45 kg
        #self.force_dimension_client.set_parameter('enable_force', True)
        #self.force_dimension_client.set_parameter('effector_mass_kg', 1.45)
        #self.force_dimension_client.set_parameter('gravity_compensation', True)

    def logger(self, msg):
        self.get_logger().info(msg)

    def disp_msg(self, msg, new_msg=False):
        """Prints the message only once. The 'new_msg' parameter needs to be set to True to enable print"""
        if self.new_msg or new_msg:
            print(msg)
            self.new_msg = False

    def initialize_parameters(self):

        self.declare_parameters(
            namespace='',
            parameters=[
                ('sample_rate', 0.001),
                #('transform', default_transform),
                ('sigma7_force_command_topic', '/robot/command/sigma7_force'),
                ('enable_position_restriction', True),
                ('enable_orientation_restriction', True),
                ('enable_manipulandum_restriction', False),
                ('enable_circle_restriction', False),
                ('stiffness', 2000.0),
                ('damping', 1.0),
                ('offset', [0.0, 0.0, 0.0]),
                ('roll.kp',      0.01),
                ('pitch.kp',     0.01),
                ('yaw.kp',       0.01),
                ('roll.kd',      0.001),
                ('pitch.kd',     0.001),
                ('yaw.kd',       0.001),
                ('effector_mass_kg', 1.45),
                ('attractor_basis', [+1.0, +0.0, +0.0, # Just default values, it changes later
                                               +0.0, +1.0, +0.0,
                                               +0.0, +0.0, +1.0]),
            ],
        )

    def initialize_publishers(self):
        #self.origin_pub = self.create_publisher(Point, '/robot/origin', qos_profile=qos_profile_default)

        self.sigma7_force_pub = self.create_publisher(Sigma7, self.get_parameter('sigma7_force_command_topic').value,
                                                        qos_profile=qos_profile_default)

    def initialize_subscribers(self):
        self.robot_pose_sub = self.create_subscription(Pose, '/robot/feedback/pose', self.pose_callback,
                                                 qos_profile=qos_profile_default)
        self.robot_vel_sub = self.create_subscription(Twist, '/robot/feedback/twist', self.velocity_callback,
                                                    qos_profile=qos_profile_default)

    def initialize_forces(self):
        """ Initialize the force dimension client to enable forces.
        """
        self.force_dimension_client.set_parameter('enable_force', False)
        self.basis = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] # Enable the whole space
        self.send_sigma7_force([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Should default of zero forces
        self.f_e = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.force_dimension_client.set_parameter('enable_force', True)
        self.send_sigma7_force([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Should default of zero
        self.force_dimension_client.set_parameter('gravity_compensation', True)
        #self.force_initializer.timer = self.create_timer(0.2, self.init_force_fib_callback)

    def init_force_fib_callback(self):
        """ Update the table stiffness and other forces by slowly ramping up the force to the desired value."""
        # Fibonacci sequence style!
        if self.force_initializer.nth < self.force_initializer.max_stiffness:
                self.force_initializer.nth = self.force_initializer.n1 + self.force_initializer.n2
                self.force_initializer.nth = self.force_initializer.max_stiffness if self.force_initializer.nth > self.force_initializer.max_stiffness else self.force_initializer.nth
                self.logger(f'Sending stiffness parameter: {self.force_initializer.nth}')
                self.blocks_node_client.set_parameter('table.stiffness', self.force_initializer.nth)
                self.force_initializer.n1 = self.force_initializer.n2
                self.force_initializer.n2 = self.force_initializer.nth
        else:
            self.logger(f'Final stiffness parameter: {self.force_initializer.max_stiffness}')
            self.destroy_timer(self.force_initializer.timer)

    def keyboard_callback(self, key):
        self.last_key = key
        #print("Last key: ", self.last_key)

    def get_last_key(self):
        last_key = self.last_key
        self.last_key = None

        if last_key is not None:
            last_key.lower()
        return last_key

    def pose_callback(self, msg):
        self.robot_pose = msg

    def projected_radius_ellipsoid(self, direction, a, b, c):
        """
        Calculate the projected radius of the ellipsoid from the origin along a given direction.

        Args:
            direction (numpy array): A 1D array of shape (3,) representing the direction vector.
            a (float): Major axis radius of the ellipsoid (semi-major axis length along the x-axis).
            b (float): Minor axis radius of the ellipsoid (semi-minor axis length along the y-axis).
            c (float): Minor axis radius of the ellipsoid (semi-minor axis length along the z-axis).

        Returns:
            float: The projected radius of the ellipsoid from the origin along the given direction.
        """

        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Calculate the denominator (squared)
        # Scaling the direction vector with the ellipsoid axes
        scaled_direction_squared = (
                (direction[0] / a) ** 2 +
                (direction[1] / b) ** 2 +
                (direction[2] / c) ** 2
        )

        # Calculate the projected radius
        projected_radius = 1 / np.sqrt(scaled_direction_squared)

        return projected_radius

    def velocity_callback(self, msg):
        self.robot_vel = msg

    def position_display_timer_callback(self):
        """
        Simply shows the robot's current position
        """
        # delete the previous line and print the new one
        print("Press 'y' and 'enter' when satisfied: [{:2.3f} {:2.3f} {:2.3f}]".format(self.robot_pose.position.x, self.robot_pose.position.y, self.robot_pose.position.z), end="\r")

    def angular_range_display_timer_callback(self):
        """
        Display the minimum and maximum angles of the wrist flexion and radial deviation
        """
        data = self.pose_data[self.axis_type]
        # delete the previous line and print the new one
        print("{}: [{:2.3f} {:2.3f}]".format(self.axis_type,data['min_angle'], data['max_angle']), end="\r")

    def save_pos_timer_callback(self):
        """Save the robot current position and append it to the list of the flexion positions
        """
        # Extract the x, y, z data and append it to the list of flexion positions
        data = [self.robot_pose.position.x, self.robot_pose.position.y, self.robot_pose.position.z]
        self.pose_data[self.axis_type]['pos_data'].append(data)

    def sphere_perimeter_restriction(self):
        """ Restrict robot movement to a 3D spherical perimiter boundary along the surface of the sphere

        The radius of the sphere is 0.03 meters with the center at origin (0, 0, 0).

        """
        if True:

            sphere_radius = self.pose_data[self.axis_type]['radius']
            origin = self.pose_data[self.axis_type]['center']
            # Calculate the distance between the robot current position and origin
            #origin = [0.0, 0.0, 0.0] # Is stable
            #origin = [self.pose_data['flexion']['center'][0],
            #          self.pose_data['origin']['center'][1],
            #          self.pose_data['origin']['center'][2]
            pose = self.robot_pose
            dx = (pose.position.x - origin[0])
            dy = (pose.position.y - origin[1])
            dz = (pose.position.z - origin[2])
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # If the distance is greater than the radius of the sphere, then calculate the desired position of the robot
            # as the point on the sphere that is closest to the robot current position.
            #sphere_radius = 0.06
            desired_position = Pose()
            # Calculate the desired position of the robot
            if distance > sphere_radius or distance < sphere_radius:
                desired_position.position.x = sphere_radius * pose.position.x / distance
                desired_position.position.y = sphere_radius * pose.position.y / distance
                desired_position.position.z = sphere_radius * pose.position.z / distance
            else:
                desired_position.position.x = pose.position.x
                desired_position.position.y = pose.position.y
                desired_position.position.z = pose.position.z

            desired_position.orientation.x = pose.orientation.x
            desired_position.orientation.y = pose.orientation.y
            desired_position.orientation.z = pose.orientation.z
            desired_position.orientation.w = pose.orientation.w

            return desired_position

    def sphere_perimeter_restriction_old(self):
        """ Restrict robot movement to a 3D spherical perimiter boundary along the surface of the sphere
        """
        if self.axis_type == 'flexion':
            sphere_radius = self.pose_data['flexion']['radius']
            origin = self.pose_data['flexion']['center']
        elif self.axis_type == 'radial':
            sphere_radius = self.pose_data['radial']['radius']
            origin = self.pose_data['radial']['center']
        else:
            sphere_radius = self.pose_data['origin']['radius']
            origin = self.pose_data['origin']['center']

        if True:
            print("sphere center: [{:2.3f} {:2.3f} {:2.3f}]".format(origin[0], origin[1], origin[2]), end="\r")
            # Calculate the distance between the robot current position and origin
            pose = self.robot_pose
            dx = (pose.position.x - origin[0])
            dy = (pose.position.y - origin[1])
            #dz = (pose.position.z - origin[2])
            dz = pose.position.z - self.pose_data['front']['pose'].position.z
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            #print("dz: {:2.5f}".format(dz), end="\r")

            # If the distance is greater than the radius of the sphere, then calculate the desired position of the robot
            # as the point on the sphere that is closest to the robot current position.

            if distance > sphere_radius or distance < sphere_radius:
                # Calculate the desired position of the robot
                desired_position = Pose()
                desired_position.position.x = sphere_radius * pose.position.x / distance
                desired_position.position.y = sphere_radius * pose.position.y / distance
                desired_position.position.z = sphere_radius * pose.position.z / distance
            else:
                # Calculate the desired position of the robot
                desired_position = pose_msg()
                desired_position.position.x = pose.position.x
                desired_position.position.y = pose.position.y
                desired_position.position.z = pose.position.z

            desired_position.orientation.x = pose.orientation.x
            desired_position.orientation.y = pose.orientation.y
            desired_position.orientation.z = pose.orientation.z
            desired_position.orientation.w = pose.orientation.w

            return desired_position

    def calculate_resisting_force(self, current_pose, current_velocity, desired_pose):
        """
        Calculate the resisting force and torque to restrict motion in undesired directions.

        Parameters:
        -----------
        current_position     (geometry_msgs.msg.Pose) : The current pose of the robot/wrist.
        current_velocity    (geometry_msgs.msg.Twist) : The current velocity of the robot/wrist.
        desired_pose         (geometry_msgs.msg.Pose) : The desired pose of the robot/wrist.

        Return:
        ----------
        force_msg (geometry_msgs.msg.Wrench) : The resisting force and torque to be applied.
        """
        force_msg = Wrench()
        force_msg.force.x = 0.0
        force_msg.force.y = 0.0
        force_msg.force.z = 0.0
        force_msg.torque.x = 0.0
        force_msg.torque.y = 0.0
        force_msg.torque.z = 0.0

        if current_pose is None or current_velocity is None or desired_pose is None:
            return force_msg

        if self.get_parameter('enable_position_restriction').value:
            # Calculate the difference between the robot's current position and desired position
            dp_x = desired_pose.position.x - current_pose.position.x
            dp_y = desired_pose.position.y - current_pose.position.y
            dp_z = desired_pose.position.z - current_pose.position.z

            # Calculate the force needed to keep x, y, z positions to zero. Use the spring damping model.
            # F = -k * dx - b * dx/dt
            v = [current_velocity.linear.x, current_velocity.linear.y, current_velocity.linear.z]
            k = self.get_parameter('stiffness').value
            d = self.get_parameter('damping').value
            force_msg.force.x = k * dp_x - d * v[0]
            force_msg.force.y = k * dp_y - d * v[1]
            force_msg.force.z = k * dp_z - d * v[2]

        if self.get_parameter('enable_orientation_restriction').value:
            # Calculate the difference between the robot's current orientation and desired orientation
            (current_roll, current_pitch, current_yaw) = quaternion_to_euler(current_pose.orientation.x,
                                                                             current_pose.orientation.y,
                                                                             current_pose.orientation.z,
                                                                             current_pose.orientation.w)
            (desired_roll, desired_pitch, desired_yaw) = quaternion_to_euler(desired_pose.orientation.x,
                                                                             desired_pose.orientation.y,
                                                                             desired_pose.orientation.z,
                                                                             desired_pose.orientation.w)

            #self.logger("desired roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(desired_roll, desired_pitch, desired_yaw))
            #self.logger("current roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(current_roll, current_pitch, current_yaw))

            dr = desired_roll - current_roll
            dp = desired_pitch - current_pitch
            dy = desired_yaw - current_yaw

            #self.logger("roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(dr, dp, dy))


            # Calculate the torque needed to keep roll, pitch, yaw to zero. Use PD control.
            dtheta = [current_velocity.angular.x, current_velocity.angular.y, current_velocity.angular.z]
            force_msg.torque.x = (self.get_parameter('roll.kp').value * dr - self.get_parameter(
                'roll.kd').value * dtheta[0])
            force_msg.torque.y = (self.get_parameter('pitch.kp').value * dp - self.get_parameter(
                'pitch.kd').value * dtheta[1])
            force_msg.torque.z = (self.get_parameter('yaw.kp').value * dy - self.get_parameter(
                'yaw.kd').value * dtheta[2])

        return force_msg

    def update_force_callback(self):
        """ Assigns a new basis for the robot's workspace to restrict movement in the x-y axis and displays the robot's current position
        """
        if self.get_parameter('enable_circle_restriction').value:
            desired_pose = self.sphere_perimeter_restriction()
            f_e = self.calculate_resisting_force(self.robot_pose, self.robot_vel, desired_pose)
            self.f_e = [f_e.force.x, f_e.force.y, f_e.force.z,
                        f_e.torque.x, f_e.torque.y, f_e.torque.z, 0.0]
        elif self.get_parameter('enable_manipulandum_restriction').value:
            desired_pose = self.wrist_manipulandum_restriction()
            f_e = self.calculate_resisting_force(self.robot_pose, self.robot_vel, desired_pose)
            self.f_e = [f_e.force.x, f_e.force.y, f_e.force.z,
                        f_e.torque.x, f_e.torque.y, f_e.torque.z, 0.0]
        basis_force = self.compute_effector_force()
        new_force = [self.f_e[0] + basis_force[0], self.f_e[1] + basis_force[1], self.f_e[2] + basis_force[2], self.f_e[3], self.f_e[4], self.f_e[5], self.f_e[6]]
        self.send_sigma7_force(new_force)

    def find_intersection(self, pose1, pose2):
        # Calculate the rotation matrices from the quaternions
        q1 = [pose1.orientation.x, pose1.orientation.y, pose1.orientation.z, pose1.orientation.w]
        q2 = [pose2.orientation.x, pose2.orientation.y, pose2.orientation.z, pose2.orientation.w]
        pose1_rot_matrix = quaternion_to_rotation_matrix(q1)
        pose2_rot_matrix = quaternion_to_rotation_matrix(q2)

        # Calculate the normal vectors from the rotation matrices
        normal1 = calculate_normal_vector(pose1_rot_matrix)
        normal2 = calculate_normal_vector(pose2_rot_matrix)

        # Convert the positions from the Pose messages to NumPy arrays
        pos1 = np.array([pose1.position.x, pose1.position.y, pose1.position.z])
        pos2 = np.array([pose2.position.x, pose2.position.y, pose2.position.z])

        # Find the line of intersection and a point on it
        point_of_intersection, line_direction = find_line_of_intersection(normal1, pos1, normal2, pos2)

        print(f"Point of intersection: {point_of_intersection}")
        print(f"Direction of line of intersection: {line_direction}")

        return point_of_intersection, line_direction

    def fit_circle_2d(self, points):
        """ Fit a circle to a set of 2D points using least squares optimization
        """
        # Define the function to optimize
        def circle_residuals(params, points):
            # Unpack the parameters
            x0, y0, r = params
            # Calculate the residuals
            residuals = np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2) - r
            return residuals

        # Initial guess for the circle parameters
        x0, y0 = np.array([0.1, 0.1])
        #x0, y0 = np.mean(points, axis=0)
        #x0, y0 = np.array([self.pose_data['origin']['pose'].position.x, self.pose_data['origin']['pose'].position.y])
        r = np.mean(np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2))

        # Optimize the circle parameters
        optimized_params = least_squares(circle_residuals, [x0, y0, r], args=(points,))
        center = optimized_params.x[:2]
        radius = optimized_params.x[2]

        return list(center), radius

    def calculate_wrist_flexion_angle(self):
        """ Calculate the wrist flexion angle from the wrist flexion plane and the x-z-front plane. Returns the adjusted angle in degrees.
        """
        pose = self.robot_pose
        ORIGIN = self.pose_data['flexion']['center']
        dx = (pose.position.x - ORIGIN[0])
        dy = (pose.position.y - ORIGIN[1])
        cur_yaw = math.atan2(dy, dx)
        # -pi transitions to pi,  Reorient the yaw angle so 0 is at the front of the sphere.
        sign = 1 if cur_yaw < 0 else -1
        cur_yaw = sign * (math.pi - abs(cur_yaw))
        return math.degrees(cur_yaw)

    def calculate_wrist_radial_angle(self):
        """ Calculate the wrist radial deviation angle from the wrist radial plane and the x-y-front plane. Returns the adjusted angle in degrees.
        """
        pose = self.robot_pose
        dx = (pose.position.x - self.pose_data['radial']['center'][0])
        dy = (pose.position.y - self.pose_data['radial']['center'][1])
        dz = (pose.position.z - self.pose_data['radial']['center'][2])
        # Calculate the pitch of the handle, and factor in the orientation of the wrist with respect to the handle. This makes it super convenient to get thr rotation at any point instead of just the origin.
        z1 = self.pose_data['radial']['center'][2]
        x1 = self.pose_data['radial']['center'][0]
        wrist_offset = math.atan2(z1, x1)
        cur_pitch = -(math.atan2(-dz, math.sqrt(dx ** 2 + dy ** 2)) + wrist_offset)
        return math.degrees(cur_pitch)

    def compute_effector_force(self):
        """
        "modeled as a spring+damper_ system that pull the device towards its
         projection on the constraint segment"
        K_ is the spring constant, or stiffness_.
        damper_
        C is the damping coefficient
        damping_

        .. _K: https://en.wikipedia.org/wiki/Spring_(device)#Physics
        .. _stiffness: https://en.wikipedia.org/wiki/Stiffness
        .. _spring+damper: https://en.wikipedia.org/wiki/Mass-spring-damper_model
        .. _damper: https://en.wikipedia.org/wiki/Dashpot
        .. _damping: https://en.wikipedia.org/wiki/Damping
        """

        # Convert the cursor force to robot / effector force.
        # This is the force to be applied in the absence of an attractor
        # guiding force.
        #f_e = super().compute_effector_force(cursor_force)

        # Initialize a utility function for getting parameters.
        get_parameter = lambda n: self.get_parameter(n).value

        # Initialize a utility function for converting messages to numpy arrays.
        to_odict = rosidl_runtime_py.convert.message_to_ordereddict
        msg_to_array = lambda m: np.array(tuple(to_odict(m).values()))
        to_column = lambda s: np.array(s)[:][np.newaxis].T

        # Initialize local variables.
        K = get_parameter('stiffness')
        C = get_parameter('damping')
        o = to_column(get_parameter('offset'))
        current_pos = [self.robot_pose.position.x, self.robot_pose.position.y, self.robot_pose.position.z]
        current_vel = [self.robot_vel.linear.x, self.robot_vel.linear.y, self.robot_vel.linear.z]

        p = to_column(msg_to_array(self.robot_pose.position))
        v = to_column(msg_to_array(self.robot_vel.linear))

        #p = to_column(msg_to_array(current_pos))
        #v = to_column(msg_to_array(current_vel))

        # Create a projection matrix from the attractor basis parameter.
        b = self.basis
        B = np.array(b).reshape((3, 3))
        P_a = B @ np.linalg.pinv(B)

        # Compute the projection of the position vector onto the attractor
        # surface -- that is, the permissible plane in which the effector
        # should be constrained to move -- after subtracting the offset.
        p_a = P_a @ (p - o) + o

        # Compute the difference vector between the position vector and the
        # offset plane.
        p_d = p_a - p

        # Compute the "guidance force" from the spring + damper model.
        # Project the current position onto the attractor surface (i.e., a line
        # or plane).
        f_g = K * (p_a - p) - C * v

        # Project the guidance force onto the difference vector; that is, the
        # tangent to the permissible plane.
        # This is recommended by the Force Dimension example, and seems
        # necessary due to the velocity term in the spring + damper model,
        # which can introduce viscous force components on the permissible
        # attractor surface (i.e., the "free" plane).
        P_g = p_d @ np.linalg.pinv(p_d)
        f_g = P_g @ f_g

        # Compute the aggregate effector force by adding in the attractor force.
        # self.get_logger().info(f'{f_e + f_g.squeeze()} (f_e: {f_e}; f_g: {f_g})')
        #f_e = self.f_e[0:3] + f_g.squeeze()
        f_e = f_g.squeeze()

        if (len(f_e) != 3) or not isinstance(f_e[0], float): raise Exception()

        # Return the result as a tuple.
        return f_e

    def save_angles_to_yaml(self, file_path=None):
        """ Save the wrist flexion and radial deviation angles to a YAML file

         Parameters:
        -----------
        file_path      (str): The path to the YAML file where the angles will be saved.
        angles        (dict): A dictionary containing the angles to save.

        """
        print("Saving angles to YAML file...")
        # Create lambda function to convert the ROS message to a dictionary
        to_odict = rosidl_runtime_py.convert.message_to_ordereddict
        msg_to_array = lambda m: np.array(tuple(to_odict(m).values()))
        msg_to_values = lambda m: to_odict(m).values()


        flex_pos_data = dict(to_odict(self.pose_data['flexion']['pose'].position))
        flex_rot_data = dict(to_odict(self.pose_data['flexion']['pose'].orientation))
        flex_center = self.pose_data['flexion']['center']
        radial_pos_data = dict(to_odict(self.pose_data['radial']['pose'].position))
        radial_rot_data = dict(to_odict(self.pose_data['radial']['pose'].orientation))
        radial_center = self.pose_data['radial']['center']
        front_pos_data = dict(to_odict(self.pose_data['front']['pose'].position))
        front_rot_data = dict(to_odict(self.pose_data['front']['pose'].orientation))
        try:
            if file_path is None:
                # Get the current date and time formatted as a string
                formatted_time = datetime.datetime.now().strftime('%m%d%y_%H%M%S')
                file_path = f"data/wrist_angles_{formatted_time}.yaml"
                print("filename: ", file_path)

            data = [{'flexion': {'angle_min': self.pose_data['flexion']['min_angle'],
                                   'angle_max': self.pose_data['flexion']['max_angle'],
                                    'position': flex_pos_data,
                                 'orientation': flex_rot_data,
                                 'axis_center': {'x': flex_center[0], 'y': flex_center[1], 'z': flex_center[2]},
                                      'radius': self.pose_data['flexion']['radius']}},
                    {'radial': {'angle_min': self.pose_data['radial']['min_angle'],
                                   'angle_max': self.pose_data['radial']['max_angle'],
                                    'position': radial_pos_data,
                                 'orientation': radial_rot_data,
                                 'axis_center': {'x': radial_center[0], 'y': radial_center[1], 'z': radial_center[2]},
                                      'radius': self.pose_data['radial']['radius']}},
                    {'front': {'position': front_pos_data,
                                 'orientation': front_rot_data}}]


            with open(file_path, '+w') as f:
                yaml.dump(data, f)

            print(open(file_path).read())

        except Exception as e:
            self.logger(f"Error saving angles to YAML file: {e}")

    def send_sigma7_force(self, force):
        """ Sends a force command to the robot to keep it in place
        """
        # Create the sigma7 force message at the beginning
        sigma7_force = Sigma7()
        sigma7_force.force.x = force[0]
        sigma7_force.force.y = force[1]
        sigma7_force.force.z = force[2]
        sigma7_force.torque.x = force[3]
        sigma7_force.torque.y = force[4]
        sigma7_force.torque.z = force[5]

        # Publish the force message
        self.sigma7_force_pub.publish(sigma7_force)

    def wrist_manipulandum_restriction(self):
        """ Restrict the robot movement to a wrist manipulandum's workspace. The position is fixed to origin in space, and rotation is disabled, but pitch and yaw are allowed.

        Assumes that the points for the flexion and radial deviation axes have been calculated.

        """

        flex_r = self.pose_data['flexion']['radius']
        radial_r = self.pose_data['radial']['radius']

        # Calculate the distance between the robot current position and origin
        pose = self.robot_pose
        ORIGIN = self.pose_data['flexion']['center']
        dx = (pose.position.x - ORIGIN[0])
        dy = (pose.position.y - ORIGIN[1])
        dz = (pose.position.z - ORIGIN[2])

        # Calculate the euler angles for the desired orientation.
        # Desired orientation is the pitch and yaw euler angle components using the vector between the origin and the robot's current position on the sphere.
        cur_roll = 0.00 # No roll allowed

        # Calculate the pitch of the handle, and factor in the orientation of the wrist with respect to the handle
        z1 = self.pose_data['radial']['center'][2]
        x1 = self.pose_data['radial']['center'][0]
        wrist_offset = math.atan2(z1, x1)
        cur_pitch = -(math.atan2(-dz, math.sqrt(dx ** 2 + dy ** 2)) + wrist_offset)
        cur_yaw = math.atan2(dy, dx)
        # -pi transitions to pi,  Reorient the yaw angle so 0 is at the front of the sphere.
        sign = 1 if cur_yaw < 0 else -1
        cur_yaw = sign * (math.pi - abs(cur_yaw))

        #self.logger("orientation: roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(cur_roll, cur_pitch, cur_yaw))

        # Convert the pitch and yaw angles to quaternions and add to pose message
        desired_pose = Pose()
        desired_pose.orientation.x, desired_pose.orientation.y, desired_pose.orientation.z, desired_pose.orientation.w = euler_to_quaternion(cur_roll, cur_pitch, cur_yaw)

        # Calculate the position of the robot
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # If the distance is greater or less than the radius of the sphere, then find the point on the sphere that is closest to the robot current position.
        x_r = math.sqrt(flex_r**2 + radial_r**2)
        direction_vector = np.array([dx, dy, dz])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Projected radius given ellipsoid (almost done, need to choose the best radius conbination since x-y-z coordinate frame feels different from the robot coordinate frame)
        #sphere_radius = self.projected_radius_ellipsoid(direction_vector, flex_r, radial_r, x_r) # super close!
        #sphere_radius = self.projected_radius_ellipsoid(direction_vector, x_r, radial_r, flex_r) # too far out
        #sphere_radius = self.projected_radius_ellipsoid(direction_vector, flex_r, x_r, radial_r) # much better, possibly this one
        sphere_radius = flex_r # for simple spherical boundary
        if distance > sphere_radius or distance < sphere_radius:
            temp = [sphere_radius * pose.position.x / distance,
                    sphere_radius * pose.position.y / distance,
                    sphere_radius * pose.position.z / distance]
        else:
            temp = [pose.position.x, pose.position.y, pose.position.z]

        desired_pose.position.x = temp[0]
        desired_pose.position.y = temp[1]
        desired_pose.position.z = temp[2]

        return desired_pose
    def begin(self):
        """ State machine for the range of motion data collection  """
        try:
            if self.state == 'user_input':
                """The 'user_input' state is the initial state where the user is asked if they are ready to begin data collection."""
                sr = self.get_parameter('sample_rate').value
                if self.force_timer is None:
                    self.force_timer = self.create_timer(sr, self.update_force_callback)
                self.disp_msg(main_msg)
                if not self.user_input:
                    #key = getKey()
                    key = self.get_last_key()
                    if key == 'y':
                        self.disp_msg("Starting data collection...",new_msg=True)
                        self.user_input = True
                        if self.position_display_timer is None:
                            self.position_display_timer = self.create_timer(self.get_parameter('sample_rate').value, self.position_display_timer_callback)
                        self.state = 'collect_front'
                    elif key == 'n':
                        print("Exiting...")
                        self.user_input=True
                        self.state = 'exit'
                    else:
                        pass

            if self.state=='collect_front':
                """The 'collect_origin' state asks the user to press 'y' to save the origin position. The robot's position is constantly updated on-screen"""
                self.disp_msg(msg_1)
                last_key = self.get_last_key()
                if last_key == 'y':
                    self.destroy_timer(self.position_display_timer)
                    self.position_display_timer = None
                    p = self.robot_pose
                    self.pose_data['front']['pose'] = p
                    print('\nFront chosen at : [{:2.3f} {:2.3f} {:2.3f}]'.format(p.position.x, p.position.y, p.position.z))
                    # Update the basis 'offset' parameter with the current position
                    self.rom_node_client.set_parameter('offset', [p.position.x, p.position.y, p.position.z])
                    self.state = 'start_flexion_timer'

            # ----------------- Flexion Axis Collection -----------------
            if self.state=='start_flexion_timer':
                """ The 'start_flexion_timer' state begins the position display timer, restricts movement to the 
                    flexion/extension plane, and begins collecting robot position data
                """
                sr = self.get_parameter('sample_rate').value
                if self.position_display_timer is None:
                    self.position_display_timer = self.create_timer(sr, self.position_display_timer_callback)
                if self.pos_timer is None:
                    self.axis_type = 'flexion'
                    self.pos_timer = self.create_timer(0.02, self.save_pos_timer_callback)
                if self.force_timer is None:
                    self.force_timer = self.create_timer(sr, self.update_force_callback)
                self.basis = [+1.0, +0.0, +0.0, # x
                              +0.0, +1.0, +0.0, # y
                              +0.0, +0.0, +0.0]
                self.new_msg = True
                self.state = 'collect_flexion_data'

            if self.state=='collect_flexion_data':
                """ The 'collect_flexion_axis' state asks the user to rotate their wrist to a 90 degree flexion position. The robot's position is constantly updated on-screen"""
                self.disp_msg(msg_2)
                last_key = self.get_last_key()
                if last_key == 'y':
                    self.destroy_timer(self.pos_timer)
                    self.pos_timer = None
                    self.destroy_timer(self.position_display_timer)
                    self.position_display_timer = None
                    # We now have a list of [x, y, z] robot position values in a curve which we can fit to a circle in the 2D x-y plane
                    flex_pos_points = np.array(self.pose_data['flexion']['pos_data'])
                    [p, r] = self.fit_circle_2d(flex_pos_points[:,:2])
                    self.logger("flexion circle data: ")
                    self.pose_data['flexion']['center'] = [float(p[0]), float(p[1]), self.pose_data['front']['pose'].position.z] # Keep the z value the same as the front position
                    self.pose_data['flexion']['radius'] = float(r)

                    # Enable movement restrictions to the circle boundary
                    self.rom_node_client.set_parameter('enable_circle_restriction', True)
                    print("\nFlexion rotation center: {} radius: {:2.3f}m".format([round(i,3) for i in self.pose_data['flexion']['center']], round(r,3)))
                    self.state="test_flexion_axis"
                    self.new_msg = True

            if self.state=='test_flexion_axis':
                self.disp_msg(confirm_msg)
                last_key = self.get_last_key()
                if last_key == 'y':
                    # If the user likes it we can polish the position of the flexion axis
                    self.new_msg = True
                    self.state = 'adjust_flexion_axis'
                elif last_key == 'n':
                    print('n pressed. Restarting flexion axis collection...')
                    self.rom_node_client.set_parameter('enable_circle_restriction', False)
                    self.destroy_timer(self.angular_range_display_timer)
                    self.angular_range_display_timer = None
                    self.initialize_forces()
                    self.pose_data['flexion']['pos_data'] = []
                    self.pose_data['flexion']['max_angle'] = 0.0
                    self.pose_data['flexion']['min_angle'] = 0.0
                    self.state = 'start_flexion_timer'

            if self.state=='adjust_flexion_axis':
                """ The 'adjust_flexion_axis' state allows the user to input slight position adjustments to the center of the flexion_axis"""
                self.disp_msg(flex_adjust_msg)
                last_key = self.get_last_key()
                if last_key in flexMoveBindings.keys():
                    print(flexMoveBindings[last_key])
                    x = flexMoveBindings[last_key][0]
                    y = flexMoveBindings[last_key][1]
                    z = flexMoveBindings[last_key][2]
                    # Update the center position of the flexion axis
                    self.pose_data['flexion']['center'][0] += x * moveStep
                    self.pose_data['flexion']['center'][1] += y * moveStep
                    self.pose_data['flexion']['center'][2] += z * moveStep
                    print("\nFlexion rotation center: {} r".format([round(i,3) for i in self.pose_data['flexion']['center']]), end="\r")
                if last_key=='y':
                    # We no longer need to display the cartesian position of the robot handle, but we do want to save the min and max angles
                    if self.angular_range_display_timer is None:
                        self.angular_range_display_timer = self.create_timer(self.get_parameter('sample_rate').value, self.angular_range_display_timer_callback)
                    self.state = "get_flexion_range"
                    self.new_msg = True

            if self.state=='get_flexion_range':
                """ The 'get_flexion_range' state asks the user to rotate their wrist to a maximum flexion position. The robot's position is constantly updated on-screen"""
                self.disp_msg(range_msg)
                last_key = self.get_last_key()
                angle = self.calculate_wrist_flexion_angle()
                if angle < self.pose_data['flexion']['min_angle']:
                    self.pose_data['flexion']['min_angle'] = angle
                if angle > self.pose_data['flexion']['max_angle']:
                    self.pose_data['flexion']['max_angle'] = angle
                if last_key == 'y':
                    self.destroy_timer(self.angular_range_display_timer)
                    self.angular_range_display_timer = None
                    self.state = 'confirm_flexion_axis'
                elif last_key == 'n':
                    print('n pressed. Restarting flexion axis collection...')
                    self.rom_node_client.set_parameter('enable_circle_restriction', False)
                    self.destroy_timer(self.angular_range_display_timer)
                    self.angular_range_display_timer = None
                    self.initialize_forces()
                    self.pose_data['flexion']['pos_data'] = []
                    self.pose_data['flexion']['max_angle'] = 0.0
                    self.pose_data['flexion']['min_angle'] = 0.0
                    self.state = 'start_flexion_timer'

            if self.state=='confirm_flexion_axis':
                """ The 'confirm_flexion_axis' state asks the user if the flexion axis is correct and displays the center position, radius, and angle range. If the user is satisfied, the radial axis collection is started. Otherwise, the user is asked to repeat the process."""
                msg = "Flexion axis center: {} radius: {:2.3f}m\nAngle range: {:2.3f} to {:2.3f}\n".format(self.pose_data['flexion']['center'], self.pose_data['flexion']['radius'],self.pose_data['flexion']['min_angle'], self.pose_data['flexion']['max_angle']) + axis_confirm_msg
                self.disp_msg(msg)
                last_key = self.get_last_key()
                if last_key == 'y':
                    self.rom_node_client.set_parameter('enable_circle_restriction', False)
                    print('\nNext step: radial deviation axis collection...')
                    self.basis = [+1.0, +0.0, +0.0,  # x
                                  +0.0, +1.0, +0.0,  # y
                                  +0.0, +0.0, +1.0]
                    self.new_msg = True
                    self.state = 'prep_radial_axis'

            # ----------------- Radial Axis Collection -----------------
            if self.state=='prep_radial_axis':
                """ Asks the user to continue to the next step to collect the radial axis data"""

                self.disp_msg(begin_msg)
                last_key = self.get_last_key()
                if last_key == 'y':
                    self.state = 'start_radial_timer'
                elif last_key == 'n':
                    self.state = 'exit'

            if self.state=='start_radial_timer':
                """ The 'start_radial_timer' state begins the radial timer and initialize the movement boundary for the radial/ulnar deviation plane"""
                sr = self.get_parameter('sample_rate').value
                if self.position_display_timer is None:
                    self.position_display_timer = self.create_timer(sr, self.position_display_timer_callback)
                if self.pos_timer is None:
                    self.axis_type = 'radial'
                    self.pos_timer = self.create_timer(0.02, self.save_pos_timer_callback)
                if self.force_timer is None:
                    self.force_timer = self.create_timer(sr, self.update_force_callback)
                self.basis = [+1.0, +0.0, +0.0,
                              +0.0, +0.0, +0.0,
                              +0.0, +0.0, +1.0]
                self.new_msg = True
                self.state = 'collect_radial_data'

            if self.state=='collect_radial_data':
                """ The 'collect_radial_axis' state asks the user to rotate their wrist to a maximum radial deviation position. The robot's position is constantly updated on-screen"""
                self.disp_msg(msg_3)
                last_key = self.get_last_key()
                if last_key == 'y':
                    self.destroy_timer(self.pos_timer)
                    self.pos_timer = None

                    # Get the first and third columns of the radial position points
                    radial_pos_points = np.array(self.pose_data['radial']['pos_data'])
                    new_points = np.zeros((radial_pos_points.shape[0], 2))
                    new_points[:,0] = radial_pos_points[:,0]
                    new_points[:,1] = radial_pos_points[:,2]
                    [p, r] = self.fit_circle_2d(new_points)
                    print(r)
                    print(type(r))
                    self.pose_data['radial']['center'] = [float(p[0]), self.pose_data['front']['pose'].position.y, float(p[1])]
                    self.pose_data['radial']['radius'] = float(r)

                    # Implement the circle boundary for robot movement around the radial axis
                    self.rom_node_client.set_parameter('enable_circle_restriction', True)
                    print("\nRadial rotation center: {} radius: {:2.3f}m".format(p, r))
                    self.destroy_timer(self.position_display_timer)
                    self.position_display_timer = None
                    self.new_msg = True
                    self.state = 'test_radial_axis'

            if self.state=='test_radial_axis':
                """ The 'test_radial_axis' state asks the user if the radial axis is correct. If not, the user is asked to repeat the process"""
                self.disp_msg(axis_confirm_msg)
                last_key = self.get_last_key()
                if last_key == 'y':
                    # If the new user likes it we can polish the position of the radial axis
                    self.new_msg = True
                    self.state = 'adjust_radial_axis'
                elif last_key == 'n':
                    print('n pressed. Restarting flexion axis collection...')
                    self.rom_node_client.set_parameter('enable_circle_restriction', False)
                    self.destroy_timer(self.angular_range_display_timer)
                    self.angular_range_display_timer = None
                    self.initialize_forces()
                    self.pose_data['radial']['pos_data'] = []
                    self.pose_data['radial']['max_angle'] = 0.0
                    self.pose_data['radial']['min_angle'] = 0.0
                    self.state = 'start_radial_timer'

            if self.state=='adjust_radial_axis':
                """ The 'adjust_radial_axis' state allows the user to input slight position adjustments to the center of the radial axis"""
                self.disp_msg(radial_adjust_msg)
                last_key = self.get_last_key()
                if last_key in radialMoveBindings.keys():
                    print(flexMoveBindings[last_key])
                    x = radialMoveBindings[last_key][0]
                    y = radialMoveBindings[last_key][1]
                    z = radialMoveBindings[last_key][2]
                    # Update the center position of the flexion axis
                    self.pose_data['radial']['center'][0] += x * moveStep
                    self.pose_data['radial']['center'][1] += y * moveStep
                    self.pose_data['radial']['center'][2] += z * moveStep
                    print("\nRadial rotation center: {} r".format(
                        [round(i, 3) for i in self.pose_data['radial']['center']]), end="\r")
                if last_key == 'y':
                    # We no longer need to display the cartesian position of the robot handle, but we do want to save the min and max angles
                    if self.angular_range_display_timer is None:
                        self.angular_range_display_timer = self.create_timer(self.get_parameter('sample_rate').value,
                                                                             self.angular_range_display_timer_callback)
                    self.state = "get_radial_range"
                    self.new_msg = True

            if self.state=='get_radial_range':
                """ The 'get_radial_range' state asks the user to rotate their wrist to a maximum radial deviation position. The robot's position is constantly updated on-screen"""
                self.disp_msg(msg_3)
                last_key = self.get_last_key()
                angle = self.calculate_wrist_radial_angle()
                if angle < self.pose_data['radial']['min_angle']:
                    self.pose_data['radial']['min_angle'] = angle
                if angle > self.pose_data['radial']['max_angle']:
                    self.pose_data['radial']['max_angle'] = angle
                if last_key == 'y':
                    self.destroy_timer(self.angular_range_display_timer)
                    self.angular_range_display_timer = None
                    self.state = 'confirm_radial_axis'

            if self.state=='confirm_radial_axis':
                """ The 'confirm_radial_axis' state asks the user if the radial axis is correct and displays the center position, radius, and angle range. If the user is satisfied, the radial axis collection is complete. Otherwise, the user is asked to repeat the process."""
                msg = "Radial axis center: {} radius: {:2.3f}m\nAngle range: {:2.3f} to {:2.3f}\n".format(self.pose_data['radial']['center'], self.pose_data['radial']['radius'], self.pose_data['radial']['min_angle'], self.pose_data['radial']['max_angle']) + axis_confirm_msg
                self.disp_msg(msg)
                last_key = self.get_last_key()
                if last_key == 'y':
                    self.basis = [+1.0, +0.0, +0.0, # x
                                    +0.0, +1.0, +0.0, # y
                                    +0.0, +0.0, +1.0]
                    self.rom_node_client.set_parameter('enable_circle_restriction', False)
                    self.destroy_timer(self.angular_range_display_timer)
                    self.angular_range_display_timer = None
                    #self.initialize_forces()
                    self.basis = [+1.0, +0.0, +0.0,  # x
                                  +0.0, +1.0, +0.0,  # y
                                  +0.0, +0.0, +1.0]
                    print('\nNext step: applying wrist manipulandum restriction...')
                    self.new_msg = True
                    self.state = 'apply_restriction'
                elif last_key == 'n':
                    print('n pressed. Restarting radial axis collection...')
                    self.rom_node_client.set_parameter('enable_circle_restriction', False)
                    self.destroy_timer(self.angular_range_display_timer)
                    self.angular_range_display_timer = None
                    self.initialize_forces()
                    self.pose_data['radial']['pos_data'] = []
                    self.pose_data['radial']['max_angle'] = 0.0
                    self.pose_data['radial']['min_angle'] = 0.0
                    self.state = 'start_radial_timer'

            # ----------------- Apply Restriction -----------------
            if self.state=='apply_restriction':
                self.rom_node_client.set_parameter('enable_manipulandum_restriction', True)
                self.rom_node_client.set_parameter('enable_position_restriction', True)
                self.rom_node_client.set_parameter('enable_orientation_restriction', True)
                self.new_msg = True
                self.state = 'confirm_restriction'

            if self.state=='confirm_restriction':
                """ The 'apply_restriction' state restricts robot movement to the wrist manipulandum workspace using the calculated axes of rotation"""
                self.disp_msg(msg_4)
                last_key = self.get_last_key()
                if last_key == 'y':
                    self.basis = [+1.0, +0.0, +0.0,  # x
                                  +0.0, +1.0, +0.0,  # y
                                  +0.0, +0.0, +1.0]
                    self.rom_node_client.set_parameter('enable_circle_restriction', False)
                    self.rom_node_client.set_parameter('enable_manipulandum_restriction', False)
                    self.initialize_forces()
                    self.force_dimension_client.set_parameter('enable_forces', False)
                    self.new_msg = True
                    self.state = 'confirm_axes'
                elif last_key == 'n':
                    self.state = 'exit'

            if self.state=='confirm_axes':
                """ The 'confirm_axes' state asks the user if the axes are correct. If not, the user is asked to repeat the process"""
                self.disp_msg(axis_confirm_msg)
                last_key = self.get_last_key()
                if last_key == 'y':
                    # Display the angular limits of both flexion and radial deviation
                    print("Flexion max angle: {:2.3f} min angle: {:2.3f}".format(self.pose_data['flexion']['max_angle'], self.pose_data['flexion']['min_angle']))
                    print("Radial max angle: {:2.3f} min angle: {:2.3f}".format(self.pose_data['radial']['max_angle'], self.pose_data['radial']['min_angle']))

                    # Save angles to yaml file as 'wrist_angles.yaml'
                    print('Saving angles to YAML file...')
                    self.save_angles_to_yaml()
                    self.state = 'exit'

                elif last_key == 'n':
                    print('n pressed. Restarting axis collection...')
                    self.rom_node_client.set_parameter('enable_manipulandum_restriction', False)
                    self.new_msg = True
                    self.state = 'user_input'

            if self.state=='exit':
                print("exiting")
                self.destroy_node()
                self.keyboard_thread.stop = True
                self.destroy_timer(self.begin_timer)
                self.destroy_timer(self.position_display_timer)
                self.destroy_timer(self.force_timer)
                self.destroy_timer(self.pos_timer)
                rclpy.shutdown()

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            self.keyboard_thread.stop = True
            self.destroy_timer(self.begin_timer)
            self.destroy_timer(self.position_display_timer)
            self.destroy_timer(self.force_timer)
            self.destroy_timer(self.pos_timer)

def main(args=None):
    """

    """
    if args is None:
        args = sys.argv

    try:
        rclpy.init(args=args)
        node = ROM()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("KeyboardInterrupt2")
        pass

    finally:
        print("\nshutting down")
        node.destroy_node()
        #rclpy.shutdown()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':
    main()