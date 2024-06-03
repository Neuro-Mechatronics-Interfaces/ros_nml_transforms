""" This script implements a ROS2 node that computes the passive joint stiffness of the wrist constrained to a manipulandum configuration.

It first restricts the position of the robot to a spherical/wrist manip boundary restriction.
Then it implements an impedance controller to compute the stiffness of the wrist joint by asking the user to relax their adrm and wrist, and allow the robot to apply a Sigma7 command to move the robot hand a linear fashion to one of 24 different paths equally spaced in the wrist manip boundary.
The angular velocity of the wrist manip is kept at 0.02 rad/sec

"""

#!/usr/bin/env python

# Import ROS2 libraries
import rclpy
import rosidl_runtime_py
from rclpy.node import Node
from rclpy.qos import qos_profile_default
from rcl_interfaces.msg import SetParametersResult
#from rclpy_message_converter import message_converter

# Importing ROS2 message types and services
from geometry_msgs.msg import Pose, Point, Twist, Wrench
from example_interfaces.msg import Float64
from haptic_device_interfaces.msg import Sigma7
from ros2_utilities_py import ros2serviceclient as r2sc

import csv
import math
import numpy as np
import threading


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

class JointStiffness(Node):
    """ This class implements a ROS2 node that computes the passive joint stiffness of the wrist constrained to a
        manipulandum configuration.
    """
    def __init__(self, verbose=False):
        super().__init__('stiffness_node')
        self.initialize_parameters()
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.initialize_subscribers()
        self.initialize_publishers()
        self.verbose = verbose

        # Handles for the robot's current state
        self.current_pose = Pose()
        self.current_vel  = Twist()

        # Create timers for the main (init now) and force dimension callback functions
        self.main_timer = self.create_timer(self.get_parameter('sample_rate').value, self.main_callback)
        self.force_timer = None # cb name is 'update_force_callback'
        self.arc_timer = None  # cb name is 'sinusoidal_arc_callback'

        # Start the keyboard thread
        self.keyboard_thread = KeyboardThread(input_cbk=self.keyboard_callback)
        self.last_key = None
        self.new_msg = True

        # Create a service client connection to the force_dimension node
        self.force_dimension_client = r2sc.ROS2ServiceClient(parent_node=self,
                                                             external_node_name='robot/sigma7',
                                                             tag='sigma7_client')
        # Just some global variables
        self.n1 = 0
        self.n2 = 0
        self.nth = 1
        self.max_stiffness = self.get_parameter('basis.stiffness').value
        self.force_init_timer = None
        self.f_e = [0.0, 0.0, 0.0]
        self.projected_radius = 0.0
        self.desired_angle = 0.0

        self.node_client = r2sc.ROS2ServiceClient(parent_node=self,
                                               external_node_name='robot/stiffness_node',
                                                  tag='stiffness_client')

        self.counter = 0.0
        self.completed_sweeps = 0
        self.collect_data = False
        self.state = 'begin'

        # The transform data dictionary will hold the robot force, robot position, and estimated radius at each time
        # step, for each rotation angle. Position and force will be in [x, y, z] vector components.
        # Ex: {'angle': [0, 15, ..., 180],
        #      'force': [[0.1234, 0.4231, 0.4223], ..., [0.1234, 0.4233, 0.4223]]     # angle 0
        #                ...
        #               [[0.1234, 0.4233, 0.4223], ..., [0.1234, 0.4233, 0.4223]],    # angle 180
        #      'position': [[0.1234, 0.4231, 0.0000], ..., [0.1234, 0.4233, 0.0000]]  # angle 0
        #                   ...
        #                  [[0.1234, 0.4233, 0.0000], ..., [0.1234, 0.4233, 0.0000]], # angle 180
        #      'radius': [[0.1234, 0.4231, 0.4223], ..., [0.1234, 0.4233, 0.4223]]    # angle 0
        #                 ...
        #                [[0.1234, 0.4233, 0.4223], ..., [0.1234, 0.4233, 0.4223]],   # angle 180
        #      'manip': None}
        #      }
        self.transform_data = {}

        self.logger('Joint stiffness node has been started')

    def initialize_parameters(self):
        """ Define all parameters in this function.
        """
        default_transform = [+1.0, +0.0, +0.0,
                             +0.0, +1.0, +0.0,
                             +0.0, +0.0, +0.0]

        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_node_name', 'robot/manipulandum'),
                ('robot_pose_topic', '/robot/feedback/pose'),
                ('robot_twist_topic', '/robot/feedback/twist'),
                ('gripper_angle_topic', '/robot/feedback/gripper_angle'),
                ('sigma7_force_command_topic', '/robot/command/sigma7_force'),
                ('environment_boundary', 'sphere_rotation'),
                ('sample_rate', 0.001),

                # Robot physics parameters
                ('enable_position_restriction', False),
                ('enable_orientation_restriction', False),
                ('gripper_stiffness', 0.0000001),
                ('basis.stiffness',  1000.0),
                ('basis.damping',      10.0),
                ('basis.transform', default_transform),
                ('roll.kp',      0.01),
                ('pitch.kp',     0.01),
                ('yaw.kp',       0.01),
                ('roll.kd',      0.02),
                ('pitch.kd',     0.02),
                ('yaw.kd',       0.02),

                # Wrist manip properties
                ('ellipsoid_a', 0.07),
                ('ellipsoid_b', 0.08),
                ('ellipsoid_c', 0.10),
                #('position_origin', [0.00, 0.0, 0.00]),
                ('position_origin', [0.05, 0.0, 0.02]),
                ('orientation_origin', [0.0, 0.0, 0.0, 1.0]),
                ('wrist.x_offset',  0.06),
                ('wrist.z_offset',  0.06),
                ('wrist.stiffness', 10.0),
                ('wrist.damping',    0.1),
                ('roll.range',      30.0),
                ('pitch.range',     30.0),
                ('yaw.range',       30.0),

                # Wave generation properties
                ('wave_axis', 'x'),
                ('enable_wave', False),
                ('wave_amplitude', 45.0),
                ('wave_frequency', 0.1),
                ('rotation_angle', 0.0),
                ('spherical_boundary_radius', 0.06),
                ('sweep_repetitions', 2),
                ('angular_step_size', 15.0),
            ],
        )

        # I also want to make a callback function whenever the parameters are changed

    def parameter_callback(self, params):
        """ Callback function to update the parameters when they are changed.
        """
        #self.logger("Parameters have been updated.")
        #for param in params:
        #    self.logger(f'{param.name}: {param.value}')
        #if param.name == 'basis.stiffness':
            #self.max_stiffness = param.value
            #self.logger(f'Max stiffness: {self.max_stiffness}')
        return SetParametersResult(successful=True)

    def initialize_subscribers(self):

        self.pose_subscription = self.create_subscription(Pose, self.get_parameter('robot_pose_topic').value,
                                                         self.pose_callback,  qos_profile=qos_profile_default)
        self.twist_subscription = self.create_subscription(Twist, self.get_parameter('robot_twist_topic').value,
                                                         self.twist_callback, qos_profile=qos_profile_default)
        self.gripper_subscription = self.create_subscription(Float64, self.get_parameter('gripper_angle_topic').value,
                                                             self.gripper_callback, qos_profile=qos_profile_default)

    def initialize_publishers(self):
        """
        Initialize the publisher to publish the Sigma7 force command.
        """
        self.sigma7_force_publisher = self.create_publisher(Sigma7, self.get_parameter('sigma7_force_command_topic').value,
                                                        qos_profile=qos_profile_default)

    def init_force_fib_callback(self):
        """ Update the table stiffness and other forces by slowly ramping up the force to the desired value."""
        # Fibonacci sequence style!
        if self.nth < self.max_stiffness:
            self.nth = self.n1 + self.n2
            self.nth = self.max_stiffness if self.nth > self.max_stiffness else self.nth
            if self.verbose:
                self.logger(f'Sending stiffness parameter: {float(self.nth)}')
            self.node_client.set_parameter('basis.stiffness', float(self.nth))
            self.n1 = self.n2
            self.n2 = self.nth
        else:
            if self.verbose:
                self.logger(f'Final stiffness parameter: {self.nth}')
            self.destroy_timer(self.force_init_timer)

    def disp_msg(self, msg, new_msg=False):
        """Prints the message only once. The 'new_msg' parameter needs to be set to True to enable print"""
        if self.new_msg or new_msg:
            print(msg)
            self.new_msg = False

    def publish_sigma7_forces(self, sigma7_force):
        """
        Publish the sigma7 force command.

        :param sigma7_force: The sigma7 force to be applied.
        """
        self.sigma7_force_publisher.publish(sigma7_force)

    def logger(self, msg):
        self.get_logger().info(msg)

    def keyboard_callback(self, key):
        self.last_key = key
        #print("Last key: ", self.last_key)

    def pose_callback(self, msg):
        self.current_pose = msg

    def twist_callback(self, msg):
        self.current_vel = msg

    def gripper_callback(self, msg):
            self.gripper_angle = msg.data

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

    def calculate_resisting_force(self, desired_pose):
        """
        Calculate the resisting force and torque to restrict motion in undesired directions.

        Parameters:
        -----------
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

        current_pose = self.current_pose
        current_velocity = self.current_vel

        if self.get_parameter('enable_position_restriction').value:
            # Calculate the difference between the robot's current position and desired position
            dp_x = desired_pose.position.x - current_pose.position.x
            dp_y = desired_pose.position.y - current_pose.position.y
            dp_z = desired_pose.position.z - current_pose.position.z

            # Calculate the force needed to keep x, y, z positions to zero. Use the spring damping model.
            # F = -k * dx - b * dx/dt
            k = self.get_parameter('basis.stiffness').value
            d = self.get_parameter('basis.damping').value
            force_msg.force.x = k * dp_x - d * current_velocity.linear.x
            force_msg.force.y = k * dp_y - d * current_velocity.linear.y
            force_msg.force.z = k * dp_z - d * current_velocity.linear.z

        if self.get_parameter('enable_orientation_restriction').value:
            # Calculate the difference between the robot's current orientation and desired orientation
            (current_roll, current_pitch, current_yaw) = r2sc.quaternion_to_euler(current_pose.orientation.x,
                                                                             current_pose.orientation.y,
                                                                             current_pose.orientation.z,
                                                                             current_pose.orientation.w)
            (desired_roll, desired_pitch, desired_yaw) = r2sc.quaternion_to_euler(desired_pose.orientation.x,
                                                                             desired_pose.orientation.y,
                                                                             desired_pose.orientation.z,
                                                                             desired_pose.orientation.w)

            #self.logger("current orientation: roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(current_roll, current_pitch, current_yaw))
            #self.logger("desired orientation: roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(desired_roll, desired_pitch, desired_yaw))

            # Calculate the torque needed to keep roll, pitch, yaw to zero. Use PD control.
            d_theta = [desired_roll - current_roll, desired_pitch - current_pitch, desired_yaw - current_yaw]
            d_omega = [current_velocity.angular.x, current_velocity.angular.y, current_velocity.angular.z]
            force_msg.torque.x = (self.get_parameter('roll.kp').value * d_theta[0] - self.get_parameter(
                'roll.kd').value * d_omega[0])
            force_msg.torque.y = (self.get_parameter('pitch.kp').value * d_theta[1] - self.get_parameter(
                'pitch.kd').value * d_omega[1])
            force_msg.torque.z = (self.get_parameter('yaw.kp').value * d_theta[2] - self.get_parameter(
                'yaw.kd').value * d_omega[2])

        return force_msg

    def sinusoidal_movement_callback(self):
        """ Perform a sinusoidal movement test to calculate the passive joint stiffness of the wrist."""
        # given that my sine wav has an amplitude of 0.1, and a frequency of 0.02 rad/sec, write a function that will
        # output a position on the wave where my input counter increases 0.001 every run of the loop
        # The function should output a position on the wave that is 0.1*sin(0.02*counter)

        # Calculate the desired position on the sinusoidal wave
        desired_position = self.get_parameter('wave_amplitude').value * math.sin(self.get_parameter('wave_frequency').value * self.counter)
        self.counter += 0.001

        desired_pose = Pose()
        wave_axis = self.get_parameter('wave_axis').value
        if wave_axis == 'x':
            desired_pose.position.x = desired_position
            desired_pose.position.y = 0.0
            desired_pose.position.z = 0.0
        elif wave_axis == 'y':
            desired_pose.position.x = 0.0
            desired_pose.position.y = desired_position
            desired_pose.position.z = 0.0
        else:
            desired_pose.position.x = 0.0
            desired_pose.position.y = 0.0
            desired_pose.position.z = desired_position

        return desired_pose

    def get_robot_orientation(self):
        """ Calculate the orientation of the robot's end effector in terms of roll, pitch, and yaw."""
        # Calculate the distance between the robot current position and origin
        pose = self.current_pose
        ORIGIN = self.get_parameter('position_origin').value
        dx = (pose.position.x - ORIGIN[0])
        dy = (pose.position.y - ORIGIN[1])
        dz = (pose.position.z - ORIGIN[2])

        # Calculate the euler angles for the desired orientation.
        # Desired orientation is the pitch and yaw euler angle components using the vector between the origin and the robot's current position on the sphere.
        cur_roll = 0.00  # No roll allowed

        # Calculate the pitch of the handle, and factor in the orientation of the wrist with respect to the handle
        z1 = self.get_parameter('wrist.z_offset').value
        x1 = self.get_parameter('wrist.x_offset').value
        wrist_offset = math.atan2(z1, x1)
        cur_pitch = -(math.atan2(-dz, math.sqrt(dx ** 2 + dy ** 2)) + wrist_offset)
        cur_yaw = math.atan2(dy, dx)
        # -pi transitions to pi,  Reorient the yaw angle so 0 is at the front of the sphere.
        sign = 1 if cur_yaw < 0 else -1
        cur_yaw = sign * (math.pi - abs(cur_yaw))

        return cur_roll, cur_pitch, cur_yaw

    def calculate_torque(self, desired_angle):
        """ Computes the generated torque based on the desired angle of the wrist manipulandum.
        The equation follows the format of T = k * theta - b * omega, where k is the stiffness and b is the damping coefficient.
        theta is the difference between the robot's current orientation (in yaw or whatever axis is being used) and the desired orientation.
        omega is the angular velocity of the robot in the same axis.

        Parameters:
        -----------
        desired_angle  (list): The desired angle of the wrist manip in terms of roll, pitch, and yaw
        """

        cur_roll, cur_pitch, cur_yaw = self.get_robot_orientation()
        omega_roll = self.current_vel.angular.x
        omega_pitch = self.current_vel.angular.y
        omega_yaw = self.current_vel.angular.z

        # Calculate the difference between the current yaw and the desired angle
        theta_pitch = cur_pitch - desired_angle[1]
        theta_yaw = cur_yaw - desired_angle[2]

        ## Calculate the torque
        K = self.get_parameter('wrist.stiffness').value
        d = self.get_parameter('wrist.damping').value
        T_pitch = K * theta_pitch - d * omega_pitch
        T_yaw = K * theta_yaw - d * omega_yaw

        # Calculate torque vectors
        torque_parallel_pitch = np.array([0, T_pitch, 0])
        torque_parallel_yaw = np.array([0, 0, T_yaw])

        # Calculate perpendicular torques
        torque_perpendicular_pitch = np.cross(torque_parallel_pitch, np.array([0, 0, 1]))
        torque_perpendicular_yaw = np.cross(torque_parallel_yaw, np.array([0, 1, 0]))

        return [torque_parallel_pitch, torque_parallel_yaw, torque_perpendicular_pitch, torque_perpendicular_yaw]

    def compute_basis_force(self):
        """
        "modeled as a spring+damper_ system that pull the device towards its
         projection on the constraint segment, either a line, plane, or unrestricted"
        K_ is the spring constant, or stiffness_.
        C is the damping coefficient, or damping_.

        . _K: https://en.wikipedia.org/wiki/Spring_(device)#Physics
        . _stiffness: https://en.wikipedia.org/wiki/Stiffness
        . _spring+damper: https://en.wikipedia.org/wiki/Mass-spring-damper_model
        . _damper: https://en.wikipedia.org/wiki/Dashpot
        . _damping: https://en.wikipedia.org/wiki/Damping
        """

        # Initialize a utility function for getting parameters.
        get_parameter = lambda n: self.get_parameter(n).value

        # Initialize a utility function for converting messages to numpy arrays.
        to_odict = rosidl_runtime_py.convert.message_to_ordereddict
        msg_to_array = lambda m: np.array(tuple(to_odict(m).values()))
        to_column = lambda s: np.array(s)[:][np.newaxis].T

        # Get the current position and velocity of the robot.
        p = to_column(msg_to_array(self.current_pose.position))
        v = to_column(msg_to_array(self.current_vel.linear))

        # Create a projection matrix from the attractor basis parameter.
        b = self.get_parameter('basis.transform').value
        B = np.array(b).reshape((3, 3))

        # Get the rotation matrix based on the specified axis
        angle = self.get_parameter('rotation_angle').value * math.pi / 180  # in radians
        rot_mat =  r2sc.get_rotation_matrix(self.get_parameter('wave_axis').value, angle)

        # Apply rotation matrix to basis matrix
        B_rotated = np.dot(rot_mat, B)

        # Create projection matrix from the rotated basis matrix
        P_a = B @ np.linalg.pinv(B_rotated)

        # Compute the projection of the position vector onto the attractor
        # surface -- that is, the permissible plane in which the effector
        # should be constrained to move -- after subtracting the offset.
        o = to_column(get_parameter('position_origin'))
        p_a = P_a @ (p - o) + o

        # Compute the difference vector between the position vector and the
        # offset plane.
        p_d = p_a - p

        # Compute the "guidance force" from the spring + damper model.
        K = get_parameter('basis.stiffness')
        #print("Using K of ", K)
        C = get_parameter('basis.damping')
        f_g = K * (p_a - p) - C * v

        # Project the guidance force onto the difference vector; that is, the
        # tangent to the permissible plane.
        # This is recommended by the Force Dimension example, and seems
        # necessary due to the velocity term in the spring + damper model,
        # which can introduce viscous force components on the permissible
        # attractor surface (i.e., the "free" plane).
        P_g = p_d @ np.linalg.pinv(p_d)
        f_g = P_g @ f_g

        f_e = f_g.squeeze()

        if (len(f_e) != 3) or not isinstance(f_e[0], float): raise Exception()

        # Return the result as a tuple.
        return f_e

    def sinusoidal_arc_callback(self):
        """ The callback function that causes the robot to perform a sinusoidal movement by calculating the cartesian
            position of the robot's end effector based on the projected/desired angle. If the position and force data
            is also being recorded, then several other parameters are updated

        The 'rotation_angle' parameter starts at 0 degrees, and changes by the 'angular_step_size' parameter after the
            number of sweeps in the 'sweeps_repetitions' parameter is satisfied.
        The 'angular_step_size' parameter is in degrees, and the 'rotation_angle' parameter is in degrees.
        The 'sweeps_repetitions' parameter is the number of times the robot will sweep through the sinusoidal wave.
        The 'wave_frequency' parameter is the frequency of the sinusoidal wave.
        """

        if not self.get_parameter('enable_position_restriction').value:
            return self.current_pose, 0.0

        # If the wave has gone through a full cycle, increment the number of sweeps
        if self.counter >= 1/(self.get_parameter('wave_frequency').value):
            print("completed")
            self.completed_sweeps += 1
            self.counter = 0.0

        if self.completed_sweeps >= self.get_parameter('sweep_repetitions').value:
            # Adjust the rotation angle by the angular step size
            current_angle = self.get_parameter('rotation_angle').value
            new_angle = current_angle + self.get_parameter('angular_step_size').value
            self.logger("new angle: {:2.4f}".format(new_angle))
            self.node_client.set_parameter('rotation_angle', new_angle)
            self.completed_sweeps = 0

        des_rot_angle = self.get_parameter('rotation_angle').value


        # Calculate the desired angle on the 2D sinusoidal wave
        f = self.get_parameter('wave_frequency').value
        A = self.get_parameter('wave_amplitude').value
        # (also need to spin 180 degrees because of the orientation of the robot)
        desired_angle = 180 - A * math.sin(2 * math.pi * f * self.counter)
        self.desired_angle = desired_angle
        #self.logger("angle in degrees: {:2.4f}".format(desired_angle))
        #self.logger("counter: {:2.4f}".format(self.counter))

        # The handle should stay at the neutral position when the wave is disabled, even if the position restriction
        # parameter is enabled
        if self.get_parameter('enable_wave').value:
            self.logger("sweeps: {:2.0f}, angle: {:2.4f}".format(self.completed_sweeps, des_rot_angle))
            self.counter += self.get_parameter('sample_rate').value

        # convert polar to cartesian coordinates
        angle_rad = desired_angle * math.pi / 180
        # Get the norm direction vector
        direction_vector = np.array([math.cos(angle_rad), math.sin(angle_rad), 0])

        # Create a projection matrix from the attractor basis parameter.
        b = self.get_parameter('basis.transform').value
        B = np.array(b).reshape((3, 3))

        # Get the rotation matrix based on the specified axis
        angle = des_rot_angle * math.pi / 180  # in radians
        rot_mat = r2sc.get_rotation_matrix(self.get_parameter('wave_axis').value, angle)

        # Apply rotation matrix to basis matrix
        B_rotated = np.dot(rot_mat, B)

        # Rotate the direction vector by the basis matrix
        dir_vec = np.dot(B_rotated, direction_vector)

        # Does the magnitude of the dir_vec variable equal to 1?
        #print("Magnitude of dir_vec: ", np.linalg.norm(dir_vec))

        # Projected radius given ellipsoid
        e_x = self.get_parameter('ellipsoid_a').value
        e_y = self.get_parameter('ellipsoid_b').value
        e_z = self.get_parameter('ellipsoid_c').value
        r = self.projected_radius_ellipsoid(dir_vec, e_x, e_y, e_z) # Very happy how this turned out!
        #r = 0.03
        self.projected_radius = r # Save this for data collection

        # Rescale to match the radius of the ellipsoid
        scaled_dir_vec = r * dir_vec

        # Last, subtract the position origin from the scaled direction vector
        ORIGIN = self.get_parameter('position_origin').value
        x = ORIGIN[0] + scaled_dir_vec[0]
        y = ORIGIN[1] + scaled_dir_vec[1]
        z = ORIGIN[2] + scaled_dir_vec[2]

        desired_pose = Pose()
        desired_pose.position.x = x
        desired_pose.position.y = y
        desired_pose.position.z = z

        return desired_pose, angle_rad

    def get_wave_vector(self):
        """ Calculate the desired directional vector based on a sinusoidal wave.
        """
        # Convert 15 degrees to radians
        max_amplitude_radians = 15 * math.pi / 180

        # Calculate the desired angle on the sinusoidal wave
        f = self.get_parameter('wave_frequency').value
        A = self.get_parameter('wave_amplitude').value
        desired_angle = max_amplitude_radians * A * math.sin(f * self.counter)

        # Calculate the desired directional vector based on the desired angle
        # Using trigonometric functions to create a unit vector
        # In a 2D plane, you can use cos and sin of the desired angle for the unit vector
        if self.wave_axis == 'x':
            # The wave is in the xy-plane
            desired_vector = np.array([math.cos(desired_angle), math.sin(desired_angle), 0])
        elif self.wave_axis == 'y':
            # The wave is in the yz-plane
            desired_vector = np.array([0, math.cos(desired_angle), math.sin(desired_angle)])
        else:
            # Default to wave in the xz-plane
            desired_vector = np.array([math.cos(desired_angle), 0, math.sin(desired_angle)])

        # Increment the counter for the next iteration
        self.counter += 0.001

        # Normalize the vector
        desired_vector = desired_vector / np.linalg.norm(desired_vector)

        # Return the normalized directional vector
        return desired_vector

    def get_last_key(self):
        last_key = self.last_key
        self.last_key = None

        if last_key is not None:
            last_key.lower()
        return last_key

    def update_force_callback(self):
        """ Assigns a new basis for the robot's workspace to restrict movement in the x-y axis and displays the robot's current position
        """

        # Create the sigma7 force message at the beginning
        sigma7_force = Sigma7()

        # Calculate the basis force to restrict the space the robot can move in
        basis_force = self.compute_basis_force()

        # Calculate the resisting force based on the desired pose
        # Apply the wrist manipulandum restriction to the robot's current pose, which gets us the desired pose
        # desired_pose = self.wrist_manipulandum_restriction()
        # desired_pose = self.sinusoidal_movement_callback()
        desired_pose, desired_angle = self.sinusoidal_arc_callback()
        resisting_force = self.calculate_resisting_force(desired_pose)

        #self.logger("resisting force: x: {:2.4f}, y: {:2.4f}, z: {:2.4f}".format(resisting_force.force.x, resisting_force.force.y, resisting_force.force.z))
        #generated_torque = self.calculate_torque([0, 0, desired_angle])
        #self.logger("Torque: {:2.4f}".format(generated_torque[1]))

        # Apply the basis restriction to the robot resistive force
        #f_x = resisting_force.force.x + basis_force[0]
        #f_y = resisting_force.force.y + basis_force[1]
        #f_z = resisting_force.force.z + basis_force[2]
        # Or not use the basis force
        f_x = resisting_force.force.x
        f_y = resisting_force.force.y
        f_z = resisting_force.force.z
        self.f_e = [f_x, f_y, f_z]

        # Apply the resisting force to the robot
        sigma7_force.force.x = f_x
        sigma7_force.force.y = f_y
        sigma7_force.force.z = f_z
        sigma7_force.torque.x = resisting_force.torque.x
        sigma7_force.torque.y = resisting_force.torque.y
        sigma7_force.torque.z = resisting_force.torque.z

        # Publish the sigma7 force command
        self.publish_sigma7_forces(sigma7_force)

    def wrist_manipulandum_restriction(self):
        """ Restrict the robot movement to a wrist manipulandum's workspace. The position is fixed to origin in space, and rotation is disabled, but pitch and yaw are allowed.

        Assumes that the points for the flexion and radial deviation axes have been calculated.

        """

        # Calculate the distance between the robot current position and origin
        pose = self.current_pose
        ORIGIN = self.get_parameter('position_origin').value
        dx = (pose.position.x - ORIGIN[0])
        dy = (pose.position.y - ORIGIN[1])
        dz = (pose.position.z - ORIGIN[2])

        # Calculate the euler angles for the desired orientation.
        # Desired orientation is the pitch and yaw euler angle components using the vector between the origin and the robot's current position on the sphere.
        cur_roll = 0.00 # No roll allowed

        # Calculate the pitch of the handle, and factor in the orientation of the wrist with respect to the handle
        z1 = self.get_parameter('wrist.z_offset').value
        x1 = self.get_parameter('wrist.x_offset').value
        wrist_offset = math.atan2(z1, x1)
        cur_pitch = -(math.atan2(-dz, math.sqrt(dx ** 2 + dy ** 2)) + wrist_offset)
        cur_yaw = math.atan2(dy, dx)
        # -pi transitions to pi,  Reorient the yaw angle so 0 is at the front of the sphere.
        sign = 1 if cur_yaw < 0 else -1
        cur_yaw = sign * (math.pi - abs(cur_yaw))
        #self.logger("orientation: roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(cur_roll, cur_pitch, cur_yaw))

        # Convert the pitch and yaw angles to quaternions and add to pose message
        desired_pose = Pose()
        desired_pose.orientation.x, desired_pose.orientation.y, desired_pose.orientation.z, desired_pose.orientation.w = r2sc.euler_to_quaternion(cur_roll, cur_pitch, cur_yaw)

        # Calculate the position of the robot
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if distance == 0:
            distance = 1e-6

        # If the distance is greater or less than the radius of the sphere, then find the point on the sphere that is closest to the robot current position.
        direction_vector = np.array([dx, dy, dz])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Projected radius given ellipsoid
        e_x = self.get_parameter('ellipsoid_a').value
        e_y = self.get_parameter('ellipsoid_b').value
        e_z = self.get_parameter('ellipsoid_c').value
        sphere_radius = self.projected_radius_ellipsoid(direction_vector, e_x, e_y, e_z)
        #sphere_radius = self.get_parameter('spherical_boundary_radius').value
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

    def write_data_to_csv(self, data_dict, csv_file):
        """
        Write data from a dictionary to a CSV file.

        Parameters:
            data_dict (dict): Dictionary containing data.
            csv_file (str): Path to the CSV file.

        Returns:
            None
        """
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['roll_angle', 'yaw_angle', 'position_x', 'position_y', 'position_z',
                          'force_x', 'force_y', 'force_z', 'radius']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for angle, data in data_dict.items():
                for i in range(len(data['position'])):
                    writer.writerow({
                        'roll_angle': angle,
                        'yaw_angle': data['desired_angle'][i],
                        'position_x': data['position'][i][0],
                        'position_y': data['position'][i][1],
                        'position_z': data['position'][i][2],
                        'force_x': data['force'][i][0],
                        'force_y': data['force'][i][1],
                        'force_z': data['force'][i][2],
                        'radius': data['radius'][i]
                    })

    def main_callback(self):
        # This is the main callback function that runs the state machine for the impedance controller

        if self.state == 'begin':
            # First we need to ask if we will load wrist parameters from a yaml file (type '1'), or use the default settings (type '2')
            # If the user types '1', then we will load the yaml file and set the parameters
            # If the user types '2', then we will use the default settings

            # Ask the user to relax their arm and wrist
            self.disp_msg("Please relax your arm and wrist. Press 'y' to continue. Press q at any time to quit.")
            last_key = self.get_last_key()
            if last_key == 'q':
                self.disp_msg("Quitting the program.", new_msg=True)
                self.new_msg = True
                self.state = 'quit'
            elif last_key == 'y':
                self.new_msg = True
                self.state = 'init_force'
            elif last_key is None:
                pass

        if self.state == 'init_force':
            sr = self.get_parameter('sample_rate').value
            self.force_dimension_client.set_parameter('enable_force', False)
            self.publish_sigma7_forces(Sigma7())
            self.node_client.set_parameter('basis.stiffness', 0.0)
            self.force_dimension_client.set_parameter('enable_force', True)
            self.force_dimension_client.set_parameter('gravity_compensation', True)
            # The feedback sample decimation needs to be set to a low number for smooth force feedback
            self.force_dimension_client.set_parameter('feedback_sample_decimation.pose', 10)
            if self.force_timer is None:
                self.force_timer = self.create_timer(sr, self.update_force_callback)
            # Initialize the force restrictions for the basis (forces already initialized)
            self.nth, self.n1, self.n2 = 0, 1, 0
            if self.force_init_timer is None:
                self.force_init_timer = self.create_timer(0.2, self.init_force_fib_callback) # Stops itself after it's done
            self.state = 'parameter_config'

        if self.state== 'parameter_config':
            self.disp_msg("Would you like to load wrist parameters from a yaml file or use the default settings? Type '1' to load yaml, '2' for default.")
            last_key = self.get_last_key()
            if last_key == '1':
                # Load the yaml file and set the parameters
                self.logger("Loading wrist parameters from yaml file.")
                # Load the yaml file
                # Set the parameters
                self.logger("not supported yet! Using default")
                self.new_msg = True
                self.state = 'setup_data_collection'
            elif last_key == '2':
                # Use the default settings. The parameters when the node starts should have the default values
                self.disp_msg("Using default wrist parameters.", new_msg=True)
                self.new_msg = True
                self.state = 'setup_data_collection'
            elif last_key == 'q':
                self.logger("Quitting the program.")
                self.state = 'quit'
            elif last_key is None:
                pass
            else:
                self.logger("Invalid input. Please type '1' for yes, '2' for no.")

        if self.state == 'setup_data_collection':
            # In this state we enable the position restrictions with the wrist in the neutral position. We still
            # have a separate parameter to enable the wave generation.
            if not self.get_parameter('enable_position_restriction').value:
                self.node_client.set_parameter('enable_position_restriction', True)
            self.disp_msg("Ready to begin data collection. Press 'y' to continue. Press 'q' to quit.")
            last_key = self.get_last_key()
            if last_key == 'q':
                self.state = 'quit'
            elif last_key == 'y':
                # Start the position wave generation time
                self.counter = 0.0
                self.collect_data = True
                self.node_client.set_parameter('enable_wave', True)
                sr = self.get_parameter('sample_rate').value
                self.arc_timer = self.create_timer(sr, self.sinusoidal_arc_callback)
                self.new_msg = True
                self.state = 'data_collection'
            elif last_key is None:
                pass

        if self.state == 'data_collection':
            # Data should now be collected with the sweeping motion of the arc for all the angles desired, until a 180
            # degree sweep is completed. Until then, we will keep collecting data.
            self.disp_msg("Collecting data. Please wait for the process to finish")
            def save_data(angle, position, force, radius, desired_angle, data_dict):
                """
                Helper function to save data for a given angle/rotation to a dictionary.

                Parameters:
                    angle       (float): Angle/rotation value.
                    position (np.array): Position data.
                    force    (np.array): Force data.
                    radius      (float): Estimated radius.
                    desired_angle (float): Desired angle.
                    data_dict    (dict): Dictionary to save the data into.

                Returns: None
                """
                # Check if the angle is already in the dictionary
                if angle in data_dict:
                    # Append the data to the existing list for this angle/rotation
                    data_dict[angle]['position'].append(position)
                    data_dict[angle]['desired_angle'].append(desired_angle)
                    data_dict[angle]['force'].append(force)
                    data_dict[angle]['radius'].append(radius)
                else:
                    # Create a new list for this angle/rotation
                    data_dict[angle] = {'position': [position],
                                        'desired_angle': [desired_angle],
                                        'force': [force],
                                        'radius': [radius]}

                    print(data_dict[angle])

            angle = self.get_parameter('rotation_angle').value
            position = [self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z]
            save_data(str(angle), position, self.f_e, self.projected_radius, self.desired_angle, self.transform_data)

            if angle == 180.0:
                # We've gotten a full sweep, and we can stop collecting data and disable the wave
                self.collect_data = False
                self.node_client.set_parameter('enable_wave', False)
                self.new_msg = True
                self.state = 'save'

        if self.state == 'save':
            # Save the data to a file
            self.disp_msg("Data collection complete. Press 'y' to save the data. Press 'q' to quit.")
            last_key = self.get_last_key()
            if last_key == 'q':
                self.new_msg = True
                self.state = 'quit'
            elif last_key == 'y':
                self.destroy_timer(self.arc_timer)
                self.logger("Saving data to file.")
                self.write_data_to_csv(self.transform_data, "data.csv")
                self.new_msg = True
                self.state = 'quit'
            elif last_key is None:
                pass

        if self.state=='quit':
            self.disp_msg("Quitting the program.", new_msg=True)
            self.keyboard_thread.join()
            self.destroy_timer(self.main_timer)
            self.destroy_timer(self.force_timer)
            self.destroy_timer(self.arc_timer)
            self.destroy_timer(self.force_init_timer)
            self.destroy_node()

def main(args=None):

    try:
        rclpy.init(args=args)
        node = JointStiffness()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down the node.")

    finally:
        #node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()





