#!/usr/bin/env python3

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose as pose_msg
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Quaternion
from example_interfaces.msg import Float64
from haptic_device_interfaces.msg import Sigma7

# Declare default quality-of-service settings for ROS2.
DEFAULT_QOS = rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value

# Conversion from x, y, z translation in pixels to meters (measured as 'm' workspace room, and 'p' pixel values)
px_to_m = 1 # 240 / 0.26 / 1000
py_to_m = 1 # 190 / 0.20 / 1000
pz_to_m = 1 # 130 / 0.12 / 1000

#SPHERE_RADIUS = 0.05
#ORIGIN = [0.02, 0.0, 0.0]

# This function is a stripped down version of the code in
# https://github.com/matthew-brett/transforms3d/blob/f185e866ecccb66c545559bc9f2e19cb5025e0ab/transforms3d/euler.py
# Besides simplifying it, this version also inverts the order to return x,y,z,w, which is
# the way that ROS prefers it.
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


class ForceDimensionNode(Node):
    def __init__(self):
        """
        Initialize the ForceDimensionNode class.
        """
        super().__init__('ros_env')

        self.initialize_parameters()
        self.initialize_subscribers()
        self.initialize_publishers()

        self.current_rotation_velocity = None
        self.current_position_velocity = None
        self.gripper_position = 0.000
        self.gripper_force = 0.000
        self.previous_pose = None  # Store the previous pose of the robot
        self.current_pose = None  # Store the current pose of the robot
        self.t_prev = time.time()  # Store the previous time

        # Create a timer to handle physics and forces at a specified rate
        self.timer = self.create_timer(self.get_parameter('sample_rate').value, self.physics_callback)

        # Add a "on set" parameter callback.
        #self.add_on_set_parameters_callback(self._parameters_callback)

        self.logger("Environment boundary: {}".format(self.get_parameter('environment_boundary').value))


    def logger(self, msg):
        self.get_logger().info(msg)

    def initialize_parameters(self):
        """ Define all parameters in this function.
        """
        default_transform = [+1.0, +0.0, +0.0,
                             +0.0, +1.0, +0.0,
                             +0.0, +0.0, +1.0]

        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_node_name', 'robot/manipulandum'),
                ('robot_pose_topic', '/robot/feedback/pose'),
                ('gripper_angle_topic', '/robot/feedback/gripper_angle'),
                ('wrench_command_topic', '/robot/command/wrench'),
                ('sigma7_force_command_topic', '/robot/command/sigma7_force'),
                ('enable_position_restriction', True),
                ('enable_orientation_restriction', False),
                ('environment_boundary', 'sphere_rotation'),
                ('gripper_stiffness', 0.0000001),
                ('position_origin', [0.0, 0.0, 0.0]),
                ('orientation_origin', [0.0, 0.0, 0.0, 1.0]),
                ('wrist.x_offset', 0.07),
                ('wrist.z_offset', 0.003),
                ('x.stiffness',  10.0),
                ('y.stiffness',  10.0),
                ('z.stiffness',  100.0),
                ('x.damping',    0.1),
                ('y.damping',    0.1),
                ('z.damping',    0.1),
                ('roll.kp',      0.01),
                ('pitch.kp',     0.01),
                ('yaw.kp',       0.01),
                ('roll.kd',      5.0),
                ('pitch.kd',     5.0),
                ('yaw.kd',       5.0),
                ('roll.range',   30.0),
                ('pitch.range',  30.0),
                ('yaw.range',    30.0),
                ('sample_rate',  0.001),
                ('transform', default_transform),
                ('spherical_boundary_radius', 0.06),
            ],
        )

    def initialize_subscribers(self):

        self.pose_subscription = self.create_subscription(pose_msg, self.get_parameter('robot_pose_topic').value,
                                                         self.pose_callback,  qos_profile=DEFAULT_QOS)
        self.gripper_subscription = self.create_subscription(Float64, self.get_parameter('gripper_angle_topic').value,
                                                             self.gripper_callback, qos_profile=DEFAULT_QOS)

    def initialize_publishers(self):
        """
        Initialize the publisher to publish the resisting wrench command.
        """
        self.wrench_publisher = self.create_publisher(Wrench, self.get_parameter('wrench_command_topic').value,
                                                      qos_profile=DEFAULT_QOS)

        self.sigma7_force_publisher = self.create_publisher(Sigma7, self.get_parameter('sigma7_force_command_topic').value,
                                                        qos_profile=DEFAULT_QOS)
    def pose_callback(self, msg):
        """
        Callback function to process received pose messages.

        Parameters:
        -----------
        msg (geometry_msgs.msg.Pose) : The received Pose message.
        """
        # Store the current position and orientation
        if self.current_pose is not None:
            self.previous_pose = self.current_pose
        self.current_pose = msg

        # Create a Pose message and initialize it with values for the position.x field and orientation.x field
        if self.previous_pose is not None:
            # Calculate the current cartesional velocity
            dx = self.current_pose.position.x - self.previous_pose.position.x
            dy = self.current_pose.position.y - self.previous_pose.position.y
            dz = self.current_pose.position.z - self.previous_pose.position.z
            self.current_position_velocity = np.array([dx, dy, dz])

            # Calculate the current angular velocity
            droll = self.current_pose.orientation.x - self.previous_pose.orientation.x
            dpitch = self.current_pose.orientation.y - self.previous_pose.orientation.y
            dyaw = self.current_pose.orientation.z - self.previous_pose.orientation.z
            self.current_rotation_velocity = np.array([droll, dpitch, dyaw])

        # Update the last time call
        self.t_prev = time.time()

    def gripper_callback(self, msg):
        """
        Callback function to process received gripper messages.

        Parameters:
        -----------
        msg (std_msgs.msg.Float64) : The received gripper message.
        """
        self.gripper_position = msg.data

    def enforce_angle_bounds(self, angle, bound, unit="degrees"):
        """ Enforce the angle within the specified bounds.

        Parameters:
            ----------
            angle (float) : The desired angle
            bound (float) : The specified lower/upper bound
        """
        if unit == "radians":
            angle = math.degrees(angle)

        if abs(angle) > bound:
            sign = -1 if angle < 0 else 1
            angle = sign * bound

        return angle

    def origin_restriction(self):
        """ Restrict the robot to the origin of the workspace.
        """
        desired_pose = pose_msg()
        ORIGIN = self.get_parameter('position_origin').value
        desired_pose.position.x = ORIGIN[0]
        desired_pose.position.y = ORIGIN[1]
        desired_pose.position.z = ORIGIN[2]

        # Convert the current robot orientation to euler angles
        cur_roll, cur_pitch, cur_yaw = quaternion_to_euler(self.current_pose.orientation.x,
                                                           self.current_pose.orientation.y,
                                                           self.current_pose.orientation.z,
                                                           self.current_pose.orientation.w)
        # Enforce the angle within the specified bounds
        cur_roll = self.enforce_angle_bounds(cur_roll, self.get_parameter('roll.range').value)
        cur_pitch = self.enforce_angle_bounds(cur_pitch, self.get_parameter('pitch.range').value)
        cur_yaw = self.enforce_angle_bounds(cur_yaw, self.get_parameter('yaw.range').value)

        # self.logger("desired orientation: roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(cur_roll, cur_pitch, cur_yaw))

        desired_pose.orientation.x, desired_pose.orientation.y, desired_pose.orientation.z, desired_pose.orientation.w = euler_to_quaternion(
            math.radians(cur_roll), math.radians(cur_pitch), math.radians(cur_yaw))

        return desired_pose

    def wrist_manipulandum_restriction(self):
        """ Restrict the robot movement to a wrist manipulandum's workspace. The position is fixed to origin in space, and rotation is disabled, but pitch and yaw are allowed.
        """

        # Calculate the distance between the robot current position and origin
        pose = self.current_pose
        ORIGIN = self.get_parameter('position_origin').value
        dx = (pose.position.x - ORIGIN[0])
        dy = (pose.position.y - ORIGIN[1])
        dz = (pose.position.z - ORIGIN[2])

        # Calculate the euler angles for the desired orientation.
        # Desired orientation is the pitch and yaw euler angle components using the vector between the origin and
        # the robot's current position on the sphere. Just need the pitch and yaw angles
        cur_roll = 0.00 # No roll allowed

        # Calculate the pitch of the handle, and factor in the orientation of the wrist with respect to the handle
        wrist_offset = math.atan2(self.get_parameter('wrist.z_offset').value, self.get_parameter('wrist.x_offset').value)
        cur_pitch = -(math.atan2(-dz, math.sqrt(dx ** 2 + dy ** 2)) + wrist_offset)
        cur_yaw = math.atan2(dy, dx)
        # -pi transitions to pi,  Reorient the yaw angle so 0 is at the front of the sphere.
        sign = 1 if cur_yaw < 0 else -1
        cur_yaw = sign * (math.pi - abs(cur_yaw))

        # Enforce the angle within the specified bounds
        # TO-DO: output goes to bounds but I want the angle to be anything between -bound:bound

        #cur_roll, lock_roll = self.enforce_angle_bounds(cur_roll, math.radians(self.get_parameter('roll.range').value), "radians")
        #cur_pitch = self.enforce_angle_bounds(cur_pitch, math.radians(self.get_parameter('pitch.range').value), "radians")
        #cur_yaw = self.enforce_angle_bounds(cur_yaw, math.radians(self.get_parameter('yaw.range').value), "radians")

        self.logger("orientation: roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(cur_roll, cur_pitch, cur_yaw))

        # Convert the pitch and yaw angles to quaternions and add to pose message
        desired_pose = pose_msg()
        desired_pose.orientation.x, desired_pose.orientation.y, desired_pose.orientation.z, desired_pose.orientation.w = euler_to_quaternion(cur_roll, cur_pitch, cur_yaw)

        # TO-DO: the position of the robot should be restricted to the cone of the rotation boundary with the manipulandum


        # Calculate the position of the robot
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        sphere_radius = self.get_parameter('wrist.x_offset').value

        # Calculate the

        # If the distance is greater or less than the radius of the sphere, then find the point on the sphere that is closest to the robot current position.
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

    def sphere_volume_restriction(self):
        """ Restrict robot movement to a 3D spherical volumetric boundary

        The radius of the sphere is 0.03 meters with the center at origin (0, 0, 0).

        The desired position of the robot is calculated as follows:
        1. Calculate the distance between the robot current position and origin
        2. If the distance is greater than the radius of the sphere, then calculate the desired position of the robot
              as the point on the sphere that is closest to the robot current position.
        3. If the distance is less than the radius of the sphere, then calculate the desired position of the robot
              as the point on the sphere that is closest to the robot current position.

        """
        if self.current_pose is not None:
            # Calculate the distance between the robot current position and origin
            origin = self.get_parameter('position_origin').value
            pose = self.current_pose
            dx = (pose.position.x - origin[0])
            dy = (pose.position.y - origin[1])
            dz = (pose.position.z - origin[2])
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # If the distance is greater than the radius of the sphere, then calculate the desired position of the robot
            # as the point on the sphere that is closest to the robot current position.
            sphere_radius = self.get_parameter('spherical_boundary_radius').value
            desired_position = pose_msg()
            if distance > sphere_radius:
                # Calculate the desired position of the robot
                desired_position.position.x = sphere_radius * self.current_pose.position.x / distance
                desired_position.position.y = sphere_radius * self.current_pose.position.y / distance
                desired_position.position.z = sphere_radius * self.current_pose.position.z / distance
            else:
                # Calculate the desired position of the robot
                desired_position.position.x = self.current_pose.position.x
                desired_position.position.y = self.current_pose.position.y
                desired_position.position.z = self.current_pose.position.z

            desired_position.orientation.x = self.current_pose.orientation.x
            desired_position.orientation.y = self.current_pose.orientation.y
            desired_position.orientation.z = self.current_pose.orientation.z
            desired_position.orientation.w = self.current_pose.orientation.w

            return desired_position

    def sphere_perimeter_restriction(self):
        """ Restrict robot movement to a 3D spherical perimiter boundary along the surface of the sphere

        The radius of the sphere is 0.03 meters with the center at origin (0, 0, 0).

        """
        if self.current_pose is not None:
            # Calculate the distance between the robot current position and origin
            origin = self.get_parameter('position_origin').value
            pose = self.current_pose
            dx = (pose.position.x - origin[0])
            dy = (pose.position.y - origin[1])
            dz = (pose.position.z - origin[2])
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # If the distance is greater than the radius of the sphere, then calculate the desired position of the robot
            # as the point on the sphere that is closest to the robot current position.
            sphere_radius = self.get_parameter('spherical_boundary_radius').value
            if distance > sphere_radius or distance < sphere_radius:
                # Calculate the desired position of the robot
                desired_position = pose_msg()
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

    def sphere_rotation_restriction(self):
        """ Restrict robot movement and the orientation of the robot to a 3D spherical perimiter boundary along the surface of the sphere, where the angle of rotation is restricted to the vector between the origin and the robot current position.

        Returns:
        ---------
        desired_pose        (geometry_msgs.msg.Pose) : The desired position and orientation of the robot/wrist.
        """
        origin = self.get_parameter('position_origin').value
        if self.current_pose is not None:
            # Calculate the distance between the robot current position and origin
            pose = self.current_pose
            dx = (pose.position.x - origin[0])
            dy = (pose.position.y - origin[1])
            dz = (pose.position.z - origin[2])

            # First calculate the euler angles for the desired orientation.
            # Desired orientation is the pitch and yaw euler angle components using the vector between the origin and
            # the robot's current position on the sphere. Just need the pitch and yaw angles
            roll = 0.00
            pitch = -math.atan2(-dz, math.sqrt(dx ** 2 + dy ** 2))
            yaw = math.atan2(dy, dx)
            # -pi transitions to pi,  Reorient the yaw angle so 0 is at the front of the sphere.
            sign = 1
            if yaw < 0:
                sign = -1
            yaw = - sign*(math.pi - abs(yaw))


            # Limit the angle to within the ranges specified by the parameters
            #roll = max(-math.radians(self.get_parameter('roll.range').value), min(roll, math.radians(self.get_parameter('roll.range').value)))
            #pitch = max(-math.radians(self.get_parameter('pitch.range').value), min(pitch, math.radians(self.get_parameter('pitch.range').value)))
            #yaw = max(-math.radians(self.get_parameter('yaw.range').value), min(yaw, math.radians(self.get_parameter('yaw.range').value)))

            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)

            #self.logger("pitch: {:2.4f}, yaw: {:2.4f}".format(pitch_deg, yaw_deg))

            desired_pose = pose_msg()

            # Convert the pitch and yaw angles to quaternions
            desired_pose.orientation.x, desired_pose.orientation.y, desired_pose.orientation.z, desired_pose.orientation.w = euler_to_quaternion(roll, pitch, yaw)

            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            sphere_radius = self.get_parameter('spherical_boundary_radius').value

            # If the distance is greater or less than the radius of the sphere, then find the point on the sphere that is closest to the robot current position.
            if distance > sphere_radius or distance < sphere_radius:
                temp = [sphere_radius * pose.position.x / distance,
                        sphere_radius * pose.position.y / distance,
                        sphere_radius * pose.position.z / distance]
            else:
                temp = [pose.position.x, pose.position.y, pose.position.z]

            # Limit the x, y, z position of the robot to the sphere boundary such that the rotation angles are within the specified ranges
            # TO-DO

            desired_pose.position.x = temp[0]
            desired_pose.position.y = temp[1]
            desired_pose.position.z = temp[2]


            return desired_pose

    def table_restriction(self):
        """
        Generate a table boundary so the robot cannot pass through the table, effectively limiting its possible movement space to anywhere in the x and y planes but z always has to be greater than or equal to zero

        Workspace range: x: [-0.065, 0.065], y: [-0.065, 0.065], z: [0, 0.065]

        Returns:
        ---------
        desired_position        (geometry_msgs.msg.Point) : The desired position of the robot/wrist.
        desired_orientation (geometry_msgs.msg.Quaternion) : The desired orientation of the robot/wrist.
        """
        if self.current_pose is not None:

            #self.logger("current z: {:2.10f}".format(self.current_pose.position.z))

            # Have to make a new topic, otherwise copying the reference will overwrite values from the current position
            desired_pose = pose_msg()
            desired_pose.position.x = self.current_pose.position.x
            desired_pose.position.y = self.current_pose.position.y
            desired_pose.position.z = self.current_pose.position.z
            desired_pose.orientation.x = self.current_pose.orientation.x
            desired_pose.orientation.y = self.current_pose.orientation.y
            desired_pose.orientation.z = self.current_pose.orientation.z
            desired_pose.orientation.w = self.current_pose.orientation.w

            if self.current_pose.position.z < 0:
                desired_pose.position.z = 0e12
                #self.logger("adjusting z pose from {:2.10f} to 0.00".format(self.current_pose.position.z))

            return desired_pose

    def physics_callback(self):
        """
        Callback function to handle physics and forces at a specified rate.
        """
        if self.current_pose is None and self.previous_pose is None:
            return

        # Create the sigma7 force message at the beginning
        sigma7_force = Sigma7()

        #self.logger("GET RANGE OF MOTION")
        #self.get_range_of_motion()

        # Check to see if we are doing any environment restrictions
        if self.get_parameter('environment_boundary').value == 'wall':
            self.wall_restriction()
        elif self.get_parameter('environment_boundary').value == 'table':
            desired_pose = self.table_restriction()
        elif self.get_parameter('environment_boundary').value == 'origin':
            desired_pose = self.origin_restriction()
        elif self.get_parameter('environment_boundary').value == 'sphere_volume':
            desired_pose = self.sphere_volume_restriction()
        elif self.get_parameter('environment_boundary').value == 'sphere_perimeter':
            desired_pose = self.sphere_perimeter_restriction()
        elif self.get_parameter('environment_boundary').value == 'sphere_rotation':
            desired_pose = self.sphere_rotation_restriction()
        elif self.get_parameter('environment_boundary').value == 'wrist_manipulandum':
            desired_pose = self.wrist_manipulandum_restriction()
            #self.logger()
        else:
            self.logger("unknown environment boundary. Using default origin restriction")
            desired_pose = self.origin_restriction()

        # Check to see if there are any interactions or collisions with the robot current position and objects in the scene
        # If there are, then calculate the resisting force and torque to restrict motion in undesired directions


        # Calculate the gripper force
        grip_force = self.calculate_gripper_force()

        # Extract position and orientation from the current pose
        cur_pose = self.current_pose
        #(roll, pitch, yaw) = quaternion_to_euler(cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w)
        # self.logger("roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(roll, pitch, yaw))

        # Calculate the resisting force to restrict motion in undesired directions
        resist_force = self.calculate_resisting_force(cur_pose.position, cur_pose.orientation,
                                                      self.current_position_velocity, self.current_rotation_velocity,
                                                      desired_pose.position, desired_pose.orientation)

        # Combine the translational forces, torques, and gripper force into a single Sigma7 message
        sigma7_force.force.x = resist_force.force.x
        sigma7_force.force.y = resist_force.force.y
        sigma7_force.force.z = resist_force.force.z
        sigma7_force.torque.x = resist_force.torque.x
        sigma7_force.torque.y = resist_force.torque.y
        sigma7_force.torque.z = resist_force.torque.z
        sigma7_force.gripper.data = grip_force.data
        # self.logger("Griper current position: {:2.6f}".format(self.gripper_position))
        #self.logger("Gripper force: {:2.12f}".format(sigma7_force.gripper.data))

        # roll, pitch, yaw = quaternion_to_euler(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

        #self.logger(" Forces: x: {:2.6f}N, y: {:2.6f}N, z: {:2.6f}N".format(resist_force.force.x, resist_force.force.y, resist_force.force.z))
        #self.logger(" Torques: roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(resist_force.torque.x, resist_force.torque.y, resist_force.torque.z))
        self.publish_sigma7_forces(sigma7_force)
        # self.publish_resisting_wrench(resist_force)

    def calculate_gripper_force(self):
        """
        Calculate the gripper force to be applied.

        Return:
        ----------
        gripper_force (std_msgs.msg.Float64) : The gripper force to be applied.
        """
        gripper_force = Float64()
        gripper_force.data = 0.0

        #if self.gripper_position is not None and self.gripper_position < 0.10:
        #    gripper_force.data = self.get_parameter('gripper_stiffness').value * (0.10 - self.gripper_position)

        return gripper_force

    def calculate_resisting_force(self, current_position, current_orientation, current_position_velocity, current_rotation_velocity, desired_position, desired_orientation):
        """
        Calculate the resisting force and torque to restrict motion in undesired directions.

        Parameters:
        -----------
        current_position         (geometry_msgs.msg.Point) : The current position of the robot/wrist.
        current_orientation (geometry_msgs.msg.Quaternion) : The current orientation of the robot/wrist.
        desired_position         (geometry_msgs.msg.Point) : The desired position of the robot/wrist.
        desired_orientation (geometry_msgs.msg.Quaternion) : The desired orientation of the robot/wrist.

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

        if (current_position is None
                or current_orientation is None
                or current_position_velocity is None
                or current_rotation_velocity is None
                or desired_position is None
                or desired_orientation is None):
            return force_msg

        if self.get_parameter('enable_position_restriction').value:
            # Calculate the difference between the robot's current position and desired position
            dp_x = px_to_m * (desired_position.x - current_position.x)
            dp_y = py_to_m * (desired_position.y - current_position.y)
            dp_z = pz_to_m * (desired_position.z - current_position.z)
            #self.logger("dz: {:2.6f}".format(dp_z))

            # Calculate the force needed to keep x, y, z positions to zero. Use the spring damping model.
            # F = -k * dx - b * dx/dt
            v = current_position_velocity
            force_msg.force.x = (
                    self.get_parameter('x.stiffness').value * dp_x - self.get_parameter('x.damping').value * v[0])
            force_msg.force.y = (
                    self.get_parameter('y.stiffness').value * dp_y - self.get_parameter('y.damping').value * v[1])
            force_msg.force.z = (
                    self.get_parameter('z.stiffness').value * dp_z - self.get_parameter('z.damping').value * v[2])

        if self.get_parameter('enable_orientation_restriction').value:
            # Calculate the difference between the robot's current orientation and desired orientation
            (current_roll, current_pitch, current_yaw) = quaternion_to_euler(current_orientation.x,
                                                                             current_orientation.y,
                                                                             current_orientation.z,
                                                                             current_orientation.w)
            (desired_roll, desired_pitch, desired_yaw) = quaternion_to_euler(desired_orientation.x,
                                                                             desired_orientation.y,
                                                                             desired_orientation.z,
                                                                             desired_orientation.w)

            #self.logger("desired roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(desired_roll, desired_pitch, desired_yaw))
            #self.logger("current roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(current_roll, current_pitch, current_yaw))

            dr = desired_roll - current_roll
            dp = desired_pitch - current_pitch
            dy = desired_yaw - current_yaw

            #self.logger("roll: {:2.4f}, pitch: {:2.4f}, yaw: {:2.4f}".format(dr, dp, dy))


            # Calculate the torque needed to keep roll, pitch, yaw to zero. Use PD control.
            dtheta = current_rotation_velocity
            force_msg.torque.x = (self.get_parameter('roll.kp').value * dr - self.get_parameter(
                'roll.kd').value * dtheta[0])
            force_msg.torque.y = (self.get_parameter('pitch.kp').value * dp - self.get_parameter(
                'pitch.kd').value * dtheta[1])
            force_msg.torque.z = (self.get_parameter('yaw.kp').value * dy - self.get_parameter(
                'yaw.kd').value * dtheta[2])

        return force_msg

    def publish_resisting_wrench(self, resisting_force):
        """
        Publish the resisting wrench command.

        :param resisting_force: The resisting force to be applied as a wrench.
        """
        self.wrench_publisher.publish(resisting_force)

    def publish_sigma7_forces(self, sigma7_force):
        """
        Publish the sigma7 force command.

        :param sigma7_force: The sigma7 force to be applied.
        """
        self.sigma7_force_publisher.publish(sigma7_force)

    def get_range_of_motion(self):
        """ Collects the range of motion of the user for the 3 rotational axes of the wrist and attempts to determine
        the location of each axis of rotation from the intersection of planes. For each step, assume that the "plane"
        references are the 2D plane in the x and z axes. The process is as follows:

        1. With the user's wrist in a neutral position, determine the position of the handle that is most comfortable to the user for grip and save the position as origin
        2. Instruct the user to perform a wrist flexion rotation at 90 degrees and record the position and orientation of the handle, saving the flex-plane. Determine where the flex-plane and x-z-origin plane intersect, and save this position as the flexion/extension axis.
        3. Instruct the user to perform a wrist radial deviation rotation at 90 degrees and record the position and orientation of the handle, saving the radial-plane. Determine where the radial-plane and x-y-origin plane intersect, and save this position as the radial/ulnar deviation axis.
        4. Apply the "wrist_manipulandum_restriction" function to restrict robot movement to the wrist manipulandum workspace using the calculated axes of rotation.
        5. Ask the user if the axes are correct and feel okay to use. If not, repeat the process.

        """
        origin_found=False
        while not origin_found:
            # Check if the user is ready to begin
            user_ready = input("Are you ready to begin? (y/n): ")
            if user_ready == 'y':
                while True:
                    origin = [self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z]
                    self.logger("Please move the handle to the most comfortable position for grip. Press ENTER to continue.")
                    self.logger("Current origin: x: {:2.4f}, y: {:2.4f}, z: {:2.4f}".format(origin[0], origin[1], origin[2]))
                    # Determine the position of the handle that is most comfortable to the user for grip and save the position as origin
                    if keyboard.read_key == '':
                        self.logger("Origin set to x: {:2.4f}, y: {:2.4f}, z: {:2.4f}".format(origin[0], origin[1], origin[2]))
                        break
            else:
                continue

def main(args=None):
    """
    Main function to create and spin the ForceDimensionNode.
    """
    rclpy.init(args=args)
    node = ForceDimensionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
