#!/usr/bin/env python3

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose as pose_msg
from geometry_msgs.msg import Twist as twist_msg
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Quaternion
from example_interfaces.msg import Float64
from geometry_msgs.msg import Vector3, Point
from haptic_device_interfaces.msg import Sigma7
from ros2_utilities_py import ros2serviceclient as r2sc
from .objects import Table, Cube, Plane

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

    return [X, Y, Z]

class ForceInit:
    """ Class to handle the initialization of the force to a desired value.
    """
    def __init__(self, max_stiffness=2000):
        self.max_stiffness = max_stiffness
        self.nth = 0
        self.n1 = 0
        self.n2 = 0.1
        self.start = time.time()
        self.timer = None # placeholder for ROS2 timer


class TransformsNode(Node):
    def __init__(self,
                 name="blocks_node",
                 environment="table_with_blocks",
                 has_gripper=True,
                 ):
        """
        Initialize the TransformsNode class.
        """
        super().__init__(name)
        self.has_gripper = has_gripper
        self.environment = environment

        # We're setting up an environment where there's a table surface at z=-0.24. Robot movements will be restricted to not being able to pass through the table, but if the robot position z < 0 then it can increase and pass through the table.
        # The robot gripper thumb and index finger positions can be acquired from the /robot/feedback/gripper/thumb_position and /robot/feedback/gripper/index_position topics.
        # The robot's current position and orientation can be acquired from the /robot/feedback/pose topic.

        self.initialize_parameters()
        self.initialize_subscribers()
        self.initialize_publishers()

        # Internal robot state
        self.tool_pose = [None, None]
        self.adjusted_tool_pose = [None, None]
        self.prev_tool_pose = [None, None]
        self.contact_pose = [None, None]
        self.small_cube = Cube('SmallCube', 0.02, 0.02, 0.02, logger=self.logger) # 40mm cube
        #self.logger("cube type: {}".format(self.small_cube.cube_pose))
        self.big_cube = Cube('BigCube', 0.03, 0.03, 0.03, logger=self.logger) # 60mm cube
        self.current_rotation_velocity = None
        self.current_position_velocity = None
        self.gripper_position = 0.000
        self.gripper_force = 0.000
        #self.t_prev = time.time()  # Store the previous time

        # Environment
        #self.table = Table(10, 10, 0.0) # huge table at origin with default stiffness of 2000 N/m
        self.plane = Plane('z', 0.001) # Create a plane on the xy-axis at z=0.0, with default stiffness of 2000 N/m
        self.table_stiffness = None
        self.force_initializer = ForceInit(max_stiffness=self.get_parameter('table.stiffness').value)
        # Update the cube sizes to match the initialized ones
        small_size = self.get_parameter('small_cube_size').value
        self.small_cube_size_publisher.publish(Vector3(x=small_size[0], y=small_size[1], z=small_size[2]))
        big_size = self.get_parameter('big_cube_size').value
        self.big_cube_size_publisher.publish(Vector3(x=big_size[0], y=big_size[1], z=big_size[2]))

        # Create a timer to handle physics and forces at a specified rate
        self.timer = self.create_timer(self.get_parameter('sample_rate').value, self.physics_callback)
        self.init_force_timer = None # For later during the force initialization stage

        # Add an "on set" parameter callback.
        #self.add_on_set_parameters_callback(self._parameters_callback)

        self.logger("Environment boundary: {}".format(self.environment))

        # Create a service client connection to the force_dimension node
        self.force_dimension_client = r2sc.ROS2ServiceClient(parent_node=self,
                                                             external_node_name='robot/sigma7',
                                                             tag='sigma7_client')
        # Also want to change this node's parameters
        self.blocks_node_client = r2sc.ROS2ServiceClient(self, 'robot/blocks_node', 'blocks_node_client')

        # Enable forces from the force dimension client (and wait until the robot node initializes)
        time.sleep(1)
        self.initialize_forces()

    def logger(self, msg):
        self.get_logger().info(msg)

    def initialize_parameters(self):
        """ Define all parameters in this function.
        """
        default_transform = [+1.0, +0.0, +0.0,
                             +0.0, +1.0, +0.0,
                             +0.0, +0.0, +1.0]
        parameters = [
            # ROS2 specific topic names
            ('robot_node_name', 'robot/manipulandum'),
            ('robot_pose_topic', '/robot/feedback/pose'),
            ('robot_twist_topic', '/robot/feedback/twist'),
            ('gripper_angle_topic', '/robot/feedback/gripper_angle'),
            ('wrench_command_topic', '/robot/command/wrench'),
            ('sigma7_force_command_topic', '/robot/command/sigma7_force'),
            ('small_cube_pose_topic', '/SmallCube/pose'),
            ('small_cube_size_topic', '/SmallCube/size'),
            ('small_cube_size', [0.025, 0.025, 0.025]),
            ('big_cube_pose_topic', '/BigCube/pose'),
            ('big_cube_size_topic', '/BigCube/size'),
            ('big_cube_size', [0.05, 0.05, 0.05]),
            ('thumb.radius', 0.001),
        ]
        if self.has_gripper:
            parameters += [
                # For robots that have a gripper
                ('thumb_pose_feedback_topic', '/robot/feedback/gripper/thumb_position'),
                ('index_pose_feedback_topic', '/robot/feedback/gripper/index_position'),
                # We also want to publish the transform of the positions to the environment
                ('thumb_pos_command_topic', '/environment/thumb/position'),
                ('index_pos_command_topic', '/environment/index/position'),
                ('finger_pos_transform', [+1.0, +1.0, +1.0]),
            ]
        parameters += [
            # Environment-specific parameters like stiffness and damping parameters to model the spring-damper system
            ('sample_rate', 0.001),
            ('position_origin', [0.0, 0.0, 0.0]),
            ('orientation_origin', [0.0, 0.0, 0.0, 1.0]),
            ('box.stiffness', 4000.0),
            ('table.stiffness', 4000.0),
            ('box.damping', 0.01),
            ('table.damping', 0.01),

            # Rotation limits for the wrist (in degrees)
            #('roll.range', 90.0),
            #('pitch.range', 90.0),
            #('yaw.range', 90.0),
            #('transform', default_transform),
        ]
        self.declare_parameters(
            namespace='',
            parameters=parameters,
        )

    def initialize_subscribers(self):

        self.pose_subscription = self.create_subscription(pose_msg, self.get_parameter('robot_pose_topic').value,
                                                         self.pose_callback,  qos_profile=DEFAULT_QOS)

        self.twist_subscription = self.create_subscription(twist_msg, self.get_parameter('robot_twist_topic').value,
                                                         self.twist_callback,  qos_profile=DEFAULT_QOS)
        if self.has_gripper:
            self.thumb_subscription = self.create_subscription(Point, self.get_parameter('thumb_pose_feedback_topic').value,
                                                              self.thumb_pose_callback, qos_profile=DEFAULT_QOS)
            self.index_subscription = self.create_subscription(Point, self.get_parameter('index_pose_feedback_topic').value,
                                                                self.index_pose_callback, qos_profile=DEFAULT_QOS)

        self.small_cube_pose_subscription = self.create_subscription(pose_msg, self.get_parameter('small_cube_pose_topic').value,
                                                                self.small_cube_pose_callback, qos_profile=DEFAULT_QOS)

        self.big_cube_pose_subscription = self.create_subscription(pose_msg, self.get_parameter('big_cube_pose_topic').value,
                                                                self.big_cube_pose_callback, qos_profile=DEFAULT_QOS)

    def initialize_publishers(self):
        """
        Initialize the publisher to publish the resisting wrench command.
        """
        self.wrench_publisher = self.create_publisher(Wrench, self.get_parameter('wrench_command_topic').value,
                                                      qos_profile=DEFAULT_QOS)

        self.sigma7_force_publisher = self.create_publisher(Sigma7, self.get_parameter('sigma7_force_command_topic').value,
                                                        qos_profile=DEFAULT_QOS)

        self.small_cube_size_publisher = self.create_publisher(Vector3, self.get_parameter('small_cube_size_topic').value,
                                                                qos_profile=DEFAULT_QOS)

        self.big_cube_size_publisher = self.create_publisher(Vector3, self.get_parameter('big_cube_size_topic').value,
                                                                qos_profile=DEFAULT_QOS)

        if self.has_gripper:
            self.thumb_pos_publisher = self.create_publisher(Point, self.get_parameter('thumb_pos_command_topic').value,
                                                             qos_profile=DEFAULT_QOS)
            self.index_pos_publisher = self.create_publisher(Point, self.get_parameter('index_pos_command_topic').value,
                                                             qos_profile=DEFAULT_QOS)

    def initialize_forces(self):
        """ Initialize the force dimension client to enable forces.
        """
        self.force_dimension_client.set_parameter('enable_force', False)

        # We want to be careful with the stiffness of the table and the cube, so we will save the current max stiffness set the stiffness to a low value
        self.blocks_node_client.set_parameter('table.stiffness', 0.0)
        self.publish_sigma7_forces(Sigma7()) # Should default of zero forces

        # Slowly ramp up the force to the desired value
        self.force_dimension_client.set_parameter('enable_force', True)
        self.force_initializer.timer = self.create_timer(0.2, self.init_force_fib_callback)

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

    def pose_callback(self, msg):
        """
        Callback function to process received pose messages.

        Parameters:
        -----------
        msg (geometry_msgs.msg.Pose) : The received Pose message.
        """
        # Store the current position and orientation
        self.current_pose = msg

    def twist_callback(self, msg):
        """
        Callback function to process received twist messages.

        Parameters:
        -----------
        msg (geometry_msgs.msg.Twist) : The received Twist message.
        """
        self.current_twist = msg

    def gripper_callback(self, msg):
        """
        Callback function to process received gripper messages.

        Parameters:
        -----------
        msg (std_msgs.msg.Float64) : The received gripper message.
        """
        self.gripper_position = msg.data

    def thumb_pose_callback(self, msg):
        """
        Callback function that processes the thumb position

        Parameters:
        -----------
        msg (geometry_msgs.msg.Point) : The received Point message.
        """
        if self.tool_pose[0] is None:
            self.tool_pose[0] = msg
            self.prev_tool_pose[0] = msg
        else:
            self.prev_tool_pose[0] = self.tool_pose[0]
            self.tool_pose[0] = msg

        if self.adjusted_tool_pose[0] is None:
            self.adjusted_tool_pose[0] = msg

        # We also want to publish the transform of the thumb position too, after considering all environmental restrictions
        finger_transform = self.get_parameter('finger_pos_transform').value
        pos = Point()
        pos.x = finger_transform[0] * self.adjusted_tool_pose[0].x
        pos.y = finger_transform[1] * self.adjusted_tool_pose[0].y
        pos.z = finger_transform[2] * self.adjusted_tool_pose[0].z
        #self.logger('z position: {}'.format(pos.z))
        self.thumb_pos_publisher.publish(pos)

    def index_pose_callback(self, msg):
        """
        Callback function that processes the index position

        Parameters:
        -----------
        msg (geometry_msgs.msg.Point) : The received Point message.
        """
        if self.tool_pose[1] is None:
            self.tool_pose[1] = msg
            self.prev_tool_pose[1] = msg
        else:
            self.prev_tool_pose[1] = self.tool_pose[1]
            self.tool_pose[1] = msg

        if self.adjusted_tool_pose[1] is None:
            self.adjusted_tool_pose[1] = msg

        # We also want to publish the transform of the index position too
        finger_transform = self.get_parameter('finger_pos_transform').value
        pos = Point()
        pos.x = finger_transform[0] * self.adjusted_tool_pose[1].x
        pos.y = finger_transform[1] * self.adjusted_tool_pose[1].y
        pos.z = finger_transform[2] * self.adjusted_tool_pose[1].z
        self.index_pos_publisher.publish(pos)

    def small_cube_pose_callback(self, msg):
        """
        Callback function that processes the small cube position into a list

        Parameters:
        -----------
        msg (geometry_msgs.msg.Pose) : The received Pose message.
        """
        x, y, z = msg.position.x, msg.position.y, msg.position.z
        q1, q2, q3, q4 = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.small_cube.pose = [x, y, z, q1, q2, q3, q4]

    def big_cube_pose_callback(self, msg):
        """
        Callback function that processes the big cube position

        Parameters:
        -----------
        msg (geometry_msgs.msg.Pose) : The received Pose message.
        """
        x, y, z = msg.position.x, msg.position.y, msg.position.z
        q1, q2, q3, q4 = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.big_cube_pose = [x, y, z, q1, q2, q3, q4]

    def table_restriction_old(self, current_pose=None, previous_pose=None):
        """
        Generate a table boundary so the robot cannot pass through the table, effectively limiting its possible movement space to anywhere in the x and y planes but z always has to be greater than or equal to zero

        Workspace range: x: [-2.5, 2.5], y: [-2.5, 2.5], z: [0.0, 0.0]

        Parameters:
        -----------
        current_pose           (list) : A list of the current pose of the tool tip(s) (optional)
        previous_pose          (list) : A list of the previous pose of the tool tip(s) (optional)

        Returns:
        ---------
        new_force              (list) : The adjusted forces of the tool tip(s).
        gripper_force_magnitude (num) : The magnitude of the force applied to the gripper.
        """
        if current_pose is None:
            current_pose = self.tool_pose

        if previous_pose is None:
            previous_pose = self.prev_tool_pose

        # Compute the interaction between the table and each tool
        force_tool = []
        #current_pos_i = []
        #previous_pos_i = []
        for i, pos in enumerate(self.tool_pose):

            # Check if the message type of pos is either pose_msg() or Point()
            if type(pos) == pose_msg:
                current_pos_i = [pos.position.x, pos.position.y, pos.position.z]
                previous_pos_i = [previous_pose[i].position.x, previous_pose[i].position.y, previous_pose[i].position.z]

            elif type(pos) == Point:
                current_pos_i = [pos.x, pos.y, pos.z]
                previous_pos_i = [previous_pose[i].x, previous_pose[i].y, previous_pose[i].z]

            #finger = "thumb" if i == 0 else "index"
            #self.logger("{}: {}".format(finger, current_pos_i))
            # If the previous tool position was above the table and the current tool position is below, then the tool tip is in a colliding state
            over_table = self.table.over_table(current_pos_i)
            under_table = self.table.under_table(previous_pos_i)
            self.logger("currently over table: {} | previously under table: {}".format(over_table, under_table))
            #if ( self.table.over_table(current_pos_i) and self.table.under_table(previous_pos_i) ):
            if over_table and under_table:
                self.table.is_colliding = True
                self.logger("Colliding")
            else:
                # If the previous and current pose were under the table, then we don't need to apply any kind of boundaries
                # If the current tool position is above the table but the previous tool position was below, we still don't need to do anything
                self.table.is_colliding = False

               # If the tool is colliding with the table, then compute the resisting force
            penetration_distance = 0.0
            if self.table.is_colliding:
                # compute the penetration distance between the tool and the table
                def normalize(x):
                    x = np.asarray(x)
                    return (x - x.min()) / (np.ptp(x))

                #pose_list = [pos.position.x, pos.position.y, pos.position.z]
                direction = normalize(np.subtract(np.array(current_pos_i), np.array(self.table.origin)))

                penetration_distance = np.linalg.norm(np.array(current_pos_i), np.array(self.table.origin)) # - tool_radius # Set this to a non-zero value in the future

            # Compute the penetration force
            if penetration_distance < 0.0:
                force_tool.append(-1.0 * self.get_parameter('table.stiffness').value * penetration_distance * direction)
            else:
                force_tool.append([0.0, 0.0, 0.0])

            #self.logger("force_tool: {}".format(force_tool[i]))

        # If the haptic device has a gripper then compute the projected force onto each finger, otherwise just return the force
        gripper_force_magnitude = 0.0
        if not self.has_gripper:
            new_force = force_tool[0]
        else:
            new_force = force_tool[0] + force_tool[1]

            # Check if the message type of pos is either pose_msg() or Point()
            if type(current_pose[0]) == pose_msg:
                thumb_pose = [current_pose[0].position.x, current_pose[0].position.y, current_pose[0].position.z]
                finger_pose = [current_pose[1].position.x, current_pose[1].position.y, current_pose[1].position.z]

            elif type(current_pose[0]) == Point:
                thumb_pose = np.array([current_pose[0].x, current_pose[0].y, current_pose[0].z])
                finger_pose = np.array([current_pose[1].x, current_pose[1].y, current_pose[1].z])

            #thumb_pose = [current_pose[0].position.x, current_pose[0].position.y, current_pose[0].position.z]
            #index_pose = [current_pose[1].position.x, current_pose[1].position.y, current_pose[1].position.z]
            gripper_direction = np.subtract(thumb_pose, finger_pose)

            if np.linalg.norm(gripper_direction) > 0.00001:
                # Project the mobile gripper finger force (force_global[1]) onto the gripper opening vector (gripper_direction).
                gripper_direction /= np.linalg.norm(gripper_direction)
                gripper_force = np.dot(force_tool[1], gripper_direction) / np.linalg.norm(gripper_direction)**2 * gripper_direction
                gripper_force_magnitude = np.linalg.norm(gripper_force)

                # Compute the direction of the force based on the angle between the gripper
                # force vector (gripper_force) and the gripper opening vector (gripper_direction)
                if np.linalg.norm(new_force) > 0.001:
                    cos_angle = np.dot(gripper_direction, gripper_force) / (
                                np.linalg.norm(gripper_direction) * np.linalg.norm(gripper_force))
                    cos_angle = min(1.0, cos_angle)
                    cos_angle = max(-1.0, cos_angle)
                    angle = np.arccos(cos_angle)
                    if angle > np.pi / 2.0 or angle < -np.pi / 2.0:
                        gripper_force_magnitude = -gripper_force_magnitude

            return new_force, gripper_force_magnitude

    def table_restriction(self, current_pose=None, previous_pose=None):
        """ Check if the height of the robot is above the table and if it is not or is colliding, apply a z force to push it up.
        """
        if current_pose is None:
            current_pose = self.tool_pose

        if previous_pose is None:
            previous_pose = self.prev_tool_pose

        # Compute the interaction between the table and each tool
        force_tool = []
        for i, pos in enumerate(current_pose):

            # Check if the message type of pos is either pose_msg() or Point()
            if type(pos) == pose_msg:
                current_pos_i = [pos.position.x, pos.position.y, pos.position.z]
            elif type(pos) == Point:
                current_pos_i = [pos.x, pos.y, pos.z]

            # If the previous tool position was above the table and the current tool position is below, then the tool
            # tip is in a colliding state. If contact is true, remember the previous position
            over_table = self.plane.over_plane(current_pos_i)
            contact, penetration_distance_vector = self.plane.through_plane(current_pos_i, self.get_parameter('thumb.radius').value)

            # If contact is true, then depending on which element is less than or equal to zero, save that element to
            # the contact pose. We want to update the displayed position of the tool to be with respect to the physical
            # forces it is feeling
            if contact and self.contact_pose[i] is None:
                self.contact_pose[i] = current_pose[i]
            elif not contact and self.contact_pose[i] is not None:
                self.contact_pose[i] = None

            # Adjust the pose if contact is made with the table. We will do this here instead of the higher functions to avoid
            # having to recompute the adjusted pose for each tool tip, as the rendering will show a different position
            self.adjusted_tool_pose[i] = Point()
            self.adjusted_tool_pose[i].x = current_pos_i[0]
            self.adjusted_tool_pose[i].y = current_pos_i[1]
            self.adjusted_tool_pose[i].z = current_pos_i[2]

            # Finally adapt the right position of the tool tip if it is in contact with the table
            if contact:
                if self.plane.axis == 'x':
                    self.adjusted_tool_pose[i].x = self.contact_pose[i].x
                elif self.plane.axis == 'y':
                    self.adjusted_tool_pose[i].y = self.contact_pose[i].y
                if self.plane.axis == 'z':
                    self.adjusted_tool_pose[i].z = self.contact_pose[i].z

            if sum(penetration_distance_vector) < 0.0:
                new_force = [-1 * self.get_parameter('table.stiffness').value * i for i in penetration_distance_vector]
                force_tool.append(new_force)
            else:
                force_tool.append([0.0, 0.0, 0.0])


        # If the haptic device has a gripper then compute the projected force onto each finger, otherwise just return the force
        gripper_force_magnitude = 0.0
        if not self.has_gripper:
            new_force = force_tool[0]
        else:
            new_force = list(np.array(force_tool[0]) + np.array(force_tool[1]))

        #self.logger("new force: {}".format(new_force))
        return new_force, gripper_force_magnitude

    def cube_interaction(self, parent_cube, current_pose=None, previous_pose=None):
        """ Function that calculates the interaction forces between the tool tip(s) and the cube in the environment.

        Parameters:
        -----------
        parent_cube (Cube) : The cube object in the environment.

        Returns:
        ---------
        new_force              (list) : The adjusted forces of the tool tip(s).
        gripper_force_magnitude (num) : The magnitude of the force applied to the gripper.

        """

        if current_pose is None:
            current_pose = self.tool_pose

        if previous_pose is None:
            previous_pose = self.prev_tool_pose

        # Compute the interaction between the cube and each tool
        force_tool = []
        for i, pos in enumerate(current_pose):
            # Check if the message type of pos is either pose_msg() or Point()
            if type(pos) == pose_msg:
                current_pos_i = [pos.position.x, pos.position.y, pos.position.z]
            elif type(pos) == Point:
                current_pos_i = [pos.x, pos.y, pos.z]

            # Show me the 8 cube corners
            #self.logger("type: {}".format(parent_cube.cube_pose))
            #cube_corners = parent_cube.cube_corners()

            # Find the points that are outside the cube
            #is_inside, distances = parent_cube.is_inside_cube(current_pos_i)

            #if i==0:
            #    self.logger("is inside cube: {} | {}".format(is_inside, distances))
            #    self.logger("box pos: {} | thumb dist: {}".format([round(i,3) for i in self.small_cube.pose[0:3]], [round(i,3) for i in  distances2]))

        #self.logger("inside points: {}".format(outside_points))

                # Compute the interaction between the cube and each tool
        #for i, pos in enumerate(current_pose):
        #    tool_pose[i] = [current_pose[i].position.x,
        #                        current_pose[i].position.y,
        #                        current_pose[i].position.z]#

        #    # Compute the position of the tool in the local coordinates of the cube
        #    tool_local_position[i] = cube_orientation.transpose() * (tool_pose[i] - cube_position)



            # Case when both of the fingers are making contact with the cube
        #    finger_distance = np.linalg.norm(thumb_position - index_position)

        return [0.0, 0.0, 0.0], 0.0

    def physics_callback(self):
        """
        Callback function to handle the environment physics and forces.
        """
        #if True:
        #    return

        if len(self.tool_pose) <= 0 or self.tool_pose[0] is None:
            return

        # Check to see if there are any interactions or collisions with the robot current position and objects in the scene
        # If there are, then calculate the resisting force and torque to restrict motion in undesired directions
        # For lack of a better term, I'm running the "collision engine" here to check for any interactions between the
        # robot and the objects in the scene
        force, torque, gripper = self.collision_engine()

        # Create the sigma7 force message
        msg = Sigma7()

        # Combine the translational forces, torques, and gripper force into a single Sigma7 message
        msg.force.x = force[0]
        msg.force.y = force[1]
        msg.force.z = force[2]
        msg.torque.x = torque[0]
        msg.torque.y = torque[1]
        msg.torque.z = torque[2]
        msg.gripper.data = gripper

        # Publish the sigma7 force command
        self.publish_sigma7_forces(msg)

    def collision_engine(self):
        """
        Calculate the haptic and resistive forces to be applied to the end effector based on object and environment
        interactions.

        Return:
        ----------
        resist_force (geometry_msgs.msg.Wrench) : The resisting force and torque to be applied.
        grip_force (std_msgs.msg.Float64) : The gripper force to be applied.
        """

        # First we create a new instance of the tool pose by setting the adjusted tool tip to match the current tool tip
        # (no forces acting on it yet)
        #self.adjusted_tool_pose = self.tool_pose

        # Calculate the resistance forces from the table. This should return a list of ROS2 messgages with interaction
        # forces between the tool tip(s) and the table. The adjusted tool tip pose is also updated here.
        table_force, table_grip = self.table_restriction()
        #self.logger("table force: {}".format(table_force))

        # Next we want to calculate the resistance forces from the interaction between the tool tips and the cubes
        small_cube_force, small_cube_grip = self.cube_interaction(self.small_cube)
        #big_cube_force, big_cube_grip = self.cube_interaction(self.big_cube_pose)

        # Let's just make sure the right position and orientation are reported with the small cube
        # convert from quaternion to euler angles of the small cube
        small_cube_orientation = [self.small_cube.pose[3], self.small_cube.pose[4], self.small_cube.pose[5], self.small_cube.pose[6]]
        small_cube_euler = quaternion_to_euler(small_cube_orientation[0], small_cube_orientation[1], small_cube_orientation[2], small_cube_orientation[3])
        #self.logger("small cube pose: {}".format([round(i, 3) for i in self.small_cube.pose[0:3] + small_cube_euler]))






        # Sum the forces and torques from the table and the cubes
        #resist_force = table_force + small_cube_force + big_cube_force
        #grip_force = table_grip + small_cube_grip + big_cube_grip

        resist_force = table_force
        resist_torque = [0.0, 0.0, 0.0]
        grip_force = table_grip

        return resist_force, resist_torque, grip_force

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
            [current_roll, current_pitch, current_yaw] = quaternion_to_euler(current_orientation.x,
                                                                             current_orientation.y,
                                                                             current_orientation.z,
                                                                             current_orientation.w)
            [desired_roll, desired_pitch, desired_yaw] = quaternion_to_euler(desired_orientation.x,
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

def main(args=None):
    """
    Main function to create and spin the TransformsNode.
    """
    rclpy.init(args=args)
    node = TransformsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
