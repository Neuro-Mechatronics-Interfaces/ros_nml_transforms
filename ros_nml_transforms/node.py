""" A ROS2 node that continuously applies linear transformations to position 
    and force vectors transmitted as messages across specific topics.

Examples
--------

>>>

"""

# Copyright 2022-2023 Carnegie Mellon University Neuromechatronics Lab (a.whit)
# 
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# 
# Contact: a.whit (nml@whit.contact)


# Import numpy.
import numpy

# Import ROS2.
import rclpy
import rclpy.node
import rclpy.qos
import rosidl_runtime_py

# Local imports.
from ros_nml_transforms.msg import position_message
from ros_nml_transforms.msg import force_message


# Declare default quality-of-service settings for ROS2.
DEFAULT_QOS = rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value


# Define the node class.
class Node(rclpy.node.Node):
    
    def __init__(self, *args, 
                       node_name='transforms', 
                       namespace='robot', 
                       verbose=False,
                       **kwargs):
        super().__init__(*args, node_name=node_name, **kwargs)
        
        self.current_effector_position = position_message()
        self.initialize_parameters()

        # Initializing list to store N force inputs
        self.N_inputs = self.get_parameter('n_force_inputs').value
        self.force_cmd = [[0.0, 0.0, 0.0] for _ in range(self.N_inputs)]  
        self.verbose = verbose

        self.initialize_publishers()
        self.initialize_subscriptions()
        
        self.logger("'{}' node created".format(node_name))
        
    def logger(self, msg=""):
        self.get_logger().info(msg)

        
    def initialize_parameters(self):
        default_transform = [+1.0, +0.0, +0.0,
                             +0.0, +1.0, +0.0,
                             +0.0, +0.0, +1.0]
        self.declare_parameter('cursor.position_transform', default_transform)
        self.declare_parameter('cursor.force_transform', default_transform)
        self.declare_parameter('n_force_inputs', 2)
        
    def initialize_publishers(self):
        
        # Initialize a publisher for cursor position messages.
        kwargs = dict(topic='/cursor/position',
                      msg_type=position_message,
                      qos_profile=DEFAULT_QOS) 
                      #rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.position_publisher = self.create_publisher(**kwargs)
        
        # Initialize a publisher for robot force messages.
        kwargs = dict(topic='command/force',
                      msg_type=force_message,
                      qos_profile=DEFAULT_QOS)
        self.force_publisher = self.create_publisher(**kwargs)
        
    def initialize_subscriptions(self):
        
        # Initialize a subscription for the robot position.
        kwargs = dict(topic='feedback/position',
                      msg_type=position_message,
                      callback=self.publish_cursor_position,
                      qos_profile=DEFAULT_QOS)
                      #rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(**kwargs)
        
        # Initialize a subscription for the cursor force.
        #kwargs = dict(topic='/cursor/force',
        #              msg_type=force_message,
        #              callback=self.publish_cursor_force,
        #              qos_profile=DEFAULT_QOS)
        #self.create_subscription(**kwargs)
        
        # Instead of one topic listening to a force input and publishing the force command to the 
        # '/command/force' topic, we can use the summation of force inputs and update the 
        # corresponding force command with each change. Doing it this way, we can have a force command
        # from an attractor node, an independent force perturbation command, potentially a 3rd
        # input from object colisions, and others
        self.input_sub = []
        #N_inputs = self.get_parameter('n_force_inputs').value
        for i in range(self.N_inputs):
            def create_callback(idx):
               return lambda msg, idx = idx: self.input_force_cb(msg, idx)
               
            topic_str = 'force/input'+str(i+1)
            kwargs = dict(topic=topic_str,
                          msg_type=force_message,
                          callback=create_callback(i),
                          qos_profile=DEFAULT_QOS)
            self.input_sub.append(self.create_subscription(**kwargs))
        
    def publish_cursor_position(self, position):
        
        # Record the current position.
        self.current_effector_position = position
        
        # Convert the robot position to cursor position.
        p_r = numpy.array([position.x, position.y, position.z])
        t   = numpy.array(self.get_parameter('cursor.position_transform').value)
        T   = t.reshape((3, 3))
        p_c = T @ p_r
        
        # Publish the position.
        kwargs = dict(zip(['x', 'y', 'z'], p_c))
        message = position_message(**kwargs) #x=p_c[0], y=p_c[1], z=p_c[2])
        self.position_publisher.publish(message)
        
    def input_force_cb(self, force, idx):
    
        # Update force data for input by index
        self.force_cmd[idx] = [force.x, force.y, force.z]
        
        if self.verbose: self.logger("current force inputs:")
        if self.verbose: self.logger("{}".format([i for i in self.force_cmd]))
        
        # Send new force message
        self.publish_sum_of_forces()
        
    def publish_sum_of_forces(self):
        # Calculate summation of forces
        f_x = sum(force[0] for force in self.force_cmd)  # Sum of x forces
        f_y = sum(force[1] for force in self.force_cmd)  # Sum of y forces
        f_z = sum(force[2] for force in self.force_cmd)  # Sum of z forces
        
        if self.verbose: self.logger("Summation of forces: {}".format([f_x, f_y, f_z]))
        
  
        # Publish the force.
        kwargs = dict(zip(['x', 'y', 'z'], [f_x, f_y, f_z]))
        message = force_message(**kwargs)
        self.force_publisher.publish(message)
    
    
    def publish_cursor_force(self, force):
        
        # Convert the cursor force to robot force.
        f_e = self.compute_effector_force(force)
        
        # Publish the force.
        kwargs = dict(zip(['x', 'y', 'z'], f_e))
        message = force_message(**kwargs)
        self.force_publisher.publish(message)
        
    def compute_effector_force(self, cursor_force):
        """
        """
        
        # Initialize a utility function for converting messages to numpy arrays.
        to_odict = rosidl_runtime_py.convert.message_to_ordereddict
        msg_to_array = lambda m: numpy.array(tuple(to_odict(m).values()))
        to_column = lambda s: numpy.array(s)[:][numpy.newaxis].T
        
        # Convert the message to a column numpy array.
        f_c = to_column(msg_to_array(cursor_force))
        
        # Convert the cursor force to robot / effector force.
        t   = numpy.array(self.get_parameter('cursor.force_transform').value)
        T   = t.reshape((3, 3))
        f_e = T @ f_c
        
        # Return the result.
        return f_e.squeeze()
        

        
    


def main(args=None, Node=Node):
    
    rclpy.init(args=args)
    try:
        node = Node()
        try: rclpy.spin(node)
        except KeyboardInterrupt: pass
        finally: node.destroy_node()
    finally: rclpy.shutdown()
    


if __name__ == '__main__': main()


