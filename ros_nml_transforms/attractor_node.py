""" A ROS2 node that implements an attractor transformaion.

See `segment.cpp` in the Force Dimension SDK examples.

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
from numpy import linalg

# Import ROS2.
import rclpy
import rclpy.qos
import rosidl_runtime_py

# Local imports.
from ros_nml_transforms import node
from ros_nml_transforms.msg import velocity_message


# Declare default quality-of-service settings for ROS2.
DEFAULT_QOS = rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value


# Define the node class.
class Node(node.Node):
        
    def __init__(self, *args, **kwargs):
        self.current_effector_velocity = velocity_message()
        super().__init__(*args, **kwargs)
        
    def initialize_parameters(self):
        super().initialize_parameters()
        default_basis = [+1.0, +0.0, +0.0,
                         +0.0, +1.0, +0.0,
                         +0.0, +0.0, +1.0]
        self.declare_parameter('effector.attractor_basis', default_basis)
        self.declare_parameter('effector.attractor_offset', [0.0, 0.0, 0.0])
        self.declare_parameter('effector.attractor_stiffness', 2000.0)
        self.declare_parameter('effector.attractor_damping', 20.0)        
        
    def initialize_subscriptions(self):
        """
        """
        
        # Invoke superclass method.
        super().initialize_subscriptions()
        
        # Initialize a subscription for the robot position.
        kwargs = dict(topic='feedback/velocity',
                      msg_type=velocity_message,
                      callback=self.velocity_callback,
                      qos_profile=DEFAULT_QOS)
                      #rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(**kwargs)
        
    def velocity_callback(self, message):
        """
        """
        self.current_effector_velocity = message
        
    def compute_effector_force(self, cursor_force):
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
        f_e = super().compute_effector_force(cursor_force)
        
        # Initialize a utility function for getting parameters.
        get_parameter = lambda n: self.get_parameter(n).value
        
        # Initialize a utility function for converting messages to numpy arrays.
        to_odict = rosidl_runtime_py.convert.message_to_ordereddict
        msg_to_array = lambda m: numpy.array(tuple(to_odict(m).values()))
        to_column = lambda s: numpy.array(s)[:][numpy.newaxis].T
        
        # Initialize local variables.
        K = get_parameter('effector.attractor_stiffness')
        C = get_parameter('effector.attractor_damping')
        o = to_column(get_parameter('effector.attractor_offset'))
        p = to_column(msg_to_array(self.current_effector_position))
        v = to_column(msg_to_array(self.current_effector_velocity))
        
        # Create a projection matrix from the attractor basis parameter.
        b   = self.get_parameter('effector.attractor_basis').value
        B   = numpy.array(b).reshape((3, 3))
        P_a = B @ linalg.pinv(B)
        
        # Compute the projection of the position vector onto the attractor 
        # surface -- that is, the permissable plane in which the effector 
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
        # tangent to the permissable plane.
        # This is recommended by the Force Dimension example, and seems 
        # necessary due to the velocity term in the spring + damper model, 
        # which can introduce viscous force components on the permissable 
        # attractor surface (i.e., the "free" plane).
        P_g = p_d @ linalg.pinv(p_d)
        f_g = P_g @ f_g
        
        # Compute the aggregate effector force by adding in the attractor force.
        #self.get_logger().info(f'{f_e + f_g.squeeze()} (f_e: {f_e}; f_g: {f_g})')
        f_e = f_e + f_g.squeeze()
        
        if (len(f_e) != 3) or not isinstance(f_e[0], float): raise Exception()
        
        # Return the result as a tuple.
        return f_e

    


def main(args=None, Node=Node): return node.main(Node=Node)
    


if __name__ == '__main__': main()


