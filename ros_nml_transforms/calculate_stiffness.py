# This script calculates the stiffness parameter K by estimatine the slope of the linear region of the force-displacement curve.
# The torque can be calculate by finding the perpendicular vector between the origin point and the robot end effector position



import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
#import sys
#import math
import csv
import pandas as pd

# From the 'stiffness_node.py' file, the origin at default is at [0.04, 0.0, 0.02]
ORIGIN = np.array([0.04, 0.0, 0.02]) # x, y, z

def read_csv_files(base_path, participant_path):

    # Read the data
    base_data = pd.read_csv(base_path)
    participant_data = pd.read_csv(participant_path)

    # Make sure the shape of the data is the same
    if base_data.shape[0] > participant_data.shape[0]:
        base_data = base_data[:participant_data.shape[0]]
    else:
        participant_data = participant_data[:base_data.shape[0]]

    # What is the size of the base_data df?
    assert base_data.shape == participant_data.shape

    return base_data, participant_data

def get_torque_angle_phase_data(input_df, angle_pad=0.05):
    # This pipeline computes the torque felt at each angle of the robot end effector by using the cross product of the
    # perpendicular vector between origin and robot position as well as the forces felt at the endpoint. The data is then
    # split into 4 phases: up from 0, down from max, down from 0, up from min. Lastly, the data is regressed to find K

    # Subtract 180 from the yaw angle to get the correct angle
    df = input_df.copy()
    df['yaw_angle'] -= 180

    # Convert deg to rad
    df['yaw_angle'] = np.deg2rad(df['yaw_angle'])

    # Find the max value in desired_angle (use 10% pad)
    max_angle = df['yaw_angle'].max()
    max_angle_pad = angle_pad * max_angle
    min_angle = df['yaw_angle'].min()
    min_angle_pad = angle_pad * min_angle

    data = {'up_from_0': {'torque': None, 'angle': None},
            'down_from_max': {'torque': None, 'angle': None},
            'down_from_0': {'torque': None, 'angle': None},
            'up_from_min': {'torque': None, 'angle': None}}

    # Get the torque data and angles for each condition
    dfc = df.copy()
    print(dfc)
    up_from_0_df = dfc[(dfc['yaw_angle'].diff() > 0) & (dfc['yaw_angle'] > max_angle_pad) & (dfc['yaw_angle'] < max_angle)]
    data['up_from_0']['torque'] = np.cross(up_from_0_df[['position_x', 'position_y']].values,
                                up_from_0_df[['force_x', 'force_y']].values)
    print(data['up_from_0']['torque'])
    data['up_from_0']['torque'] -= data['up_from_0']['torque'][0]
    data['up_from_0']['angle'] = up_from_0_df['yaw_angle'].to_numpy()

    down_from_max_df = dfc[(dfc['yaw_angle'].diff() < 0) & (dfc['yaw_angle'] > max_angle_pad) & (dfc['yaw_angle'] < max_angle)]
    data['down_from_max']['torque'] = np.cross(down_from_max_df[['position_x', 'position_y']].values,
                                down_from_max_df[['force_x', 'force_y']].values)
    data['down_from_max']['torque'] -= data['down_from_max']['torque'][0]
    data['down_from_max']['angle'] = down_from_max_df['yaw_angle'].to_numpy()

    down_from_0_df = dfc[(dfc['yaw_angle'].diff() < 0) & (dfc['yaw_angle'] > min_angle) & (dfc['yaw_angle'] < min_angle_pad)]
    data['down_from_0']['torque'] = np.cross(down_from_0_df[['position_x', 'position_y']].values,
                                down_from_0_df[['force_x', 'force_y']].values)
    data['down_from_0']['torque'] -= data['down_from_0']['torque'][0]
    data['down_from_0']['angle'] = down_from_0_df['yaw_angle'].to_numpy()

    up_from_min_df = dfc[(dfc['yaw_angle'].diff() > 0) & (dfc['yaw_angle'] > min_angle) & (dfc['yaw_angle'] < min_angle_pad)]
    data['up_from_min']['torque'] = np.cross(up_from_min_df[['position_x', 'position_y']].values,
                                up_from_min_df[['force_x', 'force_y']].values)
    data['up_from_min']['torque'] -= data['up_from_min']['torque'][0]
    data['up_from_min']['angle'] = up_from_min_df['yaw_angle'].to_numpy()

    return data

def get_phase_regression(df_data):
    # Performs regression for each phase in the data

    def regress(data):
        # Performs linear regression and returns the slope and intercept
        model = LinearRegression()
        model.fit(data['angle'].reshape(-1, 1), data['torque'].reshape(-1, 1))
        return model.coef_, model.intercept_

    # Perform linear regression for each phase, return slope and intercept
    phase_data = {}
    for phase, values in df_data.items():
        #print(phase)
        #print(values)
        phase_data[phase] = regress(values)

    return phase_data


def stiffness_1D(ds_df):

    # Calculate the torque by finding the perpendicular vector between the origin and the end effector position
    end_effector_position = ds_df[['position_x', 'position_y', 'position_z']].values
    robot_vector = end_effector_position - ORIGIN
    torque_xyz = np.cross(robot_vector, ds_df[['force_x', 'force_y', 'force_z']].values) # This gets the x, y, z torque components but we need the perpendicular components

    # Single DOF measurement, starting with wrist flexion/extension. desired_angle should report 0 degrees rotation
    roll0_df = ds_df[ds_df['roll_angle'] == 0]
    robot_vector_0 = robot_vector[ds_df['roll_angle'] == 0]
    yaw0 = roll0_df['yaw_angle'].to_numpy()
    yaw0 -= 180
    wrist_fe_torque = np.cross(robot_vector_0[:, :2], roll0_df[['force_x', 'force_y']].values)

    # Plot the torque as a function of the yaw angle, this gets a hysteresis loop
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        t = np.arange(yaw0.shape[0])
        ax.scatter(yaw0, wrist_fe_torque, c=t, cmap='rainbow')
        ax.set_title('Wrist FE Torque vs. Angle')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Torque (N/m)')
        plt.show()

    # Since the torque vs angle plot shows a hysteretic loop, it's harder to find the stiffness coefficient which expects
    # a linear behavior. We can try to find the linear region of the plot by splitting the data into 4 phases:
    data = get_torque_angle_phase_data(roll0_df)

    # Perform linear regression for each phase, return slope and intercept
    reg_data = get_phase_regression(data)
    print(reg_data)

    # Find the max value in desired_angle (use 10% pad)
    #max_angle = ds_df['yaw_angle'].max()
    #max_angle_5 = 0.01 * max_angle
    #min_angle = ds_df['yaw_angle'].min()
    #min_angle_5 = 0.01 * min_angle

    # Get the torque data and angles for each condition
    #up_from_0_df = roll0_df[(roll0_df['yaw_angle'].diff() > 0) & (roll0_df['yaw_angle'] > max_angle_5) & (roll0_df['yaw_angle'] < max_angle)]
    #up_from_0_torque = np.cross(up_from_0_df[['position_x', 'position_y']].values, up_from_0_df[['force_x', 'force_y']].values)
    #up_from_0_angle = up_from_0_df['yaw_angle'].to_numpy()

    # Perform linear regression
    #model = LinearRegression()
    #model.fit(angle, torque)

    # Predict torque values using the model
    #predicted_torque = model.predict(angle)

    # Find number of times the header index (first column number) difference is greater than 1
    #diff_indices = up_from_0_df.index.to_series().diff()
    #N = len(diff_indices[diff_indices > 100])

    #down_from_max_df = ds_df[(ds_df['yaw_angle'].diff() < 0) & (ds_df['yaw_angle'] > max_angle) & (ds_df['yaw_angle'] < 0)]
    #down_from_0_df = ds_df[(ds_df['yaw_angle'].diff() < 0) & (ds_df['yaw_angle'] > min_angle_5) & (ds_df['yaw_angle'] < 0)]
    #up_from_min_df = ds_df[(ds_df['yaw_angle'].diff() > 0) & (ds_df['yaw_angle'] > min_angle) & (ds_df['yaw_angle'] < 0)]


    #Compensate for the hysteresis loop by removing the offset, so first element is 0. Make sure angle data starts at 0
    #up_from_0_torque -= up_from_0_torque[0]
    #up_from_0_angle -= up_from_0_angle[0]
    #down_from_max_torque -= down_from_max_torque[0]
    #down_from_0_torque -= down_from_0_torque[0]
    #up_from_min_torque -= up_from_min_torque[0]

    # Plot the torque data
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(up_from_0_angle, up_from_0_torque, c='b', s=0.5)
    #ax.set_xlabel('Angle (deg)')
    #ax.set_ylabel('Torque (N/m)')
    #plt.show()

if __name__ == '__main__':

    # Need to load the baseline data and participant data csv files
    base_path = r'/home/nml/haptics_ws/baseline_data.csv'
    participant_path = r'/home/nml/haptics_ws/jonathan_data.csv'

    # Read the data
    baseline_df, participant_df = read_csv_files(base_path, participant_path)

    # Subtract the forces between the two dataframes to get the actual change in force
    diff_df = participant_df.copy()
    diff_df[['force_x', 'force_y', 'force_z']] -= baseline_df[['force_x', 'force_y', 'force_z']]

    # Downsample by taking every 10th row
    ds_df = diff_df.iloc[::10]

    # perform the 1D analysis with the data
    stiffness_1D(ds_df)



    # Plot the torque
    #ax.quiver(downsampled_df['position_x'], downsampled_df['position_y'], downsampled_df['position_z'], torque[:, 0], torque[:, 1], torque[:, 2], color='r')


    # Add axis limits
    #ax.set_xlim([-0.05, 0.05])
    #ax.set_ylim([-0.05, 0.05])
    #ax.set_zlim([-0.05, 0.05])

    # Add axis labels
    #ax.set_zlabel('Z')


    # To get the perpendicular components, we need to find the angle between the torque vector and the x-axis. We can
    # assume that since the "plane of movement" is rotating about the x-axis, the x-component of the torque vector is true

    # Let's get the perpendicular vector from origin up along the z-axis, dependent on the value of the 'desired_angle' column
    # We can also use this plane vetor to get the torque with respect to the plane of motion, like the wrist abduction/aduction
    #plane_vector = np.array([0, np.sin(downsampled_df['roll_angle']), np.cos(downsampled_df['roll_angle'])])

    # Get the vector from the cross product of the plane_vector and the torque_xyz
    #torque_vector = np.cross(plane_vector, torque_xyz)

    # The torque with respect to the plane of motion

    # To-DO calculate the new torque values when a rotation around the x-axis is applied
    # Calculate the perpendicular torque components
    #perpendicular_torque_x = torque_xyz[:, 0]
    #perpendicular_torque_y = torque_y * np.cos(angle_rad) - torque_z * np.sin(angle_rad)
    #perpendicular_torque_z = torque_y * np.sin(angle_rad) + torque_z * np.cos(angle_rad)


    #angle0_data = diff_df[diff_df['angle'] == 0]
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(angle0_data['position_x'], angle0_data['position_y'], angle0_data['position_z'])
    #plt.show()


    # Plot data
    #plt.plot(displacement, force)
    #plt.xlabel('Displacement')
    #plt.ylabel('Force')
    #plt.show()


