import os
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cv2

# Compute the sphere distance with the unit directional vectors in cartesian coordinate system
def compute_orth_dist_with_unit_dir_vec(position_a, position_b):
    yaw_true = (position_a[:, 0] - 0.5) * 2 * pi
    pitch_true = (position_a[:, 1] - 0.5) * pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (position_b[:, 0] - 0.5) * 2 * pi
    pitch_pred = (position_b[:, 1] - 0.5) * pi
    # Transform into directional vector in Cartesian Coordinate System
    x_true = np.sin(yaw_true)*np.cos(pitch_true)
    y_true = np.sin(pitch_true)
    z_true = np.cos(yaw_true)*np.cos(pitch_true)
    x_pred = np.sin(yaw_pred)*np.cos(pitch_pred)
    y_pred = np.sin(pitch_pred)
    z_pred = np.cos(yaw_pred)*np.cos(pitch_pred)
    # Finally compute orthodromic distance
    # great_circle_distance = np.arccos(x_true*x_pred+y_true*y_pred+z_true*z_pred)
    # To keep the values in bound between -1 and 1
    great_circle_distance = np.arccos(np.maximum(np.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0))
    return great_circle_distance

import pickle
#load data from pickle file.
if 'salient_ds_dict' not in locals():
    with open('Nguyen_MM_18/dataset/salient_ds_dict_w16_h9', 'rb') as file_in:
        u = pickle._Unpickler(file_in)
        u.encoding = 'latin1'
        salient_ds_dict = u.load()

video_names = salient_ds_dict['360net'].keys()
print (video_names)

# Compute the sphere distance with the unit directional vectors in cartesian coordinate system
def compute_orth_dist_with_unit_dir_vec(position_a, position_b):
    x_true = position_a[0]
    y_true = position_a[1]
    z_true = position_a[2]

    x_pred = position_b[0]
    y_pred = position_b[1]
    z_pred = position_b[2]

    # To keep the values in bound between -1 and 1
    great_circle_distance = np.arccos(np.maximum(np.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0))
    return great_circle_distance

# for prediction_window in [0.5, 1, 1.5, 2, 2.5]:

# We found that the dataset has two timestamps one contained in the other
# and there are videos without head position data
def pre_process_dataset(salient_ds_dict):
    processed_dataset = {}
    for VID_NAME in salient_ds_dict['360net'].keys():
        if (len(salient_ds_dict['360net'][VID_NAME]['headpos']) > 0):
            processed_dataset[VID_NAME] = {}
            last_timestamp = -1.0
            cut_index = 0
            for x_i in range(len(salient_ds_dict['360net'][VID_NAME]['timestamp'])):
                curr_timestamp = salient_ds_dict['360net'][VID_NAME]['timestamp'][x_i]
                # Find where the timestamp restarts
                if curr_timestamp < last_timestamp:
                    cut_index = x_i
                last_timestamp = curr_timestamp
            processed_dataset[VID_NAME]['timestamp'] = list(np.copy(salient_ds_dict['360net'][VID_NAME]['timestamp'][cut_index:]))
            processed_dataset[VID_NAME]['salient'] = list(np.copy(salient_ds_dict['360net'][VID_NAME]['salient'][cut_index:]))
            processed_dataset[VID_NAME]['headpos'] = []
            for UID in range(len(salient_ds_dict['360net'][VID_NAME]['headpos'])):
                processed_dataset[VID_NAME]['headpos'].append(list(np.copy(salient_ds_dict['360net'][VID_NAME]['headpos'][UID][cut_index:])))
    return processed_dataset


# To show the effect of pre-processing:
# processed_data = pre_process_dataset(salient_ds_dict)
# for VID_NAME in processed_data.keys():
#     print ('Original', len(salient_ds_dict['360net'][VID_NAME]['timestamp']),
#            len(salient_ds_dict['360net'][VID_NAME]['headpos']), len(salient_ds_dict['360net'][VID_NAME]['headpos'][0]),
#            len(salient_ds_dict['360net'][VID_NAME]['salient']), salient_ds_dict['360net'][VID_NAME]['timestamp'])
#     print ('Processed', len(processed_data[VID_NAME]['timestamp']), len(salient_ds_dict['360net'][VID_NAME]['headpos']),
#            len(processed_data[VID_NAME]['headpos'][0]), len(processed_data[VID_NAME]['salient']),
#            processed_data[VID_NAME]['timestamp'])


# compute the fartest motion from the last position
compute_furthest_mot = True

if compute_furthest_mot:
    processed_data = pre_process_dataset(salient_ds_dict)

    # Number of seconds per time-step
    step = 0.063

    max_velocities_per_pred_w = []
    for prediction_window in [0.2, 0.5, 1.0, 2.0, 5.0, 15.0]:
        print ('prediction_window', prediction_window)
        m_window = int(np.round(prediction_window/step))
        max_velocities = []
        for video in processed_data.keys():
            for user in range(len(processed_data[video]['headpos'])):
                positions = processed_data[video]['headpos'][user]
                for x_i in range(1, len(positions)-m_window):
                    prediction = positions[x_i-1]
                    max_vel = 0.0
                    # Compute furthest motion from last position
                    for t in range(m_window):
                        groundtruth = positions[t+x_i]
                        max_vel = max(max_vel, compute_orth_dist_with_unit_dir_vec(groundtruth, prediction))
                    max_velocities.append(max_vel)
        max_velocities = np.array(max_velocities) * 180 / pi
        max_velocities_per_pred_w.append(max_velocities)
        n, bins, patches = plt.hist(max_velocities,bins=np.arange(0, 360), density=True, histtype='step', cumulative=True, label=str(prediction_window) + 's')
        # hist, bin_edges = np.histogram(max_velocities, bins=np.arange(0, 360), density=True)
        # for i in range(len(hist)):
        #     print (bin_edges[i], hist[i])

    plt.xlabel('Motion from last position (t -> t+T) [Degrees]')
    plt.ylabel('Data proportion')
    plt.legend()
    plt.xlim(0, 180)
    plt.ylim(0.0, 1.0)
    plt.title('CDF - All Videos - MM18')
    plt.show()
