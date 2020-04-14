import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from math import pi

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

# For GazePred dataset
scanpath_folder = 'Xu_CVPR_18/dataset/Gaze_txt_files'

users = range(1, 45)
videos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214, 215]
videos_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202]
videos_test = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214, 215]

traces_all = {}
traces_train = {}
traces_test = {}
for user in users:
    traces_all[user] = []
    traces_train[user] = []
    traces_test[user] = []
    foldername = '%s/p%03d' % (scanpath_folder, user)
    list_videos = os.listdir(foldername)
    for video in list_videos:
        video_id = int(video.split('.')[0])
        traces_all[user].append(video)
        if video_id in videos_test:
            traces_test[user].append(video)
        if video_id in videos_train:
            traces_train[user].append(video)

def get_trace(person, video):
    filename = '%s/p%03d/%s' % (scanpath_folder, person, video)
    dataframe = pd.read_table(filename, header=None, sep=",", engine='python')
    # The columns in the file that correspond to the head are 3 and 4
    head_values = np.array(dataframe.values[:, 3:5])
    return head_values

n_window = 1
for m_window in [5, 12, 25, 50, 125, 375]:
    print ('m_window', m_window * 1.0 / 25.0)
    average_velocities = []
    for user in traces_all.keys():
        for video in traces_all[user]:
            # print m_window, video, user
            positions = get_trace(user, video)
            for x_i in range(n_window, len(positions) - m_window):
                # This one computes the farthest motion from the last position
                av_vel = np.max(compute_orth_dist_with_unit_dir_vec(np.array(positions[x_i:x_i + m_window], dtype=np.float32), np.array(np.repeat(positions[x_i - 1:x_i], m_window, axis=0), dtype=np.float32)))
                # This one computes the additive motion
                # av_vel = np.sum(compute_orth_dist_with_unit_dir_vec(np.array(positions[x_i:x_i + m_window], dtype=np.float32), np.array(positions[x_i - 1:x_i + m_window - 1], dtype=np.float32)))
                average_velocities.append(av_vel)
    average_velocities = np.array(average_velocities) * 180 / pi
    n, bins, patches = plt.hist(average_velocities, bins=np.arange(0, 360), density=True, histtype='step', cumulative=True, label=str(m_window*1.0/25.0)+'s')
    # hist, bin_edges = np.histogram(average_velocities, bins=np.arange(0, 360), density=True)
plt.xlabel('Motion from last position (t -> t+T) [Degrees]')
plt.ylabel('Data proportion')
plt.legend()
plt.title('CVPR18')
plt.xlim(0, 180)
plt.ylim(0.0, 1.0)
plt.show()
