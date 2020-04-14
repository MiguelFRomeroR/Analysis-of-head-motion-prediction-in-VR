# For the dataset of Fixation prediction (NOSSDAV 17)
from pandas import read_csv
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
import random

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

path360Dataset = 'Fan_NOSSDAV_17/dataset'
# Returns yaw ([:, 0]) and pitch ([:, 1]), both in the range [0, 1].
def read_orientation_info(usr_id, video_name):
    csv_filename = path360Dataset+'/sensory/orientation/%s_user%02d_orientation.csv' % (video_name, usr_id)
    data_frame_user_i = read_csv(csv_filename, usecols=[7,8], engine='python', header=0)
    dataset_usr_i = data_frame_user_i.values
    dataset_usr_i = dataset_usr_i.astype('float32')
    dataset_usr_i = dataset_usr_i / np.array([360.0, 180.0])
    dataset_usr_i = dataset_usr_i + 0.5
    return dataset_usr_i

videos = np.array(['coaster2', 'coaster', 'diving', 'drive', 'game', 'landscape', 'pacman', 'panel', 'ride', 'sport'])
users = range(50)
proportion_train_set = 0.8

videos = np.array(['coaster2', 'coaster', 'diving', 'drive', 'game', 'landscape', 'pacman', 'panel', 'ride', 'sport'])
videos_indices = np.arange(len(videos))

# fix random seed for reproducibility
random.seed(7)
np.random.seed(7)
# Select 12 users for training and validation as in the paper
users_train = random.sample(range(50), 12)
traces_indices = np.array(np.meshgrid(users_train, videos_indices)).T.reshape(-1, 2)
np.random.shuffle(traces_indices)

num_train_traces = int(len(traces_indices)*proportion_train_set)
training_traces = traces_indices[:num_train_traces, :]
validation_traces = traces_indices[num_train_traces:, :]

n_window = 1
for time_t in [0.2, 0.5, 1, 2, 5, 15]:
    print (time_t)
    m_window = int(round(time_t * 30.0))
    traces_counter = 0
    average_velocities = []
    for trace in traces_indices:
        user = trace[0]
        video = videos[trace[1]]
        traces_counter += 1
        # print video, user, "video: %s/%s" % (traces_counter, len(validation_traces))
        positions = read_orientation_info(user+1, video)
        for x_i in range(n_window, len(positions) - m_window):
            # This one computes the farthest motion from the last position
            av_vel = np.max(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i + m_window], np.repeat(positions[x_i - 1:x_i], m_window, axis=0)))
            # This one computes the additive motion
            # av_vel = np.sum(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i + m_window], positions[x_i - 1:x_i + m_window - 1]))
            average_velocities.append(av_vel)
    average_velocities = np.array(average_velocities) * 180 / pi
    # Compute the CDF
    n, bins, patches = plt.hist(average_velocities, bins=np.arange(0, 360), density=True, histtype='step', cumulative=True, label=str(time_t)+'s')
plt.xlabel('Motion from last position (t -> t+T) [Degrees]')
plt.ylabel('Data proportion')
plt.title('NOSSDAV 17')
plt.xlim(0, 180)
plt.ylim(0.0, 1.0)
plt.legend()
plt.show()
