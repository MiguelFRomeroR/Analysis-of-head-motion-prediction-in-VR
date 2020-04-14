import os
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

## For the dataset of MMSys18
dataFolder = 'DatasetAnalysis/MMSys18/scanpaths'
videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
videos_ids = np.arange(0, len(videos))
users = np.arange(0, 57)

# Select at random the users for each set
np.random.shuffle(users)
num_train_users = int(len(users)*0.5)
users_train = users[:num_train_users]
users_test = users[num_train_users:]

# Select at random the videos for each set
np.random.shuffle(videos_ids)
num_train_videos = int(len(videos_ids)*0.8)
videos_ids_train = videos_ids[:num_train_videos]
videos_ids_test = videos_ids[num_train_videos:]

'''
Return longitude \in {0, 1} and latitude \in {0, 1}
'''
def get_positions_for_trace(video, user):
    foldername = os.path.join(dataFolder, video + '_fixations')
    filename = os.path.join(foldername, video + '_fixations_user_' + str(user) + '.csv')
    dataframe = pd.read_table(filename, header=None, sep=",", engine='python')
    dataframe_values = np.array(dataframe.values[:, 0:2])
    return dataframe_values

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

n_window = 1
for time_t in [0.2, 0.5, 1, 2, 5, 15]:
    print ('time_t', time_t)
    m_window = int(round(time_t * 5.0))
    average_velocities = []
    for video_id in videos_ids:
        video = videos[video_id]
        velocities_per_video = []
        for user in users:
            # print video, user
            velocities_for_trace = []
            positions = get_positions_for_trace(video, user)
            for x_i in range(n_window, len(positions)-m_window):
                # This one computes the farthest motion from the last position
                av_vel = np.max(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i + m_window], np.repeat(positions[x_i - 1:x_i], m_window, axis=0)))
                # This one computes the additive motion
                # av_vel = np.sum(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i + m_window], positions[x_i - 1:x_i + m_window - 1]))
                average_velocities.append(av_vel)
                velocities_for_trace.append(av_vel)
                velocities_per_video.append(av_vel)
    average_velocities = np.array(average_velocities) * 180 / pi
    # Compute the CDF
    n, bins, patches = plt.hist(average_velocities, bins=np.arange(0, 360), density=True, histtype='step', cumulative=True, label=str(time_t)+'s')
    # hist, bin_edges = np.histogram(average_velocities, bins=np.arange(0, 360), density=True)
    # for i in range(len(hist)):
    #     print (bin_edges[i], hist[i])
    # print n.shape, bins.shape, patches.shape
plt.xlabel('Motion from last position (t -> t+T) [Degrees]')
plt.ylabel('Data proportion')
plt.legend()
plt.xlim(0, 180)
plt.ylim(0.0, 1.0)
plt.title('MMSys 18')
plt.show()
