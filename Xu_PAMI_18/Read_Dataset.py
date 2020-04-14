import sys
sys.path.insert(0, './')

import scipy.io
import os
import numpy as np
import pandas as pd
from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SampledDataset import get_video_ids, get_user_ids, get_users_per_video
import cv2
import argparse

ROOT_FOLDER = './Xu_PAMI_18/dataset/'
DATA_FILENAME = 'FULLdata_per_video_frame.mat'
OUTPUT_FOLDER = './Xu_PAMI_18/sampled_dataset'

OUTPUT_TRUE_SALIENCY_FOLDER = './Xu_PAMI_18/true_saliency'
NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

SAMPLING_RATE = 0.2

# .mat dataset was obtained from https://github.com/YuhangSong/DHP
def get_dot_mat_data():
    mat = scipy.io.loadmat(os.path.join(ROOT_FOLDER, DATA_FILENAME))
    dataset = {}
    for video in mat.keys():
        if video in ['__globals__', '__version__', '__header__']:
            continue
        longitudes = mat[video][:, ::2]
        latitudes = mat[video][:, 1::2]
        for user in range(int(mat[video].shape[1]/2)):
            if user not in dataset.keys():
                dataset[user] = {}
            long = longitudes[:, user]
            lat = latitudes[:, user]
            dataset[user][video] = np.concatenate((long[:, np.newaxis], lat[:, np.newaxis]), axis=1)
    return dataset

# return the list of unique videos from the "dataset"
def get_videos_list(dataset):
    videos = []
    for user in dataset.keys():
        for video in dataset[user].keys():
            videos.append(video)
    return list(set(videos))

# The rate in the file is in the first colum, we perform a cumulative sum and divide by 1000 since the rate is in milliseconds
# Returns the column corresponding to the rate
def get_orig_rates_for_trace(filename):
    dataframe = pd.read_table(filename, header=None, sep=r"\s*", engine='python')
    # the sampling rate is given in the first column (the last row is the sum of the rates)
    rate = dataframe[[0]].iloc[:-1]
    cum_sum_rate = np.cumsum(rate.values)/1000.0
    return cum_sum_rate

# The files are composed as follows: rate, latitude, longitude, x, y, z (coordinates of the gaze)
# Returns in order the latitude and longitude
def get_orig_positions_for_trace(filename):
    dataframe = pd.read_table(filename, header=None, sep=r"\s*", engine='python')
    # To read pitch (latitude) and yaw (longitude) in order
    data = dataframe[[1, 2]].iloc[:-1]
    return data.values

# dataset is an older version of .mat dataset found before under the same link https://github.com/YuhangSong/DHP
def get_original_data():
    dataset = {}
    for root, directories, files in os.walk(os.path.join(ROOT_FOLDER, 'data')):
        for user in directories:
            dataset[user] = {}
            for r_2, d_2, sub_files in os.walk(os.path.join(ROOT_FOLDER, 'data', user)):
                for video_txt in sub_files:
                    video = video_txt.split('.')[0]
                    filename = os.path.join(ROOT_FOLDER, 'data', user, video_txt)
                    dataset[user][video] = get_orig_positions_for_trace(filename)
    return dataset

# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset(videos_list):
    dataset = {}
    for root, directories, files in os.walk(os.path.join(ROOT_FOLDER, 'data')):
        for user in directories:
            dataset[user] = {}
            for video in videos_list:
                print ('getting original dataset', user, video)
                video_txt = video+'.txt'
                filename = os.path.join(ROOT_FOLDER, 'data', user, video_txt)
                positions = get_orig_positions_for_trace(filename)
                rates = get_orig_rates_for_trace(filename)
                samples = []
                for pos, rate in zip(positions, rates):
                    samples.append({'sec':rate, 'yaw':pos[1], 'pitch':pos[0]})
                dataset[user][video] = samples
    return dataset

# This comparison was performed to find if there were differences between both datasets
# we found that both datasets are the same except for a few NaN entries in .mat dataset
# and some traces on videos in the original dataset that do not exist (like LOL, Help2, RollerCoaster, Dota2 and Soldier)
def compare_dot_mat_dataset_and_original_dataset(dot_mat_dataset, original_dataset):
    flag_more_vids_in_orig = False
    flag_more_vids_in_dot_mat = False
    flag_nan_in_orig = False
    flag_nan_in_dot_mat = False
    flag_equal_datasets = True
    # First let's compare if the list of videos is the same
    dot_mat_videos = get_videos_list(dot_mat_dataset)
    orig_dat_videos = get_videos_list(original_dataset)
    for video in orig_dat_videos:
        if video not in dot_mat_videos:
            print ('video', video, 'is not in dot_mat_dataset videos')
            flag_more_vids_in_orig = True
    for video in dot_mat_videos:
        if video not in orig_dat_videos:
            print ('video', video, 'is not in original_dataset videos')
            flag_more_vids_in_dot_mat = True
    # now let's compare sample by sample
    users = list(original_dataset.keys())
    users.sort()
    for i, user in enumerate(users):
        for video in dot_mat_dataset[i].keys():
            for sample_dot, sample_orig in zip(dot_mat_dataset[i][video], original_dataset[user][video]):
                if np.isnan(sample_dot[0]) or np.isnan(sample_dot[1]):
                    print('Nan sample', sample_dot, 'in sample_dot from video', video, 'user', user)
                    flag_nan_in_dot_mat = True
                if np.isnan(sample_orig[0]) or np.isnan(sample_orig[1]):
                    print('Nan sample', sample_orig, 'in sample_orig from video', video, 'user', user)
                    flag_nan_in_orig = True
                if not np.isclose(sample_dot[0], sample_orig[0]) and np.isclose(sample_dot[1], sample_orig[1]):
                    print('not close', i, user, video, sample_dot, sample_orig)
                    flag_equal_datasets = False
    print('---- SUMMARY ----')
    if flag_more_vids_in_orig:
        print('There are some videos in the original dataset that are not in the .mat dataset')
    if flag_more_vids_in_dot_mat:
        print('There are some videos in the .mat dataset that are not in the original dataset')
    if flag_nan_in_orig:
        print('There are NaN values in the original dataset')
    if flag_nan_in_dot_mat:
        print('There are NaN values in the .mat dataset')
    if flag_equal_datasets:
        print('Appart from these differences, the datasets are equal')

# The HM data takes the front center as the origin, and the upper & left as the positive direction
# Thus the longitudes ranges from -180 to 180, and the latitude ranges from -90 to 90.
# The subject starts to watch the video at position yaw=0, turning the head left leads to positive yaw.
## In other words, yaw = 0, pitch = 0 is equal to the position (1, 0, 0) in cartesian coordinates
# Pitching the head up results in a positive pitch value.
## In other words, yaw = Any, pitch = 90\degree is equal to the position (0, 0, 1) in cartesian coordinates
### We will transform these coordinates so that
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
def transform_the_degrees_in_range(yaw, pitch):
    yaw = yaw + 180
    pitch = -pitch + 90
    return degrees_to_radian(yaw), degrees_to_radian(pitch)

# Performs the opposite transformation than transform_the_degrees_in_range
def transform_the_radians_to_original(yaw, pitch):
    yaw = (yaw-np.pi)
    pitch = -(pitch-np.pi/2.0)
    return radian_to_degrees(yaw), radian_to_degrees(pitch)

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def create_sampled_dataset(original_dataset, rate):
    dataset = {}
    for user in original_dataset.keys():
        dataset[user] = {}
        for video in original_dataset[user].keys():
            print('creating sampled dataset', user, video)
            sample_orig = np.array([1, 0, 0])
            data_per_video = []
            for sample in original_dataset[user][video]:
                sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
                sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
                quat_rot = rotationBetweenVectors(sample_orig, sample_new)
                # append the quaternion to the list
                data_per_video.append([sample['sec'], quat_rot[0], quat_rot[1], quat_rot[2], quat_rot[3]])
                # update the values of time and sample
            # interpolate the quaternions to have a rate of 0.2 secs
            data_per_video = np.array(data_per_video)
            dataset[user][video] = interpolate_quaternions(data_per_video[:, 0], data_per_video[:, 1:], rate=rate)
    return dataset

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def recover_original_angles_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample_new[0], sample_new[1], sample_new[2])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

def recover_original_angles_from_xyz_trace(xyz_trace):
    angles_per_video = []
    for sample in xyz_trace:
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample[1], sample[2], sample[3])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

### Check if the quaternions are good
# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def compare_sample_vs_original(original_dataset, sampled_dataset):
    for user in original_dataset.keys():
        for video in original_dataset[user].keys():
            pitchs = []
            yaws = []
            times = []
            for sample in original_dataset[user][video]:
                times.append(sample['sec'])
                yaws.append(sample['yaw'])
                pitchs.append(sample['pitch'])
            angles_per_video = recover_original_angles_from_xyz_trace(sampled_dataset[user][video])
            plt.subplot(1, 2, 1)
            plt.plot(times, yaws, label='yaw')
            plt.plot(times, pitchs, label='pitch')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 0], label='yaw')
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 1], label='pitch')
            plt.legend()
            plt.show()

# Returns the values (x, y, z) of a unit sphere with center in (0, 0, 0)
# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def recover_xyz_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        angles_per_video.append(quat_rot.rotate(orig_vec))
    return np.concatenate((quaternions_trace[:, 0:1], np.array(angles_per_video)), axis=1)

# Return the dataset
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def get_xyz_dataset(sampled_dataset):
    dataset = {}
    for user in sampled_dataset.keys():
        dataset[user] = {}
        for video in sampled_dataset[user].keys():
            dataset[user][video] = recover_xyz_from_quaternions_trace(sampled_dataset[user][video])
    return dataset

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def plot_3d_trace(positions, user, video):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='parametric curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('User: %s, Video: %s' % (user, video))
    plt.show()

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def plot_all_traces_in_3d(xyz_dataset):
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            plot_3d_trace(xyz_dataset[user][video][:, 1:], user, video)

def store_dataset(xyz_dataset):
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            video_folder = os.path.join(OUTPUT_FOLDER, video)
            # Create the folder for the video if it doesn't exist
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            path = os.path.join(video_folder, user)
            df = pd.DataFrame(xyz_dataset[user][video])
            df.to_csv(path, header=False, index=False)

# ToDo This is the Main function of this file
def create_and_store_sampled_dataset(plot_comparison=False, plot_3d_traces=False):
    print('Getting .mat data')
    dot_mat_data = get_dot_mat_data()
    # original_data = get_original_data()
    # compare_dot_mat_dataset_and_original_dataset(dot_mat_data, original_data)
    print('Getting original dataset')
    original_dataset = get_original_dataset(get_videos_list(dot_mat_data))
    print('Creating sampled dataset')
    sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
    if plot_comparison:
        compare_sample_vs_original(original_dataset, sampled_dataset)
    xyz_dataset = get_xyz_dataset(sampled_dataset)
    if plot_3d_traces:
        plot_all_traces_in_3d(xyz_dataset)
    store_dataset(xyz_dataset)

# Returns the maximum number of samples among all users (the length of the largest trace)
def get_max_num_samples_for_video(video, sampled_dataset):
    max_len = 0
    for user in sampled_dataset.keys():
        curr_len = len(sampled_dataset[user][video])
        if curr_len > max_len:
            max_len = curr_len
    return max_len

def create_and_store_true_saliency(sampled_dataset):
    if not os.path.exists(OUTPUT_TRUE_SALIENCY_FOLDER):
        os.makedirs(OUTPUT_TRUE_SALIENCY_FOLDER)

    # Returns an array of size (NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL) with values between 0 and 1 specifying the probability that a tile is watched by the user
    # We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
    def from_position_to_tile_probability_cartesian(pos):
        yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, NUM_TILES_WIDTH_TRUE_SAL, endpoint=False),
                                           np.linspace(0, 1, NUM_TILES_HEIGHT_TRUE_SAL, endpoint=False))
        yaw_grid += 1.0 / (2.0 * NUM_TILES_WIDTH_TRUE_SAL)
        pitch_grid += 1.0 / (2.0 * NUM_TILES_HEIGHT_TRUE_SAL)
        yaw_grid = yaw_grid * 2 * np.pi
        pitch_grid = pitch_grid * np.pi
        x_grid, y_grid, z_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)
        great_circle_distance = np.arccos(np.maximum(np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0), -1.0))
        gaussian_orth = np.exp((-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance))
        return gaussian_orth

    videos = get_video_ids(OUTPUT_FOLDER)

    for enum_video, video in enumerate(videos):
        print('creating true saliency for video', video, '-', enum_video, '/', len(videos))
        real_saliency_for_video = []

        max_num_samples = get_max_num_samples_for_video(video, sampled_dataset)

        for x_i in range(max_num_samples):
            tileprobs_for_video_cartesian = []
            for user in sampled_dataset.keys():
                if len(sampled_dataset[user][video]) > x_i:
                    tileprobs_cartesian = from_position_to_tile_probability_cartesian(sampled_dataset[user][video][x_i, 1:])
                    tileprobs_for_video_cartesian.append(tileprobs_cartesian)
            tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
            real_saliency_cartesian = np.sum(tileprobs_for_video_cartesian, axis=0) / tileprobs_for_video_cartesian.shape[0]
            real_saliency_for_video.append(real_saliency_cartesian)
        real_saliency_for_video = np.array(real_saliency_for_video)

        true_sal_out_file = os.path.join(OUTPUT_TRUE_SALIENCY_FOLDER, video)
        np.save(true_sal_out_file, real_saliency_for_video)

# ToDo copied integrally from David_MMSys_18/Reading_Dataset.py
def load_sampled_dataset():
    list_of_videos = [o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith('.gitkeep')]
    dataset = {}
    for video in list_of_videos:
        for user in [o for o in os.listdir(os.path.join(OUTPUT_FOLDER, video)) if not o.endswith('.gitkeep')]:
            if user not in dataset.keys():
                dataset[user] = {}
            path = os.path.join(OUTPUT_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the input parameters to parse the dataset.')
    parser.add_argument('-creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset.')
    parser.add_argument('-creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('-compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('-plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')
    parser.add_argument('-analyze_data', action="store_true", dest='_analyze_orig_data', help='Flag that tells if we want to verify if the tile probability is correctly computed.')

    args = parser.parse_args()

    #print('Use this file to create sampled dataset')
    # Create sampled dataset
    if args._create_sampled_dataset:
        create_and_store_sampled_dataset()
    
    if args._analyze_orig_data:
        dot_mat_data = get_dot_mat_data()
        original_data = get_original_data()
        compare_dot_mat_dataset_and_original_dataset(dot_mat_data, original_data)

    if args._compare_traces:
        dot_mat_data = get_dot_mat_data()
        original_dataset = get_original_dataset(get_videos_list(dot_mat_data))
        sampled_dataset = load_sampled_dataset()
        compare_sample_vs_original(original_dataset, sampled_dataset)

    if args._plot_3d_traces:
        sampled_dataset = load_sampled_dataset()
        plot_all_traces_in_3d(sampled_dataset)

    if args._create_true_saliency:
        if os.path.isdir(OUTPUT_FOLDER):
            sampled_dataset = load_sampled_dataset()
            create_and_store_true_saliency(sampled_dataset)
        else:
            print('Please verify that the sampled dataset has been created correctly under the folder', OUTPUT_FOLDER)
