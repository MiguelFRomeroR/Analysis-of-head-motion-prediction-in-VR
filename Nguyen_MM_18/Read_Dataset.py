import sys
sys.path.insert(0, './')

import pickle
import os
import numpy as np
import cv2
from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees
from pyquaternion import Quaternion
import pandas as pd
from SampledDataset import get_video_ids, get_user_ids, get_users_per_video
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT_FOLDER = './Nguyen_MM_18/dataset/'
OUTPUT_FOLDER = './Nguyen_MM_18/sampled_dataset'
OUTPUT_SALIENCY_FOLDER = './Nguyen_MM_18/extract_saliency/saliency'
OUTPUT_TRUE_SALIENCY_FOLDER = './Nguyen_MM_18/true_saliency'

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

ORIGINAL_SAMPLING_RATE = 0.063
SAMPLING_RATE = 0.2

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
#CALCULATE DEGREE DISTANCE BETWEEN TWO 3D VECTORS
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def vector_to_ang(_v):
    _v = np.array(_v)
    alpha = degree_distance(_v, [0, 1, 0])#degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0] #proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = _v - proj1#proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = degree_distance(proj2, [1, 0, 0])#theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi


def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h / 2.0 - (_h / 2.0) * np.sin(_phi / 180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0 / 360 * _w)
    return int(x), int(y)

H = 10
W = 20
def create_fixation_map(v):
    theta, phi = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H - hi - 1, W - wi - 1] = 1
    return result

def load_dataset():
    if 'salient_ds_dict' not in locals():
        with open(os.path.join(ROOT_FOLDER, 'salient_ds_dict_w16_h9'), 'rb') as file_in:
            u = pickle._Unpickler(file_in)
            u.encoding = 'latin1'
            salient_ds_dict = u.load()
    return salient_ds_dict

# This function is used to show how the time-stamps of the dataset are taken
# We find that the traces of most of the videos are splitted and concatenated
# From MM_18 paper:
# For each video, we select one segment with a length of 20-45 seconds. The video segment is selected such that there are one or more events in the video that introduce new salient regions
# (usually when new video scene is shown) and lead to fast head movement of users.
# We extract the timestamped saliency maps and head orientation maps from these videos, generating a total of 300,000 data samples from 432 time series using viewing logs of 48 users
def analyze_time_stamps():
    number_of_time_series = 0
    number_of_data_samples = 0
    # load data from pickle file.
    salient_ds_dict = load_dataset()
    for video_id, video in enumerate(salient_ds_dict['360net'].keys()):
        time_stamps = salient_ds_dict['360net'][video]['timestamp']
        frame_rates = np.array(time_stamps[1:]) - np.array(time_stamps[:-1])
        flag_splitted = False
        for id, rate in enumerate(frame_rates):
            if not np.isclose(rate, ORIGINAL_SAMPLING_RATE):
                # print('video', video, 'has consecutive time-stamps', time_stamps[id-1], time_stamps[id], time_stamps[id+1], time_stamps[id+2], 'meaning that there are two parts of the video concatenated')
                print('video', video, 'is splitted, there is a part that goes from', time_stamps[0], 'to', time_stamps[id], 'and another part that goes from', time_stamps[id+1], 'to', time_stamps[-1])
                flag_splitted = True
        if not flag_splitted:
            print('video', video, 'is not splitted, goes from', time_stamps[0], 'to', time_stamps[-1])
            print('video length:', time_stamps[-1] - time_stamps[0])
        num_users_for_video = len(salient_ds_dict['360net'][video]['headpos'])
        number_of_time_series += num_users_for_video
        number_of_data_samples += len(time_stamps) * num_users_for_video
    print('number_of_time_series', number_of_time_series)
    print('number_of_data_samples', number_of_data_samples)


# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
# In equirectangular projection, the longitude ranges from left to right as follows: 0 +90 +180 -180 -90 0
# and the latitude ranges from top to bottom: -90 90
def get_original_dataset():
    dataset = {}
    # load data from pickle file.
    salient_ds_dict = load_dataset()
    for video_id, video in enumerate(salient_ds_dict['360net'].keys()):
        time_stamps = salient_ds_dict['360net'][video]['timestamp']
        id_sort_tstamps = np.argsort(time_stamps)
        for user_id in range(len(salient_ds_dict['360net'][video]['headpos'])):
            print('get head positions from original dataset', 'video', video_id, '/', len(salient_ds_dict['360net'].keys()), 'user', user_id, '/', len(salient_ds_dict['360net'][video]['headpos']))
            user = str(user_id)
            if user not in dataset.keys():
                dataset[user] = {}
            positions_vector = salient_ds_dict['360net'][video]['headpos'][user_id]
            samples = []
            # Sorted time-stamps
            for id_sort in id_sort_tstamps:
                yaw_true, pitch_true = (vector_to_ang(positions_vector[id_sort]))
                samples.append({'sec': time_stamps[id_sort], 'yaw': yaw_true, 'pitch': pitch_true})
            dataset[user][video] = samples
    return dataset

# In equirectangular projection, the longitude ranges from left to right as follows: 0 +90 +180 -180 -90 0
# and the latitude ranges from top to bottom: -90 90
### We will transform these coordinates so that
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates (after applying eulerian_to_cartesian function)
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
# For this, we will transform:
# pitch = -90 into pitch = 0
# pitch = 90 into pitch = pi
# yaw = -180 into yaw = 0
# yaw = -90 into yaw = pi/2
# yaw = 0 into yaw = pi
# yaw = 90 into yaw = 3pi/2
# yaw = 180 into yaw = 2pi
def transform_the_degrees_in_range(yaw, pitch):
    yaw = ((yaw+180.0)/360.0)*2*np.pi
    pitch = ((pitch+90.0)/180.0)*np.pi
    return yaw, pitch

# Performs the opposite transformation than transform_the_degrees_in_range
def transform_the_radians_to_original(yaw, pitch):
    yaw = ((yaw/(2*np.pi))*360.0)-180.0
    pitch = ((pitch/np.pi)*180.0)-90.0
    return yaw, pitch

# ToDo Copied exactly from David_MMSys_18/Reading_Dataset (Author: Miguel Romero)
def create_sampled_dataset(original_dataset, rate):
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating sampled dataset', 'video', enum_video, '/', len(original_dataset[user].keys()), 'user', enum_user, '/', len(original_dataset.keys()))
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
            # In this case the time starts counting at random parts of the video
            dataset[user][video] = interpolate_quaternions(data_per_video[:, 0], data_per_video[:, 1:], rate=rate, time_orig_at_zero=False)
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

def compare_integrals(original_dataset, sampled_dataset):
    error_per_trace = []
    traces = []
    for user in original_dataset.keys():
        for video in original_dataset[user].keys():
            integ_yaws_orig = 0
            integ_pitchs_orig = 0
            for count, sample in enumerate(original_dataset[user][video]):
                if count == 0:
                    prev_sample = original_dataset[user][video][0]
                else:
                    dt = sample['sec'] - prev_sample['sec']
                    integ_yaws_orig += sample['yaw'] * dt
                    integ_pitchs_orig += sample['pitch'] * dt
                    prev_sample = sample
            angles_per_video = recover_original_angles_from_quaternions_trace(sampled_dataset[user][video])
            integ_yaws_sampl = 0
            integ_pitchs_sampl = 0
            for count, sample in enumerate(angles_per_video):
                if count == 0:
                    prev_time = sampled_dataset[user][video][count, 0]
                else:
                    dt = sampled_dataset[user][video][count, 0] - prev_time
                    integ_yaws_sampl += angles_per_video[count, 0] * dt
                    integ_pitchs_sampl += angles_per_video[count, 1] * dt
                    prev_time = sampled_dataset[user][video][count, 0]
            error_per_trace.append(np.sqrt(np.power(integ_yaws_orig-integ_yaws_sampl, 2) + np.power(integ_pitchs_orig-integ_pitchs_sampl, 2)))
            traces.append({'user': user, 'video': video})
    return error_per_trace, traces

### Check if the quaternions are good
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

# ToDo Copied exactly from Xu_PAMI_18/Reading_Dataset (Author: Miguel Romero)
def recover_xyz_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        angles_per_video.append(sample_new)
    return np.concatenate((quaternions_trace[:, 0:1], np.array(angles_per_video)), axis=1)

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
# Return the dataset
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
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

# Store the dataset in xyz coordinates form into the folder_to_store
def store_dataset(xyz_dataset, folder_to_store):
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            video_folder = os.path.join(folder_to_store, video)
            # Create the folder for the video if it doesn't exist
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            path = os.path.join(video_folder, user)
            df = pd.DataFrame(xyz_dataset[user][video])
            df.to_csv(path, header=False, index=False)

def create_and_store_sampled_dataset(plot_comparison=False, plot_3d_traces=False):
    analyze_time_stamps()
    original_dataset = get_original_dataset()
    sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
    if plot_comparison:
        compare_sample_vs_original(original_dataset, sampled_dataset)
    xyz_dataset = get_xyz_dataset(sampled_dataset)
    if plot_3d_traces:
        plot_all_traces_in_3d(xyz_dataset)
    store_dataset(xyz_dataset, OUTPUT_FOLDER)

# ToDo Copied exactly from Extract_Saliency/panosalnet
def post_filter(_img):
    result = np.copy(_img)
    result[:3, :3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result

def create_saliency_maps():
    salient_ds_dict = load_dataset()
    for video in salient_ds_dict['360net'].keys():
        sal_per_vid = {}
        video_sal_folder = os.path.join(OUTPUT_SALIENCY_FOLDER, video)
        if not os.path.exists(video_sal_folder):
            os.makedirs(video_sal_folder)

        time_stamps = salient_ds_dict['360net'][video]['timestamp']
        id_sort_tstamps = np.argsort(time_stamps)

        time_stamps_by_rate = np.arange(time_stamps[id_sort_tstamps[0]], time_stamps[id_sort_tstamps[-1]] + SAMPLING_RATE / 2.0, SAMPLING_RATE)

        for tstap_id, sampled_timestamp in enumerate(time_stamps_by_rate):
            # get the saliency with closest time-stamp
            sal_id = np.argmin(np.power(time_stamps-sampled_timestamp, 2.0))
            saliency = salient_ds_dict['360net'][video]['salient'][sal_id]
            salient = cv2.resize(saliency, (NUM_TILES_WIDTH, NUM_TILES_HEIGHT))

            salient = (salient * 1.0 - salient.min())
            salient = (salient / salient.max()) * 255
            salient = post_filter(salient)

            frame_id = "%03d" % (tstap_id+1)
            sal_per_vid[frame_id] = salient

            output_file = os.path.join(video_sal_folder, frame_id+'.jpg')
            cv2.imwrite(output_file, salient)
            print('saved image %s' % (output_file))

        pickle.dump(sal_per_vid, open(os.path.join(video_sal_folder, video), 'wb'))

# From MM_18 paper:
# We use 5 videos from our dataset to model training and another 4 videos for model validation
def split_in_train_and_test():
    from SampledDataset import get_video_ids, get_user_ids, get_users_per_video
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    videos = get_video_ids(OUTPUT_FOLDER)
    videos_train = np.random.choice(videos, 5, replace=False)
    videos_test = np.setdiff1d(videos, videos_train)
    print(videos_train)
    print(videos_test)


def plot_traces():
    dataset = {}
    # load data from pickle file.
    salient_ds_dict = load_dataset()
    # for video_id, video in enumerate(salient_ds_dict['360net'].keys()):
    for video_id, video in enumerate(['1', '2', '3', '4', '5', '6', '7', '8', '0']):
        time_stamps = salient_ds_dict['360net'][video]['timestamp']
        for user_id in range(len(salient_ds_dict['360net'][video]['headpos'])):
            print('get head positions from original dataset', 'video', video_id, '/', len(salient_ds_dict['360net'].keys()), 'user', user_id, '/', len(salient_ds_dict['360net'][video]['headpos']))
            user = str(user_id)
            if user not in dataset.keys():
                dataset[user] = {}
            positions_vector = salient_ds_dict['360net'][video]['headpos'][user_id]
            yaws = []
            pitchs = []
            yaws1 = []
            pitchs1 = []
            yaws2 = []
            pitchs2 = []
            time_stamps1 = []
            time_stamps2 = []
            flag_second_trace = False
            for count, pos_vec in enumerate(positions_vector):
                yaw_true, pitch_true = vector_to_ang(pos_vec)
                if count == 0:
                    yaws1.append(yaw_true)
                    pitchs1.append(pitch_true)
                    time_stamps1.append(time_stamps[count])
                else:
                    if not flag_second_trace and (time_stamps[count] > time_stamps[count-1]):
                        yaws1.append(yaw_true)
                        pitchs1.append(pitch_true)
                        time_stamps1.append(time_stamps[count])
                    else:
                        flag_second_trace = True
                        yaws2.append(yaw_true)
                        pitchs2.append(pitch_true)
                        time_stamps2.append(time_stamps[count])
                yaws.append(yaw_true)
                pitchs.append(pitch_true)
            plt.plot(yaws, label='longitude')
            plt.plot(pitchs, label='latitude')
            plt.xlabel('increasing counter')
            plt.ylabel('degrees')
            plt.legend()
            plt.show()
            print('video name:', video)
            print('first set of time stamps:', time_stamps1)
            plt.plot(time_stamps1, yaws1, color='blue', linewidth=3.0, label='longitude1')
            plt.plot(time_stamps1, pitchs1, color='red', linewidth=3.0, label='latitude1')
            if flag_second_trace:
                plt.plot(time_stamps2, yaws2, color='orange', label='longitude2')
                plt.plot(time_stamps2, pitchs2, color='green', label='latitude2')
                print('second set of time stamps:', time_stamps2)
            plt.xlabel('time-stamp')
            plt.ylabel('degrees')
            plt.legend()
            plt.show()
            if video in ['1', '2', '4', '5', '6', '7']:
                if video == '1':
                    inters_sec = 20
                if video == '2':
                    inters_sec = 90
                if video == '4':
                    inters_sec = 50
                if video == '5':
                    inters_sec = 140
                if video == '6':
                    inters_sec = 30
                if video == '7':
                    inters_sec = 90
                sal_id = np.argmin(np.power(np.array(time_stamps1) - inters_sec, 2.0))
                print('id of a saliency map in the first set of time-stamps:', sal_id)
                sal_id2_tstamp = np.argmin(np.power(np.array(time_stamps2) - inters_sec, 2.0))
                sal_id2 = sal_id2_tstamp + len(time_stamps1)
                # print(time_stamps2[sal_id2_tstamp-4:sal_id2_tstamp+5])
                print('id of a saliency map in the second set of time-stamps:', sal_id2)
                # print(len(time_stamps), len(time_stamps1), len(time_stamps2))
                # print(video, user)
                print('plotting both saliency maps to compare and observe they are the same. So the traces are in fact splitted and concatenated.')
                saliency = salient_ds_dict['360net'][video]['salient'][sal_id]
                saliency2 = salient_ds_dict['360net'][video]['salient'][sal_id2]
                plt.subplot(1, 2, 1)
                plt.imshow(saliency)
                plt.subplot(1, 2, 2)
                plt.imshow(saliency2)
                plt.show()
            # # Sorted time-stamps
            # for id_sort in id_sort_tstamps:
            #     yaw_true, pitch_true = (vector_to_ang(positions_vector[id_sort]))
            #     samples.append({'sec': time_stamps[id_sort], 'yaw': yaw_true, 'pitch': pitch_true})
            # dataset[user][video] = samples
    return dataset

# Returns the maximum number of samples among all users (the length of the largest trace)
def get_max_num_samples_for_video(video, sampled_dataset, users_in_video):
    max_len = 0
    for user in users_in_video:
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
    users_per_video = get_users_per_video(OUTPUT_FOLDER)

    for enum_video, video in enumerate(videos):
        print('creating true saliency for video', video, '-', enum_video, '/', len(videos))
        real_saliency_for_video = []

        max_num_samples = get_max_num_samples_for_video(video, sampled_dataset, users_per_video[video])

        for x_i in range(max_num_samples):
            tileprobs_for_video_cartesian = []
            for user in users_per_video[video]:
                if len(sampled_dataset[user][video]) > x_i:
                    tileprobs_cartesian = from_position_to_tile_probability_cartesian(sampled_dataset[user][video][x_i, 1:])
                    tileprobs_for_video_cartesian.append(tileprobs_cartesian)
            tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
            real_saliency_cartesian = np.sum(tileprobs_for_video_cartesian, axis=0) / tileprobs_for_video_cartesian.shape[0]
            real_saliency_for_video.append(real_saliency_cartesian)
        real_saliency_for_video = np.array(real_saliency_for_video)

        true_sal_out_file = os.path.join(OUTPUT_TRUE_SALIENCY_FOLDER, video)
        np.save(true_sal_out_file, real_saliency_for_video)

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
    parser.add_argument('-split_traces', action="store_true", dest='_split_traces_and_store', help='Flag that tells if we want to create the files to split the traces into train and test.')
    parser.add_argument('-creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset.')
    parser.add_argument('-creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('-analyze_data', action="store_true", dest='_analyze_orig_data', help='Flag that tells if we want to verify if the tile probability is correctly computed.')
    parser.add_argument('-creat_cb_sal', action="store_true", dest='_create_cb_saliency', help='Flag that tells if we want to create the content-based saliency maps.')
    parser.add_argument('-compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('-plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')

    args = parser.parse_args()

    if args._split_traces_and_store:
        split_in_train_and_test()

    if args._analyze_orig_data:
        analyze_time_stamps()
        plot_traces()

    if args._create_sampled_dataset:
        create_and_store_sampled_dataset()

    if args._create_cb_saliency:
        create_saliency_maps()

    if args._compare_traces:
        original_dataset = get_original_dataset()
        sampled_dataset = load_sampled_dataset()
        compare_sample_vs_original(original_dataset, sampled_dataset)

    if args._plot_3d_traces:
        sampled_dataset = load_sampled_dataset()
        plot_all_traces_in_3d(sampled_dataset)

    #print('use this file to create sampled dataset or to create saliency maps or to get the division between train and test traces')

    if args._create_true_saliency:
        if os.path.isdir(OUTPUT_FOLDER):
            sampled_dataset = load_sampled_dataset()
            create_and_store_true_saliency(sampled_dataset)
        else:
            print('Please verify that the sampled dataset has been created correctly under the folder', OUTPUT_FOLDER)
