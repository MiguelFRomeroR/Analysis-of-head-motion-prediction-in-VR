import sys
sys.path.insert(0, './')

import os
import numpy as np
import pandas as pd
from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees, compute_orthodromic_distance, load_dict_from_csv, store_dict_as_csv
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import csv
import pickle
from SampledDataset import get_video_ids, get_user_ids
import argparse

ROOT_FOLDER = './Fan_NOSSDAV_17/dataset/'
OUTPUT_FOLDER = './Fan_NOSSDAV_17/sampled_dataset'
OUTPUT_TILE_PROB_FOLDER = './Fan_NOSSDAV_17/dataset/sensory/tile_replica'

FOLDER_IMAGES_SAL = 'content/saliency'
FOLDER_IMAGES_MOT = 'content/motion'
OUTPUT_SALIENCY_FOLDER = './Fan_NOSSDAV_17/extract_saliency/saliency'

OUTPUT_TRUE_SALIENCY_FOLDER = './Fan_NOSSDAV_17/true_saliency'

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

SAMPLING_RATE = 0.2
# From https://people.cs.nctu.edu.tw/~chuang/pubs/pdf/2017mmsys.pdf 360° Video Viewing Dataset:
# We divide each frame, which is mapped in equirectangular model, into 192x192 tiles, so there are *200* tiles in total.
# Then we number the tiles from upper-left to lower-right.
NUM_TILES_HEIGHT = 10
NUM_TILES_WIDTH = 20
# From https://people.cs.nctu.edu.tw/~chuang/pubs/pdf/2017mmsys.pdf 360° Video Viewing Dataset:
# We assume the FoVs are modeled by 100°x100° circles.
FOV_SIZE = 110.0

# -1 means that the most negative point is in the south pole, +1 means that the most negative point is in the north pole
PITCH_DIRECTION_PER_USER = {'user01': -1, 'user02': -1, 'user03': -1, 'user04': -1, 'user05': -1, 'user06': -1,
                            'user07': -1, 'user08': -1, 'user09': -1, 'user10': -1, 'user11': -1, 'user12': -1,
                            'user13': -1, 'user14': -1, 'user15': -1, 'user16': -1, 'user17': -1, 'user18': -1,
                            'user19': -1, 'user20': -1, 'user21': +1, 'user22': +1, 'user23': +1, 'user24': +1,
                            'user25': +1, 'user26': +1, 'user27': +1, 'user28': +1, 'user29': +1, 'user30': +1,
                            'user31': +1, 'user32': +1, 'user33': +1, 'user34': +1, 'user35': +1, 'user36': +1,
                            'user37': +1, 'user38': +1, 'user39': +1, 'user40': +1, 'user41': +1, 'user42': +1,
                            'user43': +1, 'user44': +1, 'user45': +1, 'user46': +1, 'user47': +1, 'user48': +1,
                            'user49': +1, 'user50': +1}

fps_per_video = {'drive': 0.03333333333333333, 'pacman': 0.03333333333333333, 'landscape': 0.03333333333333333, 'diving': 0.03333333333333333, 'game': 0.03333333333333333, 'ride': 0.03333333333333333, 'coaster2': 0.03333333333333333, 'coaster': 0.03333333333333333, 'sport': 0.03333333333333333, 'panel': 0.03333333333333333}

def get_orientations_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=', ')
    data = dataframe[['cal. yaw', 'cal. pitch']]
    return data.values

# ToDo Copied (changed the frame_id position from dataframe[1] to dataframe[0]) from Xu_CVPR_18/Reading_Dataset (Author: Miguel Romero)
def get_frame_indices_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=', ')
    data = dataframe['no. frames']
    return data.values

# returns the frame rate of a video using openCV
# ToDo Copied (changed videoname to videoname+'_saliency' and video_path folder) from Xu_CVPR_18/Reading_Dataset (Author: Miguel Romero)
def get_frame_rate(videoname, use_dict=False):
    if use_dict:
        return 1.0/fps_per_video[videoname]
    video_mp4 = videoname+'_saliency.mp4'
    video_path = os.path.join(ROOT_FOLDER, 'content/saliency', video_mp4)
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset():
    dataset = {}
    for root, directories, files in os.walk(os.path.join(ROOT_FOLDER, 'sensory/orientation')):
        for enum_trace, filename in enumerate(files):
            print('get head orientations from original dataset trace', enum_trace, '/', len(files))
            splitted_filename = filename.split('_')
            user = splitted_filename[1]
            video = splitted_filename[0]
            if user not in dataset.keys():
                dataset[user] = {}
            file_path = os.path.join(root, filename)
            positions = get_orientations_for_trace(file_path)
            frame_ids = get_frame_indices_for_trace(file_path)
            video_rate = 1.0 / get_frame_rate(video, use_dict=True)
            samples = []
            for pos, frame_id in zip(positions, frame_ids):
                samples.append({'sec': frame_id*video_rate, 'yaw': pos[0], 'pitch': pos[1]})
            dataset[user][video] = samples
    return dataset

# The viewer orientations, including yaw, pitch and roll in the range of [-180, 180].
# Transform the original yaw degrees from range [-180, 180] to the range [0, 2pi]
# Transform the original pitch degrees from range [-180, 180] to the range [0, pi]
def transform_the_degrees_in_range(yaw, pitch):
    yaw = (yaw/360.0+0.5)*2*np.pi
    pitch = (pitch/180.0+0.5)*np.pi
    return yaw, pitch

# Performs the opposite transformation than transform_the_degrees_in_range
# Transform the yaw values from range [0, 2pi] to range [-180, 180]
# Transform the pitch values from range [0, pi] to range [-90, 90]
def transform_the_radians_to_original(yaw, pitch):
    yaw = (yaw/(2*np.pi)-0.5)*360.0
    pitch = (pitch/np.pi-0.5)*180.0
    return yaw, pitch

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def create_sampled_dataset(original_dataset, rate):
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating sampled dataset', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
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

def read_tile_info(video, user):
    filename = os.path.join(ROOT_FOLDER, 'sensory/tile', video + '_' + user + '_tile.csv')
    csv.register_dialect('nospaces', delimiter=',', skipinitialspace=True)
    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect='nospaces')
        # Skip the headers
        next(reader, None)
        tiles_per_trace = []
        for row in reader:
            viewed_tiles = np.zeros(NUM_TILES_HEIGHT*NUM_TILES_WIDTH, dtype=int)
            # subtract 1 to have the indices starting from zero
            tile_indices = np.array(row[1:]).astype(int) - 1
            viewed_tiles[tile_indices] = 1
            tiles_per_trace.append(viewed_tiles)
    csvFile.close()
    tiles_per_trace = np.array(tiles_per_trace)
    return tiles_per_trace.reshape(-1, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)

# From https://people.cs.nctu.edu.tw/~chuang/pubs/pdf/2017mmsys.pdf 360° Video Viewing Dataset:
# While view orientation log files give the center of viewer's FoVs, determining which tiles are needed to render the
# FoVs equire extra calculations. We assume the FoVs are modeled by 100°x100° circles.
def from_position_to_tile_probability_cartesian(pos):
    yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, NUM_TILES_WIDTH, endpoint=False), np.linspace(0, 1, NUM_TILES_HEIGHT, endpoint=False))
    yaw_grid += 1.0 / (2.0 * NUM_TILES_WIDTH)
    pitch_grid += 1.0 / (2.0 * NUM_TILES_HEIGHT)
    yaw_grid = yaw_grid * 2*np.pi
    pitch_grid = pitch_grid * np.pi
    x_grid, y_grid, z_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)
    great_circle_distance = np.arccos(np.maximum(np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0), -1.0))
    binary_orth = np.where(great_circle_distance < (((FOV_SIZE/2.0)/180.0)*np.pi), 1, 0)
    return binary_orth


def create_and_store_tile_probability_replica(original_dataset):
    if not os.path.exists(OUTPUT_TILE_PROB_FOLDER):
        os.makedirs(OUTPUT_TILE_PROB_FOLDER)
    for enum_user, user in enumerate(original_dataset.keys()):
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating tiles for', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            tile_prob_for_trace = []
            for sample_id, sample in enumerate(original_dataset[user][video]):
                sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
                sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
                gen_tile_prob_cartesian = from_position_to_tile_probability_cartesian(sample_new)
                tile_prob_for_trace.append(gen_tile_prob_cartesian)
            filename = '%s_%s_created_tile.npy' % (video, user)
            file_path = os.path.join(OUTPUT_TILE_PROB_FOLDER, filename)
            np.save(file_path, np.array(tile_prob_for_trace))

def read_replica_tile_info(video, user):
    replica_filename = '%s_%s_created_tile.npy' % (video, user)
    file_path = os.path.join(OUTPUT_TILE_PROB_FOLDER, replica_filename)
    replica_tile_prob_for_trace = np.load(file_path)
    return replica_tile_prob_for_trace

# With this function we can verify that the replica (created) tile probability is similar to the original (by the authors of the paper in NOSSDAV17)
# You will find that appart from the trace: user: 17, video: sport (which has an error greater than 25%, probably due to an error when creating the tile probabilities).
# the other traces have a small rounding error (~8%) at most, it can be observed in the plots of this function.
# Observe also the way PITCH_DIRECTION_PER_USER is used in this function, meaning that when going from north pole to south pole,
# for some traces (users between 1 and 20) the pitch ranges from 90 to -90, and
# for other traces (users between 21 and 50) the pitch ranges from -90 to 90.
def verify_tile_probability_replica(original_dataset, use_pitch_direction=True):
    errors = []
    traces = []
    for enum_user, user in enumerate(original_dataset.keys()):
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('comparing tiles from', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            orig_tiles_map = read_tile_info(video, user)
            repl_tiles_map = read_replica_tile_info(video, user)
            for sample_id, sample in enumerate(original_dataset[user][video]):
                orig_tile_prob = orig_tiles_map[sample_id]
                repl_tile_prob = repl_tiles_map[sample_id]
                if use_pitch_direction and PITCH_DIRECTION_PER_USER[user] == -1:
                    repl_tile_prob = repl_tile_prob[::-1]
                error = np.sum(np.abs(repl_tile_prob - orig_tile_prob))
                errors.append(error)
                traces.append({'user': user, 'video': video, 'sample_id': sample_id})
    indices = np.argsort(-np.array(errors))
    print('-------------------------------------------------')
    for trace_id in indices:
        trace = traces[trace_id]
        user = trace['user']
        video = trace['video']
        if errors[trace_id] > 15:
            print('user', user, 'video', video, 'error', errors[trace_id]/(NUM_TILES_HEIGHT*NUM_TILES_WIDTH))
    print('-------------------------------------------------')
    if use_pitch_direction:
        print('From the previous data, we can see how only the trace corresponding to Video: Sport and User: 17 has 25% of the replicated tiles wrong. '
              '\nFor the rest of the traces less than 8% of the replicated tiles are wrong (from our algorithm to replicate the tile generation of NOSSDAV17).'
              '\nIn the following plots you can observe in the top row the plots of the replicated tiles (with our algorithm) versus the original tiles in the bottom row.'
              '\nNote that the parameter flag use_pitch_direction is true, meaning that we had to change the position of north and south pole to generate these tiles, \n'
              'check the usage of the variable PITCH_DIRECTION_PER_USER in the file ./Fan_NOSSDAV_17/Read_Dataset.py and set the flag use_pitch_direction to false.')
    else:
        print('Did you see the difference?, when setting the variable to true we had an error of 8% at most. Now look the plots.')
    count = 0
    for trace_id in indices:
        trace = traces[trace_id]
        user = trace['user']
        video = trace['video']
        sample_id = trace['sample_id']
        # We exclude the user17 and video sport, because the error on this trace is high (reaching even more than 25% of the tiles wrong)
        if user != 'user17' and video != 'sport':
            count += 1
            orig_tiles_map = read_tile_info(video, user)
            repl_tiles_map = read_replica_tile_info(video, user)
            orig_tile_prob = orig_tiles_map[sample_id]
            repl_tile_prob = repl_tiles_map[sample_id]
            if use_pitch_direction and PITCH_DIRECTION_PER_USER[user] == -1:
                repl_tile_prob = repl_tile_prob[::-1]
            plt.subplot(2, 10, (count % 10)+1)
            plt.imshow(repl_tile_prob)
            plt.title('Replica')
            plt.subplot(2, 10, (count % 10)+11)
            plt.imshow(orig_tile_prob)
            plt.title('True')
            print('user', user, 'video', video, 'error', errors[trace_id]/(NUM_TILES_HEIGHT*NUM_TILES_WIDTH))
            if count % 10 == 0:
                plt.show()

# After using the function verify_tile_probability_replica, we found that for users from 1 to 20 the pitch value seems
# to be upside down, and that the trace for (user17, sport) has a strange behavior, for this reason we decided to use
# only users from 21 to 50 for our experiments, since the dataset is more consistent in these traces.
def filter_dataset_strange_traces(original_dataset):
    filtered_dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        if PITCH_DIRECTION_PER_USER[user] == 1:
            filtered_dataset[user] = {}
            for enum_video, video in enumerate(original_dataset[user].keys()):
                    filtered_dataset[user][video] = original_dataset[user][video]
    return filtered_dataset

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

# ToDo, transform in a class this is the main function of this file
def create_and_store_sampled_dataset(plot_comparison=False, plot_3d_traces=False):
    original_dataset = get_original_dataset()
    # create_and_store_tile_probability_replica(original_dataset)
    # verify_tile_probability_replica(original_dataset)
    # After using the function above we found that for users from 1 to 20 the pitch value seems to be upside down, and
    # that the trace for (user17, sport) has a strange behavior, for this reason we decided to use only users
    # from 21 to 50 for our experiments, since the dataset is more consistent in these traces.
    filtered_original_dataset = filter_dataset_strange_traces(original_dataset)
    # verify_tile_probability_replica(filtered_original_dataset, use_pitch_direction=False)
    filtered_sampled_dataset = create_sampled_dataset(filtered_original_dataset, rate=SAMPLING_RATE)
    if plot_comparison or plot_3d_traces:
        error_per_trace, traces = compare_integrals(filtered_original_dataset, filtered_sampled_dataset)
    if plot_comparison:
        compare_sample_vs_original(filtered_original_dataset, filtered_sampled_dataset)
    filtered_xyz_dataset = get_xyz_dataset(filtered_sampled_dataset)
    if plot_3d_traces:
        plot_all_traces_in_3d(filtered_xyz_dataset)
    store_dataset(filtered_xyz_dataset, OUTPUT_FOLDER)

# ToDo Copied exactly from Extract_Saliency/panosalnet
def post_filter(_img):
    result = np.copy(_img)
    result[:3, :3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result

def create_saliency_maps():
    folder_sal_map = os.path.join(ROOT_FOLDER, FOLDER_IMAGES_SAL)
    folder_mot_vec = os.path.join(ROOT_FOLDER, FOLDER_IMAGES_MOT)
    videos = [d.split('_')[0] for d in os.listdir(folder_sal_map) if os.path.isdir(os.path.join(folder_sal_map, d))]

    for video in videos:
        sal_per_vid = {}
        video_sal_folder = os.path.join(OUTPUT_SALIENCY_FOLDER, video)
        if not os.path.exists(video_sal_folder):
            os.makedirs(video_sal_folder)
        video_input_salmap_folder = os.path.join(folder_sal_map, video + '_saliency')
        video_input_motvec_folder = os.path.join(folder_mot_vec, video + '_motion')
        for image_name in os.listdir(video_input_salmap_folder):
            file_dir_sal = os.path.join(video_input_salmap_folder, image_name)
            file_dir_mot = os.path.join(video_input_motvec_folder, image_name)
            img_sal = cv2.imread(file_dir_sal, 0).astype(np.float)
            img_mot = cv2.imread(file_dir_mot, 0).astype(np.float)

            salient = (np.array(img_mot)+np.array(img_sal))/512.0

            salient = (salient * 1.0 - salient.min())
            salient = (salient / salient.max()) * 255
            salient = post_filter(salient)

            frame_id = image_name.split('.')[0].split('_')[-1]
            sal_per_vid[frame_id] = salient
            output_file = os.path.join(video_sal_folder, image_name)

            print('saved image %s' % (output_file))
            cv2.imwrite(output_file, salient)

        pickle.dump(sal_per_vid, open(os.path.join(video_sal_folder, video), 'wb'))

# From NOSSDAV_17:
# We use the traces from 12 viewers (training set) to train the two proposed fixation prediction networks.
# Among the traces, we randomly divide them into 80% and 20% for training and validation, respectively.
# We give the training and validation results of the two models in Tables 1 and 2.
NUM_USERS_EXPERIMENT = 12
PROPORTION_TRAIN_SET = 0.8
def get_traces_for_train_and_test():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    videos = get_video_ids(OUTPUT_FOLDER)
    all_users = get_user_ids(OUTPUT_FOLDER)
    users = np.random.choice(all_users, NUM_USERS_EXPERIMENT, replace=False)

    traces_index = []
    traces = []
    count_traces = 0
    for video in videos:
        for user in users:
            traces.append({'video': video, 'user': user})
            traces_index.append(count_traces)
            count_traces += 1
    traces_index = np.array(traces_index)
    np.random.shuffle(traces_index)
    num_train_traces = int(len(traces_index) * PROPORTION_TRAIN_SET)

    train_traces_ids = traces_index[:num_train_traces]
    test_traces_ids = traces_index[num_train_traces:]

    train_traces = []
    for trace_id in train_traces_ids:
        train_traces.append(traces[trace_id])
    test_traces = []
    for trace_id in test_traces_ids:
        test_traces.append(traces[trace_id])
    return train_traces, test_traces

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

def split_traces_and_store():
    train_traces, test_traces = get_traces_for_train_and_test()
    store_dict_as_csv('train_set', ['user', 'video'], train_traces)
    store_dict_as_csv('test_set', ['user', 'video'], test_traces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the input parameters to parse the dataset.')
    parser.add_argument('-split_traces', action="store_true", dest='_split_traces_and_store', help='Flag that tells if we want to create the files to split the traces into train and test.')
    parser.add_argument('-creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset.')
    parser.add_argument('-creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('-analyze_data', action="store_true", dest='_analyze_orig_data', help='Flag that tells if we want to verify if the tile probability is correctly computed.')
    parser.add_argument('-creat_cb_sal', action="store_true", dest='_create_cb_saliency', help='Flag that tells if we want to create the content-based saliency maps.')
    parser.add_argument('-compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('-plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')
    parser.add_argument('-create_tile_replica', action="store_true", dest='_create_tile_replica', help='Flag that tells if we want to create the tile replica using our tile mapping function.')

    args = parser.parse_args()

    if args._split_traces_and_store:
        split_traces_and_store()

    if args._analyze_orig_data:
        original_dataset = get_original_dataset()
        if not os.path.isdir(OUTPUT_TILE_PROB_FOLDER):
            print('Folder Tile Probability Replica is not created.')
            create_and_store_tile_probability_replica(original_dataset)
        verify_tile_probability_replica(original_dataset, use_pitch_direction=True)

    if args._create_sampled_dataset:
        create_and_store_sampled_dataset()

    if args._create_cb_saliency:
        create_saliency_maps()

    if args._compare_traces:
        original_dataset = get_original_dataset()
        filtered_original_dataset = filter_dataset_strange_traces(original_dataset)
        filtered_sampled_dataset = load_sampled_dataset()
        compare_sample_vs_original(filtered_original_dataset, filtered_sampled_dataset)

    if args._plot_3d_traces:
        sampled_dataset = load_sampled_dataset()
        plot_all_traces_in_3d(sampled_dataset)

    if args._create_true_saliency:
        if os.path.isdir(OUTPUT_FOLDER):
            sampled_dataset = load_sampled_dataset()
            create_and_store_true_saliency(sampled_dataset)
        else:
            print('Please verify that the sampled dataset has been created correctly under the folder', OUTPUT_FOLDER)

    if args._create_tile_replica:
        original_dataset = get_original_dataset()
        create_and_store_tile_probability_replica(original_dataset)