import os
import numpy as np
import pandas as pd
from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import cv2

GAZE_TXT_FOLDER = '/home/twipsy/PycharmProjects/UniformHeadMotionDataset/Xu_CVPR_18/dataset/Gaze_txt_files'
VIDEOS_PATH = '/media/twipsy/ADATAHD7104/UniformHeadMotionDataset/Xu_CVPR_18/dataset/Videos'
# From the paper in http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Gaze_Prediction_in_CVPR_2018_paper.pdf
# "the interval between two neighboring frames in our experiments corresponds to 0.2 seconds"
SAMPLING_RATE = 0.2

def get_gaze_positions_for_trace(filename):
    dataframe = pd.read_csv(filename, header=None, sep=",", engine='python')
    data = dataframe[[6, 7]]
    return data.values

def get_head_positions_for_trace(filename):
    dataframe = pd.read_csv(filename, header=None, sep=",", engine='python')
    data = dataframe[[3, 4]]
    return data.values

def get_indices_for_trace(filename):
    dataframe = pd.read_csv(filename, header=None, sep=",", engine='python')
    data = dataframe[1]
    return data.values


# returns the frame rate of a video using openCV
def get_frame_rate(videoname):
    video_mp4 = videoname+'.mp4'
    video_path = os.path.join(VIDEOS_PATH, video_mp4)
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

# returns the number of frames of a video using openCV
def get_frame_count(videoname):
    video_mp4 = videoname+'.mp4'
    video_path = os.path.join(VIDEOS_PATH, video_mp4)
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        count = video.get(cv2.cv.CAP_PROP_FRAME_COUNT)
    else:
        count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    return count

# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset_gaze():
    dataset = {}
    # ToDo replaced this to Debug
    for root, directories, files in os.walk(os.path.join(GAZE_TXT_FOLDER)):
        for enum_user, user in enumerate(directories):
            dataset[user] = {}
            for r_2, d_2, sub_files in os.walk(os.path.join(GAZE_TXT_FOLDER, user)):
                for enum_video, video_txt in enumerate(sub_files):
                    video = video_txt.split('.')[0]
                    print('get gaze positions from original dataset', 'user', enum_user, '/', len(directories), 'video', enum_video, '/', len(sub_files))
                    filename = os.path.join(GAZE_TXT_FOLDER, user, video_txt)
                    positions = get_gaze_positions_for_trace(filename)
                    frame_ids = get_indices_for_trace(filename)
                    video_rate = 1.0 / get_frame_rate(video)
                    samples = []
                    for pos, frame_id in zip(positions, frame_ids):
                        # ToDo Check if gaze position x corresponds to yaw and gaze position y corresponds to pitch
                        samples.append({'sec':frame_id*video_rate, 'yaw':pos[0], 'pitch':pos[1]})
                    dataset[user][video] = samples
    return dataset

# The HM data takes the position in the panorama image, they are fractional from 0.0 to 1.0 with respect to the panorama image
# and computed from left bottom corner.
# Thus the longitudes ranges from 0 to 1, and the latitude ranges from 0 to 1.
# The subject starts to watch the video at position yaw=0.5, pitch=0.5.
## In other words, yaw = 0.5, pitch = 0.5 is equal to the position (1, 0, 0) in cartesian coordinates
# Pitching the head up results in a positive pitch value.
## In other words, yaw = Any, pitch = 1.0 is equal to the position (0, 0, 1) in cartesian coordinates
### We will transform these coordinates so that
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
def transform_the_degrees_in_range(yaw, pitch):
    yaw = yaw*2*np.pi
    pitch = (-pitch+1)*np.pi
    return yaw, pitch

# Performs the opposite transformation than transform_the_degrees_in_range
def transform_the_radians_to_original(yaw, pitch):
    yaw = (yaw)/(2*np.pi)
    pitch = (-(pitch/np.pi)+1)
    return yaw, pitch

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

# Check if the quaternions are good
def compare_sample_vs_original(original_dataset, sampled_dataset, plot=False):
    new_dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        new_dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('recovering eulerian angles from subsampled trace', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            new_dataset[user][video] = {}
            pitchs = []
            yaws = []
            times = []
            for sample in original_dataset[user][video]:
                times.append(sample['sec'])
                yaws.append(sample['yaw'])
                pitchs.append(sample['pitch'])
            angles_per_video = recover_original_angles_from_quaternions_trace(sampled_dataset[user][video])
            new_dataset[user][video]['times'] = sampled_dataset[user][video][:, 0]
            new_dataset[user][video]['yaw'] = angles_per_video[:, 0]
            new_dataset[user][video]['pitch'] = angles_per_video[:, 1]
            if plot:
                plt.subplot(1, 2, 1)
                plt.plot(times, yaws, label='yaw')
                plt.plot(times, pitchs, label='pitch')
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(new_dataset[user][video]['times'], new_dataset[user][video]['yaw'], label='yaw')
                plt.plot(new_dataset[user][video]['times'], new_dataset[user][video]['pitch'], label='pitch')
                plt.legend()
                plt.show()
    return new_dataset

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

if __name__ == "__main__":
    original_dataset_gaze = get_original_dataset_gaze()
    sampled_dataset_gaze_quat = create_sampled_dataset(original_dataset_gaze, SAMPLING_RATE)
    sampled_dataset_gaze_eulerian = compare_sample_vs_original(original_dataset_gaze, sampled_dataset_gaze_quat, plot=False)

    # Here you can store the dataset, I printed it so you can observe its structure
    print('users', sampled_dataset_gaze_eulerian.keys())
    for user in sampled_dataset_gaze_eulerian.keys():
        print('videos for user', user, ':', sampled_dataset_gaze_eulerian[user].keys())
        for video in sampled_dataset_gaze_eulerian[user]:
            print('shape of yaw in trace', user, video, ':', sampled_dataset_gaze_eulerian[user][video]['yaw'].shape)
            print('shape of pitch in trace', user, video, ':', sampled_dataset_gaze_eulerian[user][video]['pitch'].shape)
            print('times in trace', user, video, ':', sampled_dataset_gaze_eulerian[user][video]['times'])
