import sys
sys.path.insert(0, './')

import os
import numpy as np
import pandas as pd
from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees, compute_orthodromic_distance, store_dict_as_csv
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse

ROOT_FOLDER = './David_MMSys_18/dataset/'
OUTPUT_FOLDER = './David_MMSys_18/sampled_dataset'

OUTPUT_FOLDER_ORIGINAL_XYZ = './David_MMSys_18/original_dataset_xyz'

OUTPUT_TRUE_SALIENCY_FOLDER = './David_MMSys_18/true_saliency'

SAMPLING_RATE = 0.2

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

VIDEOS = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']

# From "David_MMSys_18/dataset/Videos/Readme_Videos.md"
# Text files are provided with scanpaths from head movement with 100 samples per observer
NUM_SAMPLES_PER_USER = 100

def get_orientations_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
    data = dataframe[[' longitude', ' latitude']]
    return data.values

def get_time_stamps_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
    data = dataframe[' start timestamp']
    return data.values

# returns the frame rate of a video using openCV
# ToDo Copied (changed videoname to videoname+'_saliency' and video_path folder) from Xu_CVPR_18/Reading_Dataset (Author: Miguel Romero)
def get_frame_rate(videoname):
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
    for root, directories, files in os.walk(os.path.join(ROOT_FOLDER, 'Videos/H/Scanpaths')):
        for enum_trace, filename in enumerate(files):
            print('get head orientations from original dataset traces for video', enum_trace, '/', len(files))
            splitted_filename = filename.split('_')
            video = '_'.join(splitted_filename[:-1])
            file_path = os.path.join(root, filename)
            positions_all_users = get_orientations_for_trace(file_path)
            time_stamps_all_users = get_time_stamps_for_trace(file_path)
            num_users = int(positions_all_users.shape[0]/NUM_SAMPLES_PER_USER)
            for user_id in range(num_users):
                user = str(user_id)
                if user not in dataset.keys():
                    dataset[user] = {}
                positions = positions_all_users[user_id * NUM_SAMPLES_PER_USER:(user_id + 1) * (NUM_SAMPLES_PER_USER)]
                time_stamps = time_stamps_all_users[user_id * NUM_SAMPLES_PER_USER:(user_id + 1) * (NUM_SAMPLES_PER_USER)]
                samples = []
                for pos, t_stamp in zip(positions, time_stamps):
                    samples.append({'sec': t_stamp/1000.0, 'yaw': pos[0], 'pitch': pos[1]})
                dataset[user][video] = samples
    return dataset

# From "dataset/Videos/Readme_Videos.md"
# Latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied according to the
# resolution of the desired equi-rectangular image output dimension).
# Participants started exploring omnidirectional contents either from an implicit longitudinal center
# (0-degrees and center of the equirectangular projection) or from the opposite longitude (180-degrees).
def transform_the_degrees_in_range(yaw, pitch):
    yaw = yaw*2*np.pi
    pitch = pitch*np.pi
    return yaw, pitch

# Performs the opposite transformation than transform_the_degrees_in_range
# Transform the yaw values from range [0, 2pi] to range [0, 1]
# Transform the pitch values from range [0, pi] to range [0, 1]
def transform_the_radians_to_original(yaw, pitch):
    yaw = yaw/(2*np.pi)
    pitch = pitch/np.pi
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
# def plot_all_traces_in_3d(xyz_dataset, error_per_trace, traces):
#     indices = np.argsort(-np.array(error_per_trace))
#     for trace_id in indices:
#         trace = traces[trace_id]
#         user = trace['user']
#         video = trace['video']
#         plot_3d_trace(xyz_dataset[user][video][:, 1:])

def plot_all_traces_in_3d(xyz_dataset):
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            plot_3d_trace(xyz_dataset[user][video][:, 1:], user, video)

# ToDo, transform in a class this is the main function of this file
def create_and_store_sampled_dataset(plot_comparison=False, plot_3d_traces=False):
    original_dataset = get_original_dataset()
    sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
    if plot_comparison:
        compare_sample_vs_original(original_dataset, sampled_dataset)
    xyz_dataset = get_xyz_dataset(sampled_dataset)
    if plot_3d_traces:
        plot_all_traces_in_3d(xyz_dataset)
    store_dataset(xyz_dataset, OUTPUT_FOLDER)

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

    for enum_video, video in enumerate(VIDEOS):
        print('creating true saliency for video', video, '-', enum_video, '/', len(VIDEOS))
        real_saliency_for_video = []
        for x_i in range(NUM_SAMPLES_PER_USER):
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

def get_most_salient_points_per_video():
    from skimage.feature import peak_local_max
    most_salient_points_per_video = {}
    for video in VIDEOS:
        saliencies_for_video_file = os.path.join(OUTPUT_TRUE_SALIENCY_FOLDER, video+'.npy')
        saliencies_for_video = np.load(saliencies_for_video_file)
        most_salient_points_in_video = []
        for id, sal in enumerate(saliencies_for_video):
            coordinates = peak_local_max(sal, exclude_border=False, num_peaks=5)
            coordinates_normalized = coordinates / np.array([NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL])
            coordinates_radians = coordinates_normalized * np.array([np.pi, 2.0*np.pi])
            cartesian_pts = np.array([eulerian_to_cartesian(sample[1], sample[0]) for sample in coordinates_radians])
            most_salient_points_in_video.append(cartesian_pts)
        most_salient_points_per_video[video] = np.array(most_salient_points_in_video)
    return  most_salient_points_per_video

def predict_most_salient_point(most_salient_points, current_point):
    pred_window_predicted_closest_sal_point = []
    for id, most_salient_points_per_fut_frame in enumerate(most_salient_points):
        distances = np.array([compute_orthodromic_distance(current_point, most_sal_pt) for most_sal_pt in most_salient_points_per_fut_frame])
        closest_sal_point = np.argmin(distances)
        predicted_closest_sal_point = most_salient_points_per_fut_frame[closest_sal_point]
        pred_window_predicted_closest_sal_point.append(predicted_closest_sal_point)
    return pred_window_predicted_closest_sal_point

def most_salient_point_baseline(dataset):
    most_salient_points_per_video = get_most_salient_points_per_video()
    error_per_time_step = {}
    for enum_user, user in enumerate(dataset.keys()):
        for enum_video, video in enumerate(dataset[user].keys()):
            print('computing error for user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
            trace = dataset[user][video]
            for x_i in range(5, 75):
                model_prediction = predict_most_salient_point(most_salient_points_per_video[video][x_i+1:x_i+25+1], trace[x_i, 1:])
                for t in range(25):
                    if t not in error_per_time_step.keys():
                        error_per_time_step[t] = []
                    error_per_time_step[t].append(compute_orthodromic_distance(trace[x_i+t+1, 1:], model_prediction[t]))
    for t in range(25):
        print(t*0.2, np.mean(error_per_time_step[t]))

def create_original_dataset_xyz(original_dataset):
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating original dataset in format for', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            data_per_video = []
            for sample in original_dataset[user][video]:
                sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
                sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
                data_per_video.append([sample['sec'], sample_new[0], sample_new[1], sample_new[2]])
            dataset[user][video] = np.array(data_per_video)
    return dataset

def create_and_store_original_dataset():
    original_dataset = get_original_dataset()
    original_dataset_xyz = create_original_dataset_xyz(original_dataset)
    store_dataset(original_dataset_xyz, OUTPUT_FOLDER_ORIGINAL_XYZ)

def get_traces_for_train_and_test():
    videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight',
              '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino',
              '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']

    # Fixing random state for reproducibility
    np.random.seed(7)

    videos_ids = np.arange(len(videos))
    users = np.arange(57)

    # Select at random the users for each set
    np.random.shuffle(users)
    num_train_users = int(len(users) * 0.5)
    users_train = users[:num_train_users]
    users_test = users[num_train_users:]

    videos_ids_train = [1, 3, 5, 7, 8, 9, 11, 14, 16, 18]
    videos_ids_test = [0, 2, 4, 13, 15]

    train_traces = []
    for video_id in videos_ids_train:
        for user_id in users_train:
            train_traces.append({'video': videos[video_id], 'user': str(user_id)})

    test_traces = []
    for video_id in videos_ids_test:
        for user_id in users_test:
            test_traces.append({'video': videos[video_id], 'user': str(user_id)})

    return train_traces, test_traces

def split_traces_and_store():
    train_traces, test_traces = get_traces_for_train_and_test()
    store_dict_as_csv('train_set', ['user', 'video'], train_traces)
    store_dict_as_csv('test_set', ['user', 'video'], test_traces)

if __name__ == "__main__":
    #print('use this file to create sampled dataset or to create true_saliency or to create original dataset in xyz format')

    parser = argparse.ArgumentParser(description='Process the input parameters to parse the dataset.')
    parser.add_argument('-split_traces', action="store_true", dest='_split_traces_and_store', help='Flag that tells if we want to create the files to split the traces into train and test.')
    parser.add_argument('-creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset.')
    parser.add_argument('-creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('-compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('-plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')

    args = parser.parse_args()

    if args._split_traces_and_store:
        split_traces_and_store()

    # create_and_store_original_dataset()

    if args._create_sampled_dataset:
        create_and_store_sampled_dataset()

    if args._compare_traces:
        original_dataset = get_original_dataset()
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

    # sampled_dataset = load_sampled_dataset()
    # most_salient_point_baseline(sampled_dataset)
