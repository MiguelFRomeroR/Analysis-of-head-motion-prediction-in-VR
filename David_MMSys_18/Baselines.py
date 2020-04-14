import os
import pandas as pd
import numpy as np
from Utils import compute_orthodromic_distance, cartesian_to_eulerian, radian_to_degrees
# import matplotlib.pyplot as plt
from position_only_baseline import create_pos_only_model

# ToDo import from training_procedure (for now we just copied the function)
# from training_procedure import transform_batches_cartesian_to_normalized_eulerian
def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def normalized_eulerian_to_cartesian(theta, phi):
    theta = theta * 2*np.pi
    phi = phi * np.pi
    # ToDo here we can use directly eulerian_to_cartesian
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z])

EXPERIMENT_FOLDER = './David_MMSys_18/sampled_dataset'
TRAIN_TEST_SET_FILE = 'train_test_set.xlsx'


def load_gaze_dataset():
    list_of_videos = os.listdir(EXPERIMENT_FOLDER)
    gaze_dataset = {}
    for video in list_of_videos:
        for user in os.listdir(os.path.join(EXPERIMENT_FOLDER, video)):
            if user not in gaze_dataset.keys():
                gaze_dataset[user] = {}
            path = os.path.join(EXPERIMENT_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            gaze_dataset[user][video] = data.values
    return gaze_dataset


# Computes orthodromic distance categorized per video for the baseline (using the position at t for prediction up to time t+prediction_horizon)
# for all the samples in the dataset with video belonging to videos_list
def compute_no_motion_baseline_error_xyz(dataset, videos_list, users_list, history_window, prediction_horizon):
    errors_per_video = {}
    for enum_user, user in enumerate(dataset.keys()):
        if user in users_list:
            for enum_video, video in enumerate(dataset[user].keys()):
                if video in videos_list:
                    if video not in errors_per_video.keys():
                        errors_per_video[video] = {}
                    print('computing error for trace', 'user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
                    xyz_per_video = dataset[user][video]
                    for t in range(history_window, len(xyz_per_video)-prediction_horizon):
                        sample_t = xyz_per_video[t, 1:]
                        for x_i in range(prediction_horizon):
                            if x_i not in errors_per_video[video].keys():
                                errors_per_video[video][x_i] = []
                            sample_t_n = xyz_per_video[t+x_i+1, 1:]
                            errors_per_video[video][x_i].append(compute_orthodromic_distance(sample_t, sample_t_n))
    return errors_per_video


def compute_pos_only_baseline_error_xyz(dataset, videos_list, model, history_window, prediction_horizon):
    intersection_angle_error = []
    for enum_user, user in enumerate(dataset.keys()):
        for enum_video, video in enumerate(dataset[user].keys()):
            if video in videos_list:
                print('computing error for trace', 'user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
                xyz_per_video = dataset[user][video]
                for t in range(history_window, len(xyz_per_video)-prediction_horizon):
                    encoder_pos_inputs_for_batch = [xyz_per_video[t-history_window:t, 1:]]
                    decoder_pos_inputs_for_batch = [xyz_per_video[t:t+1, 1:]]
                    prediction = model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)])
                    for x_i in range(prediction_horizon):
                        pred_t_n = normalized_eulerian_to_cartesian(prediction[0, x_i, 0], prediction[0, x_i, 1])
                        sample_t_n = xyz_per_video[t+x_i+1, 1:]
                        int_ang_err = compute_orthodromic_distance(pred_t_n, sample_t_n)
                        intersection_angle_error.append(radian_to_degrees(int_ang_err))
    return intersection_angle_error


# This function is used to compute the metrics used in the CVPR18 paper
def compute_pretrained_model_error_xyz(dataset, videos_list, model, model_name, history_window, prediction_horizon):
    intersection_angle_error = []
    for enum_user, user in enumerate(dataset.keys()):
        for enum_video, video in enumerate(dataset[user].keys()):
            if video in videos_list:
                print('computing error for trace', 'user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
                xyz_per_video = dataset[user][video]
                for t in range(history_window, len(xyz_per_video)-prediction_horizon):
                    encoder_pos_inputs_for_batch = [xyz_per_video[t-history_window:t, 1:]]
                    decoder_pos_inputs_for_batch = [xyz_per_video[t:t+1, 1:]]
                    prediction = model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)])
                    for x_i in range(prediction_horizon):
                        pred_t_n = normalized_eulerian_to_cartesian(prediction[0, x_i, 0], prediction[0, x_i, 1])
                        sample_t_n = xyz_per_video[t+x_i+1, 1:]
                        int_ang_err = compute_orthodromic_distance(pred_t_n, sample_t_n)
                        intersection_angle_error.append(radian_to_degrees(int_ang_err))
    return intersection_angle_error

# Print the CDF of the input 1D array
def get_data_proportion(int_ang_err):
    data_sorted = np.sort(int_ang_err)
    # Calculate the proportional values of samples
    p = 1. * np.arange(len(int_ang_err)) / (len(int_ang_err) - 1)
    for i in range(0, len(data_sorted), 5000):
        if data_sorted[i] <= 20.0:
            print("%.4f, %.4f" % (data_sorted[i], p[i]))

def get_list_of_videos_and_users_for_experiment():
    # fix random seed for reproducibility
    np.random.seed(7)

    videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight',
              '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino',
              '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
    users = np.arange(57)

    # Select at random the users for each set
    np.random.shuffle(users)
    num_train_users = int(len(users) * 0.5)
    users_train = users[:num_train_users]
    users_test = users[num_train_users:]

    videos_ids_train = [1, 3, 5, 7, 8, 9, 11, 14, 16, 18]
    videos_ids_test = [0, 2, 4, 13, 15]

    videos_train = [videos[video_id] for video_id in videos_ids_train]
    videos_test = [videos[video_id] for video_id in videos_ids_test]
    users_train = [str(user) for user in users_train]
    users_test = [str(user) for user in users_test]
    return videos_train, users_train, videos_test, users_test

if __name__ == "__main__":
    videos_train, users_train, videos_test, users_test = get_list_of_videos_and_users_for_experiment()

    H_WINDOW = 25

    # Compute the different results
    xyz_dataset_gaze = load_gaze_dataset()

    # No-motion baseline
    errors_per_video = compute_no_motion_baseline_error_xyz(xyz_dataset_gaze, videos_test, users_test, history_window=5, prediction_horizon=H_WINDOW)
    for video_name in videos_test:
        for t in range(H_WINDOW):
            print(video_name, t, np.mean(errors_per_video[video_name][t]), end=';')
            # print video_name, t, np.mean(errors_per_video[video_name][t]),';',
        print()

    # Position only baseline
    # BEST_SNAPSHOT_WEIGHTS = '/home/twipsy/PycharmProjects/UniformHeadMotionDataset/Xu_CVPR_18/pos_only/Models_EncDec_eulerian_Paper_Exp_init_5_in_5_out_5_end_5/weights_023.hdf5'
    # model = create_pos_only_model(M_WINDOW=5, H_WINDOW=5)
    # model.load_weights(BEST_SNAPSHOT_WEIGHTS)
    # int_ang_err_xyz_no_motion = compute_pos_only_baseline_error_xyz(xyz_dataset_gaze, videos_test, model, history_window=5, prediction_horizon=5)
    # get_data_proportion(int_ang_err_xyz_no_motion)

