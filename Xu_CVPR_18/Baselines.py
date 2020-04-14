import sys
sys.path.insert(0, './')

import os
import pandas as pd
import numpy as np
from Utils import compute_orthodromic_distance, cartesian_to_eulerian, radian_to_degrees, eulerian_to_cartesian
# import matplotlib.pyplot as plt
from TRACK_model import create_TRACK_model
from CVPR18_model import create_CVPR18_model
from CVPR18_original_model import create_CVPR18_orig_Model, auto_regressive_prediction
from position_only_baseline import create_pos_only_model
from SampledDataset import load_saliency, load_true_saliency
import argparse

# ToDo import from training_procedure (for now we just copied the function)
# from training_procedure import transform_batches_cartesian_to_normalized_eulerian
def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2*np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)

parser = argparse.ArgumentParser(description='Process the input parameters to evaluate the network.')


# usage: python Baselines.py -server_name bird -gpu_id 1 -model_name TRACK

parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')

args = parser.parse_args()

ROOT_FOLDER = './Xu_CVPR_18'
EXPERIMENT_GAZE_FOLDER = 'sampled_dataset_replica'
TRAIN_TEST_SET_FILE = 'dataset/train_test_set.xlsx'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

TRAINED_PREDICTION_HORIZON = 5

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

# ToDo: Copied in Reading_Dataset
# Read the split of the video ids for the train and test set in the experiments from CVPR18
def get_videos_train_and_test_from_file():
    xl_train_test_set_file = pd.ExcelFile(os.path.join(ROOT_FOLDER, TRAIN_TEST_SET_FILE))
    df_train = xl_train_test_set_file.parse('train_set', header=None)
    df_test = xl_train_test_set_file.parse('test_set', header=None)
    videos_train = df_train[0].values
    videos_test = df_test[0].values
    videos_train = ['%03d'%video_id for video_id in videos_train]
    videos_test = ['%03d'%video_id for video_id in videos_test]
    return videos_train, videos_test


def load_gaze_dataset():
    list_of_videos = os.listdir(os.path.join(ROOT_FOLDER, EXPERIMENT_GAZE_FOLDER))
    gaze_dataset = {}
    for video in list_of_videos:
        for user in os.listdir(os.path.join(ROOT_FOLDER, EXPERIMENT_GAZE_FOLDER, video)):
            if user not in gaze_dataset.keys():
                gaze_dataset[user] = {}
            # path = os.path.join(EXPERIMENT_GAZE_FOLDER, video, user)
            path = os.path.join(ROOT_FOLDER, EXPERIMENT_GAZE_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            gaze_dataset[user][video] = data.values
    return gaze_dataset


# Computes the intersection angle error for the baseline (using the position at t for prediction up to time t+prediction_horizon)
# for all the samples in the dataset with video belonging to videos_list
def compute_no_motion_baseline_error_xyz(dataset, videos_list, history_window, prediction_horizon):
    intersection_angle_error = []
    for enum_user, user in enumerate(dataset.keys()):
        for enum_video, video in enumerate(dataset[user].keys()):
            if video in videos_list:
                print('computing error for trace', 'user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
                xyz_per_video = dataset[user][video]
                for t in range(history_window, len(xyz_per_video)-prediction_horizon):
                    sample_t = xyz_per_video[t, 1:]
                    for x_i in range(prediction_horizon):
                        sample_t_n = xyz_per_video[t+x_i+1, 1:]
                        int_ang_err = compute_orthodromic_distance(sample_t, sample_t_n)
                        intersection_angle_error.append(radian_to_degrees(int_ang_err))
    return intersection_angle_error


# def compute_pos_only_baseline_error_xyz(dataset, videos_list, model, history_window, prediction_horizon):
#     intersection_angle_error = []
#     for enum_user, user in enumerate(dataset.keys()):
#         for enum_video, video in enumerate(dataset[user].keys()):
#             if video in videos_list:
#                 print('computing error for trace', 'user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
#                 xyz_per_video = dataset[user][video]
#                 for t in range(history_window, len(xyz_per_video)-prediction_horizon):
#                     encoder_pos_inputs_for_batch = [xyz_per_video[t-history_window:t, 1:]]
#                     decoder_pos_inputs_for_batch = [xyz_per_video[t:t+1, 1:]]
#                     prediction = model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)])
#                     for x_i in range(prediction_horizon):
#                         pred_t_n = normalized_eulerian_to_cartesian(prediction[0, x_i, 0], prediction[0, x_i, 1])
#                         sample_t_n = xyz_per_video[t+x_i+1, 1:]
#                         int_ang_err = compute_orthodromic_distance(pred_t_n, sample_t_n)
#                         intersection_angle_error.append(radian_to_degrees(int_ang_err))
#     return intersection_angle_error

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216
# This function is used to compute the metrics used in the CVPR18 paper
def compute_pretrained_model_error_xyz(dataset, videos_list, model_name, history_window, prediction_horizon, model_weights_path):
    if model_name == 'TRACK':
        model = create_TRACK_model(history_window, TRAINED_PREDICTION_HORIZON, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
    elif model_name == 'CVPR18':
        model = create_CVPR18_model(history_window, TRAINED_PREDICTION_HORIZON, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
    elif model_name == 'CVPR18_orig':
        model = create_CVPR18_orig_Model(history_window, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
    elif model_name == 'pos_only':
        model = create_pos_only_model(history_window, TRAINED_PREDICTION_HORIZON)

    ###
    if os.path.isfile(model_weights_path):
        model.load_weights(model_weights_path)
    else:
        command = 'python training_procedure.py -train -gpu_id 0 -dataset_name Xu_CVPR_18 -model_name %s -m_window 5 -h_window 5 -exp_folder sampled_dataset_replica -provided_videos' % model_name
        if model_name not in ['no_motion', 'pos_only', 'TRACK']:
            command += ' -use_true_saliency'

        raise Exception('Sorry, the folder ./Xu_CVPR_18/'+model_name+'/ doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"'+command+'\" or \n* Download the files from:\n\thttps://unice-my.sharepoint.com/:f:/g/personal/miguel_romero-rondon_unice_fr/EjhbHp5qgDRKrtkqODKayq0BoCqUY76cmm8bDwdbMOTqeQ?e=fGRFjo')
    ###

    saliency_folder = os.path.join(ROOT_FOLDER, 'extract_saliency/saliency')
    true_saliency_folder = os.path.join(ROOT_FOLDER, 'true_saliency')

    if model_name not in ['pos_only']:
        all_saliencies = {}
        for video in videos_list:
            # for model CVPR18_orig we use the true saliency:
            if model_name == 'CVPR18_orig':
                if os.path.isdir(true_saliency_folder):
                    all_saliencies[video] = load_true_saliency(true_saliency_folder, video)
                else:
                    raise Exception('Sorry, the folder ./Xu_CVPR_18/true_saliency doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"python ./Xu_CVPR_18/Read_Dataset.py -creat_true_sal\" or \n* Download the folder from:\n\thttps://unice-my.sharepoint.com/:f:/g/personal/miguel_romero-rondon_unice_fr/EsOFppF2mSRBtCtlmUM0TV4BGFRb1plZWgtUxSEo_E-I7w?e=pKXxCf')
            else:
                if os.path.isdir(saliency_folder):
                    all_saliencies[video] = load_saliency(saliency_folder, video)
                else:
                    raise Exception('Sorry, the folder ./Xu_CVPR_18/extract_saliency doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"./Xu_CVPR_18/dataset/creation_of_scaled_images.sh\n\tpython ./Extract_Saliency/panosalnet.py -dataset_name CVPR_18\" or \n* Download the folder from:\n\thttps://unice-my.sharepoint.com/:f:/g/personal/miguel_romero-rondon_unice_fr/EvRCuy0v5BpDmADTPUuA8JgBoIgaWcFbR0S7wIXlevIIGQ?e=goOz7o')

    intersection_angle_error = []
    for enum_user, user in enumerate(dataset.keys()):
        for enum_video, video in enumerate(dataset[user].keys()):
            if video in videos_list:
                print('computing error for trace', 'user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
                xyz_per_video = dataset[user][video]
                for t in range(history_window, len(xyz_per_video)-prediction_horizon):
                    if model_name not in ['pos_only', 'no_motion']:
                        encoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][t - history_window + 1:t + 1], axis=-1)])
                        # ToDo: Be careful here, we are using TRAINED_PREDICTION_HORIZON to load future saliencies
                        if model_name == 'CVPR18_orig':
                            decoder_sal_inputs_for_sample = np.zeros((1, TRAINED_PREDICTION_HORIZON, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL, 1))
                        else:
                            decoder_sal_inputs_for_sample = np.zeros((1, TRAINED_PREDICTION_HORIZON, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1))
                        taken_saliencies = all_saliencies[video][t + 1:min(t + TRAINED_PREDICTION_HORIZON + 1, len(all_saliencies[video]))]
                        # decoder_sal_inputs_for_sample = np.array([np.expand_dims(taken_saliencies, axis=-1)])
                        decoder_sal_inputs_for_sample[0, :len(taken_saliencies), :, :, 0] = taken_saliencies
                    encoder_pos_inputs_for_sample = [xyz_per_video[t-history_window:t, 1:]]
                    decoder_pos_inputs_for_sample = [xyz_per_video[t:t+1, 1:]]

                    if model_name == 'TRACK':
                        model_prediction = model.predict(
                            [np.array(encoder_pos_inputs_for_sample), np.array(encoder_sal_inputs_for_sample),
                             np.array(decoder_pos_inputs_for_sample), np.array(decoder_sal_inputs_for_sample)])[0]
                    elif model_name == 'CVPR18':
                        model_prediction = model.predict(
                            [np.array(encoder_pos_inputs_for_sample), np.array(decoder_pos_inputs_for_sample),
                             np.array(decoder_sal_inputs_for_sample)])[0]
                    elif model_name == 'CVPR18_orig':
                        initial_pos_inputs = transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample)
                        model_pred = auto_regressive_prediction(model, initial_pos_inputs, decoder_sal_inputs_for_sample, history_window, prediction_horizon)
                        model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
                    elif model_name == 'pos_only':
                        model_pred = model.predict(
                            [transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample),
                             transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
                        model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)

                    for x_i in range(prediction_horizon):
                        pred_t_n = model_prediction[x_i]
                        sample_t_n = xyz_per_video[t+x_i+1, 1:]
                        int_ang_err = compute_orthodromic_distance(pred_t_n, sample_t_n)
                        intersection_angle_error.append(radian_to_degrees(int_ang_err))
    return intersection_angle_error

# Print the CDF of the input 1D array
def get_data_proportion(int_ang_err):
    data_sorted = np.sort(int_ang_err)
    # Calculate the proportional values of samples
    p = 1. * np.arange(len(int_ang_err)) / (len(int_ang_err) - 1)
    x_vals = []
    y_vals = []
    for i in range(0, len(data_sorted), 5000):
        if data_sorted[i] <= 20.0:
            print("%.4f, %.4f" % (data_sorted[i], p[i]))
            x_vals.append(data_sorted[i])
            y_vals.append(p[i])
    return x_vals, y_vals


if __name__ == "__main__":
    ### This dataset follows the specifications in Xu_CVPR_18 Paper:
    # "Following the common setting in trajectory prediction for crowd, we downsample one frame from every five frames for model training and performance evaluation.
    # In this way, the interval between two neighboring frames in our experiments corresponds to 5/25 seconds, and such setting makes our gaze prediction task more challenging
    # than that for the neighboring frames in original videos. In the following sections, the frames mentioned correspond to the sampled ones. Further, we propose to use the
    # history gaze path in the first five frames to predict the gaze points in next five frames (history_window=5, prediction_horizon=5). We use the observation (the result of gaze tracking)
    # in the first 1 second to predict the gaze points in the frames of upcoming 1 second."
    ### Check function "create_and_store_gaze_sampled_dataset()" in Reading_Dataset.py the sampled dataset is read here with the function "load_gaze_dataset()"
    # "[...] In our experiments, we randomly select 134 videos as training data, and use the remaining 74 videos as testing. Some participants are shared in training and testing, but the videos
    # in training/testing have no overlap.#
    ### Check function "get_videos_train_and_test_from_file()"
    # "We propose to use the viewing angle between the predicted gaze point and its ground truth to measure the performance of gaze prediction. A smaller angle means the predicted point agrees
    # its prediction better. [...] For gaze point in the i^{th} frame (i = 1, ..., T) with ground truth (x_{i}^{p}, y_{i}^{p}) and prediction (\hat{x}_{i}^{p}, \hat{y}_{i}^{p}), the viewing angle between
    # them can be represented as d_i (the way to compute d_i will be provided in supplementary material), [...], we also use cumulative distribution function (CDF) of all gaze points for performance
    # evaluation. A higher CDF curve corresponds to a method with smaller MAE."
    ### Check function "get_data_proportion()"
    # "Intersection Angle Error: For a given gaze point (x, y) where x is the longitude and y is the latitude, its coordinate in the unit sphere is P = (cos y cos x, cos y sin x, sin y), then for a
    # ground truth gaze point (x, y) and its predicted gaze point (\hat{x}, \hat{y}), we can get corresponding coordinates in unit sphere as P and \hat{P}, the intersection angle error between them
    # can be computed as $d = arccos(<P, \hat{P}>), where <, > is inner product"
    ### Check function compute_orthodromic_distance

    videos_train, videos_test = get_videos_train_and_test_from_file()

    xyz_dataset_gaze = load_gaze_dataset()

    # No-motion baseline
    # int_ang_err_xyz_no_motion = compute_no_motion_baseline_error_xyz(xyz_dataset_gaze, videos_test, history_window=5, prediction_horizon=5)
    # get_data_proportion(int_ang_err_xyz_no_motion)

    # Position only baseline trained on the correct prediction window
    # weights_path = os.path.join(ROOT_FOLDER, 'pos_only/Models_EncDec_eulerian_Paper_Exp_init_5_in_5_out_5_end_5/weights_023.hdf5')
    # int_ang_err_xyz_no_motion = compute_pretrained_model_error_xyz(xyz_dataset_gaze, videos_test, 'pos_only', history_window=5, prediction_horizon=5, model_weights_path=weights_path)
    # get_data_proportion(int_ang_err_xyz_no_motion)

    # # TRACK model trained on 5 seconds prediction window
    if args.model_name == 'no_motion':
        int_ang_err_xyz_no_motion = compute_no_motion_baseline_error_xyz(xyz_dataset_gaze, videos_test, history_window=5, prediction_horizon=5)
        x_vals, y_vals = get_data_proportion(int_ang_err_xyz_no_motion)
    elif args.model_name == 'pos_only':
        weights_path = os.path.join(ROOT_FOLDER, 'pos_only/Models_EncDec_eulerian_Paper_Exp_init_5_in_5_out_5_end_5/weights.hdf5')
        int_ang_err_xyz_no_motion = compute_pretrained_model_error_xyz(xyz_dataset_gaze, videos_test, 'pos_only', history_window=5, prediction_horizon=5, model_weights_path=weights_path)
        x_vals, y_vals = get_data_proportion(int_ang_err_xyz_no_motion)
    elif args.model_name == 'TRACK':
        weights_path = os.path.join(ROOT_FOLDER, 'TRACK/Models_EncDec_3DCoords_ContSal_Paper_Exp_init_5_in_5_out_5_end_5', 'weights.hdf5')
        int_ang_err_xyz_no_motion = compute_pretrained_model_error_xyz(xyz_dataset_gaze, videos_test, args.model_name, history_window=5, prediction_horizon=5, model_weights_path=weights_path)
        x_vals, y_vals = get_data_proportion(int_ang_err_xyz_no_motion)
    elif args.model_name == 'CVPR18':
        weights_path = os.path.join(ROOT_FOLDER, 'CVPR18/Models_EncDec_3DCoords_ContSal_Paper_Exp_init_5_in_5_out_5_end_5', 'weights.hdf5')
        int_ang_err_xyz_no_motion = compute_pretrained_model_error_xyz(xyz_dataset_gaze, videos_test, args.model_name, history_window=5, prediction_horizon=5, model_weights_path=weights_path)
        x_vals, y_vals = get_data_proportion(int_ang_err_xyz_no_motion)
    elif args.model_name == 'CVPR18_orig':
        weights_path = os.path.join(ROOT_FOLDER, 'CVPR18_orig/Models_EncDec_2DNormalized_TrueSal_Paper_Exp_init_5_in_5_out_5_end_5', 'weights.hdf5')
        # print(weights_path)
        int_ang_err_xyz_no_motion = compute_pretrained_model_error_xyz(xyz_dataset_gaze, videos_test, args.model_name, history_window=5, prediction_horizon=5, model_weights_path=weights_path)
        x_vals, y_vals = get_data_proportion(int_ang_err_xyz_no_motion)

    import matplotlib.pyplot as plt

    plt.plot(
        [0.0175, 0.1172, 0.2172, 0.3171, 0.4168, 0.5168, 0.6165, 0.7164, 0.8161, 0.9161, 1.0160, 1.1157, 1.2157, 1.3154,
         1.4153, 1.5150, 1.6150, 1.7149, 1.8146, 1.9145, 2.0142, 2.1142, 2.2139, 2.3138, 2.4135, 2.5135, 2.6134, 2.7131,
         2.8131, 2.9128, 3.0127, 3.1124, 3.2124, 3.3123, 3.4120, 3.5120, 3.6117, 3.7116, 3.8113, 3.9113, 4.0112, 4.1109,
         4.2108, 4.3105, 4.4105, 4.5102, 4.6101, 4.7098, 4.8098, 4.9097, 5.0094, 5.1094, 5.2093, 5.3090, 5.4087, 5.5087,
         5.6086, 5.7083, 5.8083, 5.9080, 6.0079, 6.1076, 6.2076, 6.3073, 6.4072, 6.5071, 6.6068, 6.7068, 6.8065, 6.9064,
         7.0061, 7.1061, 7.2060, 7.3057, 7.4057, 7.5054, 7.6053, 7.7050, 7.8050, 7.9049, 8.0046, 8.1046, 8.2043, 8.3042,
         8.4039, 8.5039, 8.6038, 8.7035, 8.8034, 8.9031, 9.0031, 9.1030, 9.2027, 9.3027, 9.4024, 9.5023, 9.6020, 9.7020,
         9.8017, 9.9016, 10.0016, 10.1013, 10.2010, 10.3009, 10.4009, 10.5006, 10.6005, 10.7004, 10.8002, 10.8999,
         10.9998, 11.0997, 11.1994, 11.2994, 11.3991, 11.4990, 11.5987, 11.6987, 11.7986, 11.8983, 11.9983, 12.0980,
         12.1979, 12.2976, 12.3976, 12.4975, 12.5972, 12.6972, 12.7969, 12.8968, 12.9962, 13.0965, 13.1964, 13.2961,
         13.3960, 13.4957, 13.5957, 13.6954, 13.7953, 13.8953, 13.9950, 14.0949, 14.1946, 14.2946, 14.3943, 14.4942,
         14.5942, 14.6939, 14.7938, 14.8935, 14.9935, 15.0932, 15.1931, 15.2930, 15.3928, 15.4927, 15.5924, 15.6923,
         15.7920, 15.8920, 15.9919, 16.0916, 16.1913, 16.2913, 16.3912, 16.4909, 16.5909, 16.6906, 16.7905, 16.8902,
         16.9902, 17.0901, 17.1898, 17.2898, 17.3895, 17.4894, 17.5891, 17.6891, 17.7890, 17.8887, 17.9886, 18.0883,
         18.1883, 18.2880, 18.3879, 18.4879, 18.5876, 18.6875, 18.7872, 18.8872, 18.9869, 19.0868, 19.1868, 19.2865,
         19.3864, 19.4861, 19.5861, 19.6858, 19.7857, 19.8856, 19.9854],
        [0.00008, 0.00060, 0.00170, 0.00365, 0.00692, 0.01041, 0.01460, 0.01984, 0.02515, 0.03062, 0.03634, 0.04288,
         0.04988, 0.05680, 0.06338, 0.07067, 0.07816, 0.08483, 0.09195, 0.09836, 0.10509, 0.11134, 0.11775, 0.12356,
         0.12991, 0.13642, 0.14183, 0.14749, 0.15312, 0.15891, 0.16449, 0.16934, 0.17456, 0.17976, 0.18472, 0.18938,
         0.19408, 0.19864, 0.20303, 0.20765, 0.21228, 0.21676, 0.22070, 0.22484, 0.22873, 0.23324, 0.23724, 0.24126,
         0.24526, 0.24966, 0.25352, 0.25751, 0.26186, 0.26562, 0.26925, 0.27244, 0.27562, 0.27924, 0.28294, 0.28671,
         0.29037, 0.29407, 0.29747, 0.30085, 0.30421, 0.30707, 0.31052, 0.31357, 0.31624, 0.31909, 0.32232, 0.32557,
         0.32895, 0.33183, 0.33514, 0.33801, 0.34092, 0.34372, 0.34641, 0.34918, 0.35176, 0.35482, 0.35786, 0.36046,
         0.36334, 0.36638, 0.36929, 0.37193, 0.37472, 0.37751, 0.38030, 0.38287, 0.38554, 0.38853, 0.39051, 0.39303,
         0.39559, 0.39816, 0.40095, 0.40382, 0.40632, 0.40909, 0.41167, 0.41417, 0.41681, 0.41934, 0.42137, 0.42402,
         0.42688, 0.42928, 0.43156, 0.43434, 0.43685, 0.43907, 0.44158, 0.44375, 0.44605, 0.44786, 0.45019, 0.45283,
         0.45506, 0.45745, 0.45972, 0.46167, 0.46364, 0.46570, 0.46805, 0.47027, 0.47245, 0.47482, 0.47707, 0.47910,
         0.48129, 0.48367, 0.48590, 0.48794, 0.48977, 0.49192, 0.49352, 0.49527, 0.49735, 0.49962, 0.50163, 0.50357,
         0.50584, 0.50808, 0.51034, 0.51290, 0.51503, 0.51730, 0.51915, 0.52150, 0.52351, 0.52534, 0.52744, 0.52935,
         0.53137, 0.53377, 0.53599, 0.53787, 0.53997, 0.54174, 0.54355, 0.54540, 0.54746, 0.54952, 0.55208, 0.55429,
         0.55613, 0.55834, 0.56005, 0.56189, 0.56400, 0.56616, 0.56781, 0.56988, 0.57191, 0.57376, 0.57546, 0.57737,
         0.57935, 0.58110, 0.58304, 0.58507, 0.58665, 0.58862, 0.59069, 0.59260, 0.59453, 0.59612, 0.59787, 0.60017,
         0.60170, 0.60364, 0.60519, 0.60684, 0.60859, 0.61013, 0.61187, 0.61341, 0.61537], 'r--',
        label='CVPR18 reported')

    plt.plot(
        [0.0015, 0.1213, 0.1754, 0.2195, 0.26, 0.2973, 0.3325, 0.3664, 0.3997, 0.4319, 0.4635, 0.4946, 0.5252, 0.557, 0.5883, 0.6196, 0.6506, 0.6821, 0.7138, 0.7447, 0.7756, 0.8069, 0.8394, 0.8709, 0.9032, 0.9356, 0.9682, 1.0015, 1.0347, 1.0682, 1.1014, 1.1361, 1.1715, 1.2066, 1.2421, 1.2778, 1.3138, 1.3504, 1.3865, 1.4248, 1.4625, 1.501, 1.5397, 1.5788, 1.6188, 1.6601, 1.6996, 1.7408, 1.7833, 1.8257, 1.8685, 1.9121, 1.956, 2.0003, 2.0447, 2.0896, 2.1363, 2.1823, 2.2304, 2.2773, 2.3262, 2.375, 2.4245, 2.4747, 2.5259, 2.5771, 2.6308, 2.6844, 2.7385, 2.7943, 2.8497, 2.9059, 2.9633, 3.0219, 3.0796, 3.1381, 3.1991, 3.2601, 3.3211, 3.3825, 3.4451, 3.5079, 3.5714, 3.6362, 3.7039, 3.7705, 3.8398, 3.9077, 3.9763, 4.0452, 4.1164, 4.1875, 4.2619, 4.3354, 4.4091, 4.4831, 4.5586, 4.6324, 4.7104, 4.7857, 4.8609, 4.9396, 5.017, 5.0944, 5.1722, 5.252, 5.333, 5.4149, 5.4958, 5.5754, 5.657, 5.7396, 5.8201, 5.9018, 5.9847, 6.0683, 6.1524, 6.2348, 6.3233, 6.4101, 6.494, 6.5821, 6.6701, 6.7561, 6.8437, 6.932, 7.0224, 7.1087, 7.2004, 7.2904, 7.3802, 7.4691, 7.559, 7.6504, 7.7415, 7.833, 7.925, 8.0161, 8.1091, 8.2022, 8.2957, 8.3875, 8.4821, 8.5755, 8.6703, 8.7652, 8.8601, 8.9564, 9.0524, 9.1519, 9.2463, 9.3426, 9.4425, 9.5436, 9.6428, 9.7417, 9.8409, 9.9415, 10.0408, 10.1409, 10.2436, 10.347, 10.4472, 10.5504, 10.6523, 10.7551, 10.8588, 10.9651, 11.0664, 11.1733, 11.2778, 11.3831, 11.4894, 11.5952, 11.701, 11.8101, 11.9215, 12.0299, 12.1408, 12.2477, 12.3598, 12.4713, 12.5823, 12.6917, 12.8046, 12.9205, 13.0364, 13.1531, 13.2694, 13.3877, 13.507, 13.6251, 13.7435, 13.8627, 13.9846, 14.1029, 14.2263, 14.3509, 14.4752, 14.6021, 14.7247, 14.8522, 14.9752, 15.1028, 15.2309, 15.3595, 15.485, 15.613, 15.7454, 15.8746, 16.0094, 16.1415, 16.2755, 16.4118, 16.5464, 16.6815, 16.8177, 16.9554, 17.0926, 17.2298, 17.3728, 17.5131, 17.6549, 17.798, 17.9403, 18.0847, 18.2322, 18.3774, 18.5262, 18.6731, 18.8242, 18.9753, 19.1239, 19.2777, 19.431, 19.5848, 19.7431, 19.8961],
        [0, 0.0026, 0.0052, 0.0078, 0.0104, 0.013, 0.0156, 0.0182, 0.0208, 0.0234, 0.026, 0.0286, 0.0312, 0.0338, 0.0365, 0.0391, 0.0417, 0.0443, 0.0469, 0.0495, 0.0521, 0.0547, 0.0573, 0.0599, 0.0625, 0.0651, 0.0677, 0.0703, 0.0729, 0.0755, 0.0781, 0.0807, 0.0833, 0.0859, 0.0885, 0.0911, 0.0937, 0.0963, 0.0989, 0.1015, 0.1041, 0.1068, 0.1094, 0.112, 0.1146, 0.1172, 0.1198, 0.1224, 0.125, 0.1276, 0.1302, 0.1328, 0.1354, 0.138, 0.1406, 0.1432, 0.1458, 0.1484, 0.151, 0.1536, 0.1562, 0.1588, 0.1614, 0.164, 0.1666, 0.1692, 0.1718, 0.1745, 0.1771, 0.1797, 0.1823, 0.1849, 0.1875, 0.1901, 0.1927, 0.1953, 0.1979, 0.2005, 0.2031, 0.2057, 0.2083, 0.2109, 0.2135, 0.2161, 0.2187, 0.2213, 0.2239, 0.2265, 0.2291, 0.2317, 0.2343, 0.2369, 0.2395, 0.2421, 0.2448, 0.2474, 0.25, 0.2526, 0.2552, 0.2578, 0.2604, 0.263, 0.2656, 0.2682, 0.2708, 0.2734, 0.276, 0.2786, 0.2812, 0.2838, 0.2864, 0.289, 0.2916, 0.2942, 0.2968, 0.2994, 0.302, 0.3046, 0.3072, 0.3098, 0.3124, 0.3151, 0.3177, 0.3203, 0.3229, 0.3255, 0.3281, 0.3307, 0.3333, 0.3359, 0.3385, 0.3411, 0.3437, 0.3463, 0.3489, 0.3515, 0.3541, 0.3567, 0.3593, 0.3619, 0.3645, 0.3671, 0.3697, 0.3723, 0.3749, 0.3775, 0.3801, 0.3827, 0.3854, 0.388, 0.3906, 0.3932, 0.3958, 0.3984, 0.401, 0.4036, 0.4062, 0.4088, 0.4114, 0.414, 0.4166, 0.4192, 0.4218, 0.4244, 0.427, 0.4296, 0.4322, 0.4348, 0.4374, 0.44, 0.4426, 0.4452, 0.4478, 0.4504, 0.453, 0.4557, 0.4583, 0.4609, 0.4635, 0.4661, 0.4687, 0.4713, 0.4739, 0.4765, 0.4791, 0.4817, 0.4843, 0.4869, 0.4895, 0.4921, 0.4947, 0.4973, 0.4999, 0.5025, 0.5051, 0.5077, 0.5103, 0.5129, 0.5155, 0.5181, 0.5207, 0.5234, 0.526, 0.5286, 0.5312, 0.5338, 0.5364, 0.539, 0.5416, 0.5442, 0.5468, 0.5494, 0.552, 0.5546, 0.5572, 0.5598, 0.5624, 0.565, 0.5676, 0.5702, 0.5728, 0.5754, 0.578, 0.5806, 0.5832, 0.5858, 0.5884, 0.591, 0.5937, 0.5963, 0.5989, 0.6015, 0.6041, 0.6067, 0.6093, 0.6119, 0.6145, 0.6171], 'g--',
        label='Pos-Only reported')

    plt.plot(
        [0.00135, 0.1924, 0.28115, 0.36, 0.41345, 0.4754, 0.5376, 0.58785, 0.64875, 0.70975, 0.76075, 0.8124, 0.86405, 0.9172, 0.98325, 1.0374, 1.0914, 1.1477, 1.20345, 1.261, 1.3196, 1.3795, 1.4398, 1.50075, 1.56345, 1.6263, 1.69085, 1.73995, 1.80725, 1.8763, 1.94665, 2.01785, 2.07055, 2.1451, 2.2209, 2.2776, 2.35505, 2.4356, 2.5173, 2.5777, 2.66265, 2.74875, 2.81215, 2.90235, 2.96745, 3.0585, 3.15075, 3.24485, 3.31495, 3.3842, 3.4822, 3.5555, 3.6577, 3.7328, 3.83785, 3.9457, 4.0247, 4.1352, 4.217, 4.3302, 4.41235, 4.52925, 4.61195, 4.69405, 4.81435, 4.89965, 5.02145, 5.10925, 5.19645, 5.2832, 5.41165, 5.5015, 5.63025, 5.72385, 5.81495, 5.94565, 6.03765, 6.1321, 6.2684, 6.36415, 6.4598, 6.5977, 6.6937, 6.83255, 6.9312, 7.0304, 7.1306, 7.27625, 7.376, 7.47715, 7.62195, 7.7228, 7.823, 7.9239, 8.07245, 8.17465, 8.27725, 8.38155, 8.53225, 8.63725, 8.7419, 8.8483, 8.95565, 9.1127, 9.22275, 9.3319, 9.44175, 9.55045, 9.7087, 9.8192, 9.9303, 10.04125, 10.1524, 10.31495, 10.4259, 10.53985, 10.6535, 10.81695, 10.93005, 11.0427, 11.1586, 11.32565, 11.44165, 11.55645, 11.67685, 11.84875, 11.96805, 12.08905, 12.20815, 12.32725, 12.4466, 12.5668, 12.74225, 12.8629, 12.98545, 13.1097, 13.2355, 13.3607, 13.5449, 13.66885, 13.79665, 13.9238, 14.04965, 14.17675, 14.36425, 14.4935, 14.6255, 14.7573, 14.8886, 15.02185, 15.15675, 15.35205, 15.4892, 15.6214, 15.75795, 15.89325, 16.0298, 16.16915, 16.30945, 16.4504, 16.6592, 16.79875, 16.94075, 17.085, 17.22685, 17.36955, 17.58455, 17.73315, 17.8817, 18.0281, 18.1769, 18.3252, 18.4765, 18.62695, 18.7795, 18.9338, 19.0896, 19.24635, 19.40205, 19.63275, 19.79175, 19.95265],
        [0, 0.00956, 0.01736, 0.02508, 0.0292, 0.03516, 0.04112, 0.04524, 0.0512, 0.05716, 0.06128, 0.06548, 0.0696, 0.07372, 0.0797, 0.08382, 0.08794, 0.09214, 0.09626, 0.10038, 0.1045, 0.1087, 0.11282, 0.11694, 0.12106, 0.12526, 0.12938, 0.13174, 0.13586, 0.13998, 0.1441, 0.1483, 0.15058, 0.1547, 0.1589, 0.16118, 0.16538, 0.1695, 0.17362, 0.17598, 0.1801, 0.18424, 0.1866, 0.19072, 0.19308, 0.1972, 0.20132, 0.20544, 0.2078, 0.21016, 0.21428, 0.21664, 0.22076, 0.22304, 0.22724, 0.23136, 0.23364, 0.23784, 0.24012, 0.24432, 0.2466, 0.25072, 0.25308, 0.25544, 0.25956, 0.26192, 0.26604, 0.26834, 0.2707, 0.27306, 0.27718, 0.27954, 0.28366, 0.28594, 0.2883, 0.29242, 0.29478, 0.29714, 0.30126, 0.30354, 0.3059, 0.31002, 0.31238, 0.3165, 0.31886, 0.32114, 0.3235, 0.32762, 0.32998, 0.33234, 0.33646, 0.33874, 0.3411, 0.34346, 0.3476, 0.34996, 0.35224, 0.3546, 0.35872, 0.36108, 0.36336, 0.36572, 0.36808, 0.3722, 0.37448, 0.37684, 0.3792, 0.38148, 0.38568, 0.38796, 0.39032, 0.3926, 0.39496, 0.39908, 0.40144, 0.4038, 0.40608, 0.4102, 0.41256, 0.41492, 0.4172, 0.42142, 0.4237, 0.42606, 0.42834, 0.43254, 0.43482, 0.43718, 0.43954, 0.44182, 0.44418, 0.44646, 0.45066, 0.45294, 0.4553, 0.45766, 0.45994, 0.4623, 0.46642, 0.46878, 0.47106, 0.47342, 0.47578, 0.47806, 0.48218, 0.48454, 0.4869, 0.48918, 0.49156, 0.49384, 0.4962, 0.50032, 0.50268, 0.50504, 0.50732, 0.50968, 0.51196, 0.51432, 0.51668, 0.51896, 0.52316, 0.52544, 0.5278, 0.53008, 0.53244, 0.5348, 0.53892, 0.54128, 0.54356, 0.54592, 0.5482, 0.55056, 0.55292, 0.5552, 0.55756, 0.55994, 0.56222, 0.56458, 0.56686, 0.57106, 0.57334, 0.5757], 'y--',
        label='CVPR18-repro')

    plt.plot(
        [0.0008, 0.101, 0.1473, 0.185, 0.2181, 0.2492, 0.2789, 0.3076, 0.336, 0.3637, 0.3906, 0.4178, 0.4451, 0.4717, 0.4983, 0.5259, 0.5529, 0.5793, 0.6066, 0.6345, 0.6617, 0.6888, 0.7163, 0.7449, 0.7738, 0.802, 0.8308, 0.8598, 0.89, 0.9197, 0.95, 0.9805, 1.0116, 1.0426, 1.0732, 1.1049, 1.138, 1.171, 1.2039, 1.237, 1.2719, 1.306, 1.3405, 1.376, 1.412, 1.4475, 1.4843, 1.5218, 1.5597, 1.598, 1.6364, 1.6757, 1.7154, 1.7569, 1.7976, 1.8394, 1.8821, 1.9252, 1.9698, 2.0142, 2.0587, 2.1036, 2.1498, 2.1973, 2.246, 2.2943, 2.3425, 2.3927, 2.4444, 2.4958, 2.5491, 2.6032, 2.6579, 2.7132, 2.7702, 2.828, 2.8864, 2.9437, 3.0016, 3.0611, 3.1204, 3.1815, 3.2434, 3.3076, 3.3706, 3.4344, 3.4996, 3.5666, 3.6336, 3.703, 3.7733, 3.8427, 3.915, 3.9893, 4.0644, 4.137, 4.2131, 4.2888, 4.366, 4.4448, 4.5229, 4.6031, 4.6824, 4.7651, 4.8478, 4.93, 5.0133, 5.0969, 5.1815, 5.2672, 5.3552, 5.4421, 5.5327, 5.6222, 5.715, 5.8058, 5.897, 5.9874, 6.0814, 6.1758, 6.2709, 6.3645, 6.4596, 6.5564, 6.6513, 6.7461, 6.8431, 6.9411, 7.0397, 7.141, 7.2432, 7.3418, 7.4441, 7.5442, 7.644, 7.7451, 7.8452, 7.9476, 8.048, 8.1515, 8.2562, 8.359, 8.4636, 8.5667, 8.6708, 8.7767, 8.8838, 8.9943, 9.1055, 9.216, 9.3261, 9.439, 9.549, 9.6583, 9.7718, 9.883, 9.9944, 10.1087, 10.2215, 10.3355, 10.4495, 10.5649, 10.6792, 10.791, 10.9057, 11.0206, 11.1352, 11.2515, 11.3675, 11.4867, 11.6063, 11.7287, 11.8506, 11.9751, 12.0976, 12.2202, 12.3405, 12.462, 12.5852, 12.7076, 12.8335, 12.957, 13.0832, 13.2092, 13.3372, 13.4641, 13.5897, 13.7196, 13.8488, 13.9793, 14.1094, 14.2407, 14.372, 14.5036, 14.6343, 14.7684, 14.9048, 15.0398, 15.1759, 15.3142, 15.4545, 15.5916, 15.7353, 15.8736, 16.0145, 16.1572, 16.3003, 16.4474, 16.5927, 16.734, 16.8817, 17.0313, 17.1805, 17.3294, 17.4797, 17.6297, 17.7865, 17.9366, 18.0913, 18.2459, 18.4009, 18.5622, 18.7194, 18.8853, 19.045, 19.208, 19.368, 19.5328, 19.6931, 19.86],
        [0, 0.0026, 0.0052, 0.0078, 0.0104, 0.013, 0.0156, 0.0182, 0.0208, 0.0234, 0.026, 0.0286, 0.0312, 0.0338, 0.0365, 0.0391, 0.0417, 0.0443, 0.0469, 0.0495, 0.0521, 0.0547, 0.0573, 0.0599, 0.0625, 0.0651, 0.0677, 0.0703, 0.0729, 0.0755, 0.0781, 0.0807, 0.0833, 0.0859, 0.0885, 0.0911, 0.0937, 0.0963, 0.0989, 0.1015, 0.1041, 0.1068, 0.1094, 0.112, 0.1146, 0.1172, 0.1198, 0.1224, 0.125, 0.1276, 0.1302, 0.1328, 0.1354, 0.138, 0.1406, 0.1432, 0.1458, 0.1484, 0.151, 0.1536, 0.1562, 0.1588, 0.1614, 0.164, 0.1666, 0.1692, 0.1718, 0.1745, 0.1771, 0.1797, 0.1823, 0.1849, 0.1875, 0.1901, 0.1927, 0.1953, 0.1979, 0.2005, 0.2031, 0.2057, 0.2083, 0.2109, 0.2135, 0.2161, 0.2187, 0.2213, 0.2239, 0.2265, 0.2291, 0.2317, 0.2343, 0.2369, 0.2395, 0.2421, 0.2448, 0.2474, 0.25, 0.2526, 0.2552, 0.2578, 0.2604, 0.263, 0.2656, 0.2682, 0.2708, 0.2734, 0.276, 0.2786, 0.2812, 0.2838, 0.2864, 0.289, 0.2916, 0.2942, 0.2968, 0.2994, 0.302, 0.3046, 0.3072, 0.3098, 0.3124, 0.3151, 0.3177, 0.3203, 0.3229, 0.3255, 0.3281, 0.3307, 0.3333, 0.3359, 0.3385, 0.3411, 0.3437, 0.3463, 0.3489, 0.3515, 0.3541, 0.3567, 0.3593, 0.3619, 0.3645, 0.3671, 0.3697, 0.3723, 0.3749, 0.3775, 0.3801, 0.3827, 0.3854, 0.388, 0.3906, 0.3932, 0.3958, 0.3984, 0.401, 0.4036, 0.4062, 0.4088, 0.4114, 0.414, 0.4166, 0.4192, 0.4218, 0.4244, 0.427, 0.4296, 0.4322, 0.4348, 0.4374, 0.44, 0.4426, 0.4452, 0.4478, 0.4504, 0.453, 0.4557, 0.4583, 0.4609, 0.4635, 0.4661, 0.4687, 0.4713, 0.4739, 0.4765, 0.4791, 0.4817, 0.4843, 0.4869, 0.4895, 0.4921, 0.4947, 0.4973, 0.4999, 0.5025, 0.5051, 0.5077, 0.5103, 0.5129, 0.5155, 0.5181, 0.5207, 0.5234, 0.526, 0.5286, 0.5312, 0.5338, 0.5364, 0.539, 0.5416, 0.5442, 0.5468, 0.5494, 0.552, 0.5546, 0.5572, 0.5598, 0.5624, 0.565, 0.5676, 0.5702, 0.5728, 0.5754, 0.578, 0.5806, 0.5832, 0.5858, 0.5884, 0.591, 0.5937, 0.5963], 'm--',
        label='No-Motion reported')

    plt.plot(
        [0, 0.2917, 0.4133, 0.5087, 0.5887, 0.6623, 0.7287, 0.7908, 0.8489, 0.9041, 0.9569, 1.0082, 1.0592, 1.1086, 1.1573, 1.2063, 1.2525, 1.2984, 1.3441, 1.3888, 1.434, 1.4794, 1.5248, 1.5687, 1.6142, 1.6595, 1.7054, 1.7522, 1.7982, 1.8439, 1.8892, 1.935, 1.9816, 2.0283, 2.0762, 2.1225, 2.1686, 2.2154, 2.2625, 2.3099, 2.3575, 2.4048, 2.4527, 2.5013, 2.5506, 2.6001, 2.6501, 2.6999, 2.75, 2.7985, 2.8458, 2.8961, 2.9464, 2.9952, 3.0459, 3.097, 3.1476, 3.1995, 3.2512, 3.3036, 3.3558, 3.4097, 3.4618, 3.5153, 3.5692, 3.6249, 3.68, 3.7343, 3.789, 3.844, 3.9008, 3.9567, 4.0146, 4.0714, 4.1283, 4.1847, 4.2427, 4.3019, 4.3607, 4.4218, 4.4815, 4.5425, 4.6038, 4.6649, 4.7258, 4.7862, 4.8482, 4.9112, 4.9754, 5.0388, 5.103, 5.1675, 5.2319, 5.2974, 5.3636, 5.4299, 5.4979, 5.567, 5.6352, 5.7032, 5.7737, 5.8424, 5.9138, 5.9851, 6.0558, 6.1276, 6.2004, 6.2731, 6.3464, 6.4188, 6.49, 6.5645, 6.6385, 6.713, 6.7885, 6.8637, 6.9382, 7.0161, 7.0917, 7.1683, 7.2465, 7.324, 7.4004, 7.4779, 7.5582, 7.6378, 7.7178, 7.7975, 7.8791, 7.9597, 8.0406, 8.1248, 8.2076, 8.2927, 8.3765, 8.4605, 8.5454, 8.6302, 8.7162, 8.7995, 8.8858, 8.9724, 9.0604, 9.1476, 9.2359, 9.3255, 9.4146, 9.5029, 9.5919, 9.6832, 9.775, 9.8664, 9.9587, 10.0499, 10.1405, 10.2352, 10.3275, 10.4225, 10.5165, 10.6119, 10.7065, 10.8028, 10.8999, 10.9956, 11.0933, 11.1912, 11.2886, 11.3876, 11.4865, 11.5841, 11.6835, 11.7865, 11.8923, 11.9946, 12.0965, 12.1993, 12.3024, 12.4062, 12.5122, 12.6135, 12.7169, 12.8215, 12.9294, 13.0368, 13.147, 13.2564, 13.3635, 13.4756, 13.5866, 13.6995, 13.8101, 13.9247, 14.0389, 14.1541, 14.2664, 14.382, 14.4982, 14.6203, 14.7393, 14.8608, 14.9838, 15.1039, 15.2233, 15.3433, 15.4657, 15.5909, 15.7162, 15.8381, 15.9635, 16.0894, 16.2129, 16.3406, 16.4672, 16.597, 16.7285, 16.8582, 16.9896, 17.1251, 17.2562, 17.3927, 17.5314, 17.6698, 17.8118, 17.9501, 18.0856, 18.2288, 18.3707, 18.5155, 18.6624, 18.8057, 18.9512, 19.0975, 19.2452, 19.3936, 19.5412, 19.6897, 19.839, 19.9925],
        [0, 0.0026, 0.0052, 0.0078, 0.0104, 0.013, 0.0156, 0.0182, 0.0208, 0.0234, 0.026, 0.0286, 0.0312, 0.0338, 0.0365, 0.0391, 0.0417, 0.0443, 0.0469, 0.0495, 0.0521, 0.0547, 0.0573, 0.0599, 0.0625, 0.0651, 0.0677, 0.0703, 0.0729, 0.0755, 0.0781, 0.0807, 0.0833, 0.0859, 0.0885, 0.0911, 0.0937, 0.0963, 0.0989, 0.1015, 0.1041, 0.1068, 0.1094, 0.112, 0.1146, 0.1172, 0.1198, 0.1224, 0.125, 0.1276, 0.1302, 0.1328, 0.1354, 0.138, 0.1406, 0.1432, 0.1458, 0.1484, 0.151, 0.1536, 0.1562, 0.1588, 0.1614, 0.164, 0.1666, 0.1692, 0.1718, 0.1745, 0.1771, 0.1797, 0.1823, 0.1849, 0.1875, 0.1901, 0.1927, 0.1953, 0.1979, 0.2005, 0.2031, 0.2057, 0.2083, 0.2109, 0.2135, 0.2161, 0.2187, 0.2213, 0.2239, 0.2265, 0.2291, 0.2317, 0.2343, 0.2369, 0.2395, 0.2421, 0.2448, 0.2474, 0.25, 0.2526, 0.2552, 0.2578, 0.2604, 0.263, 0.2656, 0.2682, 0.2708, 0.2734, 0.276, 0.2786, 0.2812, 0.2838, 0.2864, 0.289, 0.2916, 0.2942, 0.2968, 0.2994, 0.302, 0.3046, 0.3072, 0.3098, 0.3124, 0.3151, 0.3177, 0.3203, 0.3229, 0.3255, 0.3281, 0.3307, 0.3333, 0.3359, 0.3385, 0.3411, 0.3437, 0.3463, 0.3489, 0.3515, 0.3541, 0.3567, 0.3593, 0.3619, 0.3645, 0.3671, 0.3697, 0.3723, 0.3749, 0.3775, 0.3801, 0.3827, 0.3854, 0.388, 0.3906, 0.3932, 0.3958, 0.3984, 0.401, 0.4036, 0.4062, 0.4088, 0.4114, 0.414, 0.4166, 0.4192, 0.4218, 0.4244, 0.427, 0.4296, 0.4322, 0.4348, 0.4374, 0.44, 0.4426, 0.4452, 0.4478, 0.4504, 0.453, 0.4557, 0.4583, 0.4609, 0.4635, 0.4661, 0.4687, 0.4713, 0.4739, 0.4765, 0.4791, 0.4817, 0.4843, 0.4869, 0.4895, 0.4921, 0.4947, 0.4973, 0.4999, 0.5025, 0.5051, 0.5077, 0.5103, 0.5129, 0.5155, 0.5181, 0.5207, 0.5234, 0.526, 0.5286, 0.5312, 0.5338, 0.5364, 0.539, 0.5416, 0.5442, 0.5468, 0.5494, 0.552, 0.5546, 0.5572, 0.5598, 0.5624, 0.565, 0.5676, 0.5702, 0.5728, 0.5754, 0.578, 0.5806, 0.5832, 0.5858, 0.5884, 0.591, 0.5937, 0.5963, 0.5989, 0.6015, 0.6041, 0.6067, 0.6093, 0.6119, 0.6145, 0.6171], 'c--',
        label='TRACK reported')

    plt.plot(
        x_vals,
        y_vals, 'b',
        label=args.model_name + ' obtained')

    plt.xlabel('Intersection Angle Error')
    plt.ylabel('Data Proportion')
    plt.legend()
    plt.show()
