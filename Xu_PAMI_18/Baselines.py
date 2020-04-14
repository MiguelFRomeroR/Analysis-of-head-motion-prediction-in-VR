import sys
sys.path.insert(0, './')

import numpy as np
import os
import cv2
import pandas as pd
import argparse

from Xu_PAMI_18.Read_Dataset import get_dot_mat_data, get_original_dataset, recover_original_angles_from_quaternions_trace, get_videos_list, transform_the_degrees_in_range, transform_the_radians_to_original, recover_xyz_from_quaternions_trace
# from Reading_Dataset import get_dot_mat_data, get_original_dataset, recover_original_angles_from_quaternions_trace, get_videos_list, transform_the_degrees_in_range, transform_the_radians_to_original, recover_xyz_from_quaternions_trace
from Utils import eulerian_to_cartesian, rotationBetweenVectors, interpolate_quaternions, cartesian_to_eulerian
from Xu_PAMI_18.MeanOverlap import MeanOverlap
# from MeanOverlap import MeanOverlap
from TRACK_model import create_TRACK_model
from CVPR18_model import create_CVPR18_model
from position_only_baseline import create_pos_only_model

from SampledDataset import load_saliency


parser = argparse.ArgumentParser(description='Process the input parameters to evaluate the network.')

parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')

args = parser.parse_args()

ROOT_FOLDER = './Xu_PAMI_18'

DATASET_ROOT_FOLDER = os.path.join(ROOT_FOLDER, 'dataset')
OUTPUT_QUATERNION_FOLDER = os.path.join(ROOT_FOLDER, 'sampled_by_frame_dataset')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

RATE = 0.2

TRAINED_PREDICTION_HORIZON = 25
VIDEOS_TEST = ['KingKong', 'SpaceWar2', 'StarryPolar', 'Dancing', 'Guitar', 'BTSRun', 'InsideCar', 'RioOlympics', 'SpaceWar', 'CMLauncher2', 'Waterfall', 'Sunset', 'BlueWorld', 'Symphony', 'WaitingForLove']

# returns the frame rate of a video using openCV
def get_frame_rate(videoname, hardcoded=False):
    if hardcoded:
        dict_rates = {'A380': 29.97002997002997, 'AcerEngine': 30.0, 'AcerPredator': 25.0, 'AirShow': 30.0, 'BFG': 23.976023976023978, 'Bicycle': 29.97002997002997, 'BlueWorld': 29.97002997002997, 'BTSRun': 29.97002997002997, 'Camping': 29.97002997002997, 'CandyCarnival': 23.976023976023978, 'Castle': 25.0, 'Catwalks': 25.0, 'CMLauncher': 30.0, 'CMLauncher2': 30.0, 'CS': 30.0, 'DanceInTurn': 29.97002997002997, 'Dancing': 29.97002997002997, 'DrivingInAlps': 25.0, 'Egypt': 29.97002997002997, 'F5Fighter': 25.0, 'Flight': 23.976023976023978, 'GalaxyOnFire': 29.97002997002997, 'Graffiti': 25.0, 'GTA': 29.97002997002997, 'Guitar': 29.97002997002997, 'HondaF1': 29.97002997002997, 'InsideCar': 29.97002997002997, 'IRobot': 23.976023976023978, 'KasabianLive': 29.97002997002997, 'KingKong': 30.0, 'Lion': 29.97002997002997, 'LoopUniverse': 25.0, 'Manhattan': 29.97002997002997, 'MC': 30.0, 'MercedesBenz': 30.0, 'Motorbike': 29.97002997002997, 'Murder': 29.97002997002997, 'NotBeAloneTonight': 29.97002997002997, 'Orion': 25.0, 'Parachuting': 29.97002997002997, 'Parasailing': 29.97002997002997, 'Pearl': 30.0, 'Predator': 24.0, 'ProjectSoul': 29.97002997002997, 'Rally': 25.0, 'RingMan': 25.0, 'RioOlympics': 23.976023976023978, 'Roma': 24.0, 'Shark': 23.976023976023978, 'Skiing': 29.97002997002997, 'Snowfield': 29.97002997002997, 'SnowRopeway': 25.0, 'SpaceWar': 30.0, 'SpaceWar2': 30.0, 'Square': 30.0, 'StarryPolar': 12.0, 'StarWars': 25.0, 'StarWars2': 25.0, 'Stratosphere': 29.97002997002997, 'StreetFighter': 30.0, 'Sunset': 29.97002997002997, 'Supercar': 25.0, 'SuperMario64': 23.976023976023978, 'Surfing': 25.0, 'SurfingArctic': 25.0, 'Symphony': 25.0, 'TalkingInCar': 29.97002997002997, 'Terminator': 30.0, 'TheInvisible': 29.97002997002997, 'Village': 25.0, 'VRBasketball': 25.0, 'WaitingForLove': 30.0, 'Waterfall': 29.97002997002997, 'Waterskiing': 29.97002997002997, 'WesternSichuan': 30.0, 'Yacht': 29.97002997002997}
        return dict_rates[videoname]
    else:
        video_mp4 = videoname+'.mp4'
        video_path = os.path.join(DATASET_ROOT_FOLDER, 'videos', video_mp4)
        video = cv2.VideoCapture(video_path)
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps

# Sample the dataset to have a value each frame
def create_sampled_dataset_per_video_frame(original_dataset):
    dataset = {}
    for user in original_dataset.keys():
        dataset[user] = {}
        for video in original_dataset[user].keys():
            video_rate = 1.0 / get_frame_rate(video)
            print('creating sampled dataset', user, video, 'with rate', video_rate)
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
            dataset[user][video] = interpolate_quaternions(data_per_video[:, 0], data_per_video[:, 1:], rate=video_rate)
    return dataset

# Compute the Average MO score for all the videos in the dataset
def compute_baseline_error(dataset):
    mo_calculator = MeanOverlap(3840, 1920, 65.5 / 2, 3.0 / 4.0)
    error_per_video = {}
    for user in dataset.keys():
        for video in VIDEOS_TEST:
            print('computing error for user', user, 'and video', video)
            angles_per_video = recover_original_angles_from_quaternions_trace(dataset[user][video])
            if video not in error_per_video.keys():
                error_per_video[video] = []
            for t in range(len(angles_per_video)-1):
                sample_t = angles_per_video[t]
                sample_t_n = angles_per_video[t+1]
                mo_score = mo_calculator.calc_mo_deg([sample_t[0], sample_t[1]], [sample_t_n[0], sample_t_n[1]], is_centered=True)
                error_per_video[video].append(mo_score)
    avg_error_per_video = {}
    for video in VIDEOS_TEST:
        avg_error_per_video[video] = np.mean(error_per_video[video])
    return avg_error_per_video


def transform_angles_for_model(trace):
    new_trace = []
    for sample in trace:
        sample_yaw, sample_pitch = transform_the_degrees_in_range(sample[0], sample[1])
        sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
        new_trace.append(sample_new)
    return np.array(new_trace)

def transform_back_predicted_angles(sample):
    restored_yaw, restored_pitch = cartesian_to_eulerian(sample[0], sample[1], sample[2])
    restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
    return restored_yaw, restored_pitch

def compute_pos_only_baseline_error(dataset):
    m_window = 5
    h_window = 5
    model = create_pos_only_model(M_WINDOW=m_window, H_WINDOW=h_window)
    weights_file = os.path.join(ROOT_FOLDER,'pos_only','Models_EncDec_eulerian_Paper_Exp_init_5_in_5_out_5_end_5','weights.hdf5')
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
    else:
        raise Exception('Sorry, the file ./Xu_PAMI_18/pos_only/Models_EncDec_eulerian_Paper_Exp_init_5_in_5_out_5_end_5/weights.hdf5 doesn\'t exist.\nYou can:\n* Create it using the command:\n\t\"python training_procedure.py -train -gpu_id 0 -dataset_name Xu_PAMI_18 -model_name pos_only -m_window 5 -h_window 5 -exp_folder sampled_by_frame_dataset -provided_videos\" or \n* Download the file from:\n\thttps://unice-my.sharepoint.com/:u:/g/personal/miguel_romero-rondon_unice_fr/EaRpN6_HJZVCowdyrn5hs-sBK_pd0nH-p_re1jONYEyxPQ?e=7Pm1kU')
    mo_calculator = MeanOverlap(3840, 1920, 65.5 / 2, 3.0 / 4.0)
    error_per_video = {}
    for user in dataset.keys():
        for video in VIDEOS_TEST:
            print('computing error for user', user, 'and video', video)
            angles_per_video = recover_original_angles_from_quaternions_trace(dataset[user][video])
            transformed_angles_for_model = transform_angles_for_model(angles_per_video)
            if video not in error_per_video.keys():
                error_per_video[video] = []
            for t in range(len(angles_per_video)-1):
                if t < m_window+1:
                    inputs = transformed_angles_for_model[:t+1]
                    diff_len = m_window+1-len(inputs)
                    repeated_last_inp = np.repeat(inputs[-1:], diff_len, axis=0)
                    repeated_inputs = np.concatenate((inputs, repeated_last_inp))
                    encoder_inputs = np.array([repeated_inputs[-m_window-1:-1]])
                    decoder_inputs = np.array([repeated_inputs[-1:]])
                else:
                    encoder_inputs = np.array([transformed_angles_for_model[t-m_window:t]])
                    decoder_inputs = np.array([transformed_angles_for_model[t:t+1]])
                sample_t_pred = model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_inputs), transform_batches_cartesian_to_normalized_eulerian(decoder_inputs)])[0][0:1]
                sample_t_pred = transform_normalized_eulerian_to_cartesian(sample_t_pred)[0]
                sample_t = transform_back_predicted_angles(sample_t_pred)

                sample_t_n = angles_per_video[t+1]
                mo_score = mo_calculator.calc_mo_deg([sample_t[0], sample_t[1]], [sample_t_n[0], sample_t_n[1]], is_centered=True)
                error_per_video[video].append(mo_score)
    avg_error_per_video = {}
    for video in VIDEOS_TEST:
        avg_error_per_video[video] = np.mean(error_per_video[video])
    return avg_error_per_video


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

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216
# This function is used to compute the metrics used in the CVPR18 paper
def compute_pretrained_model_error(dataset, videos_list, model_name, history_window, model_weights_path):
    if model_name == 'TRACK':
        model = create_TRACK_model(history_window, TRAINED_PREDICTION_HORIZON, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
    elif model_name == 'CVPR18':
        model = create_CVPR18_model(history_window, TRAINED_PREDICTION_HORIZON, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
    elif model_name == 'pos_only':
        model = create_pos_only_model(history_window, TRAINED_PREDICTION_HORIZON)

    if os.path.isfile(model_weights_path):
        model.load_weights(model_weights_path)
    else:
        command = 'python training_procedure.py -train -gpu_id 0 -dataset_name Xu_PAMI_18 -model_name %s -m_window 5 -h_window 5 -exp_folder sampled_by_frame_dataset -provided_videos' % model_name
        raise Exception(
            'Sorry, the file '+model_weights_path+' doesn\'t exist.\nYou can:\n* Create it using the command:\n\t\"'+command+'\" or \n* Download the file from:\n\thttps://unice-my.sharepoint.com/:f:/g/personal/miguel_romero-rondon_unice_fr/EvQmshggLahKnBjIehzAbY0Bd-JDlzzFPYw9_R8IrGjQPA?e=Yk7f3c')

    saliency_folder = os.path.join(ROOT_FOLDER, 'extract_saliency/saliency')
    if model_name not in ['pos_only']:
        all_saliencies = {}
        for video in videos_list:
            if os.path.isdir(saliency_folder):
                all_saliencies[video] = load_saliency(saliency_folder, video)
            else:
                raise Exception('Sorry, the folder ./Xu_PAMI_18/extract_saliency doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"./Xu_PAMI_18/dataset/creation_of_scaled_images.sh\n\tpython ./Extract_Saliency/panosalnet.py -dataset_name PAMI_18\" or \n* Download the folder from:https://unice-my.sharepoint.com/:f:/g/personal/miguel_romero-rondon_unice_fr/Eir98fXEHKRBq9j-bgKGNTYBNN-_FQkvisJ1j9kOeVrB-Q?e=50lCOb\n\t')

    mo_calculator = MeanOverlap(3840, 1920, 65.5 / 2, 3.0 / 4.0)
    error_per_video = {}
    for user in dataset.keys():
        for video in VIDEOS_TEST:
            time_stamps_in_saliency = np.arange(0.0, len(all_saliencies[video]) * 0.2, 0.2)
            print('computing error for user', user, 'and video', video)
            angles_per_video = recover_original_angles_from_quaternions_trace(dataset[user][video])
            if video not in error_per_video.keys():
                error_per_video[video] = []
            # 1. Find the first time-stamp greater than 1 second (so that the input of the trace is greater than 5 when sampled in 0.2)
            # 1.1 Get the video rate (This is also the index of the first time-stamp at 1 sec)
            video_rate = int(np.ceil(get_frame_rate(video, hardcoded=True)))
            for t in range(video_rate, len(angles_per_video) - 1):
                # Remember that python arrays do not include the last index when sliced, e.g. [0, 1, 2, 3][:2] = [0, 1], in this case the input_data doesn't include the value at t+1
                input_data = dataset[user][video][t-video_rate:t+1]
                sample_t_n = angles_per_video[t+1]

                sampled_input_data = interpolate_quaternions(input_data[:, 0], input_data[:, 1:], rate=RATE, time_orig_at_zero=False)
                sampled_input_data_xyz = recover_xyz_from_quaternions_trace(sampled_input_data)

                # 2. For the saliency, get the time-stamps of the input trace and find the closest
                first_decoder_saliency_timestamp = input_data[-1, 0] + 0.2
                first_decoder_sal_id = np.argmin(np.power(time_stamps_in_saliency - first_decoder_saliency_timestamp, 2.0))

                if model_name not in ['pos_only', 'no_motion']:
                    encoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][first_decoder_sal_id - history_window:first_decoder_sal_id], axis=-1)])
                    # ToDo: Be careful here, we are using TRAINED_PREDICTION_HORIZON to load future saliencies
                    decoder_sal_inputs_for_sample = np.zeros((1, TRAINED_PREDICTION_HORIZON, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1))
                    taken_saliencies = all_saliencies[video][first_decoder_sal_id:min(first_decoder_sal_id + TRAINED_PREDICTION_HORIZON, len(all_saliencies[video]))]
                    # decoder_sal_inputs_for_sample = np.array([np.expand_dims(taken_saliencies, axis=-1)])
                    decoder_sal_inputs_for_sample[0, :len(taken_saliencies), :, :, 0] = taken_saliencies

                encoder_pos_inputs_for_sample = [sampled_input_data_xyz[-history_window - 1:-1, 1:]]
                decoder_pos_inputs_for_sample = [sampled_input_data_xyz[-1:, 1:]]

                # 3. predict
                if model_name == 'TRACK':
                    model_prediction = model.predict(
                        [np.array(encoder_pos_inputs_for_sample), np.array(encoder_sal_inputs_for_sample),
                         np.array(decoder_pos_inputs_for_sample), np.array(decoder_sal_inputs_for_sample)])[0]
                elif model_name == 'CVPR18':
                    model_prediction = model.predict(
                        [np.array(encoder_pos_inputs_for_sample), np.array(decoder_pos_inputs_for_sample),
                         np.array(decoder_sal_inputs_for_sample)])[0]
                elif model_name == 'pos_only':
                    model_pred = model.predict(
                        [transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample),
                         transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
                    model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)

                # 4. upsample the predicted trace from 0.2 sec to the video rate
                sample_orig = np.array([1, 0, 0])
                quat_rot_1 = rotationBetweenVectors(sample_orig, sampled_input_data_xyz[-1, 1:])
                quat_rot_1 = np.array([quat_rot_1[0], quat_rot_1[1], quat_rot_1[2], quat_rot_1[3]])
                quat_rot_2 = rotationBetweenVectors(sample_orig, model_prediction[0])
                quat_rot_2 = np.array([quat_rot_2[0], quat_rot_2[1], quat_rot_2[2], quat_rot_2[3]])

                interpolated = interpolate_quaternions([0.0, RATE], [quat_rot_1, quat_rot_2], rate=1.0/video_rate)
                pred_samples = recover_original_angles_from_quaternions_trace(interpolated)


                pred_sample_t_n = pred_samples[1]

                mo_score = mo_calculator.calc_mo_deg([pred_sample_t_n[0], pred_sample_t_n[1]], [sample_t_n[0], sample_t_n[1]], is_centered=True)
                error_per_video[video].append(mo_score)
    avg_error_per_video = {}
    for video in VIDEOS_TEST:
        avg_error_per_video[video] = np.mean(error_per_video[video])
    return avg_error_per_video

# ToDo Copied exactly from Reading_Dataset, but changed the output_folder to OUTPUT_QUATERNION_FOLDER
def store_dataset(dataset):
    for user in dataset.keys():
        for video in dataset[user].keys():
            video_folder = os.path.join(OUTPUT_QUATERNION_FOLDER, video)
            # Create the folder for the video if it doesn't exist
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            path = os.path.join(video_folder, user)
            df = pd.DataFrame(dataset[user][video])
            df.to_csv(path, header=False, index=False)

# ToDo Copied exactly from Xu_CVPR_18/Baselines, changed the folder to OUTPUT_QUATERNION_FOLDER
def load_dataset_sampled_per_video_frame():
    list_of_videos = os.listdir(os.path.join(OUTPUT_QUATERNION_FOLDER))
    dataset = {}
    for video in list_of_videos:
        for user in os.listdir(os.path.join(OUTPUT_QUATERNION_FOLDER, video)):
            if user not in dataset.keys():
                dataset[user] = {}
            path = os.path.join(OUTPUT_QUATERNION_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset

if __name__ == "__main__":
    ### This experiment follows the specifications in Xu_PAMI_18 Paper:
    # "the metric of mean overlap (MO), which measures how close the predicted HM position is to the groundtruth HM position. MO ranges from 0 to 1, and a larger MO indicates a more
    # precise prediction. Specifically, MO is defined as, MO = A(FoV_p \cap FoV_g) / A(FoV_p \cup FoV_g), where FoV_p and FoV_g represent the FoVs at the predicted and ground-truth HM positions,
    # respectively. A represents the area of a panoramic region, which accounts for number fo pixels."
    ### We took the metric MO from the code of Xu_PAMI_18 authors in "https://github.com/YuhangSong/DHP/blob/master/MeanOverlap.py"
    # "The online-DHP approach refers to predicting a specific subject's HM position (\hat{x}_{t+1}, \hat{y}_{t+1}) at frame t+1, given his/her HM positions
    # {(x_1, y_1), ..., (x_t, y_t)} till frame t. [...] the output is the predicted HM position (\hat{x}_{t+1}, \hat{y}_{t+1}) at the next frame for the viewer.
    # Stage II: Prediction: [...] When entering the prediction stage, the DRL model trained in the first stage is used to produce the HM position as follows.
    # [...] the HM position (\hat{x}_{t+1}, \hat{y}_{t+1}) can be predicted, given the ground-truth HM position (x_t, y_t) and the estimated HM scanpath (\hat{\alpha}, \hat{v_t}) at frame t."
    ### Meaning that the ground truth of the previous HM position is used to predict each HM position frame by frame.
    # dot_mat_data = get_dot_mat_data()
    # original_dataset = get_original_dataset(get_videos_list(dot_mat_data))
    # sampled_dataset_per_vid_frame = create_sampled_dataset_per_video_frame(original_dataset)
    # store_dataset(sampled_dataset_per_vid_frame)

    if args.model_name == 'no_motion':
        sampled_dataset_per_vid_frame = load_dataset_sampled_per_video_frame()
        avg_error_per_video = compute_baseline_error(sampled_dataset_per_vid_frame)
    elif args.model_name == 'pos_only':
        sampled_dataset_per_vid_frame = load_dataset_sampled_per_video_frame()
        avg_error_per_video = compute_pos_only_baseline_error(sampled_dataset_per_vid_frame)
    else:
        if args.model_name == 'TRACK':
            model_weights_path = os.path.join(ROOT_FOLDER, 'TRACK/Models_EncDec_3DCoords_ContSal_init_5_in_5_out_25_end_25/weights.hdf5')
        elif args.model_name == 'CVPR18':
            model_weights_path = os.path.join(ROOT_FOLDER, 'CVPR18/Models_EncDec_3DCoords_ContSal_init_5_in_5_out_25_end_25/weights.hdf5')
        sampled_dataset_per_vid_frame = load_dataset_sampled_per_video_frame()
        avg_error_per_video = compute_pretrained_model_error(sampled_dataset_per_vid_frame, videos_list=VIDEOS_TEST, model_name=args.model_name, history_window=5, model_weights_path=model_weights_path)
    print(avg_error_per_video)

    print('| Method                 | KingKong | SpaceWar2 | StarryPolar | Dancing | Guitar | BTSRun | InsideCar | RioOlympics | SpaceWar | CMLauncher2 | Waterfall | Sunset | BlueWorld | Symphony | WaitingForLove | Average |')
    print('| ---------------------- | -------- | --------- | ----------- | ------- | ------ | ------ | --------- | ----------- | -------- | ----------- | --------- | ------ | --------- | -------- | -------------- | ------- |')
    print('| PAMI18 reported        |    0.809 |     0.763 |       0.549 |   0.859 |  0.785 |  0.878 |     0.847 |       0.820 |    0.626 |       0.763 |     0.667 |  0.659 |     0.693 |    0.747 |          0.863 |   0.753 |')
    print('| No-motion reported     |    0.974 |     0.963 |       0.906 |   0.979 |  0.970 |  0.983 |     0.976 |       0.966 |    0.965 |       0.981 |     0.973 |  0.964 |     0.970 |    0.968 |          0.978 |   0.968 |')
    print('| Position-only reported |    0.983 |     0.977 |       0.930 |   0.984 |  0.977 |  0.987 |     0.982 |       0.976 |    0.976 |       0.989 |     0.984 |  0.973 |     0.979 |    0.976 |          0.982 |   0.977 |')
    print('| TRACK-CBSal reported   |    0.974 |     0.964 |       0.912 |   0.978 |  0.968 |  0.982 |     0.974 |       0.965 |    0.965 |       0.981 |     0.972 |  0.964 |     0.970 |    0.969 |          0.977 |   0.968 |')

    average_for_all_videos = np.mean([avg_error_per_video[key] for key in avg_error_per_video.keys()])
    print('| %s obtained | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (args.model_name, avg_error_per_video['KingKong'], avg_error_per_video['SpaceWar2'], avg_error_per_video['StarryPolar'], avg_error_per_video['Dancing'], avg_error_per_video['Guitar'], avg_error_per_video['BTSRun'], avg_error_per_video['InsideCar'], avg_error_per_video['RioOlympics'], avg_error_per_video['SpaceWar'], avg_error_per_video['CMLauncher2'], avg_error_per_video['Waterfall'], avg_error_per_video['Sunset'], avg_error_per_video['BlueWorld'], avg_error_per_video['Symphony'], avg_error_per_video['WaitingForLove'], average_for_all_videos))