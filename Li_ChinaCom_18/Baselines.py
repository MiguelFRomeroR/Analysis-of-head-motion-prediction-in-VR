import sys
sys.path.insert(0, './')

import os
import numpy as np
import pandas as pd
from Utils import eulerian_to_cartesian, cartesian_to_eulerian
import matplotlib.pyplot as plt
import cv2
import csv
from sklearn.metrics import accuracy_score, f1_score, label_ranking_loss
from position_only_baseline import create_pos_only_model
import argparse

parser = argparse.ArgumentParser(description='Process the input parameters to evaluate the network.')

parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')
parser.add_argument('-video', action='store', dest='video', help='The video name on which the model will be tested.')

args = parser.parse_args()

if args.video is None:
    print('you should specify which video to test in with the parameter -video "videoname"')

ROOT = '.'
ROOT_FOLDER = './Fan_NOSSDAV_17'
OUR_TILE_PROB_FOLDER = './Fan_NOSSDAV_17/dataset/sensory/tile_replica'

DATASET_ROOT_FOLDER = os.path.join(ROOT_FOLDER, 'dataset')
SALIENCY_FOLDER = os.path.join(ROOT_FOLDER, 'extract_saliency', 'saliency')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

NUM_TILES_WIDTH_SAL = 384
NUM_TILES_HEIGHT_SAL = 216

# From https://people.cs.nctu.edu.tw/~chuang/pubs/pdf/2017mmsys.pdf 360\degree Video Viewing Dataset:
# We divide each frame, which is mapped in equirectangular model, into 192x192 tiles, so there are *200* tiles in total.
# Then we number the tiles from upper-left to lower-right.
NUM_TILES_HEIGHT = 10
NUM_TILES_WIDTH = 20
# From https://people.cs.nctu.edu.tw/~chuang/pubs/pdf/2017mmsys.pdf 360\degree Video Viewing Dataset:
# We assume the FoVs are modeled by 100\degreex100\degree circles.
FOV_SIZE = 110.0

# Check it with get_frame_rate(video) for all videos
ORIGINAL_SAMPLING_RATE = 1./30.0

MODEL_SAMPLING_RATE = 0.2

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

# From NOSSDAV17 paper:
# "We let both the sliding window size M_WINDOW and prediction window size H_WINDOW to be 30.
M_WINDOW = 30
H_WINDOW = 30

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
def get_frame_rate(videoname, hardcoded=False):
    if hardcoded:
        dict_rates = {'coaster': 30.0, 'coaster2': 30.0, 'diving': 30.0, 'drive': 30.0, 'game': 30.0, 'landscape': 30.0, 'pacman': 30.0, 'panel': 30.0, 'ride': 30.0, 'sport': 30.0}
        return dict_rates[videoname]
    else:
        video_mp4 = videoname+'_saliency.mp4'
        video_path = os.path.join(DATASET_ROOT_FOLDER, 'content/saliency', video_mp4)
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
    for root, directories, files in os.walk(os.path.join(DATASET_ROOT_FOLDER, 'sensory/orientation')):
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
            video_rate = 1.0 / get_frame_rate(video, hardcoded=True)
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

def read_tile_info(video, user):
    filename = os.path.join(DATASET_ROOT_FOLDER, 'sensory/tile', video + '_' + user + '_tile.csv')
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

# From https://people.cs.nctu.edu.tw/~chuang/pubs/pdf/2017mmsys.pdf 360\degree Video Viewing Dataset:
# While view orientation log files give the center of viewer's FoVs, determining which tiles are needed to render the
# FoVs equire extra calculations. We assume the FoVs are modeled by 100\degreex100\degree circles.
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

def read_replica_tile_info(video, user):
    replica_filename = '%s_%s_created_tile.npy' % (video, user)
    file_path = os.path.join(OUR_TILE_PROB_FOLDER, replica_filename)
    if os.path.isfile(file_path):
        replica_tile_prob_for_trace = np.load(file_path)
    else:
        raise Exception('Sorry, the folder ./Fan_NOSSDAV_17/dataset/sensory/tile_replica doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"python ./Fan_NOSSDAV_17/Read_Dataset.py -create_tile_replica\" or \n* Download the tile_replica folder from:\n\thttps://unice-my.sharepoint.com/:f:/g/personal/miguel_romero-rondon_unice_fr/EgbWOKSxKwNCj2qKsuuKK1YBQPLubATD4MSmI5-seDEYjQ?e=WzHYVw')
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


def no_motion_baseline_metrics(original_dataset_cartesian, video):
    accuracy_results = []
    f1_score_results = []
    # ranking_results = []
    for num_user, user in enumerate(original_dataset_cartesian.keys()):
        repl_tiles_map = read_replica_tile_info(video, user)
        for t in range(M_WINDOW, len(original_dataset_cartesian[user][video]) - H_WINDOW):
            print('ChinaCom', video, 'computing no_motion metrics for user', num_user, '/', len(original_dataset_cartesian.keys()), 'time-stamp:', t)
            pred_tile_map = repl_tiles_map[t]
            future_tile_maps = repl_tiles_map[t+1:t+H_WINDOW+1]
            for x_i, tile_map in enumerate(future_tile_maps):
                accuracy_results.append(accuracy_score(np.ndarray.flatten(tile_map), np.ndarray.flatten(pred_tile_map)))
                f1_score_results.append(f1_score(np.ndarray.flatten(tile_map), np.ndarray.flatten(pred_tile_map)))
                # ranking_results.append(label_ranking_loss(tile_map, pred_tile_map))
    # print('Accuracy', np.mean(accuracy_results) * 100, 'F-Score', np.mean(f1_score_results), 'Rank. Loss', np.mean(ranking_results))
    return np.mean(accuracy_results) * 100, np.mean(f1_score_results)


# ToDo import from training_procedure (for now we just copied the function)
# from training_procedure import transform_batches_cartesian_to_normalized_eulerian
def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

# ToDo copied from Nguyen_MM_18/Baselines.py
def transform_normalized_eulerian_to_cartesian(position):
    position = position * np.array([2*np.pi, np.pi])
    eulerian_samples = eulerian_to_cartesian(position[0], position[1])
    return np.array(eulerian_samples)

def get_pos_only_prediction(model, trace, m_window_trained_model):
    indices_input_trace = np.linspace(0, len(trace) - 1, m_window_trained_model + 1, dtype=int)

    subsampled_trace = trace[indices_input_trace]

    encoder_pos_inputs = subsampled_trace[-m_window_trained_model - 1:-1]
    decoder_pos_inputs = subsampled_trace[-1:]

    # ToDo Hardcoded value to slice the prediction for the first second
    model_prediction = model.predict([np.array(transform_batches_cartesian_to_normalized_eulerian([encoder_pos_inputs])), np.array(transform_batches_cartesian_to_normalized_eulerian([decoder_pos_inputs]))])[0]

    head_map_pred = [from_position_to_tile_probability_cartesian(transform_normalized_eulerian_to_cartesian(pos)) for pos in model_prediction]

    # ToDo the repetition is hardcoded
    return np.repeat(head_map_pred, [6, 6, 6, 6, 6], axis=0)


def pos_only_metrics(original_dataset_cartesian, video):
    M_WINDOW_TRAINED_MODEL = 5
    H_WINDOW_TRAINED_MODEL = 5
    model = create_pos_only_model(M_WINDOW_TRAINED_MODEL, H_WINDOW_TRAINED_MODEL)

    weights_file = os.path.join(ROOT, 'Li_ChinaCom_18', 'pos_only', 'Models_EncDec_eulerian_init_5_in_5_out_5_end_5_'+video, 'weights.hdf5')
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
    else:
        raise Exception('Sorry, the folder ./Li_ChinaCom_18/pos_only/ doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"python training_procedure.py -train -gpu_id 0 -dataset_name Li_ChinaCom_18 -model_name pos_only -m_window 5 -h_window 5 -video_test_chinacom VIDEO_NAME -provided_videos\" or \n* Download the file from:\n\thttps://unice-my.sharepoint.com/:f:/g/personal/miguel_romero-rondon_unice_fr/EmSloFxbiLFKiQcG0Br3KdoBnWjJ_CiuQPaauWI9ID6j0g?e=arhfqf')

    accuracy_results = []
    f1_score_results = []
    # ranking_results = []
    for num_user, user in enumerate(original_dataset_cartesian.keys()):
        repl_tiles_map = read_replica_tile_info(video, user)
        for t in range(M_WINDOW, len(original_dataset_cartesian[user][video]) - H_WINDOW):
            print('ChinaCom', video, 'computing no_motion metrics for user', num_user, '/', len(original_dataset_cartesian.keys()), 'time-stamp:', t)
            past_positions = original_dataset_cartesian[user][video][t-M_WINDOW:t+1]
            pred_tile_map = get_pos_only_prediction(model, past_positions, M_WINDOW_TRAINED_MODEL)
            # future_positions = original_dataset_cartesian[user][video][t+1:t+H_WINDOW+1]
            future_tile_maps = repl_tiles_map[t+1:t+H_WINDOW+1]
            for x_i, tile_map in enumerate(future_tile_maps):
                accuracy_results.append(accuracy_score(np.ndarray.flatten(tile_map), np.ndarray.flatten(pred_tile_map[x_i])))
                f1_score_results.append(f1_score(np.ndarray.flatten(tile_map), np.ndarray.flatten(pred_tile_map[x_i])))
                # ranking_results.append(label_ranking_loss(tile_map, pred_tile_map[x_i]))
    # print('Accuracy', np.mean(accuracy_results) * 100, 'F-Score', np.mean(f1_score_results), 'Rank. Loss', np.mean(ranking_results))
    return np.mean(accuracy_results) * 100, np.mean(f1_score_results)


def transform_dataset_in_cartesian(original_dataset):
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating cartesian dataset', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            data_per_video = []
            for sample in original_dataset[user][video]:
                sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
                sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
                data_per_video.append(sample_new)
            dataset[user][video] = np.array(data_per_video)
    return dataset

if __name__ == "__main__":
    original_dataset = get_original_dataset()
    filtered_dataset = filter_dataset_strange_traces(original_dataset)
    cartesian_dataset = transform_dataset_in_cartesian(filtered_dataset)

    if args.model_name == 'no_motion':
        accuracy_result, f1_score_result = no_motion_baseline_metrics(cartesian_dataset, args.video)
    elif args.model_name == 'pos_only':
        accuracy_result, f1_score_result = pos_only_metrics(cartesian_dataset, args.video)

    print('| Videoname       | No-motion baseline | Position-only baseline |     ChinaCom18     |')
    print('| --------------- | ------------------ | ---------------------- | ------------------ |')
    print('| Videoname       | Accuracy | F-score |  Accuracy  |  F-score  | Accuracy | F-score |')
    print('| Hog Rider       |   96.29% |  0.8858 |     96.97% |    0.9066 |   77.09% |  0.2742 |')
    print('| Driving with    |   95.96% |  0.8750 |     96.59% |    0.9843 |   77.34% |  0.2821 |')
    print('| Shark Shipwreck |   95.23% |  0.8727 |     96.12% |    0.8965 |   83.26% |  0.5259 |')
    print('| Mega Coaster    |   97.20% |  0.9144 |     97.71% |    0.9299 |   88.90% |  0.7011 |')
    print('| Roller Coaster  |   96.99% |  0.9104 |     97.50% |    0.9256 |   88.28% |  0.6693 |')
    print('| Chariot-Race    |   97.07% |  0.8802 |     96.91% |    0.9056 |   87.79% |  0.6040 |')
    print('| SFR Sport       |   96.00% |  0.8772 |     96.91% |    0.9054 |   89.29% |  0.7282 |')
    print('| Pac-Man         |   96.83% |  0.8985 |     97.16% |    0.9089 |   87.45% |  0.6826 |')
    print('| Peris Panel     |   95.60% |  0.8661 |     96.54% |    0.8947 |   89.12% |  0.7246 |')
    print('| Kangaroo Island |   95.35% |  0.8593 |     96.54% |    0.8954 |   82.62% |  0.5308 |')
    print('|                 |          |         |            |           |          |         |')
    print('| Average         |   96.15% |  0.8840 |     96.90% |    0.9063 |   72.54% |  0.5155 |')

    print('Obtained result is:')
    print('| Videoname       | No-motion baseline | Position-only baseline |     ChinaCom18     |')
    if args.model_name == 'no_motion':
        if args.video == "game":
            print('| Hog Rider       |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "driving":
            print('| Driving with    |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "diving":
            print('| Shark Shipwreck |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "coaster2":
            print('| Mega Coaster    |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "coaster":
            print('| Roller Coaster  |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "ride":
            print('| Chariot-Race    |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "sport":
            print('| SFR Sport       |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "pacman":
            print('| Pac-Man         |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "panel":
            print('| Peris Panel     |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "landscape":
            print('| Kangaroo Island |   %.2f%% |  %.4f |     ------ |    ------ |   ------ |  ------ |' % (accuracy_result, f1_score_result))
    elif args.model_name == 'pos_only':
        if args.video == "game":
            print('| Hog Rider       |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "driving":
            print('| Driving with    |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "diving":
            print('| Shark Shipwreck |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "coaster2":
            print('| Mega Coaster    |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "coaster":
            print('| Roller Coaster  |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "ride":
            print('| Chariot-Race    |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "sport":
            print('| SFR Sport       |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "pacman":
            print('| Pac-Man         |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "panel":
            print('| Peris Panel     |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))
        elif args.video == "landscape":
            print('| Kangaroo Island |   ------ |  ------ |     %.2f%% |    %.4f |   ------ |  ------ |' % (accuracy_result, f1_score_result))