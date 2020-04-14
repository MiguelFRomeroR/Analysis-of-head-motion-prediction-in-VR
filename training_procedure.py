import sys
sys.path.insert(0, './')

import numpy as np
import keras
from SampledDataset import read_sampled_positions_for_trace, load_saliency, load_true_saliency, get_video_ids, get_user_ids, get_users_per_video, split_list_by_percentage, partition_in_train_and_test_without_any_intersection, partition_in_train_and_test_without_video_intersection, partition_in_train_and_test
from TRACK_model import create_TRACK_model
from CVPR18_model import create_CVPR18_model
import MM18_model
from position_only_baseline import create_pos_only_model
from CVPR18_original_model import create_CVPR18_orig_Model, auto_regressive_prediction
from Xu_CVPR_18.Read_Dataset import get_videos_train_and_test_from_file
import os
from Utils import cartesian_to_eulerian, eulerian_to_cartesian, get_max_sal_pos, load_dict_from_csv, all_metrics
import argparse
from Saliency_only_baseline import get_most_salient_points_per_video, predict_most_salient_point
from TRACK_AblatSal_model import create_TRACK_AblatSal_model
from TRACK_AblatFuse_model import create_TRACK_AblatFuse_model
from ContentBased_Saliency_baseline import get_most_salient_content_based_points_per_video, predict_most_salient_cb_point

# e.g. usage: python training_procedure.py -server_name octopus -gpu_id 7 -dataset_name Xu_PAMI_18 -model_name TRACK -m_window 5 -h_window 25
# uses the server octopus
# GPU: 7
# Dataset: Xu_PAMI_18
# Model: TRACK
# M_WINDOW: 5 (input window)
# H_WINDOW: 25 (prediction window)

# e.g. python training_procedure.py -server_name bird -gpu_id 1 -dataset_name Xu_CVPR_18 -model_name pos_only -m_window 5 -h_window 5 -exp_folder sampled_dataset_replica -provided_videos
# python training_procedure.py -validate -server_name octopus -gpu_id "" -dataset_name Xu_PAMI_18 -model_name no_motion -m_window 5 -h_window 25
#python training_procedure.py -train -gpu_id 0 -dataset_name Li_ChinaCom_18 -model_name pos_only -m_window 5 -h_window 5 -video_test_chinacom VIDEONAME

parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')

parser.add_argument('-train', action="store_true", dest='train_flag', help='Flag that tells if we will run the training procedure.')
parser.add_argument('-evaluate', action="store_true", dest='evaluate_flag', help='Flag that tells if we will run the evaluation procedure.')
parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-dataset_name', action='store', dest='dataset_name', help='The name of the dataset used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')
parser.add_argument('-init_window', action='store', dest='init_window', help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
parser.add_argument('-m_window', action='store', dest='m_window', help='Past history window.', type=int)
parser.add_argument('-h_window', action='store', dest='h_window', help='Prediction window.', type=int)
parser.add_argument('-end_window', action='store', dest='end_window', help='(Optional) Final buffer (to avoid having samples with less outputs).', type=int)
parser.add_argument('-exp_folder', action='store', dest='exp_folder', help='Used when the dataset folder of the experiment is different than the default dataset.')
parser.add_argument('-provided_videos', action="store_true", dest='provided_videos', help='Flag that tells whether the list of videos is provided in a global variable.')
parser.add_argument('-use_true_saliency', action="store_true", dest='use_true_saliency', help='Flag that tells whether to use true saliency (if not set, then the content based saliency is used).')
parser.add_argument('-num_of_peaks', action="store", dest='num_of_peaks', help='Value used to get number of peaks from the true_saliency baseline.')
parser.add_argument('-video_test_chinacom', action="store", dest='video_test_chinacom', help='Which video will be used to test in ChinaCom, the rest of the videos will be used to train')
parser.add_argument('-metric', action="store", dest='metric', help='Which metric to use, by default, orthodromic distance is used.')

args = parser.parse_args()

gpu_id = args.gpu_id
dataset_name = args.dataset_name
model_name = args.model_name
# Buffer window in timesteps
M_WINDOW = args.m_window
# Forecast window in timesteps (5 timesteps = 1 second) (Used in the network to predict)
H_WINDOW = args.h_window
# Initial buffer (to avoid stationary part)
if args.init_window is None:
    INIT_WINDOW = M_WINDOW
else:
    INIT_WINDOW = args.init_window
# final buffer (to avoid having samples with less outputs)
if args.end_window is None:
    END_WINDOW = H_WINDOW
else:
    END_WINDOW = args.end_window
# This variable is used when we use the dataset of the experiment run in the respective paper
# e.g. Xu_CVPR_18 predicts the gaze using the last 5 gaze positions to predict the next 5 gaze positions
EXP_FOLDER = args.exp_folder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

assert dataset_name in ['Xu_PAMI_18', 'AVTrack360', 'Xu_CVPR_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'David_MMSys_18', 'Li_ChinaCom_18']
assert model_name in ['TRACK', 'CVPR18', 'pos_only', 'no_motion', 'most_salient_point', 'true_saliency', 'content_based_saliency', 'CVPR18_orig', 'TRACK_AblatSal', 'TRACK_AblatFuse', 'MM18']

if model_name in ['true_saliency', 'content_based_saliency']:
    assert int(args.num_of_peaks) > 0

if args.metric is None:
    metric = all_metrics['orthodromic']
else:
    assert args.metric in all_metrics.keys()
    metric = all_metrics[args.metric]

# Fixing random state for reproducibility
np.random.seed(19680801)

EPOCHS = 500

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

RATE = 0.2

root_dataset_folder = os.path.join('./', dataset_name)

# If EXP_FOLDER is defined, add "Paper_exp" to the results name and use the folder in EXP_FOLDER as dataset folder
if EXP_FOLDER is None:
    EXP_NAME = '_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, 'sampled_dataset')
else:
    EXP_NAME = '_Paper_Exp_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, EXP_FOLDER)

if dataset_name == 'Li_ChinaCom_18':
    EXP_NAME = EXP_NAME + '_' + args.video_test_chinacom

SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'extract_saliency/saliency')

if model_name == 'MM18':
    TRUE_SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'mm18_true_saliency')
else:
    TRUE_SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'true_saliency')

if model_name == 'TRACK':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
if model_name == 'TRACK_AblatSal':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
if model_name == 'TRACK_AblatFuse':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
elif model_name == 'CVPR18':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
elif model_name == 'pos_only':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'pos_only/Results_EncDec_eulerian' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'pos_only/Models_EncDec_eulerian' + EXP_NAME)
elif model_name == 'CVPR18_orig':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Results_EncDec_2DNormalized_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Models_EncDec_2DNormalized_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Results_Enc_2DNormalized_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Models_Enc_2DNormalized_ContSal' + EXP_NAME)
elif model_name == 'MM18':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'MM18/Results_Seq2One_2DNormalized_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'MM18/Models_Seq2One_2DNormalized_TrueSal' + EXP_NAME)

PERC_VIDEOS_TRAIN = 0.8
PERC_USERS_TRAIN = 0.5

BATCH_SIZE = 128.0

TRAIN_MODEL = False
EVALUATE_MODEL = False
if args.train_flag:
    TRAIN_MODEL = True
if args.evaluate_flag:
    EVALUATE_MODEL = True

videos = get_video_ids(SAMPLED_DATASET_FOLDER)
users = get_user_ids(SAMPLED_DATASET_FOLDER)
users_per_video = get_users_per_video(SAMPLED_DATASET_FOLDER)

if args.provided_videos:
    if dataset_name == 'Xu_CVPR_18':
        videos_train, videos_test = get_videos_train_and_test_from_file(root_dataset_folder)
        partition = partition_in_train_and_test_without_video_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_per_video)
    if dataset_name == 'Xu_PAMI_18':
        # From PAMI_18 paper:
        # For evaluating the performance of offline-DHP, we randomly divided all 76 panoramic sequences of our PVS-HM database into a training set (61 sequences) and a test set (15 sequences).
        # For evaluating the performance of online-DHP [...]. Since the DRL network of offline-DHP was learned over 61 training sequences and used as the initial model of online-DHP, our comparison was conducted on all 15 test sequences of our PVS-HM database.
        videos_test = ['KingKong', 'SpaceWar2', 'StarryPolar', 'Dancing', 'Guitar', 'BTSRun', 'InsideCar', 'RioOlympics', 'SpaceWar', 'CMLauncher2', 'Waterfall', 'Sunset', 'BlueWorld', 'Symphony', 'WaitingForLove']
        videos_train = ['A380', 'AcerEngine', 'AcerPredator', 'AirShow', 'BFG', 'Bicycle', 'Camping', 'CandyCarnival', 'Castle', 'Catwalks', 'CMLauncher', 'CS', 'DanceInTurn', 'DrivingInAlps', 'Egypt', 'F5Fighter', 'Flight', 'GalaxyOnFire', 'Graffiti', 'GTA', 'HondaF1', 'IRobot', 'KasabianLive', 'Lion', 'LoopUniverse', 'Manhattan', 'MC', 'MercedesBenz', 'Motorbike', 'Murder', 'NotBeAloneTonight', 'Orion', 'Parachuting', 'Parasailing', 'Pearl', 'Predator', 'ProjectSoul', 'Rally', 'RingMan', 'Roma', 'Shark', 'Skiing', 'Snowfield', 'SnowRopeway', 'Square', 'StarWars', 'StarWars2', 'Stratosphere', 'StreetFighter', 'Supercar', 'SuperMario64', 'Surfing', 'SurfingArctic', 'TalkingInCar', 'Terminator', 'TheInvisible', 'Village', 'VRBasketball', 'Waterskiing', 'WesternSichuan', 'Yacht']
        partition = partition_in_train_and_test_without_video_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_per_video)
    if dataset_name == 'Fan_NOSSDAV_17':
        train_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'train_set'), ['user', 'video'])
        test_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'test_set'), ['user', 'video'])
        videos_test = ['coaster', 'drive', 'pacman', 'game', 'diving', 'coaster2', 'sport', 'ride', 'panel', 'landscape']
        print(train_traces)
        print(test_traces)
        partition = partition_in_train_and_test(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, train_traces, test_traces)
    if dataset_name == 'David_MMSys_18':
        train_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'train_set'), ['user', 'video'])
        test_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'test_set'), ['user', 'video'])
        print(train_traces)
        print(test_traces)
        partition = partition_in_train_and_test(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, train_traces, test_traces)
        videos_test = ['1_PortoRiverside', '3_PlanEnergyBioLab', '5_Waterpark', '14_Warship', '16_Turtle']
    if dataset_name == 'Nguyen_MM_18':
        videos_train = ['0', '1', '2', '3', '6']
        videos_test = ['4', '5', '7', '8']
        partition = partition_in_train_and_test_without_video_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_per_video)
    if dataset_name == 'Li_ChinaCom_18':
        videos_train = []
        for video in videos:
            if video != args.video_test_chinacom:
                videos_train.append(video)
        videos_test = [args.video_test_chinacom]
        partition = partition_in_train_and_test_without_video_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_per_video)
else:
    videos_train, videos_test = split_list_by_percentage(videos, PERC_VIDEOS_TRAIN)
    users_train, users_test = split_list_by_percentage(users, PERC_USERS_TRAIN)
    # Datasets
    partition = partition_in_train_and_test_without_any_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_train, users_test)

# Dictionary that stores the traces per video and user
all_traces = {}
for video in videos:
    all_traces[video] = {}
    for user in users_per_video[video]:
        all_traces[video][user] = read_sampled_positions_for_trace(SAMPLED_DATASET_FOLDER, str(video), str(user))

if model_name == 'MM18':
    MM18_model.create_gt_sal(TRUE_SALIENCY_FOLDER, all_traces)

# Load the saliency only if it's not the position_only baseline
if model_name not in ['pos_only', 'no_motion', 'true_saliency', 'content_based_saliency']:
    all_saliencies = {}
    if args.use_true_saliency:
        for video in videos:
            if model_name == 'MM18':
                all_saliencies[video] = MM18_model.get_true_saliency(TRUE_SALIENCY_FOLDER, video)
            else:
                all_saliencies[video] = load_true_saliency(TRUE_SALIENCY_FOLDER, video)
    else:
        for video in videos:
            all_saliencies[video] = load_saliency(SALIENCY_FOLDER, video)

if model_name == 'MM18':
    all_headmaps = {}
    for video in videos:
        all_headmaps[video] = {}
        for user in users_per_video[video]:
            all_headmaps[video][user] = MM18_model.get_headmaps(all_traces[video][user])

if model_name == 'true_saliency':
    most_salient_points_per_video = get_most_salient_points_per_video(videos, TRUE_SALIENCY_FOLDER, k=int(args.num_of_peaks))
if model_name == 'content_based_saliency':
    most_salient_points_per_video = get_most_salient_content_based_points_per_video(videos, SALIENCY_FOLDER, k=int(args.num_of_peaks))

def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2*np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)

def generate_arrays(list_IDs, future_window):
    while True:
        encoder_pos_inputs_for_batch = []
        encoder_sal_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        decoder_sal_inputs_for_batch = []
        decoder_outputs_for_batch = []
        count = 0
        np.random.shuffle(list_IDs)
        for ID in list_IDs:
            user = ID['user']
            video = ID['video']
            x_i = ID['time-stamp']
            # Load the data
            if model_name not in ['pos_only', 'MM18']:
                encoder_sal_inputs_for_batch.append(np.expand_dims(all_saliencies[video][x_i-M_WINDOW+1:x_i+1], axis=-1))
                decoder_sal_inputs_for_batch.append(np.expand_dims(all_saliencies[video][x_i+1:x_i+future_window+1], axis=-1))
            if model_name == 'CVPR18_orig':
                encoder_pos_inputs_for_batch.append(all_traces[video][user][x_i-M_WINDOW+1:x_i+1])
                decoder_outputs_for_batch.append(all_traces[video][user][x_i+1:x_i+1+1])
            elif model_name == 'MM18':
                encoder_sal_inputs_for_batch.append(np.concatenate((all_saliencies[video][x_i-M_WINDOW+1:x_i+1], all_headmaps[video][user][x_i-M_WINDOW+1:x_i+1]), axis=1))
                decoder_outputs_for_batch.append(all_headmaps[video][user][x_i+future_window+1])
            else:
                encoder_pos_inputs_for_batch.append(all_traces[video][user][x_i-M_WINDOW:x_i])
                decoder_pos_inputs_for_batch.append(all_traces[video][user][x_i:x_i+1])
                decoder_outputs_for_batch.append(all_traces[video][user][x_i+1:x_i+future_window+1])
            count += 1
            if count == BATCH_SIZE:
                count = 0
                if model_name in ['TRACK', 'TRACK_AblatSal', 'TRACK_AblatFuse']:
                    yield ([np.array(encoder_pos_inputs_for_batch), np.array(encoder_sal_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
                elif model_name == 'CVPR18':
                    yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
                elif model_name == 'pos_only':
                    yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
                elif model_name == 'CVPR18_orig':
                    yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)[:, 0, :, :, 0]], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch)[:, 0])
                elif model_name == 'MM18':
                    yield (np.array(encoder_sal_inputs_for_batch), np.array(decoder_outputs_for_batch))
                encoder_pos_inputs_for_batch = []
                encoder_sal_inputs_for_batch = []
                decoder_pos_inputs_for_batch = []
                decoder_sal_inputs_for_batch = []
                decoder_outputs_for_batch = []
        if count != 0:
            if model_name == 'TRACK':
                yield ([np.array(encoder_pos_inputs_for_batch), np.array(encoder_sal_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
            elif model_name == 'CVPR18':
                yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
            elif model_name == 'pos_only':
                yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
            elif model_name == 'CVPR18_orig':
                yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)[:, 0, :, :, 0]], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch)[:, 0])
            elif model_name == 'MM18':
                yield (np.array(encoder_sal_inputs_for_batch), np.array(decoder_outputs_for_batch))

if model_name == 'TRACK':
    if args.use_true_saliency:
        model = create_TRACK_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
    else:
        model = create_TRACK_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
elif model_name == 'TRACK_AblatSal':
    if args.use_true_saliency:
        model = create_TRACK_AblatSal_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
    else:
        model = create_TRACK_AblatSal_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
elif model_name == 'TRACK_AblatFuse':
    if args.use_true_saliency:
        model = create_TRACK_AblatFuse_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
    else:
        model = create_TRACK_AblatFuse_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
elif model_name == 'CVPR18':
    if args.use_true_saliency:
        model = create_CVPR18_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
    else:
        model = create_CVPR18_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
elif model_name == 'MM18':
    mm18_models = []
    for _h_window in range(H_WINDOW):
        mm18_models.append(MM18_model.create_MM18_model())
elif model_name == 'pos_only':
    model = create_pos_only_model(M_WINDOW, H_WINDOW)
elif model_name == 'CVPR18_orig':
    if args.use_true_saliency:
        model = create_CVPR18_orig_Model(M_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
    else:
        model = create_CVPR18_orig_Model(M_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)


if model_name not in ['no_motion', 'most_salient_point', 'true_saliency', 'content_based_saliency', 'MM18']:
    model.summary()
elif model_name == 'MM18':
    mm18_models[0].summary()

steps_per_ep_train = np.ceil(len(partition['train'])/BATCH_SIZE)
steps_per_ep_validate = np.ceil(len(partition['test'])/BATCH_SIZE)

if TRAIN_MODEL:
    # Create results folder and models folder
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    if model_name == 'MM18':
        for _h_window in range(H_WINDOW):
            csv_logger = keras.callbacks.CSVLogger(RESULTS_FOLDER + '/results_h_window_' + str(_h_window) + '.csv')
            model_checkpoint = keras.callbacks.ModelCheckpoint(MODELS_FOLDER + '/weights_h_window_' + str(_h_window) + '.hdf5', save_best_only=True, save_weights_only=True, mode='auto', period=1)
            mm18_models[_h_window].fit_generator(generator=generate_arrays(partition['train'], future_window=_h_window), verbose=1, steps_per_epoch=steps_per_ep_train, epochs=EPOCHS, callbacks=[csv_logger, model_checkpoint], validation_data=generate_arrays(partition['test'], future_window=_h_window), validation_steps=steps_per_ep_validate)
    else:
        csv_logger = keras.callbacks.CSVLogger(RESULTS_FOLDER + '/results.csv')
        model_checkpoint = keras.callbacks.ModelCheckpoint(MODELS_FOLDER + '/weights.hdf5', save_best_only=True, save_weights_only=True, mode='auto', period=1)
        model.fit_generator(generator=generate_arrays(partition['train'], future_window=H_WINDOW), verbose=1, steps_per_epoch=steps_per_ep_train, epochs=EPOCHS, callbacks=[csv_logger, model_checkpoint], validation_data=generate_arrays(partition['test'], future_window=H_WINDOW), validation_steps=steps_per_ep_validate)
if EVALUATE_MODEL:
    if model_name not in ['no_motion', 'most_salient_point', 'true_saliency', 'content_based_saliency', 'MM18']:
        model.load_weights(MODELS_FOLDER + '/weights.hdf5')
    elif model_name == 'MM18':
        for _h_window in range(H_WINDOW):
            mm18_models[_h_window].load_weights(MODELS_FOLDER + '/weights_h_window_' + str(_h_window) + '.hdf5')

    traces_count = 0
    errors_per_video = {}
    errors_per_timestep = {}
    for ID in partition['test']:
        traces_count += 1
        print('Progress:', traces_count, '/', len(partition['test']))

        user = ID['user']
        video = ID['video']
        x_i = ID['time-stamp']

        if video not in errors_per_video.keys():
            errors_per_video[video] = {}

        # Load the data
        if model_name not in ['pos_only', 'no_motion', 'true_saliency', 'content_based_saliency', 'MM18']:
            encoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i - M_WINDOW + 1:x_i + 1], axis=-1)])
            decoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i + 1:x_i + H_WINDOW + 1], axis=-1)])
        if model_name in ['true_saliency', 'content_based_saliency']:
            decoder_true_sal_inputs_for_sample = most_salient_points_per_video[video][x_i + 1:x_i + H_WINDOW + 1]
        if model_name == 'CVPR18_orig':
            # ToDo when is CVPR18_orig the input is the concatenation of encoder_pos_inputs and decoder_pos_inputs
            encoder_pos_inputs_for_sample = np.array([all_traces[video][user][x_i - M_WINDOW + 1:x_i + 1]])
        elif model_name == 'MM18':
            encoder_sal_inputs_for_sample = np.array([np.concatenate((all_saliencies[video][x_i-M_WINDOW+1:x_i+1], all_headmaps[video][user][x_i-M_WINDOW+1:x_i+1]), axis=1)])
        else:
            encoder_pos_inputs_for_sample = np.array([all_traces[video][user][x_i-M_WINDOW:x_i]])
            decoder_pos_inputs_for_sample = np.array([all_traces[video][user][x_i:x_i + 1]])


        groundtruth = all_traces[video][user][x_i+1:x_i+H_WINDOW+1]

        if model_name in ['TRACK', 'TRACK_AblatSal', 'TRACK_AblatFuse']:
            model_prediction = model.predict([np.array(encoder_pos_inputs_for_sample), np.array(encoder_sal_inputs_for_sample), np.array(decoder_pos_inputs_for_sample), np.array(decoder_sal_inputs_for_sample)])[0]
        elif model_name == 'CVPR18':
            model_prediction = model.predict([np.array(encoder_pos_inputs_for_sample), np.array(decoder_pos_inputs_for_sample), np.array(decoder_sal_inputs_for_sample)])[0]
        elif model_name == 'pos_only':
            model_pred = model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
            model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
        elif model_name == 'no_motion':
            model_prediction = np.repeat(decoder_pos_inputs_for_sample[0], H_WINDOW, axis=0)
        elif model_name == 'most_salient_point':
            model_prediction = np.array([get_max_sal_pos(sal, dataset_name) for sal in decoder_sal_inputs_for_sample[0, :, :, 0]])
        elif model_name == 'true_saliency':
            model_prediction = predict_most_salient_point(decoder_true_sal_inputs_for_sample, decoder_pos_inputs_for_sample[0, 0])
        elif model_name == 'content_based_saliency':
            model_prediction = predict_most_salient_cb_point(decoder_true_sal_inputs_for_sample, decoder_pos_inputs_for_sample[0, 0])
        elif model_name == 'CVPR18_orig':
            initial_pos_inputs = transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample)
            model_pred = auto_regressive_prediction(model, initial_pos_inputs, decoder_sal_inputs_for_sample, M_WINDOW, H_WINDOW)
            model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
        elif model_name == 'MM18':
            model_prediction = []
            groundtruth = []
            for _h_window in range(H_WINDOW):
                model_pred = mm18_models[_h_window].predict(encoder_sal_inputs_for_sample)[0]
                model_pred_norm_eul = MM18_model.model_pred_in_normalized_eulerian(model_pred)
                model_prediction.append(eulerian_to_cartesian(model_pred_norm_eul[0], model_pred_norm_eul[1]))

                groundtruth_eulerian = MM18_model.model_pred_in_normalized_eulerian(all_headmaps[video][user][x_i+_h_window+1])
                groundtruth.append(eulerian_to_cartesian(groundtruth_eulerian[0], groundtruth_eulerian[1]))

        for t in range(len(groundtruth)):
            if t not in errors_per_video[video].keys():
                errors_per_video[video][t] = []
            errors_per_video[video][t].append(metric(groundtruth[t], model_prediction[t]))
            if t not in errors_per_timestep.keys():
                errors_per_timestep[t] = []
            errors_per_timestep[t].append(metric(groundtruth[t], model_prediction[t]))

    for video_name in videos_test:
        for t in range(H_WINDOW):
            print(video_name, t, np.mean(errors_per_video[video_name][t]), end=';')
            # print video_name, t, np.mean(errors_per_video[video_name][t]),';',
        print()
        # print '\n'

    import matplotlib.pyplot as plt
    avg_error_per_timestep = []
    for t in range(H_WINDOW):
        print('Average', t, np.mean(errors_per_timestep[t]), end=';')
        avg_error_per_timestep.append(np.mean(errors_per_timestep[t]))
        # print t, np.mean(errors_per_timestep[t]), ';'
    print()
    plt.plot(np.arange(H_WINDOW)+1*RATE, avg_error_per_timestep, label=model_name)
    met = args.metric
    if args.metric is None:
        met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, dataset_name, model_name))
    plt.ylabel(met)
    plt.xlabel('Prediction step s (sec.)')
    plt.legend()
    plt.show()
    # print '\n'
