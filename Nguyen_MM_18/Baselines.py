import sys
sys.path.insert(0, './')

import numpy as np
import os
import cv2
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt

from Utils import eulerian_to_cartesian, cartesian_to_eulerian
from TRACK_model import create_TRACK_model
from CVPR18_model import create_CVPR18_model
from position_only_baseline import create_pos_only_model

import pickle


parser = argparse.ArgumentParser(description='Process the input parameters to evaluate the network.')

parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')

args = parser.parse_args()

ROOT_FOLDER = './Nguyen_MM_18'


DATASET_ROOT_FOLDER = os.path.join(ROOT_FOLDER, 'dataset')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# From the paper MM18 About partition into train and test sets
# "We use 5 videos from our dataset for model training and another 4 videos for model validation". For each video, we select one segment with a length of 20-45 seconds.
# The video segment is selected such that there are one or more events in the video that introduce new salient regions (usually when new video scene is shown= and lead to fast
# head movement of users. We extract the timestamped saliency maps and head orientation maps from these videos, generating a total of 300,000 data samples from 432 time series
# using viewing logs of 48 users.
# Before the model training, we normalize the values in saliency maps and head orientation maps to (-1, 1).
import matplotlib.pyplot as plt
VIDEOS_TEST = ['4', '5', '7', '8']
USERS = range(48)

# From the paper MM18 The equirectangular frame is spatially configured into 16x9 tiles.
# Height of the equirectangular frame
H = 9
# Width of the equirectangular frame
W = 16

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216

# 5*0.2 = 1 second of history
M_WINDOW = 5
# 13*0.2 = 2.6 seconds of prediction
H_WINDOW = 13

ORIGINAL_SAMPLING_RATE = 0.063
MODEL_SAMPLING_RATE = 0.2

mmscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

def load_dataset():
    if os.path.isfile(os.path.join(DATASET_ROOT_FOLDER, 'salient_ds_dict_w16_h9')):
        if 'salient_ds_dict' not in locals():
            with open(os.path.join(DATASET_ROOT_FOLDER, 'salient_ds_dict_w16_h9'), 'rb') as file_in:
                u = pickle._Unpickler(file_in)
                u.encoding = 'latin1'
                salient_ds_dict = u.load()
        return salient_ds_dict
    else:
        raise Exception('Sorry, the file ./Nguyen_MM_18/dataset/salient_ds_dict_w16_h9 doesn\'t exist.\nYou can:\n* Download the file from:\n\thttps://www.dropbox.com/s/r09s1yqr2qyluyr/salient_ds.zip?dl=0')

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

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h / 2.0 - (_h / 2.0) * np.sin(_phi / 180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0 / 360 * _w)
    return int(x), int(y)

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def create_fixation_map(v, cartesian):
    if cartesian:
        theta, phi = vector_to_ang(v)
    else:
        theta = v[0]
        phi = v[1]
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H - hi - 1, W - wi - 1] = 1
    return result

# ToDo: inspired from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py lines 67, 68
# And from the paper MM18:
# "We generate the head orientation map by first identifiying the tile pointed by current head orientation vector and set its likelihood to be viewed as 1.0.
# Using the tile as the center, we then apply a Gaussian kernel to gradually select other tiles with a lower likelihood to be viewed around the center tile until the selected tiles can cover a viewport.
### There is a better way to perform the computation of the head-pos probability map (Check function from_position_to_tile_probability_cartesian in Reading_Dataset.py), but we wanted to stick
### to the way it was computed on their experiments, the results with either of the methods are similar
gblur_size = 5
def create_head_map(v, cartesian=True):
    headmap = create_fixation_map(v, cartesian)
    headmap = cv2.GaussianBlur(headmap, (gblur_size, gblur_size), 0)
    # To binarize the headmap
    headmap = np.ceil(headmap)
    return headmap

salient_ds_dict = load_dataset()


# From the paper MM18: About evaluation metric:
# "The evaluation metric is the accuracy of head movement prediction. Accuracy is calculated based on the ratio of the number of overlapping
# tiles between predicted and ground truth head orientation map over the total number of predicted and viewed tiles."
# Note that the validation set used in this evaluation was never used to train the LSTM model.
def compute_accuracy_metric(binary_pred, binary_true):
    binary_pred = binary_pred.reshape(-1)
    binary_true = binary_true.reshape(-1)
    sum_of_binary = binary_true + binary_pred
    Intersection = np.sum(sum_of_binary == 2)
    Union = np.sum(sum_of_binary>0)
    return Intersection / np.float(Union)

## To keep a backup
# def compute_error(model_name):
#     # From the paper MM18:
#     # We use the input feature from the past one second to predict the head orientation in the future.
#     initial_one_second = int(np.ceil(1.0/ORIGINAL_SAMPLING_RATE))
#     # From the paper MM18:
#     # "The default prediction window k is set to be 0.5 seconds.
#     # To explore the effect of prediction window k on the accuracy of the proposed model and other three benchmarks, we vary k from 0.5 seconds to 2.5 seconds."
#     prediction_horizons = [0.5, 1, 1.5, 2, 2.5]
#     results_for_pred_horizon = {}
#     for pred_hor in prediction_horizons:
#         results_for_pred_horizon[pred_hor] = []
#         prediction_horizon_in_timesteps = int(np.ceil(pred_hor/ORIGINAL_SAMPLING_RATE))
#         for enum_video, video in enumerate(VIDEOS_TEST):
#             for user in USERS:
#                 print('computing no-motion baseline error for', enum_video, '/', len(VIDEOS_TEST), 'user', user, '/', len(USERS), 'prediction_horizon', pred_hor)
#                 trace = salient_ds_dict['360net'][video]['headpos'][user]
#                 for t in range(initial_one_second, len(trace)-prediction_horizon_in_timesteps):
#                     if model_name == 'no_motion':
#                         model_pred = create_head_map(trace[t])
#                     groundtruth = create_head_map(trace[t+prediction_horizon_in_timesteps])
#                     results_for_pred_horizon[pred_hor].append(compute_accuracy_metric(model_pred, groundtruth))
#     return results_for_pred_horizon

# ToDo Copied exactly from Reading_Dataset, you can just import it
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

# ToDo Copied exactly from Extract_Saliency/panosalnet (Author: Miguel Romero)
def post_filter(_img):
    result = np.copy(_img)
    result[:3, :3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result

# ToDo Copied exactly from Reading_Dataset, you can just import it
# Performs the opposite transformation than transform_the_degrees_in_range
def transform_the_radians_to_original(yaw, pitch):
    yaw = ((yaw/(2*np.pi))*360.0)-180.0
    pitch = ((pitch/np.pi)*180.0)-90.0
    return yaw, pitch


def get_TRACK_prediction(model, trace, saliency, pred_hor):
    indices_input_trace = np.linspace(0, len(trace)-1, M_WINDOW+1, dtype=int)
    indices_input_saliency = np.linspace(0, len(saliency) - 1, M_WINDOW+H_WINDOW, dtype=int)

    subsampled_trace = trace[indices_input_trace]

    subsampled_saliency = saliency[indices_input_saliency]

    encoder_pos_inputs = subsampled_trace[-M_WINDOW-1:-1]
    decoder_pos_inputs = subsampled_trace[-1:]
    encoder_sal_inputs = subsampled_saliency[:M_WINDOW]
    decoder_sal_inputs = subsampled_saliency[M_WINDOW:]

    model_prediction = model.predict(
        [np.array([encoder_pos_inputs]), np.array([encoder_sal_inputs]),
         np.array([decoder_pos_inputs]), np.array([decoder_sal_inputs])])[0][int(pred_hor/MODEL_SAMPLING_RATE)]

    yaw_pred, pitch_pred = cartesian_to_eulerian(model_prediction[0], model_prediction[1], model_prediction[2])
    yaw_pred, pitch_pred = transform_the_radians_to_original(yaw_pred, pitch_pred)
    return create_head_map(np.array([yaw_pred, pitch_pred]), cartesian=False)


def get_CVPR18_prediction(model, trace, saliency, pred_hor):
    indices_input_trace = np.linspace(0, len(trace)-1, M_WINDOW+1, dtype=int)
    indices_input_saliency = np.linspace(0, len(saliency) - 1, M_WINDOW+H_WINDOW, dtype=int)

    subsampled_trace = trace[indices_input_trace]

    subsampled_saliency = saliency[indices_input_saliency]

    encoder_pos_inputs = subsampled_trace[-M_WINDOW-1:-1]
    decoder_pos_inputs = subsampled_trace[-1:]
    # encoder_sal_inputs = subsampled_saliency[:M_WINDOW]
    decoder_sal_inputs = subsampled_saliency[M_WINDOW:]

    model_prediction = model.predict([np.array([encoder_pos_inputs]), np.array([decoder_pos_inputs]), np.array([decoder_sal_inputs])])[0][int(pred_hor/MODEL_SAMPLING_RATE)]

    yaw_pred, pitch_pred = cartesian_to_eulerian(model_prediction[0], model_prediction[1], model_prediction[2])
    yaw_pred, pitch_pred = transform_the_radians_to_original(yaw_pred, pitch_pred)
    return create_head_map(np.array([yaw_pred, pitch_pred]), cartesian=False)

# ToDo import from training_procedure (for now we just copied the function)
# from training_procedure import transform_batches_cartesian_to_normalized_eulerian
def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(position):
    position = position * np.array([2*np.pi, np.pi])
    eulerian_samples = eulerian_to_cartesian(position[0], position[1])
    return np.array(eulerian_samples)

def get_pos_only_prediction(model, trace, pred_hor):
    indices_input_trace = np.linspace(0, len(trace) - 1, M_WINDOW + 1, dtype=int)

    subsampled_trace = trace[indices_input_trace]

    encoder_pos_inputs = subsampled_trace[-M_WINDOW - 1:-1]
    decoder_pos_inputs = subsampled_trace[-1:]

    model_prediction = model.predict([np.array(transform_batches_cartesian_to_normalized_eulerian([encoder_pos_inputs])), np.array(transform_batches_cartesian_to_normalized_eulerian([decoder_pos_inputs]))])[0][int(pred_hor / MODEL_SAMPLING_RATE)]
    model_prediction = transform_normalized_eulerian_to_cartesian(model_prediction)

    yaw_pred, pitch_pred = cartesian_to_eulerian(model_prediction[0], model_prediction[1], model_prediction[2])
    yaw_pred, pitch_pred = transform_the_radians_to_original(yaw_pred, pitch_pred)
    return create_head_map(np.array([yaw_pred, pitch_pred]), cartesian=False)

def compute_error(model_name):
    if model_name == 'CVPR18':
        model = create_CVPR18_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
        model.load_weights(os.path.join(ROOT_FOLDER, 'CVPR18', 'Models_EncDec_3DCoords_ContSal_init_5_in_5_out_13_end_13', 'weights.hdf5'))
    if model_name == 'TRACK':
        model = create_TRACK_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)

        weights_file = os.path.join(ROOT_FOLDER, 'TRACK', 'Models_EncDec_3DCoords_ContSal_init_5_in_5_out_13_end_13', 'weights.hdf5')
        if os.path.isfile(weights_file):
            model.load_weights(weights_file)
        else:
            raise Exception('Sorry, the folder ./Nguyen_MM_18/TRACK/ doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"python training_procedure.py -train -gpu_id 0 -dataset_name Nguyen_MM_18 -model_name TRACK -m_window 5 -h_window 13 -provided_videos\" or \n* Download the file from:\n\thttps://unice-my.sharepoint.com/:u:/g/personal/miguel_romero-rondon_unice_fr/EYNvRsxKh1FCiJrhudfBMUsBhp1oB5m3fxTYa8kkZHOcSA?e=eC2Plz')
    if model_name == 'pos_only':
        model = create_pos_only_model(M_WINDOW, H_WINDOW)

        weights_file = os.path.join(ROOT_FOLDER, 'pos_only', 'Models_EncDec_eulerian_init_5_in_5_out_13_end_13', 'weights.hdf5')
        if os.path.isfile(weights_file):
            model.load_weights(weights_file)
        else:
            raise Exception('Sorry, the folder ./Nguyen_MM_18/pos_only/ doesn\'t exist or is incomplete.\nYou can:\n* Create it using the command:\n\t\"python training_procedure.py -train -gpu_id 0 -dataset_name Nguyen_MM_18 -model_name pos_only -m_window 5 -h_window 13 -provided_videos\" or \n* Download the file from:\n\thttps://unice-my.sharepoint.com/:u:/g/personal/miguel_romero-rondon_unice_fr/EWO4VEQP2GtMp6NEZBMZA-QBpuXFo6WG2jQb-muvPc_ejw?e=iaPbYp')

    # From the paper MM18:
    # We use the input feature from the past one second to predict the head orientation in the future.
    one_second_in_timesteps = int(np.ceil(1.0/ORIGINAL_SAMPLING_RATE))
    one_timestep_models = MODEL_SAMPLING_RATE / ORIGINAL_SAMPLING_RATE
    # From the paper MM18:
    # "The default prediction window k is set to be 0.5 seconds.
    # To explore the effect of prediction window k on the accuracy of the proposed model and other three benchmarks, we vary k from 0.5 seconds to 2.5 seconds."
    prediction_horizons = [0.5, 1, 1.5, 2, 2.5]
    results_for_pred_horizon = {}
    for pred_hor in prediction_horizons:
        results_for_pred_horizon[pred_hor] = []
        prediction_horizon_in_timesteps = int(np.ceil(pred_hor/ORIGINAL_SAMPLING_RATE))
        for enum_video, video in enumerate(VIDEOS_TEST):
            saliency = salient_ds_dict['360net'][video]['salient']
            # preprocess saliency
            saliency = np.array([cv2.resize(sal, (NUM_TILES_WIDTH, NUM_TILES_HEIGHT)) for sal in saliency])
            saliency = np.array([(sal * 1.0 - sal.min()) for sal in saliency])
            saliency = np.array([(sal / sal.max()) * 255 for sal in saliency])
            saliency = np.array([post_filter(sal) for sal in saliency])
            saliency = np.array([mmscaler.fit_transform(salmap.ravel().reshape(-1, 1)).reshape(salmap.shape) for salmap in saliency])
            saliency = np.expand_dims(saliency, -1)
            for user in USERS:
                print('computing', args.model_name, 'baseline error for video', enum_video, '/', len(VIDEOS_TEST), 'user', user, '/', len(USERS), 'prediction_horizon', pred_hor)
                trace = salient_ds_dict['360net'][video]['headpos'][user]

                trace_for_model = np.array([vector_to_ang(point) for point in trace])
                trace_for_model = np.array([transform_the_degrees_in_range(sample[0], sample[1]) for sample in trace_for_model])
                trace_for_model = np.array([eulerian_to_cartesian(sample[0], sample[1]) for sample in trace_for_model])
                for t in range(one_second_in_timesteps, len(trace)-prediction_horizon_in_timesteps):
                    if model_name == 'no_motion':
                        model_pred = create_head_map(trace[t])
                    if model_name == 'TRACK':
                        pos_input = trace_for_model[t-one_second_in_timesteps:t]
                        saliency_input = saliency[t-int(np.ceil((M_WINDOW-1)*one_timestep_models)):t+int(np.ceil(one_timestep_models*(H_WINDOW+1)))]
                        model_pred = get_TRACK_prediction(model, pos_input, saliency_input, pred_hor)
                    if model_name == 'CVPR18':
                        pos_input = trace_for_model[t - one_second_in_timesteps:t]
                        saliency_input = saliency[t - int(np.ceil((M_WINDOW - 1) * one_timestep_models)):t + int(np.ceil(one_timestep_models * (H_WINDOW + 1)))]
                        model_pred = get_CVPR18_prediction(model, pos_input, saliency_input, pred_hor)
                    if model_name == 'pos_only':
                        pos_input = trace_for_model[t - one_second_in_timesteps:t]
                        model_pred = get_pos_only_prediction(model, pos_input, pred_hor)
                    groundtruth = create_head_map(trace[t+prediction_horizon_in_timesteps])
                    results_for_pred_horizon[pred_hor].append(compute_accuracy_metric(model_pred, groundtruth))
        print(pred_hor, np.mean(results_for_pred_horizon[pred_hor]))
    return results_for_pred_horizon

if __name__ == "__main__":
    # results_for_pred_horizon = compute_error('no_motion')
    # for pred_hor in results_for_pred_horizon.keys():
    #     print(pred_hor, np.mean(results_for_pred_horizon[pred_hor]))

    results_for_pred_horizon = compute_error(args.model_name)
    pred_horizons = []
    results_per_horizon = []
    print('\n\n', args.model_name, 'results:')
    for pred_hor in results_for_pred_horizon.keys():
        average_for_horizon = np.mean(results_for_pred_horizon[pred_hor])
        pred_horizons.append(pred_hor)
        results_per_horizon.append(average_for_horizon)
        print(pred_hor, average_for_horizon)

    plt.plot([0.5, 1, 1.5, 2, 2.5], [0.76, 0.6, 0.57, 0.52, 0.44], 'r*--', label='MM18 reported')
    plt.plot([0.5, 1, 1.5, 2, 2.5], [0.7916673712436443, 0.6686546370264778, 0.5939437746136321, 0.53875768596792, 0.4999404410546121], 'gs--', label='Pos-only reported')
    plt.plot([0.5, 1, 1.5, 2, 2.5], [0.8138432634030537, 0.7004443996979024, 0.6257524854330238, 0.572281867283657, 0.534381004066448], 'cD--', label='No-motion reported')
    plt.plot([0.5, 1, 1.5, 2, 2.5], [0.8126569832, 0.6919552981, 0.6165277754, 0.5578093616, 0.5152421880746652], 'mo--', label='TRACK reported')
    plt.plot(pred_horizons, results_per_horizon, 'b', label=args.model_name + ' obtained')
    plt.xlabel('Prediction step (sec.)')
    plt.ylabel('Score (Intersection over union)')
    plt.legend()
    plt.show()
