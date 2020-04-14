import numpy as np
import cv2
from Quaternion import Quat
from pyquaternion import Quaternion
from sklearn import preprocessing
from keras import models
from scipy import stats
import os

H = 9
W = 16
gblur_size = 5
mmscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / np.pi * 180

def vector_to_ang(_v):
    #v = np.array(vector_ds[0][600][1])
    #v = np.array([0, 0, 1])
    _v = np.array(_v)
    alpha = degree_distance(_v, [0, 1, 0])#degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0] #proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = _v - proj1#proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = degree_distance(proj2, [1, 0, 0])#theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi

def geoy_to_phi(_geoy, _height):
    d = (_height / 2 - _geoy) * 1.0 / (_height / 2)
    s = -1 if d < 0 else 1
    return s * np.arcsin(np.abs(d)) / np.pi * 180

def pixel_to_ang(_x, _y, _geo_h, _geo_w):
    phi = geoy_to_phi(_x, _geo_h)
    theta = -(_y * 1.0 / _geo_w) * 360
    if theta < -180: theta = 360 + theta
    return theta, phi

def extract_direction(_q):
    v = [1, 0, 0]
    return _q.rotate(v)

def create_pixel_vecmap(_geo_h, _geo_w):
    vec_map = np.zeros((_geo_h, _geo_w)).tolist()
    for i in range(_geo_h):
        for j in range(_geo_w):
            theta, phi = pixel_to_ang(i, j, _geo_h, _geo_w)
            t = Quat([0.0, theta, phi]).q  # nolonger use Quat
            q = Quaternion([t[3], t[2], -t[1], t[0]])
            vec_map[i][j] = extract_direction(q)
    return vec_map

def gaussian_from_distance(_d, _gaussian_dict):
    temp = np.around(_d, 1)
    return _gaussian_dict[temp] if temp in _gaussian_dict else 0.0

def create_salient(_fixation_list, _vec_map, _width, _height, _gaussian_dict):
    idx = 0
    heat_map = np.zeros((_height, _width))
    for i in range(heat_map.shape[0]):
        for j in range(heat_map.shape[1]):
            qxy = _vec_map[i][j]
            for fixation in _fixation_list:
                d = degree_distance(fixation, qxy)
                heat_map[i, j] += 1.0 * gaussian_from_distance(d, _gaussian_dict)
            idx += 1
    return heat_map

def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h/2.0 - (_h/2.0) * np.sin(_phi/180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0/360 * _w)
    return int(x), int(y)

def create_fixation_map(_X, _y, _idx):
    v = _y[_idx]
    theta, phi = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H-hi-1, W-wi-1] = 1
    return result

def get_headmaps(head_positions):
    headmap = np.array([create_fixation_map(None, head_positions, idx) for idx, _ in enumerate(head_positions)])
    headmap = np.array([cv2.GaussianBlur(item, (gblur_size, gblur_size), 0) for item in headmap])
    headmap = mmscaler.fit_transform(headmap.ravel().reshape(-1, 1)).reshape(headmap.shape)
    return headmap.reshape(-1, H * W)

def get_true_saliency(saliency_folder, video):
    filename = '%s/%s.npy' % (saliency_folder, video)
    saliency_maps = np.load(filename)
    saliency_maps = mmscaler.fit_transform(saliency_maps.ravel().reshape(-1, 1)).reshape(saliency_maps.shape)
    return saliency_maps.reshape(-1, H*W)

def create_gt_sal(saliency_folder, all_traces):
    # first verify that the folder doesn't exist
    if not os.path.isdir(saliency_folder):
        os.makedirs(saliency_folder)
        var = 20
        # Create Saliency maps each 1/15th of second
        for video in all_traces.keys():
            saliency_per_video = []
            max_trace_len = 0
            for user in all_traces[video].keys():
                trace_len_for_user = all_traces[video][user].shape[0]
                if trace_len_for_user > max_trace_len:
                    max_trace_len = trace_len_for_user
            for x_i in range(max_trace_len):
                fixation_list = []
                for user in all_traces[video].keys():
                    print('creating salmap for video', video, 'x_i', x_i, 'user', user)
                    if len(all_traces[video][user]) >= max_trace_len:
                        orientations = all_traces[video][user][x_i]
                        fixation_list.append(orientations)
                vec_map = create_pixel_vecmap(H, W)
                gaussian_dict = {np.around(_d, 1): stats.multivariate_normal.pdf(_d, mean=0, cov=var) for _d in np.arange(0.0, 180, .1)}
                heat_map = create_salient(fixation_list, vec_map, W, H, gaussian_dict)
                saliency_per_video.append(heat_map)
            saliency_per_video = np.array(saliency_per_video)
            filename = '%s/%s' % (saliency_folder, video)
            np.save(filename, saliency_per_video)
            print('saved file for video %s' % (video))

def create_MM18_model():
    model = models.load_model('./model3_360net_128_w16_h9_8000')
    return model

def model_pred_in_normalized_eulerian(model_prediction):
    highest_salient_pred_pix = np.unravel_index(np.argmax(model_prediction, axis=None), (H, W))
    highest_salient_pred = (highest_salient_pred_pix + np.array([0.5, 0.5])) / np.array([H, W])
    eulerian_pred = (highest_salient_pred[1], highest_salient_pred[0])
    # pred_3dCoord = from_eulerian_to_3D(eulerian_pred)
    return eulerian_pred * np.array([2*np.pi, np.pi])