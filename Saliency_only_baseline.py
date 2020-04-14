import os
import numpy as np
from skimage.feature import peak_local_max
from Utils import eulerian_to_cartesian, compute_orthodromic_distance

NUM_TILES_HEIGHT_TRUE_SAL = 256
NUM_TILES_WIDTH_TRUE_SAL = 256

def get_most_salient_points_per_video(videos, true_saliency_folder, k=1):
    most_salient_points_per_video = {}
    for video in videos:
        saliencies_for_video_file = os.path.join(true_saliency_folder, video+'.npy')
        saliencies_for_video = np.load(saliencies_for_video_file)
        most_salient_points_in_video = []
        for id, sal in enumerate(saliencies_for_video):
            coordinates = peak_local_max(sal, exclude_border=False, num_peaks=k)
            coordinates_normalized = coordinates / np.array([NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL])
            coordinates_radians = coordinates_normalized * np.array([np.pi, 2.0*np.pi])
            cartesian_pts = np.array([eulerian_to_cartesian(sample[1], sample[0]) for sample in coordinates_radians])
            most_salient_points_in_video.append(cartesian_pts)
        most_salient_points_per_video[video] = np.array(most_salient_points_in_video)
    return most_salient_points_per_video

def predict_most_salient_point(most_salient_points, current_point):
    pred_window_predicted_closest_sal_point = []
    for id, most_salient_points_per_fut_frame in enumerate(most_salient_points):
        distances = np.array([compute_orthodromic_distance(current_point, most_sal_pt) for most_sal_pt in most_salient_points_per_fut_frame])
        closest_sal_point = np.argmin(distances)
        predicted_closest_sal_point = most_salient_points_per_fut_frame[closest_sal_point]
        pred_window_predicted_closest_sal_point.append(predicted_closest_sal_point)
    return pred_window_predicted_closest_sal_point

def most_salient_point_baseline(dataset, videos, true_saliency_folder):
    most_salient_points_per_video = get_most_salient_points_per_video(videos, true_saliency_folder)
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