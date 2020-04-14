import os
import numpy as np
from skimage.feature import peak_local_max
from Utils import eulerian_to_cartesian, compute_orthodromic_distance
from SampledDataset import load_saliency

# ToDo merge with Saliency_only_baseline.py

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216

def get_most_salient_content_based_points_per_video(videos, saliency_folder, k=1):
    most_salient_points_per_video = {}
    for video in videos:
        saliencies_for_video = load_saliency(saliency_folder, video, RUN_IN_SERVER=False)
        most_salient_points_in_video = []
        for id, sal in enumerate(saliencies_for_video):
            coordinates = peak_local_max(sal, exclude_border=False, num_peaks=k)
            coordinates_normalized = coordinates / np.array([NUM_TILES_HEIGHT, NUM_TILES_WIDTH])
            coordinates_radians = coordinates_normalized * np.array([np.pi, 2.0*np.pi])
            cartesian_pts = np.array([eulerian_to_cartesian(sample[1], sample[0]) for sample in coordinates_radians])
            most_salient_points_in_video.append(cartesian_pts)
        most_salient_points_per_video[video] = np.array(most_salient_points_in_video)
    return most_salient_points_per_video

def predict_most_salient_cb_point(most_salient_points, current_point):
    pred_window_predicted_closest_sal_point = []
    for id, most_salient_points_per_fut_frame in enumerate(most_salient_points):
        distances = np.array([compute_orthodromic_distance(current_point, most_sal_pt) for most_sal_pt in most_salient_points_per_fut_frame])
        closest_sal_point = np.argmin(distances)
        predicted_closest_sal_point = most_salient_points_per_fut_frame[closest_sal_point]
        pred_window_predicted_closest_sal_point.append(predicted_closest_sal_point)
    return pred_window_predicted_closest_sal_point