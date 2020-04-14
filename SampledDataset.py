import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

# returns the whole data organized as follows:
# time-stamp, x, y, z (in 3d coordinates)
def read_sampled_data_for_trace(sampled_dataset_folder, video, user):
    path = os.path.join(sampled_dataset_folder, video, user)
    data = pd.read_csv(path, header=None)
    return data.values

# returns only the positions from the trace
# ~time-stamp~ is removed from the output, only x, y, z (in 3d coordinates) is returned
def read_sampled_positions_for_trace(sampled_dataset_folder, video, user):
    path = os.path.join(sampled_dataset_folder, video, user)
    data = pd.read_csv(path, header=None)
    return data.values[:, 1:]

# Returns the ids of the videos in the dataset
def get_video_ids(sampled_dataset_folder):
    list_of_videos = [o for o in os.listdir(sampled_dataset_folder) if not o.endswith('.gitkeep')]
    # Sort to avoid randomness of keys(), to guarantee reproducibility
    list_of_videos.sort()
    return list_of_videos

# returns the unique ids of the users in the dataset
def get_user_ids(sampled_dataset_folder):
    videos = get_video_ids(sampled_dataset_folder)
    users = []
    for video in videos:
        for user in [o for o in os.listdir(os.path.join(sampled_dataset_folder, video)) if not o.endswith('.gitkeep')]:
            users.append(user)
    list_of_users = list(set(users))
    # Sort to avoid randomness of keys(), to guarantee reproducibility
    list_of_users.sort()
    return list_of_users

# Returns a dictionary indexed by video, and under each index you can find the users for which the trace has been stored for this video
def get_users_per_video(sampled_dataset_folder):
    videos = get_video_ids(sampled_dataset_folder)
    users_per_video = {}
    for video in videos:
        users_per_video[video] = [user for user in os.listdir(os.path.join(sampled_dataset_folder, video))]
    return users_per_video

# divides a list into two sublists with the first sublist having samples proportional to "percentage"
def split_list_by_percentage(the_list, percentage):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Shuffle to select randomly
    np.random.shuffle(the_list)
    num_samples_first_part = int(len(the_list) * percentage)
    train_part = the_list[:num_samples_first_part]
    test_part = the_list[num_samples_first_part:]
    return train_part, test_part

# returns a dictionary partition with two indices:
# partition['train'] and partition['test']
# partition['train'] will contain randomly perc_videos_train percent of videos and perc_users_train from each video
# partition['test'] the remaining samples
# the sample consists on a structure of {'video':video_id, 'user':user_id, 'time-stamp':time-stamp_id}
# In this case we don't have any intersection between users nor between videos in train and test sets
# init_window and end_window are used to crop the initial and final indices of the sequence
# e.g. if the indices are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and init_window = 3, end_window=2
# the resulting indices of the sequence will be: [3, 4, 5, 6, 7, 8]
# ToDo perform a better split taking into account that a video may be watched by different amounts of users
def partition_in_train_and_test_without_any_intersection(sampled_dataset_folder, init_window, end_window, videos_train, videos_test, users_train, users_test):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    for video in videos_train:
        for user in users_train:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(init_window, trace_length-end_window):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['train'].append(ID)
    for video in videos_test:
        for user in users_test:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(init_window, trace_length - end_window):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['test'].append(ID)
    return partition

# returns a dictionary partition with two indices:
# partition['train'] and partition['test']
# partition['train'] will contain randomly perc_videos_train percent of videos and perc_users_train from each video
# partition['test'] the remaining samples
# the sample consists on a structure of {'video':video_id, 'user':user_id, 'time-stamp':time-stamp_id}
# In this case the partition is performed only by videos
def partition_in_train_and_test_without_video_intersection(sampled_dataset_folder, init_window, end_window, videos_train, videos_test, users_per_video):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    for video in videos_train:
        for user in users_per_video[video]:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(init_window, trace_length-end_window):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['train'].append(ID)
    for video in videos_test:
        for user in users_per_video[video]:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(init_window, trace_length - end_window):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['test'].append(ID)
    return partition


# returns a dictionary partition with two indices:
# partition['train'] and partition['test']
# partition['train'] will contain the traces in train_traces
# partition['test'] the traces in test_traces
# the sample consists on a structure of {'video':video_id, 'user':user_id, 'time-stamp':time-stamp_id}
# In this case the samples in train and test may belong to the same user or (exclusive or) the same video
def partition_in_train_and_test(sampled_dataset_folder, init_window, end_window, train_traces, test_traces):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    for trace in train_traces:
        user = str(trace[0])
        video = trace[1]
        # to get the length of the trace
        trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
        for tstap in range(init_window, trace_length - end_window):
            ID = {'video': video, 'user': user, 'time-stamp': tstap}
            partition['train'].append(ID)
    for trace in test_traces:
        user = str(trace[0])
        video = trace[1]
        # to get the length of the trace
        trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
        for tstap in range(init_window, trace_length - end_window):
            ID = {'video': video, 'user': user, 'time-stamp': tstap}
            partition['test'].append(ID)
    return partition

# load the saliency maps for a "video" normalized between -1 and 1
# RUN_IN_SERVER is a flag used to load the file in a different manner if is stored in the server.
# ToDo check that the saliency and the traces are sampled at the same rate, for now we assume the saliency is sampled manually when running the scripts to create the scaled_images before extracting the saliency in '/home/twipsy/PycharmProjects/UniformHeadMotionDataset/AVTrack360/dataset/videos/cut_wogrey/creation_of_scaled_images'
def load_saliency(saliency_folder, video):
    mmscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    saliency_list = []
    with open(('%s/%s/%s' % (saliency_folder, video, video)), 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        for frame_id in range(1, len(p.keys())+1):
            salmap = p['%03d' % frame_id]
            salmap_norm = mmscaler.fit_transform(salmap.ravel().reshape(-1, 1)).reshape(salmap.shape)
            saliency_list.append(salmap_norm)
    return np.array(saliency_list)

def load_true_saliency(saliency_folder, video):
    saliencies_for_video_file = os.path.join(saliency_folder, video + '.npy')
    saliencies_for_video = np.load(saliencies_for_video_file)
    return saliencies_for_video
