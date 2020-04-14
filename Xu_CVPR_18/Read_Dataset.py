import sys
sys.path.insert(0, './')

import os
import numpy as np
import pandas as pd
from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SampledDataset import get_video_ids, get_user_ids, get_users_per_video
import cv2
import argparse

ROOT_FOLDER = './Xu_CVPR_18/dataset/'
GAZE_TXT_FOLDER = 'Gaze_txt_files'
# GAZE_TXT_FOLDER = 'Selected_Test_Gaze_txt_files'
OUTPUT_GAZE_FOLDER = './Xu_CVPR_18/sampled_dataset_replica'
OUTPUT_FOLDER = './Xu_CVPR_18/sampled_dataset'
TRAIN_TEST_SET_FILE = 'dataset/train_test_set.xlsx'
# From the paper in http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Gaze_Prediction_in_CVPR_2018_paper.pdf
# "the interval between two neighboring frames in our experiments corresponds to 0.2 seconds"
SAMPLING_RATE = 0.2

OUTPUT_TRUE_SALIENCY_FOLDER = './Xu_CVPR_18/true_saliency'
NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

# train_videos = ['A380', 'AcerEngine', 'AcerPredator', 'AirShow', 'BFG', 'Bicycle', 'Camping', 'CandyCarnival', 'Castle', 'Catwalks', 'CMLauncher', 'CS', 'DanceInTurn', 'DrivingInAlps', 'Egypt', 'F5Fighter', 'Flight', 'GalaxyOnFire', 'Graffiti', 'GTA', 'HondaF1', 'IRobot', 'KasabianLive', 'Lion', 'LoopUniverse', 'Manhattan', 'MC', 'MercedesBenz', 'Motorbike', 'Murder', 'NotBeAloneTonight', 'Orion', 'Parachuting', 'Parasailing', 'Pearl', 'Predator', 'ProjectSoul', 'Rally', 'RingMan', 'Roma', 'Shark', 'Skiing', 'Snowfield', 'SnowRopeway', 'Square', 'StarWars', 'StarWars2', 'Stratosphere', 'StreetFighter', 'Supercar', 'SuperMario64', 'Surfing', 'SurfingArctic', 'TalkingInCar', 'Terminator', 'TheInvisible', 'Village', 'VRBasketball', 'Waterskiing', 'WesternSichuan', 'Yacht']
# test_videos = ['KingKong', 'SpaceWar2', 'StarryPolar', 'Dancing', 'Guitar', 'BTSRun', 'InsideCar', 'RioOlympics', 'SpaceWar', 'CMLauncher2', 'Waterfall', 'Sunset', 'BlueWorld', 'Symphony', 'WaitingForLove']
orig_users = ['chenmeiling_w1_23', 'CRYL_m1', 'diaopengfei_m1', 'fanchao_m1_22', 'fangyizhong_m1', 'fanshiyang_m1_23', 'fengyuting_w1', 'gaoweiqing_m1_25', 'gaoxi_w1', 'gaoyuan_m1', 'guozhanpeng_m1_24', 'haodongdong_m1_23', 'hewenjing_w1', 'huangweihan_m1', 'huruitao_m1', 'lande_m1', 'liangyankuan_m1_24', 'liantianye_m1', 'lichen_m', 'lijing_m1', 'liliu_w1_22', 'linxin_m1', 'linzhixing_m1', 'lisai_w1', 'liushijie_m1', 'liutong_m1', 'liwenli_w1', 'lucan_m1', 'lujiaxin_w1', 'luyunpeng_m1_21', 'mafu_m11', 'mashang_m1', 'ouliyang_w1', 'pengshuxue_m1', 'qiuyao_w1', 'renjie_m1', 'renzan_m1', 'shaowei_m1_23', 'shixiaonan_m1_20', 'subowen_m1', 'tianmiaomiao_w1', 'wangrui_m1', 'weiwu_m1', 'wuguanqun_m1', 'xujingyao_w1', 'xusanjia_m1', 'yangpengshuai_m1', 'yangren_m1', 'yangyandan_w1_21', 'yinjiayuan_m1', 'yumengyue_w1_24', 'yuwenhai_m1', 'zhangwenjing_w1', 'zhaosiyu_m1', 'zhaoxinyue_w1', 'zhaoyilin_w1', 'zhongyueyou_m1', 'zhuxudong_m1']

fps_per_video = {'001': 0.04, '002': 0.03999998263888889, '003': 0.033366657890117606, '004': 0.03336662470627727, '005': 0.03333333333333333, '006': 0.03336666666666667, '007': 0.041708247630570676, '008': 0.03330890516890517, '009': 0.033366654097536454, '010': 0.03999998036906164, '011': 0.03336666666666667, '012': 0.041711111928495234, '013': 0.03336666666666667, '014': 0.033366652777777776, '015': 0.03333188657407407, '016': 0.03333333333333333, '017': 0.04, '018': 0.0399999838501292, '019': 0.041711111928495234, '020': 0.03336665534035564, '021': 0.03994386785102073, '022': 0.041708318865740744, '023': 0.03999998759920635, '024': 0.03333332638888889, '025': 0.03331449652777778, '026': 0.03336666666666667, '027': 0.03996762777328663, '028': 0.039999989149305554, '029': 0.03336665219907407, '030': 0.039999985532407405, '031': 0.03336666666666667, '032': 0.016683325004164584, '033': 0.03999998759920635, '034': 0.03999998255712541, '035': 0.03336666666666667, '037': 0.033366634783994896, '038': 0.03336666666666667, '039': 0.03336666666666667, '040': 0.03336666666666667, '041': 0.03333332561728395, '042': 0.041666666666666664, '043': 0.033347238390893805, '044': 0.03334347993827161, '045': 0.03999998761303109, '046': 0.03336666666666667, '047': 0.0399999838501292, '048': 0.03999997040982394, '049': 0.03336666666666667, '050': 0.016683337817236122, '051': 0.0399999881291548, '052': 0.03336666666666667, '053': 0.03333333333333333, '054': 0.033366652777777776, '055': 0.04, '056': 0.04, '057': 0.033397896361273556, '058': 0.033366652777777776, '059': 0.03336666666666667, '060': 0.0333210686095932, '062': 0.033366657529239764, '063': 0.03333333333333333, '064': 0.03333333333333333, '065': 0.03336664930555555, '066': 0.039999985532407405, '067': 0.03336666666666667, '068': 0.03336665798611111, '069': 0.03336664930555555, '070': 0.03336664930555555, '071': 0.03336666666666667, '072': 0.03999999035493827, '073': 0.03999998387356878, '074': 0.03999999060768292, '075': 0.03336666666666667, '076': 0.033366662082874955, '077': 0.041666647802301456, '078': 0.039973797569671836, '079': 0.03328899231678487, '080': 0.03336665679889481, '081': 0.03339789603960396, '082': 0.039978076103500765, '083': 0.04170831926151075, '085': 0.03333333333333333, '087': 0.03333333333333333, '088': 0.03336664682539683, '089': 0.03336666666666667, '090': 0.03336666666666667, '091': 0.03333333333333333, '092': 0.03999999210858586, '093': 0.03336666666666667, '094': 0.03336666666666667, '095': 0.03336666666666667, '096': 0.03336664988251091, '097': 0.03336666666666667, '098': 0.03336666666666667, '099': 0.03336666666666667, '100': 0.03336666666666667, '101': 0.03333333333333333, '102': 0.04, '103': 0.03336666666666667, '104': 0.03336666666666667, '105': 0.03334730600670276, '106': 0.033366650326797385, '109': 0.04170832798116035, '110': 0.03333333333333333, '111': 0.03333333333333333, '112': 0.04170833333333333, '113': 0.03336666666666667, '114': 0.03336666666666667, '115': 0.04166379698802324, '116': 0.04, '117': 0.03333333333333333, '118': 0.033338138651471984, '119': 0.03336666666666667, '120': 0.04, '121': 0.03333333333333333, '122': 0.03333333333333333, '123': 0.03333333333333333, '124': 0.03336666666666667, '125': 0.03333333333333333, '126': 0.03336666666666667, '127': 0.016666666666666666, '128': 0.03333333333333333, '129': 0.03333333333333333, '130': 0.04, '131': 0.03336590157154673, '132': 0.03336666666666667, '133': 0.033338138651471984, '134': 0.03336666666666667, '135': 0.04, '136': 0.041622234067756454, '137': 0.04, '138': 0.03336666666666667, '139': 0.03333333333333333, '140': 0.033293778832212996, '141': 0.03336666666666667, '142': 0.04, '143': 0.03333532603025561, '144': 0.03336666666666667, '145': 0.04, '146': 0.03333333333333333, '147': 0.03336666666666667, '148': 0.03332834545222605, '149': 0.03336666666666667, '150': 0.03336632352941177, '151': 0.03336666666666667, '152': 0.03336666666666667, '153': 0.03336666666666667, '154': 0.03336666666666667, '155': 0.03327503885003885, '156': 0.03333333333333333, '157': 0.03336666666666667, '158': 0.03333333333333333, '159': 0.04170833333333333, '160': 0.03333333333333333, '161': 0.033366158536585366, '162': 0.04170833333333333, '163': 0.03333333333333333, '164': 0.03336666666666667, '165': 0.03336666666666667, '166': 0.03336666666666667, '167': 0.04, '168': 0.033320372938819544, '169': 0.03333333333333333, '170': 0.033320372938819544, '171': 0.03333333333333333, '172': 0.03333333333333333, '173': 0.03336666666666667, '174': 0.03333333333333333, '175': 0.03336666666666667, '176': 0.03996866788479944, '177': 0.03333333333333333, '178': 0.04, '179': 0.03336666666666667, '180': 0.04, '181': 0.041666666666666664, '182': 0.04170832765280618, '183': 0.04170833333333333, '184': 0.03336666666666667, '185': 0.03336666666666667, '186': 0.03336666666666667, '187': 0.03333333333333333, '188': 0.04, '189': 0.04, '190': 0.03336666666666667, '191': 0.04, '192': 0.03336666666666667, '193': 0.03333333333333333, '194': 0.04, '195': 0.03333333333333333, '196': 0.03336666666666667, '197': 0.041708328364142316, '198': 0.03336666666666667, '199': 0.04, '200': 0.039953038601982266, '201': 0.04, '202': 0.03336666666666667, '203': 0.03336666666666667, '204': 0.03334138047138047, '205': 0.04, '206': 0.04, '208': 0.03336666666666667, '209': 0.03336666666666667, '210': 0.04, '211': 0.03332834545222605, '212': 0.04, '213': 0.03336666666666667, '214': 0.03336666666666667, '215': 0.03333333333333333}

def get_gaze_positions_for_trace(filename):
    dataframe = pd.read_csv(filename, header=None, sep=",", engine='python')
    data = dataframe[[6, 7]]
    return data.values

def get_head_positions_for_trace(filename):
    dataframe = pd.read_csv(filename, header=None, sep=",", engine='python')
    data = dataframe[[3, 4]]
    return data.values

def get_indices_for_trace(filename):
    dataframe = pd.read_csv(filename, header=None, sep=",", engine='python')
    data = dataframe[1]
    return data.values


# returns the frame rate of a video using openCV
# ToDo Copied exactly from Xu_PAMI_18/Reading_Dataset (Author: Miguel Romero)
def get_frame_rate(videoname, use_dict=False):
    if use_dict:
        return 1.0/fps_per_video[videoname]
    video_mp4 = videoname+'.mp4'
    video_path = os.path.join(ROOT_FOLDER, 'Videos', video_mp4)
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

# returns the number of frames of a video using openCV
def get_frame_count(videoname):
    video_mp4 = videoname+'.mp4'
    video_path = os.path.join(ROOT_FOLDER, 'Videos', video_mp4)
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        count = video.get(cv2.cv.CAP_PROP_FRAME_COUNT)
    else:
        count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    return count

# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset():
    dataset = {}
    for root, directories, files in os.walk(os.path.join(ROOT_FOLDER, GAZE_TXT_FOLDER)):
        for enum_user, user in enumerate(directories):
            dataset[user] = {}
            for r_2, d_2, sub_files in os.walk(os.path.join(ROOT_FOLDER, GAZE_TXT_FOLDER, user)):
                for enum_video, video_txt in enumerate(sub_files):
                    video = video_txt.split('.')[0]
                    print('get head positions from original dataset', 'user', enum_user, '/', len(directories), 'video', enum_video, '/', len(sub_files))
                    filename = os.path.join(ROOT_FOLDER, GAZE_TXT_FOLDER, user, video_txt)
                    positions = get_head_positions_for_trace(filename)
                    frame_ids = get_indices_for_trace(filename)
                    video_rate = 1.0 / get_frame_rate(video, use_dict=True)
                    samples = []
                    for pos, frame_id in zip(positions, frame_ids):
                        # ToDo Check if head position x corresponds to yaw and gaze position y corresponds to pitch
                        samples.append({'sec': frame_id*video_rate, 'yaw': pos[0], 'pitch': pos[1]})
                    dataset[user][video] = samples
    return dataset

# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset_gaze():
    dataset = {}
    # ToDo replaced this to Debug
    for root, directories, files in os.walk(os.path.join(ROOT_FOLDER, GAZE_TXT_FOLDER)):
        for enum_user, user in enumerate(directories):
            dataset[user] = {}
            for r_2, d_2, sub_files in os.walk(os.path.join(ROOT_FOLDER, GAZE_TXT_FOLDER, user)):
                for enum_video, video_txt in enumerate(sub_files):
                    video = video_txt.split('.')[0]
                    print('get gaze positions from original dataset', 'user', enum_user, '/', len(directories), 'video', enum_video, '/', len(sub_files))
                    filename = os.path.join(ROOT_FOLDER, GAZE_TXT_FOLDER, user, video_txt)
                    positions = get_gaze_positions_for_trace(filename)
                    frame_ids = get_indices_for_trace(filename)
                    video_rate = 1.0 / get_frame_rate(video, use_dict=True)
                    samples = []
                    for pos, frame_id in zip(positions, frame_ids):
                        # ToDo Check if gaze position x corresponds to yaw and gaze position y corresponds to pitch
                        samples.append({'sec':frame_id*video_rate, 'yaw':pos[0], 'pitch':pos[1]})
                    dataset[user][video] = samples
    return dataset

# The HM data takes the position in the panorama image, they are fractional from 0.0 to 1.0 with respect to the panorama image
# and computed from left bottom corner.
# Thus the longitudes ranges from 0 to 1, and the latitude ranges from 0 to 1.
# The subject starts to watch the video at position yaw=0.5, pitch=0.5.
## In other words, yaw = 0.5, pitch = 0.5 is equal to the position (1, 0, 0) in cartesian coordinates
# Pitching the head up results in a positive pitch value.
## In other words, yaw = Any, pitch = 1.0 is equal to the position (0, 0, 1) in cartesian coordinates
### We will transform these coordinates so that
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
def transform_the_degrees_in_range(yaw, pitch):
    yaw = yaw*2*np.pi
    pitch = (-pitch+1)*np.pi
    return yaw, pitch

# Performs the opposite transformation than transform_the_degrees_in_range
def transform_the_radians_to_original(yaw, pitch):
    yaw = (yaw)/(2*np.pi)
    pitch = (-(pitch/np.pi)+1)
    return yaw, pitch

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def create_sampled_dataset(original_dataset, rate):
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating sampled dataset', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
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
            dataset[user][video] = interpolate_quaternions(data_per_video[:, 0], data_per_video[:, 1:], rate=rate)
    return dataset

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def recover_original_angles_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample_new[0], sample_new[1], sample_new[2])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

def recover_original_angles_from_xyz_trace(xyz_trace):
    angles_per_video = []
    for sample in xyz_trace:
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample[1], sample[2], sample[3])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

### Check if the quaternions are good
# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def compare_sample_vs_original(original_dataset, sampled_dataset):
    for user in original_dataset.keys():
        for video in original_dataset[user].keys():
            pitchs = []
            yaws = []
            times = []
            for sample in original_dataset[user][video]:
                times.append(sample['sec'])
                yaws.append(sample['yaw'])
                pitchs.append(sample['pitch'])
            angles_per_video = recover_original_angles_from_xyz_trace(sampled_dataset[user][video])
            plt.subplot(1, 2, 1)
            plt.plot(times, yaws, label='yaw')
            plt.plot(times, pitchs, label='pitch')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 0], label='yaw')
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 1], label='pitch')
            plt.legend()
            plt.show()

# ToDo Copied exactly from Xu_PAMI_18/Reading_Dataset (Author: Miguel Romero)
def recover_xyz_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        angles_per_video.append(sample_new)
    return np.concatenate((quaternions_trace[:, 0:1], np.array(angles_per_video)), axis=1)


# Return the dataset
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
def get_xyz_dataset(sampled_dataset):
    dataset = {}
    for user in sampled_dataset.keys():
        dataset[user] = {}
        for video in sampled_dataset[user].keys():
            dataset[user][video] = recover_xyz_from_quaternions_trace(sampled_dataset[user][video])
    return dataset

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def plot_3d_trace(positions_gaze, positions_head, user, video):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.plot(positions_gaze[:, 0], positions_gaze[:, 1], positions_gaze[:, 2], label='gaze trace')
    ax.plot(positions_head[:, 0], positions_head[:, 1], positions_head[:, 2], label='head trace', color="r")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('User: %s, Video: %s' % (user, video))
    ax.legend()
    plt.show()

def plot_all_traces_in_3d(gaze_xyz_dataset, head_xyz_dataset):
    for user in gaze_xyz_dataset.keys():
        for video in gaze_xyz_dataset[user].keys():
            plot_3d_trace(gaze_xyz_dataset[user][video][:, 1:], head_xyz_dataset[user][video][:, 1:], user, video)

# Store the dataset in xyz coordinates form into the folder_to_store
def store_dataset(xyz_dataset, folder_to_store):
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            video_folder = os.path.join(folder_to_store, video)
            # Create the folder for the video if it doesn't exist
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            path = os.path.join(video_folder, user)
            df = pd.DataFrame(xyz_dataset[user][video])
            df.to_csv(path, header=False, index=False)

def create_and_store_gaze_sampled_dataset():
    original_dataset_gaze = get_original_dataset_gaze()
    sampled_dataset_gaze = create_sampled_dataset(original_dataset_gaze, SAMPLING_RATE)
    # compare_sample_vs_original(original_dataset_gaze, sampled_dataset_replica)
    xyz_dataset_gaze = get_xyz_dataset(sampled_dataset_gaze)
    store_dataset(xyz_dataset_gaze, OUTPUT_GAZE_FOLDER)

# ToDo This is the Main function of this file
def create_and_store_sampled_dataset(plot_comparison=False, plot_3d_traces=False):
    print('Getting original dataset')
    original_dataset = get_original_dataset()
    print('Creating sampled dataset')
    sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
    if plot_comparison:
        compare_sample_vs_original(original_dataset, sampled_dataset)
    xyz_dataset = get_xyz_dataset(sampled_dataset)
    store_dataset(xyz_dataset, OUTPUT_FOLDER)

# Read the split of the video ids for the train and test set in the experiments from CVPR18
def get_videos_train_and_test_from_file(root_folder):
    xl_train_test_set_file = pd.ExcelFile(os.path.join(root_folder, TRAIN_TEST_SET_FILE))
    df_train = xl_train_test_set_file.parse('train_set', header=None)
    df_test = xl_train_test_set_file.parse('test_set', header=None)
    videos_train = df_train[0].values
    videos_test = df_test[0].values
    videos_train = ['%03d'%video_id for video_id in videos_train]
    videos_test = ['%03d'%video_id for video_id in videos_test]
    return videos_train, videos_test

# Returns the maximum number of samples among all users (the length of the largest trace)
def get_max_num_samples_for_video(video, sampled_dataset_gaze, users_in_video):
    max_len = 0
    for user in users_in_video:
        curr_len = len(sampled_dataset_gaze[user][video])
        if curr_len > max_len:
            max_len = curr_len
    return max_len

# ToDo: Copied from Xu_PAMI_18/Reading_Dataset
def create_and_store_true_saliency(sampled_dataset_gaze):
    if not os.path.exists(OUTPUT_TRUE_SALIENCY_FOLDER):
        os.makedirs(OUTPUT_TRUE_SALIENCY_FOLDER)

    # Returns an array of size (NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL) with values between 0 and 1 specifying the probability that a tile is watched by the user
    # We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
    def from_position_to_tile_probability_cartesian(pos):
        yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, NUM_TILES_WIDTH_TRUE_SAL, endpoint=False),
                                           np.linspace(0, 1, NUM_TILES_HEIGHT_TRUE_SAL, endpoint=False))
        yaw_grid += 1.0 / (2.0 * NUM_TILES_WIDTH_TRUE_SAL)
        pitch_grid += 1.0 / (2.0 * NUM_TILES_HEIGHT_TRUE_SAL)
        yaw_grid = yaw_grid * 2 * np.pi
        pitch_grid = pitch_grid * np.pi
        x_grid, y_grid, z_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)
        great_circle_distance = np.arccos(np.maximum(np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0), -1.0))
        gaussian_orth = np.exp((-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance))
        return gaussian_orth

    videos = get_video_ids(OUTPUT_GAZE_FOLDER)
    users_per_video = get_users_per_video(OUTPUT_GAZE_FOLDER)

    for enum_video, video in enumerate(videos):
        print('creating true saliency for video', video, '-', enum_video, '/', len(videos))
        real_saliency_for_video = []

        max_num_samples = get_max_num_samples_for_video(video, sampled_dataset_gaze, users_per_video[video])

        for x_i in range(max_num_samples):
            tileprobs_for_video_cartesian = []
            for user in users_per_video[video]:
                if len(sampled_dataset_gaze[user][video]) > x_i:
                    tileprobs_cartesian = from_position_to_tile_probability_cartesian(sampled_dataset_gaze[user][video][x_i, 1:])
                    tileprobs_for_video_cartesian.append(tileprobs_cartesian)
            tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
            real_saliency_cartesian = np.sum(tileprobs_for_video_cartesian, axis=0) / tileprobs_for_video_cartesian.shape[0]
            real_saliency_for_video.append(real_saliency_cartesian)
        real_saliency_for_video = np.array(real_saliency_for_video)

        true_sal_out_file = os.path.join(OUTPUT_TRUE_SALIENCY_FOLDER, video)
        np.save(true_sal_out_file, real_saliency_for_video)

def load_sampled_dataset():
    list_of_videos = [o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith('.gitkeep')]
    dataset = {}
    for video in list_of_videos:
        for user in [o for o in os.listdir(os.path.join(OUTPUT_FOLDER, video)) if not o.endswith('.gitkeep')]:
            if user not in dataset.keys():
                dataset[user] = {}
            path = os.path.join(OUTPUT_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset

def load_gaze_sampled_dataset():
    list_of_videos = [o for o in os.listdir(OUTPUT_GAZE_FOLDER) if not o.endswith('.gitkeep')]
    dataset = {}
    for video in list_of_videos:
        for user in [o for o in os.listdir(os.path.join(OUTPUT_GAZE_FOLDER, video)) if not o.endswith('.gitkeep')]:
            if user not in dataset.keys():
                dataset[user] = {}
            path = os.path.join(OUTPUT_GAZE_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the input parameters to parse the dataset.')
    parser.add_argument('-creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset from the gaze positions.')
    parser.add_argument('-creat_samp_dat_head', action="store_true", dest='_create_sampled_dataset_head', help='Flag that tells if we want to create and store the sampled dataset from the gaze positions.')
    parser.add_argument('-creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('-creat_cb_sal', action="store_true", dest='_create_cb_saliency', help='Flag that tells if we want to create the content-based saliency maps.')
    parser.add_argument('-compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('-plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')

    args = parser.parse_args()

    # When creating the sampled head motion dataset
    #print('Use this file to create and store sampled dataset, or to create true saliency maps')
    if args._create_sampled_dataset_head:
        create_and_store_sampled_dataset()
    
    # When comparing with the original experiment
    if args._create_sampled_dataset:
        create_and_store_gaze_sampled_dataset()

    if args._compare_traces:
        original_dataset = get_original_dataset()
        sampled_dataset = load_sampled_dataset()
        compare_sample_vs_original(original_dataset, sampled_dataset)

    if args._plot_3d_traces:
        gaze_sampled_dataset = load_gaze_sampled_dataset()
        head_sampled_dataset = load_sampled_dataset()
        plot_all_traces_in_3d(gaze_sampled_dataset, head_sampled_dataset)

    if args._create_true_saliency:
        if os.path.isdir(OUTPUT_FOLDER):
            sampled_dataset = load_gaze_sampled_dataset()
            create_and_store_true_saliency(sampled_dataset)
        else:
            print('Please verify that the sampled dataset has been created correctly under the folder', OUTPUT_FOLDER)
