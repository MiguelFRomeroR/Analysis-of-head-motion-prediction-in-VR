import os
import pandas as pd
import numpy as np
import cv2
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Compute the sphere distance with the unit directional vectors in cartesian coordinate system
def compute_orth_dist_with_unit_dir_vec(position_a, position_b):
    yaw_true = (position_a[:, 0] - 0.5) * 2 * pi
    pitch_true = (position_a[:, 1] - 0.5) * pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (position_b[:, 0] - 0.5) * 2 * pi
    pitch_pred = (position_b[:, 1] - 0.5) * pi
    # Transform into directional vector in Cartesian Coordinate System
    x_true = np.sin(yaw_true)*np.cos(pitch_true)
    y_true = np.sin(pitch_true)
    z_true = np.cos(yaw_true)*np.cos(pitch_true)
    x_pred = np.sin(yaw_pred)*np.cos(pitch_pred)
    y_pred = np.sin(pitch_pred)
    z_pred = np.cos(yaw_pred)*np.cos(pitch_pred)
    # Finally compute orthodromic distance
    # great_circle_distance = np.arccos(x_true*x_pred+y_true*y_pred+z_true*z_pred)
    # To keep the values in bound between -1 and 1
    great_circle_distance = np.arccos(np.maximum(np.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0))
    return great_circle_distance

# Returns the trace (yaw, pitch) both in the range (0, 1)
def get_trace(video, user):
    filename = 'DatasetAnalysis/PAMI18/trace_%s_%s.csv' % (user, video)
    dataframe = pd.read_table(filename, header=None, sep=",", engine='python')
    dataframe_values = np.array(dataframe.values)
    dataframe_values = dataframe_values + np.array([180, 90])
    dataframe_values = dataframe_values / np.array([360, 180])
    return dataframe_values

fps_per_video = {'A380':30.0, 'AcerEngine':30.0, 'AcerPredator':25.0, 'AirShow':30.0, 'BFG':24.0, 'Bicycle':30.0, 'Camping':30.0, 'CandyCarnival':24.0, 'Castle':25.0, 'Catwalks':25.0, 'CMLauncher':30.0, 'CS':30.0, 'DanceInTurn':30.0, 'DrivingInAlps':25.0, 'Egypt':30.0, 'F5Fighter':25.0, 'Flight':24.0, 'GalaxyOnFire':30.0, 'Graffiti':25.0, 'GTA':30.0, 'HondaF1':30.0, 'IRobot':24.0, 'KasabianLive':30.0, 'Lion':30.0, 'LoopUniverse':25.0, 'Manhattan':30.0, 'MC':30.0, 'MercedesBenz':30.0, 'Motorbike':30.0, 'Murder':30.0, 'NotBeAloneTonight':30.0, 'Orion':25.0, 'Parachuting':30.0, 'Parasailing':30.0, 'Pearl':30.0, 'Predator':24.0, 'ProjectSoul':30.0, 'Rally':25.0, 'RingMan':25.0, 'Roma':24.0, 'Shark':24.0, 'Skiing':30.0, 'Snowfield':30.0, 'SnowRopeway':25.0, 'Square':30.0, 'StarWars':25.0, 'StarWars2':25.0, 'Stratosphere':30.0, 'StreetFighter':30.0, 'Supercar':25.0, 'SuperMario64':24.0, 'Surfing':25.0, 'SurfingArctic':25.0, 'TalkingInCar':30.0, 'Terminator':30.0, 'TheInvisible':30.0, 'Village':25.0, 'VRBasketball':25.0, 'Waterskiing':30.0, 'WesternSichuan':30.0, 'Yacht':30.0, 'KingKong':30.0, 'SpaceWar2':30.0, 'StarryPolar':12.0, 'Dancing':30.0, 'Guitar':30.0, 'BTSRun':30.0, 'InsideCar':30.0, 'RioOlympics':24.0, 'SpaceWar':30.0, 'CMLauncher2':30.0, 'Waterfall':30.0, 'Sunset':30.0, 'BlueWorld':30.0, 'Symphony':25.0, 'WaitingForLove':30.0}
videos = ['A380', 'AcerEngine', 'AcerPredator', 'AirShow', 'BFG', 'Bicycle', 'Camping', 'CandyCarnival', 'Castle', 'Catwalks', 'CMLauncher', 'CS', 'DanceInTurn', 'DrivingInAlps', 'Egypt', 'F5Fighter', 'Flight', 'GalaxyOnFire', 'Graffiti', 'GTA', 'HondaF1', 'IRobot', 'KasabianLive', 'Lion', 'LoopUniverse', 'Manhattan', 'MC', 'MercedesBenz', 'Motorbike', 'Murder', 'NotBeAloneTonight', 'Orion', 'Parachuting', 'Parasailing', 'Pearl', 'Predator', 'ProjectSoul', 'Rally', 'RingMan', 'Roma', 'Shark', 'Skiing', 'Snowfield', 'SnowRopeway', 'Square', 'StarWars', 'StarWars2', 'Stratosphere', 'StreetFighter', 'Supercar', 'SuperMario64', 'Surfing', 'SurfingArctic', 'TalkingInCar', 'Terminator', 'TheInvisible', 'Village', 'VRBasketball', 'Waterskiing', 'WesternSichuan', 'Yacht', 'KingKong', 'SpaceWar2', 'StarryPolar', 'Dancing', 'Guitar', 'BTSRun', 'InsideCar', 'RioOlympics', 'SpaceWar', 'CMLauncher2', 'Waterfall', 'Sunset', 'BlueWorld', 'Symphony', 'WaitingForLove']
train_videos = ['A380', 'AcerEngine', 'AcerPredator', 'AirShow', 'BFG', 'Bicycle', 'Camping', 'CandyCarnival', 'Castle', 'Catwalks', 'CMLauncher', 'CS', 'DanceInTurn', 'DrivingInAlps', 'Egypt', 'F5Fighter', 'Flight', 'GalaxyOnFire', 'Graffiti', 'GTA', 'HondaF1', 'IRobot', 'KasabianLive', 'Lion', 'LoopUniverse', 'Manhattan', 'MC', 'MercedesBenz', 'Motorbike', 'Murder', 'NotBeAloneTonight', 'Orion', 'Parachuting', 'Parasailing', 'Pearl', 'Predator', 'ProjectSoul', 'Rally', 'RingMan', 'Roma', 'Shark', 'Skiing', 'Snowfield', 'SnowRopeway', 'Square', 'StarWars', 'StarWars2', 'Stratosphere', 'StreetFighter', 'Supercar', 'SuperMario64', 'Surfing', 'SurfingArctic', 'TalkingInCar', 'Terminator', 'TheInvisible', 'Village', 'VRBasketball', 'Waterskiing', 'WesternSichuan', 'Yacht']
test_videos = ['KingKong', 'SpaceWar2', 'StarryPolar', 'Dancing', 'Guitar', 'BTSRun', 'InsideCar', 'RioOlympics', 'SpaceWar', 'CMLauncher2', 'Waterfall', 'Sunset', 'BlueWorld', 'Symphony', 'WaitingForLove']
users = ['chenmeiling_w1_23', 'CRYL_m1', 'diaopengfei_m1', 'fanchao_m1_22', 'fangyizhong_m1', 'fanshiyang_m1_23', 'fengyuting_w1', 'gaoweiqing_m1_25', 'gaoxi_w1', 'gaoyuan_m1', 'guozhanpeng_m1_24', 'haodongdong_m1_23', 'hewenjing_w1', 'huangweihan_m1', 'huruitao_m1', 'lande_m1', 'liangyankuan_m1_24', 'liantianye_m1', 'lichen_m', 'lijing_m1', 'liliu_w1_22', 'linxin_m1', 'linzhixing_m1', 'lisai_w1', 'liushijie_m1', 'liutong_m1', 'liwenli_w1', 'lucan_m1', 'lujiaxin_w1', 'luyunpeng_m1_21', 'mafu_m11', 'mashang_m1', 'ouliyang_w1', 'pengshuxue_m1', 'qiuyao_w1', 'renjie_m1', 'renzan_m1', 'shaowei_m1_23', 'shixiaonan_m1_20', 'subowen_m1', 'tianmiaomiao_w1', 'wangrui_m1', 'weiwu_m1', 'wuguanqun_m1', 'xujingyao_w1', 'xusanjia_m1', 'yangpengshuai_m1', 'yangren_m1', 'yangyandan_w1_21', 'yinjiayuan_m1', 'yumengyue_w1_24', 'yuwenhai_m1', 'zhangwenjing_w1', 'zhaosiyu_m1', 'zhaoxinyue_w1', 'zhaoyilin_w1', 'zhongyueyou_m1', 'zhuxudong_m1']

n_window = 1

for time_t in [0.2, 0.5, 1, 2, 5, 15]:
    print (time_t)
    average_velocities = []
    for video in videos:
        m_window = int(round(time_t * fps_per_video[video]))
        # print video, time_t, m_window
        for user in users:
            # print video, user
            positions = get_trace(video, user)
            for x_i in range(n_window, len(positions)-m_window):
                # This one computes the farthest motion from the last position
                av_vel = np.max(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i+m_window], np.repeat(positions[x_i-1:x_i], m_window, axis=0)))
                # This one computes the additive motion
                # av_vel = np.sum(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i+m_window], positions[x_i-1:x_i+m_window-1]))
                average_velocities.append(av_vel)
    average_velocities = np.array(average_velocities) * 180 / pi
    n, bins, patches = plt.hist(average_velocities, bins=np.arange(0, 360), density=True, histtype='step', cumulative=True, label=str(time_t)+'s')
    # hist, bin_edges = np.histogram(average_velocities, bins=np.arange(0, 360), density=True)
    # for i in range(len(hist)):
    #     print bin_edges[i], hist[i]
plt.xlabel('Motion from last position (t -> t+T) [Degrees]')
plt.ylabel('Data proportion')
plt.title('PAMI 18')
plt.xlim(0, 180)
plt.ylim(0.0, 1.0)
plt.legend()
plt.show()
