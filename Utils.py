import numpy as np
from numpy import cross, dot
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import csv
import pandas as pd

# The (input) corresponds to (x, y, z) of a unit sphere centered at the origin (0, 0, 0)
# Returns the values (theta, phi) with:
# theta in the range 0, to 2*pi, theta can be negative, e.g. cartesian_to_eulerian(0, -1, 0) = (-pi/2, pi/2) (is equal to (3*pi/2, pi/2))
# phi in the range 0 to pi (0 being the north pole, pi being the south pole)
def cartesian_to_eulerian(x, y, z):
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/r)
    # remainder is used to transform it in the positive range (0, 2*pi)
    theta = np.remainder(theta, 2*np.pi)
    return theta, phi

# The (input) values of theta and phi are assumed to be as follows:
# theta = Any              phi =   0    : north pole (0, 0, 1)
# theta = Any              phi =  pi    : south pole (0, 0, -1)
# theta = 0, 2*pi          phi = pi/2   : equator facing (1, 0, 0)
# theta = pi/2             phi = pi/2   : equator facing (0, 1, 0)
# theta = pi               phi = pi/2   : equator facing (-1, 0, 0)
# theta = -pi/2, 3*pi/2    phi = pi/2   : equator facing (0, -1, 0)
# In other words
# The longitude ranges from 0, to 2*pi
# The latitude ranges from 0 to pi, origin of equirectangular in the top-left corner
# Returns the values (x, y, z) of a unit sphere with center in (0, 0, 0)
def eulerian_to_cartesian(theta, phi):
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z])

## Use this function to debug the behavior of the function eulerian_to_cartesian()
def test_eulerian_to_cartesian():
    # trace starting from (1, 0, 0) and moving to point (0, 0, 1)
    yaw_1 = np.linspace(0, np.pi/2, 50, endpoint=True)
    pitch_1 = np.linspace(np.pi/2, 0, 50, endpoint=True)
    positions_1 = np.array([eulerian_to_cartesian(yaw_samp, pitch_samp) for yaw_samp, pitch_samp in zip(yaw_1, pitch_1)])
    yaw_2 = np.linspace(3*np.pi/2, np.pi, 50, endpoint=True)
    pitch_2 = np.linspace(np.pi, np.pi/2, 50, endpoint=True)
    positions_2 = np.array([eulerian_to_cartesian(yaw_samp, pitch_samp) for yaw_samp, pitch_samp in zip(yaw_2, pitch_2)])
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.plot(positions_1[:, 0], positions_1[:, 1], positions_1[:, 2], color='b')
    ax.plot(positions_2[:, 0], positions_2[:, 1], positions_2[:, 2], color='g')
    ax.scatter(positions_1[0, 0], positions_1[0, 1], positions_1[0, 2], color='r')
    ax.scatter(positions_2[0, 0], positions_2[0, 1], positions_2[0, 2], color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax2 = fig.add_subplot(122)
    ax2.plot(yaw_1, pitch_1, color='b')
    ax2.plot(yaw_2, pitch_2, color='g')
    ax2.scatter(yaw_1[0], pitch_1[0], color='r')
    ax2.scatter(yaw_2[0], pitch_2[0], color='r')
    # to turn around the y axis, starting from 0 on the top
    ax2.set_ylim(ax2.get_ylim()[1], ax2.get_ylim()[0])
    ax2.set_xlabel('yaw')
    ax2.set_ylabel('pitch')
    plt.show()

# Transforms the eulerian angles from range (0, 2*pi) and (0, pi) to (-pi, pi) and (-pi/2, pi/2)
def eulerian_in_range(theta, phi):
    theta = theta - np.pi
    phi = (phi - (np.pi / 2.0))
    return theta, phi

# Returns an array of size (numOfTilesHeight, numOfTilesWidth) with values between 0 and 1 specifying the probability that a tile is watched by the user
# We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
def from_position_to_tile_probability(pos, numTilesWidth, numTilesHeight):
    yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, numTilesWidth, endpoint=False), np.linspace(0, 1, numTilesHeight, endpoint=False))
    yaw_grid += 1.0 / (2.0 * numTilesWidth)
    pitch_grid += 1.0 / (2.0 * numTilesHeight)
    yaw_grid = (yaw_grid - 0.5) * 2 * np.pi
    pitch_grid = (pitch_grid - 0.5) * np.pi
    cp_yaw = pos[0]
    cp_pitch = pos[1]
    delta_long = np.abs(np.arctan2(np.sin(yaw_grid - cp_yaw), np.cos(yaw_grid - cp_yaw)))
    numerator = np.sqrt(np.power(np.cos(cp_pitch) * np.sin(delta_long), 2.0) + np.power(np.cos(pitch_grid) * np.sin(cp_pitch) - np.sin(pitch_grid) * np.cos(cp_pitch) * np.cos(delta_long), 2.0))
    denominator = np.sin(pitch_grid) * np.sin(cp_pitch) + np.cos(pitch_grid) * np.cos(cp_pitch) * np.cos(delta_long)
    second_ort = np.abs(np.arctan2(numerator, denominator))
    gaussian_orth = np.exp((-1.0/(2.0*np.square(0.1))) * np.square(second_ort))
    return gaussian_orth

# Returns an array of size (numOfTilesHeight, numOfTilesWidth) with 1 in the tile where the head is and 0s elsewhere
def from_position_to_tile(pos, numTilesWidth, numTilesHeight):
    yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, numTilesWidth, endpoint=False), np.linspace(0, 1, numTilesHeight, endpoint=False))
    yaw_grid += 1.0 / (2.0 * numTilesWidth)
    pitch_grid += 1.0 / (2.0 * numTilesHeight)
    yaw_grid = (yaw_grid - 0.5) * 2 * np.pi
    pitch_grid = (pitch_grid - 0.5) * np.pi
    cp_yaw = pos[0]
    cp_pitch = pos[1]
    delta_long = np.abs(np.arctan2(np.sin(yaw_grid - cp_yaw), np.cos(yaw_grid - cp_yaw)))
    numerator = np.sqrt(np.power(np.cos(cp_pitch) * np.sin(delta_long), 2.0) + np.power(np.cos(pitch_grid) * np.sin(cp_pitch) - np.sin(pitch_grid) * np.cos(cp_pitch) * np.cos(delta_long), 2.0))
    denominator = np.sin(pitch_grid) * np.sin(cp_pitch) + np.cos(pitch_grid) * np.cos(cp_pitch) * np.cos(delta_long)
    second_ort = np.abs(np.arctan2(numerator, denominator))
    gaussian_orth = np.exp((-1.0/(2.0*np.square(0.1))) * np.square(second_ort))
    max_pos = np.where(gaussian_orth==np.max(gaussian_orth), 1, 0)
    return max_pos

def orthogonal(v):
    x = abs(v[0])
    y = abs(v[1])
    z = abs(v[2])
    other = (1, 0, 0) if (x < y and x < z) else (0, 1, 0) if (y < z) else (0, 0, 1)
    return cross(v, other)

def normalized(v):
    return normalize(v[:, np.newaxis], axis=0).ravel()


def rotationBetweenVectors(u, v):
    u = normalized(u)
    v = normalized(v)

    if np.allclose(u, v):
        return Quaternion(angle=0.0, axis=u)
    if np.allclose(u, -v):
        return Quaternion(angle=np.pi, axis=normalized(orthogonal(u)))

    quat = Quaternion(angle=np.arccos(dot(u, v)), axis=normalized(cross(u, v)))
    return quat

def degrees_to_radian(degree):
    return degree*np.pi/180.0

def radian_to_degrees(radian):
    return radian*180.0/np.pi

# time_orig_at_zero is a flag to determine if the time must start counting from zero, if so, the trace is forced to start at 0.0
def interpolate_quaternions(orig_times, quaternions, rate, time_orig_at_zero=True):
    # if the first time-stamps is greater than (half) the frame rate, put the time-stamp 0.0 and copy the first quaternion to the beginning
    if time_orig_at_zero and (orig_times[0] > rate/2.0):
        orig_times = np.concatenate(([0.0], orig_times))
        # ToDo use the quaternion rotation to predict where the position was at t=0
        quaternions = np.concatenate(([quaternions[0]], quaternions))
    key_rots = R.from_quat(quaternions)
    slerp = Slerp(orig_times, key_rots)
    # we add rate/2 to the last time-stamp so we include it in the possible interpolation time-stamps
    times = np.arange(orig_times[0], orig_times[-1]+rate/2.0, rate)
    # to bound it to the maximum original-time in the case of rounding errors
    times[-1] = min(orig_times[-1], times[-1])
    interp_rots = slerp(times)
    return np.concatenate((times[:, np.newaxis], interp_rots.as_quat()), axis=1)

# Compute the orthodromic distance between two points in 3d coordinates
def compute_orthodromic_distance(true_position, pred_position):
    norm_a = np.sqrt(np.square(true_position[0]) + np.square(true_position[1]) + np.square(true_position[2]))
    norm_b = np.sqrt(np.square(pred_position[0]) + np.square(pred_position[1]) + np.square(pred_position[2]))
    x_true = true_position[0] / norm_a
    y_true = true_position[1] / norm_a
    z_true = true_position[2] / norm_a
    x_pred = pred_position[0] / norm_b
    y_pred = pred_position[1] / norm_b
    z_pred = pred_position[2] / norm_b
    great_circle_distance = np.arccos(np.maximum(np.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0))
    return great_circle_distance

def compute_mse(true_position, pred_position):
    return mean_squared_error(true_position, pred_position)

# Returns the position in cartesian coordinates of the point with maximum saliency of the equirectangular saliency map
def get_max_sal_pos(saliency_map, dataset_name):
    row, col = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    y = row/float(saliency_map.shape[0])
    if dataset_name == 'Xu_CVPR_18':
        x = np.remainder(col/float(saliency_map.shape[1])-0.5, 1.0)
    else:
        x = col / float(saliency_map.shape[1])
    pos_cartesian = eulerian_to_cartesian(x*2*np.pi, y*np.pi)
    return pos_cartesian

def store_dict_as_csv(csv_file, csv_columns, dict_data):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

def load_dict_from_csv(filename, columns, sep=',', header=0, engine='python'):
    dataframe = pd.read_csv(filename, engine=engine, header=header, sep=sep)
    data = dataframe[columns]
    return dataframe.values

all_metrics = {}
all_metrics['orthodromic'] = compute_orthodromic_distance
all_metrics['mse'] = compute_mse