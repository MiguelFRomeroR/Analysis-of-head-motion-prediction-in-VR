import tensorflow as tf
import numpy as np

def metric_orth_dist(position_a, position_b):
    # Transform into directional vector in Cartesian Coordinate System
    norm_a = tf.sqrt(tf.square(position_a[:, :, 0:1]) + tf.square(position_a[:, :, 1:2]) + tf.square(position_a[:, :, 2:3]))
    norm_b = tf.sqrt(tf.square(position_b[:, :, 0:1]) + tf.square(position_b[:, :, 1:2]) + tf.square(position_b[:, :, 2:3]))
    x_true = position_a[:, :, 0:1]/norm_a
    y_true = position_a[:, :, 1:2]/norm_a
    z_true = position_a[:, :, 2:3]/norm_a
    x_pred = position_b[:, :, 0:1]/norm_b
    y_pred = position_b[:, :, 1:2]/norm_b
    z_pred = position_b[:, :, 2:3]/norm_b
    # Finally compute orthodromic distance
    # great_circle_distance = np.arccos(x_true*x_pred+y_true*y_pred+z_true*z_pred)
    # To keep the values in bound between -1 and 1
    great_circle_distance = tf.acos(tf.maximum(tf.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0))
    return great_circle_distance

# This way we ensure that the network learns to predict the delta angle
def toPosition(values):
    orientation = values[0]
    # The network returns values between 0 and 1, we force it to be between -1/2 and 1/2
    motion = values[1]
    return (orientation + motion)

def selectImageInModel(input_to_selector, curr_idx):
    selected_image = input_to_selector[:, curr_idx:curr_idx+1]
    return selected_image

def add_timestep_axis(input):
    return tf.expand_dims(input, 1)

# if __name__ == "__main__":
