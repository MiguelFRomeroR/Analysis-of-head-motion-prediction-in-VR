from keras.models import Model
from keras.layers import Dense, LSTM, Lambda, Input, Reshape, Convolution2D, TimeDistributed, Concatenate, Flatten, ConvLSTM2D, MaxPooling2D
from keras import optimizers
import tensorflow as tf
from keras import backend as K
import numpy as np

def metric_orth_dist(true_position, pred_position):
    yaw_true = (true_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_true = (true_position[:, :, 1:2] - 0.5) * np.pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (pred_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_pred = (pred_position[:, :, 1:2] - 0.5) * np.pi
    # Finally compute orthodromic distance
    delta_long = tf.abs(tf.atan2(tf.sin(yaw_true - yaw_pred), tf.cos(yaw_true - yaw_pred)))
    numerator = tf.sqrt(tf.pow(tf.cos(pitch_pred)*tf.sin(delta_long), 2.0) + tf.pow(tf.cos(pitch_true)*tf.sin(pitch_pred)-tf.sin(pitch_true)*tf.cos(pitch_pred)*tf.cos(delta_long), 2.0))
    denominator = tf.sin(pitch_true)*tf.sin(pitch_pred)+tf.cos(pitch_true)*tf.cos(pitch_pred)*tf.cos(delta_long)
    great_circle_distance = tf.abs(tf.atan2(numerator, denominator))
    return great_circle_distance

# This way we ensure that the network learns to predict the delta angle
def toPosition(values):
    orientation = values[0]
    magnitudes = values[1]/2.0
    directions = values[2]
    # The network returns values between 0 and 1, we force it to be between -2/5 and 2/5
    motion = magnitudes * directions

    yaw_pred_wo_corr = orientation[:, :, 0:1] + motion[:, :, 0:1]
    pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]

    cond_above = tf.cast(tf.greater(pitch_pred_wo_corr, 1.0), tf.float32)
    cond_correct = tf.cast(tf.logical_and(tf.less_equal(pitch_pred_wo_corr, 1.0), tf.greater_equal(pitch_pred_wo_corr, 0.0)), tf.float32)
    cond_below = tf.cast(tf.less(pitch_pred_wo_corr, 0.0), tf.float32)

    pitch_pred = cond_above * (1.0 - (pitch_pred_wo_corr - 1.0)) + cond_correct * pitch_pred_wo_corr + cond_below * (-pitch_pred_wo_corr)
    yaw_pred = tf.math.mod(cond_above * (yaw_pred_wo_corr - 0.5) + cond_correct * yaw_pred_wo_corr + cond_below * (yaw_pred_wo_corr - 0.5),1.0)
    return tf.concat([yaw_pred, pitch_pred], -1)

# ----------------------------- TRAIN ----------------------------
def create_pos_only_model(M_WINDOW, H_WINDOW):
    # Defining model structure
    encoder_inputs = Input(shape=(M_WINDOW, 2))
    decoder_inputs = Input(shape=(1, 2))

    lstm_layer = LSTM(1024, return_sequences=True, return_state=True)
    decoder_dense_mot = Dense(2, activation='sigmoid')
    decoder_dense_dir = Dense(2, activation='tanh')
    To_Position = Lambda(toPosition)

    # Encoding
    encoder_outputs, state_h, state_c = lstm_layer(encoder_inputs)
    states = [state_h, state_c]

    # Decoding
    all_outputs = []
    inputs = decoder_inputs
    for curr_idx in range(H_WINDOW):
        # # Run the decoder on one timestep
        decoder_pred, state_h, state_c = lstm_layer(inputs, initial_state=states)
        outputs_delta = decoder_dense_mot(decoder_pred)
        outputs_delta_dir = decoder_dense_dir(decoder_pred)
        outputs_pos = To_Position([inputs, outputs_delta, outputs_delta_dir])
        # Store the current prediction (we will concantenate all predictions later)
        all_outputs.append(outputs_pos)
        # Reinject the outputs as inputs for the next loop iteration as well as update the states
        inputs = outputs_pos
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    # decoder_outputs = all_outputs

    # Define and compile model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model_optimizer = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=model_optimizer, loss=metric_orth_dist)
    return model