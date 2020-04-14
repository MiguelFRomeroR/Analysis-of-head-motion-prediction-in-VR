from keras.models import Model
from keras.layers import Dense, LSTM, Lambda, Input, Concatenate, Flatten
from keras import optimizers
import tensorflow as tf

# This way we ensure that the network learns to predict the displacement
def toPosition(values):
    last_pos = values[0][:, -1, :]
    pred_disp = values[1]*2.0-1.0
    return tf.math.mod(last_pos + pred_disp, 1)

# ----------------------------- TRAIN ----------------------------
def create_CVPR18_orig_Model(M_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH):
    # Model inputs
    pos_inputs = Input(shape=(M_WINDOW, 2))
    image_inputs = Input(shape=(NUM_TILES_HEIGHT, NUM_TILES_WIDTH))

    # Propioception stack
    lstm_pos_1 = LSTM(128, name='dense_prop_lstm_1', return_sequences=True)
    lstm_pos_2 = LSTM(128, name='dense_prop_lstm_2')

    # Image stack
    flat_image = Flatten()(image_inputs)

    # Act stack
    out_dense_1 = Dense(1000, name='act_dense_1')
    out_dense_2 = Dense(2, name='act_dense_2', activation='sigmoid')

    To_Position = Lambda(toPosition)

    lstm_1_out = lstm_pos_1(pos_inputs)
    lstm_2_out = lstm_pos_2(lstm_1_out)

    conc_out = Concatenate(axis=-1)([flat_image, lstm_2_out])

    act_out_1 = out_dense_1(conc_out)
    pred_disp = out_dense_2(act_out_1)

    pred_pos = To_Position([pos_inputs, pred_disp])

    # Define and compile model
    model = Model([pos_inputs, image_inputs], pred_pos)
    sgd = optimizers.SGD(lr=0.1, decay=0.0005, momentum=0.9)
    model.compile(optimizer=sgd, loss='mse')
    return model

import numpy as np
def auto_regressive_prediction(model, pos_inputs, saliency_inputs, M_WINDOW, H_WINDOW):
    reg_pos_inputs = np.copy(pos_inputs)
    model_predictions = []
    for t in range(H_WINDOW):
        model_prediction = model.predict([pos_inputs[:, -M_WINDOW:], saliency_inputs[:, t, :, :, 0]])
        reg_pos_inputs = np.array([np.concatenate((reg_pos_inputs[0], model_prediction))])
        model_predictions.append(model_prediction[0])
    return np.array(model_predictions)

# if __name__ == "__main__":
#     model = create_CVPR18_orig_Model(5, 20, 20)
#     res = auto_regressive_prediction(model, np.zeros((1, 5, 2)), np.zeros((1, 25, 20, 20, 1)), 5, 25)
#     print(res.shape)