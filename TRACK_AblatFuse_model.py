from keras.models import Model
from keras.layers import Dense, LSTM, Lambda, Input, Reshape, Convolution2D, TimeDistributed, Concatenate, Flatten, ConvLSTM2D, MaxPooling2D
from keras import optimizers
from keras import backend as K

from models_utils import metric_orth_dist, toPosition, selectImageInModel

# ----------------------------- TRAIN ----------------------------
def create_TRACK_AblatFuse_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH):
    # Defining model structure
    encoder_position_inputs = Input(shape=(M_WINDOW, 3))
    encoder_saliency_inputs = Input(shape=(M_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1))
    decoder_position_inputs = Input(shape=(1, 3))
    decoder_saliency_inputs = Input(shape=(H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1))

    sense_pos_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_enc')

    sense_sal_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_enc')

    sense_pos_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_dec')

    sense_sal_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_dec')

    fuse_1_dec = Dense(units=256)

    fuse_2 = Dense(units=256)

    fuse_3 = Dense(units=256)

    # Act stack
    fc_layer_out = Dense(3)
    To_Position = Lambda(toPosition)

    # Encoding
    out_enc_pos, state_h_1, state_c_1 = sense_pos_enc(encoder_position_inputs)
    states_1 = [state_h_1, state_c_1]

    out_flat_enc = TimeDistributed(Flatten())(encoder_saliency_inputs)
    out_enc_sal, state_h_2, state_c_2 = sense_sal_enc(out_flat_enc)
    states_2 = [state_h_2, state_c_2]

    # Decoding
    all_pos_outputs = []
    inputs = decoder_position_inputs
    for curr_idx in range(H_WINDOW):
        out_enc_pos, state_h_1, state_c_1 = sense_pos_dec(inputs, initial_state=states_1)
        states_1 = [state_h_1, state_c_1]

        selected_timestep_saliency = Lambda(selectImageInModel, arguments={'curr_idx': curr_idx})(decoder_saliency_inputs)
        flatten_timestep_saliency = Reshape((1, NUM_TILES_WIDTH * NUM_TILES_HEIGHT))(selected_timestep_saliency)
        out_enc_sal, state_h_2, state_c_2 = sense_sal_dec(flatten_timestep_saliency, initial_state=states_2)
        states_2 = [state_h_2, state_c_2]

        conc_out_dec = Concatenate(axis=-1)([out_enc_sal, out_enc_pos])

        fuse_out_dec_1 = TimeDistributed(fuse_1_dec)(conc_out_dec)
        fuse_out_dec_2 = TimeDistributed(fuse_2)(fuse_out_dec_1)
        fuse_out_dec_3 = TimeDistributed(fuse_3)(fuse_out_dec_2)

        outputs_delta = fc_layer_out(fuse_out_dec_3)

        decoder_pred = To_Position([inputs, outputs_delta])

        all_pos_outputs.append(decoder_pred)
        # Reinject the outputs as inputs for the next loop iteration as well as update the states
        inputs = decoder_pred

    # Concatenate all predictions
    decoder_outputs_pos = Lambda(lambda x: K.concatenate(x, axis=1))(all_pos_outputs)

    # Define and compile model
    model = Model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs], decoder_outputs_pos)
    model_optimizer = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=model_optimizer, loss='mean_squared_error', metrics=[metric_orth_dist])
    return model
