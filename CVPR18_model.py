from keras.models import Model
from keras.layers import Dense, LSTM, Lambda, Input, Reshape, Convolution2D, TimeDistributed, Concatenate, Flatten, ConvLSTM2D, MaxPooling2D
from keras import optimizers
import tensorflow as tf
from keras import backend as K

from models_utils import metric_orth_dist, toPosition, selectImageInModel, add_timestep_axis

# ----------------------------- TRAIN ----------------------------
def create_CVPR18_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH):
    # Defining model structure
    encoder_position_inputs = Input(shape=(M_WINDOW, 3))
    decoder_saliency_inputs = Input(shape=(H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1))
    decoder_position_inputs = Input(shape=(1, 3))

    # Propioception stack
    sense_pos_1_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_enc')
    sense_pos_2_enc = LSTM(units=256, return_sequences=False, return_state=True, name='prop_lstm_2_enc')

    sense_pos_1_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_dec')
    sense_pos_2_dec = LSTM(units=256, return_sequences=False, return_state=True, name='prop_lstm_2_dec')

    # Fuse stack
    fuse_1 = Dense(units=256)
    fuse_2 = Dense(units=256)

    # Act stack
    fc_layer_out = Dense(3)
    To_Position = Lambda(toPosition)

    prop_out_enc_1, state_h_1, state_c_1 = sense_pos_1_enc(encoder_position_inputs)
    states_1 = [state_h_1, state_c_1]
    print ('prop_out_enc_1.shape', prop_out_enc_1.shape)
    prop_out_enc_2, state_h_2, state_c_2 = sense_pos_2_enc(prop_out_enc_1)
    states_2 = [state_h_2, state_c_2]
    print ('prop_out_enc_2.shape', prop_out_enc_2.shape)

    # Decoding
    all_pos_outputs = []
    inputs = decoder_position_inputs
    for curr_idx in range(H_WINDOW):
        selected_timestep_saliency = Lambda(selectImageInModel, arguments={'curr_idx': curr_idx})(decoder_saliency_inputs)
        flatten_timestep_saliency = Reshape((1, NUM_TILES_WIDTH*NUM_TILES_HEIGHT))(selected_timestep_saliency)
        print ('inputs.shape', inputs.shape)
        prop_out_dec_1, state_h_1, state_c_1 = sense_pos_1_dec(inputs, initial_state=states_1)
        states_1 = [state_h_1, state_c_1]
        print ('prop_out_dec_1.shape', prop_out_dec_1.shape)
        prop_out_dec_2, state_h_2, state_c_2 = sense_pos_2_dec(prop_out_dec_1, initial_state=states_2)
        states_2 = [state_h_2, state_c_2]
        print ('prop_out_dec_2.shape', prop_out_dec_2.shape)
        prop_out_dec_2_timestep = Lambda(add_timestep_axis)(prop_out_dec_2)
        print ('prop_out_dec_2_timestep.shape', prop_out_dec_2_timestep.shape)

        conc_out_dec = Concatenate(axis=-1)([flatten_timestep_saliency, prop_out_dec_2_timestep])
        print ('conc_out_dec.shape', conc_out_dec.shape)

        fuse_out_1_dec = TimeDistributed(fuse_1)(conc_out_dec)
        print ('fuse_out_1_dec.shape', fuse_out_1_dec.shape)
        fuse_out_2_dec = TimeDistributed(fuse_2)(fuse_out_1_dec)
        print ('fuse_out_2_dec.shape', fuse_out_2_dec.shape)

        outputs_delta = fc_layer_out(fuse_out_2_dec)
        decoder_pred = To_Position([inputs, outputs_delta])
        print ('decoder_pred.shape', decoder_pred.shape)

        all_pos_outputs.append(decoder_pred)
        # Reinject the outputs as inputs for the next loop iteration as well as update the states
        inputs = decoder_pred

    # Concatenate all predictions
    decoder_outputs_pos = Lambda(lambda x: K.concatenate(x, axis=1))(all_pos_outputs)
    print ('decoder_outputs_pos.shape', decoder_outputs_pos.shape)
    # decoder_outputs_img = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)


    # Define and compile model
    model = Model([encoder_position_inputs, decoder_position_inputs, decoder_saliency_inputs], decoder_outputs_pos)
    model_optimizer = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=model_optimizer, loss='mean_squared_error', metrics=[metric_orth_dist])
    return model