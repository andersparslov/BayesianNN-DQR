import tensorflow as tf
#import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, Activation, BatchNormalization, Flatten, Reshape

def tilted_loss(q,y,f):
    e = (y[:,:,:,0,0]-f[:,:,:,0,0])
    return K.mean(K.maximum(q*e, (q-1)*e))

def mse_loss(y, f):
    return K.mean(K.square(y[:,:,:,0,0]-f[:,:,:,0,0]), axis = -1)

## Tilted loss for both mean and quantiles
def joint_tilted_loss(quantiles, y, f):
    loss = K.mean(K.square(y[:,:,:,0,0]-f[:,:,:,0,0]), axis = -1)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,:,:,0,k+1]-f[:,:,:,0,k+1])
        loss += K.mean(K.maximum(q*e, (q-1)*e))
    return loss

## Encoder-decoder convolutional LSTM for jointly estimating quantiles and mean predictions.
def joint_convLstm(num_filters, kernel_length, input_timesteps, num_links, output_timesteps, quantiles, prob, loss):
    model = Sequential()
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links, 1, 1)))
    model.add(ConvLSTM2D(name ='conv_lstm_1',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences = False))

    model.add(Dropout(prob, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(Flatten())
    model.add(RepeatVector(output_timesteps))
    model.add(Reshape((output_timesteps, num_links, 1, num_filters)))

    model.add(ConvLSTM2D(name ='conv_lstm_2',filters = num_filters, kernel_size = (kernel_length, 1), padding='same',return_sequences = True))
    model.add(Dropout(prob, name = 'dropout_2'))

    model.add(TimeDistributed(Dense(units = len(quantiles) + 1, name = 'dense_1')))
    model.compile(loss = loss, optimizer = 'nadam')
    return model

## Encoder-decoder LSTM for mean 
def convLstm(num_filters, kernel_length, input_timesteps, num_links, output_timesteps, prob, loss):
    model = Sequential()
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links, 1, 1)))

    model.add(ConvLSTM2D(name ='conv_lstm_1',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences = False))

    model.add(Dropout(prob, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(Flatten())
    model.add(RepeatVector(output_timesteps))
    model.add(Reshape((output_timesteps, num_links, 1, num_filters)))

    model.add(ConvLSTM2D(name ='conv_lstm_2',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences = True))
    model.add(Dropout(prob, name = 'dropout_2'))
    model.add(TimeDistributed(Dense(units = 1, name = 'dense_1')))
    model.compile(loss = loss, optimizer = 'nadam')
    return model
