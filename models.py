import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, Activation, BatchNormalization, Flatten, Reshape

## Alpha affects the number of hidden nodes in LSTM layers.
def encoderDecoder(alpha, num_samples, input_dim, output_dim, loss = 'mse'):
    mod = Sequential()
    mod.add(LSTM(int(num_samples / (alpha*(input_dim + output_dim))), activation='relu', input_shape=(input_dim, 1)))
    mod.add(Dropout(0.2))
    mod.add(RepeatVector(output_dim))
    mod.add(LSTM(int(num_samples / (alpha*(input_dim + output_dim))), activation='relu', return_sequences=True))
    mod.add(Dropout(0.2))
    mod.add(TimeDistributed(Dense(1)))
    mod.compile(optimizer='adam', loss = loss)
    return mod
    
def twoLayerLTSM(input_dim, output_dim, loss = 'mse'):
    mod = Sequential()
    mod.add(LSTM(50, input_shape=(input_dim, 1), activation='relu', return_sequences = True))
    mod.add(LSTM(50, activation='relu'))
    mod.add(Dropout(0.5))
    mod.add(Dense(output_dim))
    mod.add(Dropout(0.2))
    mod.compile(optimizer='adam', loss = loss)
    return mod

def multi_tilted_loss2(quantiles, y, f):
    loss = K.mean(K.square(y[:,:,:,0,0]-f[:,:,:,0,0]), axis=-1)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,:,:,0,k+1]-f[:,:,:,0,k+1])
        loss += K.mean(K.maximum(q*e, (q-1)*e))
    return loss

def build_model(num_nodes, input_dim, output_dim, loss):
    mod = Sequential()
    mod.add(LSTM(100, activation='relu', input_shape=(input_dim, 1)))
    mod.add(Dropout(0.2))
    mod.add(RepeatVector(output_dim))
    mod.add(LSTM(100, activation='relu', return_sequences=True))
    mod.add(Dropout(0.2))
    mod.add(TimeDistributed(Dense(output_dim)))
    mod.compile(loss=loss, optimizer = 'nadam')
    return mod

def joint_multilink(num_filters, kernel_length, input_timesteps, num_links, output_timesteps, quantiles, loss):
    model = Sequential()
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links, 1, 1)))
    model.add(ConvLSTM2D(name ='conv_lstm_1',
                         filters = num_filters, kernel_size = (kernel_length, 1),                       
                         padding = 'same', 
                         return_sequences = True))

    model.add(Dropout(0.21, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(ConvLSTM2D(name ='conv_lstm_2',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences = False))

    model.add(Dropout(0.20, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))

    model.add(Flatten())
    model.add(RepeatVector(output_timesteps))
    model.add(Reshape((output_timesteps, num_links, 1, num_filters)))

    model.add(ConvLSTM2D(name ='conv_lstm_3',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences = True))

    model.add(Dropout(0.20, name = 'dropout_3'))
    model.add(BatchNormalization(name = 'batch_norm_3'))

    model.add(ConvLSTM2D(name ='conv_lstm_4',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences = True))

    model.add(TimeDistributed(Dense(units = len(quantiles) + 1, name = 'dense_1')))
    model.compile(loss = loss, optimizer = 'nadam')
    return model

def build_convLSTM(timesteps, num_links, output_dim, loss):
    model = Sequential()
    model.add(ConvLSTM2D(filters = 20, kernel_size=(2, 2),
                   input_shape = (timesteps, num_links, 1, 1),
                   padding = 'same', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=20, kernel_size=(2, 2),
                       padding='same', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units = output_dim))
    model.add(Activation("relu"))
    model.compile(loss = loss, optimizer = 'nadam')
    return model

def build_convLSTM2(input_timesteps, num_links, output_timesteps, loss):
    model = Sequential()
    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links)))
    model.add(ConvLSTM2D(name ='conv_lstm_1',
                         filters = 64, kernel_size = (10, 1),                       
                         padding = 'same', 
                         return_sequences = True))
    
    model.add(Dropout(0.21, name = 'dropout_1'))
    model.add(BatchNormalization(name = 'batch_norm_1'))

    model.add(ConvLSTM2D(name ='conv_lstm_2',
                         filters = 64, kernel_size = (5, 1), 
                         padding='same',
                         return_sequences = False))
    
    model.add(Dropout(0.20, name = 'dropout_2'))
    model.add(BatchNormalization(name = 'batch_norm_2'))
    
    model.add(Flatten())
    model.add(RepeatVector(output_timesteps))
    model.add(Reshape((output_timesteps, num_links, 1, 64)))
    
    model.add(ConvLSTM2D(name ='conv_lstm_3',
                         filters = 64, kernel_size = (10, 1), 
                         padding='same',
                         return_sequences = True))
    
    model.add(Dropout(0.20, name = 'dropout_3'))
    model.add(BatchNormalization(name = 'batch_norm_3'))
    
    model.add(ConvLSTM2D(name ='conv_lstm_4',
                         filters = 64, kernel_size = (5, 1), 
                         padding='same',
                         return_sequences = True))
    
    model.add(TimeDistributed(Dense(units = 1, name = 'dense_1', activation = 'relu')))
    model.compile(loss = loss, optimizer = 'rmsprop')
    return model

    