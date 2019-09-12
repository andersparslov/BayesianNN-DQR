import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed

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

## Here quantiles is array with multiple quantile values
## y is (N, J+1) array where J is number of quantiles.
def multi_tilted_loss(quantiles, y, f):
    mean_loss = K.mean(K.square(y[:,0]-f[:,0]))
    quan_loss = 0
    for j in range(len(quantiles)):
        q = quantiles[j]
        quan_loss += K.mean(K.maximum(q*(y[:,j+1]-f[:,j+1]), (q-1)*(y[:,j+1]-f[:,j+1])), axis=-1)
    return mean_loss + quan_loss

def multi_tilted_loss2(quantiles,y,f):
    loss = K.mean(K.square(y[:,0]-f[:,0]), axis=-1)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k+1]-f[:,k+1])
        loss += K.mean(K.maximum(q*e, (q-1)*e))
    return loss

def tilted_loss(q, y, f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def build_model(num_nodes, input_dim, output_dim, loss):
    mod = Sequential()
    mod.add(LSTM(100, activation='relu', input_shape=(input_dim, 1)))
    mod.add(Dropout(0.2))
    mod.add(RepeatVector(output_dim))
    mod.add(LSTM(100, activation='relu', return_sequences=True))
    mod.add(Dropout(0.2))
    mod.add(TimeDistributed(Dense(output_dim)))
    mod.compile(loss=loss, optimizer='nadam')
    return mod