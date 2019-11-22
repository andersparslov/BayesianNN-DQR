
# coding: utf-8

# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd


# In[1]:


def write_3d(X, filename):
    X_list = X.tolist()
    with open(filename+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(X_list)

def transform(data, means_df, scales_df, order, freq = '15min'):
  tss = { }
  ws = { }
  removed_mean = { }
  removed_scale = { }
  lnk_list = []
  for lnk, data_link in data.groupby('link_ref', sort = False):
      # Link Data Time Indexed
      link_time_ix = pd.DatetimeIndex(data_link.index)
      data_link = data_link.set_index(link_time_ix)
      # Link Reference Data Index
      ix_week = data_link['Weekday'].tolist()
      ix_tod = data_link['TOD'].tolist()
      ## Create multi index for the two lists
      mult_ind = pd.MultiIndex.from_arrays([ix_week, ix_tod])

      link_travel_time_k = data_link['link_travel_time'].resample(freq).mean()
      removed_mean[lnk] = pd.Series(data=means_df[lnk].loc[mult_ind].values, 
                                    index = link_time_ix).resample(freq).mean()
      removed_scale[lnk] = pd.Series(data =scales_df[lnk].loc[mult_ind].values, 
                                      index = link_time_ix).resample(freq).mean()
      tss[lnk] = (link_travel_time_k - removed_mean[lnk].values) / removed_scale[lnk].values
      ws[lnk] = data_link['link_travel_time'].resample(freq).count()
      lnk_list.append(lnk)

  ts = pd.DataFrame(data = tss).fillna(method='pad').fillna(0) 
  df_removed_mean = pd.DataFrame(data = removed_mean, index = ts.index).fillna(method='pad').fillna(method='bfill') 
  df_removed_scale = pd.DataFrame(data = removed_scale, index = ts.index).fillna(method='pad').fillna(method='bfill')    
  w = pd.DataFrame(data = ws).fillna(0) # Link Travel Time Weights, e.g. number of measurements
  return ts[order], df_removed_mean[order], df_removed_scale[order]

def fit_scale(data, order, ref_freq = '15min'):
  means = { }
  scales = { }
  low = { }
  upr = { }

  grouping = data[data['link_travel_time'].notnull()].groupby('link_ref', sort = False)
  for link_ref, data_link in grouping:
      # Fit outlier bounds using MAD
      median = data_link.groupby('Weekday')['link_travel_time'].median()
      error = pd.concat([data_link['Weekday'], np.abs(data_link['link_travel_time'] - median[data_link['Weekday']].values)], axis = 1)
      mad = 1.4826 * error.groupby('Weekday')['link_travel_time'].median()

      _low = median - 3 * mad
      _upr = median + 3 * mad
      mask = (_low[data_link['Weekday']].values < data_link['link_travel_time']) & (data_link['link_travel_time'] < _upr[data_link['Weekday']].values)
      data_link_no = data_link[mask]

      _mean = data_link_no.groupby(['Weekday', 'TOD'])['link_travel_time'].mean()
      means[link_ref] = _mean
      scale = data_link_no.groupby(['Weekday', 'TOD'])['link_travel_time'].std()
      scales[link_ref] = scale

      low[link_ref] = _low
      upr[link_ref] = _upr

  means_df = pd.DataFrame(data=means).interpolate()
  scales_df = pd.DataFrame(data=scales).interpolate()
  low_df = pd.DataFrame(data=low).interpolate()
  upr_df = pd.DataFrame(data=upr).interpolate()

  ## Correct order of links
  means_df = means_df[order]
  scales_df = scales_df[order]
  low_df = low_df[order]
  upr_df = upr_df[order]

  # Fill NaNs    
  means_df = means_df.fillna(method='pad').fillna(method='bfill')
  scales_df = scales_df.fillna(method='pad').fillna(method='bfill')
  low_df = low_df.fillna(method='pad').fillna(method='bfill')
  upr_df = upr_df.fillna(method='pad').fillna(method='bfill')
  
  return means_df, scales_df

def roll(ix, ts, removed_mean, removed_scale, lags, preds):
  X = np.stack([np.roll(ts, i, axis = 0) for i in range(lags, 0, -1)], axis = 1)[lags:-preds,]
  Y = np.stack([np.roll(ts, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
  Y_ix = ix[lags:-preds]
  Y_mean = np.stack([np.roll(removed_mean, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
  Y_scale = np.stack([np.roll(removed_scale, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
  return X, Y, Y_ix, Y_mean, Y_scale

def sort_links(data, start_link, end_link):
  ordered_list = [start_link]
  links = data['link_ref'].unique()
  stop_end = start_link.rpartition(':')[2]
  
  while True:
      stop_start = stop_end
      for lnk in links:
          if(lnk.rpartition(':')[0] == stop_start):
              if( (lnk in ordered_list) or (lnk == end_link) ):
                  break
              else:
                  ordered_list.append(lnk)
                  stop_end = lnk.rpartition(':')[2]
      if(stop_start == stop_end):
          break
  ordered_list.append(end_link)
  ## Only include links in ordered list.
  data = data[data['link_ref'].isin(ordered_list)]
  return data, ordered_list

def tod_interval(x):
    if(x < 2):
        return 0
    elif(x < 4):
        return 1
    elif(x < 6):
        return 2
    elif(x < 8):
        return 3
    elif(x < 10):
        return 4
    elif(x < 12):
        return 5
    elif(x < 14):
        return 6
    elif(x < 16):
        return 7
    elif(x < 18):
        return 8
    elif(x < 20):
        return 9
    elif(x < 22):
        return 10
    elif(x < 24):
        return 11

def split_df(data, start_train, end_train, end_test):
    data_train = data[start_train:end_train]
    data_test = data[end_train:end_test]
    return data_train, data_test


# In[4]:


def load_data(lags, start_train, end_train, end_test):
  preds = 1
  data = pd.read_csv('data/link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0)

  ## Sort links by order 
  data, order = sort_links(data, '1973:1412', '7057:7058')
  ## Make a link order column e.g here the neighbouring links for link 1 are 0 and 2.
  data['link_order'] = data['link_ref'].astype('category')
  not_in_list = data['link_order'].cat.categories.difference(order)
  data['link_order'] = data['link_order'].cat.set_categories(np.hstack((order, not_in_list)), ordered=True)
  data['link_order'] = data['link_order'].cat.codes
  ## Add week of day column [Monday, ..., Sunday] = [0, ..., 6]
  data['Weekday'] = data.index.weekday
  ## Add hour of the time to dataframe
  data['Hour'] = data.index.hour
  ## Add time of day variables to data frame
  data['TOD'] = data.Hour.apply(tod_interval)
  data = data.sort_values('link_order')
  data_train, data_test = split_df(data, start_train = start_train, end_train = end_train, end_test = end_test)

  ## Transform train and test set using the mean and std for train set.
  means_df_train, scales_df_train = fit_scale(data_train, order)
  ts_train_df, mean_train_df, scale_train_df = transform(data_train, 
                                                    means_df_train, 
                                                    scales_df_train, 
                                                    order,
                                                    freq = '15min')
  ts_test_df, mean_test_df, scale_test_df = transform(data_test, 
                                                  means_df_train, 
                                                  scales_df_train, 
                                                  order,
                                                  freq = '15min')
  return ts_train_df, mean_train_df, scale_train_df, ts_test_df, mean_test_df, scale_test_df


# In[5]:


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, Activation, BatchNormalization, Flatten, Reshape
sum_all = tf.math.reduce_sum

def tilted_loss(q,y,f):
    e = (y[:,:,:,0,0]-f[:,:,:,0,0])
    # The term inside k.mean is a one line simplification of the first equation
    return K.mean(K.maximum(q*e, (q-1)*e))

def mse_loss(y, f):
	return K.mean(K.square(y[:,:,:,0,0]-f[:,:,:,0,0]), axis = -1)

## Tilted loss for both mean and quantiles
def joint_tilted_loss(quantiles, y, f):
	loss = K.mean(K.square(y[:,:,:,0,0]-f[:,:,:,0,0]), axis = -1)
	for i in range(len(quantiles)):
		q = quantiles[i]
		e = (y[:,:,:,0,i+1]-f[:,:,:,0,i+1])
		loss += K.mean(K.maximum(q*e, (q-1)*e))
	return loss

## Encoder-decoder convolutional LSTM for jointly estimating quantiles and mean predictions.
def joint_convLstm(num_filters, kernel_length, input_timesteps, num_links, output_timesteps, quantiles, loss):
	model = Sequential()
	model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links, 1, 1)))
	model.add(ConvLSTM2D(name ='conv_lstm_0',
                         filters = num_filters, kernel_size = (kernel_length, 1), 
                         padding='same',
                         return_sequences = False))

	model.add(Dropout(0.20, name = 'dropout_0'))
	model.add(BatchNormalization(name = 'batch_norm_1'))

	model.add(Flatten())
	model.add(RepeatVector(output_timesteps))
	model.add(Reshape((output_timesteps, num_links, 1, num_filters)))

	model.add(ConvLSTM2D(name ='conv_lstm_1',filters = num_filters, kernel_size = (kernel_length, 1), padding='same',return_sequences = True))
	model.add(Dropout(0.20, name = 'dropout_1'))

	model.add(TimeDistributed(Dense(units = len(quantiles) + 1, name = 'dense_1')))
	model.compile(loss = loss, optimizer = 'nadam')
	return model

## Encoder-decoder LSTM for mean 
def convLstm(num_filters, kernel_length, input_timesteps, num_links, output_timesteps, loss):
	model = Sequential()
	model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links, 1, 1)))
	model.add(ConvLSTM2D(name ='conv_lstm_0',
													filters = num_filters, kernel_size = (kernel_length, 1), 
													padding='same',
													return_sequences = False))

	model.add(Dropout(0.20, name = 'dropout_0'))
	model.add(BatchNormalization(name = 'batch_norm_1'))

	model.add(Flatten())
	model.add(RepeatVector(output_timesteps))
	model.add(Reshape((output_timesteps, num_links, 1, num_filters)))

	model.add(ConvLSTM2D(name ='conv_lstm_1',
													filters = num_filters, kernel_size = (kernel_length, 1), 
													padding='same',
													return_sequences = True))
	model.add(Dropout(0.20, name = 'dropout_1'))

	model.add(TimeDistributed(Dense(units = 1, name = 'dense_1')))
	model.compile(loss = loss, optimizer = 'nadam')
	return model


# In[ ]:


lags = 10
preds = 1

start_train_lst = ['2019-01-01', '2019-01-07', '2019-01-14', '2019-01-21', '2019-02-01']
end_train_lst = ['2019-01-31', '2019-02-07', '2019-02-14', '2019-02-21', '2019-03-01']
end_test_lst = ['2019-02-07', '2019-02-14', '2019-02-21', '2019-03-01', '2019-03-07']
    
num_partitions = 3 
num_links = 16
batch_size = 80

quantiles = np.array([0.05, 0.95])
units_lst = np.arange(6, 72, 12)
kernel_lengths = np.arange(3, 14, 2)

epochs = 200
patience = 5

mse_i =  np.empty((num_partitions, len(units_lst), len(kernel_lengths)))
icp_i =  np.empty((num_partitions, len(units_lst), len(kernel_lengths)))
mil_i =  np.empty((num_partitions, len(units_lst), len(kernel_lengths)))

mse_j =  np.empty((num_partitions, len(units_lst), len(kernel_lengths)))
icp_j =  np.empty((num_partitions, len(units_lst), len(kernel_lengths)))
mil_j =  np.empty((num_partitions, len(units_lst), len(kernel_lengths)))

time_mean = np.empty((num_partitions, len(units_lst), len(kernel_lengths)))
time_quan = np.empty((num_partitions, len(units_lst), len(kernel_lengths)))
time_joint = np.empty((num_partitions, len(units_lst), len(kernel_lengths)))
  
for u, units in enumerate(units_lst):
  print("Units {}".format(units))
  for k, kern in enumerate(kernel_lengths):
    print("Kernel length {}".format(kern))
    ## Initialise models
    ## Model for mean predictions
    mod_mean = convLstm(units,kern,lags, num_links,1, loss = lambda y, f: mse_loss(y,f))
		
		## Model for quantiles
    mod_quan = []
    for q, quan in enumerate(quantiles):
      mod_quan.append(convLstm(units,kern,lags,  num_links, 1, loss = lambda y, f: tilted_loss(quan, y, f)))
		## Joint model
    mod_joint = joint_convLstm(units, kern,lags, num_links, 1, quantiles, loss = lambda y, f: joint_tilted_loss(quantiles, y, f))

    for part in range(num_partitions):
      start_train = start_train_lst[part]
      end_train = end_train_lst[part]
      end_test = end_test_lst[part]
      ts_train_df, mean_train_df, scale_train_df, ts_test_df, mean_test_df, scale_test_df = load_data(lags, start_train, end_train, end_test)

      X_train, y_train, y_ix_train, y_mean_train, y_std_train = roll(ts_train_df.index, 
                                                                      ts_train_df.values,
                                                                      mean_train_df.values,
                                                                      scale_train_df.values,
                                                                      lags, 
                                                                      preds)
      X_test, y_test, y_ix_test, y_mean_test, y_std_test = roll(ts_test_df.index, 
                                                                ts_test_df.values, 
                                                                mean_test_df.values,
                                                                scale_test_df.values,
                                                                lags, 
                                                                preds)
      X_train = X_train[:,:,:,np.newaxis,np.newaxis]
      y_train = y_train[:,:,:,np.newaxis,np.newaxis]
      X_test = X_test[:,:,:,np.newaxis,np.newaxis]
      y_test = y_test[:,:,:,np.newaxis,np.newaxis]
      
      y_traink = np.zeros((y_train.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
      y_testk  = np.zeros((y_test.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
      for i in range(len(quantiles)):
        y_traink[:,:,:,:,i+1] = y_train[:,:,:,:,0]
        y_testk[:,:,:,:,i+1] = y_test[:,:,:,:,0]
				
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=8)
      check_mean = ModelCheckpoint('models/mean_weights_a{}_k{}_p{}.hdf5'.format(units, kern, part), monitor='val_loss', mode='min', save_best_only=True)
      check_q = []
      for q, quan in enumerate(quantiles):
        check_q.append(ModelCheckpoint('models/q{}_weights_a{}_k{}_p{}.hdf5'.format(q, units, kern, part), monitor='val_loss', mode='min', save_best_only=True))
      check_joint = ModelCheckpoint('models/joint_weights_a{}_k{}_p{}.hdf5'.format(units, kern, part), monitor='val_loss', mode='min', save_best_only=True)
			
	  ## If not the first partition initialise weights from last partition
      if part != 0:
        mod_mean.load_weights('models/mean_weights_a{}_k{}_p{}.hdf5'.format(units, kern, part-1))
        for q, mod in enumerate(mod_quan):
          mod.load_weights('models/q{}_weights_a{}_k{}_p{}.hdf5'.format(q, units, kern, part-1))
        mod_joint.load_weights('models/joint_weights_a{}_k{}_p{}.hdf5'.format(units, kern, part-1))
			################### INDEPENDENT ############################
      t1 = datetime.now()
      mod_mean.fit(X_train, y_train, epochs = epochs,validation_data = (X_test, y_test),batch_size = batch_size, callbacks = [es, check_mean],verbose=0)
      mod_mean.load_weights('models/mean_weights_a{}_k{}_p{}.hdf5'.format(units, kern, part))
      t2 = datetime.now()
      time_mean[part, u, k] = (t2-t1).seconds
      y_pred = mod_mean.predict(X_test)
      y_pred_q = []
      for q, mod in enumerate(mod_quan):
        t1 = datetime.now()
        mod.fit(X_train, y_train, epochs = epochs,validation_data = (X_test, y_test),batch_size=batch_size,callbacks=[es, check_q[q]],verbose = 0)
        mod.load_weights('models/q{}_weights_a{}_k{}_p{}.hdf5'.format(q, units, kern, part))
        y_pred_q.append(mod.predict(X_test))
      t2 = datetime.now()
      time_quan[part, u, k] = (t2-t1).seconds

      Y_pred_lwr  = (y_pred_q[0][:,:,:,0,0] * y_std_test) + y_mean_test
      Y_pred_upr  = (y_pred_q[1][:,:,:,0,0] * y_std_test) + y_mean_test
      Y_pred_mean = (y_pred[:,:,:,0,0] * y_std_test) + y_mean_test
      Y_true = y_test[:,:,:,0,0]* y_std_test + y_mean_test
			
      Y_true_total = np.sum(Y_true, axis = 2)
      Y_pred_mean_total = np.sum(Y_pred_mean, axis = 2)
        
      icp_lnks = np.zeros(num_links)
      mil_lnks = np.zeros(num_links)  
      for lnk in range(num_links):
            q1 = Y_pred_lwr[:,:,lnk]
            q2 = Y_pred_upr[:,:,lnk]
            icp_lnks[lnk] = 1 - (np.sum(y_test[:,:,lnk] < q1) + np.sum(y_test[:,:,lnk] > q2) )/len(y_test)
            mil_lnks[lnk] = np.sum(np.maximum(0, q2 - q1)) / len(y_test)
      icp_i[part, u, k] = np.mean(icp_lnks)
      mil_i[part, u, k] = np.mean(mil_lnks)
      mse_i[part,  u, k] = np.sum((Y_pred_mean_total - Y_true_total)**2)/len(Y_true_total)
							
			
			#################### JDQR MODEL #####################
      t1 = datetime.now()
      mod_joint.fit(X_train,y_traink, epochs = epochs,validation_data = (X_test, y_testk),batch_size = batch_size,callbacks=[es, check_joint],verbose=0)
      mod_joint.load_weights('models/joint_weights_a{}_k{}_p{}.hdf5'.format(units, kern, part))
      t2 = datetime.now()
      time_joint[part, u, k] = (t2-t1).seconds
      y_pred = mod_joint.predict(X_test)
      Y_true = y_test[:,:,:,0,0]* y_std_test + y_mean_test
      Y_pred_mean = (y_pred[:,:,:,0,0] * y_std_test) + y_mean_test
      Y_pred_lwr  = (y_pred[:,:,:,0,1] * y_std_test) + y_mean_test
      Y_pred_upr  = (y_pred[:,:,:,0,2] * y_std_test) + y_mean_test
		
      Y_true_total = np.sum(Y_true , axis = 2)
      Y_pred_mean_total = np.sum(Y_pred_mean, axis = 2)
	  
      icp_lnks = np.zeros(num_links)
      mil_lnks = np.zeros(num_links)  
      for lnk in range(num_links):
            q1 = Y_pred_lwr[:,:,lnk]
            q2 = Y_pred_upr[:,:,lnk]
            icp_lnks[lnk] = 1 - (np.sum(y_test[:,:,lnk] < q1) + np.sum(y_test[:,:,lnk] > q2) )/len(y_test)
            mil_lnks[lnk] = np.sum(np.maximum(0, q2 - q1)) / len(y_test)
      icp_j[part, u, k] = np.mean(icp_lnks)
      mil_j[part, u, k] = np.mean(mil_lnks)
      mse_j[part, u, k] = np.sum((Y_pred_mean_total - Y_true_total)**2)/len(Y_true_total)
      
			
      write_3d(time_mean, "time_mean")
      write_3d(time_quan, "time_quan")
      write_3d(time_joint, "time_joint")
      write_3d(mse_j, "mse_j")
      write_3d(icp_j, "icp_j")
      write_3d(mil_j, "mil_j")
      write_3d(mse_i, "mse_i")
      write_3d(icp_i, "icp_i")
      write_3d(mil_i, "mil_i")

