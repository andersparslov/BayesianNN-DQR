from load import split_df, sort_links
from models import convLstm, convLstm_quan, joint_convLstm, joint_tilted_loss, tilted_loss, mse_loss
from common import transform, fit_scale, roll, eval_quantiles, compute_error
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, Activation, BatchNormalization, Flatten, Reshape

lags = 16
preds = 1
path = 'data/link_travel_time_local.csv.gz'
data = pd.read_csv(path, compression='gzip', parse_dates = True, index_col = 0)

## Sort links by order 
data, order = sort_links(data, '1973:1412', '7057:7058')
## Make a link order column e.g here the neighbouring links for link 1 are 0 and 2.
data['link_order'] = data['link_ref'].astype('category')
not_in_list = data['link_order'].cat.categories.difference(order)
data['link_order'] = data['link_order'].cat.set_categories(np.hstack((order, not_in_list)), ordered=True)
data['link_order'] = data['link_order'].cat.codes
## Add week of day column [Monday, ..., Sunday] = [0, ..., 6]
data['Weekday'] = data.index.weekday_name
data = data.sort_values('link_order')

#data_train, data_test = split_df(data, start_train = '2019-01-01', end_train = '2019-03-21', end_test = '2019-03-28')
data_train, data_test = split_df(data, start_train = '2019-01-01', end_train = '2019-01-31', end_test = '2019-02-03')

## Transform train and test set using the mean and std for train set.
means_df, scales, low_df, upr_df = fit_scale(data_train, order)
ix_train, ts_train,  rm_mean_train, rm_scale_train, w_train, lns_train = transform(data_train, 
																					   means_df, 
																					   scales, 
																					   order,
																					   freq = '15min')
ix_test, ts_test, rm_mean_test, rm_scale_test, w_test, lns_test = transform(data_test, 
																				 means_df, 
																				 scales, 
																				 order,
																				 freq = '15min')
## Create rolling window tensor
##  - y_mean and y_std are arrays where columns are each link and 
##    the rows corresponding to the mean and std of each data point
##    at that weekday. 

##  - y_num_meas indicates how many measurements are in the time window
##    for a given link
X_train, y_train, y_ix_train, y_mean_train, y_std_train, y_num_meas_train = roll(ix_train, 
																				 ts_train, 
																				 rm_mean_train, 
																				 rm_scale_train, 
																				 w_train, 
																				 lags, 
																				 preds)
X_test, y_test, y_ix_test, y_mean_test, y_std_test, y_num_meas_test = roll(ix_test, 
																		   ts_test, 
																		   rm_mean_test, 
																		   rm_scale_test, 
																		   w_test, 
																		   lags, 
																		   preds)
X_train = X_train[:,:,:,np.newaxis,np.newaxis]
y_train = y_train[:,:,:,np.newaxis,np.newaxis]
X_test = X_test[:,:,:,np.newaxis,np.newaxis]
y_test = y_test[:,:,:,np.newaxis,np.newaxis]

quantiles = np.array([0.05, 0.95])
y_traink = np.zeros((y_train.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
y_testk  = np.zeros((y_test.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
y_traink[:,:,:,:,0] = y_train[:,:,:,:,0]
y_testk[:,:,:,:,0] = y_test[:,:,:,:,0]
for k in range(len(quantiles)):
    y_traink[:,:,:,:,k+1] = y_train[:,:,:,:,0]
    y_testk[:,:,:,:,k+1] = y_test[:,:,:,:,0]


n_epochs = 400
batch_size = 100
num_units = 20
kernel_length = 8

mod_mean = convLstm(num_units, 
                    kernel_length, 
                    X_train.shape[1], 
                    X_train.shape[2], 
                    y_train.shape[1], 
                    loss = lambda y, f: mse_loss(y,f))
history_mean = mod_mean.fit(X_train, 
             y_train, 
             epochs = n_epochs,
             validation_data = (X_test, y_test),
             batch_size = batch_size, verbose = 0)
print("   ", np.array(history_mean.history["loss"]))

## This list holds on to the training histories
histories_q = []
for q in quantiles:    
    mod_quan = convLstm(num_units, 
                        kernel_length, 
                        X_train.shape[1], 
                        X_train.shape[2], 
                        y_train.shape[1], 
                        loss = lambda y, f: tilted_loss(q, y, f))
    history_q = mod_quan.fit(X_train, 
                 y_train, 
                 epochs = n_epochs,
                 validation_data = (X_test, y_test),
                 batch_size = batch_size, verbose = 0)
    histories_q.append(history_q)
    
mod_joint  = joint_convLstm(num_units, 
                            kernel_length, 
                            X_train.shape[1], 
                            X_train.shape[2], 
                            y_train.shape[1], 
                            loss = lambda y, f: joint_tilted_loss(quantiles, y, f),
                            quantiles = quantiles)

history_joint = mod_joint.fit(X_train, 
                                y_traink, 
                                epochs = n_epochs,
                                validation_data = (X_test, y_testk),
                                batch_size = batch_size, verbose = 0)
    
ind_train = np.array([history_mean.history["loss"]] + [h.history["loss"] for h in histories_q]).sum(axis=0)
jnt_train = np.array(history_joint.history["loss"])
ind_test  = np.array([history_mean.history["val_loss"]] + [h.history["val_loss"] for h in histories_q]).sum(axis=0)
jnt_test  = np.array(history_joint.history["val_loss"])

np.savetxt("ind_train", ind_train)
np.savetxt("ind_test", ind_test)
np.savetxt("jnt_train", jnt_train)
np.savetxt("jnt_test", jnt_test)
