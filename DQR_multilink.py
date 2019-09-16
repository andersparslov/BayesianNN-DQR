from load import split_df, sort_links
import models
from common import transform, fit_scale, roll, eval_quantiles, compute_error

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from datetime import timedelta

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, Activation, BatchNormalization, Flatten, Reshape

data = pd.read_csv('data/link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0)

start = data.index[ 0]
end   = data.index[-1]
period = (end - start).days
period_train_days = 100
period_test_days = 10
advance_days = 1
num_partitions = int( (period - period_train_days - period_test_days ) /advance_days)

## Sort links by order 
data, order = sort_links(data, '1416:1417', '7051:2056')
## Make a link order column e.g here the neighbouring links for link 1 are 0 and 2.
data['link_order'] = data['link_ref'].astype('category')
not_in_list = data['link_order'].cat.categories.difference(order)
data['link_order'] = data['link_order'].cat.set_categories(np.hstack((order, not_in_list)), ordered=True)
data['link_order'] = data['link_order'].cat.codes
## Add week of day column [Monday, ..., Sunday] = [0, ..., 6]
data['Weekday'] = data.index.weekday_name
data = data.sort_values('link_order')

lags = np.arange(6, 60, 8)
kernel_lengths = np.arange(2, len(data['link_ref'].unique()), 2)
rmse = np.zeros(( num_partitions, len(lags), len(kernel_lengths) ))
icp  = np.zeros(( num_partitions, len(lags), len(kernel_lengths) ))

for partition in range(num_partitions):
    print("Partition = ", partition)
    train_ind = np.arange(partition, period_train_days + partition)
    test_ind = np.arange(period_train_days + partition, period_train_days + partition + period_test_days)

    start_train = start.to_datetime() + timedelta(minutes = int(train_ind[0]*24*60))
    end_train   = start + timedelta(minutes = int(train_ind[-1]*24*60))
    end_test    = start + timedelta(minutes = int(test_ind[-1]*24*60))
    
    data_train, data_test = split_df(data.sort_index(), start_train , end_train, end_test)
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
    for l in range(len(lags)):
        lag = lags[l]
        print("Lag = ", lag)
        preds = 1
        X_train, y_train, y_ix_train, y_mean_train, y_std_train, y_num_meas_train = roll(ix_train, 
                                                                                 ts_train, 
                                                                                 rm_mean_train, 
                                                                                 rm_scale_train, 
                                                                                 w_train, 
                                                                                 lag, 
                                                                                 preds)
        X_test, y_test, y_ix_test, y_mean_test, y_std_test, y_num_meas_test = roll(ix_test, 
                                                                                   ts_test, 
                                                                                   rm_mean_test, 
                                                                                   rm_scale_test, 
                                                                                   w_test, 
                                                                                   lag, 
                                                                                   preds)

        X_train = X_train[:,:,:,np.newaxis,np.newaxis]
        y_train = y_train[:,:,:,np.newaxis,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis,np.newaxis]
        y_test = y_test[:,:,:,np.newaxis,np.newaxis]

        quantiles = np.array([0.05, 0.95])
        y_traink = np.zeros((y_train.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
        y_testk  = np.zeros((y_test.shape[0], y_train.shape[1], y_train.shape[2], 1, len(quantiles)+1))
        for k in range(len(quantiles)):
            y_traink[:,:,:,:,k] = y_train[:,:,:,:,0]
            y_testk[:,:,:,:,k] = y_test[:,:,:,:,0]
        
        for k in range(len(kernel_lengths)):
            print("k = ", k)
            kern = kernel_lengths[k]
            mod = models.joint_multilink(num_filters = 64, 
                                         kernel_length = kern,
                                         input_timesteps = X_train.shape[1],  
                                         num_links       = X_train.shape[2], 
                                         output_timesteps = y_train.shape[1], 
                                         quantiles = quantiles, 
                                         loss = lambda y, f: models.multi_tilted_loss2(quantiles, y, f))
            mod.fit(X_train, y_traink, validation_data = (X_test, y_testk), verbose = 0)
            y_pred = mod.predict(X_test)

            Y_true = y_test[:,:,:,0,0]* y_std_test + y_mean_test
            Y_naive = y_mean_test
            Y_pred_mean = (y_pred[:,:,:,0,0] * y_std_test) + y_mean_test
            Y_pred_lwr  = (y_pred[:,:,:,0,1] * y_std_test) + y_mean_test
            Y_pred_upr  = (y_pred[:,:,:,0,2] * y_std_test) + y_mean_test

            Y_true_total = np.sum(Y_true * y_num_meas_test, axis = 2)
            Y_naive_total = np.sum(Y_naive * y_num_meas_test, axis = 2)
            Y_pred_mean_total = np.sum(Y_pred_mean * y_num_meas_test, axis = 2)
            Y_pred_lwr_total = np.sum(Y_pred_lwr * y_num_meas_test, axis = 2)
            Y_pred_upr_total = np.sum(Y_pred_upr * y_num_meas_test, axis = 2)
            
            rmse[partition, k, l] = np.sqrt(np.mean((Y_pred_mean_total - Y_true_total)**2))
            icp[partition, k, l] = np.sum(np.logical_and( (Y_true_total > Y_pred_lwr_total),(Y_true_total < Y_pred_upr_total) )) / len(Y_true_total)
            
rmse_mean = np.mean(rmse, axis = 0)
icp_mean = np.mean(icp, axis = 0)

np.savetxt("data/rmse.csv", rmse_mean)
np.savetxt("data/icp.csv", icp_mean)