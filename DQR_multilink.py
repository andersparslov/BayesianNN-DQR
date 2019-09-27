from load import split_df, sort_links, skip_row, write_3d
import models
from common import transform, fit_scale, roll, eval_quantiles, compute_error

import pandas as pd
import numpy as np
from datetime import timedelta, datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, Activation, BatchNormalization, Flatten, Reshape

start = datetime.strptime('19/01/01', "%y/%m/%d")
end   = datetime.strptime('19/04/14', "%y/%m/%d")
period = (end - start).days
period_train_days = 7*9  ## Train on 9 weeks
period_test_days =  7*1  ## Test on  1 week
advance_days = 7         ## Advance by 1 week
num_partitions = int( (period - period_train_days - period_test_days ) /advance_days)

num_links = 16 

##filters = np.array([2**exp for exp in range(4, 7)]) ## [16, 32, 64]
alphas = np.arange(3, 10)
kernel_lengths = np.arange(3, num_links, 2) ## [3, 5, 7, 9, 11, 13, 15]
rmse = np.zeros(( num_partitions, len(kernel_lengths), len(alphas) ))
icp  = np.zeros(( num_partitions, len(kernel_lengths), len(alphas) ))
mil  = np.zeros(( num_partitions, len(kernel_lengths), len(alphas) ))

rmse[:] = np.nan
icp[:] = np.nan
mil[:] = np.nan

for partition in range(num_partitions):
	print("Partition", partition)
	train_ind = np.arange(partition*advance_days, period_train_days + partition*advance_days)
	test_ind = np.arange(period_train_days + partition*advance_days, period_train_days + partition*advance_days + period_test_days)
	keep_train = range(int(2297920*train_ind[0]/period), int(2297920*train_ind[-1]/period))
	keep_test  = range(int(2297920*train_ind[-1]/period) + 1, int(2297920*test_ind[-1]/period))
	data_train = pd.read_csv('data/link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0,
                         skiprows = lambda x: skip_row(x, keep_train))
	data_test  = pd.read_csv('data/link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0,
                         skiprows = lambda x: skip_row(x, keep_test))
	## Sort links by order 
	data_train, order_train = sort_links(data_train, '1973:1412', '7057:7058')
	## Make a link order column e.g here the neighbouring links for link 1 are 0 and 2.
	data_train['link_order'] = data_train['link_ref'].astype('category')
	not_in_list = data_train['link_order'].cat.categories.difference(order_train)
	data_train['link_order'] = data_train['link_order'].cat.set_categories(np.hstack((order_train, not_in_list)), ordered=True)
	data_train['link_order'] = data_train['link_order'].cat.codes
	## Add week of day column [Monday, ..., Sunday] = [0, ..., 6]
	data_train['Weekday'] = data_train.index.weekday_name
	data_train = data_train.sort_values('link_order')
	
	## Sort links by order 
	data_test, order_test = sort_links(data_test, '1973:1412', '7057:7058')
	## Make a link order column e.g here the neighbouring links for link 1 are 0 and 2.
	data_test['link_order'] = data_test['link_ref'].astype('category')
	not_in_list = data_test['link_order'].cat.categories.difference(order_test)
	data_test['link_order'] = data_test['link_order'].cat.set_categories(np.hstack((order_test, not_in_list)), ordered=True)
	data_test['link_order'] = data_test['link_order'].cat.codes
	## Add week of day column [Monday, ..., Sunday] = [0, ..., 6]
	data_test['Weekday'] = data_test.index.weekday_name
	data_test = data_test.sort_values('link_order')
	
	## Transform train and test set using the mean and std for train set.
	means_df, scales, low_df, upr_df = fit_scale(data_train, order_train)
	ix_train, ts_train,  rm_mean_train, rm_scale_train, w_train, lns_train = transform(data_train, 
                                                                                           means_df, 
                                                                                           scales, 
                                                                                           order_train,
                                                                                           freq = '15min')
	ix_test, ts_test, rm_mean_test, rm_scale_test, w_test, lns_test = transform(data_test, 
                                                                                     means_df, 
                                                                                     scales, 
                                                                                     order_test,
                                                                                     freq = '15min')
	lag = 26
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
		y_traink[:,:,:,:,k+1] = y_train[:,:,:,:,0]
		y_testk[:,:,:,:,k+1] = y_test[:,:,:,:,0]
	
	## Loop over filter size and kernel lengths and train and evaluate models
	for a in range(len(alphas)):
		alpha = alphas[a]
		filt = int(X_train.shape[0] / (alpha*(X_train.shape[1] + y_train.shape[1])))
		print(" Filter", filt)
		for k in range(len(kernel_lengths)):
			kern = kernel_lengths[k]
			print("      Kernel", kern)
			#################### INDEPENDENT MODEL FIT #################
			
			
			#################### JDQR MODEL FIT #####################
			
			mod = models.joint_multilink(num_filters = filt, 
										 kernel_length = kern,
										 input_timesteps = X_train.shape[1],  
										 num_links       = X_train.shape[2], 
										 output_timesteps = y_train.shape[1], 
										 quantiles = quantiles, 
										 loss = lambda y, f: models.multi_tilted_loss(quantiles, y, f))
			mod.fit(X_train, y_traink, validation_data = (X_test, y_testk), verbose = 0)
			
			###################### DJQR PREDICTIONS ########################
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
			
			rmse[partition, l, k, f] = np.sqrt(np.mean((Y_pred_mean_total - Y_true_total)**2))
			icp[partition,  l, k, f] = np.sum(np.logical_and( (Y_true_total > Y_pred_lwr_total),
															(Y_true_total <= Y_pred_upr_total) )) / len(Y_true_total)
			mil[partition,  l, k, f] = np.sum(np.maximum(0, Y_pred_upr_total - Y_pred_lwr_total)) / len(Y_true_total)
			
			print("         RMSE", rmse[partition, l, k, f])
			print("         ICP ", icp[partition, l, k, f])
			print("         MIL ", mil[partition, l, k, f])
			############################################################

write_3d(rmse, "output/rmse")
write_3d(icp, "output/icp")
write_3d(mil, "output/mil")


## Fit best models again and save them to data folder.




######################## RMSE MODEL ###################################
inds_rmse_min = np.argwhere(rmse_mean == np.amin(rmse_mean))[0]
lag = lags[inds_rmse_min[0]]
kernel = kernel_lengths[inds_rmse_min[1]]
filt = filters[inds_rmse_min[1]]


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

mod = models.joint_multilink(num_filters = filt, 
                             kernel_length = kernel,
                             input_timesteps = X_train.shape[1],  
                             num_links       = X_train.shape[2], 
                             output_timesteps = y_train.shape[1], 
                             quantiles = quantiles, 
                             loss = lambda y, f: models.multi_tilted_loss(quantiles, y, f))
mod.fit(X_train, y_traink, validation_data = (X_test, y_testk), verbose = 0)
## Save model
mod.save('models/mod_rmse.h5')

######################################################################






######################## ICP MODEL ###################################

## Closest to PI = 0.90
### REMEMBER TO CHANGE THIS WHEN QUANTILES CHANGE!!
inds_icp_max = np.argwhere(np.absolute((icp_mean-0.90)) == np.amin(np.absolute(icp_mean - 0.90) ))
lag = lags[inds_icp_max[0]]
kernel = kernel_lengths[inds_icp_max[1]]
filt = filters[inds_icp_max[2]]

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

mod = models.joint_multilink(num_filters = filt, 
                             kernel_length = kernel,
                             input_timesteps = X_train.shape[1],  
                             num_links       = X_train.shape[2], 
                             output_timesteps = y_train.shape[1], 
                             quantiles = quantiles, 
                             loss = lambda y, f: models.multi_tilted_loss(quantiles, y, f))
mod.fit(X_train, y_traink, validation_data = (X_test, y_testk), verbose = 0)
## Save model
mod.save('models/mod_icp.h5')

######################## MIL MODEL ###################################

inds_mil_min = np.argwhere(mil_mean == np.amin(mil_mean))[0]
lag = lags[inds_mil_min[0]]
kernel = kernel_lengths[inds_mil_min[1]]
filt = filters[inds_mil_min[2]]

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

mod = models.joint_multilink(num_filters = filt, 
                             kernel_length = kernel,
                             input_timesteps = X_train.shape[1],  
                             num_links       = X_train.shape[2], 
                             output_timesteps = y_train.shape[1], 
                             quantiles = quantiles, 
                             loss = lambda y, f: models.multi_tilted_loss(quantiles, y, f))
mod.fit(X_train, y_traink, validation_data = (X_test, y_testk), verbose = 0)
## Save model
mod.save('models/mod_mil.h5')
