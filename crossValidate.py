import load
import models

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline 
import numpy as np
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed

print("Testing")

## Load data (just one link)
data = load.traveltimes_onelink(start_date = '2019-01-01', end_date = '2019-04-07')
start = data.index[0]
end   = data.index[len(data)-1]
period = (end - start).days
period_train_days = 12*7
period_test_days = 7
advance_days = 1
num_partitions = int( (period - period_train_days - period_test_days ) /advance_days)
print("Training over", period_train_days, "days and testing on", period_test_days, "days.")
print(num_partitions, "partitions are trained over.")

time_res_mins = 15
preds = 1
alphas = np.arange(2, 9)
num_lags = 4
rmse  = np.zeros((num_partitions, num_lags, len(alphas)))
icp  = np.zeros((num_partitions, num_lags, len(alphas)))
time_res_mins = 15
preds = 1

for partition in range(num_partitions):
    rmse_best = 1000000
    print("PARTITION ", partition + 1)
    for lag_ind in range(num_lags):
        lag = 12 + 3*lag_ind
        train_ind = np.arange(partition, period_train_days + partition)
        test_ind = np.arange(period_train_days + partition, period_train_days + partition + period_test_days)
        
        X, y, times = load.timeseries(data, lag, preds, time_res_mins)
        X_train, y_train, X_test, y_test, times_train, times_test = load.split_time2(X, y, times, train_ind, test_ind)
        
        for a in range(len(alphas)):
            print("   LAG = ", lag)
            print("       Alpha = ", alphas[a])
            ## Reformat y to have (J+1) columns
            quantiles = np.array([0.05, 0.50, 0.95])
            y_traink = y_train
            for k in range(len(quantiles)):
                y_traink = np.concatenate((y_traink, y_train), axis=1)
                
            num_nodes = int(X_train.shape[0] / (alphas[a]*(X_train.shape[1]+ y_train.shape[1])))
            mod = models.build_model(num_nodes, 
                                     input_dim = X_train.shape[1], 
                                     output_dim = 1 + len(quantiles), 
                                     loss = lambda y, f: models.multi_tilted_loss(quantiles, y, f))
            
            mod.fit(X_train, y_traink)
            y_pred = mod.predict(X_test)[:,:,0] ## Ask Filipe why we take just one column out of the four
            y_pred_mean = y_pred[:,0]
            y_pred_lwr  = y_pred[:,1]
            y_pred_med  = y_pred[:,2]
            y_pred_upr  = y_pred[:,3]
            
            rmse[partition, lag_ind, a] = np.sqrt(np.mean((y_pred_mean - y_test[:,:,0])**2))
                
            icp[partition, lag_ind, a] = np.sum(np.logical_and( (y_test[:,0,0] > y_pred_lwr),(y_test[:,0,0] < y_pred_upr) )) / y_test.shape[0]
            print("Model error (RMSE) = ", "%.2f" % rmse[partition, lag_ind, a], "seconds")
            if(rmse[partition, lag_ind, a] < rmse_best):
                print("    --- > BEST SO FAR ON THIS PARTITION!")
                rmse_best = rmse[partition, lag_ind, a]
            print("Perecentage in 90% Prediction Interval (ICP) = ", "%.3f" % icp[partition, lag_ind, a])

np.savetext("data/rmse.csv", rmse, delimiter = ",")
np.savetext("data/icp.csv", icp, delimiter = ",")
