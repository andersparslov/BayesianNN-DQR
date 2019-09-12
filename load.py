##### LOADS DATA AND REDUCES TO ONE DAY FOR ONE LINK   #####
##### CONVERTS DATA TO [SAMPLES, TIME_STEPS, FEATURES] #####

import pandas as pd
import numpy as np
import statistics as stats
from math import isnan
import time
from datetime import datetime, timedelta

def timeseries(traveltimes, lags, preds, time_res_mins):
    start = traveltimes.index[0]
    end   = traveltimes.index[len(traveltimes)-1]
    time_minutes = ((end - start).days*24*60*60 + (end - start).seconds) / 60
    
    ## Number of windows in period to consider
    total_windows = int(time_minutes / time_res_mins)
    ## Number of samples is total_windows minus the lag and predictions
    N = total_windows - lags - preds
    T  = np.zeros(N + lags + preds)
    mean_times = np.zeros((N + lags + preds), dtype='datetime64[s]')
    X  = np.zeros((N, lags + 1))
    y  = np.zeros((N, preds))
    
    ## Initially loop over lags
    for l in range(lags):
        time = start + timedelta(minutes = (l+1)*time_res_mins)
        mean_times[l] = time
        ## Travel times that are in current window are averaged and saved.
        T[l]  = np.mean(traveltimes[  (traveltimes.index <= time) 
                                       & (traveltimes.index >= time - timedelta(minutes = time_res_mins)) ])
        
    ## Then loop over remaining time-steps
    for t in range(lags, N + preds + lags):
        time = start + timedelta(minutes = (t + 1)*time_res_mins)
        mean_times[t] = time
        
        ## Mean travel time in the window [time - time_res_mins, time]
        T[t] = np.mean(traveltimes[ (traveltimes.index < time) & 
                                       (traveltimes.index > time - timedelta(minutes = time_res_mins)) ])
    
    ## Linear interpolation for missing values
    nan_ind, non = np.isnan(T), lambda z: z.nonzero()[0]
    T[nan_ind] = np.interp(non(nan_ind), non(~nan_ind), T[~nan_ind])
    ## Convert times to exclude initial lag times and last predictions
    mean_times = mean_times[lags:(total_windows-preds)]
    
    for t in range(N):
        X[t, lags] = T[t + lags]
        for p in range(preds):
            y[t, p] = T[t + lags + p + 1]
        for l in range(lags):
            X[t, lags - l - 1] = T[t + lags - (l+1)]
        #print("Time steps = ", X[t], " outcome = ", y[t])
    return X, y, mean_times

## Use this to convert data from multiple links into timeseries format
## Input: traveltimes [SAMPLES, LINKS]
## Output: X [N, LAGS, LINKS]
##         y [N, PREDS, LINKS]
def timeseries_all(traveltimes, lags, preds, time_res_mins):
    start = traveltimes.index[0]
    end   = traveltimes.index[len(traveltimes)-1]
    time_minutes = ((end - start).days*24*60*60 + (end - start).seconds) / 60
    
    num_links = traveltimes.shape[1]
    ## Number of windows in period to consider
    total_windows = int(time_minutes / time_res_mins)
    ## Number of samples is total_windows minus the lag and predictions
    N = total_windows - lags - preds
    T  = np.zeros( (N + lags + preds, num_links) )
    mean_times = np.zeros((N + lags + preds), dtype='datetime64[s]')
    X  = np.zeros((N, lags + 1, num_links))
    y  = np.zeros((N, preds, num_links))
    
    ## Initially loop over lags
    for l in range(lags):
        time = start + timedelta(minutes = (l+1)*time_res_mins)
        mean_times[l] = time
        ## For each link travel times that are in current window are averaged and saved.
        for lnk in range(num_links):
            T[l, lnk]  = np.mean(traveltimes[(traveltimes.index <= time) 
                                       & (traveltimes.index >= time - timedelta(minutes = time_res_mins)), lnk])
        
    ## Then loop over remaining time-steps
    for t in range(lags, N + preds + lags):
        time = start + timedelta(minutes = (t + 1)*time_res_mins)
        mean_times[t] = time
        
        ## Mean travel time in the window [time - time_res_mins, time]
        for lnk in range(num_links):
            T[t, lnk] = np.mean(traveltimes[ (traveltimes.index < time) & 
                                       (traveltimes.index > time - timedelta(minutes = time_res_mins)), lnk])
    
    ## For each link fill in missing values using linear interpolation
    for lnk in range(num_links):
        nan_ind, non = np.isnan(T[:,lnk]), lambda z: z.nonzero()[0]
        T[nan_ind, lnk] = np.interp(non(nan_ind), non(~nan_ind), T[~nan_ind, lnk])
    
    ## Convert times to exclude initial lag times and last predictions
    mean_times = mean_times[lags:(total_windows-preds)]

    for t in range(N):
        X[t, lags] = T[t + lags]
        for p in range(preds):
            y[t, p] = T[t + lags + p + 1]
        for l in range(lags):
            X[t, lags - l - 1] = T[t + lags - (l+1)]
        #print("Time steps = ", X[t], " outcome = ", y[t])
    return X, y, mean_times

def split(X, y, times, split_pct):
    ## Convert data to [samples, time steps, features (1 link)]
    #X, y, times, total = slide(data, lags, preds, time_res)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1], 1))
    
    ## Split into train and test
    split = int(split_pct*len(X))
    X_train = X[:split]
    X_test  = X[split:]
    y_train = y[:split]
    y_test  = y[split:]
    times_train = times[:split]
    times_test  = times[split:]
    
    return X_train, X_test, y_train, y_test, times_train, times_test

def split_time(data_timeseries, train_days_ind, test_days_ind):
    start = data_timeseries.index[0]
    start_train = start + timedelta(minutes = int(train_days_ind[0]*24*60))
    end_train = start + timedelta(minutes = int(train_days_ind[-1]*24*60))
    start_test = start + timedelta(minutes = int(test_days_ind[0]*24*60))
    end_test = start + timedelta(minutes = int(test_days_ind[-1]*24*60))
    ## Partition data into training and test based on specified times.
    dat_train = data_timeseries[(data_timeseries.index > start_train) & (data_timeseries.index < end_train)]
    dat_test = data_timeseries[(data_timeseries.index > start_test) & (data_timeseries.index < start_test)]
    return dat_train, dat_test

def split_time2(X, y, times, train_days_ind, test_days_ind):
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1], 1))
    start = times[0]
    start_train = start + train_days_ind[0]*24*60*60
    end_train = start + train_days_ind[-1]*24*60*60
    end_test = start + test_days_ind[-1]*24*60*60
    
    ## Partition data into training and test based on specified times.
    X_train = X[ (times > start_train) & (times < end_train)]
    y_train = y[ (times > start_train) & (times < end_train)]
    X_test  = X[ (times >= end_train ) & (times < end_test )]
    y_test  = y[ (times >= end_train ) & (times < end_test )]
    times_train  = times[ (times > start_train) & (times < end_train)]
    times_test   = times[ (times >= end_train ) & (times < end_test )]
    
    return X_train, y_train, X_test, y_test, times_train, times_test
    

def traveltimes_onelink(start_date = '2019-01-04', end_date = '2019-01-04', link_ref = '1357:1358'):
    data = pd.read_csv('data/link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0)

    ## Reduce to dates specified
    dat_1day = (data[start_date:end_date])
    
    ## Reduce data-set to just link 1357:1358 on this day
    return dat_1day[:][dat_1day['link_ref'] == link_ref]['link_travel_time']

def traveltimes_all(start_date = '2019-01-04', end_date = '2019-01-04'):
    data = pd.read_csv('data/link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0)
    data = pd.pivot_table(data, index = 'time', columns='link_ref', values='link_travel_time')
    return data

## Filipe's prediction evaluation function
def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    mse = np.mean((predicted - trues)**2)
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, mse, rae, rmse, r2

## Filipe's quantile evaluation function
def eval_quantiles(lower, upper, trues, preds):
    N = len(trues)
    icp = np.sum(np.logical_and( (trues>lower),(trues<upper) )) / N
    diffs = np.maximum(0, upper-lower)
    mil = np.sum(diffs) / N
    rmil = 0.0
    for i in range(N):
        if trues[i] != preds[i]:
            rmil += diffs[i] / (np.abs(trues[i]-preds[i]))
    rmil = rmil / N
    clc = np.exp(-rmil*(icp-0.95))
    return icp, mil, rmil, clc