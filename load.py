##### LOADS DATA AND REDUCES TO ONE DAY FOR ONE LINK   #####
##### CONVERTS DATA TO [SAMPLES, TIME_STEPS, FEATURES] #####

import pandas as pd
import numpy as np
import csv
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

def timeseries_link(traveltimes, times, start, end, total_windows, lags, preds, time_res_mins):
    #nan_ind = np.argwhere(np.isnan(traveltimes))
    #traveltimes = traveltimes[-nan_ind]
    #times       = times[-nan_ind]
    
    #end   = times[-1]
    #start = times[ 0]
    
    ## Number of samples is total_windows minus the lag and predictions
    N = total_windows - lags - preds
    T  = np.zeros(N + lags + preds)
    mean_times = np.zeros((N + lags + preds), dtype='datetime64[s]')
    X  = np.zeros((N, lags + 1))
    y  = np.zeros((N, preds))
    ## Initially loop over lags
    for l in range(lags):
        time = start + np.timedelta64((l - 1)*time_res_mins, 'm')
        mean_times[l] = time
        ## Travel times that are in current window are averaged and saved.
        T[l]  = np.mean(traveltimes[     (times <= time) 
                                       & (times >= time - np.timedelta64(time_res_mins, 'm')) ])
    ## Then loop over remaining time-steps
    for t in range(lags, N + preds + lags):
        time = start + np.timedelta64((t + 1)*time_res_mins, 'm')
        mean_times[t] = time
        
        ## Mean travel time in the window [time - time_res_mins, time]
        T[t] = np.mean(traveltimes[ (times <= time) & 
                                    (times > time - np.timedelta64(time_res_mins, 'm')) ])
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
    return X, y, mean_times


## Use this to convert data from multiple links into timeseries format
## Input: traveltimes [SAMPLES, LINKS]
## Output: X [N, LAGS, LINKS]
##         y [N, PREDS, LINKS]
def timeseries_all(traveltimes, lags, preds, time_res_mins):
    time = traveltimes.index.values
    start = time[ 0]
    end   = time[-1]
    time_minutes = 24*60*(end - start)/np.timedelta64(1, 'D')
    
    num_links = traveltimes.shape[1]
    ## Number of windows in period to consider
    total_windows = int(time_minutes / time_res_mins)
    ## Number of samples is total_windows minus the lag and predictions
    N = total_windows - lags - preds
    T  = np.zeros( (N + lags + preds, num_links) )
    mean_times = np.zeros(N, dtype='datetime64[s]')
    X  = np.zeros((N, lags + 1, num_links))
    y  = np.zeros((N, preds, num_links))
    
    for lnk in range(num_links):
        travel_link = traveltimes.iloc[:,lnk]
        X[:,:,lnk], y[:,:,lnk], mean_times = timeseries_link(travel_link, time, start, end, total_windows, lags, preds, time_res_mins)
    return X, y, mean_times


def split(X, y, times, split_pct):
    ## Convert data to [samples, time steps, links (1 link)]
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1, 1))
    y = y.reshape((y.shape[0], y.shape[1], y.shape[2], 1, 1))
    
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

## Sorts links from start to end and returns
## data with only those links
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

def split_df(data, start_train, end_train, end_test):
    data_train = data[start_train:end_train]
    data_test = data[end_train:end_test]
    return data_train, data_test

    
## Returns the data with times as rows and links as columns
def traveltimes_all(data_train, data_test):
    ## Standardize each link by hour of the day and day of the week
    #data['link_travel_time'] = data.groupby(by = [data.index.weekday_name, data.index.hour, 'link_ref']).transform(lambda x: (x - x.mean()) / x.std())
    means = data_train.groupby(by = [data_train.index.weekday_name, data_train.index.hour, 'link_ref']).mean()
    std = data_train.groupby(by = [data_train.index.weekday_name, data_train.index.hour, 'link_ref']).std()
    
    data_train['link_travel_time'] = data_train.groupby([data_train.index.weekday_name, 
                                                         data_train.index.hour, 
                                                         'link_ref']).transform(lambda x: (x - means) / std)
    data_test['link_travel_time']  = data_test.groupby([data_test.index.weekday_name, 
                                                         data_test.index.hour, 
                                                         'link_ref']).transform(lambda x: (x - means) / std)
    
    ## Spread link variables
    data_train = pd.pivot_table(data_train, index = 'time', columns='link_ref', values='link_travel_time')
    data_test  = pd.pivot_table(data_test, index = 'time', columns='link_ref', values='link_travel_time')
    
    return data_train.sort_index(), data_test.sort_index()
    
def skip_row(index, keep_list):
	if (index == 0):
		return False ## Never want to skip the header
	return index not in keep_list

def write_3d(X, filename):
    X_list = X.tolist()
    with open(filename+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(X_list)

## Reads a 3d array into np.array 
def read_3d(filename):
    with open(filename+'.csv', 'r') as f:
      reader = csv.reader(f)
      examples = list(reader)

    output = []
    for row in examples:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        output.append(nwrow)
        
    N, num_lags, num_links = len(output), len(output[0]), len(output[0][0])
    X = np.zeros((N, num_lags, num_links))
    for n in range(N):
        for lag in range(num_lags):
            for lnk in range(num_links):
                X[n, lag, lnk] = output[n][lag][lnk]
    return X
