## For writing 3d arrays to memory
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

## For only loading some of data 
def skip_row(index, keep_list):
    if (index == 0):
        return False 
    return index not in keep_list

## Computes mean and std for link/time-of-day/day-of-week groups
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

## Standardise data by means and scales
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

## Rolls data into timeseries structure
def roll(ix, ts, removed_mean, removed_scale, lags, preds):
    X = np.stack([np.roll(ts, i, axis = 0) for i in range(lags, 0, -1)], axis = 1)[lags:-preds,]
    Y = np.stack([np.roll(ts, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    Y_ix = ix[lags:-preds]
    Y_mean = np.stack([np.roll(removed_mean, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    Y_scale = np.stack([np.roll(removed_scale, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)[lags:-preds,]
    return X, Y, Y_ix, Y_mean, Y_scale

## Sorts links by neighbouring links
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

## Returns time of day group
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

## Splits data into training and testing
def split_df(data, start_train, end_train, end_test):
    data_train = data[start_train:end_train]
    data_test = data[end_train:end_test]
    return data_train, data_test

## Drops remainder samples
def drop_remainder(X, y, y_ix, y_mean, y_std, drop):
    return X[:-drop], y[:-drop], y_ix[:-drop], y_mean[:-drop], y_std[:-drop]

## Loads, standardises and rolls data.
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
    ts_train_df, mean_train_df, scale_train_df = transform(data_train, means_df_train,scales_df_train,order, freq = '15min')
    ts_test_df, mean_test_df, scale_test_df = transform(data_test, means_df_train, scales_df_train, order, freq = '15min')
    return ts_train_df, mean_train_df, scale_train_df, ts_test_df, mean_test_df, scale_test_df