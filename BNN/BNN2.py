
# coding: utf-8

# In[ ]:

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import math
from datetime import datetime
tfd = tfp.distributions
tfb = tfp.bijectors
import pandas as pd


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

# In[1]:


def write_3d(X, filename):
    X_list = X.tolist()
    with open(filename+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(X_list)

def skip_row(index, keep_list):
	if (index == 0):
		return False ## Never want to skip the header
	return index not in keep_list

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

def drop_remainder(X, y, y_ix, y_mean, y_std, drop):
  return X[:-drop], y[:-drop], y_ix[:-drop], y_mean[:-drop], y_std[:-drop]


# In[2]:


def load_data(lags, start_train, end_train, end_test):
  preds = 1
  data = pd.read_csv('link_travel_time_local.csv.gz', compression='gzip', parse_dates = True, index_col = 0)

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


# In[ ]:


class MixturePrior(object):
    def __init__(self, pi, sigma1, sigma2):
        self.mu, self.pi, self.sigma1, self.sigma2 = (np.float32(v) for v in (0.0, pi, sigma1, sigma2))
        self.dist = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                    probs=[1-self.pi, self.pi]),
                    components_distribution=tfd.Normal(
                      loc=[0, 0],       
                      scale=[self.sigma1, self.sigma2]))
        
    def sample(self):
      return self.dist.sample()

    def log_prob(self, x):
        x = tf.cast(x, tf.float32)
        return self.dist.log_prob(x)


# In[ ]:


class VariationalPosterior(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.stdNorm = tfd.Normal(0,1)
    
    @property
    def sigma(self):
        return tf.math.softplus(self.rho)
    
    def sample(self, training, sampling=True):
      if training:
        epsilon = self.stdNorm.sample(tf.shape(self.rho))
        return self.mu + self.sigma*epsilon
      elif sampling:
        return tfd.Normal(self.mu, self.sigma).sample()
      else:
        return self.mu
    
    def log_prob(self, x):
        return tf.reduce_sum(-tf.math.log(tf.math.sqrt(2 * math.pi))
                - tf.math.log(self.sigma)
                - ((x - self.mu) ** 2) / (2 * self.sigma ** 2))


# In[ ]:


sum_all = tf.math.reduce_sum
class MinimalRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, training, init, prior, **kwargs):
        super(MinimalRNNCell, self).__init__(**kwargs)
        self.init = init
        self.is_training = training
        self.units = units
        self.state_size = units
        self.prior = prior
        
    def initialise_cell(self, links):
        self.W_mu = self.add_weight(shape=(links, self.units),
                                      initializer=self.init,
                                      name='W_mu', trainable=True)
        self.W_rho = self.add_weight(shape=(links, self.units),
                                      initializer=self.init,
                                      name='W_rho', trainable=True)
        self.U_mu = self.add_weight(shape=(self.units, self.units),
                                    initializer=self.init,
                                    name='U_mu', trainable=True)
        self.U_rho = self.add_weight(shape=(self.units, self.units),
                                    initializer=self.init,
                                    name='U_rho', trainable=True)
        self.B_mu = self.add_weight(shape=(1,self.units),
                                    initializer=self.init,
                                    name='B_mu', trainable=True)
        self.B_rho = self.add_weight(shape=(1,self.units),
                                    initializer=self.init,
                                    name='B_rho', trainable=True)
        
        ## Make sure following is only printed once during training and not for testing!
        print("  Basic cell has been built (in:", links, ") (out:", self.units, ")")
        self.W_dist = VariationalPosterior(self.W_mu, self.W_rho)
        self.U_dist = VariationalPosterior(self.U_mu, self.U_rho)
        self.B_dist = VariationalPosterior(self.B_mu, self.B_rho)
        self.sampling = False
        self.built = True
    
    def call(self, inputs, states):
        self.W = self.W_dist.sample(self.is_training, self.sampling)
        self.U = self.U_dist.sample(self.is_training, self.sampling)
        self.B = self.B_dist.sample(self.is_training, self.sampling)
        if self.is_training:
            self.log_prior = sum_all(self.prior.log_prob(self.B)) + sum_all(self.prior.log_prob(self.W)) + sum_all(self.prior.log_prob(self.U)) 
            self.log_variational_posterior  = sum_all(self.W_dist.log_prob(self.W))
            self.log_variational_posterior += sum_all(self.U_dist.log_prob(self.U))
            self.log_variational_posterior += sum_all(self.B_dist.log_prob(self.B))
        h = tf.linalg.matmul(inputs, self.W)
        output = tf.math.tanh(self.B + h + tf.linalg.matmul(states[0], self.U))
        return output, [output]

    def get_initial_state(self, inputs = None, batch_size = None, dtype = None):
        return [tf.zeros((batch_size, self.state_size), dtype = dtype)]


# In[ ]:


class BayesianLSTMCell_Untied(tf.keras.Model):
    def __init__(self, num_units, training, init, prior, **kwargs):
        super(BayesianLSTMCell_Untied, self).__init__(num_units, **kwargs)
        self.init = init
        self.units = num_units
        self.is_training = training
        self.state_size = self.units
        self.prior = prior
        
    def initialise_cell(self, links):
        self.num_links = links
        self.Ui_mu = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.init,
                                      name='Ui_mu', trainable=True)
        self.Ui_rho = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.init,
                                      name='Ui_rho', trainable=True)
        self.Uo_mu = self.add_weight(shape=(self.units, self.units),
                                    initializer=self.init,
                                    name='Uo_mu', trainable=True)
        self.Uo_rho = self.add_weight(shape=(self.units, self.units),
                                    initializer=self.init,
                                    name='Uo_rho', trainable=True)
        self.Uf_mu = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.init,
                                      name='Uf_mu', trainable=True)
        self.Uf_rho = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.init,
                                      name='Uf_rho', trainable=True)
        self.Ug_mu = self.add_weight(shape=(self.units, self.units),
                                    initializer=self.init,
                                    name='Ug_mu', trainable=True)
        self.Ug_rho = self.add_weight(shape=(self.units, self.units),
                                    initializer=self.init,
                                    name='Ug_rho', trainable=True)
        
        self.Wi_mu = self.add_weight(shape=(self.num_links, self.units),
                                      initializer=self.init,
                                      name='Wi_mu', trainable=True)
        self.Wi_rho = self.add_weight(shape=(self.num_links, self.units),
                                      initializer=self.init,
                                      name='Wi_rho', trainable=True)
        self.Wo_mu = self.add_weight(shape=(self.num_links, self.units),
                                    initializer=self.init,
                                    name='Wo_mu', trainable=True)
        self.Wo_rho = self.add_weight(shape=(self.num_links, self.units),
                                    initializer=self.init,
                                    name='Wo_rho', trainable=True)
        self.Wf_mu = self.add_weight(shape=(self.num_links, self.units),
                                      initializer=self.init,
                                      name='Wf_mu', trainable=True)
        self.Wf_rho = self.add_weight(shape=(self.num_links, self.units),
                                      initializer=self.init,
                                      name='Wf_rho', trainable=True)
        self.Wg_mu = self.add_weight(shape=(self.num_links, self.units),
                                    initializer=self.init,
                                    name='Wg_mu', trainable=True)
        self.Wg_rho = self.add_weight(shape=(self.num_links, self.units),
                                    initializer=self.init,
                                    name='Wg_rho', trainable=True)
        
        self.Bi_mu = self.add_weight(shape=(1, self.units),
                                      initializer=self.init,
                                      name='Wi_mu', trainable=True)
        self.Bi_rho = self.add_weight(shape=(1, self.units),
                                      initializer=self.init,
                                      name='Wi_rho', trainable=True)
        self.Bo_mu = self.add_weight(shape=(1, self.units),
                                    initializer=self.init,
                                    name='Wo_mu', trainable=True)
        self.Bo_rho = self.add_weight(shape=(1, self.units),
                                    initializer=self.init,
                                    name='Wo_rho', trainable=True)
        self.Bf_mu = self.add_weight(shape=(1, self.units),
                                      initializer=self.init,
                                      name='Wf_mu', trainable=True)
        self.Bf_rho = self.add_weight(shape=(1, self.units),
                                      initializer=self.init,
                                      name='Wf_rho', trainable=True)
        self.Bg_mu = self.add_weight(shape=(1, self.units),
                                    initializer=self.init,
                                    name='Wg_mu', trainable=True)
        self.Bg_rho = self.add_weight(shape=(1, self.units),
                                    initializer=self.init,
                                    name='Wg_rho', trainable=True)
        
        self.Ui_dist = VariationalPosterior(self.Ui_mu, self.Ui_rho)
        self.Uo_dist = VariationalPosterior(self.Uo_mu, self.Uo_rho)
        self.Uf_dist = VariationalPosterior(self.Uf_mu, self.Uf_rho)
        self.Ug_dist = VariationalPosterior(self.Ug_mu, self.Ug_rho)
        self.Wi_dist = VariationalPosterior(self.Wi_mu, self.Wi_rho)
        self.Wo_dist = VariationalPosterior(self.Wo_mu, self.Wo_rho)
        self.Wf_dist = VariationalPosterior(self.Wf_mu, self.Wf_rho)
        self.Wg_dist = VariationalPosterior(self.Wg_mu, self.Wg_rho)
        self.Bi_dist = VariationalPosterior(self.Bi_mu, self.Bi_rho)
        self.Bo_dist = VariationalPosterior(self.Bo_mu, self.Bo_rho)
        self.Bf_dist = VariationalPosterior(self.Bf_mu, self.Bf_rho)
        self.Bg_dist = VariationalPosterior(self.Bg_mu, self.Bg_rho)
        ## Make sure following is only printed once during training and not for testing!
        print("  Untied cell has been built (in:", links, ") (out:", self.units, ")")
        self.sampling = False
        self.built = True
    
    def call(self, inputs, states):
        Ui = self.Ui_dist.sample(self.is_training, self.sampling)
        Uo = self.Uo_dist.sample(self.is_training, self.sampling)
        Uf = self.Uf_dist.sample(self.is_training, self.sampling)
        Ug = self.Ug_dist.sample(self.is_training, self.sampling)
        Wi = self.Wi_dist.sample(self.is_training, self.sampling)
        Wo = self.Wo_dist.sample(self.is_training, self.sampling)
        Wf = self.Wf_dist.sample(self.is_training, self.sampling)
        Wg = self.Wg_dist.sample(self.is_training, self.sampling)
        Bi = self.Bi_dist.sample(self.is_training, self.sampling)
        Bo = self.Bo_dist.sample(self.is_training, self.sampling)
        Bf = self.Bf_dist.sample(self.is_training, self.sampling)
        Bg = self.Bg_dist.sample(self.is_training, self.sampling)

        c_t, h_t = tf.split(value=states[0], num_or_size_splits=2, axis=0)
        
        inputs = tf.cast(inputs, tf.float32)
        i = tf.sigmoid(Bi + tf.linalg.matmul(h_t, Ui) + tf.linalg.matmul(inputs, Wi))
        o = tf.sigmoid(Bo + tf.linalg.matmul(h_t, Uo) + tf.linalg.matmul(inputs, Wo))
        f = tf.sigmoid(Bf + tf.linalg.matmul(h_t, Uf) + tf.linalg.matmul(inputs, Wf))
        g = tf.math.tanh(Bg + tf.linalg.matmul(h_t, Ug) + tf.linalg.matmul(inputs, Wg))
        
        self.log_prior  =  sum_all(self.prior.log_prob(Ui) + self.prior.log_prob(Uo) + self.prior.log_prob(Uf) + self.prior.log_prob(Ug))
        self.log_prior +=  sum_all(self.prior.log_prob(Wi) + self.prior.log_prob(Wo) + self.prior.log_prob(Wf) + self.prior.log_prob(Wg))
        self.log_prior +=  sum_all(self.prior.log_prob(Bi) + self.prior.log_prob(Bo) + self.prior.log_prob(Bf) + self.prior.log_prob(Bg))
        self.log_variational_posterior  = sum_all(self.Ui_dist.log_prob(Ui) + self.Uo_dist.log_prob(Uo) + self.Uf_dist.log_prob(Uf) + self.Ug_dist.log_prob(Ug))
        self.log_variational_posterior += sum_all(self.Wi_dist.log_prob(Wi) + self.Wo_dist.log_prob(Wo) + self.Wf_dist.log_prob(Wf) + self.Wg_dist.log_prob(Wg))
        self.log_variational_posterior += sum_all(self.Bi_dist.log_prob(Bi) + self.Bo_dist.log_prob(Bo) + self.Bf_dist.log_prob(Bf) + self.Bg_dist.log_prob(Bg))
        
        c_new = f*c_t + i*g
        h_new = o*tf.math.tanh(c_new)
        new_state = tf.concat([c_new, h_new], axis=0)
        return h_new, new_state
    
    def get_initial_state(self, inputs = None, batch_size = None, dtype = None):
        return tf.zeros((2*batch_size, self.units), dtype = dtype)


# In[ ]:


class BayesianLSTMCellTied(tf.keras.Model):
    def __init__(self, num_units, training, init, prior, **kwargs):
        super(BayesianLSTMCellTied, self).__init__(num_units, **kwargs)
        self.init = init
        self.prior = prior 
        self.units = num_units
        self.state_size = num_units
        self.is_training = training
        
    def initialise_cell(self, links):
        self.num_links = links
        self.W_mu = self.add_weight(shape=(self.units+self.num_links, 4*self.units),
                                      initializer=self.init,
                                      name='W_mu', trainable=True)
        self.W_rho = self.add_weight(shape=(self.units+self.num_links, 4*self.units),
                                      initializer=self.init,
                                      name='W_rho', trainable=True)
        self.B_mu = self.add_weight(shape=(1, 4*self.units),
                                    initializer=self.init,
                                    name='B_mu', trainable=True)
        self.B_rho = self.add_weight(shape=(1, 4*self.units),
                                    initializer=self.init,
                                    name='B_rho', trainable=True)
        
        self.W_dist = VariationalPosterior(self.W_mu, self.W_rho)
        self.B_dist = VariationalPosterior(self.B_mu, self.B_rho)
        ## Make sure following is only printed once during training and not for testing!
        print("  Tied Cell has been built (in:", links, ") (out:", self.units, ")")
        self.sampling = False
        self.built = True

    def call(self, inputs, states):
        W = self.W_dist.sample(self.is_training, self.sampling)
        B = self.B_dist.sample(self.is_training, self.sampling)
        c_t, h_t = tf.split(value=states[0], num_or_size_splits=2, axis=0)
        concat_inputs_hidden = tf.concat([tf.cast(inputs, tf.float32), h_t], 1)
        concat_inputs_hidden = tf.nn.bias_add(tf.matmul(concat_inputs_hidden, tf.squeeze(W)), 
                                              tf.squeeze(B))
        
        self.log_prior =  sum_all(self.prior.log_prob(W)) + sum_all(self.prior.log_prob(B))
        self.log_variational_posterior = sum_all(self.W_dist.log_prob(W)) + sum_all(self.B_dist.log_prob(B))
        
        # Gates: Input, New, Forget and Output
        i, j, f, o = tf.split(value = concat_inputs_hidden, num_or_size_splits = 4, axis = 1)
        c_new = c_t*tf.sigmoid(f) + tf.sigmoid(i)*tf.math.tanh(j)
        h_new = tf.math.tanh(c_new)*tf.sigmoid(o)
        new_state = tf.concat([c_new, h_new], axis=0)
        return h_new, new_state
    
    def get_initial_state(self, inputs = None, batch_size = None, dtype = None):
        return tf.zeros((2*batch_size, self.units), dtype = dtype)


# In[ ]:


class BayesianRNN(tf.keras.Model):
    def __init__(self, num_units, num_links, batch_size, init, cell_type, prior, **kwargs):
        super(BayesianRNN, self).__init__(**kwargs)
        self.cell_type = cell_type
        self.init = init
        self.num_units_lst = num_units
        self.num_links = num_links
        self.batch_size = batch_size
        self.cell_prior = prior
        self.prior = prior
        self.build()
    
    def build(self):
        print("Building net...")
        self.cell_lst = []
        state_size = self.num_links
        for i, num_units in enumerate(self.num_units_lst):
          if self.cell_type == 'Basic':
              self.cell_lst.append(MinimalRNNCell(num_units, training=True, init=self.init, prior=self.cell_prior))
          elif self.cell_type == 'TiedLSTM':
              self.cell_lst.append(BayesianLSTMCellTied(num_units, training=True, init=self.init, prior=self.cell_prior))
          else:
              self.cell_lst.append(BayesianLSTMCell_Untied(num_units, training=True, init=self.init, prior=self.cell_prior))
          self.cell_lst[-1].initialise_cell(state_size)
          state_size = num_units
            
        self.weight_mu = self.add_weight(shape=(self.num_units_lst[-1],self.num_links),
                                 initializer=self.init,
                                 name='weight_mu')
        self.weight_rho = self.add_weight(shape=(self.num_units_lst[-1],self.num_links),
                                 initializer=self.init,
                                 name='weight_mu')
        self.bias_mu = self.add_weight(shape=(self.num_links,),
                                     initializer=self.init,
                                     name='bias_mu', trainable=True)
        self.bias_rho = self.add_weight(shape=(self.num_links,),
                                     initializer=self.init,
                                     name='bias_mu', trainable=True)
        self.weight_dist = VariationalPosterior(self.weight_mu, self.weight_rho) 
        self.bias_dist = VariationalPosterior(self.bias_mu, self.bias_rho)     
        print("  Output layer has been built (in:", self.num_units_lst[-1], ") (out:", 1, ")")

        ## The diagonal of the correlation matrix
        self.scale_prior = tfd.LKJ(dimension=self.num_links, concentration=10, input_output_cholesky=True)
        self.y_rho = self.add_weight(shape=(self.num_links*((self.num_links-1)/2 + 1),), 
                                     initializer='zeros',
                                     name='y_rho',
                                     trainable=True)
        self.built = True
    @property
    def y_std(self):
        cor = tfb.ScaleTriL(diag_bijector=tfb.Softplus(),
                            diag_shift=None)
        return cor.forward(self.y_rho)

    def call(self, batch_x, training, sampling):
        self.weight = self.weight_dist.sample(training, sampling)
        self.bias = self.bias_dist.sample(training, sampling)
        if training:
            self.log_prior_dense = sum_all(self.prior.log_prob(self.weight)) + sum_all(self.prior.log_prob(self.bias))
            self.log_variational_posterior_dense  = self.weight_dist.log_prob(self.weight) 
            self.log_variational_posterior_dense += self.bias_dist.log_prob(self.bias)
        for cell in self.cell_lst:
          cell.is_training = training
          cell.sampling = sampling

        inputs = tf.convert_to_tensor(batch_x)
        rnn = tf.keras.layers.RNN(self.cell_lst)
        ## RNN layer
        final_rnn_output = rnn(inputs)
        ## Dense layer
        self.outputs = tf.linalg.matmul(final_rnn_output, self.weight) + self.bias   
        return self.outputs
    
    def log_prior(self):
        return sum(sum_all(cell.log_prior) for cell in self.cell_lst) + sum_all(self.log_prior_dense) + sum_all(self.scale_prior.log_prob(self.y_std))
    
    def log_variational_posterior(self):
        return sum(sum_all(cell.log_variational_posterior) for cell in self.cell_lst) + sum_all(self.log_variational_posterior_dense)
    
    def elbo(self, batch_x, batch_y, batch_ind, num_batches,  training, sampling=True):
        output = self(batch_x, training, sampling)
        assert(batch_y.shape[1] == self.num_links)
        assert(output.shape == batch_y.shape)
        pred_dist = tfd.MultivariateNormalTriL(output, scale_tril=self.y_std)
        self.nll = -tf.math.reduce_sum(pred_dist.log_prob(batch_y))
        kl_weight = 2**(num_batches - batch_ind) / (2**num_batches - 1)
        return (self.log_variational_posterior() - self.log_prior())/num_batches + self.nll, sum_all((output - batch_y)**2) / self.batch_size


# In[ ]:


def train_step(mod, data_train, lr):
    elbo_sum = 0
    mse_sum = 0
    batch_ind = 1
    optimizer = tf.keras.optimizers.Adam(lr = lr)
    for batch_x, batch_y in data_train:
        x = tf.cast(batch_x, tf.float32)
        y = tf.cast(batch_y, tf.float32)
        with tf.GradientTape() as tape:
            loss, mse = mod.elbo(x, y[:,0], batch_ind, num_batch_train, training=True)
        gradients = tape.gradient(loss, mod.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mod.trainable_variables))
        batch_ind = batch_ind + 1
        elbo_sum += loss
        mse_sum += mse
    return elbo_sum, mse_sum

def val_loss(mod, data_test):
    elbo_sum = 0
    mse_sum = 0
    batch_ind = 1
    for batch_x, batch_y in data_test:
        x = tf.cast(batch_x, tf.float32)
        y = tf.cast(batch_y, tf.float32)
        loss, mse = mod.elbo(x, y[:,0], batch_ind, num_batch_test, training=False, sampling=False)
        elbo_sum += loss
        mse_sum += mse
        batch_ind = batch_ind + 1
    return elbo_sum, mse_sum

def plot_val(mod, data_test):
    y_pred = np.zeros((y_test.shape))
    T = 0
    for x_batch, _ in data_test:
        x = tf.cast(x_batch,tf.float32)
        y_pred[T*batch_size:(T+1)*batch_size, 0] = mod(x, training=False, sampling=False)
        T = T+1
    for lnk in range(y_test.shape[2]):
        print("  Link ", lnk)
        plt.plot(y_ix_test, y_test[:,0,lnk], 'bo', alpha = 0.4)
        plt.plot(y_ix_test, y_pred[:,0,lnk], c = 'r')
        plt.show()


# In[ ]:


sigma1 = 1
sigma2 = np.exp(-6)
lags = 10
preds = 1

start_train_lst = ['2019-01-01', '2019-01-07', '2019-01-14', '2019-01-21', '2019-02-01']
end_train_lst = ['2019-01-31', '2019-02-07', '2019-02-14', '2019-02-21', '2019-03-01']
end_test_lst = ['2019-02-07', '2019-02-14', '2019-02-21', '2019-03-01', '2019-03-07']
    
num_partitions = 3
num_links = 16
batch_size = 80

units_lst = np.arange(6, 72, 12)
cell_types = ['Basic', 'TiedLSTM', 'UntiedLSTM']
prior = MixturePrior(0.10, sigma1, sigma2)
init = 'uniform'

lr = 1e-2
epochs = 200
patience = 8

mse =  np.empty((num_partitions, len(units_lst), len(cell_types)))
icp =  np.empty((num_partitions, len(units_lst), len(cell_types)))
mil =  np.empty((num_partitions, len(units_lst), len(cell_types)))
time = np.empty((num_partitions, len(units_lst), len(cell_types)))
  
for u, units in enumerate(units_lst):
  print("Units {}".format(units))
  ## Instantiate nets
  nets = []
  best_weights = [None, None, None]
  for cell in cell_types:
    nets.append(BayesianRNN([units], num_links, batch_size, init, cell, prior))

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
    num_batch_train = int(X_train.shape[0]/batch_size)
    num_batch_test = int(X_test.shape[0]/batch_size)

    data_train = tf.data.Dataset.from_tensor_slices((X_train, 
                                                    y_train)).shuffle(1000).batch(batch_size, drop_remainder=True)
    data_test = tf.data.Dataset.from_tensor_slices((X_test, 
                                                    y_test)).batch(batch_size, drop_remainder=True)
    drop_train = len(y_train) - num_batch_train*batch_size
    drop_test = len(y_test) - num_batch_test*batch_size
    X_train, y_train, y_ix_train, y_mean_train, y_std_train = drop_remainder(X_train, y_train, y_ix_train, y_mean_train, y_std_train, drop_train)
    X_test, y_test, y_ix_test, y_mean_test, y_std_test = drop_remainder(X_test, y_test, y_ix_test, y_mean_test, y_std_test, drop_test)
    
    for n, net in enumerate(nets):
      init_lr = 1e-2
      t1 = datetime.now()
      ## Initialise weights using last partition
      if part != 0:
        init_lr = 5e-3
        net.set_weights(best_weights[n])

      best_elbo = 10000000000000000000000
      best_mse = 10000000000000000000000
      lr = init_lr
      for epoch in range(epochs):
        ## Training
        elbo_sum, mse_sum = train_step(net, data_train, lr)
        mse_avg_train = mse_sum.numpy() / num_batch_train
        elbo_avg = elbo_sum.numpy() / num_batch_train
        ## Validation
        elbo_test, mse_sum = val_loss(net, data_test)
        mse_avg_test = mse_sum.numpy() / num_batch_test
        elbo_avg_test = elbo_test.numpy() / num_batch_test
        
        if elbo_avg_test < best_elbo:
            best_weights[n] = net.get_weights()
            best_elbo = elbo_avg_test
            patience_counter = 0
        elif mse_avg_test < best_mse:
            best_weights[n] = net.get_weights()
            best_mse = mse_avg_test
            patience_counter = 0
        patience_counter += 1
        
        if patience_counter > patience:
            print("Partition {} early stopping after {} epochs".format(part, epoch))
            break
        if patience_counter >= int(patience/2):
            lr = init_lr*np.exp(-0.1*epoch)
      ## Set weights back to the best model
      net.set_weights(best_weights[n])
      net.save_weights("bnn_u{}_p{}_n{}.hdf5".format(u, part, n))
      t2 = datetime.now()
      time[part, u, n] = (t2-t1).seconds

      nsamples = 50
      y_pred = np.empty((nsamples, y_test.shape[0],y_test.shape[2]))
      T = 0
      for x_batch, _ in data_test:
          x = tf.cast(x_batch,tf.float32)
          for samp in range(nsamples):
            out = net(x, training=False, sampling=True)
            y_pred[samp, T*batch_size:(T+1)*batch_size] = tfd.MultivariateNormalTriL(out, scale_tril=net.y_std).sample()
          T = T+1
      icp_lnks = np.zeros(num_links)  
      mil_lnks = np.zeros(num_links)
      quantiles = np.array([0.05, 0.95])
      for lnk in range(num_links):
            q1 = np.quantile(y_pred[:,:,lnk], quantiles[ 0], axis=0)
            q2 = np.quantile(y_pred[:,:,lnk], quantiles[ 1], axis=0)
            icp_lnks[lnk] = 1 - (np.sum(y_test[:,:,lnk] < q1) + np.sum(y_test[:,:,lnk] > q2) )/len(y_test)
            mil_lnks[lnk] = np.sum(np.maximum(0, q2 - q1)) / len(y_test)
      icp[part, u, n] = np.mean(icp_lnks)
      mil[part, u, n] = np.mean(mil_lnks)
    
      y_pred_all = np.sum(y_pred*y_std_test[:,0] + y_mean_test[:,0], axis=2)
      y_test_all = np.sum(y_test[:,0]*y_std_test[:,0] + y_mean_test[:,0], axis=1)
      quantiles = np.array([0.05, 0.95])
      y_mean = np.mean(y_pred_all, axis=0)
      q1 = np.quantile(y_pred_all, quantiles[ 0], axis=0)
      q2 = np.quantile(y_pred_all, quantiles[ 1], axis=0)
      
      mse[part, u, n] = np.sum((y_mean - y_test_all)**2) / len(y_test_all)

