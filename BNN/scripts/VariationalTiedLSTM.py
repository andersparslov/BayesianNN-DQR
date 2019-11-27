import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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
