import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
sum_all = tf.math.reduce_sum
from VariationalPosterior import VariationalPosterior

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
