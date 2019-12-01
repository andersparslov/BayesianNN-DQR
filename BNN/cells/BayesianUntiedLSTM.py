import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
sum_all = tf.math.reduce_sum
from VariationalPosterior import VariationalPosterior

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
