import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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