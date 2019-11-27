import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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
