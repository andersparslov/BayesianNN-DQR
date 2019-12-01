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
