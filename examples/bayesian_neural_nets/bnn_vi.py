# Copyright TODO

""" BNN with VI.ELBO  example code """
import sys
import os
import math
import numpy as np

import oneflow.experimental as flow

flow.enable_eager_execution()

sys.path.append('..')
sys.path.append('../..')
import conf

from zhusuan_of.framework.bn import BayesianNet
from zhusuan_of.variational.elbo import ELBO

from utils import load_uci_boston_housing, standardize


class Net(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles
        self.y_logstd = flow.nn.Parameter(flow.Tensor(1))
        flow.nn.init.uniform_(self.y_logstd)

    def forward(self, observed):
        self.observe(observed)
        
        x = self.observed['x']
        h = flow.tile(x, reps=(self.n_particles, *len(x.shape)*[1]))

        batch_size = x.shape[0]

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w = self.sn('Normal',
                        name="w" + str(i), 
                        mean=flow.zeros((n_out, n_in + 1), dtype=flow.float32), 
                        std=flow.ones((n_out, n_in +1), dtype=flow.float32),
                        group_ndims=2, 
                        n_samples=self.n_particles,
                        reduce_mean_dims=[0])
            w = flow.unsqueeze(w, dim=1)
            w = flow.tile(w, reps=(1, batch_size, 1,1))
            h = flow.cat([h, flow.ones((*(list(h.shape)[:-1]), 1), dtype=flow.float32).to("cuda")], -1)
            h = flow.reshape(h, list(h.shape) + [1])
            p = flow.sqrt(flow.Tensor([h.shape[2]], dtype=flow.float32).to("cuda"))


            wd1, wd2, wd3, wd4 = w.shape
            w = flow.reshape(w, [wd1*wd2, wd3, wd4])
            hd1, hd2, hd3, hd4 = h.shape
            h = flow.reshape(h, [hd1*hd2, hd3, hd4])
            h = flow.bmm(w, h) / p

            h = flow.reshape(h, [hd1, hd2, wd3, hd4])

            h = flow.squeeze(h, [-1])

            if i < len(self.layer_sizes) - 2:
                h = flow.nn.ReLU()(h)

        y_mean = flow.squeeze(h, [2])

        y = self.observed['y']
        y_pred = flow.mean(y_mean, dim=[0])
        self.cache['rmse'] = flow.sqrt(flow.mean((y - y_pred)**2))

        self.sn('Normal',
                name='y',
                mean=y_mean,
                logstd=self.y_logstd,
                reparameterize=True,
                reduce_mean_dims=[0,1],
                multiplier=456,) ## training data size

        return self


class Variational(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles

        self.w_means = [] 
        self.w_logstds = [] 

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w_mean_ = flow.nn.Parameter(flow.Tensor(n_out, n_in + 1))
            self.w_means.append(w_mean_)
            w_logstd_ = flow.nn.Parameter(flow.Tensor(n_out, n_in + 1))
            self.w_logstds.append(w_logstd_)

        self.w_means = flow.nn.ParameterList(self.w_means)
        self.w_logstds = flow.nn.ParameterList(self.w_logstds)

    def forward(self, observed):
        self.observe(observed)
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            self.sn('Normal',
                    name='w' + str(i),
                    mean=self.w_means[i],
                    logstd=self.w_logstds[i],
                    group_ndims=2,
                    n_samples=self.n_particles,
                    reparametrize=True,
                    reduce_mean_dims=[0])
        return self

def main():
    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, "housing.data")
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, x_dim = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = standardize(
        y_train, y_test)
    
    print('data size: ', len(x_train))

    # Define model parameters
    lb_samples = 512
    epoch_size = 5000
    batch_size = 114

    n_hiddens = [50]
    layer_sizes = [x_dim] + n_hiddens + [1]
    print('layer size: ', layer_sizes)

    # create the network
    net = Net(layer_sizes, lb_samples).to("cuda")
    variational = Variational(layer_sizes, lb_samples).to("cuda")

    model = ELBO(net, variational).to("cuda")
    lr = 0.001
    optimizer = flow.optim.Adam(parameters=model.parameters(), lr=lr)
    
    # do train
    len_ = len(x_train)
    num_batches = math.floor(len_ / batch_size)

    # Define training/evaluation parameters
    test_freq = 20

    for epoch in range(epoch_size):
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm, :]
        y_train = y_train[perm]

        for step in range(num_batches):
            x = flow.Tensor(x_train[step*batch_size:(step+1)*batch_size]).to("cuda")
            y = flow.Tensor(y_train[step*batch_size:(step+1)*batch_size]).to("cuda")

            lbs = model({'x':x, 'y':y})
            lbs.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % num_batches == 0:
                rmse = net.cache['rmse'].numpy()
                print("Epoch[{}/{}], Step [{}/{}], Lower bound: {:.4f}, RMSE: {:.4f}"
                      .format(epoch + 1, epoch_size, step + 1, num_batches, float(lbs.numpy()), float(rmse )* std_y_train))

        # eval
        if epoch % test_freq == 0:
            x_t = flow.Tensor(x_test).to("cuda")
            y_t = flow.Tensor(y_test).to("cuda")
            lbs = model({'x':x_t, 'y':y_t})
            rmse = net.cache['rmse'].numpy()
            print('>> TEST')
            print('>> Test Lower bound: {:.4f}, RMSE: {:.4f}'.format(float(lbs.numpy()), float(rmse) * std_y_train))


if __name__ == '__main__':
    main()
