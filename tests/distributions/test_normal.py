#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import stats

import unittest

from tests.distributions import utils

import oneflow.experimental as flow
from zhusuan_of.distributions.normal import *

flow.enable_eager_execution()

# TODO: test sample value
class TestNormal(unittest.TestCase):
    def setUp(self):        
        self._Normal_std = lambda mean, std, **kwargs: Normal(
            mean=mean, std=std, **kwargs)
        self._Normal_logstd = lambda mean, logstd, **kwargs: Normal(
            mean=mean, logstd=logstd, **kwargs)

    def test_init(self):
        try:
            Normal(mean=flow.ones((2, 1)),
                           std=flow.zeros((2, 4, 3)), logstd=flow.zeros((2, 2, 3)))
        except:
            raise ValueError("Either.*should be passed")

        try:
            Normal(mean=flow.ones((2, 1)), logstd=flow.zeros((2, 4, 3)))
        except:
            raise ValueError("should be broadcastable to match")

        try:
            Normal(mean=flow.ones((2, 1)), std=flow.ones((2, 4, 3)))
        except:
            raise ValueError("should be broadcastable to match")
        Normal(mean=flow.ones((32, 1), dtype=flow.float32),
               logstd=flow.ones((32, 1, 3), dtype=flow.float32))
        Normal(mean=flow.ones((32, 1), dtype=flow.float32),
               std=flow.ones((32, 1, 3), dtype=flow.float32) )

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(
            self, self._Normal_std, np.zeros, np.ones)
        utils.test_2parameter_sample_shape_same(
            self, self._Normal_logstd, np.zeros, np.zeros)

    def test_sample_reparameterized(self):
        mean = flow.ones((2, 3), requires_grad=True)
        logstd = flow.ones((2, 3), requires_grad=True)
        norm_rep = Normal(mean=mean, logstd=logstd)
        samples = norm_rep.sample()
        mean_grads, logstd_grads = flow.autograd.grad(
            outputs=[samples], inputs=[mean, logstd], out_grads=[flow.ones_like(samples)])

        self.assertTrue(mean_grads is not None)
        self.assertTrue(logstd_grads is not None)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(
            self, self._Normal_std, np.zeros, np.ones, np.zeros)
        utils.test_2parameter_log_prob_shape_same(
            self, self._Normal_logstd, np.zeros, np.zeros, np.zeros)
            

    def test_value(self):
        def _test_value(given, mean, logstd):
            mean = np.array(mean, np.float32)
            given = np.array(given, np.float32)
            logstd = np.array(logstd, np.float32)
            std = np.exp(logstd)
            target_log_p = np.array(stats.norm.logpdf(given, mean, np.exp(logstd)), np.float32)

            mean = flow.Tensor(mean)
            logstd = flow.Tensor(logstd)
            std = flow.Tensor(std)
            given = flow.Tensor(given)
            norm1 = Normal(mean=mean, logstd=logstd)
            log_p1 = norm1.log_prob(given)
            np.testing.assert_allclose(log_p1.numpy(), target_log_p, rtol= 1e-03)

            norm2 = Normal(mean=mean, std=std)
            log_p2 = norm2.log_prob(given)
            np.testing.assert_allclose(log_p2.numpy(), target_log_p, rtol= 1e-03)

        _test_value([0.], [0.], [0.])
        _test_value([0.99, 0.9, 9., 99.], [1.], [-3., -1., 1., 10.])
        _test_value([7.], [0., 4.], [[1., 2.], [3., 5.]])

    def test_distribution_shape(self):
        param1 = flow.zeros((1))
        param2 = flow.ones((1))
        distribution = self._Normal_logstd(param1, param2)
        utils.test_and_save_distribution_img(distribution)