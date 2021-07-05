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
from zhusuan_of.distributions.bernoulli import *

flow.enable_eager_execution()

class TestBernoulli(unittest.TestCase):

    def setUp(self):
        self._Bernoulli = lambda probs, **kwargs: Bernoulli(probs=probs,  **kwargs)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(self, self._Bernoulli, np.zeros)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_same(
            self, self._Bernoulli, np.zeros)

    def test_log_prob_shape(self):
        utils.test_1parameter_log_prob_shape_same(
            self, self._Bernoulli, np.zeros, np.zeros)

    def test_value(self):
        def _test_value(logits, given):
            logits = np.array(logits, np.float32)
            given = np.array(given, np.float32)

            target_log_p = stats.bernoulli.logpmf(
                given, -logits+ 1e-8)

            logits = flow.Tensor(logits)
            given = flow.Tensor(given)

            bernoulli = self._Bernoulli(logits)
            log_p = bernoulli.log_prob(given)
            target_log_p = target_log_p.astype(log_p.numpy().dtype)
            np.testing.assert_allclose(np.around(log_p.numpy(), decimals=6),
                                       np.around(target_log_p, decimals=6), rtol=1e-03)

        _test_value([0.], [0, 1])
        _test_value([-50., -10., -50.], [1, 1, 0])
        _test_value([0., 4.], [[0, 1], [0, 1]])
        _test_value([[2., 3., 1.], [5., 7., 4.]],
                    np.ones([3, 2, 3], dtype=np.int32))

    def test_distribution_shape(self):
        param = flow.ones((1))*.8
        distribution = self._Bernoulli(param)
        utils.test_and_save_distribution_img(distribution)
