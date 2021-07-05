#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import unittest

import oneflow.experimental as flow
from zhusuan_of.distributions.base import *

flow.enable_eager_execution()

class Dist(Distribution):
    def __init__(self,
                 dtype=flow.float32,
                 param_dtype=flow.float32,
                 group_ndims=0,
                 shape_fully_defined=True,
                 **kwargs):
        super(Dist, self).__init__(dtype,
                                   param_dtype,
                                   is_continuous=True,
                                   is_reparameterized=True,
                                   group_ndims=group_ndims,
                                   **kwargs)
        self._shape_fully_defined = shape_fully_defined

    def _value_shape(self):
        return [5]

    def _get_value_shape(self):
        if self._shape_fully_defined:
            return [5]
        return None

    def _batch_shape(self):
        return [2,3,4]

    def _get_batch_shape(self):
        if self._shape_fully_defined:
            return [2,3,4]
        return [None,3,4]

    def _sample(self, n_samples):
        return flow.ones((n_samples, 2, 3, 4, 5))

    def _log_prob(self, given):
        return flow.sum(flow.zeros_like(given), -1)


class TestDistributions(unittest.TestCase):
    def test_baseclass(self):
        dist = Distribution(
                 dtype=flow.float32,
                 param_dtype=flow.float32,
                 is_continuous = True,
                 is_reparameterized = True,
                 use_path_derivative=False,
                 group_ndims=2)
        self.assertEqual(dist.dtype, flow.float32)
        self.assertEqual(dist.param_dtype, flow.float32)
        self.assertEqual(dist.is_continuous, True)
        self.assertEqual(dist.is_reparameterized, True)
        self.assertEqual(dist.group_ndims, 2)
        with self.assertRaises(NotImplementedError):
            dist._value_shape()
        with self.assertRaises(NotImplementedError):
            dist._get_value_shape()
        with self.assertRaises(NotImplementedError):
            dist._batch_shape()
        with self.assertRaises(NotImplementedError):
            dist._get_batch_shape()
        with self.assertRaises(NotImplementedError):
            dist._sample(n_samples=1)

        with self.assertRaises(NotImplementedError):
            dist._log_prob(flow.ones((2, 3, 4, 5)))

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            dist2 = Distribution(flow.float32, flow.float32, True, True, False, -1)

    def test_subclass(self):

        dist = Dist(group_ndims=2)
        self.assertEqual(dist.dtype, flow.float32)
        self.assertEqual(dist.is_continuous, True)
        self.assertEqual(dist.is_reparameterized, True)
        self.assertEqual(dist.group_ndims, 2)

        # shape
        get_v_shape = dist.get_value_shape()
        self.assertListEqual(get_v_shape, [5])
        v_shape = dist.value_shape
        self.assertListEqual(v_shape, [5])

        get_b_shape = dist.get_batch_shape()
        self.assertListEqual(get_b_shape, [2, 3, 4])
        b_shape = dist.batch_shape
        self.assertListEqual(b_shape, [2, 3, 4])

        # sample
        samples_1 = dist.sample()
        self.assertListEqual(samples_1.numpy().flatten().astype(np.int32).tolist(),
                                np.ones((2, 3, 4, 5), dtype=np.int32).flatten().tolist())

        for n in [1, 2]:
            samples_2 = dist.sample(n_samples=n)
            self.assertListEqual(samples_2.numpy().flatten().astype(np.int32).tolist(),
                                np.ones((n, 2, 3, 4, 5), dtype=np.int32).flatten().tolist())

        # log_prob
        given_1 = flow.ones((2, 3, 4, 5))
        log_p_1 = dist.log_prob(given_1)
        self.assertListEqual(log_p_1.numpy().astype(np.int32).tolist(),
                             np.zeros((2)).tolist())

        try:
            dist.log_prob(flow.ones((3, 3, 4, 5)))
        except:
            raise ValueError("broadcast to match batch_shape and value_shape")

        given_2 = flow.ones((1, 2, 3, 4, 5))
        log_p_2 = dist.log_prob(given_2)
        self.assertListEqual(log_p_2.numpy().astype(np.int32).tolist(), np.zeros((1, 2)).tolist())

        given_3 = flow.ones((1, 1, 2, 3, 4, 5))
        log_p_3 = dist.log_prob(given_3)
        self.assertListEqual(log_p_3.numpy().astype(np.int32).tolist(), np.zeros((1, 1, 2)).tolist())

        try:
            Dist(group_event_ndims=1)
        except:
            raise ValueError("has been deprecated")

        try:
            dist2 = Dist(group_ndims=[1, 2])
        except:
            raise TypeError("should be a scalar")

        # shape not fully defined
        dist3 = Dist(shape_fully_defined=False)

        get_v_shape = dist3.get_value_shape()
        self.assertEqual(get_v_shape, None)

        v_shape = dist3.value_shape
        self.assertListEqual(v_shape, [5])

        get_b_shape = dist3.get_batch_shape()
        self.assertListEqual(get_b_shape, [None, 3, 4])
        b_shape = dist3.batch_shape
        self.assertListEqual(b_shape, [2, 3, 4])

        # given type of log_prob and prob
        def _test_log_prob_raise(dtype, given_dtype):
            dist = Dist(dtype=dtype)
            given = flow.cast(flow.Tensor([1]), given_dtype)
            try:
                dist.log_prob(given)
            except:
                ValueError


        _test_log_prob_raise(flow.float32, flow.float64)
        _test_log_prob_raise(flow.float32, flow.int32)
        _test_log_prob_raise(flow.float32, flow.int64)
        _test_log_prob_raise(flow.float64, flow.float32)
        _test_log_prob_raise(flow.float64, flow.int32)
        _test_log_prob_raise(flow.int32, flow.float32)
        _test_log_prob_raise(flow.int32, flow.int64)
        _test_log_prob_raise(flow.int64, flow.int32)
        _test_log_prob_raise(flow.int64, flow.float64)

        # NOTE(Liang Depeng): not support data type
        # _test_log_prob_raise(flow.float32, flow.float16)


