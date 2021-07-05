#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import paddle

import oneflow.experimental as flow

import matplotlib.pyplot as plt
import os

__all__ = [
    'test_and_save_distribution_img',
    'test_1parameter_log_prob_shape_same',
    'test_2parameter_log_prob_shape_same',
    'test_1parameter_sample_shape_same',
    'test_2parameter_sample_shape_same',
    'test_batch_shape_1parameter',
    'test_batch_shape_2parameter_univariate'
]

def test_and_save_distribution_img( distribution,
                                    hist_folder=os.path.join(
                                        os.path.dirname(__file__),'hist_images')):
    # Test sample hist and save image to histogram folder
    if not os.path.isdir(hist_folder):
        os.mkdir(hist_folder)

    samples = distribution.sample(10000).numpy().flatten()

    dist_name = distribution.__class__.__name__
    img_path = os.path.join(hist_folder, dist_name+'.png')


    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))
    ax0.hist(samples, 200,  histtype='bar', facecolor='blue', alpha=0.75)
    ## Draw pdf
    ax0.set_title('pdf')
    ax1.hist(samples, 100,  histtype='bar', facecolor='pink',
             alpha=0.75, cumulative=True, rwidth=0.8)
    # Draw cdf
    ax1.set_title("cdf")
    fig.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig(img_path)


def test_1parameter_log_prob_shape_same(
        test_class, Distribution, make_param, make_given):

    def _test_dynamic(param_shape, given_shape, target_shape):

        param = flow.cast(flow.Tensor(make_param(param_shape)), flow.float32)
        dist = Distribution(param)

        given = flow.cast(flow.Tensor(make_given(given_shape)), flow.float32)
        log_p = dist.log_prob(given)
        test_class.assertEqual(list(log_p.shape), target_shape)

    _test_dynamic([2, 3], [1, 3], [2, 3])
    _test_dynamic([1, 3], [2, 2, 3], [2, 2, 3])
    _test_dynamic([1, 5], [1, 2, 3, 1], [1, 2, 3, 5])


def test_2parameter_log_prob_shape_same(
        test_class, Distribution, make_param1, make_param2, make_given):

    def _test_dynamic(param1_shape, param2_shape, given_shape,
                      target_shape):
        param1 = flow.cast(flow.Tensor(make_param1(param1_shape)), flow.float32)
        param2 = flow.cast(flow.Tensor(make_param2(param2_shape)), flow.float32)
        dist = Distribution(param1, param2)
        given = flow.cast(flow.Tensor(make_given(given_shape)), flow.float32)
        log_p = dist.log_prob(given)
        test_class.assertEqual( list(log_p.shape), target_shape)

    _test_dynamic([2, 3], [2, 1], [1, 3], [2, 3])
    _test_dynamic([1, 3], [1, 1], [2, 1, 3], [2, 1, 3])
    _test_dynamic([1, 5], [3, 1], [1, 2, 1, 1], [1, 2, 3, 5])


def test_1parameter_sample_shape_same(
        test_class, Distribution, make_param):

    def _test_dynamic(param_shape, n_samples, target_shape):
        param = flow.cast(flow.Tensor(make_param(param_shape)), flow.float32)
        dist = Distribution(param)
        samples = dist.sample(n_samples)
        test_class.assertEqual(list(samples.shape), target_shape)

    _test_dynamic([2, 3], 1, [2, 3])
    # if not only_one_sample:
    _test_dynamic([1, 3], 2, [2, 1, 3])
    _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])

def test_2parameter_sample_shape_same(
        test_class, Distribution, make_param1, make_param2):

    def _test_dynamic(param1_shape, param2_shape, n_samples,
                      target_shape):
        param1 = flow.cast(flow.Tensor(make_param1(param1_shape)), flow.float32)
        param2 = flow.cast(flow.Tensor(make_param2(param2_shape)), flow.float32)
        dist = Distribution(param1, param2)
        samples = dist.sample(n_samples)
        test_class.assertEqual(list(samples.shape), target_shape)

    # TODO: Sample_shape will be [n_samples, batch_shape]
    _test_dynamic([2, 3], [2, 1], 1, [ 2, 3])
    _test_dynamic([1, 3], [2, 1], 2, [2, 2, 3])
    _test_dynamic([2, 1, 5], [1, 3, 1], 3, [3, 2, 3, 5])


def test_batch_shape_1parameter(
        test_class, Distribution, make_param):

    # dynamic
    def _test_dynamic(param_shape):
        param = flow.cast(flow.Tensor(make_param(param_shape)), flow.float32)
        dist = Distribution(param)
        test_class.assertEqual(list(dist.batch_shape), param_shape)

    _test_dynamic([2])
    _test_dynamic([2, 3])
    _test_dynamic([2, 1, 4])


def test_batch_shape_2parameter_univariate(
        test_class, Distribution, make_param1, make_param2):

    # dynamic
    def _test_dynamic(param1_shape, param2_shape, target_shape):
        param1 = paddle.cast(paddle.to_tensor(make_param1(param1_shape)),'float32')
        param2 = paddle.cast(paddle.to_tensor(make_param2(param2_shape)),'float32')
        dist = Distribution(param1, param2)
        # test_class.assertTrue(np.array(dist.batch_shape).dtype is np.int32)
        test_class.assertEqual( dist.batch_shape, target_shape)

    # _test_dynamic([2, 3], [], [2, 3])
    _test_dynamic([2, 3], [3], [2, 3])
    _test_dynamic([2, 1, 4], [2, 3, 4], [2, 3, 4])
    _test_dynamic([2, 3, 5], [3, 1], [2, 3, 5])
    # try:
    #     _test_dynamic([2, 3, 5], [3, 2], None)
    # except:
    #     AssertionError("Incompatible shapes")


