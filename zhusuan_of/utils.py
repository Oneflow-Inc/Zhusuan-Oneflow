#!/usr/bin/env python
# -*- coding: utf-8 -*-

import oneflow.experimental as flow

def log_mean_exp(x, dim=None, keepdims=False):
    """
    Oneflow numerically stable log mean of exps across the `dim`.
    :param x: A Tensor.
    :param dim: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keepdims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.
    :return: A Tensor after the computation of log mean exp along given axes of
        x.
    """
    x_max = flow.max(x, dim=dim, keepdim=True)
    ret = flow.log(flow.mean(flow.exp(x - x_max), dim=dim,
                                keepdim=True)) + x_max
    if not keepdims:
        ret = flow.mean(ret, dim=dim)
    return ret

# TODO(Liang Depeng): delete the following test codes.
# flow.enable_eager_execution()

# import zhusuan # import log_mean_exp
# import paddle

# x = paddle.randn([4, 4], 'float32')
# x_nd = x.numpy()
# print(x_nd.shape)

# paddle_result = zhusuan.log_mean_exp(x, dim=1)
# print(paddle_result.numpy())

# of_x = flow.Tensor(x_nd)
# oneflow_result = log_mean_exp(of_x, dim=1)
# print(oneflow_result.numpy())

