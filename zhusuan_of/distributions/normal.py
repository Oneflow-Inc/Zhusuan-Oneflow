import numpy as np
import oneflow.experimental as flow

from .base import Distribution

__all__ = [
    'Normal',
]


class Normal(Distribution):
    def __init__(self,
                 dtype=flow.float32,
                 param_dtype=flow.float32,
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(Normal, self).__init__(dtype, 
                             param_dtype, 
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)
        try:
            self._std = flow.cast(flow.Tensor([kwargs['std']], requires_grad=True), dtype=self.dtype) \
                if type(kwargs['std']) in [type(1.), type(1)] else kwargs['std']
            self._logstd = flow.log(self._std)
        except:
            self._logstd = flow.cast(flow.Tensor([kwargs['logstd']], requires_grad=True), self.dtype) \
                if type(kwargs['logstd']) in [type(1.), type(1)] else kwargs['logstd']
            self._std = flow.exp(self._logstd)

        self._mean = kwargs['mean']

    @property
    def mean(self):
        """The mean of the Normal distribution."""
        return self._mean

    @property
    def logstd(self):
        """The log standard deviation of the Normal distribution."""
        try:
            return self._logstd
        except:
            self._logstd = flow.log(self._std)
            return self._logstd

    @property
    def std(self):
        """The standard deviation of the Normal distribution."""
        return self._std

    def _sample(self, n_samples=1, **kwargs):

        if n_samples > 1:
            _shape = [n_samples]
            _shape = _shape + list(self._mean.shape)
            _len = len(self._std.shape)
            _std = flow.tile(self._std, reps=(n_samples, *_len*[1]))
            _mean = flow.tile(self._mean, reps=(n_samples, *_len*[1]))
        else:
            _shape = self._mean.shape
            _std = self._std + 0.
            _mean = self._mean + 0.

        if self.is_reparameterized:
            d = "cuda" if _mean.is_cuda else "cpu"
            epsilon = flow.Tensor(np.random.normal(size=tuple(list(_shape))), device=flow.device(d))

            sample_ = _mean + _std * epsilon
        else:
            _std = _std.detach()
            _mean = _mean.detach()
            d = "cuda" if _mean.is_cuda else "cpu"
            epsilon = flow.Tensor(np.random.normal(size=tuple(list(_shape))), device=flow.device(d))
            sample_ = _mean + _std * epsilon

        self.sample_cache = sample_
        if n_samples > 1:
            assert(sample_.shape[0] == n_samples)
        return sample_

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        if sample.is_cuda:
            self._std = self._std.to('cuda')
            self._mean = self._mean.to('cuda')

        if len(sample.shape) > len(self._mean.shape):
            n_samples = sample.shape[0]
            _len = len(self._std.shape)
            _std = flow.tile(self._std, reps=(n_samples, *_len*[1])) 
            _mean = flow.tile(self._mean, reps=(n_samples, *_len*[1])) 
        else:
            _std = self._std
            _mean = self._mean

        ## Log Prob
        if not self.is_reparameterized:
            _mean = _mean.detach()
            _std = _std.detach()
        logstd = flow.log(_std)
        c = -0.5 * np.log(2 * np.pi)
        precision = flow.exp(-2 * logstd)
        log_prob = c - logstd - 0.5 * precision * flow.square(sample - _mean)
        return log_prob
