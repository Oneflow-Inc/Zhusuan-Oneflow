import oneflow.experimental as flow
import numpy as np

from .base import Distribution

__all__ = [
    'Bernoulli',
]

class Bernoulli(Distribution):
    def __init__(self,
                 dtype=flow.float32,
                 param_dtype=flow.float32,
                 is_continues=False,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(Bernoulli, self).__init__(dtype, 
                             param_dtype, 
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)
        self._probs = kwargs['probs']

    @property
    def probs(self):
        """The odds of probabilities of being 1."""
        return self._probs

    def _batch_shape(self):
        return self.probs.shape

    def _sample(self, n_samples=1, **kwargs):
        if n_samples > 1:
            sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
            _probs = self._probs * flow.ones(tuple(sample_shape_))
        else:
            _probs = self._probs

        _probs *= flow.cast(_probs <= 1, self.param_dtype)
        pre_d = "cuda" if _probs.device.type == "cuda" else "cpu"
        _probs = _probs.to("cpu")
        sample_ = flow.bernoulli(_probs)
        sample_ = sample_.to(pre_d)
        
        sample_ = flow.cast(sample_, self.dtype)

        self.sample_cache = sample_
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        ## Log Prob
        if len(sample.shape) > len(self._probs.shape):
            sample_shape_ = np.concatenate([[sample.shape[0]], self.batch_shape], axis=0).tolist()
            _probs = self._probs * flow.ones(tuple(sample_shape_), dtype=self.param_dtype)
        else:
            _probs = self._probs

        # add 1e-8 for numerical stable
        _probs = _probs + 1e-8
        log_prob = sample * flow.log( _probs ) + (1 - sample) * flow.log(1 - _probs )
        log_prob = flow.cast(log_prob, self.dtype)

        return log_prob
