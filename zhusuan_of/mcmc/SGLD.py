import math

import oneflow.experimental as flow

__all__ = [
    "SGLD",
]


class SGLD(flow.nn.Module):
    """
        SGLD
    """
    def __init__(self, learning_rate, iters=310):
        super().__init__()
        self.t = 0 
        self.lr = learning_rate
        self.iters = iters
        self.lr_min = 1e-4

    def forward(self, bn, observed, resample=False, step=1):
        if resample:
            self.t = 0
            bn.forward(observed)
            self.t += 1

            self._latent = {k:v.tensor for k,v in bn.nodes.items() if k not in observed.keys()}
            self._latent_k = self._latent.keys()
            self._var_list = [self._latent[k] for k in self._latent_k]

            sample_ = dict(zip(self._latent_k, self._var_list))

            for i in range(len(self._var_list)):
                # self._var_list[i] = self._var_list[i].detach()
                # self._var_list[i].stop_gradient = False
                self._var_list[i] = flow.Tensor(self._var_list[i].numpy(), requires_grad=True)

            return sample_

        for s in range(step):
            observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
            bn.forward(observed_)

            log_joint_ = bn.log_joint()
            grad = flow.autograd.grad(log_joint_, self._var_list, out_grads=flow.ones_like(log_joint_))

            for i,_ in enumerate(grad):
                _lr = max(self.lr_min, self.lr / math.sqrt(self.t))
                #_lr = self.lr / math.sqrt(self.t)
                # epsilon = paddle.normal(shape=self._var_list[i].shape, mean=0.0, std=paddle.sqrt(_lr))
                epsilon = flow.Tensor(*list(self._var_list[i].shape))
                flow.nn.init.normal_(epsilon, 0.0, math.sqrt(_lr))

                self._var_list[i] = self._var_list[i] + 0.5 * _lr * grad[i] + epsilon
                # self._var_list[i] = self._var_list[i].detach()
                # self._var_list[i].stop_gradient = False
                self._var_list[i] = flow.Tensor(self._var_list[i].numpy(), requires_grad=True)

            self.t += 1

        sample_ = dict(zip(self._latent_k, self._var_list))
        return sample_


    def initialize(self):
        self.t = 0

    def sample(self, bn, observed, resample=False, step=1):
        """
        Return the sampling `Operation` that runs a SGMCMC iteration and the
        statistics collected during it, given the log joint function (or a
        :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance), observed
        values and latent variables.
        """
        return self.forward(bn, observed, resample, step)
