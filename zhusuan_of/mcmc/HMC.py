
import numpy as np

import oneflow.experimental as flow

from zhusuan_of.distributions import Normal


__all__ = [
    "HMC",
]


class HMC(flow.nn.Module):
    """
        HMC
    """
    def __init__(self, step_size=0.25, n_leapfrogs=25, iters=1):
        super().__init__()
        self.t = 0 
        self.step_size = step_size
        self.n_leapfrogs = n_leapfrogs
        self.iters = iters

    def forward(self, bn, observed, initial_position):
        
        if initial_position:
            observed_ = {**initial_position, **observed}
        else:
            observed_ = observed
        bn.forward(observed_)

        q0 = [[k, v.tensor] for k, v in bn.nodes.items() if k not in observed.keys()]
        normals = [[k, Normal(mean=flow.zeros(tuple(v.shape), dtype=flow.float32), std=flow.ones(tuple(v.shape), dtype=flow.float32))]\
                    for k,v in q0]
        
        for e in range(self.iters):
            q1 = [[k, flow.Tensor(v, requires_grad=True)] for k,v in q0]
            p0 = [[k, v.sample()] for k,v in normals]
            p1 = [[k, flow.Tensor(v, requires_grad=True)] for k,v in p0]

            ###### leapfrog integrator
            for s in range(self.n_leapfrogs):
                observed_ = {**dict(q1), **observed}
                bn.forward(observed_)
                log_joint_ = bn.log_joint()
                q_v = [v for _, v in q1]
                q_grad = flow.autograd.grad(log_joint_, q_v, out_grads=flow.ones_like(log_joint_))

                for i,_ in enumerate(q_grad):
                    p1[i][1] = p1[i][1] + self.step_size * q_grad[i] / 2.0
                    q1[i][1] = q1[i][1] + self.step_size * p1[i][1]
                    # p1[i][1] = p1[i][1].detach()
                    # p1[i][1].stop_gradient = False
                    p1[i][1] = flow.Tensor(p1[i][1].numpy(), requires_grad=True)

                    # q1[i][1] = q1[i][1].detach()
                    # q1[i][1].stop_gradient = False

                    q1[i][1] = flow.Tensor(q1[i][1].numpy(), requires_grad=True)

                observed_ = {**dict(q1), **observed}
                q_v = [v for _,v in q1]
                bn.forward(observed_)
                log_joint_ = bn.log_joint()
                q_grad = flow.autograd.grad(log_joint_, q_v, out_grads=flow.ones_like(log_joint_))
                
                for i,_ in enumerate(q_grad):
                    p1[i][1] = p1[i][1] + self.step_size * q_grad[i] / 2.0
                    # p1[i][1] = p1[i][1].detach()
                    # p1[i][1].stop_gradient = False
                    p1[i][1] = flow.Tensor(p1[i][1].detach(), requires_grad=True)

            ###### reverse p1
            for i,_ in enumerate(p1):
                p1[i][1] = -1 * p1[i][1]


            ###### M-H step
            observed_ = {**dict(q0), **observed}
            bn.forward(observed_)
            log_prob_q0 = bn.log_joint()
            log_prob_p0 = None
            for i,_ in enumerate(p0):
                len_q = len(log_prob_q0.shape)
                len_p = len(p0[i][1].shape)
                assert(len_p >= len_q)
                if len_p > len_q:
                    dims = [i for i in range(len_q - len_p, 0)]
                    try:
                        log_prob_p0 = log_prob_p0 + flow.sum(p0[i][1], dim=dims)
                    except:
                        log_prob_p0 = flow.sum(p0[i][1], dim=dims)
                else:
                    try:
                        log_prob_p0 = log_prob_p0 + p0[i][1]
                    except:
                        log_prob_p0 = p0[i][1]

            observed_ = {**dict(q1), **observed}
            bn.forward(observed_)
            log_prob_q1 = bn.log_joint()
            log_prob_p1 = None
            for i,_ in enumerate(p1):
                len_q = len(log_prob_q0.shape)
                len_p = len(p1[i][1].shape)
                assert(len_p >= len_q)
                if len_p > len_q:
                    dims = [i for i in range(len_q - len_p, 0)]
                    try:
                        log_prob_p1 = log_prob_p1 + flow.sum(p1[i][1], dim=dims)
                    except:
                        log_prob_p1 = flow.sum(p1[i][1], dim=dims)
                else:
                    try:
                        log_prob_p1 = log_prob_p1 + p1[i][1]
                    except:
                        log_prob_p1 = p1[i][1]

            assert(log_prob_q0.shape == log_prob_p1.shape)

            acceptance = log_prob_q1 + log_prob_p1 - log_prob_q0 - log_prob_p0

            for i,_ in enumerate(q1):
                event = flow.Tensor(np.log(np.random.rand(*q1[i][1].shape)), dtype=flow.float32)
                a = flow.cast(acceptance > event, dtype=flow.float32)
                q0[i][1] = flow.Tensor(a * q1[i][1] + (1.0 - a) * q0[i][1])

            #print(q0[0][1])
            #print(dir(bn))
            #print(bn.clear_gradients())

        sample_ = dict(q0)
        return sample_

    def sample(self, bn, observed, initial_position=None):
        """
            sample
        """
        return self.forward(bn, observed, initial_position)


