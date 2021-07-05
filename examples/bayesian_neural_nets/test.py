import oneflow.experimental as flow
import numpy as np

flow.enable_eager_execution()

class MyLayer(flow.nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.p = flow.nn.Parameter(flow.Tensor(1, dtype=flow.float32))
        flow.nn.init.normal_(self.p)

    def forward(self, input):
        t1 = flow.Tensor(np.log(np.random.rand(*[1000,10])), dtype=flow.float32)
        t2 = flow.Tensor(np.log(np.random.rand(*[1000,10])), dtype=flow.float32)
        for i in range(1000):
            event = flow.Tensor(np.log(np.random.rand(1000,10)), dtype=flow.float32)
            a = flow.cast(t2 > event, dtype=flow.float32)
            t1 = flow.Tensor(a * t2 + (1.0 - a) * t1)
        return flow.cat([self.p, self.p], dim=0)

x = flow.Tensor(np.random.randn(10, 1))


mylayer = MyLayer()
mylayer.train()

opt = flow.optim.Adam(parameters=mylayer.parameters(), lr=0.001)

out = mylayer(x)
print(out)
out.backward(gradient=flow.ones_like(out))
print('p.grad: ', mylayer.p.grad)

opt.step()

