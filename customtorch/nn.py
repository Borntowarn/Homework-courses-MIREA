import random
from autograd import Value

class Module:

    def zero_grad(self):
        for layer in self.parameters():
            for neuron in layer:
                for w in neuron:
                    w.grad = 0

    def parameters(self):
        _parameters = []
        for param in self.__dict__:
            try:
                for in_param in getattr(self, param):
                    if isinstance(in_param, Module):
                        _parameters.append(in_param.parameters())
                    elif isinstance(in_param, Value):
                        _parameters.append(in_param)
            except:
                if isinstance(getattr(self, param), Module):
                    _parameters.append(getattr(self, param).parameters())
                elif isinstance(getattr(self, param), Value):
                    _parameters.append(getattr(self, param))
        
        return _parameters
    

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.gauss(0, 1)) for _ in range(nin)]
        self.b = Value(random.gauss(0, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum([self.w[i] * x[i] for i in range(len(x))]) + self.b
        return act.relu() if self.nonlin else act #act.step() для ступенчатой ф-ии

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, kwargs['nonlin']) for i in range(nout)]

    def __call__(self, x):
        out = []
        for item in x:
            res = [neuron(item) for neuron in self.neurons]
            out.append(res[0] if len(res) == 1 else res)
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MSELoss(Module):
    
    def __call__(self, y_true, y_pred):
        res = 0
        for true, pred in zip(y_true, y_pred):
            res += (true - pred)**2
        return res / len(y_pred)

class MLP(Module):

    def __init__(self, nin, nouts, learning_rate):
        self.learning_rate = learning_rate
        sz = [nin]
        sz.extend(nouts)
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # def parameters(self):
    #     res = []
    #     for layers in self.layers:
    #         res.extend(layers.parameters())
    #     return res

    def __repr__(self):
        repr = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [{repr}]"

    def zero_grad(self):
        for layer in self.layers: layer.zero_grad()
        
    def step(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for w in neuron.w:
                    w -= self.learning_rate * w.grad
                neuron.b -= self.learning_rate * neuron.b.grad


if __name__ == '__main__':
    model = MLP(3, [4, 4, 1], learning_rate = 0.1)
    loss = MSELoss()

    for i in model.parameters():
        print(i)