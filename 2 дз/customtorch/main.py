import nn
from pytorch_lightning import seed_everything

#seed_everything(0)

class MLP(nn.Module):

    def __init__(self, nin, nouts, learning_rate):
        self.learning_rate = learning_rate
        sz = [nin]
        sz.extend(nouts)
        self.layers = [nn.Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        repr = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [{repr}]"

 
    def step(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for w in neuron.w:
                    w -= self.learning_rate * w.grad
                neuron.b -= self.learning_rate * neuron.b.grad


model = MLP(3, [4, 4, 1], learning_rate = 0.1)
loss = nn.MSELoss()


xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

for k in range(50):
    
    model.zero_grad()
    
    # forward
    predict = model(xs)

    # calculate loss (mean square error)
    loss_val = loss(ys, predict)
    acc = sum([1 for i in range(len(ys)) if ys[i] == round(predict[i]).data]) / len(ys)
    
    # backward (zero_grad + backward)
    loss_val.backward()
    
    # update
    model.step()
    
    if k % 1 == 0:
        print(f"step {k} loss {loss_val.data}, accuracy {acc*100}%")
print([round(i.data, 2) for i in predict])