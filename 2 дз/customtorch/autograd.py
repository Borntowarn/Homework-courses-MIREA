class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data: float, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None # function 
        self._prev = set(_children) # set of Value objects
        self._op = _op # the op that produced this node, string ('+', '-', ....)


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data) # Standart expression

        def _backward():
            # Calculating the derivative of the sum
            self.grad += out.grad 
            other.grad += out.grad
        out._backward = _backward
        
        # Add children to resulting expression
        out._prev.add(other)
        out._prev.add(self)
        

        return out


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)

        def _backward():
            # Calculating the derivative of the product
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        # Add children to resulting expression
        out._prev.add(other)
        out._prev.add(self)
        
        
        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other)

        def _backward():
            # Calculating the derivative of a power func
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        
        # Add children to resulting expression
        out._prev.add(self)

        return out


    def relu(self):
        out = Value(self.data) if self.data > 0 else Value(0)

        def _backward():
            # Calculating the derivative of the ReLU
            self.grad += out.grad if out.data > 0 else 0
        out._backward = _backward

        # Add children to resulting expression
        out._prev.add(self)
        
        return out
    
    
    def exp(self):
        import math
        
        out = Value(math.e ** self.data)

        def _backward():
            # Calculating the derivative of a power func
            self.grad += out.data * out.grad
        out._backward = _backward
        
        # Add children to resulting expression
        out._prev.add(self)

        return out
    
    
    def softmax(input):
        e = [item.exp() for item in input]
        s = sum(e)
        out = [item / s for item in e]
        
        return out


    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __round__(self, n = 0):
        return Value(round(self.data, n))