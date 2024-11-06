from rpython.rlib.rarithmetic import r_longfloat, r_int64

import math

class Children(object):
    def __init__(self, out=None, other=None):
        self.out = out
        self.other = other

    def tolist(self):
        if self.out is None:
            if self.other is None:
                return []
            else:
                return [self.other]
        else:
            if self.other is None:
                return [self.out]
            else:
                return [self.out, self.other]


class Value(object):
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=Children(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        # self._backward = lambda: None
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._other_pow = 0

    def add(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, Children(self, other), '+')

        # def _backward():
        #     self.grad += out.grad
        #     other.grad += out.grad
        # out._backward = _backward

        return out

    def _backward_add(self):
        # out, other = self._prev
        out = self._prev.out
        other = self._prev.other
        self.grad += out.grad
        other.grad += out.grad

    def mul(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, Children(self, other), '*')

        # def _backward():
        #     self.grad += other.data * out.grad
        #     other.grad += self.data * out.grad
        # out._backward = _backward

        return out

    def _backward_mul(self):
        # out, other = self._prev
        out = self._prev.out
        other = self._prev.other
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    def pow(self, other):
        # assert isinstance(other, (r_int64, r_longfloat)), "only supporting int/float powers for now"
        if isinstance(other, int):
            op = '**%d' % (other)
        elif isinstance(other, float):
            op = '**%f' % (other)
        else:
            raise Exception

        out = Value(math.pow(self.data, other), Children(self,), op)
        self._other_pow = other

        # def _backward():
        #     self.grad += (other * self.data**(other-1)) * out.grad
        # out._backward = _backward

        return out

    def _backward_pow(self):
        # out, = self._prev
        out = self._prev.out
        other = self._other_pow
        self.grad += (other * math.pow(self.data, other-1)) * out.grad

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, Children(self,), 'ReLU')

        # def _backward():
        #     self.grad += (out.data > 0) * out.grad
        # out._backward = _backward

        return out

    def _backward_relu(self):
        # out, other = self._prev
        out = self._prev.out
        self.grad += (out.data > 0) * out.grad

    def _backward(self):
        if self._op == '+':
            return self._backward_add()
        elif self._op == '*':
            return self._backward_mul()
        elif self._op.startswith('**'):
            return self._backward_pow()
        elif self._op == 'ReLU':
            return self._backward_relu()
        else:
            return None

    def _build_topo(self, visited, topo):
        if self not in visited:
            visited[self] = None
            child = self._prev
            if child.out:
                child.out._build_topo(visited, topo)
            if child.other:
                child.other._build_topo(visited, topo)
            topo.append(self)

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = {}
        # def build_topo(v):
        #     if v not in visited:
        #         visited.add(v)
        #         print(v)
        #         for child in v._prev:
        #             build_topo(child)
        #         topo.append(v)
        # build_topo(self)
        self._build_topo(visited, topo)

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
        s = "Value(data=" + str(self.data) + ", grad=" + str(self.grad) +  ")"
        return s
