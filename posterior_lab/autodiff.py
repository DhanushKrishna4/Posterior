"""Fresh reverse-mode tapes. Each edge accumulates, including repeated operands."""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logsumexp as scipy_logsumexp


def unbroadcast(g, shape):
    g = np.asarray(g, dtype=np.float64)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


class Tensor:
    __array_priority__ = 1000

    def __init__(self, value, parents=(), requires_grad=False):
        self.value = np.asarray(value, dtype=np.float64)
        self.parents = parents
        self.requires_grad = requires_grad or any(p.requires_grad for p, _ in parents)
        self.grad = None
        self._stable_log = None
        self._stable_log1m = None

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def binary(self, other, value, da, db):
        b = self.wrap(other)
        if np.prod(np.broadcast_shapes(self.value.shape,b.value.shape))>10000:
            raise ValueError('broadcast result exceeds 10000 elements')
        return Tensor(value(self.value, b.value), (
            (self, lambda g: unbroadcast(g * da(self.value, b.value), self.value.shape)),
            (b, lambda g: unbroadcast(g * db(self.value, b.value), b.value.shape))))

    def __add__(self, b): return self.binary(b, np.add, lambda a,b: 1, lambda a,b: 1)
    def __radd__(self, b): return self + b
    def __sub__(self, b): return self.binary(b, np.subtract, lambda a,b: 1, lambda a,b: -1)
    def __rsub__(self, b): return self.wrap(b) - self
    def __mul__(self, b): return self.binary(b, np.multiply, lambda a,b: b, lambda a,b: a)
    def __rmul__(self, b): return self * b
    def __truediv__(self, b): return self.binary(b, np.divide, lambda a,b: 1/b, lambda a,b: -a/b**2)
    def __rtruediv__(self, b): return self.wrap(b) / self
    def __neg__(self): return self * -1

    def __pow__(self, power):
        # A literal power is part of the static graph, never a latent exponent.
        return Tensor(self.value ** power, ((self, lambda g: g * (np.zeros_like(self.value) if power == 0 else power * self.value ** (power-1))),))

    def exp(self):
        value = np.exp(self.value)
        return Tensor(value, ((self, lambda g: g * value),))

    def log(self):
        if self._stable_log is not None: return self._stable_log()
        return Tensor(np.log(self.value), ((self, lambda g: g / self.value),))

    def log1p(self):
        return Tensor(np.log1p(self.value), ((self, lambda g: g / (1+self.value)),))

    def log1m(self):
        if self._stable_log1m is not None: return self._stable_log1m()
        return (-self).log1p()

    def softplus(self):
        return Tensor(np.logaddexp(0, self.value), ((self, lambda g: g * expit(self.value)),))

    def sigmoid(self):
        value = expit(self.value)
        result = Tensor(value, ((self, lambda g: g * expit(self.value) * expit(-self.value)),))
        result._stable_log = lambda: -(-self).softplus()
        result._stable_log1m = lambda: -self.softplus()
        return result

    def sum(self, axis=None, keepdims=False):
        if isinstance(axis, list): axis = tuple(axis)
        def reverse(g):
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis,int) else axis
                for a in sorted(x % self.value.ndim for x in axes): g = np.expand_dims(g, a)
            return np.broadcast_to(g, self.value.shape)
        return Tensor(self.value.sum(axis=axis, keepdims=keepdims), ((self, reverse),))

    def mean(self, axis=None, keepdims=False):
        axes = tuple(range(self.value.ndim)) if axis is None else ((axis,) if isinstance(axis,int) else tuple(axis))
        return self.sum(axis, keepdims) / np.prod([self.value.shape[i] for i in axes])

    def reshape(self, shape):
        return Tensor(self.value.reshape(shape), ((self, lambda g: g.reshape(self.value.shape)),))

    def index(self, index):
        index = tuple(index) if isinstance(index, list) else index
        def reverse(g):
            result = np.zeros_like(self.value)
            np.add.at(result, index, g)
            return result
        return Tensor(self.value[index], ((self, reverse),))

    def matmul(self, other):
        b = self.wrap(other)
        if self.value.ndim == b.value.ndim == 1:
            if self.value.shape!=b.value.shape: raise ValueError('dot requires matching vector shapes')
            return (self*b).sum()
        if self.value.ndim == 2 and b.value.ndim == 1:
            return Tensor(self.value @ b.value, ((self, lambda g: np.outer(g,b.value)), (b, lambda g: self.value.T @ g)))
        raise ValueError('dot/matvec requires vector·vector or matrix·vector')

    def logsumexp(self, axis=None, keepdims=False):
        if isinstance(axis,list): axis=tuple(axis)
        normalizer = scipy_logsumexp(self.value,axis=axis,keepdims=True)
        value = normalizer if keepdims else (normalizer.squeeze() if axis is None else np.squeeze(normalizer,axis=axis))
        weights = np.exp(self.value-normalizer)
        def reverse(g):
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis,int) else axis
                for a in sorted(x % self.value.ndim for x in axes): g=np.expand_dims(g,a)
            return g*weights
        return Tensor(value,((self,reverse),))

    def backward(self):
        if self.value.size != 1: raise ValueError('reverse mode requires a scalar output')
        order, seen, todo = [], set(), [(self,False)]
        while todo:
            node, done=todo.pop()
            if done: order.append(node); continue
            if id(node) in seen: continue
            seen.add(id(node)); todo.append((node,True))
            todo.extend((p,False) for p,_ in node.parents if p.requires_grad)
        for node in order: node.grad = np.zeros_like(node.value)
        self.grad = np.ones_like(self.value)
        for node in reversed(order):
            for parent, reverse in node.parents:
                if parent.requires_grad: parent.grad += reverse(node.grad)


def stack(tensors, axis=0):
    tensors = [Tensor.wrap(x) for x in tensors]
    if sum(t.value.size for t in tensors)>10000: raise ValueError('stack exceeds 10000 elements')
    return Tensor(np.stack([t.value for t in tensors],axis=axis), tuple((t, lambda g,i=i: np.take(g,i,axis=axis)) for i,t in enumerate(tensors)))
