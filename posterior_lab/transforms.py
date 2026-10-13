"""Model transforms: theta=T(z), once-only log Jacobians."""
import numpy as np
from .autodiff import Tensor

def transform(z, constraint):
    kind=constraint.get('type','real')
    if kind=='real': return z, Tensor(0.)
    if kind=='positive': return z.exp(), z.sum()
    if kind=='interval':
        lo,hi=constraint['lower'],constraint['upper']
        p=z.sigmoid()
        theta=p if lo==0 and hi==1 else lo+(hi-lo)*p
        return theta, (np.log(hi-lo)-(-z).softplus()-z.softplus()).sum()
    raise ValueError(f'unknown constraint: {kind}')

def inverse(value, constraint):
    x=np.asarray(value,dtype=np.float64); kind=constraint.get('type','real')
    if kind=='real': return x
    if kind=='positive':
        if np.any(x<=0): raise ValueError('positive initial value required')
        return np.log(x)
    p=(x-constraint['lower'])/(constraint['upper']-constraint['lower'])
    if np.any((p<=0)|(p>=1)): raise ValueError('initial value must lie in open interval')
    return np.log(p)-np.log1p(-p)
