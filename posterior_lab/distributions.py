"""Normalized log density primitives, including fixed-event multivariate normals."""
import numpy as np
from scipy.special import betaln, gammaln, xlogy, xlog1py
from .autodiff import Tensor

LOG2PI=float(np.log(2*np.pi))

def log_density(distribution, value, **parameters):
    allowed={'normal':{'loc','scale'},'half_normal':{'scale'},'beta':{'alpha','beta'},'bernoulli_logits':{'logits'},'binomial':{'trials','prob','logits'},'multivariate_normal':{'loc','cov'}}
    if distribution not in allowed or set(parameters)-allowed[distribution]: raise ValueError(f'{distribution}: unsupported distribution parameters')
    x=Tensor.wrap(value); p={k:Tensor.wrap(v) for k,v in parameters.items()}
    if distribution!='multivariate_normal' and np.prod(np.broadcast_shapes(x.value.shape,*(v.value.shape for v in p.values())))>10000: raise ValueError('distribution broadcast exceeds 10000 elements')
    if distribution in ('normal','half_normal'):
        scale=p['scale']; loc=p.get('loc',Tensor(0.))
        if np.any(scale.value<=0): raise FloatingPointError('normal scale must be positive')
        if distribution=='half_normal' and np.any(x.value<0): raise FloatingPointError('half-normal outside support')
        lp=-0.5*((x-loc)/scale)**2-scale.log()-0.5*LOG2PI
        return lp + (np.log(2.) if distribution=='half_normal' else 0.)
    if distribution=='beta':
        a,b=p['alpha'].value,p['beta'].value
        if np.any(a<=0) or np.any(b<=0): raise ValueError('beta shapes must be positive')
        if np.any((x.value<0)|(x.value>1)) or (x._stable_log is None and np.any((x.value==0)|(x.value==1))): raise FloatingPointError('beta outside open support')
        return (a-1)*x.log()+(b-1)*x.log1m()-betaln(a,b)
    if distribution in ('bernoulli_logits','binomial'):
        n=p.get('trials',Tensor(1.)).value
        if np.any(x.value<0) or np.any(x.value>n) or np.any(x.value!=np.floor(x.value)): raise ValueError('count outside support')
        if 'logits' in p:
            lp=-(-p['logits']).softplus(); lq=-p['logits'].softplus()
        else:
            prob=p['prob']
            if np.any((prob.value<0)|(prob.value>1)): raise FloatingPointError('binomial probability outside [0,1]')
            if prob._stable_log is None:
                # 0*log(0) is exactly zero for observed zero counts. Its
                # derivative term is zero too; do not manufacture NaNs.
                with np.errstate(divide='ignore',invalid='ignore'):
                    likelihood=xlogy(x.value,prob.value)+xlog1py(n-x.value,-prob.value)
                    derivative=np.divide(x.value,prob.value,out=np.zeros(np.broadcast_shapes(x.value.shape,prob.value.shape)),where=x.value!=0)-np.divide(n-x.value,1-prob.value,out=np.zeros(np.broadcast_shapes((n-x.value).shape,prob.value.shape)),where=(n-x.value)!=0)
                from .autodiff import unbroadcast
                return gammaln(n+1)-gammaln(x.value+1)-gammaln(n-x.value+1)+Tensor(likelihood,((prob,lambda g:unbroadcast(g*derivative,prob.value.shape)),))
            lp=prob.log(); lq=prob.log1m()
        return gammaln(n+1)-gammaln(x.value+1)-gammaln(n-x.value+1)+x*lp+(n-x)*lq
    if distribution=='multivariate_normal':
        cov=p['cov'].value; loc=p['loc']
        chol=np.linalg.cholesky(cov); precision=np.linalg.solve(cov,np.eye(len(cov)))
        residual=x-loc
        return -0.5*(residual*Tensor(precision).matmul(residual)).sum()-np.log(np.diag(chol)).sum()-0.5*len(cov)*LOG2PI
    raise ValueError(f'unsupported distribution {distribution}')
