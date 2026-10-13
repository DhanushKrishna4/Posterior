"""Versioned data-only models and a validated fixed-shape computation graph.

References are node names; literal numerical arguments become constants. No model
text is executed. All presets and the Python builder use this compiler.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
import numpy as np
from .autodiff import Tensor, stack
from .transforms import transform
from .distributions import log_density

LIMITS={'dimensions':32,'observations':1000,'nodes':256,'array_elements':10000,'total_elements':40000,'json_bytes':2_000_000}
OPS={'add','sub','mul','div','pow','neg','exp','log','log1p','sigmoid','softplus','logsumexp','sum','mean','index','reshape','dot','matvec','stack','logpdf'}
DISTRIBUTIONS={'normal':{'loc','scale'},'half_normal':{'scale'},'beta':{'alpha','beta'},'bernoulli_logits':{'logits'},'binomial':{'trials'},'multivariate_normal':{'loc','cov'}}

class ModelError(ValueError): pass

def checked_json(value):
    if isinstance(value,str):
        if len(value.encode())>LIMITS['json_bytes']: raise ModelError('model exceeds 2 MB')
        value=json.loads(value)
    text=json.dumps(value,allow_nan=False,sort_keys=True,separators=(',',':'))
    if len(text)>LIMITS['json_bytes']: raise ModelError('model exceeds 2 MB')
    return json.loads(text)

def name_valid(name):
    return isinstance(name,str) and re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,47}',name) is not None

def canonical_model_json(value):
    # JSON/JavaScript serializes 2.0 as 2. Hash numerical meaning consistently,
    # while retaining the original data-only specification in the experiment.
    def normalize(x):
        if isinstance(x,float) and x.is_integer(): return int(x)
        if isinstance(x,list):return [normalize(v) for v in x]
        if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
        return x
    return json.dumps(normalize(value),sort_keys=True,separators=(',',':'),allow_nan=False)

class CompiledModel:
    def __init__(self,spec):
        self.spec=checked_json(spec)
        s=self.spec
        if s.get('version')!=1: raise ModelError('model.version must be 1')
        self.hash=hashlib.sha256(canonical_model_json(s).encode()).hexdigest()
        self.parameters=s.get('parameters',[]); self.nodes=s.get('nodes',[]); self.factors=s.get('factors',[])
        if not isinstance(self.parameters,list) or not self.parameters: raise ModelError('at least one parameter is required')
        if len(self.nodes)+len(self.factors)+len(self.parameters)>LIMITS['nodes']: raise ModelError('graph exceeds 256 nodes')
        self.constants={}; self.shapes={}; self.slices={}; self.dimension=0; self.names=[]; all_names=set(); elements=0
        def register(name):
            if not name_valid(name): raise ModelError(f'invalid name: {name!r}')
            if name in all_names: raise ModelError(f'duplicate name: {name}')
            all_names.add(name)
        for p in self.parameters:
            name=p.get('name'); register(name); shape=p.get('shape',[])
            if not isinstance(shape,list) or len(shape)>1 or any(type(i)!=int or i<1 or i>32 for i in shape): raise ModelError(f'{name}: parameter must be scalar or vector of length <=32')
            size=int(np.prod(shape)); self.shapes[name]=tuple(shape); self.slices[name]=slice(self.dimension,self.dimension+size); self.dimension+=size
            self.names.extend([name] if not shape else [f'{name}[{i}]' for i in range(size)])
            c=p.get('constraint',{'type':'real'}); kind=c.get('type','real')
            if kind not in ('real','positive','interval'): raise ModelError(f'{name}: invalid constraint')
            if kind=='interval' and (not isinstance(c.get('lower'),(int,float)) or not isinstance(c.get('upper'),(int,float)) or not np.isfinite(c['upper']-c['lower']) or c['lower']>=c['upper']): raise ModelError(f'{name}: finite increasing interval required')
        if self.dimension>32: raise ModelError('at most 32 unconstrained parameters')
        data=s.get('data',{})
        if not isinstance(data,dict): raise ModelError('data must map names to arrays')
        for name,value in data.items():
            register(name); a=np.asarray(value,dtype=np.float64)
            if a.size>LIMITS['array_elements'] or a.ndim>2 or (a.ndim and a.shape[0]>1000): raise ModelError(f'{name}: data exceeds array/observation limit')
            if not np.all(np.isfinite(a)): raise ModelError(f'{name}: finite data required')
            a.setflags(write=False); self.constants[name]=a; elements+=a.size
        node_map={}
        for n in self.nodes:
            name=n.get('name'); register(name); node_map[name]=n
            if n.get('op') not in OPS: raise ModelError(f'{name}: unsupported operation {n.get("op")}')
        def numeric(value):
            nonlocal elements
            if isinstance(value,str):
                if value not in all_names: raise ModelError(f'unknown reference: {value}')
                return
            if isinstance(value,(dict,bool)) or value is None: raise ModelError('arguments must be node names or finite numeric constants')
            a=np.asarray(value,dtype=np.float64)
            if a.ndim>2 or a.size>10000 or not np.all(np.isfinite(a)): raise ModelError('invalid or oversized constant')
            elements+=a.size
        def refs(n):
            args=list(n.get('args',[]))
            if n.get('op')=='logpdf': args=[n.get('value')]+list(n.get('params',{}).values())
            for a in args: numeric(a)
            return [a for a in args if isinstance(a,str) and a in node_map]
        edges={name:refs(n) for name,n in node_map.items()}; visiting=set(); visited=set(); self.order=[]
        def visit(name):
            if name in visiting: raise ModelError(f'cycle involving {name}')
            if name in visited: return
            visiting.add(name)
            for dep in edges[name]: visit(dep)
            visiting.remove(name); visited.add(name); self.order.append(node_map[name])
        for name in node_map: visit(name)
        self.constant_names=set(data)
        for n in self.order:
            args=([n.get('value')]+list(n.get('params',{}).values())) if n['op']=='logpdf' else n.get('args',[])
            if all(not isinstance(a,str) or a in self.constant_names for a in args): self.constant_names.add(n['name'])
        factor_names=set()
        for f in self.factors:
            if not name_valid(f.get('name')) or f['name'] in factor_names: raise ModelError('factor names must be unique identifiers')
            factor_names.add(f['name'])
            if 'log_density' in f: numeric(f['log_density'])
            else:
                numeric(f.get('value'))
                for v in f.get('params',{}).values(): numeric(v)
        if not self.factors: raise ModelError('at least one named log-density contribution required')
        if elements>LIMITS['total_elements']: raise ModelError('model arrays exceed total element budget')
        self.derived=s.get('derived',[])
        if not isinstance(self.derived,list) or any(n not in all_names for n in self.derived): raise ModelError('derived quantities must reference named nodes')
        # Execute once to validate fixed shapes and constant distribution parameters.
        try:
            env,jac=self._graph(np.zeros(self.dimension))
            observations=0
            for n in self.order:
                if n['op']=='logpdf':
                    self._validate_distribution(n,env)
                    if not isinstance(n['value'],str) or n['value'] in self.constant_names: observations+=self._ref(n['value'],env).value.size
            for f in self.factors:
                if 'log_density' not in f:
                    self._validate_distribution(f,env)
                    if not isinstance(f['value'],str) or f['value'] in self.constant_names: observations+=self._ref(f['value'],env).value.size
                self._factor(f,env)
            if observations>1000: raise ModelError('observed distribution values exceed 1000 observations')
        except (ValueError,IndexError,TypeError,KeyError,FloatingPointError) as e: raise ModelError(f'graph validation: {e}') from e
        self.report_nodes=s.get('report',[p['name'] for p in self.parameters]+self.derived)
        if not isinstance(self.report_nodes,list) or not self.report_nodes or any(n not in all_names for n in self.report_nodes): raise ModelError('report must contain named nodes')
        if len(set(self.report_nodes))!=len(self.report_nodes): raise ModelError('report names must be distinct')
        self.reporting_names=[]
        for n in self.report_nodes:
            a=env[n].value
            if a.size>32: raise ModelError('derived reports limited to 32 elements each')
            self.reporting_names.extend([n] if a.ndim==0 else [f'{n}[{i}]' for i in range(a.size)])
        if len(self.reporting_names)>128: raise ModelError('at most 128 reported scalar quantities')

    def _ref(self,value,env): return env[value] if isinstance(value,str) else Tensor(value)

    def _validate_distribution(self,f,env):
        d=f.get('distribution'); params=f.get('params',{})
        if d not in DISTRIBUTIONS: raise ModelError(f'unknown distribution: {d}')
        if not DISTRIBUTIONS[d]<=params.keys(): raise ModelError(f'{d}: requires {sorted(DISTRIBUTIONS[d])}')
        allowed=DISTRIBUTIONS[d]|({'prob','logits'} if d=='binomial' else set())
        if set(params)-allowed: raise ModelError(f'{d}: unsupported parameters {sorted(set(params)-allowed)}')
        if d=='binomial' and ('prob' in params)==('logits' in params): raise ModelError('binomial requires exactly one of prob or logits')
        constant=['alpha','beta'] if d=='beta' else ['trials'] if d=='binomial' else ['cov'] if d=='multivariate_normal' else []
        for key in constant:
            ref=params[key]
            if isinstance(ref,str) and ref not in self.constant_names: raise ModelError(f'{d}.{key} must be constant')
        x=self._ref(f['value'],env).value
        if d in ('bernoulli_logits','binomial') and isinstance(f['value'],str) and f['value'] not in self.constant_names: raise ModelError('discrete variables must be observed constants')
        if d=='binomial':
            n=self._ref(params['trials'],env).value
            if np.any((n<0)|(n!=np.floor(n))|(n>1_000_000)): raise ModelError('trials must be fixed nonnegative integers <=1000000')
        if d=='multivariate_normal':
            cov=self._ref(params['cov'],env).value; loc=self._ref(params['loc'],env).value
            if x.ndim!=1 or cov.shape!=(x.size,x.size) or loc.shape!=x.shape: raise ModelError('multivariate normal requires vector value and mean with matching square covariance')
            if not np.allclose(cov,cov.T,rtol=0,atol=1e-12): raise ModelError('covariance must be symmetric')
            try: np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as e: raise ModelError('covariance must be positive definite') from e

    def _factor(self,f,env):
        if 'log_density' in f: return self._ref(f['log_density'],env).sum()
        return log_density(f['distribution'],self._ref(f['value'],env),**{k:self._ref(v,env) for k,v in f['params'].items()}).sum()

    def _graph(self,z):
        env={k:Tensor(v) for k,v in self.constants.items()}; jac=Tensor(0.)
        self.leaves=[]
        for p in self.parameters:
            name=p['name']; leaf=Tensor(z[self.slices[name]].reshape(self.shapes[name]),requires_grad=True); self.leaves.append(leaf)
            env[name],j=transform(leaf,p.get('constraint',{})); jac=jac+j
        for n in self.order:
            op=n['op']; a=[self._ref(x,env) for x in n.get('args',[])]
            if op in ('neg','exp','log','log1p','sigmoid','softplus','pow','sum','mean','logsumexp','index','reshape') and len(a)!=1: raise ModelError(f'{op} requires one argument')
            if op in ('dot','matvec') and len(a)!=2: raise ModelError(f'{op} requires two arguments')
            if op in ('add','sub','mul','div'):
                if len(a)!=2: raise ModelError(f'{op} requires two arguments')
                out={'add':Tensor.__add__,'sub':Tensor.__sub__,'mul':Tensor.__mul__,'div':Tensor.__truediv__}[op](a[0],a[1])
            elif op=='pow':
                power=n.get('power')
                if not isinstance(power,(int,float)) or not np.isfinite(power) or abs(power)>16: raise ModelError('constant power in [-16,16] required')
                out=a[0]**power
            elif op in ('neg','exp','log','log1p','sigmoid','softplus'):
                out=getattr(a[0],'__neg__' if op=='neg' else op)()
            elif op in ('sum','mean','logsumexp'): out=getattr(a[0],op)(n.get('axis'),n.get('keepdims',False))
            elif op=='reshape':
                shape=n.get('shape')
                if not isinstance(shape,list) or len(shape)>2 or any(type(x)!=int or x<=0 or x>10000 for x in shape) or np.prod(shape)>10000: raise ModelError('invalid bounded reshape')
                out=a[0].reshape(shape)
            elif op=='index':
                idx=n.get('index')
                if not (type(idx)==int or isinstance(idx,list) and all(type(i)==int for i in idx)): raise ModelError('index must be a fixed integer or coordinate list')
                out=a[0].index(idx)
            elif op in ('dot','matvec'): out=a[0].matmul(a[1])
            elif op=='stack': out=stack(a,n.get('axis',0))
            elif op=='logpdf': out=log_density(n['distribution'],self._ref(n['value'],env),**{k:self._ref(v,env) for k,v in n['params'].items()})
            if out.value.size>10000: raise ModelError(f'{n["name"]}: result exceeds array budget')
            env[n['name']]=out
        return env,jac

    def value_and_grad(self,z):
        z=np.asarray(z,dtype=np.float64)
        if z.shape!=(self.dimension,) or not np.all(np.isfinite(z)): return -np.inf,np.full(self.dimension,np.nan)
        try:
            with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
                env,jac=self._graph(z); target=jac
                for f in self.factors: target=target+self._factor(f,env)
                if not np.isfinite(target.value): return -np.inf,np.full(self.dimension,np.nan)
                target.backward(); grad=np.concatenate([np.zeros_like(t.value).ravel() if t.grad is None else t.grad.ravel() for t in self.leaves])
                if not np.all(np.isfinite(grad)): return -np.inf,np.full(self.dimension,np.nan)
                return float(target.value),grad
        except (FloatingPointError,OverflowError,np.linalg.LinAlgError): return -np.inf,np.full(self.dimension,np.nan)

    __call__=value_and_grad

    def report(self,z):
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            env,_=self._graph(np.asarray(z,dtype=np.float64))
        values=[env[n].value.ravel() for n in self.report_nodes]
        return np.concatenate(values)

    def contributions(self,z):
        env,jac=self._graph(np.asarray(z,dtype=np.float64))
        return {**{f['name']:float(self._factor(f,env).value) for f in self.factors},'transform_log_jacobian':float(jac.value)}


class Model:
    """Small fluent builder producing the browser's exact JSON representation."""
    def __init__(self,name='Custom model'):
        self.spec={'version':1,'name':name,'parameters':[],'data':{},'nodes':[],'factors':[],'derived':[]}
    def parameter(self,name,shape=None,constraint=None):
        self.spec['parameters'].append({'name':name,'shape':shape or [],'constraint':constraint or {'type':'real'}}); return name
    def data(self,name,value): self.spec['data'][name]=np.asarray(value).tolist(); return name
    def node(self,name,op,*args,**attrs): self.spec['nodes'].append({'name':name,'op':op,'args':list(args),**attrs}); return name
    def observe(self,name,distribution,value,**params): self.spec['factors'].append({'name':name,'distribution':distribution,'value':value,'params':params}); return self
    prior=observe
    def factor(self,name,log_density): self.spec['factors'].append({'name':name,'log_density':log_density}); return self
    def derived(self,name): self.spec['derived'].append(name); return self
    def to_dict(self): return copy.deepcopy(self.spec)
    def compile(self): return CompiledModel(self.spec)

def compile_model(spec): return CompiledModel(spec)
