"""Predictive simulation has its own RNG: viewing plots cannot advance a chain."""
import numpy as np
from .examples import verified_preset

def predict(spec,model_draws,seed=9941,count=200):
    rows=[row for chain in model_draws for row in chain]
    if not rows: return {'status':'unavailable'}
    draws=np.asarray(rows,dtype=np.float64)
    rng=np.random.default_rng(np.random.SeedSequence([seed,9301])); chosen=draws[rng.integers(0,len(draws),size=min(count,1000))]
    meta=spec.get('metadata',{}); c=meta.get('config',{}); name=meta.get('preset')
    if name not in ('regression','beta') or verified_preset(spec) is None: return {'status':'not specified for this graph'}
    if name=='regression':
        x=np.linspace(min(c['x'])-.3,max(c['x'])+.3,100); means=chosen[:,0,None]+chosen[:,1,None]*x; future=means+rng.normal(0,c['noise'],means.shape)
        return {'status':'available','x':x.tolist(),'lines':means[:30].tolist(),'mean_band':np.quantile(means,[.05,.95],axis=0).tolist(),'predictive_band':np.quantile(future,[.05,.95],axis=0).tolist(),'label':'sample-based central 90% mean-response and future-observation intervals'}
    if name=='beta': return {'status':'available','future_successes':rng.binomial(c['trials'],chosen[:,0]).tolist(),'trials':c['trials']}
    return {'status':'not specified'}
