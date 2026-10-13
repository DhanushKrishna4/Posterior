"""Narrow JSON-only browser/CLI boundary; runtime is the sole inference authority."""
import numpy as np
from .examples import PRESETS,preset,reference
from .models import compile_model
from .runtime import Experiment
from .serialization import dumps,loads,extract_checkpoint
from .prediction import predict

_run=None

def dispatch(request):
    global _run
    op=request.get('op')
    if op=='presets': return {'presets':[preset(x[0]) for x in PRESETS]}
    if op=='preset':
        result=preset(request.get('name',request.get('id')),request.get('config')); result['reference']=reference(result['spec']); return result
    if op=='compile':
        m=compile_model(request['spec']); lp,g=m(np.asarray(request.get('position',np.zeros(m.dimension))))
        return {'spec':m.spec,'hash':m.hash,'dimension':m.dimension,'names':m.reporting_names,'logp':lp,'gradient':g,'contributions':m.contributions(np.zeros(m.dimension))}
    if op=='reference': return reference(request['spec'])
    if op=='start':
        new=Experiment(request['spec'],request.get('settings')); _run=new; return _run.snapshot()
    if op=='restore':
        data=extract_checkpoint(request.get('checkpoint',request.get('experiment')))
        new=Experiment.restore(data); _run=new; return _run.snapshot()
    if _run is None: raise ValueError('no active experiment')
    if op=='step':
        result=_run.step(request.get('count',1)); return _run.snapshot() if request.get('full',False) else result
    if op=='snapshot': return _run.snapshot(request.get('diagnostics',True))
    if op=='delta': return _run.delta()
    if op=='status':
        if request['status'] not in ('running','paused','cancelled'): raise ValueError('invalid requested status')
        if request['status']=='running' and _run.status in ('complete','failed','budget_exhausted'): raise ValueError('finished run cannot resume; create a new experiment')
        _run.set_status(request['status'],request.get('reason')); return _run.snapshot()
    if op=='predict': return predict(_run.spec,_run.samples,seed=_run.settings['seed'])
    raise ValueError(f'unknown operation: {op}')

def call(text):
    try: return dumps(dispatch(loads(text)))
    except Exception as e: return dumps({'error':f'{type(e).__name__}: {e}'})
