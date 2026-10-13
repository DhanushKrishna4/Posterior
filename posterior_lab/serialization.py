"""Data-only JSON, bounded import and complete unthinned CSV exports."""
import csv
import io
import json
import copy
import numpy as np

FORMAT='posterior-experiment'
VERSION=1
MAX_BYTES=128_000_000

def encode_rng_state(state):
    """JSON has no wide integer type: preserve PCG64's uint128 words as strings."""
    result=copy.deepcopy(state)
    result['state']['state']=str(result['state']['state'])
    result['state']['inc']=str(result['state']['inc'])
    return result

def decode_rng_state(state):
    if not isinstance(state,dict) or set(state)!={'bit_generator','state','has_uint32','uinteger'} or state.get('bit_generator')!='PCG64': raise ValueError('invalid PCG64 state schema')
    nested=state.get('state')
    if not isinstance(nested,dict) or set(nested)!={'state','inc'}: raise ValueError('invalid PCG64 integer state')
    result=copy.deepcopy(state)
    for key in ['state','inc']:
        value=nested[key]
        if isinstance(value,str):
            if not value.isascii() or not value.isdigit() or len(value)>39: raise ValueError('PCG64 wide integers must be unsigned decimal strings')
            value=int(value)
        if type(value)!=int or not 0<=value<2**128: raise ValueError('PCG64 integer exceeds uint128')
        result['state'][key]=value
    if result['state']['inc']%2!=1: raise ValueError('PCG64 increment must be odd')
    for key,maximum in [('has_uint32',1),('uinteger',2**32-1)]:
        if type(result[key])!=int or not 0<=result[key]<=maximum: raise ValueError('invalid PCG64 cached state')
    return result

def clean(value):
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    return value

def dumps(value): return json.dumps(clean(value),allow_nan=False,separators=(',',':'))

def loads(text):
    if not isinstance(text,str) or len(text.encode())>MAX_BYTES: raise ValueError('experiment import is limited to 128 MB')
    result=json.loads(text,parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f'nonfinite JSON constant {v}')))
    def visit(x,depth=0):
        if depth>40: raise ValueError('experiment nesting exceeds 40 levels')
        if isinstance(x,dict):
            if len(x)>512: raise ValueError('object exceeds schema field budget')
            for v in x.values(): visit(v,depth+1)
        elif isinstance(x,list):
            if len(x)>20000: raise ValueError('array exceeds import limit')
            for v in x: visit(v,depth+1)
        elif isinstance(x,float) and not np.isfinite(x): raise ValueError('nonfinite JSON number')
    visit(result)
    return result

def extract_checkpoint(data):
    """Accept browser exports, native snapshots, and bare versioned checkpoints."""
    if not isinstance(data,dict): raise ValueError('experiment must be an object')
    if 'checkpoint' in data:return data['checkpoint']
    if isinstance(data.get('result'),dict) and 'checkpoint' in data['result']:return data['result']['checkpoint']
    return data

def draws_csv(samples,names):
    out=io.StringIO(); writer=csv.writer(out); writer.writerow(['chain','draw',*names])
    for c,chain in enumerate(samples):
        for d,row in enumerate(chain): writer.writerow([c+1,d+1,*row])
    return out.getvalue()

def summaries_csv(diagnostics):
    out=io.StringIO(); fields=['name','status','mean','sd','q05','q50','q95','rhat','ess_bulk','ess_tail','mcse_mean']; writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(diagnostics.get('parameters',[])); return out.getvalue()
