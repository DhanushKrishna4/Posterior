"""Reproducible end-to-end engine evaluation, sequential and CPU only.

Run after the inexpensive pilot and correctness suite. The difficult examples
report outcomes rather than requiring a scripted success/failure. Raw retained
draws and exact checkpoints are saved as deterministic gzip JSON artifacts.
"""
import argparse
import gzip
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from posterior_lab.examples import preset
from posterior_lab.runtime import Experiment
from posterior_lab.serialization import dumps,clean


def evaluate_case(name,algorithm,config=None,settings=None):
    spec=preset(name,config)['spec']
    kwargs={'algorithm':algorithm,'chains':4,'warmup':200,'draws':200,'iterations':1000,'batch_size':8,'max_seconds':120,'max_gradient_evaluations':500000,'seed':42,**(settings or {})}
    run=Experiment(spec,kwargs); wall=time.perf_counter()
    while run.status=='running': run.step(25)
    snapshot=run.snapshot(); wall=time.perf_counter()-wall
    rows=snapshot['diagnostics'].get('parameters',[]); samples=[x for chain in run.samples for x in chain]
    report={'preset':name,'config':config or {},'algorithm':algorithm,'settings':run.settings,'dimension':run.model.dimension,'observations':spec['metadata']['config']['trials'] if name=='beta' else len(spec.get('data',{}).get('observed',[])),'environment':run.environment,'status':run.status,'reason':run.reason,'stats':snapshot['stats'],'end_to_end_seconds':wall,'diagnostics':snapshot['diagnostics'],'model_hash':run.model.hash,'initial_positions':run.initial_positions,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'posterior_lab').glob('*.py'))}}
    if samples and algorithm!='vi':
        a=np.array(samples);report['mean']=a.mean(0).tolist();report['covariance']=np.atleast_2d(np.cov(a.T)).tolist(); report['quantiles']=np.quantile(a,[.05,.5,.95],axis=0).tolist()
        exact=snapshot['reference'].get('mean')
        if exact is not None: report['absolute_mean_error']=np.abs(a.mean(0)-exact).tolist()
    if algorithm=='vi':
        report['fits']=[]
        for i,v in enumerate(run.vi):
            fit={'run':i,'mu':v.mu.tolist(),'unconstrained_variance':(v.scale**2).tolist(),'last_independent_elbo':v.evaluation_history[-1] if v.evaluation_history else None}
            if name=='gaussian':
                cov=np.asarray(snapshot['reference']['covariance']);precision=np.linalg.inv(cov);variance=v.scale**2
                optimal=1/np.diag(precision)
                kl=.5*(np.sum(np.diag(precision)*variance)+v.mu@precision@v.mu-len(v.mu)+np.linalg.slogdet(cov)[1]-np.log(variance).sum())
                fit.update(exact_optimal_variance=optimal.tolist(),reverse_kl=float(kl),optimal_reverse_kl=float(.5*(np.linalg.slogdet(cov)[1]-np.log(optimal).sum())),true_marginal_variance=np.diag(cov).tolist())
            report['fits'].append(fit)
    for field in ['mode_occupancy','narrow_coverage']:
        if field in snapshot:report[field]=snapshot[field]
    return report,snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    output=ROOT/'reports';output.mkdir(exist_ok=True)
    if args.pilot:
        report,_=evaluate_case('beta','nuts',settings={'warmup':25,'draws':25,'max_seconds':30})
        (output/'native-pilot.json').write_text(dumps(report)+'\n');print(dumps(report['stats']));return
    if not (output/'native-pilot.json').exists():raise SystemExit('Run scripts/evaluate.py --pilot first.')
    cases=[('beta','nuts',{},{}),('beta','vi',{},{}),('regression','nuts',{},{}),('regression','vi',{},{}),('gaussian','hmc',{},{}),('gaussian','nuts',{},{}),('gaussian','vi',{},{}),('funnel','nuts',{'noncentered':False},{}),('funnel','nuts',{'noncentered':True},{}),('mixture','nuts',{}, {'initialization':'one_mode'}),('mixture','nuts',{}, {'initialization':'dispersed'}),('mixture','vi',{}, {'initialization':'dispersed'})]
    report={'description':'Actual sequential engine outcomes. Quick runs are preliminary; no success requirement imposed on funnel or mixture. Exact targets and quality-gated tests are reported separately.','seed':42,'cases':[]}
    rawdir=output/'raw';rawdir.mkdir(exist_ok=True)
    for index,(name,algorithm,config,settings) in enumerate(cases):
        result,snapshot=evaluate_case(name,algorithm,config,settings)
        path=rawdir/f'{index+1:02d}-{name}-{algorithm}.json.gz';raw=dumps(snapshot).encode()
        path.write_bytes(gzip.compress(raw,mtime=0));result['raw_artifact']=str(path.relative_to(ROOT));result['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        report['cases'].append(result);(output/'native-evaluation.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
        print(f'{name:11s} {algorithm:4s}: {result["status"]:16s} {result["stats"]["elapsed_seconds"]:.3f}s, {result["stats"]["gradient_evaluations"]} gradients',flush=True)
    print('Saved reports/native-evaluation.json and complete raw checkpoints.')

if __name__=='__main__':main()
