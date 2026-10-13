"""Transactional, bounded, round-robin execution shared by CLI and Pyodide.

A transition works with copied RNG and adaptation state. Only a complete result
is committed. Evaluation/wall-time accounting includes abandoned work. Worker
messages are serviced between calls; force termination restores the last exported
checkpoint and never turns a partial tree into a retained draw.
"""
import copy
import platform
import time
import uuid
import numpy as np
import scipy
from . import __version__
from .models import compile_model
from .examples import initial_positions, reference, verified_preset
from .samplers import hmc_transition,nuts_transition
from .adaptation import WarmupAdapter, coordinate_target, find_reasonable_step_size
from .variational import MeanFieldVI
from .diagnostics import diagnose
from .serialization import clean, FORMAT, VERSION, encode_rng_state, decode_rng_state

DEFAULTS={'algorithm':'nuts','chains':4,'warmup':200,'draws':200,'seed':42,'step_size':.15,'steps':10,'max_depth':7,'target_accept':.8,'iterations':1000,'batch_size':8,'learning_rate':.03,'initialization':'dispersed','max_seconds':120,'max_gradient_evaluations':500000}

def settings_checked(settings):
    if not isinstance(settings,dict): raise ValueError('settings must be an object')
    unknown=set(settings)-set(DEFAULTS)-{'initial_positions'}
    if unknown: raise ValueError(f'unknown settings: {sorted(unknown)}')
    s={**DEFAULTS,**settings}
    if s['algorithm'] not in ('hmc','nuts','vi'): raise ValueError('algorithm must be hmc, nuts or vi')
    bounds={'chains':(1,4),'warmup':(0,2000),'draws':(1,2000),'seed':(0,2**32-1),'steps':(1,200),'max_depth':(1,9),'iterations':(1,1000),'batch_size':(1,64),'max_gradient_evaluations':(1,2_000_000)}
    for k,(lo,hi) in bounds.items():
        if type(s[k])!=int or not lo<=s[k]<=hi: raise ValueError(f'{k} must be an integer in [{lo}, {hi}]')
    for k,lo,hi in [('step_size',1e-5,4),('target_accept',.5,.99),('learning_rate',.0001,.2),('max_seconds',.001,600)]:
        if not isinstance(s[k],(int,float)) or not np.isfinite(s[k]) or not lo<=s[k]<=hi: raise ValueError(f'{k} must be in [{lo}, {hi}]')
    if s['initialization'] not in ('dispersed','one_mode'): raise ValueError('unknown initialization')
    return s

class BudgetExceeded(RuntimeError): pass

class Experiment:
    def __init__(self,spec,settings=None):
        self.model=compile_model(spec); self.spec=self.model.spec; self.settings=settings_checked(settings or {}); s=self.settings
        self.run_id=str(uuid.uuid4()); self.status='running'; self.reason=None; self.next_chain=0
        self.reference=reference(self.spec); self.elapsed=0.; self.evaluations=0; self.last_diagnostics={'status':'unavailable','message':'too few draws to assess','parameters':[],'as_of_retained':0}
        self.stats={'divergences':0,'warmup_divergences':0,'max_depth_events':0,'warmup_max_depth_events':0,'acceptance_sum':0.,'transitions':0,'retained_transitions':0}
        self.samples=[[] for _ in range(s['chains'])]; self.unconstrained_samples=[[] for _ in range(s['chains'])]; self.transitions=[]; self.transition_summaries=[[] for _ in range(s['chains'])]; self.chains=[]; self.vi=[]
        initial=np.asarray(s.get('initial_positions',initial_positions(spec,s['chains'],s['seed'],s['initialization'])),dtype=np.float64)
        if initial.shape!=(s['chains'],self.model.dimension) or not np.all(np.isfinite(initial)): raise ValueError('initial_positions must be finite [chain, dimension] unconstrained coordinates')
        self.initial_positions=initial.tolist()
        for i in range(s['chains']):
            if not np.isfinite(self.model(initial[i])[0]): raise ValueError(f'chain {i+1}: invalid initial position')
            if s['algorithm']=='vi':
                self.vi.append(MeanFieldVI(self.model,self.model.dimension,seed=s['seed']+i,init=initial[i],learning_rate=s['learning_rate'],batch_size=s['batch_size'],max_iterations=s['iterations']))
            else:
                rng=np.random.default_rng(np.random.SeedSequence([s['seed'],i,1901])); adapter=WarmupAdapter(self.model.dimension,s['warmup'],s['step_size'],s['target_accept'])
                lp,g=self.model(initial[i]); self.chains.append({'z':initial[i].copy(),'rng':rng,'adapter':adapter,'iteration':0,'logp':lp,'gradient':g,'search_done':s['warmup']==0})
        self.environment={'engine':__version__,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'machine':platform.machine(),'rng':'NumPy PCG64','float':'float64'}
        self._active_mark=time.perf_counter()

    def _account_between_steps(self):
        # Active wall time includes checkpoint/diagnostic work and cooperative
        # browser yields. Paused time is excluded; this is not a throughput ETA.
        if self.status=='running' and self._active_mark is not None:
            now=time.perf_counter();self.elapsed+=now-self._active_mark;self._active_mark=now

    def set_status(self,status,reason=None):
        if status not in ('running','paused','cancelled'): raise ValueError('invalid requested status')
        if status=='running' and self.status in ('complete','failed','budget_exhausted'): raise ValueError('finished run cannot resume; create a new experiment')
        self._account_between_steps();self.status=status;self.reason=reason
        self._active_mark=time.perf_counter() if status=='running' else None

    def progress(self):
        s=self.settings
        if s['algorithm']=='vi':
            done=sum(v.iteration for v in self.vi)
            return {'completed':done,'total':s['chains']*s['iterations'],'warmup_completed':0,'retained':sum(map(len,self.samples)),'chain':self.next_chain,'phase':'variational optimization'}
        counts=[c['iteration'] for c in self.chains]
        return {'completed':sum(counts),'total':s['chains']*(s['warmup']+s['draws']),'warmup_completed':sum(min(i,s['warmup']) for i in counts),'retained':sum(map(len,self.samples)),'chain':self.next_chain,'phase':'warmup' if any(i<s['warmup'] for i in counts) else 'sampling'}

    def step(self,count=1):
        if type(count)!=int or not 1<=count<=25: raise ValueError('step count must be 1–25')
        for _ in range(count):
            if self.status!='running': break
            if self.progress()['completed']>=self.progress()['total']:
                self.status='complete';self._active_mark=None;break
            self._account_between_steps()
            started=time.perf_counter(); local_evaluations=0
            def check():
                if self.elapsed+time.perf_counter()-started>=self.settings['max_seconds']: raise BudgetExceeded('wall-time budget reached; incomplete transition discarded')
                if self.evaluations+local_evaluations>=self.settings['max_gradient_evaluations']: raise BudgetExceeded('gradient-evaluation budget reached; incomplete transition discarded')
            def counted_target(z):
                nonlocal local_evaluations
                check(); local_evaluations+=1; return self.model(z)
            try:
                check()
                if self.settings['algorithm']=='vi': self._vi_step(counted_target)
                else: self._mcmc_step(counted_target,check)
                self.next_chain=(self.next_chain+1)%self.settings['chains']
                p=self.progress()
                if p['completed']>=p['total']: self.status='complete'
            except BudgetExceeded as e: self.status='budget_exhausted'; self.reason=str(e)
            except Exception as e: self.status='failed'; self.reason=f'{type(e).__name__}: {e}'
            finally:
                self.elapsed+=time.perf_counter()-started; self.evaluations+=local_evaluations
                self._active_mark=time.perf_counter() if self.status=='running' else None
        return self.compact()

    def _mcmc_step(self,target,check):
        i=self.next_chain; old=self.chains[i]; s=self.settings
        rng=np.random.default_rng(); rng.bit_generator.state=copy.deepcopy(old['rng'].bit_generator.state)
        adapter=WarmupAdapter.from_state(old['adapter'].state_dict()); z=old['z'].copy()
        working=coordinate_target(target,adapter.scales)
        if not old['search_done']:
            eps,info=find_reasonable_step_size(working,z/adapter.scales,rng,initial=adapter.step_size,check=check,return_info=True)
            adapter=WarmupAdapter(self.model.dimension,s['warmup'],eps,s['target_accept']); adapter.history.append({'iteration':0,'event':'initial step-size search',**info})
        scales=adapter.scales.copy(); working=coordinate_target(target,scales)
        iteration=old['iteration']; warmup=iteration<s['warmup']
        record=iteration in (0,s['warmup'],s['warmup']+1,s['warmup']+max(2,s['draws']//2))
        if s['algorithm']=='nuts': new,meta=nuts_transition(working,z/scales,rng,adapter.step_size,max_depth=s['max_depth'],record=record,check=check)
        else: new,meta=hmc_transition(working,z/scales,rng,adapter.step_size,steps=s['steps'],record=record,check=check)
        new_z=scales*new; reported=self.model.report(new_z)
        if not np.all(np.isfinite(reported)): raise ValueError('nonfinite reported parameter; transition discarded')
        if warmup: adapter.update(new_z,meta['acceptance'])
        lp,gradient=target(new_z)
        if not np.isfinite(lp): raise ValueError('invalid resulting density; transition discarded')
        meta.update({'chain':i,'iteration':iteration,'phase':'warmup' if warmup else 'sampling','scales':scales.tolist(),'coordinates':'scaled working coordinates x; z = scales * x'})
        summary=clean({k:v for k,v in meta.items() if k not in ('trajectory','branches')})
        new_stats=self.stats.copy()
        new_stats['warmup_divergences' if warmup else 'divergences']+=int(meta['divergent'])
        new_stats['warmup_max_depth_events' if warmup else 'max_depth_events']+=int(meta['max_depth_reached'])
        new_stats['acceptance_sum']+=meta['acceptance'];new_stats['transitions']+=1
        if not warmup:new_stats['retained_transitions']+=1
        # Commit only after all provisional values and metadata are valid.
        self.chains[i]={'z':new_z,'rng':rng,'adapter':adapter,'iteration':iteration+1,'logp':lp,'gradient':gradient,'search_done':True}
        if not warmup:
            self.samples[i].append(reported.tolist());self.unconstrained_samples[i].append(new_z.tolist())
        self.transition_summaries[i].append(summary)
        if record:self.transitions.append(clean(meta))
        self.stats=new_stats

    def _vi_step(self,target):
        i=self.next_chain; s=self.settings; old=self.vi[i]; new=old.clone(target)
        new.step()
        if new.iteration%50==0 or new.iteration==s['iterations']: new.evaluate(n=128)
        model_draws=z_draws=None
        if new.iteration==s['iterations']:
            z_draws=new.sample(s['draws']); model_draws=[self.model.report(z).tolist() for z in z_draws]
            if not np.all(np.isfinite(model_draws)): raise ValueError('VI samples produced nonfinite parameters')
        self.vi[i]=new; self.vi[i].target=self.model
        if model_draws is not None:
            self.samples[i]=model_draws; self.unconstrained_samples[i]=z_draws.tolist()

    def compact(self):
        return {'run_id':self.run_id,'status':self.status,'reason':self.reason,'progress':self.progress(),'stats':{**self.stats,'elapsed_seconds':self.elapsed,'gradient_evaluations':self.evaluations,'mean_acceptance':self.stats['acceptance_sum']/max(1,self.stats['transitions'])}}

    def delta(self):
        """Last committed chain update; offsets make repeated delivery idempotent.

        The worker requests this after one step, and waits for the main thread's
        acknowledgement before starting another transition. Periodic full
        snapshots refresh diagnostics and reconcile all bounded state.
        """
        self._account_between_steps();i=(self.next_chain-1)%self.settings['chains']
        offset=max(0,len(self.samples[i])-1);summary_offset=max(0,len(self.transition_summaries[i])-1);transition_offset=max(0,len(self.transitions)-1)
        out={**self.compact(),'delta':True,'chain':i,'offset':offset,'samples':self.samples[i][offset:],'unconstrained_samples':self.unconstrained_samples[i][offset:],'summary_offset':summary_offset,'transition_summaries':self.transition_summaries[i][summary_offset:],'transition_offset':transition_offset,'transitions':self.transitions[transition_offset:],'next_chain':self.next_chain}
        if self.settings['algorithm']=='vi':
            v=self.vi[i];h=max(0,len(v.history)-1);e=max(0,len(v.evaluation_history)-1)
            out.update(vi_state=v.state_dict(h,e),history_offset=h,evaluation_offset=e)
            # A completed VI fit commits the entire bounded independent draw set.
            if self.samples[i]:out.update(offset=0,samples=self.samples[i],unconstrained_samples=self.unconstrained_samples[i])
        else:
            c=self.chains[i];out['chain_state']={'z':c['z'],'rng':encode_rng_state(c['rng'].bit_generator.state),'adapter':c['adapter'].state_dict(),'iteration':c['iteration'],'logp':c['logp'],'gradient':c['gradient'],'search_done':c['search_done']}
        return clean(out)

    def snapshot(self,diagnostics=True,checkpoint=True):
        self._account_between_steps()
        p=self.progress(); s=self.settings
        if diagnostics:
            if s['algorithm']=='vi':
                rows=[]
                # Independently initialized VI fits are alternatives, never MCMC chains.
                for i,draws in enumerate(self.samples):
                    if draws:
                        a=np.asarray(draws)
                        rows.append({'run':i,'mean':a.mean(0).tolist(),'sd':a.std(0,ddof=1).tolist(),'q05':np.quantile(a,.05,axis=0).tolist(),'q95':np.quantile(a,.95,axis=0).tolist()})
                self.last_diagnostics={'status':'not applicable','message':'R-hat and MCMC ESS do not apply to variational draws. Each fit is reported separately.','parameters':[],'fits':rows}
            else:
                n=min(map(len,self.samples))
                if n: self.last_diagnostics=diagnose(np.array([chain[:n] for chain in self.samples]),self.model.reporting_names)
                self.last_diagnostics['as_of_retained']=n*s['chains']
        out={**self.compact(),'model_hash':self.model.hash,'spec':self.spec,'settings':s,'names':self.model.reporting_names,'samples':self.samples,'unconstrained_samples':self.unconstrained_samples,'diagnostics':self.last_diagnostics,'reference':self.reference,'transitions':self.transitions,'transition_summaries':self.transition_summaries,'adaptation':[c['adapter'].state_dict() for c in self.chains],'vi':[v.state_dict() for v in self.vi],'environment':self.environment}
        if verified_preset(self.spec)=='mixture':
            out['mode_occupancy']=[{'chain':i,'left_fraction':float(np.mean(np.asarray(a)[:,0]<0)) if a else None,'region_transitions':int(np.sum(np.diff(np.asarray(a)[:,0]<0)!=0)) if len(a)>1 else 0,'draws':len(a)} for i,a in enumerate(self.samples)]
        if verified_preset(self.spec)=='funnel':
            out['narrow_coverage']=[{'chain':i,'fraction_y_below_minus3':float(np.mean(np.asarray(a)[:,0]<-3)) if a else None} for i,a in enumerate(self.samples)]
        if checkpoint: out['checkpoint']=self.checkpoint()
        return clean(out)

    def checkpoint(self):
        self._account_between_steps()
        return clean({'format':FORMAT,'version':VERSION,'run_id':self.run_id,'model_hash':self.model.hash,'spec':self.spec,'settings':self.settings,'environment':self.environment,'status':self.status,'reason':self.reason,'next_chain':self.next_chain,'initial_positions':self.initial_positions,'elapsed':self.elapsed,'evaluations':self.evaluations,'stats':self.stats,'samples':self.samples,'unconstrained_samples':self.unconstrained_samples,'transitions':self.transitions,'transition_summaries':self.transition_summaries,'diagnostics':self.last_diagnostics,'chains':[{'z':c['z'],'rng':encode_rng_state(c['rng'].bit_generator.state),'adapter':c['adapter'].state_dict(),'iteration':c['iteration'],'logp':c['logp'],'gradient':c['gradient'],'search_done':c['search_done']} for c in self.chains],'vi':[v.state_dict() for v in self.vi]})

    @classmethod
    def restore(cls,data):
        if data.get('format')!=FORMAT or data.get('version')!=VERSION: raise ValueError('unsupported experiment format/version')
        model=compile_model(data['spec']); s=settings_checked(data['settings']); d=model.dimension; n=s['chains']
        if data.get('model_hash')!=model.hash: raise ValueError('model hash mismatch')
        for key,width in [('samples',len(model.reporting_names)),('unconstrained_samples',d)]:
            values=data.get(key)
            if not isinstance(values,list) or len(values)!=n: raise ValueError(f'{key}: incorrect chain count')
            for rows in values:
                if not isinstance(rows,list) or len(rows)>s['draws']: raise ValueError(f'{key}: draw limit exceeded')
                for row in rows:
                    if not isinstance(row,list) or len(row)!=width or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in row): raise ValueError(f'{key}: invalid finite dimensions')
        if [len(a) for a in data['samples']]!=[len(a) for a in data['unconstrained_samples']]: raise ValueError('draw arrays disagree')
        for reported,z_values in zip(data['samples'],data['unconstrained_samples']):
            for row,z in zip(reported,z_values):
                computed=model.report(z)
                if not np.all(np.isfinite(computed)) or not np.allclose(computed,row,rtol=1e-10,atol=1e-10): raise ValueError('reported draws disagree with model transforms')
        if type(data.get('next_chain'))!=int or not 0<=data['next_chain']<n: raise ValueError('invalid next chain')
        if len(data.get('transitions',[]))>n*4: raise ValueError('trajectory recording budget exceeded')
        for transition in data.get('transitions',[]):
            if len(transition.get('trajectory',[]))>512 or len(transition.get('branches',[]))>2048: raise ValueError('trajectory budget exceeded')
        result=cls(data['spec'],s)
        stats=data.get('stats')
        if not isinstance(stats,dict) or set(stats)!=set(result.stats): raise ValueError('invalid statistics schema')
        for key,value in stats.items():
            if key=='acceptance_sum':
                if not isinstance(value,(int,float)) or not np.isfinite(value) or value<0: raise ValueError('invalid acceptance aggregate')
            elif type(value)!=int or not 0<=value<=16000: raise ValueError('invalid transition counters')
        if stats['acceptance_sum']>stats['transitions']+1e-8 or any(stats[k]>stats['transitions'] for k in ['divergences','warmup_divergences','max_depth_events','warmup_max_depth_events','retained_transitions']): raise ValueError('inconsistent transition counters')
        result.run_id=data['run_id']; result.status=data['status']; result.reason=data.get('reason'); result.next_chain=data['next_chain']
        if result.status not in ('running','paused','cancelled','complete','budget_exhausted','failed'): raise ValueError('invalid run status')
        result.elapsed=float(data['elapsed']); result.evaluations=int(data['evaluations'])
        if not np.isfinite(result.elapsed) or result.elapsed<0 or not 0<=result.evaluations<=2_000_000: raise ValueError('invalid budget counters')
        result.stats=data['stats']; result.samples=copy.deepcopy(data['samples']); result.unconstrained_samples=copy.deepcopy(data['unconstrained_samples']); result.transitions=data.get('transitions',[]); result.last_diagnostics=data.get('diagnostics',result.last_diagnostics)
        initial=data.get('initial_positions')
        if not isinstance(initial,list) or len(initial)!=n or any(not isinstance(row,list) or len(row)!=d or not all(type(x) in (float,int) and np.isfinite(x) for x in row) for row in initial): raise ValueError('invalid initial positions')
        result.initial_positions=initial
        summaries=data.get('transition_summaries')
        if not isinstance(summaries,list) or len(summaries)!=n or any(not isinstance(x,list) or len(x)>s['warmup']+s['draws'] for x in summaries): raise ValueError('invalid transition summary arrays')
        result.transition_summaries=summaries
        if s['algorithm']=='vi':
            if len(data.get('vi',[]))!=n: raise ValueError('VI state count mismatch')
            result.vi=[MeanFieldVI.from_state(result.model,v) for v in data['vi']]
            for i,v in enumerate(result.vi):
                if v.dimension!=d or v.batch_size!=s['batch_size'] or v.learning_rate!=s['learning_rate'] or v.max_iterations!=s['iterations'] or v.seed!=s['seed']+i: raise ValueError('VI state disagrees with experiment settings')
                if len(result.samples[i])!=(s['draws'] if v.iteration==s['iterations'] else 0): raise ValueError('VI retained draws disagree with iteration')
        else:
            if len(data.get('chains',[]))!=n: raise ValueError('chain state count mismatch')
            restored=[]
            for i,c in enumerate(data['chains']):
                z=np.asarray(c['z'],dtype=np.float64); gradient=np.asarray(c['gradient'],dtype=np.float64)
                if z.shape!=(d,) or gradient.shape!=(d,) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(gradient)): raise ValueError('invalid chain position/gradient')
                it=c['iteration']
                if type(it)!=int or not 0<=it<=s['warmup']+s['draws'] or max(0,it-s['warmup'])!=len(result.samples[i]): raise ValueError('chain iteration disagrees with draws')
                if len(summaries[i])!=it: raise ValueError('transition summaries disagree with iteration')
                a=c['adapter']
                if a['dim']!=d or a['warmup']!=s['warmup'] or a['iteration']!=min(it,s['warmup']): raise ValueError('inconsistent adaptation counters')
                if a['target_accept']!=s['target_accept'] or a['dual']['target']!=s['target_accept']: raise ValueError('adaptation target disagrees with settings')
                if len(a['scales'])!=d or len(a['variance']['mean'])!=d or len(a['variance']['m2'])!=d: raise ValueError('invalid adaptation dimension')
                adapter=WarmupAdapter.from_state(a)
                if not np.all(np.isfinite(adapter.scales)) or np.any(adapter.scales<=0): raise ValueError('invalid coordinate scales')
                if c['rng'].get('bit_generator')!='PCG64': raise ValueError('unsupported random generator')
                rng=np.random.default_rng(); rng.bit_generator.state=decode_rng_state(c['rng']); lp,g=result.model(z)
                if not np.isfinite(lp) or not np.allclose(g,gradient,rtol=1e-8,atol=1e-10) or not np.isclose(lp,c['logp'],rtol=1e-10,atol=1e-10): raise ValueError('checkpoint gradient/density mismatch')
                restored.append({'z':z,'rng':rng,'adapter':adapter,'iteration':it,'logp':lp,'gradient':g,'search_done':bool(c['search_done'])})
            result.chains=restored
        counters=[v.iteration for v in result.vi] if s['algorithm']=='vi' else [c['iteration'] for c in result.chains]
        total=sum(counters);cycles,remainder=divmod(total,n)
        if counters!=[cycles+int(i<remainder) for i in range(n)] or result.next_chain!=remainder: raise ValueError('inconsistent round-robin checkpoint counters')
        if result.status=='complete' and total!=result.progress()['total']: raise ValueError('complete status disagrees with counters')
        if s['algorithm']=='vi':
            if any(stats.values()) or any(summaries): raise ValueError('VI must not have MCMC transition statistics')
        elif stats['transitions']!=total or stats['retained_transitions']!=sum(map(len,result.samples)): raise ValueError('transition statistics disagree with draws')
        if result.environment!=data.get('environment'): result.reason='restored in a different runtime; numerical equality of full chains is not guaranteed'
        result._active_mark=time.perf_counter() if result.status=='running' else None
        return result
