"""Five ordinary graph models, with independent analytic/generative oracles."""
import numpy as np
from scipy import stats
from .models import Model, ModelError

PRESETS=[
    ('beta','Updating a belief','A probability, ten trials, and an exact answer.'),
    ('regression','From observations to predictions','Separate uncertainty in the mean from future observations.'),
    ('gaussian','Correlation defeats factorization','An optimized approximation can still miss uncertainty.'),
    ('funnel','Difficult hierarchical geometry','Compare centered and noncentered generative coordinates.'),
    ('mixture','Two modes, misleading confidence','Agreement within one mode does not establish global exploration.'),
]

def bounded(value,low,high,name):
    value=float(value)
    if not np.isfinite(value) or not low<=value<=high: raise ModelError(f'{name} must be between {low} and {high}')
    return value

def preset(name,config=None):
    c=dict(config or {}); m=Model(dict((x[0],x[1]) for x in PRESETS).get(name,name))
    if name=='beta':
        a=bounded(c.get('alpha',2),.1,100,'alpha'); b=bounded(c.get('beta',2),.1,100,'beta')
        n=bounded(c.get('trials',10),1,1000,'trials'); k=bounded(c.get('successes',7),0,n,'successes')
        if n!=int(n) or k!=int(k): raise ModelError('successes and trials must be integers')
        c={'alpha':a,'beta':b,'trials':int(n),'successes':int(k)}
        p=m.parameter('probability',constraint={'type':'interval','lower':0,'upper':1})
        m.prior('prior','beta',p,alpha=a,beta=b).observe('observations','binomial',k,trials=n,prob=p)
    elif name=='regression':
        rng=np.random.default_rng(8128); x=np.linspace(-2,2,24); y=.7+1.25*x+rng.normal(0,.6,len(x))
        x=np.asarray(c.get('x',x),dtype=np.float64); y=np.asarray(c.get('y',y),dtype=np.float64)
        if x.ndim!=1 or y.shape!=x.shape or not 2<=len(x)<=1000 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.max(np.abs(x))>1000 or np.max(np.abs(y))>1000: raise ModelError('regression requires 2–1000 finite x/y pairs, absolute values <=1000')
        noise=bounded(c.get('noise',.6),.05,10,'noise'); prior=bounded(c.get('prior_scale',3),.1,20,'prior_scale')
        c={'x':x.tolist(),'y':y.tolist(),'noise':noise,'prior_scale':prior,'preprocessing':'none'}
        coef=m.parameter('coefficient',[2]); design=m.data('design',np.column_stack((np.ones(len(x)),x))); data=m.data('observed',y)
        mean=m.node('mean_response','matvec',design,coef)
        m.prior('coefficient_prior','normal',coef,loc=0,scale=prior).observe('observations','normal',data,loc=mean,scale=noise)
        m.spec['prediction']={'distribution':'normal','design':'intercept + x','noise':noise}
    elif name=='gaussian':
        correlation=bounded(c.get('correlation',.95),-.99,.99,'correlation'); scale=bounded(c.get('scale',1),.1,10,'scale')
        cov=[[1,correlation*scale],[correlation*scale,scale**2]]; c={'correlation':correlation,'scale':scale}
        point=m.parameter('position',[2]); m.prior('normalized_target','multivariate_normal',point,loc=[0,0],cov=cov)
    elif name=='funnel':
        noncentered=c.get('noncentered',False)
        if type(noncentered)!=bool: raise ModelError('noncentered must be a boolean')
        c={'noncentered':noncentered}; y=m.parameter('y'); m.prior('parent','normal',y,loc=0,scale=3)
        half=m.node('half_y','mul',y,.5); scale=m.node('child_scale','exp',half)
        if noncentered:
            z=m.parameter('base',[5]); m.prior('base_prior','normal',z,loc=0,scale=1)
            m.node('children','mul',z,scale); m.derived('children'); m.spec['report']=['y','children']
        else:
            x=m.parameter('children',[5]); m.prior('children_prior','normal',x,loc=0,scale=scale)
        m.spec['density_kind']='normalized generative joint (no observational data)'
    elif name=='mixture':
        c={}; point=m.parameter('position',[2]); cov=[[.25,0],[0,1]]
        left=m.node('left_logpdf','logpdf',distribution='multivariate_normal',value=point,params={'loc':[-4,0],'cov':cov})
        right=m.node('right_logpdf','logpdf',distribution='multivariate_normal',value=point,params={'loc':[4,0],'cov':cov})
        l=m.node('left_weighted','add',left,float(np.log(.7))); r=m.node('right_weighted','add',right,float(np.log(.3)))
        joined=m.node('components','stack',l,r); density=m.node('mixture_log_density','logsumexp',joined); m.factor('normalized_mixture',density)
    else: raise ModelError(f'unknown preset {name}')
    m.spec['metadata']={'preset':name,'config':c}
    m.spec.setdefault('density_kind','log joint density' if name in ('beta','regression') else 'normalized target density')
    return {'id':name,'name':m.spec['name'],'description':next(x[2] for x in PRESETS if x[0]==name),'spec':m.to_dict(),'config':c}

def verified_preset(spec):
    meta=spec.get('metadata',{});name=meta.get('preset');c=meta.get('config',{})
    try:
        return name if name in [p[0] for p in PRESETS] and preset(name,c)['spec']==spec else None
    except (ValueError,TypeError,KeyError): return None

def reference(spec):
    meta=spec.get('metadata',{}); name=meta.get('preset'); c=meta.get('config',{})
    # Only attach an oracle if the *entire graph* matches the preset, not a spoofed label.
    if verified_preset(spec) is None: return {'kind':'unavailable','description':'No independent oracle for this custom graph.'}
    if name=='beta':
        a,b=c['alpha'],c['beta']; aa=a+c['successes']; bb=b+c['trials']-c['successes']; x=np.linspace(.001,.999,200)
        prior=stats.beta.pdf(x,a,b); posterior=stats.beta.pdf(x,aa,bb); like=stats.binom.pmf(c['successes'],c['trials'],x)
        return {'kind':'analytic','label':f'Exact Beta({aa:g}, {bb:g}) posterior','x':x.tolist(),'prior':prior.tolist(),'posterior':posterior.tolist(),'likelihood':(like/max(like)*max(posterior)).tolist(),'likelihood_label':'Likelihood shape, scaled to posterior peak','mean':[aa/(aa+bb)],'variance':[aa*bb/((aa+bb)**2*(aa+bb+1))],'interval':stats.beta.ppf([.05,.95],aa,bb).tolist()}
    if name=='regression':
        x=np.asarray(c['x']); y=np.asarray(c['y']); design=np.column_stack((np.ones(len(x)),x)); precision=design.T@design/c['noise']**2+np.eye(2)/c['prior_scale']**2
        cov=np.linalg.solve(precision,np.eye(2)); mean=np.linalg.solve(precision,design.T@y/c['noise']**2)
        grid=np.linspace(x.min()-.3,x.max()+.3,100); X=np.column_stack((np.ones(len(grid)),grid)); response=X@mean; var=np.einsum('ij,jk,ik->i',X,cov,X); z=stats.norm.ppf(.95)
        return {'kind':'analytic','label':'Exact Gaussian coefficient posterior; central 90% bands','x_data':x.tolist(),'y_data':y.tolist(),'x_grid':grid.tolist(),'mean_response':response.tolist(),'mean_lower':(response-z*np.sqrt(var)).tolist(),'mean_upper':(response+z*np.sqrt(var)).tolist(),'predictive_lower':(response-z*np.sqrt(var+c['noise']**2)).tolist(),'predictive_upper':(response+z*np.sqrt(var+c['noise']**2)).tolist(),'mean':mean.tolist(),'covariance':cov.tolist()}
    if name in ('gaussian','mixture'):
        gx=np.linspace(-3.5,3.5,65) if name=='gaussian' else np.linspace(-6,6,75)
        gy=np.linspace(-3.5*c.get('scale',1),3.5*c.get('scale',1),65); xx,yy=np.meshgrid(gx,gy); xy=np.stack((xx,yy),axis=-1)
        if name=='gaussian':
            cov=np.array([[1,c['correlation']*c['scale']],[c['correlation']*c['scale'],c['scale']**2]])
            density=stats.multivariate_normal.pdf(xy,mean=[0,0],cov=cov)
            return {'kind':'analytic','label':'Exact joint-density contours (density levels, not probability regions)','grid':{'x':gx.tolist(),'y':gy.tolist(),'z':density.tolist()},'mean':[0,0],'covariance':cov.tolist(),'meanfield_variance':(1/np.diag(np.linalg.inv(cov))).tolist()}
        density=.7*stats.multivariate_normal.pdf(xy,mean=[-4,0],cov=np.diag([.25,1]))+.3*stats.multivariate_normal.pdf(xy,mean=[4,0],cov=np.diag([.25,1]))
        return {'kind':'analytic','label':'Exact mixture joint-density contours; known weights 0.7 / 0.3','grid':{'x':gx.tolist(),'y':gy.tolist(),'z':density.tolist()},'mean':[-1.6,0],'covariance':[[13.69,0],[0,1]],'weights':[.7,.3]}
    rng=np.random.default_rng(771); y=rng.normal(0,3,800); children=np.exp(y[:,None]/2)*rng.normal(size=(800,5)); grid=np.linspace(-10,10,120)
    return {'kind':'independent generative draws','label':'Known generative reference on common model axes (y, child 0)','reference_samples':np.column_stack((y,children)).tolist(),'y_marginal':{'x':grid.tolist(),'density':stats.norm.pdf(grid,0,3).tolist()},'narrow_probability':float(stats.norm.cdf(-3,0,3)),'mean_y':0,'sd_y':3,'seed':771}

def initial_positions(spec,chains,seed,initialization='dispersed'):
    from .models import compile_model
    d=compile_model(spec).dimension; rng=np.random.default_rng(np.random.SeedSequence([seed,481]))
    positions=rng.normal(0,.4,(chains,d)); name=verified_preset(spec)
    if name=='mixture': positions[:,0]=(-4 if initialization=='one_mode' else np.where(np.arange(chains)%2==0,-4,4))+rng.normal(0,.2,chains)
    if name=='funnel': positions[:,0]=np.linspace(-1,1,chains)
    return positions
