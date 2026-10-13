import copy
import json
import numpy as np
import pytest
from scipy.special import betaln, expit
from scipy.integrate import quad
from posterior_lab.autodiff import Tensor, stack
from posterior_lab.models import Model, ModelError, compile_model
from posterior_lab.examples import PRESETS, preset, reference
from posterior_lab.transforms import transform, inverse
from posterior_lab.distributions import log_density


def finite_difference(f,x,h=1e-5):
    x=np.asarray(x,dtype=float); g=np.empty_like(x)
    for i in range(x.size):
        dx=np.zeros_like(x); dx.flat[i]=h
        g.flat[i]=(f(x+dx)-f(x-dx))/(2*h)
    return g


def test_reused_operands_shared_subexpressions_fresh_tapes():
    x=Tensor([1.,2.,3.],requires_grad=True)
    shared=x*x
    loss=(shared+shared+x).sum()
    loss.backward(); np.testing.assert_array_equal(x.grad,4*x.value+1)
    loss.backward(); np.testing.assert_array_equal(x.grad,4*x.value+1)
    model=Model(); q=model.parameter('q',[3]); a=model.node('a','mul',q,q); b=model.node('b','add',a,a); model.factor('f',b)
    compiled=model.compile()
    for z in [np.array([1.,2.,3.]),np.array([-.2,.8,1.2]),np.zeros(3)]:
        np.testing.assert_allclose(compiled(z)[1],4*z)


def test_broadcast_reverse_and_axis_reduction():
    a=Tensor(np.array([[1.],[2.],[3.]]),requires_grad=True)
    b=Tensor(np.array([[.5,1.,2.,3.]]),requires_grad=True)
    loss=((a*b)**2).sum(axis=0).mean()
    loss.backward()
    np.testing.assert_allclose(a.grad,2*a.value*np.sum(b.value**2)/4)
    np.testing.assert_allclose(b.grad,2*b.value*np.sum(a.value**2)/4)
    with pytest.raises(ValueError,match='broadcast'):
        Tensor(np.zeros((101,1)))+Tensor(np.zeros((1,101)))


@pytest.mark.parametrize('operation',[
    lambda x:(x*x+x/2-1).sum(), lambda x:(x**3).sum(),lambda x:x.exp().sum(),
    lambda x:(x+2).log().sum(),lambda x:x.log1p().sum(),lambda x:x.sigmoid().sum(),
    lambda x:x.softplus().mean(),lambda x:x.logsumexp(),
    lambda x:x.reshape([2,2]).sum(axis=1).sum(),lambda x:x.index(2)**2,
    lambda x:x.matmul(Tensor([1,2,3,4])),
    lambda x:Tensor([[1,2,3,4],[2,-3,1,0]]).matmul(x).logsumexp(),
    lambda x:stack([x,x*x],axis=1).logsumexp(axis=0).sum(),
])
def test_primitive_gradients_against_central_differences(operation):
    z=np.array([.2,.3,.5,.9]); x=Tensor(z,requires_grad=True); y=operation(x); y.backward()
    expected=finite_difference(lambda q:float(operation(Tensor(q)).value),z)
    np.testing.assert_allclose(x.grad,expected,rtol=2e-7,atol=2e-7)


def test_logsumexp_extreme_finite_logits_and_zero_power():
    x=Tensor([1000.,1001.,-1000.],requires_grad=True); y=x.logsumexp();y.backward()
    assert np.isfinite(y.value)
    np.testing.assert_allclose(x.grad,[1/(1+np.e),np.e/(1+np.e),0],atol=1e-13)
    x=Tensor([0.,1.],requires_grad=True); (x**0).sum().backward();np.testing.assert_array_equal(x.grad,[0,0])


@pytest.mark.parametrize('name',[p[0] for p in PRESETS]+['noncentered'])
def test_all_compiled_model_gradients(name):
    s=preset('funnel',{'noncentered':True})['spec'] if name=='noncentered' else preset(name)['spec']
    m=compile_model(s); z=np.linspace(-.35,.35,m.dimension)
    value,g=m(z); assert np.isfinite(value)
    expected=finite_difference(lambda q:m(q)[0],z)
    np.testing.assert_allclose(g,expected,rtol=3e-6,atol=3e-7)


@pytest.mark.parametrize('z',[-1000.,-40.,0.,40.,1000.])
def test_beta_binomial_stable_exact_logit_target(z):
    m=compile_model(preset('beta')['spec']); lp,g=m([z])
    expected_g=9*expit(-z)-5*expit(z)
    assert np.isfinite(lp); assert g[0]==pytest.approx(expected_g,abs=1e-12)
    # Named joint constants are retained: compare differences to avoid evidence confusion.
    expected=-9*np.logaddexp(0,-z)-5*np.logaddexp(0,z)
    offset=m([0.])[0]+14*np.log(2)
    assert lp==pytest.approx(expected+offset,abs=1e-10)


@pytest.mark.parametrize('constraint,z',[
    ({'type':'real'},np.array([-.7,.3])),
    ({'type':'positive'},np.array([-.7,.3])),
    ({'type':'interval','lower':-2,'upper':3},np.array([-.7,.3])),
])
def test_transform_identity_and_numerical_jacobian(constraint,z):
    theta,logj=transform(Tensor(z,requires_grad=True),constraint)
    np.testing.assert_allclose(inverse(theta.value,constraint),z,atol=1e-14)
    derivatives=[]
    for i in range(2):
        derivatives.append(finite_difference(lambda q:transform(Tensor(q),constraint)[0].value[i],z)[i])
    assert float(logj.value)==pytest.approx(np.log(derivatives).sum(),abs=1e-9)


def test_change_of_variables_integrates_to_one_and_noncentered_no_extra_jacobian():
    # A constrained beta prior, with no data, normalizes after transformation.
    m=Model(); theta=m.parameter('theta',constraint={'type':'interval','lower':0,'upper':1}); m.prior('p','beta',theta,alpha=2,beta=2); target=m.compile()
    integral,error=quad(lambda z:np.exp(target([z])[0]),-40,40,epsabs=1e-10)
    assert integral==pytest.approx(1,abs=1e-10)
    nc=compile_model(preset('funnel',{'noncentered':True})['spec'])
    assert nc.contributions([1,0,0,0,0,0])['transform_log_jacobian']==0
    assert nc([1,0,0,0,0,0])[1][0]==pytest.approx(-1/9)
    positive=Model(); p=positive.parameter('scale',constraint={'type':'positive'}); positive.prior('p','half_normal',p,scale=2)
    target=positive.compile(); integral,_=quad(lambda z:np.exp(target([z])[0]),-35,5)
    assert integral==pytest.approx(1,abs=1e-10)


@pytest.mark.parametrize('mutation,match',[
    (lambda s:s['parameters'].append(copy.deepcopy(s['parameters'][0])),'duplicate'),
    (lambda s:s['parameters'][0].update(shape=[33]),'scalar or vector'),
    (lambda s:s['nodes'].append({'name':'evil','op':'exec','args':[]}), 'unsupported'),
    (lambda s:s['factors'][0].update(value='missing'),'reference'),
    (lambda s:s['parameters'][0].update(constraint={'type':'interval','lower':1,'upper':0}),'interval'),
    (lambda s:s['data'].update(bad=[float('nan')]),'compliant'),
])
def test_model_validation(mutation,match):
    s=preset('beta')['spec']; mutation(s)
    with pytest.raises(ValueError,match=match): compile_model(s)


def test_cycles_shapes_support_covariance_and_constant_hyperparameters():
    s=preset('gaussian')['spec'];s['nodes']=[{'name':'a','op':'add','args':['b',1]},{'name':'b','op':'add','args':['a',1]}]
    with pytest.raises(ModelError,match='cycle'): compile_model(s)
    for cov in [[[1,2],[0,1]],[[1,2],[2,1]],[[1,0,0],[0,1,0]]]:
        s=preset('gaussian')['spec'];s['factors'][0]['params']['cov']=cov
        with pytest.raises(ModelError): compile_model(s)
    s=preset('beta')['spec'];s['factors'][0]['params']['alpha']='probability'
    with pytest.raises(ModelError,match='constant'):compile_model(s)
    s=preset('beta')['spec'];s['factors'][1]['value']=20
    with pytest.raises(ModelError,match='support'):compile_model(s)
    s=preset('gaussian')['spec'];s['factors'][0]['params']['loc']=[0,0,0]
    with pytest.raises(ModelError):compile_model(s)
    with pytest.raises(FloatingPointError,match='scale'):log_density('normal',1,loc=0,scale=-1)


def test_builder_serialization_immutable_data_and_custom_truth_guard():
    spec=preset('regression')['spec'];m=compile_model(spec);before=m([0.,0.])
    spec['data']['observed'][0]=100
    np.testing.assert_array_equal(m([0,0])[1],before[1])
    assert reference(spec)['kind']=='unavailable'
    clone=compile_model(json.dumps(m.spec));np.testing.assert_array_equal(clone([0,0])[1],before[1])


def test_invalid_numerical_target_is_explicit():
    m=Model();p=m.parameter('p',constraint={'type':'positive'});m.prior('prior','normal',p,loc=0,scale=1)
    lp,g=m.compile()([1000]);assert lp==-np.inf and np.isnan(g).all()


def test_distribution_allowlists_dot_shape_and_reporting_limits():
    with pytest.raises(ValueError,match='parameters'):log_density('half_normal',1,scale=1,loc=10)
    with pytest.raises(ValueError,match='parameters'):log_density('bernoulli_logits',7,logits=0,trials=10)
    with pytest.raises(ValueError,match='matching'):Tensor([1,2]).matmul(Tensor([3]))
    s=preset('beta')['spec'];s['report']=['probability','probability']
    with pytest.raises(ModelError,match='distinct'):compile_model(s)


@pytest.mark.parametrize('trials,count,prob',[(10,0,0),(10,10,1),(0,0,0),(0,0,1),(10,4,.3)])
def test_binomial_boundary_constants_match_independent_scipy(trials,count,prob):
    from scipy.stats import binom
    value=Tensor(prob,requires_grad=True)
    lp=log_density('binomial',count,trials=trials,prob=value);lp.backward()
    assert float(lp.value)==pytest.approx(binom.logpmf(count,trials,prob))
    expected=(count/prob if count else 0)-((trials-count)/(1-prob) if trials-count else 0)
    assert float(value.grad)==pytest.approx(expected)


def test_inline_observations_use_same_budget_as_named_data():
    model=Model();x=model.parameter('x');model.observe('obs','normal',[0.]*1001,loc=x,scale=1)
    with pytest.raises(ModelError,match='1000 observations'):model.compile()


def test_generative_funnel_reference_and_centered_noncentered_identity():
    ref=reference(preset('funnel')['spec']);draws=np.asarray(ref['reference_samples']);y=draws[:,0];residual=draws[:,1:]/np.exp(y[:,None]/2)
    assert abs(y.mean())<6*3/np.sqrt(len(y))
    assert abs(y.std(ddof=1)-3)<6*3/np.sqrt(2*len(y))
    assert abs(residual.mean())<6/np.sqrt(residual.size)
    assert abs(residual.std(ddof=1)-1)<6/np.sqrt(2*residual.size)
    centered=compile_model(preset('funnel')['spec']);noncentered=compile_model(preset('funnel',{'noncentered':True})['spec'])
    z=np.array([-2.,.2,-.1,.5,1.,-.8]);model_space=noncentered.report(z)
    assert noncentered(z)[0]-centered(model_space)[0]==pytest.approx(5*z[0]/2)


def test_mixture_composed_normalized_density_matches_independent_oracle():
    from scipy.stats import multivariate_normal
    m=compile_model(preset('mixture')['spec'])
    for z in [np.array([-4.,0]),np.array([4.,0]),np.array([0.,1]),np.array([100.,-100])]:
        expected=np.logaddexp(np.log(.7)+multivariate_normal.logpdf(z,[-4,0],np.diag([.25,1])),np.log(.3)+multivariate_normal.logpdf(z,[4,0],np.diag([.25,1])))
        assert m(z)[0]==pytest.approx(expected)
