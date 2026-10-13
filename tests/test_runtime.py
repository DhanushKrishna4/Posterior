import copy
import numpy as np
import pytest
from posterior_lab.examples import preset
from posterior_lab.runtime import Experiment
from posterior_lab.serialization import dumps,loads,draws_csv
from posterior_lab.prediction import predict
from posterior_lab.bridge import call


def finish(run):
    while run.status=='running':run.step(25)
    assert run.status=='complete',run.reason
    return run


@pytest.mark.parametrize('algorithm',['hmc','nuts','vi'])
def test_checkpoint_resume_matches_uninterrupted_exactly(algorithm):
    settings=dict(algorithm=algorithm,chains=2,warmup=60,draws=25,iterations=60,max_seconds=30)
    a=Experiment(preset('beta')['spec'],settings);b=Experiment(preset('beta')['spec'],settings)
    finish(a);b.step(17);data=loads(dumps(b.checkpoint()));b=Experiment.restore(data);finish(b)
    np.testing.assert_array_equal(a.samples,b.samples)
    np.testing.assert_array_equal(a.unconstrained_samples,b.unconstrained_samples)
    if algorithm!='vi':
        assert a.transitions==b.transitions
        assert a.transition_summaries==b.transition_summaries
        for ca,cb in zip(a.chains,b.chains):
            assert ca['rng'].bit_generator.state==cb['rng'].bit_generator.state
            assert ca['adapter'].state_dict()==cb['adapter'].state_dict()
    else: assert [v.state_dict() for v in a.vi]==[v.state_dict() for v in b.vi]


@pytest.mark.parametrize('algorithm',['hmc','nuts','vi'])
def test_gradient_budget_rolls_back_entire_transition(algorithm):
    run=Experiment(preset('gaussian')['spec'],dict(algorithm=algorithm,chains=1,warmup=0,draws=10,max_gradient_evaluations=1,max_seconds=30))
    before=run.checkpoint();run.step();after=run.checkpoint()
    assert run.status=='budget_exhausted'
    for key in ['chains','vi','samples','unconstrained_samples','transition_summaries','next_chain']:assert before[key]==after[key]
    assert after['evaluations']==1


def test_freeze_rejections_and_summary_keeps_all_draws():
    run=finish(Experiment(preset('gaussian')['spec'],dict(algorithm='hmc',chains=2,warmup=0,draws=12,step_size=4,steps=20)))
    assert all(len(c)==12 for c in run.samples)
    for c in run.chains: assert c['adapter'].frozen
    assert all(np.all(np.asarray(c)==c[0]) for c in run.samples)
    assert run.snapshot()['diagnostics']['status']=='unavailable'
    assert len(draws_csv(run.samples,run.model.reporting_names).splitlines())==25


def test_viewing_diagnostics_prediction_never_changes_sampling_rng():
    a=Experiment(preset('regression')['spec'],dict(chains=2,warmup=12,draws=12));b=Experiment(preset('regression')['spec'],dict(chains=2,warmup=12,draws=12))
    for _ in range(36):
        a.step();a.snapshot();predict(a.spec,a.samples)
        b.step()
    finish(a);finish(b);np.testing.assert_array_equal(a.samples,b.samples)


def test_prediction_includes_noise_on_independent_stream():
    run=finish(Experiment(preset('regression')['spec'],dict(chains=2,warmup=50,draws=50)))
    result=predict(run.spec,run.samples,count=1000)
    mean=np.asarray(result['mean_band']);pred=np.asarray(result['predictive_band'])
    assert np.all(pred[1]-pred[0]>mean[1]-mean[0])
    assert result==predict(run.spec,run.samples,count=1000)


@pytest.mark.parametrize('mutation',[
    lambda s:s.update(model_hash='bad'),lambda s:s['chains'][0].update(z=[1,2]),
    lambda s:s['chains'][0].update(iteration=100000),lambda s:s['chains'][0]['adapter'].update(scales=[0]),
    lambda s:s['samples'][0].append([float('nan')]),lambda s:s['chains'][0]['rng'].update(bit_generator='evil'),
    lambda s:s['settings'].update(max_depth=25),lambda s:s.update(next_chain=100),
])
def test_checkpoint_resource_and_consistency_checks(mutation):
    run=Experiment(preset('beta')['spec'],dict(chains=1));run.step(2);data=run.checkpoint();mutation(data)
    with pytest.raises((ValueError,KeyError)):Experiment.restore(data)


def test_vi_checkpoint_must_match_outer_model_and_settings():
    run=Experiment(preset('gaussian')['spec'],dict(algorithm='vi',chains=1));run.step();data=run.checkpoint();data['settings']['batch_size']=9
    with pytest.raises(ValueError,match='settings'):Experiment.restore(data)


def test_bridge_is_data_only_and_reports_errors():
    result=loads(call('{"op":"compile","spec":{"version":5}}'))
    assert 'error' in result
    assert 'error' in loads(call('{"op":"eval","code":"print(1)"}'))
    with pytest.raises(ValueError):loads('{"x":NaN}')
    with pytest.raises(ValueError):loads(' '*32_000_001)


def test_changed_target_has_new_identity_and_no_mutation_of_running_chain():
    original=Experiment(preset('beta')['spec']);changed=Experiment(preset('beta',{'successes':3})['spec'])
    assert original.model.hash!=changed.model.hash
    assert original.run_id!=changed.run_id
    saved=copy.deepcopy(original.spec);preset('beta',{'successes':9});assert original.spec==saved


@pytest.mark.parametrize('corruption',['next_chain','stats','reported','adapter_target'])
def test_reviewed_checkpoint_corruptions_are_rejected_before_commit(corruption):
    run=Experiment(preset('beta')['spec'],dict(chains=2,warmup=0,draws=2));run.step(3);data=run.checkpoint()
    if corruption=='next_chain':data['next_chain']=0
    elif corruption=='stats':data['stats']={}
    elif corruption=='reported':data['samples'][0][0][0]=999
    else:data['chains'][0]['adapter']['target_accept']=.9
    with pytest.raises(ValueError):Experiment.restore(data)


def test_completed_runs_cannot_add_an_extra_draw_even_after_direct_status_change():
    run=finish(Experiment(preset('beta')['spec'],dict(chains=2,warmup=0,draws=2)))
    with pytest.raises(ValueError,match='finished'):run.set_status('running')
    run.status='running';run.step();assert list(map(len,run.samples))==[2,2]


def test_custom_graph_does_not_get_preset_specific_interpretation():
    spec=preset('funnel')['spec'];spec['report']=['children','y']
    run=Experiment(spec,dict(chains=1,warmup=0,draws=1))
    assert 'narrow_coverage' not in run.snapshot()
    assert run.reference['kind']=='unavailable'


def test_mcmc_json_through_javascript_keeps_exact_pcg64_state():
    import shutil,subprocess
    if not shutil.which('node'):pytest.skip('Node unavailable for browser serialization reference')
    a=Experiment(preset('beta')['spec'],dict(chains=2,warmup=12,draws=12));a.step(13)
    encoded=dumps(a.checkpoint())
    through_node=subprocess.run(['node','-e',"let s='';process.stdin.on('data',x=>s+=x);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s))));"],input=encoded,text=True,capture_output=True,check=True).stdout
    b=Experiment.restore(loads(through_node));finish(a);finish(b)
    np.testing.assert_array_equal(a.samples,b.samples)
    assert a.checkpoint()['chains'][0]['rng']==b.checkpoint()['chains'][0]['rng']
    with pytest.raises(ValueError,match='nonfinite'):loads('{"x":1e999}')


@pytest.mark.parametrize('algorithm',['hmc','nuts','vi'])
def test_incremental_javascript_checkpoints_resume_exactly(algorithm):
    import shutil,subprocess
    if not shutil.which('node'):pytest.skip('Node unavailable')
    a=Experiment(preset('beta')['spec'],dict(algorithm=algorithm,chains=2,warmup=4,draws=20,iterations=20,max_seconds=30))
    initial=a.snapshot();deltas=[]
    for _ in range(17):a.step();deltas.append(a.delta())
    script="""import {EngineClient} from './web/client.js';
let input='';for await (const chunk of process.stdin)input+=chunk;
const data=JSON.parse(input),client=new EngineClient();client.result=data.initial;
for(const delta of data.deltas){client.result=client.mergeDelta(delta);client.result=client.mergeDelta(delta);}
process.stdout.write(JSON.stringify(client.result.checkpoint));
"""
    result=subprocess.run(['node','--input-type=module','-e',script],input=dumps({'initial':initial,'deltas':deltas}),text=True,capture_output=True,check=True)
    b=Experiment.restore(loads(result.stdout));finish(a);finish(b)
    np.testing.assert_array_equal(a.samples,b.samples)
    if algorithm!='vi':assert a.transition_summaries==b.transition_summaries


def test_native_and_browser_experiment_envelopes_share_one_checkpoint():
    from posterior_lab.serialization import extract_checkpoint
    run=Experiment(preset('beta')['spec'],dict(chains=1,warmup=1,draws=2));run.step()
    snapshot=run.snapshot();checkpoint=snapshot['checkpoint']
    for envelope in [snapshot,checkpoint,{'format':'posterior-experiment','version':1,'result':{'checkpoint':checkpoint}}]:
        restored=Experiment.restore(extract_checkpoint(envelope));assert restored.chains[0]['rng'].bit_generator.state==run.chains[0]['rng'].bit_generator.state
