import numpy as np
import pandas as pd
import pytest
from task1_brent.brent_lib.neural import make_windows, chronological_inner_split, correct_quantiles, fit_neural
from task1_brent.extended_experiment import split_plan, origins_for, select_on_validation, quantile_metrics


def test_future_perturbation_cannot_change_training_windows():
    values=np.exp(np.linspace(2,4,300));other=values.copy();other[241:]*=100
    x,y,origins=make_windows(values,240)
    xx,yy,_=make_windows(other,240)
    np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
    assert origins[-1]+21==240
    tr,va=chronological_inner_split(origins,21)
    assert origins[tr[-1]]+21 < origins[va[0]]
    assert np.allclose(np.exp(y[0])*values[origins[0]],values[origins[0]+1:origins[0]+22])


def test_all_horizons_stay_in_assigned_period():
    s=pd.Series(np.ones(467),index=pd.date_range('1987-05-01',periods=467,freq='MS'))
    a,b,parts=split_plan(s)
    assert sum(p['n'] for p in parts)==467
    for lo,hi in [(a,b),(b,len(s))]:
        for t in origins_for(lo,hi,3):
            assert lo <= t+1 <= t+21 < hi


def test_test_metrics_cannot_select_model():
    metrics=pd.DataFrame([dict(period=phase,model=m,horizon=1,MAE=v) for phase,m,v in
                          [('validation','a',1),('validation','b',2),('test','a',100),('test','b',0)]])
    assert select_on_validation(metrics)=={1:'a'}
    metrics.loc[metrics.period=='test','MAE']=-1000
    assert select_on_validation(metrics)=={1:'a'}


def test_isotonic_is_ordered_without_changing_ordered_rows():
    q=np.arange(19,dtype=float);np.testing.assert_array_equal(correct_quantiles(q),q)
    q[4:7]=[8,2,1];corrected=correct_quantiles(q)
    assert (np.diff(corrected)>=0).all() and np.isfinite(corrected).all()


def test_distribution_metrics_penalize_false_narrow_interval():
    actual=np.array([0.,10.]);narrow=np.tile(np.linspace(4,6,19),(2,1));wide=np.tile(np.linspace(-1,11,19),(2,1))
    n,w=quantile_metrics(actual,narrow),quantile_metrics(actual,wide)
    assert n['coverage90']==0 and w['coverage90']==1 and n['interval_score90']>w['interval_score90']


@pytest.mark.parametrize('kind',['rnn','lstm','cnn','qrnn'])
def test_real_neural_cpu_fit_reproducible_and_no_future(kind):
    pytest.importorskip('torch')
    values=np.exp(3+np.sin(np.arange(240)/15)*.1)
    p,_,meta=fit_neural(values,210,kind,epochs=2,seeds=(11,))
    changed=values.copy();changed[211:]*=3
    pp,_,_=fit_neural(changed,210,kind,epochs=2,seeds=(11,))
    np.testing.assert_allclose(p,pp,rtol=0,atol=0)
    assert p.shape==((21,19) if kind=='qrnn' else (21,)) and (p>0).all()
    assert meta['max_training_target']<=210
    assert meta['inner_train_max_target']<meta['inner_validation_first_origin']
    if kind=='qrnn':assert (np.diff(p,axis=-1)>=0).all()
