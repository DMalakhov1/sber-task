"""Retrospective sensitivity and dependent-error uncertainty; never selects a model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .brent_lib.data import load_brent

ROOT = Path(__file__).resolve().parent


def block_interval(differences, block_length, samples=2000, seed=20261006):
    """Circular moving-block bootstrap of paired loss differences.

    Negative means candidate has smaller MAE. Exploratory CI, not a proof:
    overlapping forecasts, selection and distribution shifts remain limitations.
    """
    x=np.asarray(differences, dtype=float)
    if block_length < 1 or not np.isfinite(x).all():
        raise ValueError('Invalid observations or block length')
    result={'n':len(x), 'block_length':block_length,
            'mean_loss_difference':float(x.mean()) if len(x) else None}
    if len(x) < 4*block_length:
        return dict(result, status='insufficient_blocks', ci95=None)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(x),size=(samples,int(np.ceil(len(x)/block_length))))
    indexes=(starts[:,:,None]+np.arange(block_length))%len(x)
    means=x[indexes.reshape(samples,-1)[:,:len(x)]].mean(axis=1)
    return dict(result,status='exploratory',ci95=np.quantile(means,[.025,.975]).tolist())


def main():
    y=load_brent(); output=ROOT/'outputs'
    errors=pd.read_csv(output/'backtest_errors.csv',parse_dates=['origin','target'])
    comparisons=[]
    for h in (1,3,6,12,21):
        subset=errors[(errors.horizon==h)&(errors.origin>=pd.Timestamp('2022-01-01'))]
        paired=subset[subset.model=='mean_reversion'].merge(subset[subset.model=='naive'],on=['origin','target','horizon'],suffixes=('_model','_baseline'))
        delta=abs(paired.y_true_model-paired.y_pred_model)-abs(paired.y_true_baseline-paired.y_pred_baseline)
        comparisons.append(dict(horizon=h,**block_interval(delta,h)))
    periods={'2008':('2008-01-01','2008-12-01'), '2014_2016':('2014-01-01','2016-12-01'),
             '2020':('2020-01-01','2020-12-01'), 'validation_2022_plus':('2022-01-01','2099-12-01')}
    rows=[]
    # Same origins as each horizon's naive baseline. Crisis cohorts use TARGET dates.
    for h in (1,3,6,12,21):
        baseline=errors[(errors.model=='naive')&(errors.horizon==h)]
        for window in (60,120,180):
            for half_life in (6,12,24):
                predictions=[]
                for r in baseline.itertuples():
                    train=y.loc[:r.origin].tail(window)
                    if len(train)<window: continue
                    logmean=np.log(train).mean(); last=np.log(train.iloc[-1])
                    pred=np.exp(logmean+(last-logmean)*2**(-h/half_life))
                    predictions.append((r.origin,r.target,abs(r.y_true-pred),abs(r.y_true-r.y_pred)))
                frame=pd.DataFrame(predictions,columns=['origin','target','loss','baseline_loss'])
                for label,(a,b) in periods.items():
                    axis='origin' if label.startswith('validation') else 'target'
                    sample=frame[frame[axis].between(a,b)]
                    rows.append(dict(horizon=h,window=window,half_life=half_life,period=label,n=len(sample),
                        mae=sample.loss.mean(),naive_mae=sample.baseline_loss.mean()))
    pd.DataFrame(rows).to_csv(output/'mean_reversion_sensitivity.csv',index=False)
    (output/'paired_loss_uncertainty.json').write_text(json.dumps({
        'method':'circular moving-block bootstrap, paired absolute losses, fixed seed, 2000 resamples',
        'note':'Retrospective, not untouched testing; no correction for model selection. No CI below four blocks.',
        'results':comparisons},ensure_ascii=False,indent=2))
    print(json.dumps(comparisons,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
