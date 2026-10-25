"""Pooled54 CSP+LDA benchmark using shared train-fitted CSP/preprocessing."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
os.environ.setdefault('MNE_LOGGING_LEVEL','ERROR')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'4')
import joblib
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits
from src.plan_a.data import cached_trials


def summarize(rows):
    report={}
    for name,selected in [('low',[r for r in rows if r['training_low']]),('all',rows)]:
        metrics={}
        for method in ('csp_lda','late_dann'):
            metrics[method]={'mean_accuracy':float(np.mean([r[method+'_accuracy'] for r in selected])),
                             'at70':sum(r[method+'_accuracy']>=.7-1e-9 for r in selected),'subjects':len(selected)}
        metrics['dann_vs_csp']={'crossed_up':[r['subject'] for r in selected if r['csp_lda_accuracy']<.7-1e-9 and r['late_dann_accuracy']>=.7-1e-9],
                              'crossed_down':[r['subject'] for r in selected if r['csp_lda_accuracy']>=.7-1e-9 and r['late_dann_accuracy']<.7-1e-9]}
        metrics['dann_vs_csp']['net_at70']=len(metrics['dann_vs_csp']['crossed_up'])-len(metrics['dann_vs_csp']['crossed_down'])
        report[name]=metrics
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    p.add_argument('--dann-run',type=Path,default=Path('/mne_data/mne_data/bci_runs/warm_dann_pair_001/online/dann/seed_42'))
    p.add_argument('--dann-predictions',type=Path,default=Path('/mne_data/mne_data/bci_runs/warm_dann_pair_001_test/online/dann/seed_42'))
    p.add_argument('--data-root',type=Path,default=Path('/mne_data/mne_data'))
    p.add_argument('--cache-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/cache'))
    p.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/pooled_csp_lda_test_001'))
    args=p.parse_args()
    if args.output_root.exists():raise FileExistsError('Choose a new output directory')
    torch.set_num_threads(4)
    config=json.loads((args.dann_run/'config.json').read_text())
    prep=joblib.load(args.prepared/'preprocessing.joblib')
    actual=joblib.load(args.dann_run/'preprocessing.joblib')
    if config['subjects']!=list(range(1,55)) or config['domain_source']!='training':raise ValueError('Expected all54 training-defined groups')
    if prep['data_fingerprint']!=config['data_fingerprint']:raise ValueError('Training data mismatch')
    for key in ('mean','scale'):
        if not np.array_equal(prep[key],actual[key]):raise ValueError('DANN preprocessing differs: '+key)
    for key in ('filters_','patterns_'):
        if not np.array_equal(getattr(prep['csp'],key),getattr(actual['csp'],key)):raise ValueError('DANN CSP differs')
    for key in ('mean_','scale_'):
        if not np.array_equal(getattr(prep['csp_scaler'],key),getattr(actual['csp_scaler'],key)):raise ValueError('DANN CSP scaler differs')
    for s in config['subjects']:
        if not (args.dann_predictions/f'subject_{s}.npz').is_file():raise FileNotFoundError(args.dann_predictions/f'subject_{s}.npz')
    train=torch.load(args.prepared/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    _,labels,csp,_,subjects=train.tensors
    if len(labels)!=5400 or sorted(set(subjects.tolist()))!=config['subjects']:raise ValueError('Expected5400 training trials/all54')
    model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    with threadpool_limits(limits=4):model.fit(csp.numpy(),labels.numpy())
    if model.classes_.tolist()!=[0,1]:raise ValueError('Unexpected class order')
    print('TRAIN DONE: one global CSP+LDA, all54 subjects/5400 trials, original broadband preprocessing, no extra band filter',flush=True)
    args.output_root.mkdir(parents=True)
    joblib.dump({'lda':model,'preprocessing':prep},args.output_root/'model.joblib')
    (args.output_root/'predictions').mkdir()
    rows=[]
    for s in config['subjects']:
        eeg,y=cached_trials(args.data_root,args.cache_root,s,2,'test','online')
        saved=np.load(args.dann_predictions/f'subject_{s}.npz')
        if not np.array_equal(y,saved['labels']):raise ValueError('Test order/labels differ from DANN')
        if np.bincount(y).tolist()!=[50,50]:raise ValueError('Expected balanced100 trials as DANN benchmark')
        with threadpool_limits(limits=4):
            features=prep['csp_scaler'].transform(prep['csp'].transform(eeg.astype(np.float64)))
            probabilities=model.predict_proba(features)
        row={'subject':s,'training_low':int(prep['groups'][s]['training']),
             'csp_lda_accuracy':float(accuracy_score(y,probabilities.argmax(1))),
             'late_dann_accuracy':float(accuracy_score(y,saved['probabilities'].argmax(1)))}
        rows.append(row)
        np.savez_compressed(args.output_root/'predictions'/f'subject_{s}.npz',labels=y,csp_lda_probabilities=probabilities,dann_probabilities=saved['probabilities'])
        (args.output_root/'partial_subjects.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(f'{s}/54 CSP+LDA={row["csp_lda_accuracy"]:.2%} DANN={row["late_dann_accuracy"]:.2%}',flush=True)
    summary=summarize(rows)
    if summary['all']['late_dann']['at70']!=34:raise RuntimeError('DANN34 did not reproduce; partial results saved')
    report={'protocol':'One pooled54 CSP+LDA: same session1/train5400, identical saved train-fitted globalCSP6 and causal broadband preprocessing as warm DANN; no extra8-30Hz filter. Only LDA newly fit to TRAIN labels; session2/test prediction only. No hyperparameter search.',
            'limits':'Traditional algorithm benchmark, NOT isolated DANN contribution: deep temporal network, distillation, optimization and sampling differ. LDA fit uses all train rows once; does not replicate domain-balanced neural sampler. Known/reused test set, descriptive supplementary comparison. Fixed25 group retained.',
            'summary':summary,'subjects':rows}
    (args.output_root/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (args.output_root/'subject_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)
    print('DONE',args.output_root,flush=True)


if __name__=='__main__':main()
