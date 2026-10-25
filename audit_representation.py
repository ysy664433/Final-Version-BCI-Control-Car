"""All-subject representation audit using existing baseline and shared session1 tensors.

No neural optimizer steps, no personal fine tuning, no session2, no hyperparameter sweep.
Fits two fixed global linear probes using TRAIN labels only. This is diagnosis,
not proof of a novel method; frozen features originate from a validation-selected model.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'4')
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.plan_a.model import EEGNetLight


def metrics(probabilities, labels, domains, subjects):
    predictions=probabilities.argmax(1)
    rows=[]
    for subject in np.unique(subjects):
        sel=subjects==subject
        recalls=[float((predictions[sel & (labels==c)]==c).mean()) for c in (0,1)]
        p=probabilities[np.flatnonzero(sel),labels[sel]].clip(1e-12,1)
        rows.append({'subject':int(subject),'domain':int(domains[sel][0]),'balanced_accuracy':float(np.mean(recalls)),
                     'accuracy':float((predictions[sel]==labels[sel]).mean()),'ce':float(-np.log(p).mean()),
                     'trials':int(sel.sum()),'predicted_right_fraction':float((predictions[sel]==1).mean())})
    report={}
    for name,domain in [('low',1),('high',0),('all',None)]:
        selected=[r for r in rows if domain is None or r['domain']==domain]
        report[name]={'subjects':len(selected),'mean_bacc':float(np.mean([r['balanced_accuracy'] for r in selected])),
                      'at_least70':sum(r['balanced_accuracy']>=.7-1e-9 for r in selected),
                      'mean_ce':float(np.mean([r['ce'] for r in selected]))}
    return rows,report


@torch.inference_mode()
def extract(model, data, shared_prep, baseline_prep, device, batch_size):
    eeg,labels,_,domains,subjects=data.tensors
    if labels.ndim != 1 or domains.ndim != 1 or subjects.ndim != 1:
        raise ValueError('Expected dataset order: EEG, labels, CSP features, domains, subjects')
    outputs={'probabilities':[],'fusion':[],'csp':[]}
    for start in range(0,len(labels),batch_size):
        sl=slice(start,start+batch_size)
        # Reconstruct native values and use the BASELINE's own saved preprocessing.
        # This avoids silently assuming separately fitted CSPs have identical ordering.
        raw=eeg[sl].numpy().astype(np.float64)*shared_prep['scale']+shared_prep['mean']
        x=((raw-baseline_prep['mean'])/baseline_prep['scale']).astype(np.float32)
        c=baseline_prep['csp_scaler'].transform(baseline_prep['csp'].transform(raw)).astype(np.float32)
        logits,_,features=model(torch.as_tensor(x,device=device),torch.as_tensor(c,device=device),grl_coefficient=0.)
        if not torch.isfinite(logits).all() or not torch.isfinite(features).all(): raise ValueError('Non-finite model output')
        outputs['probabilities'].append(logits.float().softmax(1).cpu().numpy())
        outputs['fusion'].append(features.float().cpu().numpy());outputs['csp'].append(c)
        if start==0 or (start+batch_size)%800==0: print('Extracted',min(start+batch_size,len(labels)),'/',len(labels),flush=True)
    result={k:np.concatenate(v) for k,v in outputs.items()}
    result.update(labels=labels.numpy(),domains=domains.numpy(),subjects=subjects.numpy())
    return result


def fit_probes(train,validation):
    joint=train['domains']*2+train['labels']
    counts=np.bincount(joint,minlength=4)
    if (counts==0).any():raise ValueError('Every training domain/class cell is required')
    weights=len(joint)/(4*counts[joint])
    classifiers={
        'csp_linear':(make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',random_state=42)),'csp'),
        'fusion_linear':(make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',random_state=42)),'fusion'),
    }
    output={}
    for name,(classifier,features) in classifiers.items():
        start=time.monotonic()
        classifier.fit(train[features],train['labels'],standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
        if not np.array_equal(classifier.classes_,np.array([0,1])): raise ValueError('Unexpected task label order')
        output[name]={'classifier':classifier,'train_probabilities':classifier.predict_proba(train[features]),
                      'validation_probabilities':classifier.predict_proba(validation[features]),'fit_seconds':time.monotonic()-start}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    p.add_argument('--baseline',type=Path,default=Path('/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42'))
    p.add_argument('--output',type=Path,default=Path('/mne_data/mne_data/bci_runs/representation_audit_001'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new output directory; old results not overwritten')
    if args.batch_size<1: raise ValueError('Positive batch size required')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    config=json.loads((args.baseline/'config.json').read_text())
    metadata=json.loads((args.prepared_root/'metadata.json').read_text())
    baseline_prep=joblib.load(args.baseline/'preprocessing.joblib')
    shared_prep=joblib.load(args.prepared_root/'preprocessing.joblib')
    if len(config['subjects'])!=54 or config['domain_source']!='training' or config['mode']!='online': raise ValueError('Expected recorded 54-subject online training-group baseline')
    if metadata['data_fingerprint']!=config['data_fingerprint'] or baseline_prep['data_fingerprint']!=config['data_fingerprint']:
        raise ValueError('Training data fingerprint differs')
    model=EEGNetLight(**config['model']).to(args.device).eval()
    checkpoint=torch.load(args.baseline/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    args.output.mkdir(parents=True)
    datasets={}
    for split,file in [('train','train.pt'),('validation','validation.pt')]:
        print('EXTRACT',split,flush=True)
        data=torch.load(args.prepared_root/file,map_location='cpu',weights_only=False,mmap=True)
        if len(data)!=5400 or sorted(set(data.tensors[4].tolist()))!=config['subjects']:
            raise ValueError('Expected all 54 subjects and 5400 trials per split')
        expected=np.array([baseline_prep['groups'][int(s)]['training'] for s in data.tensors[4]])
        if not np.array_equal(expected,data.tensors[3].numpy()): raise ValueError('Domain membership differs')
        datasets[split]=extract(model,data,shared_prep,baseline_prep,args.device,args.batch_size)
    rows,baseline_report=metrics(datasets['validation']['probabilities'],**{k:datasets['validation'][k] for k in ['labels','domains','subjects']})
    recorded={r['subject']:r for r in json.loads((args.baseline/'validation_subjects.json').read_text())}
    deviations={r['subject']:r['balanced_accuracy']-recorded[r['subject']]['balanced_accuracy'] for r in rows}
    (args.output/'baseline_reproduction.json').write_text(json.dumps({'metrics':baseline_report,'subject_deviations':deviations},indent=2))
    if max(abs(v) for v in deviations.values())>1e-8:
        raise RuntimeError('Baseline per-subject accuracies did not reproduce. Inspect baseline_reproduction.json before interpreting probe results; one changed trial can matter at the 70% threshold.')
    probes=fit_probes(datasets['train'],datasets['validation'])
    results={'baseline_head':{split:metrics(data['probabilities'],**{k:data[k] for k in ['labels','domains','subjects']}) for split,data in datasets.items()}}
    for name,probe in probes.items():
        results[name]={split:metrics(probe[split+'_probabilities'],**{k:datasets[split][k] for k in ['labels','domains','subjects']}) for split in datasets}
        joblib.dump(probe['classifier'],args.output/(name+'.joblib'))
        print(name,'fit seconds',probe['fit_seconds'],flush=True)
    report={name:{split:values[1] for split,values in splits.items()} for name,splits in results.items()}
    paired={}
    for name,splits in results.items():
        current=splits['validation'][0]
        low=[r for r in current if r['domain']==1]
        up=[r['subject'] for r in low if recorded[r['subject']]['balanced_accuracy']<.7-1e-9 and r['balanced_accuracy']>=.7-1e-9]
        down=[r['subject'] for r in low if recorded[r['subject']]['balanced_accuracy']>=.7-1e-9 and r['balanced_accuracy']<.7-1e-9]
        paired[name]={'low_crossed_up':up,'low_crossed_down':down,'net_low_at70':len(up)-len(down)}
        (args.output/(name+'_subjects.json')).write_text(json.dumps(current,indent=2))
    np.savez_compressed(args.output/'features_and_predictions.npz',**{split+'_'+k:v for split,data in datasets.items() for k,v in data.items()},
                        **{name+'_validation_probabilities':v['validation_probabilities'] for name,v in probes.items()})
    final={'metrics':report,'paired_validation':paired,'fixed_hyperparameters':'Both probes: weighted scaler+LR C=1; inverse domain/class frequency weights match the ORIGINAL sampler expectation (four cells equally weighted); no validation tuning',
           'limits':'Global linear probes fit train labels only. Frozen baseline features came from a validation-selected model. Same repeatedly used validation; not session2 evidence or algorithm novelty. No personal tuning or neural weight updates.'}
    (args.output/'summary.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final,indent=2),flush=True)
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
