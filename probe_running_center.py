"""Past-trial-only temporal centering in preserved cache order. No label tuning."""
import argparse
import json
from pathlib import Path
from audit_representation import metrics
import numpy as np
import torch
from src.plan_a.model import EEGNetLight


def running_center(train, train_subjects, target, target_subjects, prior=20):
    result=target.copy()
    for subject in np.unique(target_subjects):
        ref=train[train_subjects==subject,:128].mean(0,dtype=np.float64)
        total=np.zeros(128,dtype=np.float64)
        for n,index in enumerate(np.flatnonzero(target_subjects==subject)):
            # Predict using PAST trials; current trial enters state only afterwards.
            result[index,:128]=target[index,:128]-total/(prior+n)
            total+=target[index,:128]-ref
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=Path('/mne_data/mne_data/bci_runs/representation_audit_002/features_and_predictions.npz'))
    p.add_argument('--baseline',type=Path,default=Path('/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42'))
    p.add_argument('--output',type=Path,default=Path('/mne_data/mne_data/bci_runs/running_center_001.json'))
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    torch.set_num_threads(4)
    z=np.load(args.features)
    model=EEGNetLight(**json.loads((args.baseline/'config.json').read_text())['model']).eval()
    model.load_state_dict(torch.load(args.baseline/'best.pt',map_location='cpu',weights_only=False)['model'])
    report={'protocol':'Fixed prior=20; temporal128 only; reference is same subject session1 training mean; only previous target trials update state; current/future labels never used for correction; cache order preserved by loader, original onset timestamps not independently verified. Frozen validation-selected model, repeatedly reused validation, no session2, no network training. Assumes sufficiently representative past class mixture; not zero-calibration unseen-subject evaluation.', 'results':{}}
    target=z['validation_fusion']
    corrected=running_center(z['train_fusion'],z['train_subjects'],target,z['validation_subjects'])
    for name,features in [('baseline',target),('running_center',corrected)]:
        with torch.inference_mode():prob=model.head(torch.from_numpy(features).float()).softmax(1).numpy()
        if name=='baseline' and np.any(prob.argmax(1)!=z['validation_probabilities'].argmax(1)):
            raise RuntimeError('Baseline predictions differ; stop before interpreting')
        rows,summary=metrics(prob,**{k:z['validation_'+k] for k in ('labels','domains','subjects')})
        report['results'][name]={'metrics':summary,'subjects':rows}
        blocks=[]
        for start in (0,25,50,75):
            selected=np.zeros(len(target),dtype=bool)
            for subject in np.unique(z['validation_subjects']):
                ids=np.flatnonzero(z['validation_subjects']==subject)[start:start+25]
                selected[ids]=True
            group=selected & (z['validation_domains']==1)
            blocks.append({'trial_positions':[start+1,start+25],'low_pooled_accuracy':float((prob[group].argmax(1)==z['validation_labels'][group]).mean())})
        report['results'][name]['quarter_blocks']=blocks
        print(name,json.dumps(summary),flush=True)
    base={r['subject']:r for r in report['results']['baseline']['subjects']}
    report['paired']={}
    for name,domain in [('low',1),('all',None)]:
        rows=[r for r in report['results']['running_center']['subjects'] if domain is None or r['domain']==domain]
        up=[r['subject'] for r in rows if base[r['subject']]['balanced_accuracy']<.7-1e-9 and r['balanced_accuracy']>=.7-1e-9]
        down=[r['subject'] for r in rows if base[r['subject']]['balanced_accuracy']>=.7-1e-9 and r['balanced_accuracy']<.7-1e-9]
        report['paired'][name]={'crossed_up':up,'crossed_down':down,'net_at70':len(up)-len(down)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['paired'],indent=2),flush=True)
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
