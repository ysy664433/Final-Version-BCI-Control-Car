"""Fixed temporal centering, offline unlabeled-batch adaptation diagnostic only."""
import argparse
import json
from pathlib import Path
from audit_representation import metrics
import numpy as np
import torch
from src.plan_a.model import EEGNetLight


def center(train, train_subjects, target, target_subjects):
    result=target.copy()
    for subject in np.unique(target_subjects):
        source=train[train_subjects==subject,:128]
        selected=target_subjects==subject
        if not len(source): raise ValueError('Missing subject training reference')
        result[selected,:128] -= target[selected,:128].mean(0)-source.mean(0)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=Path('/mne_data/mne_data/bci_runs/representation_audit_002/features_and_predictions.npz'))
    parser.add_argument('--baseline',type=Path,default=Path('/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42'))
    parser.add_argument('--output',type=Path,default=Path('/mne_data/mne_data/bci_runs/centering_probe_001.json'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    torch.set_num_threads(4)
    z=np.load(args.features)
    config=json.loads((args.baseline/'config.json').read_text())
    model=EEGNetLight(**config['model']).eval()
    model.load_state_dict(torch.load(args.baseline/'best.pt',map_location='cpu',weights_only=False)['model'])
    original=z['validation_fusion']
    if original.shape[1]!=192:raise ValueError('Expected temporal128 + CSP64')
    corrected=center(z['train_fusion'],z['train_subjects'],original,z['validation_subjects'])
    report={'protocol':'OFFLINE BATCH ADAPTATION: all 100 unlabeled validation trials per subject used jointly, train subject mean reference, temporal128 only, fixed strength 1. No label-based tuning, no network updates, no session2. Assumes compatible class proportions. Not strict online inference or proof of novelty.', 'results':{}}
    for name,features in [('baseline',original),('centered',corrected)]:
        with torch.inference_mode():
            p=model.head(torch.from_numpy(features).float()).softmax(1).numpy()
        rows,summary=metrics(p,**{k:z['validation_'+k] for k in ('labels','domains','subjects')})
        report['results'][name]={'metrics':summary,'subjects':rows}
        if name=='baseline':
            changed=int(np.sum(p.argmax(1)!=z['validation_probabilities'].argmax(1)))
            report['baseline_changed_predictions']=changed
            if changed:raise RuntimeError(f'Baseline head changed {changed} predictions; stop before interpreting correction')
        print(name,json.dumps(summary),flush=True)
    base={r['subject']:r for r in report['results']['baseline']['subjects']}
    paired={}
    for group,domain in [('low',1),('all',None)]:
        rows=[r for r in report['results']['centered']['subjects'] if domain is None or r['domain']==domain]
        up=[r['subject'] for r in rows if base[r['subject']]['balanced_accuracy']<.7-1e-9 and r['balanced_accuracy']>=.7-1e-9]
        down=[r['subject'] for r in rows if base[r['subject']]['balanced_accuracy']>=.7-1e-9 and r['balanced_accuracy']<.7-1e-9]
        paired[group]={'crossed_up':up,'crossed_down':down,'net_at70':len(up)-len(down)}
    report['paired']=paired
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(paired,indent=2),flush=True)
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
