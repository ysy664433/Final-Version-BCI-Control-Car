"""Inference-only augmentation sensitivity probe. Session1 train only."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import json
import joblib
import numpy as np
import torch
from src.plan_a.model import EEGNetLight
from src.plan_a.data import cached_trials
from src.plan_b.augment import EEGAugment, prepare_view
from diagnose_seed42 import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,default=Path('/mne_data/mne_data/bci_runs'))
    p.add_argument('--data-root',type=Path,default=Path('/mne_data/mne_data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new output filename')
    torch.set_num_threads(4)
    run=args.base/'formal_a_001/online/baseline/seed_42'
    config=json.loads((run/'config.json').read_text())
    prep=joblib.load(run/'preprocessing.joblib')
    model=EEGNetLight(**config['model']).to(args.device).eval()
    model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=False)['model'])
    conditions={'clean':(0,0,0),'band_only':(.2,0,0),'noise_only':(0,0,.05),
                'mask_only':(0,.2,0),'combined':(.2,.2,.05)}
    rows=[]
    for s in config['subjects']:
        x,y=cached_trials(args.data_root,args.base/'cache',s,1,'train',config['mode'])
        rng=np.random.default_rng(2026+s)
        ix=np.concatenate([rng.choice(np.flatnonzero(y==c),min(6,int((y==c).sum())),replace=False) for c in [0,1]])
        x,y=x[ix],y[ix]
        normalized=((x-prep['mean'])/prep['scale']).astype(np.float32)
        for name,(band,mask,noise) in conditions.items():
            augmenter=None if name=='clean' else EEGAugment(band_strength=band,channel_mask=mask,noise_std=noise)
            for repeat in range(3):
                if name=='clean' and repeat: continue
                torch.manual_seed(2026+s*10+repeat)
                eeg,features=prepare_view(torch.from_numpy(normalized),prep,augmenter)
                metrics=evaluate(model,eeg.numpy(),y,features.numpy(),args.device,4)
                rows.append(dict(subject=s,domain=prep['groups'][s]['training'],condition=name,repeat=repeat,
                                 ce=metrics['ce'],accuracy=metrics['accuracy'],predicted_right_fraction=metrics['predicted_right_fraction']))
        print(f'subject {s} complete',flush=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in conditions:
        selected=[r for r in rows if r['condition']==name]
        print(name,'mean accuracy',np.mean([r['accuracy'] for r in selected]),'mean CE',np.mean([r['ce'] for r in selected]),flush=True)
    print('DONE',args.output,flush=True)
    print('Diagnostic only: 12 balanced training trials/subject, 3 random augmentation repeats. '
          'Sensitivity does not by itself prove a training failure mechanism.')


if __name__=='__main__': main()
