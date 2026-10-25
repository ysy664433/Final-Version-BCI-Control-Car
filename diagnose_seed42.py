"""Run from repository root. Reads session1 only; never trains or reads session2."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from src.plan_a.data import cached_trials
from src.plan_a.model import EEGNetLight, ShallowConvNetTeacher
from src.plan_b.model import EEGNetLightPlanB


@torch.inference_mode()
def evaluate(model, eeg, labels, csp, device, batch):
    losses, predictions, probabilities = [], [], []
    for start in range(0, len(labels), batch):
        x = torch.as_tensor(eeg[start:start+batch], device=device)
        y = torch.as_tensor(labels[start:start+batch], dtype=torch.long, device=device)
        logits = model(x) if csp is None else model(x, torch.as_tensor(csp[start:start+batch], device=device))
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite logits')
        losses.extend(F.cross_entropy(logits.float(), y, reduction='none').cpu().tolist())
        probabilities.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    predictions = np.asarray(predictions)
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    recalls = confusion.diagonal()/confusion.sum(1).clip(1)
    return dict(ce=float(np.mean(losses)), accuracy=float((predictions == labels).mean()),
                balanced_accuracy=float(recalls.mean()), predicted_right_fraction=float((predictions == 1).mean()),
                mean_right_probability=float(np.mean(probabilities)), confusion=confusion.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('/mne_data/mne_data/bci_runs'))
    parser.add_argument('--data-root', type=Path, default=Path('/mne_data/mne_data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    records = []
    runs = [('baseline', 'formal_a_001/online/baseline/seed_42'),
            ('plan_a', 'formal_a_001/online/plan_a/seed_42'),
            ('plan_b', 'shared_baseline_b_001/online/plan_b/seed_42')]
    reference = None
    for name, relative in runs:
        run = args.base/relative
        config = json.loads((run/'config.json').read_text())
        signature = {k:config[k] for k in ['subjects','data_fingerprint','domain_source','channels','protocol']}
        if reference is None: reference = signature
        if signature != reference: raise ValueError('Data/protocol mismatch')
        prep = joblib.load(run/'preprocessing.joblib')
        cls = EEGNetLightPlanB if config.get('framework') == 'plan_b' else EEGNetLight
        model = cls(**config['model']).to(args.device).eval()
        best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(best['model'])
        teacher = ShallowConvNetTeacher(len(config['channels'])).to(args.device).eval()
        last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        teacher.load_state_dict(last['teacher'])
        last_epoch = last['epoch']+1
        del last
        for subject in config['subjects']:
            for block in ['train', 'test']:
                raw, labels = cached_trials(args.data_root, args.base/'cache', subject, 1, block, config['mode'])
                csp = prep['csp_scaler'].transform(prep['csp'].transform(raw.astype(np.float64))).astype(np.float32)
                eeg = ((raw-prep['mean'])/prep['scale']).astype(np.float32)
                for label, net, features, epoch in [('best_student', model, csp, best['epoch']+1),
                                                    ('last_teacher', teacher, None, last_epoch)]:
                    row = dict(run=name, checkpoint=label, epoch=epoch, subject=subject,
                               domain=int(prep['groups'][subject][config['domain_source']]),
                               split='train_clean' if block == 'train' else 'validation')
                    row.update(evaluate(net,eeg,labels,features,args.device,args.batch_size))
                    records.append(row)
            print(f'{name}: subject {subject} complete',flush=True)
        del model, teacher
    (args.output/'subject_metrics.json').write_text(json.dumps(records,indent=2))
    summaries=[]
    for name,_ in runs:
        for checkpoint in ['best_student','last_teacher']:
            for split in ['train_clean','validation']:
                for domain in ['all',0,1]:
                    rows=[r for r in records if r['run']==name and r['checkpoint']==checkpoint and r['split']==split and (domain=='all' or r['domain']==domain)]
                    row=dict(run=name,checkpoint=checkpoint,split=split,domain=domain,subjects=len(rows))
                    for key in ['ce','accuracy','balanced_accuracy','predicted_right_fraction','mean_right_probability']:
                        row[key]=float(np.mean([r[key] for r in rows]))
                    summaries.append(row)
                    print(row,flush=True)
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    (args.output/'README.txt').write_text('Session1 only. No augmentation. eval() and FP32. Subject means. '
        'Best student and LAST teacher are different epochs; teacher is not the best-student teacher. '
        'Train metrics differ from historical training metrics due to eval mode, no augmentation and no resampling. '
        'Only load trusted checkpoints/joblib. No session2 used.\n')
    print('DONE:',args.output,flush=True)


if __name__=='__main__':
    main()
