"""Read existing checkpoint optimizer/scaler states. No training or dataset access."""
import argparse
import json
from pathlib import Path
import torch


def optimizer_stats(state, expected):
    steps=[]
    for value in state.get('state',{}).values():
        if 'step' in value:
            step=value['step']
            steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
    return {'parameters_with_step_state':len(steps),'min_step':min(steps) if steps else None,
            'max_step':max(steps) if steps else None,
            'expected_step_calls':expected,
            'estimated_skipped_step_calls':expected-max(steps) if steps else None,
            'note':'Estimate assumes uninterrupted original loop; step counters do not locate when skips occurred. Unused parameters may have no state.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/mne_data/mne_data/bci_runs/dann_pair_43_44_001'))
    p.add_argument('--prepared',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    p.add_argument('--output',type=Path,default=Path('/mne_data/mne_data/bci_runs/dann_step_audit_001.json'))
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    torch.set_num_threads(4)
    train=torch.load(args.prepared/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    n=len(train);del train
    report=[]
    for seed in (43,44):
        for variant in ('baseline','plan_a'):
            directory=args.root/'online'/variant/f'seed_{seed}'
            config=json.loads((directory/'config.json').read_text())
            checkpoint=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
            epochs=int(checkpoint['epoch'])+1
            batch=int(config['hyperparameters']['batch_size'])
            expected=epochs*((n+batch-1)//batch)
            row={'seed':seed,'variant':variant,'completed_epochs':epochs,
                 'student':optimizer_stats(checkpoint['optimizer'],expected),
                 'teacher':optimizer_stats(checkpoint['teacher_optimizer'],expected),
                 'scaler':checkpoint.get('scaler',{}),
                 'student_current_lr':[g['lr'] for g in checkpoint['optimizer']['param_groups']],
                 'nonfinite_student_tensors':[k for k,v in checkpoint['model'].items() if torch.is_tensor(v) and not torch.isfinite(v).all()],
                 'nonfinite_teacher_tensors':[k for k,v in checkpoint['teacher'].items() if torch.is_tensor(v) and not torch.isfinite(v).all()],
                 'last_checkpoint_has_teacher':'teacher' in checkpoint}
            del checkpoint
            best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
            row['best_epoch']=int(best['epoch'])+1
            row['best_checkpoint_has_teacher']='teacher' in best
            del best
            report.append(row)
            print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
