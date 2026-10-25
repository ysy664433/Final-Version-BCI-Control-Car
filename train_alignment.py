"""Full-data paired global vs label-wise domain alignment, from scratch.

Run from the project root. Reuses the original training-only data preparation.
No source edits, personal fine tuning, synthetic data in normal runs, or session2 access.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from src.plan_a.data import MOTOR_CHANNELS
from src.plan_a.model import EEGNetLight, GradientReverseLayer, ShallowConvNetTeacher
from src.plan_a.train import prepare, set_seed, subject_metrics, checkpoint_save, write_json


class BalancedBatches(Sampler):
    def __init__(self, labels, domains, batch_size, seed):
        self.cells = [torch.where((domains == d) & (labels == y))[0] for d in (0, 1) for y in (0, 1)]
        if any(len(cell) == 0 for cell in self.cells):
            raise ValueError('Every training domain must contain both task classes')
        self.batch_size, self.seed = batch_size, seed
        self.batches = math.ceil(len(labels) / batch_size)

    def __len__(self):
        return self.batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.batches):
            parts = [cell[torch.randint(len(cell), (self.batch_size // 4,), generator=generator)] for cell in self.cells]
            indices = torch.cat(parts)
            yield indices[torch.randperm(self.batch_size, generator=generator)].tolist()


class AlignmentModel(nn.Module):
    def __init__(self, alignment, seed, dropout=.1):
        super().__init__()
        self.alignment = alignment
        self.backbone = EEGNetLight(11, dropout=dropout, enhance=True, dann=False, causal=True)
        # Domain-head construction must not change the common backbone RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 2)
            head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 2))
        self.domain_heads = nn.ModuleList([copy.deepcopy(head) for _ in range(2 if alignment == 'classwise' else 1)])

    def forward(self, eeg, csp, coefficient=None):
        logits, _, features = self.backbone(eeg, csp, grl_coefficient=0.)
        if coefficient is None:
            return logits
        reverse = GradientReverseLayer.apply(features, coefficient)
        devices = [features.device.index if features.device.index is not None else torch.cuda.current_device()] if features.is_cuda else []
        # Extra discriminator dropout must not alter later teacher/student dropout.
        with torch.random.fork_rng(devices=devices):
            domain_logits = torch.stack([head(reverse) for head in self.domain_heads], dim=1)
        return logits, domain_logits


def alignment_loss(domain_logits, domains, labels, alignment):
    parts = []
    for domain in (0, 1):
        for label in (0, 1):
            selected = (domains == domain) & (labels == label)
            if not selected.any():
                raise ValueError('Balanced batch invariant broken')
            head = label if alignment == 'classwise' else 0
            parts.append(F.cross_entropy(domain_logits[selected, head], domains[selected]))
    return torch.stack(parts).mean()


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    ys, ps, ss, ds = [], [], [], []
    total_ce, count = 0., 0
    for eeg, targets, csp, domains, subjects in loader:
        logits = model(eeg.to(device), csp.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite validation logits')
        total_ce += F.cross_entropy(logits.float(), targets.to(device), reduction='sum').item()
        count += len(targets)
        ys.extend(targets.tolist()); ps.extend(logits.argmax(1).cpu().tolist())
        ss.extend(subjects.tolist()); ds.extend(domains.tolist())
    rows = subject_metrics(np.array(ys), np.array(ps), np.array(ss), np.array(ds))
    low = [r for r in rows if r['domain'] == 1]
    return float(np.mean([r['balanced_accuracy'] for r in low])), rows, total_ce / count


def train_one(args, alignment, train_data, validation_data, preprocessing):
    output = args.output_root / 'online' / alignment / f'seed_{args.seed}'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('resume', 'output_root')}
    source_dir = Path('src/plan_a')
    config.update({'framework': 'supervised_classwise_alignment_v1', 'alignment': alignment,
                   'data_fingerprint': preprocessing['data_fingerprint'],
                   'protocol': {'train': 'session1/EEG_MI_train', 'validation': 'session1/EEG_MI_test', 'test': 'unused'},
                   'groups_csv_sha256': hashlib.sha256(args.groups_csv.read_bytes()).hexdigest() if not args.synthetic else 'synthetic',
                   'code_sha256': hashlib.sha256(Path(__file__).read_bytes() + b''.join((source_dir / name).read_bytes() for name in ('train.py','model.py','data.py'))).hexdigest(),
                   'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda': torch.version.cuda},
                   'notes': 'Supervised task-label routing; two classwise domain heads vs one global head. '
                            'All batches domain/class balanced with replacement. Original KD, CE, GRL schedule, LR scheduler. '
                            'Not standard CDAN reproduction. Not interchangeable with original checkpoint formats.'})
    if (output / 'config.json').exists():
        if not args.resume:
            raise FileExistsError(f'{output} exists; use another root or --resume with identical options')
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration/source/data differs')
    elif args.resume and output.exists():
        raise ValueError('Incomplete output directory')
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = AlignmentModel(alignment, args.seed).to(args.device)
    set_seed(args.seed + 1)
    teacher = ShallowConvNetTeacher(11).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=.5)
    amp = args.amp and args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, start_epoch, best_score, patience = [], 0, -1., 0
    actual = {'student': 0, 'teacher': 0}
    if args.resume and (output / 'last.pt').exists():
        cp = torch.load(output / 'last.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(cp['model']); teacher.load_state_dict(cp['teacher'])
        optimizer.load_state_dict(cp['optimizer']); teacher_optimizer.load_state_dict(cp['teacher_optimizer'])
        scheduler.load_state_dict(cp['scheduler']); scaler.load_state_dict(cp['scaler'])
        history, start_epoch, best_score, patience = cp['history'], cp['epoch'] + 1, cp['best_score'], cp['patience']
        actual.update(cp['actual_steps'])
        del cp
    elif args.resume and (output / 'config.json').exists():
        raise FileNotFoundError('Missing last.pt')
    def student_step(*unused): actual['student'] += 1
    def teacher_step(*unused): actual['teacher'] += 1
    optimizer.register_step_post_hook(student_step)
    teacher_optimizer.register_step_post_hook(teacher_step)
    joblib.dump(preprocessing, output / 'preprocessing.joblib')
    write_json(output / 'config.json', config)
    loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.device.startswith('cuda'))
    labels, domains = train_data.tensors[1], train_data.tensors[3]
    for epoch in range(start_epoch, args.epochs):
        if patience >= args.patience: break
        started = time.monotonic()
        set_seed(args.seed + epoch * 1009)
        batches = BalancedBatches(labels, domains, args.batch_size, args.seed + epoch)
        train_loader = DataLoader(train_data, batch_sampler=batches, num_workers=args.workers,
                                  pin_memory=args.device.startswith('cuda'), generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train(); teacher.train()
        totals = {k: 0. for k in ['loss','ce','kd','domain','teacher_ce','train_accuracy','teacher_accuracy']}
        examples = 0
        for index, (eeg, targets, csp, ds, _) in enumerate(train_loader):
            eeg, targets, csp, ds = [v.to(args.device) for v in (eeg, targets, csp, ds)]
            teacher_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, enabled=amp):
                teacher_logits = teacher(eeg)
                teacher_ce = F.cross_entropy(teacher_logits, targets)
            scaler.scale(teacher_ce).backward()
            scaler.unscale_(teacher_optimizer)
            nn.utils.clip_grad_norm_(teacher.parameters(), 5.)
            scaler.step(teacher_optimizer)
            progress = (epoch + index / len(train_loader)) / max(1, args.epochs - 1)
            coefficient = args.grl_max * (2 / (1 + math.exp(-10 * progress)) - 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, enabled=amp):
                logits, domain_logits = model(eeg, csp, coefficient)
                ce = F.cross_entropy(logits, targets)
                kd = F.kl_div(F.log_softmax(logits.float()/args.temperature, 1),
                              F.softmax(teacher_logits.detach().float()/args.temperature, 1), reduction='batchmean') * args.temperature**2
                variance = logits.float().softmax(1).var(dim=1, unbiased=False).mean()
                domain = alignment_loss(domain_logits, ds, targets, alignment)
                loss = (1-args.distill_alpha)*(ce+args.variance_weight*variance)+args.distill_alpha*kd+args.domain_weight*domain
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
            n = len(targets); examples += n
            for key, value in [('loss',loss),('ce',ce),('kd',kd),('domain',domain),('teacher_ce',teacher_ce)]:
                totals[key] += value.detach().item()*n
            totals['train_accuracy'] += (logits.argmax(1)==targets).sum().item()
            totals['teacher_accuracy'] += (teacher_logits.argmax(1)==targets).sum().item()
        score, rows, val_ce = validate(model, loader, args.device)
        scheduler.step(score)
        improved = score > best_score
        patience = 0 if improved else patience+1
        best_score = max(best_score, score)
        low = [r for r in rows if r['domain']==1]
        row = {'epoch':epoch+1, **{k:v/examples for k,v in totals.items()}, 'val_target_balanced_accuracy':score,
               'val_ce':val_ce,'val_all_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in rows])),
               'low_below70':sum(r['balanced_accuracy']<.7-1e-9 for r in low), 'grl':coefficient,
               'lr':optimizer.param_groups[0]['lr'], 'no_improvement':patience, 'seconds':time.monotonic()-started,
               'actual_student_steps':actual['student'],'actual_teacher_steps':actual['teacher'],'amp_scale':scaler.get_scale()}
        history.append(row)
        cp = {'framework':config['framework'],'alignment':alignment,'epoch':epoch,'model':model.state_dict(),
              'teacher':teacher.state_dict(),'optimizer':optimizer.state_dict(),'teacher_optimizer':teacher_optimizer.state_dict(),
              'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'history':history,'best_score':best_score,
              'patience':patience,'actual_steps':actual.copy()}
        checkpoint_save(output/'last.pt',cp)
        if improved:
            checkpoint_save(output/'best.pt',{'model':model.state_dict(),'epoch':epoch,'validation_score':score,
                                             'framework':config['framework'],'alignment':alignment,'seed':args.seed})
            write_json(output/'validation_subjects.json',rows)
        write_json(output/'history.json',history)
        print(f'[{alignment}/{args.seed}] epoch={epoch+1} CE={row["ce"]:.4f} train_acc={row["train_accuracy"]:.3f} '
              f'low_val={score:.4f} best={best_score:.4f} below70={row["low_below70"]} wait={patience}/{args.patience} '
              f'actual_steps={actual["student"]}/{actual["teacher"]} seconds={row["seconds"]:.1f}',flush=True)
    write_json(output/'completion.json',{'reason':'early_stop' if patience>=args.patience else 'max_epochs',
                                        'completed_epochs':len(history),'best_validation':best_score})
    print(f'DONE {alignment}: best={best_score:.4f}; {output}',flush=True)


def synthetic_data():
    rng = np.random.default_rng(123)
    labels = torch.tensor([0,1,0,1,0,1,0,1])
    domains = torch.tensor([0,0,1,1,0,0,1,1])
    data = TensorDataset(torch.from_numpy(rng.normal(size=(8,11,1000)).astype(np.float32)),labels,
                         torch.from_numpy(rng.normal(size=(8,6)).astype(np.float32)),domains,torch.where(domains==0,1,4))
    return data,data,{'data_fingerprint':'synthetic','groups':{1:{'training':0},4:{'training':1}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/mne_data/mne_data'))
    parser.add_argument('--groups-csv',type=Path,default=Path('/mne_data/mne_data/openbmi_bci_illiteracy_result.csv'))
    parser.add_argument('--cache-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/cache'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--alignments',nargs='+',choices=['global','classwise'],default=['global','classwise'])
    parser.add_argument('--subjects',type=int,nargs='+',default=list(range(1,55)))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--patience',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--lr',type=float,default=5e-4)
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--distill-alpha',type=float,default=.5)
    parser.add_argument('--temperature',type=float,default=3.)
    parser.add_argument('--variance-weight',type=float,default=1e-5)
    parser.add_argument('--domain-weight',type=float,default=.03)
    parser.add_argument('--grl-max',type=float,default=1.)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--synthetic',action='store_true',help='Smoke test only; not EEG evidence')
    args = parser.parse_args()
    if args.batch_size<4 or args.batch_size%4 or min(args.epochs,args.patience)<1 or args.workers<0:
        raise ValueError('Batch size divisible by 4 and positive epochs/patience required')
    if not 0<=args.distill_alpha<=1 or args.temperature<=0:
        raise ValueError('Invalid distillation parameters')
    numeric = [args.lr,args.weight_decay,args.temperature,args.domain_weight,args.grl_max,args.variance_weight]
    if not all(math.isfinite(v) for v in numeric) or min(numeric)<0 or args.lr<=0:
        raise ValueError('Invalid numeric parameters')
    if len(set(args.alignments))!=len(args.alignments) or len(set(args.subjects))!=len(args.subjects) or not args.subjects or not set(args.subjects)<=set(range(1,55)):
        raise ValueError('Invalid duplicate alignments or subjects')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    args.domain_source='training';args.group_seed=42;args.threshold=.70
    torch.set_num_threads(4)
    print('Preparing ALL requested subjects; no session2; balanced batch; alignments:',args.alignments,flush=True)
    if args.synthetic:
        train_data,val_data,prep=synthetic_data()
    else:
        import mne
        mne.set_log_level('ERROR')
        train_data,val_data,prep=prepare(args,'online')
    for alignment in args.alignments:
        train_one(args,alignment,train_data,val_data,prep)
    print('ALL DONE',flush=True)


if __name__=='__main__':main()
