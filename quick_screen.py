"""Small-budget, from-scratch algorithm screen. Run from the repository root.

No subject fine tuning, no session2, no original source changes.
Classwise domain heads use TRAIN labels: supervised alignment, not CDAN reproduction.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path.cwd()))
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.plan_a.data import cached_trials
from src.plan_a.model import EEGNetLight, GradientReverseLayer, ShallowConvNetTeacher


VARIANTS = ('baseline', 'gate', 'gate_global', 'gate_classwise')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Candidate(nn.Module):
    def __init__(self, variant, common_state, seed):
        super().__init__()
        self.variant = variant
        self.backbone = EEGNetLight(11, dropout=.1, enhance=variant != 'baseline', dann=False, causal=True)
        missing, extra = self.backbone.load_state_dict(common_state, strict=False)
        assert not extra and all(k.startswith('enhance.') for k in missing)
        # Same initial discriminator for global and both classwise heads.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 2)
            head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 2))
        self.domains = nn.ModuleList([copy.deepcopy(head) for _ in range(2 if variant == 'gate_classwise' else 1)]) if 'gate_' in variant else nn.ModuleList()

    def forward(self, x, csp, labels=None, domains=None, coefficient=0.):
        logits, _, features = self.backbone(x, csp, grl_coefficient=0.)
        auxiliary = logits.new_zeros(())
        if labels is not None and len(self.domains):
            reverse = GradientReverseLayer.apply(features, coefficient)
            if self.variant == 'gate_global':
                auxiliary = F.cross_entropy(self.domains[0](reverse), domains)
            else:
                parts = []
                for label in (0, 1):
                    selected = labels == label
                    assert set(domains[selected].tolist()) == {0, 1}
                    parts.append(F.cross_entropy(self.domains[label](reverse[selected]), domains[selected]))
                auxiliary = torch.stack(parts).mean()
        return logits, auxiliary


def balanced_indices(y, d, batch_size, rng):
    parts = [rng.choice(np.flatnonzero((d == domain) & (y == label)), batch_size // 4, replace=True)
             for domain in (0, 1) for label in (0, 1)]
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices


def take_per_class(y, count, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == label), min(count, int((y == label).sum())), replace=False)
                           for label in (0, 1)])


def prepare(args):
    run = args.base / 'formal_a_001/online/baseline/seed_42'
    prep = joblib.load(run / 'preprocessing.joblib')
    cfg = json.loads((run / 'config.json').read_text())
    assert cfg['domain_source'] == 'training' and cfg['mode'] == 'online'
    subjects = []
    for domain in (0, 1):
        candidates = [s for s in sorted(cfg['subjects']) if prep['groups'][s]['training'] == domain]
        if len(candidates) < args.subjects_per_domain:
            raise ValueError('Not enough subjects in both training-defined groups')
        subjects += candidates[:args.subjects_per_domain]
    splits = {}
    manifest = []
    for split, block in [('train', 'train'), ('validation', 'test')]:
        xs, cs, ys, ds, ss = [], [], [], [], []
        for s in subjects:
            raw, labels = cached_trials(args.data_root, args.base / 'cache', s, 1, block, 'online')
            if split == 'validation':
                ix = take_per_class(labels, args.val_per_class, np.random.default_rng(args.seed + s + 7000))
                raw, labels = raw[ix], labels[ix]
                manifest.append({'subject': s, 'validation_trial_indices': ix.tolist()})
            csp = prep['csp_scaler'].transform(prep['csp'].transform(raw.astype(np.float64))).astype(np.float32)
            xs.append(((raw - prep['mean']) / prep['scale']).astype(np.float32))
            cs.append(csp)
            ys.extend(labels.tolist())
            ds.extend([prep['groups'][s]['training']] * len(labels))
            ss.extend([s] * len(labels))
            print(f'prepare {split}: subject {s}', flush=True)
        splits[split] = (np.concatenate(xs), np.concatenate(cs), np.array(ys), np.array(ds), np.array(ss))
    metadata = {'subjects': subjects, 'selection': 'first IDs per training-defined domain; no validation ranking',
                'validation_manifest': manifest, 'source_data_fingerprint': cfg['data_fingerprint'],
                'preprocessing': 'reuse fixed CSP/scalers fitted on all session1 training subjects; subset screening only',
                'preprocessing_sha256': hashlib.sha256((run / 'preprocessing.joblib').read_bytes()).hexdigest()}
    return splits, metadata


def synthetic():
    rng = np.random.default_rng(123)
    splits = {}
    for split in ('train', 'validation'):
        y = np.tile([0, 1], 8)
        d = np.repeat([0, 1], 8)
        s = np.where(d == 0, 1, 4)
        splits[split] = (rng.normal(size=(16, 11, 1000)).astype(np.float32),
                         rng.normal(size=(16, 6)).astype(np.float32), y, d, s)
    return splits, {'subjects': [1, 4], 'synthetic': True}


@torch.inference_mode()
def evaluate(model, data, device, batch_size):
    model.eval()
    x, c, y, d, subjects = data
    preds, losses = [], []
    for start in range(0, len(y), batch_size):
        sl = slice(start, start + batch_size)
        logits, _ = model(torch.tensor(x[sl], device=device), torch.tensor(c[sl], device=device))
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite validation logits')
        preds.extend(logits.argmax(1).cpu().tolist())
        losses.extend(F.cross_entropy(logits.float(), torch.tensor(y[sl], device=device), reduction='none').cpu().tolist())
    preds, losses = np.array(preds), np.array(losses)
    rows = []
    for s in np.unique(subjects):
        selected = subjects == s
        recalls = [float((preds[selected & (y == label)] == label).mean()) for label in (0, 1)]
        rows.append({'subject': int(s), 'domain': int(d[selected][0]), 'balanced_accuracy': float(np.mean(recalls)),
                     'ce': float(losses[selected].mean()), 'predicted_right_fraction': float((preds[selected] == 1).mean())})
    low = [r for r in rows if r['domain'] == 1]
    return {'val_low_bacc': float(np.mean([r['balanced_accuracy'] for r in low])),
            'val_all_bacc': float(np.mean([r['balanced_accuracy'] for r in rows])),
            'val_ce': float(losses.mean()), 'low_below70': sum(r['balanced_accuracy'] < .7 - 1e-9 for r in low),
            'subjects': rows}


def train_one(args, variant, splits, common):
    out = args.output / variant
    out.mkdir()
    seed_all(args.seed)
    model = Candidate(variant, common, args.seed).to(args.device)
    seed_all(args.seed + 1)
    teacher = ShallowConvNetTeacher(11).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-3)
    actual_steps = {'student': 0, 'teacher': 0}
    def count_student(*unused):
        actual_steps['student'] += 1
    def count_teacher(*unused):
        actual_steps['teacher'] += 1
    optimizer.register_step_post_hook(count_student)
    teacher_optimizer.register_step_post_hook(count_teacher)
    amp = args.amp and args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    x, c, y, d, _ = splits['train']
    history, best = [], -1.
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        teacher.train()
        ramp = min(1., max(0., (epoch - args.warmup + 1) / args.ramp))
        totals = {key: 0. for key in ['ce', 'kd', 'aux', 'teacher_ce', 'train_accuracy']}
        for step in range(args.steps):
            # Explicit seeds keep batch sampling and dropout comparable across candidates.
            seed_all(args.seed + epoch * 1009 + step)
            rng = np.random.default_rng(args.seed + epoch * 1009 + step)
            ix = balanced_indices(y, d, args.batch_size, rng)
            eeg, features = torch.tensor(x[ix], device=args.device), torch.tensor(c[ix], device=args.device)
            labels, domains = torch.tensor(y[ix], device=args.device), torch.tensor(d[ix], device=args.device)
            teacher_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, enabled=amp):
                teacher_logits = teacher(eeg)
                teacher_ce = F.cross_entropy(teacher_logits, labels)
            scaler.scale(teacher_ce).backward()
            scaler.unscale_(teacher_optimizer)
            nn.utils.clip_grad_norm_(teacher.parameters(), 5.)
            scaler.step(teacher_optimizer)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, enabled=amp):
                logits, auxiliary = model(eeg, features, labels, domains, ramp)
                ce = F.cross_entropy(logits, labels)
                kd = F.kl_div(F.log_softmax(logits.float()/3, 1), F.softmax(teacher_logits.detach().float()/3, 1), reduction='batchmean') * 9
                variance = logits.float().softmax(1).var(dim=1, unbiased=False).mean()
                loss = .5 * (ce + 1e-5 * variance) + .5 * kd + args.domain_weight * auxiliary
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            for key, value in [('ce', ce), ('kd', kd), ('aux', auxiliary), ('teacher_ce', teacher_ce)]:
                totals[key] += float(value.detach()) / args.steps
            totals['train_accuracy'] += (logits.argmax(1) == labels).float().mean().item() / args.steps
        metrics = evaluate(model, splits['validation'], args.device, args.batch_size)
        row = {'epoch': epoch + 1, 'updates': (epoch + 1) * args.steps, 'grl': ramp,
               'actual_student_steps': actual_steps['student'], 'actual_teacher_steps': actual_steps['teacher'],
               'amp_scale': scaler.get_scale(),
               'seconds': time.monotonic()-started, **totals, **metrics}
        history.append(row)
        checkpoint = {'model': model.state_dict(), 'teacher': teacher.state_dict(), 'epoch': epoch + 1,
                      'variant': variant, 'screen_only': True}
        if metrics['val_low_bacc'] > best:
            best = metrics['val_low_bacc']
            torch.save(checkpoint, out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'{variant}: {epoch+1}/{args.epochs} updates={row["updates"]} CE={totals["ce"]:.4f} '
              f'train_acc={totals["train_accuracy"]:.3f} low_val={metrics["val_low_bacc"]:.3f} '
              f'below70={metrics["low_below70"]} actual_steps={actual_steps["student"]}/{actual_steps["teacher"]} '
              f'scale={scaler.get_scale():.1f} seconds={row["seconds"]:.1f}', flush=True)
    torch.save(checkpoint, out / 'last.pt')
    (out / 'train_clean_final.json').write_text(json.dumps(evaluate(model, splits['train'], args.device, args.batch_size), indent=2))
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('/mne_data/mne_data/bci_runs'))
    parser.add_argument('--data-root', type=Path, default=Path('/mne_data/mne_data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--subjects-per-domain', type=int, default=4)
    parser.add_argument('--val-per-class', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--ramp', type=int, default=8)
    parser.add_argument('--domain-weight', type=float, default=.03)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--synthetic', action='store_true', help='Installation smoke test; NOT EEG evidence')
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.subjects_per_domain, args.val_per_class, args.ramp) < 1 or args.batch_size < 4 or args.batch_size % 4:
        raise ValueError('Positive counts and batch size divisible by 4 required')
    if args.warmup < 0 or not math.isfinite(args.domain_weight) or args.domain_weight < 0 or len(set(args.variants)) != len(args.variants):
        raise ValueError('Invalid schedule/weight/variants')
    if args.output.exists():
        raise FileExistsError('Choose a NEW output directory; screening does not resume')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
                   'notes': 'From scratch; shared training-only preprocessing; fixed balanced batches; no augmentation. '
                            'Both domain variants use equal delayed GRL ramp. Two classwise heads vs one global head; '
                            'not a faithful CDAN reproduction or existing full-run reproduction. No early stopping or LR scheduler. '
                            'No session2. Small budget negative results are inconclusive if the control fails to learn.'})
    splits, metadata = synthetic() if args.synthetic else prepare(args)
    config['data'] = metadata
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    seed_all(args.seed)
    common = EEGNetLight(11, dropout=.1, enhance=False, dann=False, causal=True).state_dict()
    histories = {variant: train_one(args, variant, splits, common) for variant in args.variants}
    summaries = []
    for variant, history in histories.items():
        # Rank fixed-budget trailing means, not a cherry-picked peak.
        tail = history[-min(3, len(history)):]
        summaries.append({'variant': variant, 'updates': history[-1]['updates'],
                          'actual_student_steps': history[-1]['actual_student_steps'],
                          'actual_teacher_steps': history[-1]['actual_teacher_steps'],
                          'amp_scale_last': history[-1]['amp_scale'],
                          'seconds': history[-1]['seconds'], 'train_ce_last': history[-1]['ce'],
                          'train_accuracy_last': history[-1]['train_accuracy'],
                          'low_val_last3_mean': float(np.mean([r['val_low_bacc'] for r in tail])),
                          'low_val_best': max(r['val_low_bacc'] for r in history),
                          'low_below70_last': history[-1]['low_below70']})
    reference = next((r for r in summaries if r['variant'] == 'baseline'), None)
    control_learned = reference is not None and reference['low_val_last3_mean'] >= .6 and reference['train_accuracy_last'] >= .65
    report = {'status': 'SYNTHETIC ONLY' if args.synthetic else ('EXPLORATORY SCREEN ONLY' if control_learned else 'INCONCLUSIVE: control did not meet heuristic learning checks'),
              'heuristic_note': '0.60 low validation and 0.65 stochastic train accuracy are triage thresholds, not statistical tests.',
              'results': summaries}
    if reference:
        for row in summaries:
            row['delta_to_screen_baseline_last3'] = row['low_val_last3_mean'] - reference['low_val_last3_mean']
            if row['variant'] != 'baseline':
                base_rows = {r['subject']: r for r in histories['baseline'][-1]['subjects'] if r['domain'] == 1}
                ds = [r['balanced_accuracy'] - base_rows[r['subject']]['balanced_accuracy'] for r in histories[row['variant']][-1]['subjects'] if r['domain'] == 1]
                row['low_improved_tied_worse_last'] = [sum(d > 1e-9 for d in ds), sum(abs(d) <= 1e-9 for d in ds), sum(d < -1e-9 for d in ds)]
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
