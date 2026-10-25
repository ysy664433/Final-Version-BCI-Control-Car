"""Classwise domain-loss candidate using the ORIGINAL Plan A loop and sampler.

Run from repository root. Does not modify src. --check uses synthetic data only.
Normal training requires the recorded 0.03 run's original source fingerprint.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from src.plan_a import train as original
from src.plan_a.model import EEGNetLight, GradientReverseLayer


class ClasswiseEEGNet(EEGNetLight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.domain is None:
            raise ValueError('Classwise candidate requires the plan_a variant')
        # deepcopy consumes no RNG; original .domain is the left-hand head.
        self.right_domain = copy.deepcopy(self.domain)

    def forward(self, eeg, csp_features, grl_coefficient=None):
        result = super().forward(eeg, csp_features, grl_coefficient)
        if grl_coefficient is None:
            return result
        logits, left_logits, features = result
        device_ids = [features.device.index if features.device.index is not None else torch.cuda.current_device()] if features.is_cuda else []
        # Preserve the original RNG consumption from the first domain head.
        # The added head must not shift later teacher/student dropout draws.
        with torch.random.fork_rng(devices=device_ids):
            right_logits = self.right_domain(GradientReverseLayer.apply(features, grl_coefficient))
        return logits, torch.stack([left_logits, right_logits], dim=1), features


def classwise_loss(logits, domains, labels):
    parts = []
    for label in (0, 1):
        selected = labels == label
        if set(domains[selected].tolist()) != {0, 1}:
            continue
        parts.append(.5 * sum(F.cross_entropy(logits[selected & (domains == domain), label],
                                               domains[selected & (domains == domain)]) for domain in (0, 1)))
    if not parts:
        return logits.sum() * 0., 0
    return torch.stack(parts).mean(), len(parts)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Original source layout differs; refuse to guess a patch: ' + old[:80])
    return text.replace(old, new, 1)


def classwise_trainer():
    source = inspect.getsource(original.train_variant)
    old = '''                if domain_logits is not None:
                    if domains.unique().numel() == 2:
                        domain_loss = 0.5 * sum(
                            functional.cross_entropy(domain_logits[domains == domain], domains[domains == domain])
                            for domain in (0, 1)
                        )
                    else:
                        skipped_domain_batches += 1'''
    new = '''                if domain_logits is not None:
                    domain_loss, active_classes = classwise_loss(domain_logits, domains, targets)
                    active_class_batches += active_classes
                    if active_classes == 0:
                        skipped_domain_batches += 1'''
    source = replace_once(source, old, new)
    source = replace_once(source, '        skipped_domain_batches = 0',
                          '        skipped_domain_batches = 0\n        active_class_batches = 0')
    source = replace_once(source, '"skipped_domain_batches": skipped_domain_batches}',
                          '"skipped_domain_batches": skipped_domain_batches,\n'
                          '                        "active_class_batches": active_class_batches, '
                          '"possible_class_batches": 2 * len(train_loader)}')
    source = replace_once(source, '"format_version": 1, "mode": mode, "variant": variant, "seed": seed,',
                          '"format_version": 1, "framework": "classwise_original_v1", "mode": mode, "variant": variant, "seed": seed,')
    # All imports, prepare, validate, optimizer, scheduler, sampler, AMP and checkpoint logic
    # remain from original. __file__ deliberately remains original.train.__file__.
    namespace = dict(vars(original))
    namespace.update(EEGNetLight=ClasswiseEEGNet, classwise_loss=classwise_loss)
    exec(compile(source, '<classwise change to original.train_variant>', 'exec'), namespace)
    return namespace['train_variant']


def original_fingerprint():
    root = Path(original.__file__).parent
    return hashlib.sha256(b''.join((root / name).read_bytes() for name in ('model.py','data.py','train.py'))).hexdigest()


def check(device, amp):
    # Ordinary/global path IS the original callable, not a rewritten loop.
    assert original.train_variant is vars(original)['train_variant']
    original.set_seed(42)
    ordinary = EEGNetLight(11, dropout=.1, enhance=True, dann=True, causal=True)
    original.set_seed(42)
    conditional = ClasswiseEEGNet(11, dropout=.1, enhance=True, dann=True, causal=True)
    assert all(torch.equal(v, conditional.state_dict()[k]) for k,v in ordinary.state_dict().items())
    labels = torch.tensor([0,1,0,1,0,1,0,1])
    domains = torch.tensor([0,0,1,1,0,0,1,1])
    z = torch.randn(8,2,2,requires_grad=True)
    loss, active = classwise_loss(z, domains, labels)
    loss.backward()
    assert active == 2
    for i,label in enumerate(labels):
        assert z.grad[i,1-label].abs().sum() == 0
    empty, active = classwise_loss(z, torch.zeros_like(domains), labels)
    assert active == 0 and empty.item() == 0
    print('PASS: common initial weights; label-head routing; missing-domain handling',flush=True)
    generator = torch.Generator().manual_seed(123)
    data = TensorDataset(torch.randn(8,11,1000,generator=generator), labels,
                         torch.randn(8,6,generator=generator), domains, torch.where(domains==0,1,4))
    trainer = classwise_trainer()
    with tempfile.TemporaryDirectory(prefix='classwise_check_') as tmp:
        root = Path(tmp)
        csv = root/'groups.csv';csv.write_text('synthetic\n')
        results = []
        for name, function in [('original',original.train_variant),('original_repeat',original.train_variant),('classwise_zero_aux',trainer)]:
            args = original.parser().parse_args([])
            args.subjects=[1,4];args.seeds=[42];args.variants=['plan_a'];args.modes=['online']
            args.epochs=1;args.batch_size=4;args.workers=0;args.device=device;args.amp=amp
            args.output_root=root/name;args.groups_csv=csv;args.domain_weight=0.;args.resume=False
            function(args,'online','plan_a',42,data,data,{'data_fingerprint':'synthetic'})
            cp=torch.load(args.output_root/'online/plan_a/seed_42/last.pt',map_location='cpu',weights_only=False)
            results.append(cp)
        reference_model=EEGNetLight(11,dropout=.1,enhance=True,dann=True,causal=True).eval()
        reference_model.load_state_dict(results[0]['model'])
        parameter_names=set(dict(reference_model.named_parameters()))
        with torch.no_grad():
            reference_prob=reference_model(data.tensors[0],data.tensors[2]).softmax(1)
        report=[]
        for index,name in [(1,'original_repeat'),(2,'classwise_zero_aux')]:
            max_error=0.;numerator=0.;denominator=0.
            for key,value in results[0]['model'].items():
                diff=value.double()-results[index]['model'][key].double()
                max_error=max(max_error,diff.abs().max().item())
                if key in parameter_names:
                    numerator+=diff.square().sum().item()
                    denominator+=value.double().square().sum().item()
            candidate=(ClasswiseEEGNet if index==2 else EEGNetLight)(11,dropout=.1,enhance=True,dann=True,causal=True).eval()
            candidate.load_state_dict(results[index]['model'])
            with torch.no_grad():
                probabilities=candidate(data.tensors[0],data.tensors[2]).softmax(1)
            teacher_equal=all(torch.equal(v,results[index]['teacher'][k]) for k,v in results[0]['teacher'].items())
            row={'comparison':name,'common_state_max_abs_error':max_error,
                 'parameter_relative_l2_error':math.sqrt(numerator/max(denominator,1e-30)),
                 'fp32_eval_probability_max_abs_error':(probabilities-reference_prob).abs().max().item(),
                 'teacher_exactly_equal':teacher_equal}
            report.append(row)
            print(json.dumps(row,indent=2),flush=True)
        strict=all(r['common_state_max_abs_error']<=1e-6 and r['teacher_exactly_equal'] for r in report)
        # Do not automatically accept a looser threshold after an AMP mismatch.
        if not strict:
            print('CHECK NUMERICAL MISMATCH: routing/initialization passed, strict update equality did not. '
                  'No acceptance threshold was relaxed. Compare the FP32 check and this diagnostic report before long training.',flush=True)
            raise SystemExit(2)
    print('CHECK PASS. Synthetic one-epoch mechanism check only, not evidence of EEG benefit.',flush=True)


def main():
    parser=original.parser()
    parser.set_defaults(domain_weight=.03,seeds=[42],variants=['plan_a'],modes=['online'])
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--alignment',choices=['global','classwise'],default='classwise')
    parser.add_argument('--reference-run',default='/mne_data/mne_data/bci_runs/plan_a_domain003_001/online/plan_a/seed_42')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.check:
        check(args.device,args.amp)
        return
    reference=json.loads((Path(args.reference_run)/'config.json').read_text())
    if original_fingerprint()!=reference['code_fingerprint']:
        raise RuntimeError('Current original Plan A sources differ from the recorded 70.40% run. No training started.')
    if args.variants!=['plan_a'] or args.modes!=['online'] or args.seeds!=[42]:
        raise ValueError('This controlled experiment is online plan_a, seed 42 only')
    expected={'epochs':120,'patience':30,'batch_size':4,'workers':0,'lr':.0005,'weight_decay':.01,
              'distill_alpha':.5,'temperature':3.,'variance_weight':1e-5,'domain_weight':.03,'grl_max':1.}
    for key,value in expected.items():
        if getattr(args,key)!=value or reference['hyperparameters'][key]!=value:
            raise ValueError(f'Expected recorded common setting {key}={value}')
    if args.domain_source!='training' or args.subjects!=reference['subjects']:
        raise ValueError('Use the same training-defined groups and all recorded subjects')
    if hashlib.sha256(args.groups_csv.read_bytes()).hexdigest()!=reference['groups_csv_sha256']:
        raise ValueError('Grouping CSV differs')
    if args.amp!=reference['hyperparameters']['amp']:
        raise ValueError('AMP must match the reference')
    current_env={'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda}
    print('Environment matches recorded reference:',current_env==reference['environment'],flush=True)
    if current_env!=reference['environment']:
        print('WARNING: environment changed; this is a new comparison condition, not an exact reproduction.',flush=True)
    args.entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.sampler_protocol='original WeightedRandomSampler; no batch balancing'
    import mne
    mne.set_log_level('ERROR')
    train_data,val_data,prep=original.prepare(args,'online')
    if prep['data_fingerprint']!=reference['data_fingerprint']:
        raise ValueError('Prepared training data/groups differ from reference')
    runner=original.train_variant if args.alignment=='global' else classwise_trainer()
    print('Training',args.alignment,'with original sampler/optimizer/scheduler; no session2.',flush=True)
    runner(args,'online','plan_a',42,train_data,val_data,prep)
    print('DONE',args.alignment,flush=True)


if __name__=='__main__':main()
