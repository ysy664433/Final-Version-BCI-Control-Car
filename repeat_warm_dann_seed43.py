"""Fixed 10-epoch same-start continuation control versus DANN-only. No session2."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import joblib
import torch
from torch.utils.data import DataLoader
from src.plan_a import train as original


def replace(source,old,new):
    if source.count(old)!=1:raise RuntimeError('Original training source differs: '+old[:60])
    return source.replace(old,new)


def initialize(model,teacher,checkpoint):
    missing,extra=model.load_state_dict(checkpoint['model'],strict=False)
    if extra or any(not key.startswith('domain.') for key in missing):raise ValueError('Unexpected model state mismatch')
    teacher.load_state_dict(checkpoint['teacher'])


def build(checkpoint):
    source=inspect.getsource(original.train_variant)
    source=replace(source,'    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)',
        '    initialize(model,teacher,INITIAL)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)')
    source=replace(source,'    for epoch in range(start_epoch, args.epochs):',
        '    initial_score,initial_rows=validate(model,val_loader,device)\n'
        '    best_score=initial_score\n'
        '    write_json(output/"initial_subjects.json",initial_rows)\n'
        '    write_json(output/"validation_subjects.json",initial_rows)\n'
        '    checkpoint_save(output/"best.pt",{"model":model.state_dict(),"epoch":-1,"validation_score":initial_score})\n'
        '    print(f"INITIAL {variant} low_mean={initial_score:.4f}",flush=True)\n'
        '    for epoch in range(start_epoch, args.epochs):')
    source=replace(source,'        score, rows = validate(model, val_loader, device)',
        '        score, rows = validate(model, val_loader, device)\n'
        '        low_ok=sum(r["domain"]==1 and r["balanced_accuracy"]>=.7-1e-9 for r in rows)\n'
        '        all_ok=sum(r["balanced_accuracy"]>=.7-1e-9 for r in rows)\n'
        '        (output/"validation_epochs").mkdir(exist_ok=True)\n'
        '        write_json(output/"validation_epochs"/f"epoch_{epoch+1:03d}.json",rows)\n'
        '        print(f"COUNTS {variant} low_at70={low_ok}/25 all_at70={all_ok}/54",flush=True)')
    namespace=dict(vars(original));namespace.update(initialize=initialize,INITIAL=checkpoint)
    exec(compile(source,'<same-start continuation>','exec'),namespace)
    return namespace['train_variant']


def summarize(root, seed):
    report={}
    for variant in ('baseline','dann'):
        directory=root/'online'/variant/f'seed_{seed}'
        rows=json.loads((directory/'validation_subjects.json').read_text())
        low=[r for r in rows if r['domain']==1]
        report[variant]={'low_mean':sum(r['balanced_accuracy'] for r in low)/len(low),
                         'low_at70':sum(r['balanced_accuracy']>=.7-1e-9 for r in low),
                         'all_at70':sum(r['balanced_accuracy']>=.7-1e-9 for r in rows)}
    report['dann_minus_control']={k:report['dann'][k]-report['baseline'][k] for k in report['baseline']}
    report['limits']='Exploratory validation result. Same matched last-checkpoint initialization, fresh optimizers for BOTH branches, fixed 10 epochs, student lr1e-4, teacher lr1e-3. No new data, no enhancement, distillation retained. Best includes epoch0. Compare also to old best baseline; one seed is not robustness evidence.'
    original.write_json(root/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)


def check():
    original.set_seed(42)
    base=original.EEGNetLight(n_channels=11,n_times=20,enhance=False,dann=False)
    teacher=original.ShallowConvNetTeacher(11)
    cp={'model':base.state_dict(),'teacher':teacher.state_dict()}
    original.set_seed(42)
    candidate=original.EEGNetLight(n_channels=11,n_times=20,enhance=False,dann=True)
    other_teacher=original.ShallowConvNetTeacher(11)
    initialize(candidate,other_teacher,cp)
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in base.state_dict().items())
    assert all(torch.equal(v,other_teacher.state_dict()[k]) for k,v in teacher.state_dict().items())
    x=torch.randn(4,11,20);c=torch.randn(4,6)
    base.eval();candidate.eval()
    assert torch.equal(base(x,c),candidate(x,c))
    for m in (base,candidate):m.train();m.zero_grad(set_to_none=True)
    original.set_seed(7);base(x,c,0.)[0].square().mean().backward()
    original.set_seed(7);out,domain,_=candidate(x,c,0.);(out.square().mean()+0*domain.square().mean()).backward()
    assert all(torch.equal(v.grad,dict(candidate.named_parameters())[k].grad) for k,v in base.named_parameters())
    build(cp)
    print('PASS: matched common student/teacher, identical eval logits, zero-domain common gradients, source patch. CPU synthetic only.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--seed',type=int,default=43,choices=[43])
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/warm_dann_seed43_001'))
    parser.add_argument('--reference',type=Path,default=Path('/mne_data/mne_data/bci_runs/dann_pair_43_44_001/online/baseline/seed_43'))
    parser.add_argument('--prepared',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    args=parser.parse_args()
    if args.check:check();return
    if args.summarize:summarize(args.output_root,args.seed);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    ref=json.loads((args.reference/'config.json').read_text())
    if ref['seed']!=args.seed:raise ValueError('Reference seed differs')
    files=Path(original.__file__).parent
    source_hash=hashlib.sha256(b''.join((files/n).read_bytes() for n in ('model.py','data.py','train.py'))).hexdigest()
    if source_hash!=ref['code_fingerprint']:raise ValueError('Original source changed')
    if ref['model']['enhance'] or ref['model']['dann'] or ref['domain_source']!='training':raise ValueError('Expected training-group original baseline')
    checkpoint=torch.load(args.reference/'last.pt',map_location='cpu',weights_only=False)
    prep=joblib.load(args.prepared/'preprocessing.joblib')
    if prep['data_fingerprint']!=ref['data_fingerprint']:raise ValueError('Prepared data differs')
    train=torch.load(args.prepared/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    val=torch.load(args.prepared/'validation.pt',map_location='cpu',weights_only=False,mmap=True)
    model=original.EEGNetLight(**ref['model']).cuda().eval();model.load_state_dict(checkpoint['model'])
    score,_=original.validate(model,DataLoader(val,batch_size=4),torch.device('cuda'))
    expected=checkpoint['history'][-1]['val_target_balanced_accuracy']
    if abs(score-expected)>1e-8:raise ValueError(f'Initial validation mismatch {score} versus {expected}; do not train')
    print(f'PREFLIGHT PASS: matched last checkpoint epoch={checkpoint["epoch"]+1}, low_mean={score:.4f}; this is NOT the old best checkpoint.',flush=True)
    del model
    if args.preflight:return
    if args.output_root.exists():raise FileExistsError('Choose a new output root; no automatic resume')
    options=original.parser().parse_args([])
    for k,v in ref['hyperparameters'].items():
        if hasattr(options,k):setattr(options,k,v)
    options.groups_csv=Path('/mne_data/mne_data/openbmi_bci_illiteracy_result.csv')
    if hashlib.sha256(options.groups_csv.read_bytes()).hexdigest()!=ref['groups_csv_sha256']:raise ValueError('CSV differs')
    options.subjects=ref['subjects'];options.output_root=args.output_root;options.device='cuda';options.resume=False
    options.epochs=10;options.patience=11;options.lr=1e-4;options.domain_weight=.03
    options.seeds=[args.seed];options.variants=['baseline','dann'];options.modes=['online']
    options.continuation_note='Same matched student+teacher from original baseline last.pt; fresh optimizers in both branches. Fixed10 epochs. Added DANN only, no enhancement. No session2. Mean-selected best includes initialization.'
    options.initial_checkpoint_epoch=checkpoint['epoch']+1
    options.evidence_note='Fixed seed43 replication of seed42 late DANN recipe. Seed44 pretrained baseline failed; not silently excluded from overall training-stability reporting. Session2 already exposed, follow-up reproducibility evidence not untouched confirmatory test.'
    options.entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run=build(checkpoint)
    for variant in ('baseline','dann'):
        run(options,'online',variant,args.seed,train,val,prep)
    summarize(args.output_root,args.seed)
    print('DONE',args.output_root,flush=True)


if __name__=='__main__':main()
