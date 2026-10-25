"""Small set of full-data recipes, using original loops and samplers; no src edits."""
import argparse, hashlib, inspect, json, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from train_subject_weighted import replace_once, source_fingerprint


def protected_backward(scaler, anchor, other, parameters):
    parameters=list(parameters)
    primary=torch.autograd.grad(scaler.scale(anchor),parameters,retain_graph=True,allow_unused=True)
    auxiliary=torch.autograd.grad(scaler.scale(other),parameters,allow_unused=True)
    pairs=[(a,b) for a,b in zip(primary,auxiliary) if a is not None and b is not None]
    zero=torch.zeros((),device=parameters[0].device,dtype=torch.float64)
    dot=sum(((a.double()*b.double()).sum() for a,b in pairs),zero)
    norm=sum((a.double().square().sum() for a in primary if a is not None),zero)
    conflict=bool(torch.isfinite(dot) and torch.isfinite(norm) and norm>0 and dot<0)
    ratio=(dot/norm).float() if conflict else 0.
    for p,a,b in zip(parameters,primary,auxiliary):
        if a is None and b is None: p.grad=None
        elif a is None: p.grad=b
        elif b is None: p.grad=(1-ratio)*a
        else: p.grad=a+b-ratio*a
    return conflict


def recorded_trainer(module, recipe):
    source=inspect.getsource(module.train_variant)
    if recipe=='gradient_protected':
        source=replace_once(source,'        examples = 0','        examples = 0\n        projected_batches = 0')
        source=replace_once(source,'            scaler.scale(loss).backward()',
            '            projected_batches += int(protected_backward(scaler, (1-args.distill_alpha)*ce, '
            '(1-args.distill_alpha)*args.variance_weight*variance + args.distill_alpha*kd + args.domain_weight*domain_loss, model.parameters()))')
    if recipe=='delayed_dann':
        source=replace_once(source,
            'progress = (epoch + batch_index / len(train_loader)) / max(1, args.epochs - 1)',
            'progress = min(1.0, max(0.0, (epoch + batch_index / len(train_loader) - args.domain_warmup) / args.domain_ramp))')
    source=replace_once(source,'    start_epoch, best_score, patience, history = 0, -1.0, 0, []',
        '    start_epoch, best_score, patience, history = 0, -1.0, 0, []\n    threshold_best = (-1, -1, -1.)')
    source=replace_once(source,'        score, rows = validate(model, val_loader, device)',
        '        score, rows = validate(model, val_loader, device)\n'
        '        low_ok = sum(r["balanced_accuracy"] >= .7-1e-9 for r in rows if r["domain"] == 1)\n'
        '        all_ok = sum(r["balanced_accuracy"] >= .7-1e-9 for r in rows)\n'
        '        (output / "validation_epochs").mkdir(exist_ok=True)\n'
        '        write_json(output / "validation_epochs" / f"epoch_{epoch+1:03d}.json", rows)\n'
        '        if (low_ok, all_ok, score) > threshold_best:\n'
        '            threshold_best = (low_ok, all_ok, score)\n'
        '            checkpoint_save(output / "best_threshold.pt", {"model": model.state_dict(), "epoch": epoch, "threshold_key": threshold_best})\n'
        '            write_json(output / "validation_subjects_threshold.json", rows)')
    source=replace_once(source,'"val_target_balanced_accuracy": score,',
        '"val_target_balanced_accuracy": score, "low_at_least70": low_ok, "all_at_least70": all_ok,'
        + (' "projected_batches": projected_batches,' if recipe=='gradient_protected' else ''))
    namespace=dict(vars(module))
    namespace['protected_backward']=protected_backward
    exec(compile(source,'<recorded '+recipe+'>','exec'),namespace)
    return namespace['train_variant']


def main():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--recipe',required=True,choices=['delayed_dann','gradient_protected','weak_contrast','weak_augment'])
    recipe=pre.parse_known_args()[0].recipe
    if recipe in ['delayed_dann','gradient_protected']:
        from src.plan_a import train as module
        variant='plan_a'
    else:
        from src.plan_b import train as module
        variant='contrast' if recipe=='weak_contrast' else 'augment'
    parser=module.parser()
    parser.add_argument('--recipe',required=True,choices=['delayed_dann','gradient_protected','weak_contrast','weak_augment'])
    parser.add_argument('--domain-warmup',type=int,default=10)
    parser.add_argument('--domain-ramp',type=int,default=40)
    parser.add_argument('--reference-run',default='/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42')
    parser.add_argument('--prepared-root',default='')
    parser.set_defaults(variants=[variant],modes=['online'],seeds=[42],domain_source='training',batch_size=4)
    if recipe in ['delayed_dann','gradient_protected']: parser.set_defaults(domain_weight=.03)
    elif recipe=='weak_contrast': parser.set_defaults(contrastive_weight=.01)
    else: parser.set_defaults(band_strength=.1,channel_mask=0.,noise_std=.01)
    args=parser.parse_args()
    if args.resume: raise ValueError('Batch recipes do not resume. Preserve partial results and use a new run root.')
    if args.domain_ramp<1 or args.domain_warmup<0: raise ValueError('Invalid GRL schedule')
    ref=json.loads((Path(args.reference_run)/'config.json').read_text())
    if source_fingerprint()!=ref['code_fingerprint']: raise ValueError('Original A source differs from reference')
    if args.variants!=[variant] or args.modes!=['online'] or args.seeds!=[42] or args.domain_source!='training' or args.subjects!=ref['subjects']:
        raise ValueError('Use the fixed recipe, online, seed42, all subjects and training groups')
    for key in ['epochs','patience','batch_size','workers','lr','weight_decay','distill_alpha','temperature','variance_weight','group_seed','threshold','amp']:
        if getattr(args,key)!=ref['hyperparameters'][key]: raise ValueError('Common reference setting differs: '+key)
    if hashlib.sha256(args.groups_csv.read_bytes()).hexdigest()!=ref['groups_csv_sha256']: raise ValueError('CSV differs')
    args.entry_sha256=hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'train_subject_weighted.py').read_bytes()).hexdigest()
    args.selection_note='Primary best.pt uses original low-group mean. best_threshold.pt is exploratory until the control uses the same selection rule.'
    torch.set_num_threads(4)
    import mne
    mne.set_log_level('ERROR')
    if args.prepared_root:
        from shared_data import load_prepared
        train_data,val_data,prep=load_prepared(args)
    else:
        train_data,val_data,prep=module.prepare(args,'online')
    if prep['data_fingerprint']!=ref['data_fingerprint']: raise ValueError('Training data/groups differ')
    recorded_trainer(module,recipe)(args,'online',variant,42,train_data,val_data,prep)
    print('DONE',recipe,flush=True)


if __name__=='__main__':main()
