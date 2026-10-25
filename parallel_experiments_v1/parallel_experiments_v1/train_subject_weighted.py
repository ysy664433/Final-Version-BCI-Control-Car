"""Training-only capped subject reweighting of baseline CE, original Plan A loop.

No DANN, gate, augmentation, personal fine tuning, or session2 usage.
This is an experimental weighting heuristic, NOT a novel/faithful group-DRO claim.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from src.plan_a import train as original


def weights_for_epoch(ema, args, preprocessing, epoch):
    weights=np.ones(len(ema),dtype=np.float32)
    mode=getattr(args,'reweight_mode','adaptive_low')
    if mode=='none' or epoch<args.reweight_warmup or args.reweight_cap==1:
        return weights
    low=[int(s) for s in args.subjects if (mode=='adaptive_all' or preprocessing['groups'][int(s)]['training']==1) and np.isfinite(ema[int(s)])]
    if not low: return weights
    if mode=='uniform_low':
        weights[low]=args.reweight_cap
        return weights
    center=float(np.median(ema[low]))
    # Fixed loss-gap scale prevents tiny early fluctuations receiving maximum weight.
    for s in low:
        weights[s]=1+(args.reweight_cap-1)*np.clip((ema[s]-center)/args.reweight_gap,0,1)
    return weights


def update_ema(ema,sums,counts,beta):
    result=ema.copy()
    for s in np.flatnonzero(counts):
        current=sums[s]/counts[s]
        result[s]=current if not np.isfinite(ema[s]) else beta*ema[s]+(1-beta)*current
    return result


def subject_ce(logits, targets, subjects, weights, sums, counts):
    raw=F.cross_entropy(logits,targets,reduction='none')
    batch_weights=weights[subjects.to(logits.device)]
    if torch.all(batch_weights==batch_weights[0]):
        # Preserve canonical mean CE when weighting has no relative effect.
        loss=F.cross_entropy(logits,targets)
    else:
        loss=(raw*batch_weights).sum()/batch_weights.sum()
    ids=subjects.cpu().numpy()
    np.add.at(sums,ids,raw.detach().float().cpu().numpy())
    np.add.at(counts,ids,1)
    return loss,raw.detach().sum().item()


def threshold_counts(rows):
    low=[r for r in rows if r['domain']==1]
    low_ok=sum(r['balanced_accuracy']>=.7-1e-9 for r in low)
    all_ok=sum(r['balanced_accuracy']>=.7-1e-9 for r in rows)
    return low_ok,all_ok


def replace_once(text,old,new):
    if text.count(old)!=1: raise RuntimeError('Unsupported source layout: '+old[:70])
    return text.replace(old,new,1)


def make_trainer():
    s=inspect.getsource(original.train_variant)
    s=replace_once(s,'    start_epoch, best_score, patience, history = 0, -1.0, 0, []',
        '    start_epoch, best_score, patience, history = 0, -1.0, 0, []\n'
        '    subject_ema = np.full(55, np.nan)\n    threshold_best = (-1, -1, -1.)')
    old='        best_score, patience, history = checkpoint["best_score"], checkpoint["patience"], checkpoint["history"]'
    s=replace_once(s,old,old+'\n        subject_ema = np.asarray(checkpoint["subject_loss_ema"], dtype=float)\n'
                   '        threshold_best = tuple(checkpoint["threshold_best"])')
    s=replace_once(s,'        skipped_domain_batches = 0',
        '        skipped_domain_batches = 0\n'
        '        epoch_weights = weights_for_epoch(subject_ema, args, preprocessing, epoch)\n'
        '        weight_tensor = torch.as_tensor(epoch_weights, device=device)\n'
        '        subject_sums, subject_counts = np.zeros(55), np.zeros(55, dtype=int)\n'
        '        raw_ce_sum = 0.')
    s=replace_once(s,'for batch_index, (eeg, targets, csp_features, domains, _) in enumerate(train_loader):',
                    'for batch_index, (eeg, targets, csp_features, domains, subjects) in enumerate(train_loader):')
    s=replace_once(s,'                ce = functional.cross_entropy(logits, targets)',
        '                ce, raw_ce = subject_ce(logits, targets, subjects, weight_tensor, subject_sums, subject_counts)\n'
        '                raw_ce_sum += raw_ce')
    s=replace_once(s,'+ args.distill_alpha * kd + args.domain_weight * domain_loss)',
        '+ args.distill_alpha * getattr(args, "kd_scale", 1.0) * kd + args.domain_weight * domain_loss)')
    s=replace_once(s,'        score, rows = validate(model, val_loader, device)',
        '        subject_ema = update_ema(subject_ema, subject_sums, subject_counts, args.reweight_ema)\n'
        '        score, rows = validate(model, val_loader, device)\n'
        '        low_ok, all_ok = threshold_counts(rows)\n'
        '        (output / "validation_epochs").mkdir(exist_ok=True)\n'
        '        write_json(output / "validation_epochs" / f"epoch_{epoch+1:03d}.json", rows)\n'
        '        threshold_key = (low_ok, all_ok, score)\n'
        '        if threshold_key > threshold_best:\n'
        '            threshold_best = threshold_key\n'
        '            checkpoint_save(output / "best_threshold.pt", {"model": model.state_dict(), "epoch": epoch, "threshold_key": threshold_key})\n'
        '            write_json(output / "validation_subjects_threshold.json", rows)')
    s=replace_once(s,'"skipped_domain_batches": skipped_domain_batches}',
        '"skipped_domain_batches": skipped_domain_batches, "ce_unweighted": raw_ce_sum / examples,\n'
        '                        "low_at_least70": low_ok, "all_at_least70": all_ok,\n'
        '                        "subject_weights": {str(s): float(epoch_weights[s]) for s in args.subjects}}')
    s=replace_once(s,'"epoch": epoch, "best_score": best_score, "patience": patience, "history": history,',
        '"epoch": epoch, "best_score": best_score, "patience": patience, "history": history,\n'
        '            "subject_loss_ema": subject_ema.tolist(), "threshold_best": threshold_best,')
    s=replace_once(s,'"format_version": 1, "mode": mode, "variant": variant, "seed": seed,',
        '"format_version": 1, "framework": "capped_subject_weighting_v1", "mode": mode, "variant": variant, "seed": seed,')
    s=replace_once(s,'f"val_target_bacc={score:.4f} best={best_score:.4f}"',
        'f"val_target_bacc={score:.4f} best={best_score:.4f} low_at70={low_ok} all_at70={all_ok} max_weight={epoch_weights.max():.3f}"')
    namespace=dict(vars(original))
    namespace['VARIANTS']={**original.VARIANTS,'subject_weighted':(False,False)}
    namespace.update(weights_for_epoch=weights_for_epoch,update_ema=update_ema,subject_ce=subject_ce,threshold_counts=threshold_counts)
    exec(compile(s,'<capped subject loss change to original baseline>','exec'),namespace)
    return namespace['train_variant']


def source_fingerprint():
    root=Path(original.__file__).parent
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in ('model.py','data.py','train.py'))).hexdigest()


def check():
    from types import SimpleNamespace
    args=SimpleNamespace(subjects=[1,2,4,6],reweight_warmup=5,reweight_cap=1.5,reweight_gap=.15)
    prep={'groups':{1:{'training':0},2:{'training':0},4:{'training':1},6:{'training':1}}}
    ema=np.full(55,np.nan);ema[[1,2,4,6]]=[2.,1.,.4,1.]
    assert np.all(weights_for_epoch(ema,args,prep,0)==1)
    w=weights_for_epoch(ema,args,prep,6)
    assert w[1]==w[2]==w[4]==1 and w[6]==1.5 and w.max()<=1.5
    z=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True);y=torch.tensor([0,1]);subjects=torch.tensor([1,4])
    sums=np.zeros(55);counts=np.zeros(55,dtype=int)
    loss,_=subject_ce(z,y,subjects,torch.ones(55),sums,counts)
    assert torch.equal(loss,F.cross_entropy(z,y)) and counts[1]==counts[4]==1
    generator=torch.Generator().manual_seed(123)
    labels=torch.tensor([0,1,0,1,0,1,0,1]);domains=torch.tensor([0,0,1,1,0,0,1,1])
    data=TensorDataset(torch.randn(8,11,1000,generator=generator),labels,torch.randn(8,6,generator=generator),domains,torch.where(domains==0,1,4))
    trainer=make_trainer()
    with tempfile.TemporaryDirectory(prefix='subject_weight_check_') as tmp:
        root=Path(tmp);csv=root/'groups.csv';csv.write_text('synthetic\n')
        cps=[]
        for name,fn in [('original',original.train_variant),('weighting_disabled',trainer)]:
            a=original.parser().parse_args([])
            a.subjects=[1,4];a.epochs=1;a.batch_size=4;a.workers=0;a.device='cpu';a.amp=False;a.resume=False
            a.output_root=root/name;a.groups_csv=csv;a.reweight_warmup=5;a.reweight_cap=1.;a.reweight_gap=.15;a.reweight_ema=.8
            variant='baseline' if name=='original' else 'subject_weighted'
            fn(a,'online',variant,42,data,data,{'data_fingerprint':'synthetic','groups':{1:{'training':0},4:{'training':1}}})
            p=a.output_root/'online'/variant/'seed_42'
            cps.append(torch.load(p/'last.pt',map_location='cpu',weights_only=False))
            if name=='weighting_disabled':
                for file in ['best_threshold.pt','validation_subjects_threshold.json','validation_epochs/epoch_001.json']:
                    assert (p/file).is_file()
        error=max((v-cps[1]['model'][k]).abs().max().item() for k,v in cps[0]['model'].items())
        assert error<=1e-6
        print('Disabled weighting maximum state difference:',error)
    print('CHECK PASS: caps, training-only weights, warmup, baseline compatibility and threshold records. Synthetic CPU only.')


def main():
    parser=original.parser()
    parser.set_defaults(variants=['baseline'],modes=['online'],seeds=[42],domain_source='training',batch_size=4)
    parser.add_argument('--reweight-cap',type=float,default=1.5)
    parser.add_argument('--reweight-warmup',type=int,default=5)
    parser.add_argument('--reweight-gap',type=float,default=.15)
    parser.add_argument('--reweight-ema',type=float,default=.8)
    parser.add_argument('--reweight-mode',choices=['adaptive_low','uniform_low','adaptive_all','none'],default='adaptive_low')
    parser.add_argument('--kd-scale',type=float,default=1.,help='0 disables student KD gradient; teacher retained for controlled RNG and diagnostics')
    parser.add_argument('--prepared-root',default='')
    parser.add_argument('--reference-run',default='/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.check: check();return
    if not 1<=args.reweight_cap<=2 or args.reweight_warmup<0 or args.reweight_gap<=0 or not 0<=args.reweight_ema<1:
        raise ValueError('Invalid weighting parameters')
    if not np.isfinite([args.reweight_cap,args.reweight_gap,args.reweight_ema,args.kd_scale]).all() or args.kd_scale<0: raise ValueError('Invalid weighting or KD scale')
    ref=json.loads((Path(args.reference_run)/'config.json').read_text())
    if source_fingerprint()!=ref['code_fingerprint']: raise ValueError('Original source differs from baseline reference')
    if args.variants!=['baseline'] or args.modes!=['online'] or args.seeds!=[42] or args.domain_source!='training' or args.subjects!=ref['subjects']:
        raise ValueError('Use online baseline, seed42, all recorded subjects, training groups')
    for key in ['epochs','patience','batch_size','workers','lr','weight_decay','distill_alpha','temperature','variance_weight','group_seed','threshold','amp']:
        if getattr(args,key)!=ref['hyperparameters'][key]: raise ValueError('Baseline setting differs: '+key)
    if hashlib.sha256(args.groups_csv.read_bytes()).hexdigest()!=ref['groups_csv_sha256']: raise ValueError('CSV differs')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    args.entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.selection_note='best.pt: original low-group mean criterion. best_threshold.pt: exploratory low count, all count, low mean. Do not compare threshold-selected candidate to mean-selected old baseline as a fair algorithm test.'
    import mne
    mne.set_log_level('ERROR')
    if args.prepared_root:
        from shared_data import load_prepared
        train_data,val_data,prep=load_prepared(args)
    else:
        train_data,val_data,prep=original.prepare(args,'online')
    if prep['data_fingerprint']!=ref['data_fingerprint']: raise ValueError('Data/groups differ')
    print('Capped subject CE weighting only. No gate, DANN, personal tuning, or session2.',flush=True)
    args.variants=['subject_weighted']
    make_trainer()(args,'online','subject_weighted',42,train_data,val_data,prep)
    print('DONE subject weighting',flush=True)


if __name__=='__main__':main()
