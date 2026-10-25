"""Sequential seed43/44 paired replications of recorded baseline and PlanA .03."""
import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))


def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def worker(args):
    import joblib
    import torch
    from src.plan_a import train as module
    root=Path(module.__file__).parent
    fingerprint=hashlib.sha256(b''.join((root/name).read_bytes() for name in ('model.py','data.py','train.py'))).hexdigest()
    refs={v:json.loads((p/'config.json').read_text()) for v,p in [('baseline',args.baseline),('plan_a',args.candidate)]}
    common=['epochs','patience','batch_size','workers','lr','weight_decay','distill_alpha','temperature','variance_weight','group_seed','threshold','amp','grl_max']
    for key in common:
        if refs['baseline']['hyperparameters'][key]!=refs['plan_a']['hyperparameters'][key]:raise ValueError('Reference common setting differs: '+key)
    for ref in refs.values():
        if fingerprint!=ref['code_fingerprint']:raise ValueError('Original source changed')
        if ref['domain_source']!='training' or ref['subjects']!=list(range(1,55)) or ref['mode']!='online':raise ValueError('Unexpected reference protocol')
        if hashlib.sha256(args.csv.read_bytes()).hexdigest()!=ref['groups_csv_sha256']:raise ValueError('CSV changed')
    if refs['plan_a']['hyperparameters']['domain_weight']!=.03:raise ValueError('Expected candidate domain_weight .03')
    prep=joblib.load(args.prepared/'preprocessing.joblib')
    metadata=json.loads((args.prepared/'metadata.json').read_text())
    if metadata['source_fingerprint']!=fingerprint:raise ValueError('Cache source mismatch')
    if any(prep['data_fingerprint']!=r['data_fingerprint'] for r in refs.values()):raise ValueError('Data fingerprint mismatch')
    ref=refs[args.variant]
    options=module.parser().parse_args([])
    for key,value in ref['hyperparameters'].items():
        if hasattr(options,key):setattr(options,key,value)
    options.output_root=args.run_root;options.groups_csv=args.csv;options.device='cuda';options.resume=False
    options.seeds=[args.seed];options.variants=[args.variant];options.modes=['online'];options.subjects=ref['subjects']
    options.replication_entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    options.replication_note='Frozen recorded recipe; only training seed changes. group_seed remains42. Primary original mean-selected checkpoint. Shared preprocessing/cache for all four jobs. No session2.'
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    train=torch.load(args.prepared/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    val=torch.load(args.prepared/'validation.pt',map_location='cpu',weights_only=False,mmap=True)
    source=inspect.getsource(module.train_variant)
    old='        score, rows = validate(model, val_loader, device)'
    if source.count(old)!=1:raise RuntimeError('Training loop changed')
    source=source.replace(old,old+'\n        low_ok=sum(r["domain"]==1 and r["balanced_accuracy"]>=.7-1e-9 for r in rows)\n        all_ok=sum(r["balanced_accuracy"]>=.7-1e-9 for r in rows)\n        print(f"COUNTS seed={seed} variant={variant} low_at70={low_ok}/25 all_at70={all_ok}/54",flush=True)\n        (output/"validation_epochs").mkdir(exist_ok=True)\n        write_json(output/"validation_epochs"/f"epoch_{epoch+1:03d}.json",rows)')
    namespace=dict(vars(module));exec(compile(source,'<paired original trainer>','exec'),namespace)
    namespace['train_variant'](options,'online',args.variant,args.seed,train,val,prep)
    print('DONE',args.variant,args.seed,flush=True)


def summarize(args):
    results=[]
    for seed in (42,43,44):
        pair={}
        for variant in ('baseline','plan_a'):
            directory=(args.baseline if variant=='baseline' else args.candidate) if seed==42 else args.run_root/'online'/variant/f'seed_{seed}'
            file=directory/'validation_subjects.json'
            if not file.exists():continue
            if seed!=42 and not (args.run_root/f'{variant}_{seed}.completed').exists():continue
            rows=json.loads(file.read_text());low=[r for r in rows if r['domain']==1]
            pair[variant]={'low_mean':sum(r['balanced_accuracy'] for r in low)/len(low),'low_at70':sum(r['balanced_accuracy']>=.7-1e-9 for r in low),'all_at70':sum(r['balanced_accuracy']>=.7-1e-9 for r in rows)}
        record={'seed':seed,'results':pair}
        if len(pair)==2:
            record['candidate_minus_baseline']={k:pair['plan_a'][k]-pair['baseline'][k] for k in pair['baseline']}
        results.append(record)
    report={'pairs':results,'note':'Original mean-selected checkpoints; paired seed differences, reused validation, no session2. Only completed new jobs included.'}
    print(json.dumps(report,indent=2))
    if args.run_root.exists():write(args.run_root/'paired_summary.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/dann_pair_43_44_001'))
    p.add_argument('--prepared',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    p.add_argument('--baseline',type=Path,default=Path('/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42'))
    p.add_argument('--candidate',type=Path,default=Path('/mne_data/mne_data/bci_runs/plan_a_domain003_001/online/plan_a/seed_42'))
    p.add_argument('--csv',type=Path,default=Path('/mne_data/mne_data/openbmi_bci_illiteracy_result.csv'))
    p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int,choices=[43,44]);p.add_argument('--variant',choices=['baseline','plan_a'])
    p.add_argument('--status',action='store_true');p.add_argument('--summarize',action='store_true')
    args=p.parse_args()
    if args.worker:worker(args);return
    if args.status:
        print((args.run_root/'status.json').read_text() if (args.run_root/'status.json').exists() else 'No status file');return
    if args.summarize:summarize(args);return
    args.run_root.mkdir(parents=True,exist_ok=False)
    state={'phase':'running','jobs':{}};write(args.run_root/'status.json',state)
    for seed in (43,44):
        for variant in ('baseline','plan_a'):
            key=f'{variant}_{seed}';log=args.run_root/(key+'.log')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--seed',str(seed),'--variant',variant]
            for flag,value in [('run-root',args.run_root),('prepared',args.prepared),('baseline',args.baseline),('candidate',args.candidate),('csv',args.csv)]:command.extend(['--'+flag,str(value)])
            with log.open('w') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state['jobs'][key]={'status':'running','pid':child.pid,'log':str(log)};write(args.run_root/'status.json',state)
                print('START',key,'PID',child.pid,'LOG',log,flush=True)
                code=child.wait()
            state['jobs'][key].update(status='completed' if code==0 else 'failed',exit_code=code);write(args.run_root/'status.json',state)
            if code:
                state['phase']='failed';write(args.run_root/'status.json',state)
                raise RuntimeError(f'{key} failed; see {log}. Remaining jobs not started.')
            (args.run_root/(key+'.completed')).write_text('completed')
            print('DONE',key,flush=True)
    state['phase']='done';write(args.run_root/'status.json',state);summarize(args)


if __name__=='__main__':main()
