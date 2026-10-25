"""One-GPU conservative experiment queue. Standard library only.

No unbounded parallel launch, no automatic reruns, no deletion/overwrite of old runs.
"""
import argparse, csv, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
JOBS={
 'subject_adaptive':('train_subject_weighted.py',['--reweight-mode','adaptive_low','--reweight-cap','1.5']),
 'subject_uniform':('train_subject_weighted.py',['--reweight-mode','uniform_low','--reweight-cap','1.25']),
 'subject_all':('train_subject_weighted.py',['--reweight-mode','adaptive_all','--reweight-cap','1.5']),
 'no_kd_control':('train_subject_weighted.py',['--reweight-mode','none','--reweight-cap','1','--kd-scale','0']),
 'delayed_dann':('train_recipe.py',['--recipe','delayed_dann','--domain-warmup','10','--domain-ramp','40']),
 'gradient_protected':('train_recipe.py',['--recipe','gradient_protected']),
 'weak_contrast':('train_recipe.py',['--recipe','weak_contrast']),
 'weak_augment':('train_recipe.py',['--recipe','weak_augment']),
}


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def gpu_memory(index):
    text=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
    return tuple(int(v.strip()) for v in text.strip().splitlines()[0].split(','))


def available_ram_mb():
    host=None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'): host=int(line.split()[1])/1024
    except OSError: pass
    limits=[]
    for limit,current in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                          ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            value=Path(limit).read_text().strip()
            if value!='max': limits.append(max(0,int(value)-int(Path(current).read_text()))/1024**2)
        except (OSError,ValueError): pass
    if host is not None: limits.append(host)
    return min(limits) if limits else 0


def commands(args):
    common=['--data-root',str(args.data_root),'--groups-csv',str(args.groups_csv),
            '--cache-root',str(args.cache_root),'--reference-run',str(args.reference_run),
            '--prepared-root',str(args.run_root/'_prepared'),'--device','cuda','--amp']
    return {name:[sys.executable,'-u',str(HERE/JOBS[name][0]),*common,'--output-root',str(args.run_root/name),*JOBS[name][1]] for name in args.jobs}


def owned_process(pid,root):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        needle=str(root).encode()
        return any(a==needle or a.startswith(needle+b'/') for a in args)
    except OSError: return False


def stop_owned_group(pid,root):
    if not owned_process(pid,root): return False
    try:
        if os.getpgid(pid)!=pid: return False
        os.killpg(pid,signal.SIGTERM)
        return True
    except ProcessLookupError: return False


def refresh_progress(state,root):
    for name,row in state['jobs'].items():
        paths=list((root/name).glob('online/*/seed_42/history.json'))
        if not paths: continue
        try:
            history=json.loads(paths[0].read_text())
            if history:
                h=history[-1]
                row['progress']={k:h[k] for k in ['epoch','val_target_balanced_accuracy','low_at_least70','all_at_least70'] if k in h}
        except (OSError,ValueError): pass


def summarize(root,reference_run):
    from compare_thresholds import summarize as paired_summary
    baseline={r['subject']:r for r in json.loads((reference_run/'validation_subjects.json').read_text())}
    state=json.loads((root/'status.json').read_text())
    table=[]
    for name,job in state['jobs'].items():
        row={'job':name,'status':job['status'],'completed_epochs':'','low_mean':'','low_below70':'','low_crossed_up':'','low_crossed_down':'','low_net_new_at70':'','all_net_new_at70':''}
        files=list((root/name).glob('online/*/seed_42/validation_subjects.json'))
        if files:
            candidate=json.loads(files[0].read_text())
            if {r['subject'] for r in candidate}!=baseline.keys(): raise ValueError('Unpaired subject records')
            pairs=[dict(subject=r['subject'],domain=r['domain'],baseline=baseline[r['subject']]['balanced_accuracy'],
                        candidate=r['balanced_accuracy'],delta=r['balanced_accuracy']-baseline[r['subject']]['balanced_accuracy']) for r in candidate]
            if any(p['domain']!=baseline[p['subject']]['domain'] for p in pairs): raise ValueError('Group membership changed')
            low=paired_summary([p for p in pairs if p['domain']==1]);all_people=paired_summary(pairs)
            history=json.loads((files[0].parent/'history.json').read_text())
            row.update(completed_epochs=history[-1]['epoch'],low_mean=low['candidate_mean'],low_below70=low['candidate_below70'],
                       low_crossed_up=json.dumps(low['crossed_up_subjects']),low_crossed_down=json.dumps(low['crossed_down_subjects']),
                       low_net_new_at70=len(low['crossed_up_subjects'])-len(low['crossed_down_subjects']),
                       all_net_new_at70=len(all_people['crossed_up_subjects'])-len(all_people['crossed_down_subjects']))
        table.append(row)
    with (root/'experiment_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(table,indent=2),flush=True)
    print('Primary mean-selected checkpoints only. Partial/failed runs are labelled; no session2 evidence. Threshold-selected checkpoints are exploratory.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--data-root',type=Path,default=Path('/mne_data/mne_data'))
    parser.add_argument('--groups-csv',type=Path,default=Path('/mne_data/mne_data/openbmi_bci_illiteracy_result.csv'))
    parser.add_argument('--cache-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/cache'))
    parser.add_argument('--reference-run',type=Path,default=Path('/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42'))
    parser.add_argument('--jobs',nargs='+',choices=list(JOBS),default=list(JOBS))
    parser.add_argument('--max-parallel',type=int,default=2)
    parser.add_argument('--gpu',type=int,default=0,help='nvidia-smi visible GPU index; uses existing CUDA_VISIBLE_DEVICES for workers')
    parser.add_argument('--gpu-reserve-mb',type=int,default=8192)
    parser.add_argument('--gpu-headroom-mb',type=int,default=2048)
    parser.add_argument('--ram-reserve-mb',type=int,default=8192)
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    args.run_root=args.run_root.resolve();args.project_root=args.project_root.resolve()
    if args.status:
        state=json.loads((args.run_root/'status.json').read_text());refresh_progress(state,args.run_root)
        print(json.dumps(state,indent=2));return
    if args.stop:
        state=json.loads((args.run_root/'status.json').read_text())
        for row in state['jobs'].values():
            if row.get('status')=='running' and row.get('pid'): print('Stop child:',row['pid'],stop_owned_group(row['pid'],args.run_root))
        if state.get('preparation_pid'): print('Stop preparation:',stop_owned_group(state['preparation_pid'],args.run_root))
        pid=state['launcher_pid']
        if owned_process(pid,args.run_root):
            os.kill(pid,signal.SIGTERM);print('Launcher stop requested; no more jobs will start.')
        return
    if args.summarize:
        summarize(args.run_root,args.reference_run);return
    if args.max_parallel<1 or args.gpu_reserve_mb<1024 or args.gpu_headroom_mb<0 or args.ram_reserve_mb<1024 or len(set(args.jobs))!=len(args.jobs):
        raise ValueError('Invalid capacity settings or duplicate jobs')
    plan=commands(args)
    if args.dry_run:
        print(json.dumps(plan,indent=2));return
    if os.name!='posix': raise RuntimeError('Launch on the Linux GPU server; local --dry-run is supported')
    if args.run_root.exists(): raise FileExistsError('Use a NEW run root. Existing experiments are never overwritten or auto-restarted.')
    for file in [args.project_root/'src/plan_a/train.py',args.project_root/'src/plan_b/train.py',args.groups_csv,
                 args.reference_run/'config.json',args.reference_run/'validation_subjects.json']:
        if not file.is_file(): raise FileNotFoundError(file)
    free,total=gpu_memory(args.gpu)
    slots=max(0,(total-args.gpu_headroom_mb)//args.gpu_reserve_mb)
    if slots<1: raise RuntimeError('Insufficient total GPU memory for configured reservation')
    limit=min(args.max_parallel,slots)
    args.run_root.mkdir()
    manifest={'project_root':str(args.project_root),'reference_run':str(args.reference_run),'commands':plan,
              'maximum_concurrency':limit,'gpu_total_mb':total,'gpu_reserve_mb':args.gpu_reserve_mb,
              'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
              'note':'All subjects, seed42, 120 epochs/30 patience, original sampler. No src edits, fine tuning, or session2. Concurrency does not guarantee a speedup; VRAM reservations are estimates.'}
    atomic_json(args.run_root/'manifest.json',manifest)
    state={'launcher_pid':os.getpid(),'phase':'preparing','jobs':{name:{'status':'queued'} for name in plan}}
    stopping={'value':False}
    def request_stop(*unused): stopping['value']=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    active={}
    def save():
        refresh_progress(state,args.run_root);atomic_json(args.run_root/'status.json',state)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    prep_command=[sys.executable,'-u',str(HERE/'shared_data.py'),'--prepared-root',str(args.run_root/'_prepared'),
                  '--reference-run',str(args.reference_run),'--data-root',str(args.data_root),'--groups-csv',str(args.groups_csv),'--cache-root',str(args.cache_root)]
    print('GPU total/free MB:',total,free,'maximum simultaneous jobs:',limit,flush=True)
    print('Preparing once; log:',args.run_root/'preparation.log',flush=True)
    with (args.run_root/'preparation.log').open('w') as log:
        prep=subprocess.Popen(prep_command,cwd=args.project_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['preparation_pid']=prep.pid;save()
        while prep.poll() is None and not stopping['value']: time.sleep(2)
        if stopping['value']:
            stop_owned_group(prep.pid,args.run_root);state['phase']='stopped';save();return
        if prep.returncode:
            state['phase']='preparation_failed';save();raise RuntimeError('Preparation failed; inspect preparation.log. No training launched.')
    state.pop('preparation_pid',None);state['phase']='training';save()
    try:
        while True:
            for name,(process,log) in list(active.items()):
                code=process.poll()
                if code is not None:
                    log.close();state['jobs'][name].update(status='completed' if code==0 else 'failed',exit_code=code,ended=time.time())
                    del active[name];print(name,'finished, exit',code,flush=True)
            if stopping['value']:
                for name,(process,log) in active.items():
                    stop_owned_group(process.pid,args.run_root)
                    state['jobs'][name]['status']='stop_requested'
                for row in state['jobs'].values():
                    if row['status']=='queued': row['status']='cancelled'
                state['phase']='stopped';save();break
            queued=[name for name,row in state['jobs'].items() if row['status']=='queued']
            if not queued and not active:
                state['phase']='done';save();break
            if queued and len(active)<limit:
                free,_=gpu_memory(args.gpu);ram=available_ram_mb()
                state['resources']={'gpu_free_mb':free,'available_ram_mb':round(ram),'maximum_concurrency':limit}
                if free>=args.gpu_reserve_mb+args.gpu_headroom_mb and ram>=args.ram_reserve_mb+2048:
                    name=queued[0];log=(args.run_root/(name+'.log')).open('w')
                    process=subprocess.Popen(plan[name],cwd=args.project_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[name]=(process,log);state['jobs'][name].update(status='running',pid=process.pid,started=time.time())
                    print('Started',name,'PID',process.pid,'log',args.run_root/(name+'.log'),flush=True)
            save();time.sleep(5)
    except Exception as exc:
        state['phase']='launcher_failed';state['error']=repr(exc)
        for name in active: state['jobs'][name]['status']='stop_requested'
        for row in state['jobs'].values():
            if row['status']=='queued': row['status']='cancelled'
        save()
        raise
    finally:
        for name,(process,log) in active.items():
            stop_owned_group(process.pid,args.run_root)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print('Still shutting down:',name,process.pid,'check manually; no forced kill issued.',flush=True)
            log.close()
    summarize(args.run_root,args.reference_run)
    print('BATCH FINISHED',state['phase'],flush=True)


if __name__=='__main__':main()
