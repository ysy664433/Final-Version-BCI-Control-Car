"""Fixed session2/test evaluation of previously selected original seed42 models."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import torch
from src.plan_a.evaluate import evaluate_run
from src.plan_a.train import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/original_pair_test_001'))
    args=p.parse_args()
    if args.output_root.exists():raise FileExistsError('Output exists; choose a new directory')
    runs=[Path('/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42'),
          Path('/mne_data/mne_data/bci_runs/plan_a_domain003_001/online/plan_a/seed_42')]
    configs=[json.loads((r/'config.json').read_text()) for r in runs]
    for key in ('subjects','domain_source','data_fingerprint','code_fingerprint','protocol'):
        if configs[0][key]!=configs[1][key]:raise ValueError('Different '+key)
    if configs[0]['domain_source']!='training':raise ValueError('Training-only grouping required')
    args.data_root=Path('/mne_data/mne_data');args.cache_root=args.data_root/'bci_runs/cache'
    args.device='cuda';args.batch_size=4
    torch.set_num_threads(4)
    report={'protocol':'Fixed original seed42 best checkpoints; no training or test-based selection. Primary group is training_low; supplied groups not used.', 'runs':{}}
    results=[]
    for name,directory in zip(('original_baseline','original_plan_a_003'),runs):
        rows,_=evaluate_run(directory,args);results.append(rows)
        summary={}
        for group,selected in [('low',[r for r in rows if r['training_low']]),('all',rows)]:
            summary[group]={'mean_accuracy':sum(r['accuracy'] for r in selected)/len(selected),
                            'at70':sum(r['accuracy']>=.7-1e-9 for r in selected),'subjects':len(selected)}
        report['runs'][name]={'metrics':summary,'subjects':rows}
        write_json(args.output_root/'summary.json',report)
        print(name,json.dumps(summary),flush=True)
    base={r['subject']:r for r in results[0]}
    low=[r for r in results[1] if r['training_low']]
    report['paired_low']={'crossed_up':[r['subject'] for r in low if base[r['subject']]['accuracy']<.7-1e-9 and r['accuracy']>=.7-1e-9],
                          'crossed_down':[r['subject'] for r in low if base[r['subject']]['accuracy']>=.7-1e-9 and r['accuracy']<.7-1e-9]}
    write_json(args.output_root/'summary.json',report)
    print(json.dumps(report['paired_low']),flush=True)
    print('DONE',args.output_root,flush=True)


if __name__=='__main__':main()
