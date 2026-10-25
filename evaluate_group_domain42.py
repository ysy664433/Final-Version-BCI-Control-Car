"""Fixed session2 evaluation for repeat_group_domain42.py models. No model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import joblib
import numpy as np
import torch
from scipy.stats import binomtest
from repeat_group_domain42 import SubjectDomainModel,base
from src.plan_a.data import cached_trials


def metrics(rows):
    return {'subjects':len(rows),'mean_accuracy':float(np.mean([r['accuracy'] for r in rows])),
            'at70':sum(r['accuracy']>=.7-1e-9 for r in rows)}


def paired(reference,candidate):
    a={r['subject']:r for r in reference}
    if set(a)!={r['subject'] for r in candidate}:raise ValueError('Subject sets differ')
    up=[r['subject'] for r in candidate if a[r['subject']]['accuracy']<.7-1e-9 and r['accuracy']>=.7-1e-9]
    down=[r['subject'] for r in candidate if a[r['subject']]['accuracy']>=.7-1e-9 and r['accuracy']<.7-1e-9]
    diff=np.array([r['accuracy']-a[r['subject']]['accuracy'] for r in candidate])
    rng=np.random.default_rng(2026)
    means=rng.choice(diff,size=(10000,len(diff)),replace=True).mean(1)
    return {'mean_accuracy_gain':float(diff.mean()),'accuracy_gain_subject_bootstrap_ci95':np.quantile(means,[.025,.975]).tolist(),
            'crossed_up':up,'crossed_down':down,'net_at70':len(up)-len(down),
            'threshold_exact_mcnemar_p_two_sided':float(binomtest(len(up),len(up)+len(down),.5).pvalue) if up or down else 1.,
            'note':'Primary planned contrast: group vs control, fixed training-low group, threshold endpoint. Other contrasts exploratory; no multiplicity-adjusted claims. One seed, known/reused test set.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/group_domain_repeat42_001'))
    p.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/group_domain_repeat42_001_test'))
    p.add_argument('--data-root',type=Path,default=Path('/mne_data/mne_data'))
    p.add_argument('--cache-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/cache'))
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output_root.exists():raise FileExistsError('Choose a new output directory')
    dirs=sorted(f.parent for f in args.model_root.glob('*/config.json'))
    if not dirs:raise FileNotFoundError('No model configs')
    configs=[json.loads((d/'config.json').read_text()) for d in dirs]
    for c in configs:
        if c['framework']!='late_subject_domain_v1':raise ValueError('Unexpected framework')
        if c['entry_sha256']!=hashlib.sha256((Path(__file__).parent/'repeat_group_domain42.py').read_bytes()).hexdigest():raise ValueError('Training implementation file differs; use same script as training')
        for key in ('seed','data_fingerprint','reference','initial_epoch','settings','source_fingerprint'):
            if c[key]!=configs[0][key]:raise ValueError('Unmatched '+key)
    args.output_root.mkdir(parents=True)
    torch.set_num_threads(4);report={'runs':{},'comparisons':{},'limits':'Fixed selected checkpoints, session2/test, training-defined25 group unchanged. Follow-up exploratory analysis after prior test exposure. No test-based model/threshold selection.'};all_rows={}
    for directory,config in zip(dirs,configs):
        name=config['recipe'];out=args.output_root/name;out.mkdir()
        if not (directory/'history.json').exists() or len(json.loads((directory/'history.json').read_text()))!=10:raise ValueError('Training incomplete: '+name)
        prep=joblib.load(directory/'preprocessing.joblib')
        model=SubjectDomainModel(**config['model']).to(args.device).eval()
        checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'])
        rows=[]
        for subject in range(1,55):
            eeg,y=cached_trials(args.data_root,args.cache_root,subject,2,'test','online')
            if np.bincount(y).tolist()!=[50,50]:raise ValueError('Expected balanced100 test trials')
            raw=eeg.astype(np.float64);csp=prep['csp_scaler'].transform(prep['csp'].transform(raw)).astype(np.float32)
            x=((raw-prep['mean'])/prep['scale']).astype(np.float32);prob=[]
            with torch.inference_mode():
                for start in range(0,len(y),4):
                    logits=model(torch.as_tensor(x[start:start+4],device=args.device),torch.as_tensor(csp[start:start+4],device=args.device))
                    prob.append(logits.float().softmax(1).cpu().numpy())
            prob=np.concatenate(prob)
            d=np.full(len(y),prep['groups'][subject]['training'])
            row=base.subject_metrics(y,prob.argmax(1),np.full(len(y),subject),d)[0];rows.append(row)
            np.savez_compressed(out/f'subject_{subject}.npz',labels=y,probabilities=prob)
            print(f'{name} {subject}/54 accuracy={row["accuracy"]:.4f}',flush=True)
        base.write_json(out/'subjects.json',rows);all_rows[name]=rows
        report['runs'][name]={'selected_epoch':checkpoint['epoch'],'low':metrics([r for r in rows if r['domain']==1]),'all':metrics(rows)}
        del model,checkpoint
    for reference,candidate in [('control','group'),('control','subject'),('group','subject')]:
        if reference in all_rows and candidate in all_rows:
            report['comparisons'][candidate+'_vs_'+reference]={}
            for group in ('low','all'):
                a=[r for r in all_rows[reference] if group=='all' or r['domain']==1]
                b=[r for r in all_rows[candidate] if group=='all' or r['domain']==1]
                report['comparisons'][candidate+'_vs_'+reference][group]=paired(a,b)
    base.write_json(args.output_root/'comparison.json',report)
    print(json.dumps(report,indent=2),flush=True)
    print('DONE',args.output_root,flush=True)


if __name__=='__main__':main()
