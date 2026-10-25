"""Checkpoint gradient probe: session1 train only; no optimizer or saved-weight changes."""
from pathlib import Path
import argparse, csv, json, math, sys
sys.path.insert(0,str(Path.cwd()))
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from src.plan_a.data import cached_trials
from src.plan_a.model import EEGNetLight, ShallowConvNetTeacher
from src.plan_b.model import EEGNetLightPlanB, supervised_contrastive_loss
from src.plan_b.augment import EEGAugment, prepare_view


def gradient(loss, params):
    values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,values)]).detach()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=Path('/mne_data/mne_data/bci_runs'))
    parser.add_argument('--data-root',type=Path,default=Path('/mne_data/mne_data'))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output filename')
    torch.set_num_threads(4)
    # Math attention supports repeated autograd.grad without fused attention limitations.
    if args.device.startswith('cuda'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    rows=[]
    runs=[('baseline','formal_a_001/online/baseline/seed_42'),
          ('plan_a','formal_a_001/online/plan_a/seed_42'),
          ('plan_b','shared_baseline_b_001/online/plan_b/seed_42')]
    for name,relative in runs:
        root=args.base/relative
        c=json.loads((root/'config.json').read_text()); hp=c['hyperparameters']
        prep=joblib.load(root/'preprocessing.joblib')
        cls=EEGNetLightPlanB if name=='plan_b' else EEGNetLight
        model=cls(**c['model']).to(args.device).train()
        teacher=ShallowConvNetTeacher(11).to(args.device).train()
        cp=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(cp['model']);teacher.load_state_dict(cp['teacher'])
        epoch=cp['epoch']+1; del cp
        params=[p for n,p in model.named_parameters() if n.startswith(('temporal.','proj.','trm.','csp_fc.'))]
        pools={}
        for domain in [0,1]:
            selected=[s for s in c['subjects'] if prep['groups'][s][c['domain_source']]==domain][:2]
            for s in selected:
                raw,y=cached_trials(args.data_root,args.base/'cache',s,1,'train',c['mode'])
                pools[s]=(raw,y)
        selected={d:[s for s in pools if prep['groups'][s][c['domain_source']]==d] for d in [0,1]}
        for batch in range(8):
            rng=np.random.default_rng(2026+batch)
            xs=[];ys=[];ds=[]
            for domain in [0,1]:
                s=selected[domain][batch%len(selected[domain])];raw,y=pools[s]
                for label in [0,1]:
                    index=rng.choice(np.flatnonzero(y==label))
                    xs.append(raw[index]);ys.append(label);ds.append(domain)
            raw=np.stack(xs)
            normalized=torch.tensor(((raw-prep['mean'])/prep['scale']).astype(np.float32))
            torch.manual_seed(100+batch)
            if name=='plan_b':
                aug=EEGAugment(band_strength=hp['band_strength'],channel_mask=hp['channel_mask'],noise_std=hp['noise_std'])
                views=[prepare_view(normalized,prep,aug) for _ in range(2)]
            else:
                features=prep['csp_scaler'].transform(prep['csp'].transform(raw.astype(np.float64))).astype(np.float32)
                views=[(normalized,torch.tensor(features))]
            labels=torch.tensor(ys,device=args.device);domains=torch.tensor(ds,device=args.device)
            ces=[];kds=[];projections=[];domain_loss=None;teacher_acc=[]
            for x,features in views:
                x=x.to(args.device);features=features.to(args.device)
                with torch.no_grad(): t=teacher(x)
                teacher_acc.append((t.argmax(1)==labels).float().mean().item())
                if name=='plan_b':
                    logits,z=model(x,features,return_features=True);projections.append(z)
                else:
                    coefficient=hp.get('grl_max',1)*(2/(1+math.exp(-10*((epoch-1)/max(1,hp['epochs']-1))))-1)
                    logits,domain_logits,_=model(x,features,coefficient)
                    if domain_logits is not None:
                        domain_loss=.5*sum(F.cross_entropy(domain_logits[domains==d],domains[domains==d]) for d in [0,1])
                ces.append(F.cross_entropy(logits,labels))
                temp=hp['temperature']
                kds.append(F.kl_div(F.log_softmax(logits/temp,1),F.softmax(t/temp,1),reduction='batchmean')*temp**2)
            ce=torch.stack(ces).mean();kd=torch.stack(kds).mean()
            losses={'ce':(1-hp['distill_alpha'])*ce,'kd':hp['distill_alpha']*kd}
            if name=='plan_a': losses['aux']=hp['domain_weight']*domain_loss
            if name=='plan_b': losses['aux']=hp['contrastive_weight']*supervised_contrastive_loss(torch.stack(projections,1),labels,hp['contrastive_temperature'])
            grads={key:gradient(loss,params) for key,loss in losses.items()}
            for key,g in grads.items():
                norm=g.norm().item();base=grads['ce'].norm().item()
                if not torch.isfinite(g).all(): raise ValueError('Non-finite gradient')
                row=dict(run=name,epoch=epoch,batch=batch,term=key,weighted_loss=losses[key].item(),
                         gradient_norm=norm,ratio_to_ce=norm/max(base,1e-12),
                         cosine_to_ce=F.cosine_similarity(g,grads['ce'],dim=0).item(),teacher_accuracy=float(np.mean(teacher_acc)))
                rows.append(row)
            print(name,'batch',batch+1,'CE',ce.item(),flush=True)
        del model,teacher
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name,_ in runs:
        for term in ['ce','kd','aux']:
            group=[r for r in rows if r['run']==name and r['term']==term]
            if group: print(name,term,'median norm ratio',np.median([r['ratio_to_ce'] for r in group]),'mean cosine',np.mean([r['cosine_to_ce'] for r in group]),flush=True)
    print('DONE',args.output,flush=True)
    print('Snapshot diagnostic, FP32/train mode, 8 balanced batches, last checkpoints at different epochs. '
          'No optimizer steps, no session2. Negative cosine or large ratios are clues, not causal proof.')


if __name__=='__main__':main()
