"""Late subject-identity domain adversary versus group-domain and no-adversary controls."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Sampler
from src.plan_a import train as base
from src.plan_a.model import EEGNetLight,ShallowConvNetTeacher,GradientReverseLayer

RECIPES=('control','group')


class SubjectDomainModel(EEGNetLight):
    def __init__(self,**kwargs):
        super().__init__(**dict(kwargs,enhance=False,dann=False))
        with torch.random.fork_rng(devices=[]):
            self.domain_heads=nn.ModuleDict({
                'group':nn.Sequential(nn.Linear(192,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,2)),
                'subject':nn.Sequential(nn.Linear(192,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,54))})

    def forward(self,eeg,csp,grl_coefficient=None):
        if grl_coefficient is None:return super().forward(eeg,csp)
        logits,_,features=super().forward(eeg,csp,0.)
        reversed_features=GradientReverseLayer.apply(features,grl_coefficient)
        return logits,{k:head(reversed_features) for k,head in self.domain_heads.items()},features


class SubjectBalancedBatches(Sampler):
    def __init__(self,data,seed):
        labels=data.tensors[1];domains=data.tensors[3];subjects=data.tensors[4]
        if len(data)%8:raise ValueError('Trial count must be divisible by8')
        self.groups=[[],[]];self.cells={};self.n=len(data)//8;self.seed=seed
        for subject in subjects.unique().tolist():
            if not 1<=subject<=54:raise ValueError('Expected subject IDs1..54')
            selected=subjects==subject;d=domains[selected].unique()
            if len(d)!=1 or int(d[0]) not in (0,1):raise ValueError('Inconsistent subject grouping')
            self.groups[int(d[0])].append(subject)
            for label in (0,1):
                ids=torch.nonzero(selected & (labels==label),as_tuple=True)[0]
                if not len(ids):raise ValueError('Each subject requires both classes')
                self.cells[subject,label]=ids
        if min(map(len,self.groups))<2:raise ValueError('Need at least2 subjects per fixed group')
    def __len__(self):return self.n
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed)
        for _ in range(self.n):
            chosen=[]
            for group in self.groups:
                for i in torch.randperm(len(group),generator=g)[:2].tolist():
                    subject=group[i]
                    for label in (0,1):
                        ids=self.cells[subject,label]
                        chosen.append(int(ids[torch.randint(len(ids),(1,),generator=g)]))
            order=torch.randperm(8,generator=g).tolist()
            yield [chosen[i] for i in order]


def domain_loss(logits,labels,domains,subjects,recipe):
    # Both heads computed in every arm, so common classifier dropout calls match.
    zero=sum(x.float().sum()*0 for x in logits.values())
    if recipe=='control':return zero,0
    if recipe=='group':
        if domains.unique().numel()!=2:return zero,0
        loss=.5*sum(F.cross_entropy(logits['group'][domains==d],domains[domains==d]) for d in (0,1))
        return loss+zero,2
    if subjects.min()<1 or subjects.max()>54:raise ValueError('Subject label out of range')
    if subjects.unique().numel()<2:return zero,0
    return F.cross_entropy(logits['subject'],subjects-1)+zero,int(subjects.unique().numel())


def initialize(model,teacher,checkpoint):
    missing,extra=model.load_state_dict(checkpoint['model'],strict=False)
    if extra or any(not k.startswith('domain_heads.') for k in missing):raise ValueError('Unexpected common state mismatch')
    teacher.load_state_dict(checkpoint['teacher'])


@torch.no_grad()
def validate(model,data,device,recipe):
    model.eval();ys=[];pred=[];ss=[];ds=[];gp=[];sp=[]
    for x,y,c,d,subject in DataLoader(data,batch_size=8):
        logits,dl,_=model(x.to(device),c.to(device),0.)
        ys.extend(y.tolist());pred.extend(logits.argmax(1).cpu().tolist())
        ss.extend(subject.tolist());ds.extend(d.tolist())
        gp.extend(dl['group'].argmax(1).cpu().tolist());sp.extend((dl['subject'].argmax(1)+1).cpu().tolist())
    ds=np.array(ds);ss=np.array(ss);gp=np.array(gp);sp=np.array(sp)
    rows=base.subject_metrics(np.array(ys),np.array(pred),ss,ds)
    score=float(np.mean([r['balanced_accuracy'] for r in rows if r['domain']==1]))
    probe={'group_head_bacc':float(np.mean([np.mean(gp[ds==d]==d) for d in (0,1)])),
           'subject_head_macro_accuracy':float(np.mean([np.mean(sp[ss==sid]==sid) for sid in np.unique(ss)])),
           'chance_group':.5,'chance_subject':1/54,
           'note':'Training adversaries, not independent frozen-feature probes; inactive head accuracy is not evidence of invariance.'}
    return score,rows,probe


def train_one(args,recipe,train,val,prep,checkpoint,ref):
    out=args.output_root/recipe;out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device);base.set_seed(args.seed)
    cfg=dict(ref['model'],enhance=False,dann=False)
    model=SubjectDomainModel(**cfg).to(device)
    teacher=ShallowConvNetTeacher(11).to(device);initialize(model,teacher,checkpoint)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    teacher_opt=torch.optim.AdamW(teacher.parameters(),lr=.001,weight_decay=.001)
    amp=device.type=='cuda' and ref['hyperparameters']['amp']
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    config={'framework':'late_subject_domain_v1','recipe':recipe,'seed':args.seed,'model':cfg,
            'data_fingerprint':prep['data_fingerprint'],'reference':str(args.reference),'initial_epoch':checkpoint['epoch']+1,
            'source_fingerprint':ref['code_fingerprint'],'entry_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'replication_note':'Source pretrained seed43 last107, continuation seed44. Fixed2-arm balanced8 group-domain repeat; changes continuation RNG only. Not an independent seed44 end-to-end training; original seed44 baseline failure retained in reporting. Same previously exposed session2. Fixed10epochs, no search.',
            'source_pretraining_seed':43,
            'settings':{'epochs':10,'batch_size':8,'student_lr':.0001,'teacher_lr':.001,'domain_weight':.03,'distill_alpha':.5,'temperature':3.,'variance_weight':1e-5,'amp':amp},
            'protocol':'session1/train only; existing session1/test validation. Three arms compute same group2 and subject54 heads. Fixed8-trial batch: two subjects per performance group, left/right trial per subject. Control auxiliary0; group adversary2-way; subject adversary54-way. Same raw domain CE weight .03, not a matched-gradient-strength claim. No EA, new bands, class conditioning, extra loss or regrouping. Post-test-exposure follow-up research.'}
    base.write_json(out/'config.json',config);joblib.dump(prep,out/'preprocessing.joblib')
    score,rows,_=validate(model,val,device,recipe);best=score;history=[]
    base.write_json(out/'initial_subjects.json',rows);base.write_json(out/'validation_subjects.json',rows)
    base.checkpoint_save(out/'best.pt',{'model':model.state_dict(),'epoch':0,'validation_score':score})
    print(f'INITIAL {recipe} low_mean={score:.4f}',flush=True)
    (out/'validation_epochs').mkdir()
    for epoch in range(10):
        base.set_seed(args.seed+epoch*1009)
        loader=DataLoader(train,batch_sampler=SubjectBalancedBatches(train,args.seed+epoch),num_workers=0)
        model.train();teacher.train();totals={k:0. for k in ('ce','kd','domain','loss')};valid_domains=0
        for bi,(x,y,c,d,subjects) in enumerate(loader):
            x=x.to(device);y=y.to(device);c=c.to(device);d=d.to(device);subjects=subjects.to(device)
            teacher_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                t=teacher(x);tl=F.cross_entropy(t,y)
            scaler.scale(tl).backward();scaler.unscale_(teacher_opt)
            nn.utils.clip_grad_norm_(teacher.parameters(),5.);scaler.step(teacher_opt)
            progress=(epoch+bi/len(loader))/9
            gamma=2/(1+math.exp(-10*progress))-1
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                logits,dl,_=model(x,c,gamma if recipe!='control' else 0.)
                ce=F.cross_entropy(logits,y)
                kd=F.kl_div(F.log_softmax(logits.float()/3,dim=1),F.softmax(t.detach().float()/3,dim=1),reduction='batchmean')*9
                variance=logits.float().softmax(1).var(dim=1,unbiased=False).mean()
                aux,valid=domain_loss(dl,y,d,subjects,recipe)
                loss=.5*(ce+1e-5*variance)+.5*kd+.03*aux
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(opt);scaler.update()
            valid_domains+=valid
            for k,v in [('ce',ce),('kd',kd),('domain',aux),('loss',loss)]:totals[k]+=float(v.detach())*len(y)
        score,rows,domain_bacc=validate(model,val,device,recipe)
        low=sum(r['domain']==1 and r['balanced_accuracy']>=.7-1e-9 for r in rows);all_ok=sum(r['balanced_accuracy']>=.7-1e-9 for r in rows)
        record={'epoch':epoch+1,**{k:v/len(train) for k,v in totals.items()},'val_low_bacc':score,'low_at70':low,'all_at70':all_ok,'adversary_metrics':domain_bacc,'valid_domain_classes_per_batch':valid_domains/len(loader),'domain_ce_over_log_classes':(totals['domain']/len(train))/math.log(54 if recipe=='subject' else 2),'gamma_last':gamma}
        history.append(record);base.write_json(out/'history.json',history)
        base.write_json(out/'validation_epochs'/f'epoch_{epoch+1:02d}.json',rows)
        cp={'model':model.state_dict(),'teacher':teacher.state_dict(),'optimizer':opt.state_dict(),'teacher_optimizer':teacher_opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'history':history}
        base.checkpoint_save(out/'last.pt',cp)
        if score>best:
            best=score;base.checkpoint_save(out/'best.pt',{'model':model.state_dict(),'epoch':epoch+1,'validation_score':score});base.write_json(out/'validation_subjects.json',rows)
        print(recipe,json.dumps(record),flush=True)
    print('DONE',recipe,flush=True)


def check():
    from torch.utils.data import TensorDataset
    base.set_seed(43)
    old=EEGNetLight(n_channels=11,n_times=20,enhance=False,dann=False);teacher=ShallowConvNetTeacher(11)
    cp={'model':old.state_dict(),'teacher':teacher.state_dict()}
    model=SubjectDomainModel(n_channels=11,n_times=20);other=ShallowConvNetTeacher(11);initialize(model,other,cp)
    old.eval();model.eval();x=torch.randn(8,11,20);c=torch.randn(8,6)
    assert torch.equal(old(x,c),model(x,c))
    y=torch.tensor([0,1]*4);d=torch.tensor([0]*4+[1]*4);subjects=torch.tensor([1,1,2,2,3,3,4,4])
    dl={'group':torch.randn(8,2,requires_grad=True),'subject':torch.randn(8,54,requires_grad=True)}
    loss,n=domain_loss(dl,y,d,subjects,'subject');assert n==4
    assert torch.allclose(loss,F.cross_entropy(dl['subject'],subjects-1))
    loss.backward();assert dl['subject'].grad.abs().sum()>0 and dl['group'].grad.abs().sum()==0
    data=TensorDataset(x.repeat(2,1,1),y.repeat(2),c.repeat(2,1),d.repeat(2),subjects.repeat(2))
    batches=list(SubjectBalancedBatches(data,43));assert batches==list(SubjectBalancedBatches(data,43))
    for batch in batches:
        sd=data.tensors[4][batch];lab=data.tensors[1][batch];dom=data.tensors[3][batch]
        assert len(sd.unique())==4 and int((dom==0).sum())==4
        assert all(sorted(lab[sd==s].tolist())==[0,1] for s in sd.unique())
    grads=[]
    for recipe in RECIPES:
        model.train();model.zero_grad(set_to_none=True);base.set_seed(7)
        logits,domains,_=model(x,c,.4);aux,_=domain_loss(domains,y,d,subjects,recipe)
        (logits.square().mean()+0*aux).backward()
        grads.append({k:p.grad.clone() for k,p in model.named_parameters() if not k.startswith('domain_heads.')})
    assert all(torch.equal(grads[0][k],g[k]) for g in grads[1:] for k in grads[0])
    f=torch.ones(1,requires_grad=True);GradientReverseLayer.apply(f,.4).sum().backward();assert torch.allclose(f.grad,torch.tensor([-.4]))
    print('PASS: matched student/logits; subject IDs1..54->0..53; correct active-head gradients; deterministic four-subject/eight-trial batches; zero-weight common gradients; gradient reversal. Synthetic CPU only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true');p.add_argument('--preflight',action='store_true')
    p.add_argument('--recipes',nargs='+',choices=RECIPES,default=list(RECIPES))
    p.add_argument('--seed',type=int,choices=[44],default=44)
    p.add_argument('--reference',type=Path,default=Path('/mne_data/mne_data/bci_runs/dann_pair_43_44_001/online/baseline/seed_43'))
    p.add_argument('--prepared',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    p.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/group_domain_source43_seed44_001'))
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.check:check();return
    if args.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('Server CUDA required')
    torch.set_num_threads(4)
    ref=json.loads((args.reference/'config.json').read_text())
    root=Path(base.__file__).parent
    fingerprint=hashlib.sha256(b''.join((root/f).read_bytes() for f in ('model.py','data.py','train.py'))).hexdigest()
    if ref['code_fingerprint']!=fingerprint or ref['seed']!=43 or ref['model']['dann'] or ref['model']['enhance']:raise ValueError('Unexpected reference/source')
    if ref['subjects']!=list(range(1,55)) or ref['domain_source']!='training':raise ValueError('Expected54/train groups')
    cp=torch.load(args.reference/'last.pt',map_location='cpu',weights_only=False)
    prep=joblib.load(args.prepared/'preprocessing.joblib')
    if prep['data_fingerprint']!=ref['data_fingerprint']:raise ValueError('Data mismatch')
    reference_prep=joblib.load(args.reference/'preprocessing.joblib')
    for key in ('mean','scale'):
        if not np.array_equal(prep[key],reference_prep[key]):raise ValueError('Reference preprocessing differs: '+key)
    for key in ('filters_','patterns_'):
        if not np.array_equal(getattr(prep['csp'],key),getattr(reference_prep['csp'],key)):raise ValueError('Reference CSP differs: '+key)
    for key in ('mean_','scale_'):
        if not np.array_equal(getattr(prep['csp_scaler'],key),getattr(reference_prep['csp_scaler'],key)):raise ValueError('Reference CSP scaling differs')
    train=torch.load(args.prepared/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    val=torch.load(args.prepared/'validation.pt',map_location='cpu',weights_only=False,mmap=True)
    if len(train)!=5400 or len(val)!=5400:raise ValueError('Expected full data')
    model=SubjectDomainModel(**ref['model']).to(args.device);teacher=ShallowConvNetTeacher(11);initialize(model,teacher,cp)
    score,_,_=validate(model,val,args.device,'subject')
    expected=cp['history'][-1]['val_target_balanced_accuracy']
    if abs(score-expected)>1e-8:raise ValueError(f'Initial reproduction failed: {score} vs {expected}')
    print(f'PREFLIGHT PASS initial_epoch={cp["epoch"]+1} low_mean={score:.4f}',flush=True)
    del model,teacher
    if args.preflight:return
    if len(set(args.recipes))!=len(args.recipes):raise ValueError('Duplicate recipes')
    if any((args.output_root/r).exists() for r in args.recipes):raise FileExistsError('Recipe output exists; use a new root')
    for recipe in args.recipes:train_one(args,recipe,train,val,prep,cp,ref)
    summary={}
    for recipe in args.recipes:
        rows=json.loads((args.output_root/recipe/'validation_subjects.json').read_text())
        low=[r for r in rows if r['domain']==1]
        summary[recipe]={'low_mean':float(np.mean([r['balanced_accuracy'] for r in low])),
                         'low_at70':sum(r['balanced_accuracy']>=.7-1e-9 for r in low),
                         'all_at70':sum(r['balanced_accuracy']>=.7-1e-9 for r in rows)}
    base.write_json(args.output_root/'validation_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    print('ALL REQUESTED RUNS DONE',args.output_root,flush=True)


if __name__=='__main__':main()
