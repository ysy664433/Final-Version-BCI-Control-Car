"""Late class-conditional domain alignment; original KD retained, fixed balanced batches."""
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

RECIPES=('control','ordinary','classwise')


class ClasswiseModel(EEGNetLight):
    def __init__(self,**kwargs):
        kwargs=dict(kwargs,enhance=False,dann=False)
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            head=nn.Sequential(nn.Linear(192,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,2))
            self.domain_heads=nn.ModuleList([head,copy.deepcopy(head)])

    def forward(self,eeg,csp,grl_coefficient=None):
        if grl_coefficient is None:return super().forward(eeg,csp)
        logits,_,features=super().forward(eeg,csp,0.)
        reversed_features=GradientReverseLayer.apply(features,grl_coefficient)
        domains=torch.stack([h(reversed_features) for h in self.domain_heads],dim=1)
        return logits,domains,features


class FourCellBatches(Sampler):
    def __init__(self,data,seed):
        joint=data.tensors[3]*2+data.tensors[1]
        self.cells=[torch.flatnonzero(joint==k) if hasattr(torch,'flatnonzero') else torch.nonzero(joint==k,as_tuple=True)[0] for k in range(4)]
        if any(len(c)==0 for c in self.cells) or len(data)%4:raise ValueError('Four nonempty domain/class cells and length divisible by4 required')
        self.n=len(data)//4;self.seed=seed
    def __len__(self):return self.n
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed)
        draws=torch.stack([c[torch.randint(len(c),(self.n,),generator=g)] for c in self.cells],dim=1)
        for row in draws:yield row[torch.randperm(4,generator=g)].tolist()


def domain_loss(logits,labels,domains,recipe):
    if recipe=='control':return logits.sum()*0,0
    if recipe=='ordinary':
        if domains.unique().numel()!=2:return logits.sum()*0,0
        return .5*sum(F.cross_entropy(logits.mean(1)[domains==d],domains[domains==d]) for d in (0,1)),1
    losses=[]
    for label in (0,1):
        selected=labels==label
        if domains[selected].unique().numel()==2:
            losses.append(.5*sum(F.cross_entropy(logits[selected & (domains==d),label],domains[selected & (domains==d)]) for d in (0,1)))
    return (torch.stack(losses).mean(),len(losses)) if losses else (logits.sum()*0,0)


def initialize(model,teacher,checkpoint):
    missing,extra=model.load_state_dict(checkpoint['model'],strict=False)
    if extra or any(not k.startswith('domain_heads.') for k in missing):raise ValueError('Unexpected common state mismatch')
    teacher.load_state_dict(checkpoint['teacher'])


@torch.no_grad()
def validate(model,data,device,recipe):
    model.eval();ys=[];pred=[];ss=[];ds=[];domain_pred=[]
    for x,y,c,d,s in DataLoader(data,batch_size=8):
        logits,dl,_=model(x.to(device),c.to(device),0.)
        yp=dl.mean(1) if recipe!='classwise' else dl[torch.arange(len(y),device=device),y.to(device)]
        ys.extend(y.tolist());pred.extend(logits.argmax(1).cpu().tolist())
        ss.extend(s.tolist());ds.extend(d.tolist());domain_pred.extend(yp.argmax(1).cpu().tolist())
    ys=np.array(ys);ds=np.array(ds);dp=np.array(domain_pred)
    rows=base.subject_metrics(ys,np.array(pred),np.array(ss),ds)
    score=float(np.mean([r['balanced_accuracy'] for r in rows if r['domain']==1]))
    domain_bacc={str(c):float(np.mean([np.mean(dp[(ys==c)&(ds==d)]==d) for d in (0,1)])) for c in (0,1)}
    return score,rows,domain_bacc


def train_one(args,recipe,train,val,prep,checkpoint,ref):
    out=args.output_root/recipe;out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device);base.set_seed(args.seed)
    cfg=dict(ref['model'],enhance=False,dann=False)
    model=ClasswiseModel(**cfg).to(device)
    teacher=ShallowConvNetTeacher(11).to(device);initialize(model,teacher,checkpoint)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    teacher_opt=torch.optim.AdamW(teacher.parameters(),lr=.001,weight_decay=.001)
    amp=device.type=='cuda' and ref['hyperparameters']['amp']
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    config={'framework':'late_classwise_v1','recipe':recipe,'seed':args.seed,'model':cfg,
            'data_fingerprint':prep['data_fingerprint'],'reference':str(args.reference),'initial_epoch':checkpoint['epoch']+1,
            'source_fingerprint':ref['code_fingerprint'],'entry_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'settings':{'epochs':10,'batch_size':4,'student_lr':.0001,'teacher_lr':.001,'domain_weight':.03,'distill_alpha':.5,'temperature':3.,'variance_weight':1e-5,'amp':amp},
            'protocol':'session1/train only; session1/test validation; session2/test reserved for fixed follow-up evaluation. All54, fixed training25 groups. Same two domain heads and dropout calls in all arms; ordinary averages heads, classwise routes by TRAIN task labels. Control has zero auxiliary loss. Four-cell sampling differs from previous experiments.'}
    base.write_json(out/'config.json',config);joblib.dump(prep,out/'preprocessing.joblib')
    score,rows,_=validate(model,val,device,recipe);best=score;history=[]
    base.write_json(out/'initial_subjects.json',rows);base.write_json(out/'validation_subjects.json',rows)
    base.checkpoint_save(out/'best.pt',{'model':model.state_dict(),'epoch':0,'validation_score':score})
    print(f'INITIAL {recipe} low_mean={score:.4f}',flush=True)
    (out/'validation_epochs').mkdir()
    for epoch in range(10):
        base.set_seed(args.seed+epoch*1009)
        loader=DataLoader(train,batch_sampler=FourCellBatches(train,args.seed+epoch),num_workers=0)
        model.train();teacher.train();totals={k:0. for k in ('ce','kd','domain','loss')};valid_domains=0
        for bi,(x,y,c,d,_) in enumerate(loader):
            x=x.to(device);y=y.to(device);c=c.to(device);d=d.to(device)
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
                aux,valid=domain_loss(dl,y,d,recipe)
                loss=.5*(ce+1e-5*variance)+.5*kd+.03*aux
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(opt);scaler.update()
            valid_domains+=valid
            for k,v in [('ce',ce),('kd',kd),('domain',aux),('loss',loss)]:totals[k]+=float(v.detach())*len(y)
        score,rows,domain_bacc=validate(model,val,device,recipe)
        low=sum(r['domain']==1 and r['balanced_accuracy']>=.7-1e-9 for r in rows);all_ok=sum(r['balanced_accuracy']>=.7-1e-9 for r in rows)
        record={'epoch':epoch+1,**{k:v/len(train) for k,v in totals.items()},'val_low_bacc':score,'low_at70':low,'all_at70':all_ok,'domain_bacc_by_task':domain_bacc,'valid_domain_tasks_per_batch':valid_domains/len(loader),'gamma_last':gamma}
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
    m=EEGNetLight(n_channels=11,n_times=20,enhance=False,dann=False);t=ShallowConvNetTeacher(11)
    cp={'model':m.state_dict(),'teacher':t.state_dict()}
    n=ClasswiseModel(n_channels=11,n_times=20);nt=ShallowConvNetTeacher(11);initialize(n,nt,cp)
    m.eval();n.eval();x=torch.randn(4,11,20);c=torch.randn(4,6)
    assert torch.equal(m(x,c),n(x,c))
    y=torch.tensor([0,1,0,1]);d=torch.tensor([0,0,1,1])
    z=torch.randn(4,2,2,requires_grad=True);loss,valid=domain_loss(z,y,d,'classwise');assert valid==2
    expected=.5*(F.cross_entropy(z[[0,2],0],d[[0,2]])+F.cross_entropy(z[[1,3],1],d[[1,3]]))
    assert torch.allclose(loss,expected)
    loss.backward();assert z.grad[0,1].abs().sum()==0 and z.grad[0,0].abs().sum()>0
    zero,v=domain_loss(z,y,torch.zeros(4,dtype=torch.long),'classwise');assert zero==0 and v==0
    data=TensorDataset(torch.zeros(8,1),y.repeat(2),torch.zeros(8,1),d.repeat(2),torch.ones(8))
    for batch in FourCellBatches(data,43):assert sorted((data.tensors[3][batch]*2+data.tensors[1][batch]).tolist())==[0,1,2,3]
    # Same classifier path/RNG and zero auxiliary gradient in all arms.
    grads=[]
    for recipe in RECIPES:
        n.train();n.zero_grad(set_to_none=True);base.set_seed(9)
        logits,dl,_=n(x,c,.5);aux,_=domain_loss(dl,y,d,recipe)
        (logits.square().mean()+0*aux).backward()
        grads.append({k:p.grad.clone() for k,p in n.named_parameters() if not k.startswith('domain_heads.')})
    assert all(torch.equal(grads[0][k],g[k]) for g in grads[1:] for k in grads[0])
    f=torch.tensor([1.],requires_grad=True);GradientReverseLayer.apply(f,.4).sum().backward();assert torch.allclose(f.grad,torch.tensor([-.4]))
    print('PASS: common initialization/logits; correct class-head routing; missing-domain handling; balanced batches; zero-weight common gradients; gradient reversal. CPU synthetic only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true');p.add_argument('--preflight',action='store_true')
    p.add_argument('--recipes',nargs='+',choices=RECIPES,default=list(RECIPES))
    p.add_argument('--seed',type=int,choices=[43],default=43)
    p.add_argument('--reference',type=Path,default=Path('/mne_data/mne_data/bci_runs/dann_pair_43_44_001/online/baseline/seed_43'))
    p.add_argument('--prepared',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    p.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/late_classwise_43_001'))
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
    train=torch.load(args.prepared/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    val=torch.load(args.prepared/'validation.pt',map_location='cpu',weights_only=False,mmap=True)
    if len(train)!=5400 or len(val)!=5400:raise ValueError('Expected full data')
    model=ClasswiseModel(**ref['model']).to(args.device);teacher=ShallowConvNetTeacher(11);initialize(model,teacher,cp)
    score,_,_=validate(model,val,args.device,'classwise')
    expected=cp['history'][-1]['val_target_balanced_accuracy']
    if abs(score-expected)>1e-8:raise ValueError(f'Initial reproduction failed: {score} vs {expected}')
    print(f'PREFLIGHT PASS initial_epoch={cp["epoch"]+1} low_mean={score:.4f}',flush=True)
    del model,teacher
    if args.preflight:return
    if len(set(args.recipes))!=len(args.recipes):raise ValueError('Duplicate recipes')
    if any((args.output_root/r).exists() for r in args.recipes):raise FileExistsError('Recipe output exists; use a new root')
    for recipe in args.recipes:train_one(args,recipe,train,val,prep,cp,ref)
    print('ALL REQUESTED RUNS DONE',args.output_root,flush=True)


if __name__=='__main__':main()
