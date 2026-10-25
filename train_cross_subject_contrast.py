"""Original A baseline + fixed cross-subject training-memory contrast. No src edits."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import joblib
import torch
from torch import nn
from torch.nn import functional as F
from src.plan_a import train as original
from src.plan_a.model import EEGNetLight


class ContrastModel(EEGNetLight):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.projector=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,64))

    def forward(self,eeg,csp_features,grl_coefficient=None):
        result=super().forward(eeg,csp_features,grl_coefficient)
        if grl_coefficient is not None:
            self.aux_features=result[2]
        return result


class Memory:
    def __init__(self):
        self.keys=self.labels=self.subjects=None
        self.valid=0

    def loss(self,features,labels,subjects):
        anchors=F.normalize(features.float(),dim=1)
        loss=anchors.sum()*0
        self.valid=0
        if self.keys is not None:
            # Both positive and negative keys must come from other subjects.
            eligible=subjects[:,None]!=self.subjects[None,:]
            positive=eligible & (labels[:,None]==self.labels[None,:])
            negative=eligible & (labels[:,None]!=self.labels[None,:])
            valid=positive.any(1)&negative.any(1)
            self.valid=int(valid.sum())
            if valid.any():
                logits=anchors[valid]@self.keys.T/.1
                mask=eligible[valid];pos=positive[valid]
                logp=logits-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),dim=1,keepdim=True)
                loss=-(logp.masked_fill(~pos,0).sum(1)/pos.sum(1)).mean()
        with torch.no_grad():
            keys=anchors.detach();ys=labels.detach();ss=subjects.detach()
            if self.keys is not None:
                keys=torch.cat([self.keys,keys]);ys=torch.cat([self.labels,ys]);ss=torch.cat([self.subjects,ss])
            self.keys=keys[-256:];self.labels=ys[-256:];self.subjects=ss[-256:]
        return loss


def replace(source,old,new):
    if source.count(old)!=1:raise RuntimeError('Original source differs: '+old[:70])
    return source.replace(old,new)


def trainer():
    source=inspect.getsource(original.train_variant)
    source=replace(source,'    start_epoch, best_score, patience, history = 0, -1.0, 0, []',
                   '    start_epoch, best_score, patience, history = 0, -1.0, 0, []\n    memory=Memory()')
    source=replace(source,'        examples = 0','        examples = 0\n        aux_total=0.; valid_total=0')
    source=replace(source,'(eeg, targets, csp_features, domains, _) in enumerate(train_loader)',
                   '(eeg, targets, csp_features, domains, subjects) in enumerate(train_loader)')
    source=replace(source,'            scaler.scale(loss).backward()',
        '            weight=args.aux_weight*min(1.,max(0.,(epoch-9)/20))\n'
        '            with torch.autocast(device_type=device.type,enabled=False):\n'
        '                aux=memory.loss(model.projector(model.aux_features.float()),targets,subjects.to(device))\n'
        '            loss=loss+weight*aux\n'
        '            aux_total+=float(aux.detach())*len(targets);valid_total+=memory.valid\n'
        '            scaler.scale(loss).backward()')
    source=replace(source,'        score, rows = validate(model, val_loader, device)',
        '        score, rows = validate(model, val_loader, device)\n'
        '        low_ok=sum(r["domain"]==1 and r["balanced_accuracy"]>=.7-1e-9 for r in rows)\n'
        '        all_ok=sum(r["balanced_accuracy"]>=.7-1e-9 for r in rows)\n'
        '        (output/"validation_epochs").mkdir(exist_ok=True)\n'
        '        write_json(output/"validation_epochs"/f"epoch_{epoch+1:03d}.json",rows)')
    source=replace(source,'"val_target_balanced_accuracy": score, "grl": coefficient,',
        '"val_target_balanced_accuracy": score, "grl": coefficient, "contrastive":aux_total/examples, "aux_weight":weight, "valid_anchor_fraction":valid_total/examples, "low_at70":low_ok, "all_at70":all_ok,')
    source=replace(source,'        print(f"[{mode}/{variant}/{seed}] epoch=',
        '        print(f"cross_subject aux={aux_total/examples:.4f} weight={weight:.4f} valid={valid_total/examples:.3f} low_at70={low_ok}/25 all_at70={all_ok}/54",flush=True)\n'
        '        print(f"[{mode}/{variant}/{seed}] epoch=')
    namespace=dict(vars(original));namespace.update(EEGNetLight=ContrastModel,Memory=Memory)
    exec(compile(source,'<original baseline plus cross-subject contrast>','exec'),namespace)
    return namespace['train_variant']


def check():
    original.set_seed(42);base=EEGNetLight(n_channels=11,n_times=20,enhance=False,dann=False)
    original.set_seed(42);new=ContrastModel(n_channels=11,n_times=20,enhance=False,dann=False)
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in base.state_dict().items())
    base.eval();new.eval();x=torch.randn(4,11,20);c=torch.randn(4,6)
    assert torch.equal(base(x,c),new(x,c))
    base.train();new.train()
    original.set_seed(9);l=base(x,c,0.)[0];loss=l.square().mean();loss.backward()
    original.set_seed(9);l2=new(x,c,0.)[0];m=Memory();aux=m.loss(new.projector(new.aux_features),torch.tensor([0,1,0,1]),torch.tensor([1,1,2,2]))
    (l2.square().mean()+0*aux).backward()
    assert all(torch.equal(v.grad,dict(new.named_parameters())[k].grad) for k,v in base.named_parameters())
    m=Memory();q=torch.randn(4,64,requires_grad=True);labels=torch.tensor([0,1,0,1]);subjects=torch.tensor([1,1,2,2])
    assert m.loss(q,labels,subjects).item()==0
    loss=m.loss(q,labels,subjects);assert m.valid==4 and torch.isfinite(loss)
    loss.backward();assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
    m=Memory();m.loss(q,labels,torch.ones(4,dtype=torch.long));assert m.loss(q,labels,torch.ones(4,dtype=torch.long)).item()==0
    trainer()
    print('PASS: common initialization/logits; zero-weight common gradients; valid cross-subject positives/negatives; same-subject exclusion; finite gradients; original-loop patch. CPU synthetic only.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/cross_subject_contrast_001'))
    parser.add_argument('--prepared-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    parser.add_argument('--reference',type=Path,default=Path('/mne_data/mne_data/bci_runs/formal_a_001/online/baseline/seed_42'))
    options=parser.parse_args()
    if options.check:check();return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for server run')
    ref=json.loads((options.reference/'config.json').read_text())
    root=Path(original.__file__).parent
    fingerprint=hashlib.sha256(b''.join((root/n).read_bytes() for n in ('model.py','data.py','train.py'))).hexdigest()
    if fingerprint!=ref['code_fingerprint']:raise ValueError('Original baseline source changed')
    args=original.parser().parse_args([])
    for k,v in ref['hyperparameters'].items():
        if hasattr(args,k):setattr(args,k,v)
    args.groups_csv=Path('/mne_data/mne_data/openbmi_bci_illiteracy_result.csv')
    if hashlib.sha256(args.groups_csv.read_bytes()).hexdigest()!=ref['groups_csv_sha256']:raise ValueError('CSV differs')
    args.subjects=ref['subjects'];args.output_root=options.output_root;args.device='cuda';args.resume=False
    args.aux_weight=.01;args.aux_recipe='Cross-subject FIFO256, temperature .1, no augmentation, warmup10 ramp20; projector192-128-64. Original CE+KD untouched. No resume: memory not checkpointed.'
    args.entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if options.smoke:args.epochs=1;args.patience=1;args.aux_weight=0.
    prep=joblib.load(options.prepared_root/'preprocessing.joblib')
    metadata=json.loads((options.prepared_root/'metadata.json').read_text())
    if prep['data_fingerprint']!=ref['data_fingerprint'] or metadata['source_fingerprint']!=fingerprint:raise ValueError('Prepared data/source mismatch')
    train=torch.load(options.prepared_root/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    val=torch.load(options.prepared_root/'validation.pt',map_location='cpu',weights_only=False,mmap=True)
    torch.set_num_threads(4)
    print('All54 subjects; shared cached data; original baseline + cross-subject contrast. Distillation retained.',flush=True)
    trainer()(args,'online','baseline',42,train,val,prep)
    print('DONE',options.output_root,flush=True)


if __name__=='__main__':main()
