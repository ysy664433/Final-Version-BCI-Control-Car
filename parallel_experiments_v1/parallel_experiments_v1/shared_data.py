"""Prepare training-only tensors once, then read them in each experiment process."""
from pathlib import Path
import hashlib, json, sys
sys.path.insert(0,str(Path.cwd()))
import joblib
import torch
from src.plan_a import train as original


def fingerprint():
    root=Path(original.__file__).parent
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in ('model.py','data.py','train.py'))).hexdigest()


def load_prepared(args):
    root=Path(args.prepared_root)
    metadata=json.loads((root/'metadata.json').read_text())
    if metadata['source_fingerprint']!=fingerprint(): raise ValueError('Source changed since shared preparation')
    if metadata['groups_csv_sha256']!=hashlib.sha256(args.groups_csv.read_bytes()).hexdigest(): raise ValueError('Grouping CSV changed')
    if metadata['data_root']!=str(args.data_root.resolve()): raise ValueError('Data root changed')
    prep=joblib.load(root/'preprocessing.joblib')
    train=torch.load(root/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    val=torch.load(root/'validation.pt',map_location='cpu',weights_only=False,mmap=True)
    if prep['data_fingerprint']!=metadata['data_fingerprint']: raise ValueError('Preparation fingerprint differs')
    print('Shared preparation loaded; trials:',len(train),len(val),flush=True)
    return train,val,prep


def main():
    parser=original.parser()
    parser.add_argument('--prepared-root',required=True)
    parser.add_argument('--reference-run',required=True)
    parser.set_defaults(domain_source='training',modes=['online'])
    args=parser.parse_args()
    root=Path(args.prepared_root)
    if root.exists(): raise FileExistsError('Shared preparation directory already exists')
    ref=json.loads((Path(args.reference_run)/'config.json').read_text())
    if fingerprint()!=ref['code_fingerprint']: raise ValueError('Original source differs from recorded baseline')
    if args.subjects!=ref['subjects'] or args.domain_source!='training': raise ValueError('Use all reference subjects and training groups')
    csv_hash=hashlib.sha256(args.groups_csv.read_bytes()).hexdigest()
    if csv_hash!=ref['groups_csv_sha256']: raise ValueError('Grouping CSV differs')
    torch.set_num_threads(4)
    import mne
    mne.set_log_level('ERROR')
    train,val,prep=original.prepare(args,'online')
    if prep['data_fingerprint']!=ref['data_fingerprint']: raise ValueError('Training data/groups differ from baseline')
    root.mkdir(parents=True)
    torch.save(train,root/'train.pt');torch.save(val,root/'validation.pt')
    joblib.dump(prep,root/'preprocessing.joblib')
    (root/'metadata.json').write_text(json.dumps({'source_fingerprint':fingerprint(),'data_fingerprint':prep['data_fingerprint'],
        'data_root':str(args.data_root.resolve()),'groups_csv_sha256':csv_hash,'protocol':'session1 train/validation only; fixed all-subject train-fitted preprocessing'},indent=2))
    print('PREPARATION DONE',root,flush=True)


if __name__=='__main__':main()
