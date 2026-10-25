"""Session1-only spectral/spatial/separability audit. No new grouping or neural training."""
import argparse
import json
import os
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(name,'4')
os.environ.setdefault('MNE_LOGGING_LEVEL','ERROR')
import joblib
import numpy as np
import torch
from scipy.signal import welch
from scipy.integrate import trapezoid
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

BANDS={'theta':[4,8],'mu':[8,13],'beta':[13,30],'low_gamma':[30,45]}


def spdlog(matrix):
    values,vectors=eigh(matrix)
    return (vectors*np.log(np.maximum(values,1e-10)))@vectors.T


def describe(eeg,labels,channels):
    eeg=np.asarray(eeg,dtype=np.float64)
    if eeg.ndim!=3 or eeg.shape[1]!=len(channels) or not np.isfinite(eeg).all():raise ValueError('Expected finite EEG trials/channels/time')
    f,psd=welch(eeg,fs=250,nperseg=500,noverlap=250,axis=-1,detrend='constant')
    powers=[]
    for low,high in BANDS.values():
        selected=(f>=low)&(f<high)
        powers.append(trapezoid(psd[...,selected],f[selected],axis=-1))
    powers=np.maximum(np.stack(powers,axis=-1),1e-30)
    selected=(f>=4)&(f<45)
    total=np.maximum(trapezoid(psd[...,selected],f[selected],axis=-1),1e-30)
    relative=(powers/total[...,None]).mean(axis=(0,1))
    centered=eeg-eeg.mean(axis=-1,keepdims=True)
    cov=np.einsum('nct,ndt->cd',centered,centered,optimize=True)/(len(eeg)*(eeg.shape[-1]-1))
    energy=float(np.trace(cov))
    cov=cov/max(energy,1e-30)
    cov=(cov+cov.T)/2+np.eye(len(channels))*1e-6/len(channels)
    log_features=np.log(powers).reshape(len(eeg),-1)
    c3,c4=channels.index('C3'),channels.index('C4')
    lateral=np.log(powers[:,c3,1])-np.log(powers[:,c4,1])
    left=lateral[labels==0];right=lateral[labels==1]
    pooled=np.sqrt((left.var(ddof=1)+right.var(ddof=1))/2)
    effect=float((left.mean()-right.mean())/max(pooled,1e-12))
    psd_mean=psd.mean(axis=(0,1))
    mu_sel=(f>=8)&(f<=13)
    return {'relative_power':{name:float(v) for name,v in zip(BANDS,relative)},
            'theta_mu_ratio':float(relative[0]/max(relative[1],1e-12)),
            'mu_peak_hz':float(f[mu_sel][np.argmax(psd_mean[mu_sel])]),
            'mu_c3_c4_class_effect':effect,'total_channel_variance_native_units':energy,
            'p95_trial_channel_peak_to_peak_native_units':float(np.quantile(np.ptp(eeg,axis=-1),.95))},log_features,cov,f,psd_mean


def fit_separability(train_features,train_labels,val_features,val_labels):
    clf=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    with threadpool_limits(limits=4):
        cv=cross_val_score(clf,train_features,train_labels,cv=folds,scoring='balanced_accuracy')
        clf.fit(train_features,train_labels)
        train=float(balanced_accuracy_score(train_labels,clf.predict(train_features)))
        val=float(balanced_accuracy_score(val_labels,clf.predict(val_features)))
    return {'bandpower_lda_train_cv_bacc':float(cv.mean()),'bandpower_lda_train_fit_bacc':train,'bandpower_lda_validation_bacc':val,
            'note':'Fixed spectral probe, not the existing neural model/CSP grouping score. CV scaler and LDA fit within each training fold.'}


def plot_report(root,rows,covs,frequencies,spectra):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return 'matplotlib unavailable; JSON/NPZ results complete, plots skipped'
    ids=[r['subject'] for r in rows];logs=[spdlog(c) for c in covs]
    distance=np.array([[np.linalg.norm(a-b) for b in logs] for a in logs])
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    image=axes[0].imshow(distance,cmap='viridis',aspect='auto')
    axes[0].set(title='Training spatial covariance distance',xlabel='Subject (ordered 1-54)',ylabel='Subject (ordered 1-54)');fig.colorbar(image,ax=axes[0],label='Log-Euclidean distance')
    for group,color,label in [(0,'#2463a6','Training-defined high'),(1,'#cf6b25','Training-defined low')]:
        selected=np.array([r['training_low']==group for r in rows])
        s=np.array(spectra)[selected];mask=(frequencies>=4)&(frequencies<=45)
        s=s/np.maximum(trapezoid(s[:,mask],frequencies[mask],axis=1),1e-30)[:,None]
        axes[1].plot(frequencies[mask],np.median(s[:,mask],axis=0),color=color,label=label)
    axes[1].set(title='Training PSD: median of subject-normalized spectra',xlabel='Frequency (Hz)',ylabel='Relative spectral density (1/Hz)');axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(root/'training_overview.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
    for group,color,label in [(0,'#2463a6','High'),(1,'#cf6b25','Low')]:
        selected=[r for r in rows if r['training_low']==group]
        x=[r['separability']['bandpower_lda_train_cv_bacc'] for r in selected];y=[r['separability']['bandpower_lda_validation_bacc'] for r in selected]
        ax.scatter(x,y,color=color,label=label,alpha=.8)
        for r,a,b in zip(selected,x,y):ax.annotate(str(r['subject']),(a,b),fontsize=7)
    ax.plot([.4,1],[.4,1],ls='--',color='gray');ax.set(xlabel='Session1/train 5-fold CV balanced accuracy',ylabel='Session1/test validation balanced accuracy',title='Fixed bandpower probe: separability and transfer');ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'separability.png',dpi=160);plt.close(fig)
    return 'plots created'


def check():
    rng=np.random.default_rng(42);x=rng.normal(size=(20,11,1000));y=np.tile([0,1],10)
    channels=['FC1','FC2','FC5','FC6','C3','Cz','C4','CP1','CP2','CP5','CP6']
    summary,features,cov,f,psd=describe(x,y,channels)
    assert features.shape==(20,44) and np.isfinite(spdlog(cov)).all()
    scaled=describe(x*3,y,channels)
    assert np.allclose(cov,scaled[2]) and np.allclose(summary['relative_power']['mu'],scaled[0]['relative_power']['mu'])
    r=fit_separability(features,y,features,y)
    assert all(np.isfinite(r[k]) for k in r if k!='note')
    print('PASS: spectral/spatial feature shapes, finite SPD log, covariance scale invariance, fold-local spectral classifier. Synthetic only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--prepared',type=Path,default=Path('/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared'))
    p.add_argument('--output-root',type=Path,default=Path('/mne_data/mne_data/bci_runs/session1_eeg_profile_001'))
    args=p.parse_args()
    if args.check:check();return
    if args.output_root.exists():raise FileExistsError('Use a new output root; existing data not overwritten')
    torch.set_num_threads(4)
    prep=joblib.load(args.prepared/'preprocessing.joblib')
    train=torch.load(args.prepared/'train.pt',map_location='cpu',weights_only=False,mmap=True)
    val=torch.load(args.prepared/'validation.pt',map_location='cpu',weights_only=False,mmap=True)
    channels=list(prep['channels'])
    if len(train)!=5400 or len(val)!=5400 or len(channels)!=11:raise ValueError('Expected full54 cached data')
    train_ids=train.tensors[4].numpy();val_ids=val.tensors[4].numpy()
    if sorted(np.unique(train_ids).tolist())!=list(range(1,55)) or not np.array_equal(np.unique(train_ids),np.unique(val_ids)):raise ValueError('Subject mismatch')
    args.output_root.mkdir(parents=True)
    rows=[];covs=[];spectra=[];val_covs=[]
    for subject in range(1,55):
        print(f'{subject}/54: spectral/spatial/probe analysis',flush=True)
        tm=train_ids==subject;vm=val_ids==subject
        x=train.tensors[0][tm].numpy().astype(np.float64)*prep['scale']+prep['mean']
        v=val.tensors[0][vm].numpy().astype(np.float64)*prep['scale']+prep['mean']
        y=train.tensors[1][tm].numpy();vy=val.tensors[1][vm].numpy()
        if len(y)!=100 or len(vy)!=100 or np.bincount(y).tolist()!=[50,50] or np.bincount(vy).tolist()!=[50,50]:raise ValueError('Expected balanced100 per subject/split')
        ts,tf,tc,f,psd=describe(x,y,channels);vs,vf,vc,_,_=describe(v,vy,channels)
        probe=fit_separability(tf,y,vf,vy)
        row={'subject':subject,'training_low':int(prep['groups'][subject]['training']),
             'original_training_csp_cv_accuracy':float(prep['groups'][subject]['training_csp_lda_accuracy']),
             'train':ts,'validation':vs,'separability':probe,
             'within_subject_covariance_shift':float(np.linalg.norm(spdlog(vc)-spdlog(tc))),
             'validation_train_variance_log2_ratio':float(np.log2(vs['total_channel_variance_native_units']/max(ts['total_channel_variance_native_units'],1e-30)))}
        rows.append(row);covs.append(tc);val_covs.append(vc);spectra.append(psd)
        (args.output_root/'partial_subjects.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(f'{subject}/54 DONE bandpower CV={probe["bandpower_lda_train_cv_bacc"]:.3f} val={probe["bandpower_lda_validation_bacc"]:.3f} cov_shift={row["within_subject_covariance_shift"]:.3f}',flush=True)
        del x,v
    overall=np.mean(covs,axis=0)
    for row,cov in zip(rows,covs):row['training_covariance_distance_to_pooled']=float(np.linalg.norm(spdlog(cov)-spdlog(overall)))
    groups={}
    for group,value in [('training_low',1),('training_high',0)]:
        selected=[r for r in rows if r['training_low']==value]
        groups[group]={'n':len(selected),'median_theta_mu_ratio':float(np.median([r['train']['theta_mu_ratio'] for r in selected])),
            'median_abs_mu_lateralization_effect':float(np.median([abs(r['train']['mu_c3_c4_class_effect']) for r in selected])),
            'median_covariance_shift':float(np.median([r['within_subject_covariance_shift'] for r in selected])),
            'median_bandpower_train_cv':float(np.median([r['separability']['bandpower_lda_train_cv_bacc'] for r in selected])),
            'median_bandpower_validation':float(np.median([r['separability']['bandpower_lda_validation_bacc'] for r in selected]))}
    plots=plot_report(args.output_root,rows,covs,f,spectra)
    report={'protocol':'Session1/train and existing session1/test validation only, cached causal-preprocessed11 channels/250Hz/4s. Native values reconstructed using saved train-only mean/scale. No session2 access, neural updates, regrouping or sample exclusion.',
            'limits':'Task-period PSD, NOT resting-state SMR or ERD (no pre-cue baseline). Mu peak is descriptive maximum, not validated oscillatory peak above1/f. Covariance is trace-normalized with fixed diagonal floor. Group membership itself uses CSP separability: group differences are descriptive, not independent biomarkers or clinical findings. Spectral LDA is an auxiliary fixed probe, not a replacement baseline. No automatic choice of algorithm.',
            'data_fingerprint':prep['data_fingerprint'],'bands':BANDS,'groups':groups,'plots':plots,'subjects':rows}
    (args.output_root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.output_root/'spectra_covariances.npz',subjects=np.arange(1,55),frequency_hz=f,training_psd=np.array(spectra),training_trace_normalized_covariance=np.array(covs),validation_trace_normalized_covariance=np.array(val_covs))
    print(json.dumps(groups,indent=2),flush=True)
    print('DONE',args.output_root,flush=True)


if __name__=='__main__':main()
