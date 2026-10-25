"""Compare paired session1 validation records, no training or checkpoint loading."""
import argparse
import csv
import json
from pathlib import Path


def summarize(rows):
    # Tolerance handles balanced accuracies such as 0.6999999999999999.
    below = lambda value: value < .7 - 1e-9
    up = [r['subject'] for r in rows if below(r['baseline']) and not below(r['candidate'])]
    down = [r['subject'] for r in rows if not below(r['baseline']) and below(r['candidate'])]
    return {'subjects': len(rows),
            'baseline_mean': sum(r['baseline'] for r in rows)/len(rows),
            'candidate_mean': sum(r['candidate'] for r in rows)/len(rows),
            'baseline_below70': sum(below(r['baseline']) for r in rows),
            'candidate_below70': sum(below(r['candidate']) for r in rows),
            'crossed_up_subjects': up, 'crossed_down_subjects': down,
            'improved': sum(r['delta'] > 1e-9 for r in rows),
            'unchanged': sum(abs(r['delta']) <= 1e-9 for r in rows),
            'worsened': sum(r['delta'] < -1e-9 for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    base = Path('/mne_data/mne_data/bci_runs')
    parser.add_argument('--baseline', type=Path, default=base/'formal_a_001/online/baseline/seed_42')
    parser.add_argument('--candidate', type=Path, default=base/'classwise_original_001/online/plan_a/seed_42')
    parser.add_argument('--output', type=Path, default=base/'classwise_threshold_report_001')
    args = parser.parse_args()
    baseline = json.loads((args.baseline/'validation_subjects.json').read_text())
    candidate = json.loads((args.candidate/'validation_subjects.json').read_text())
    cb, cc = [json.loads((p/'config.json').read_text()) for p in (args.baseline, args.candidate)]
    for key in ['data_fingerprint','subjects','domain_source','protocol']:
        if cb[key] != cc[key]: raise ValueError('Mismatched comparison setting: '+key)
    b = {r['subject']:r for r in baseline}; c = {r['subject']:r for r in candidate}
    if len(b)!=len(baseline) or len(c)!=len(candidate) or b.keys()!=c.keys():
        raise ValueError('Duplicate or missing paired subjects')
    rows=[]
    for s in sorted(b):
        if b[s]['domain']!=c[s]['domain']: raise ValueError('Domain membership changed')
        x,y=b[s]['balanced_accuracy'],c[s]['balanced_accuracy']
        transition=('below_to_at_least70' if x<.7-1e-9 and y>=.7-1e-9 else
                    'at_least70_to_below' if x>=.7-1e-9 and y<.7-1e-9 else 'no_threshold_crossing')
        rows.append({'subject':s,'domain':b[s]['domain'],'baseline':x,'candidate':y,'delta':y-x,'transition':transition})
    report={group:summarize([r for r in rows if domain is None or r['domain']==domain])
            for group,domain in [('training_low_group',1),('training_high_group',0),('all',None)]}
    print(json.dumps(report,indent=2))
    print('\nThreshold crossings in the training-defined low group:')
    for r in rows:
        if r['domain']==1 and r['transition']!='no_threshold_crossing':
            print(f"subject {r['subject']}: {r['baseline']:.2%} -> {r['candidate']:.2%} ({r['transition']})")
    if args.output.exists(): raise FileExistsError('Choose another --output directory; report not overwritten')
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    with (args.output/'subjects.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('\nSaved:',args.output)
    print('Validation threshold only; not clinical diagnosis or independent session2 evidence.')


if __name__=='__main__':main()
