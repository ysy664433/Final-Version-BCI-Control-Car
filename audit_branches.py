"""Frozen branch probes; training labels only, fixed LR, no neural training/session2."""
import argparse
import json
from pathlib import Path
from audit_representation import fit_probes, metrics
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def run(root, output):
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError('Choose a new output file')
    data = np.load(root / 'features_and_predictions.npz')
    splits = {}
    for split in ('train', 'validation'):
        splits[split] = {k: data[split+'_'+k] for k in
                         ('fusion', 'csp', 'labels', 'domains', 'subjects')}
        if splits[split]['fusion'].shape[1] != 192:
            raise ValueError('Expected temporal 128 + CSP embedding 64')
    recorded = json.loads((root / 'baseline_head_subjects.json').read_text())
    baseline = {r['subject']: r for r in recorded}
    report = {'fixed_recipe': 'Weighted scaler + LR C=1; same domain/class weights for all probes',
              'results': {}, 'limits': 'Exploratory reused validation. Frozen jointly trained representations; probes do not prove end-to-end ablation benefit. No session2.'}
    for name, indices in [('full_control', slice(None)),
                          ('temporal_only', slice(0,128)),
                          ('csp_embedding_only', slice(128,192))]:
        features = {split: dict(values, fusion=values['fusion'][:,indices])
                    for split,values in splits.items()}
        with threadpool_limits(limits=4):
            probe = fit_probes(features['train'], features['validation'])['fusion_linear']
        result = {}
        for split in splits:
            rows, summary = metrics(probe[split+'_probabilities'], **{
                k:splits[split][k] for k in ('labels','domains','subjects')})
            result[split] = summary
            if split == 'validation':
                low = [r for r in rows if r['domain']==1]
                up = [r['subject'] for r in low if baseline[r['subject']]['balanced_accuracy'] < .7-1e-9 and r['balanced_accuracy'] >= .7-1e-9]
                down = [r['subject'] for r in low if baseline[r['subject']]['balanced_accuracy'] >= .7-1e-9 and r['balanced_accuracy'] < .7-1e-9]
                result['paired_vs_original_head'] = {'low_crossed_up':up, 'low_crossed_down':down, 'net_low_at70':len(up)-len(down)}
                result['validation_subjects'] = rows
        if name == 'full_control':
            previous = data['fusion_linear_validation_probabilities']
            error = float(np.abs(previous-probe['validation_probabilities']).max())
            result['previous_probe_max_probability_difference'] = error
            previous_rows, previous_summary = metrics(previous, **{
                k:splits['validation'][k] for k in ('labels','domains','subjects')})
            result['previous_probe_validation'] = previous_summary
            result['previous_probe_changed_predictions'] = int(np.sum(previous.argmax(1) != probe['validation_probabilities'].argmax(1)))
            previous_at70 = {r['subject'] for r in previous_rows if r['balanced_accuracy'] >= .7-1e-9}
            current_at70 = {r['subject'] for r in result['validation_subjects'] if r['balanced_accuracy'] >= .7-1e-9}
            result['previous_probe_changed_at70_subjects'] = sorted(previous_at70 ^ current_at70)
            result['previous_probe_probability_reproduced'] = error <= 1e-5
            result['solver_iterations'] = probe['classifier'].named_steps['logisticregression'].n_iter_.tolist()
            if error > 1e-5:
                print('NOTICE: previous full probe differs; differences saved. Compare branches against this run full_control; do not claim exact reproduction.', flush=True)
        report['results'][name] = result
        print(name, json.dumps(result['validation']), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('DONE', output, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mne_data/mne_data/bci_runs/representation_audit_002'))
    parser.add_argument('--output', type=Path, default=Path('/mne_data/mne_data/bci_runs/branch_audit_001.json'))
    args = parser.parse_args()
    run(args.root, args.output)
