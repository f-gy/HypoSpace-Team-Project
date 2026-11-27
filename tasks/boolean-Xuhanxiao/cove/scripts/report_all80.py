"""Derived, paired diagnostics and exportable plots for the frozen COVE study."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer.coverage_lora import dataset
from explorer.posterior_bank import bank
from explorer.tasks import write_json

OUT = ROOT / 'results/all80'


def main():
    d = dataset()
    summary = json.loads((OUT / 'test/summary.json').read_text())
    episodes = json.loads((OUT / 'test/episodes.json').read_text())
    task_ids = d['splits']['test']
    rng = np.random.default_rng(4311)
    indices = rng.integers(0, len(task_ids), size=(10000, len(task_ids)))
    comparisons = {}
    for protocol in ['adaptive_1x', 'fixed16']:
        def vector(method, metric):
            return np.array([np.mean([r[metric] for r in episodes if r['protocol'] == protocol and r['method'] == method and r['task'] == tid]) for tid in task_ids])
        comparisons[protocol] = {}
        for control in ['base_iid', 'frozen_iid', 'llm_catalog_wor', 'cove_only_4306', 'uniform_wor', 'symbolic_wor']:
            result = {}
            for metric in ['VR', 'NR', 'RR']:
                delta = vector('cove_llm_4306', metric) - vector(control, metric)
                ci = np.quantile(delta[indices].mean(axis=1) * 100, [.025, .975])
                result[metric] = {'difference_percentage_points': float(100 * delta.mean()),
                                  'task_paired_bootstrap_95_percent_interval_pp': ci.tolist()}
            comparisons[protocol][control] = result
    write_json(OUT / 'paired_differences.json', {'resamples': 10000, 'seed': 4311,
               'unit': '26 tasks; inference seeds averaged within task; descriptive exploratory intervals, no multiplicity-adjusted claim', 'comparisons': comparisons})
    costs = {}
    for split in ['validation', 'test']:
        scores = [json.loads(p.read_text()) for p in sorted((OUT / 'lm_scores').glob(f'{split}_*.json'))]
        costs[split] = {'tasks_scored': len(scores), 'candidate_scoring_seconds': sum(s['seconds'] for s in scores),
                        **{k: sum(s['cost'][k] for s in scores) for k in scores[0]['cost']},
                        'scores_reused_across_inference_seeds_and_catalog_controls': True}
    costs['executor_training_seconds'] = sum(r['seconds'] for r in json.loads((OUT / 'executor_training_summary.json').read_text())['runs'])
    costs['executor_parameters_per_run'] = 179
    costs['iid_extra_calls'] = 42
    costs['outputs_per_test_method'] = {'adaptive_1x': sum(len(t['reference_keys']) for t in d['tasks'] if t['id'] in task_ids) * 3, 'fixed16': 26 * 3 * 16}
    costs['interpretation'] = '66 teacher-forced sequence scores and one shared prompt prefill per task; 9 batched candidate forwards. These are additional inference computation, not N free autoregressive attempts. Output-budget matched, not equal FLOPs. Saved logit scoring is shared across seeds. Published outputs include replay and deterministic selection.'
    write_json(OUT / 'costs.json', costs)
    keys = {h['key'] for h in bank()}
    scope = {'catalog_classes': len(keys), 'all_reference_classes': len(d['universe']),
             'missing_catalog_keys': sorted({e['key'] for e in d['universe']} - keys),
             'adaptive_incomplete_tasks': [r for r in episodes if r['protocol'] == 'adaptive_1x' and r['method'] == 'cove_llm_4306' and r['seed'] == 17 and r['RR'] < 1],
             'budget_range': [min(len(t['reference_keys']) for t in d['tasks'] if t['id'] in task_ids), max(len(t['reference_keys']) for t in d['tasks'] if t['id'] in task_ids)],
             'interpretation': 'One upstream reference class lacks a validator-compatible training alias; keep full reference denominator. At N=K, unique in-reference valid selection implies VR=RR. Three headline metrics therefore are not three independent empirical gains.'}
    write_json(OUT / 'scope_audit.json', scope)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ['base_iid', 'frozen_iid', 'llm_catalog_wor', 'cove_llm_4306', 'cove_only_4306', 'symbolic_wor']
    labels = ['Base\nIID', 'Frozen adapted\nIID', 'LM catalog\nWOR', 'COVE + LM\n(proposed)', 'COVE only\ncontrol', 'Exact solver\ncontrol']
    colors = ['#3670ae', '#db8532', '#389780']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.1), sharey=True)
    x = np.arange(len(methods))
    for ax, protocol, title in zip(axes, ['adaptive_1x', 'fixed16'], ['Official README example: N = K', 'Original setting: N = 16']):
        for j, metric in enumerate(['VR', 'NR', 'RR']):
            values = [summary['metrics'][protocol][m][metric] * 100 for m in methods]
            ax.bar(x + (j - 1) * .24, values, width=.24, label=metric, color=colors[j])
        ax.axhline(80, color='#666666', ls='--', lw=1)
        ax.text(5.55, 81.5, '80%', ha='right', fontsize=9, color='#555555')
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.set_ylim(0, 108)
        ax.set_title(title, fontsize=12, pad=12)
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Macro score (%)')
    axes[0].legend(ncol=3, loc='upper left', frameon=False)
    fig.suptitle('COVE: learned compositional execution on the fixed 26-task Boolean test set', fontsize=14)
    fig.text(.5, .015, 'Output-budget matched; catalog scoring adds computation. COVE-only and exact controls match the coupled method.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'metrics.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'costs': costs, 'cove_vs_lm_catalog': comparisons['adaptive_1x']['llm_catalog_wor'], 'scope': scope}, indent=2))


if __name__ == '__main__':
    main()
