"""Strip answer-bearing source cases into query-only inference inputs."""
import argparse
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = {'dev': 'joint_three_v1/dev_evaluation.json',
           'internal_test': 'joint_three_v1/test_evaluation.json',
           'causal4_pressure': 'causal4_pressure/cases.json'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=pathlib.Path)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    batches, hashes = {}, {}
    for split, relative in SOURCES.items():
        path = ROOT / relative
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        rows = []
        for case in json.loads(data):
            task, obs = case['task'], case['observations']
            if task == 'causal':
                query = {'nodes': sorted(obs[0]['effects']), 'observations': obs}
            elif task == 'voxel':
                query = {'top_view': obs, 'max_height': 3}
            else:
                query = {'variables': ['x', 'y', 'z'], 'max_depth': 2,
                         'operators': ['AND', 'OR', 'NOT'], 'observations': obs}
            rows.append({'id': case['id'], 'task': task, 'query': query})
        batches[split] = rows
    query_file = out / 'query_only_inputs.json'
    query_file.write_text(json.dumps(batches, indent=2))
    protocol = {
        'reference_outputs_per_case': 6, 'batch_output_budget': '6 times question count',
        'policies': ['fixed_repeat', 'fixed_stop', 'shared_round_robin'],
        'order_variants': ['native', 13, 29, 47, 71, 97],
        'batch_time_limit_seconds': 30,
        'inference': 'separate process; no gold/pool/space sizes; audited JSON reads',
        'cost': 'same final output cap; CPU compute measured, not assumed equal; no LLM calls',
        'metrics': 'emitted precision/uniqueness; unique valid per shared budget; recovery; fixed6 rates only for fixed policies',
        'query_input_sha256': hashlib.sha256(query_file.read_bytes()).hexdigest(),
        'source_sha256': hashes,
        'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for directory in ['budget_v1', 'certified_generation_v1', 'code', 'vendor']
                        for p in sorted((ROOT / directory).rglob('*.py'))
                        if not p.name.startswith('._')},
        'dataset_status': 'previously used internal cases, not a new blind holdout',
        'not_official_fixed_quota_result': True,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('Prepared query-only inputs:', {k: len(v) for k, v in batches.items()})


if __name__ == '__main__':
    main()
