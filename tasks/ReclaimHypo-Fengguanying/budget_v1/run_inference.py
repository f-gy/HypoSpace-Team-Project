"""Only public task code and query-only inputs are used in this process."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORDERS = ['native', 13, 29, 47, 71, 97]
POLICIES = ['fixed_repeat', 'fixed_stop', 'shared_round_robin']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=pathlib.Path)
    out = parser.parse_args().run_dir.resolve()
    input_file = out / 'query_only_inputs.json'
    read_json = set()

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        flags = args[2]
        readable = (flags & os.O_ACCMODE) != os.O_WRONLY
        if readable and path.suffix.lower() in {'.json', '.jsonl'}:
            if ROOT == path or ROOT in path.parents:
                if path != input_file:
                    raise RuntimeError('Inference attempted disallowed data read: ' + str(path))
                read_json.add(str(path))

    sys.addaudithook(audit)
    from scheduler import allocate, backend
    data = input_file.read_bytes()
    batches = json.loads(data)
    # Warm public grammar once so policy timings share the same cache state.
    start = time.monotonic()
    backend._public_boolean(('x', 'y', 'z'), 2, ('AND', 'NOT', 'OR'))
    grammar_warmup_seconds = time.monotonic() - start
    results = []
    for split, original in batches.items():
        for seed in ORDERS:
            cases = list(original)
            if seed != 'native':
                cases.sort(key=lambda row: hashlib.sha256((str(seed) + ':' + row['id']).encode()).digest())
            for policy in POLICIES:
                result = allocate(cases, policy, per_case=6, time_limit=30)
                results.append({'split': split, 'order': seed,
                                'question_order': [q['id'] for q in cases], **result})
    result_file = out / 'inference.json'
    with result_file.open('x') as handle:
        json.dump({'query_input_sha256': hashlib.sha256(data).hexdigest(),
                   'audited_root_json_reads': sorted(read_json),
                   'only_query_file_read': read_json == {str(input_file)},
                   'llm_calls': 0, 'grammar_warmup_seconds': grammar_warmup_seconds,
                   'batches': results}, handle, indent=2)
    print('Inference complete:', len(results), 'batches; only query JSON read')


if __name__ == '__main__':
    main()
