"""Completed-only CSV exports for the authorized SimplerEnv condition-balanced run."""
import argparse
import collections
import csv
import fcntl
import json
import time
from datetime import datetime, timezone

from export_analysis_tables import ROOT, FIELDS, extract, write_json

EXTRA = ['setting', 'group', 'env_name', 'condition_id', 'condition_trial', 'official_episode_key',
         'condition_available_initializations', 'official_initialization_repeat', 'official_aggregate',
         'horizon_success', 'ever_success', 'verdict_agrees']


def export(cache):
    rows = [json.loads(x) for x in (ROOT/'manifests/simplerenv_1400.jsonl').read_text().splitlines()]
    out = ROOT/'runs/simplerenv1400'
    dest = ROOT/'analysis/simplerenv-tables'
    dest.mkdir(parents=True, exist_ok=True)
    pending, invalid = [], []
    for task in rows:
        name = task['name']
        if name in cache:
            continue
        episode = out/name
        try:
            result = json.loads((episode/'episode_result.json').read_text())
            if result.get('termination_reason') in ('infrastructure_error', 'dry_run') or type(result.get('host_final_success')) is not bool:
                invalid.append(name)
                continue
            evaluation = json.loads((episode/'simplerenv/evaluation.json').read_text())
            tables = extract(dict(suite='simplerenv', task_id=0, seed=task['condition_trial']), episode, result)
            for records in tables.values():
                for r in records:
                    r['episode_id'] = name
            r = tables['episodes'][0]
            r.update({key: task.get(key) for key in EXTRA if key in task})
            horizon = evaluation.get('horizon_eval')
            r.update(horizon_success=horizon.get('success') if horizon is not None else None,
                     ever_success=evaluation.get('ever_success'),
                     verdict_agrees=bool(evaluation['eval'].get('success')) == result['host_final_success'])
            cache[name] = tables
        except (OSError, ValueError):
            pending.append(name)
    counts = {}
    for key, original in FIELDS.items():
        fields = original + EXTRA if key == 'episodes' else original
        target = dest/(key+'.csv')
        temp = target.with_suffix('.csv.tmp')
        count = 0
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name in sorted(cache):
                for row in cache[name][key]:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
                    count += 1
        temp.replace(target)
        counts[key] = count
    episodes = [tables['episodes'][0] for tables in cache.values()]
    summary = dict(updated_utc=datetime.now(timezone.utc).isoformat(), planned=1400, completed=len(episodes),
        row_counts=counts, pending=pending, excluded_unscored=invalid,
        host_successes=sum(r['host_success'] for r in episodes), strict_successes=sum(r['strict_success'] for r in episodes),
        false_finish=sum(r['false_finish'] for r in episodes), missed_finish=sum(r['missed_finish'] for r in episodes),
        verdict_mismatches=[r['episode_id'] for r in episodes if not r['verdict_agrees']],
        per_condition_completed=dict(collections.Counter(r['condition_id'] for r in episodes)),
        checkpoint='YijiaFan/CaPtain-4B', raw_root=str(out),
        protocol='140 task/condition groups x10; condition-balanced sample of author official invocations; not full official episode grid')
    write_json(dest/'dataset_manifest.json', summary)
    write_json(dest/'data_dictionary.json', dict(tables={**FIELDS, 'episodes': FIELDS['episodes']+EXTRA},
        keys='episode_id; requests/code_cells join on request_index; capability calls/audit join on call_index',
        definitions={'seed': 'Within-condition policy trial index (0–9); official robot/object initializations are in the task JSON and official_episode_key.',
            'host_success': 'Final hidden simulator predicate.', 'strict_success': 'Final host success and model finish().',
            'horizon_success': 'Predicate at official simulator step horizon; blank when never sampled.',
            'ever_success': 'Benchmark success observed at any official control period.',
            'official_initialization_repeat': '0 for first use; positive if cycling the official condition has fewer than ten unique initializations.',
            'failure_events': 'Capability/Python/episode events have different denominators; events may overlap and are not root-cause annotations.',
            'post_success': 'Physical-call-boundary samples; approximate clock alignment, not exact first achieved state.',
            'raw_line': 'One-based JSONL line; full code, outputs, images and receipts remain in raw logs.',
            'blank': 'Missing/not applicable. Seconds for *_s, Unix seconds for t_epoch; input token totals include repeated context.'}))
    print(json.dumps({k: summary[k] for k in ('completed', 'host_successes', 'strict_successes')}), flush=True)
    return len(episodes) == 1400


if __name__ == '__main__':
    lock = (ROOT/'simplerenv-table-export.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('SimplerEnv table exporter already running.', flush=True)
        raise SystemExit(0)
    p = argparse.ArgumentParser()
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    cache = {}
    while True:
        if export(cache) or not args.watch:
            break
        time.sleep(60)
