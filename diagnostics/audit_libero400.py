"""Audit actual terminal artifacts before calling the LIBERO reproduction complete."""
import collections
import json
import subprocess
from datetime import datetime, timezone

from summarize_batch import ROOT, read_jobs, episode_dir
from analyze_trajectories import jsonl
from export_analysis_tables import write_json


jobs = read_jobs(ROOT/'manifests/libero_400.tsv')
coverage = collections.defaultdict(list)
for job in jobs:
    coverage[(job['suite'], int(job['task_id']))].append(int(job['seed']))
assert len(jobs) == 400 and len(coverage) == 40
assert all(sorted(seeds) == list(range(11, 21)) for seeds in coverage.values())
completed, pending, issues, warnings = [], [], [], []
for job in jobs:
    episode = episode_dir(ROOT/'runs/libero400', job)
    try:
        result = json.loads((episode/'episode_result.json').read_text())
    except (OSError, ValueError):
        pending.append(str(episode))
        continue
    if result.get('termination_reason') in ('infrastructure_error', 'dry_run') or type(result.get('host_final_success')) is not bool:
        issues.append(dict(episode=str(episode), issue='unscored terminal result'))
        continue
    for key, expected in [('suite', job['suite']), ('task_id', int(job['task_id'])), ('seed', int(job['seed'])), ('model', 'captain-4b')]:
        if result.get(key) != expected:
            issues.append(dict(episode=str(episode), issue='manifest/result mismatch', field=key, actual=result.get(key), expected=expected))
    required = ['result.json', 'calls.jsonl', 'hidden_success_audit.jsonl', 'driver/requests.jsonl',
                'driver/responses.jsonl', 'driver/tool_audit.jsonl']
    missing = [name for name in required if not (episode/name).is_file()]
    if missing:
        issues.append(dict(episode=str(episode), issue='missing raw evidence', files=missing))
        continue
    try:
        requests = jsonl(episode/'driver/requests.jsonl')
        responses = jsonl(episode/'driver/responses.jsonl')
        calls = jsonl(episode/'calls.jsonl')
        audit = jsonl(episode/'hidden_success_audit.jsonl')
        runtime = jsonl(episode/'driver/tool_audit.jsonl')
    except ValueError as error:
        issues.append(dict(episode=str(episode), issue='invalid JSONL', error=str(error)))
        continue
    request_ids = [r['index'] for r in requests]
    response_ids = [r['request'] for r in responses]
    if request_ids != response_ids or len(request_ids) != len(set(request_ids)):
        issues.append(dict(episode=str(episode), issue='request/response coverage mismatch'))
    if len(requests) != (result.get('driver') or {}).get('requests'):
        issues.append(dict(episode=str(episode), issue='driver/request count mismatch'))
    call_ids = {c['call_index'] for c in calls}
    if any(r.get('call_index') not in call_ids for r in audit):
        issues.append(dict(episode=str(episode), issue='hidden audit references absent capability call'))
    footer = next((r for r in reversed(runtime) if r.get('type') == 'footer'), None)
    if not footer or not footer.get('complete'):
        warnings.append(dict(episode=str(episode), warning='runtime audit not marked complete', footer=footer))
    completed.append(str(episode))
clean = {}
for name, path in [('captain', ROOT/'CaPtain'), ('harness', ROOT/'CaPtain/harness')]:
    clean[name] = subprocess.run(['git', 'diff', '--quiet'], cwd=path).returncode == 0
    if not clean[name]:
        issues.append(dict(issue='tracked official checkout edits', checkout=name))
receipt = dict(updated_utc=datetime.now(timezone.utc).isoformat(), planned=400, completed=len(completed),
               manifest_has_40_tasks_x_10_initializations=True, pending=pending, issues=issues, warnings=warnings,
               official_tracked_files_clean=clean,
               ready_for_final_analysis=len(completed) == 400 and not issues,
               protocol='Released CaPtain-4B only; states11–20; paper-matched11–16 reported separately; base Qwen excluded.')
write_json(ROOT/'analysis/libero400-integrity-audit.json', receipt)
print(json.dumps({k: receipt[k] for k in ('completed', 'ready_for_final_analysis')} | {'issues': len(issues), 'warnings': len(warnings)}))
