"""Method-level failures with arguments, schemas, Python cells and neighboring calls.

New transfer episodes have exact host-side cell boundaries. Historical receipt
matches and approximate host-clock links are labelled separately; ambiguity stays
explicit. Mechanisms label observed failures, not a guessed sole episode cause.
"""
import argparse
import ast
import collections
import csv
import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator
from analyze_trajectories import jsonl
from export_analysis_tables import ROOT, write_json


def stage(call):
    name = call['name']
    if name == 'move_to':
        return 'gripper_command' if 'gripper' in call.get('arguments', {}) else 'cartesian_motion'
    return {'observe': 'observation', 'segment': 'segmentation', 'point': 'point_localization',
            'mark_point': 'point_localization', 'select_detection': 'instance_binding',
            'measure_object_geometry': 'geometry_measurement', 'inspect_contact_geometry': 'contact_pose_check',
            'inspect_release_geometry': 'release_pose_check', 'preview_motion': 'motion_precheck',
            'check_motion': 'motion_precheck', 'release': 'release', 'recover': 'motion_recovery',
            'transform_pose': 'coordinate_transform', 'hold': 'hold'}.get(name, name)


def mechanism(call):
    msg = call.get('message') or ''
    category = call.get('failure_category')
    tests = [
        ('destination_is_target', 'target_destination_same_surface'),
        ('identity_point_does_not_match', 'point_mask_instance_mismatch'),
        ('stale detection', 'stale_visual_geometry'),
        ('recorded_motion_is_stale', 'stale_recovery_precondition'),
        ('insufficient measured selected surface', 'insufficient_visible_geometry'),
        ('insufficient', 'insufficient_geometry_evidence'),
        ('None is not of type', 'null_argument_schema_violation'),
        ('unexpected', 'unsupported_argument'),
        ('additional properties', 'unsupported_argument'),
        ('requested_cartesian_path_ik_unavailable', 'cartesian_path_ik_unavailable'),
        ('joint_limit', 'joint_limit_guard'),
        ('self_collision', 'self_collision_guard'),
        ('load_budget', 'force_load_budget_stop'),
        ('emergency_limit', 'force_emergency_stop'),
        ('closure_force_increase', 'gripper_closure_force_stop'),
        ('contact_stall', 'contact_stall_stop'),
        ('contact_detected', 'contact_guard_stop'),
        ('tracking_error', 'tracking_error_stop'),
        ('motion_stalled', 'motion_stalled'),
        ('duration_exhausted', 'motion_duration_exhausted'),
        ('horizon exhausted', 'simulator_horizon_exhausted'),
    ]
    for token, label in tests:
        if token.lower() in msg.lower():
            return label
    return category or 'unclassified'


def cells(episode):
    starts, result, definitions = {}, [], {}
    for line, row in enumerate(jsonl(episode/'driver/tool_audit.jsonl'), 1):
        if row.get('tool_name') != 'python':
            continue
        if row.get('type') == 'tool_start':
            starts[row['tool_call_id']] = row
        if row.get('type') != 'tool_end':
            continue
        code = (row.get('args') or {}).get('code', '')
        invoked = set()
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions[node.name] = ast.get_source_segment(code, node)
            invoked = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        except SyntaxError:
            pass
        start = (starts.get(row['tool_call_id']) or {}).get('host_trace') or {}
        end = row.get('host_trace') or {}
        result.append(dict(request_index=row.get('request_index'), tool_call_id=row['tool_call_id'],
            code=code, helper_code={key: definitions[key] for key in sorted(invoked & definitions.keys())},
            python_error=row.get('error'), output=(row.get('result') or {}).get('output', ''),
            before=start.get('call_index_max'), after=end.get('call_index_max'),
            boundary_valid=('boundary_error' not in start and 'boundary_error' not in end
                            and start.get('active_python_tools') == 0 and end.get('active_python_tools') == 1),
            raw_file=str(episode/'driver/tool_audit.jsonl'), raw_line=line))
    return result


def join_cell(call, executed_cells, request_times, server_started):
    clock_request = None
    if server_started is not None:
        at = server_started + call.get('t_s', 0)
        previous = [(index, stamp) for index, stamp in request_times if stamp <= at]
        if previous and all(abs(at-stamp) > .01 for _, stamp in request_times):
            clock_request = previous[-1][0]
    candidates = [c for c in executed_cells if c['boundary_valid'] and c['before'] is not None
                  and c['after'] is not None and c['before'] < call['call_index'] <= c['after']]
    if len(candidates) == 1:
        return candidates[0], 'exact_cell_boundary', []
    if len(candidates) > 1:
        return None, 'ambiguous_overlapping_boundaries', [c['request_index'] for c in candidates]
    receipt = (call.get('receipt') or '')[:300]
    candidates = [c for c in executed_cells if receipt and receipt in c['output']]
    if len(candidates) == 1 and (clock_request is None or candidates[0]['request_index'] == clock_request):
        return candidates[0], 'unique_receipt_match', []
    # Retrospective host-clock alignment is explicitly labelled approximate.
    # Reject request-boundary proximity instead of pretending millisecond-rounded
    # server timestamps give an exact request/cell association.
    if clock_request is not None:
        timed = [c for c in executed_cells if c['request_index'] == clock_request]
        if len(timed) == 1:
            return timed[0], 'approximate_host_clock_interval', []
    return None, 'ambiguous_receipt_match' if candidates else 'unlinked', [c['request_index'] for c in candidates]


def compact(call):
    return {key: call.get(key) for key in ('call_index', 'name', 'arguments', 'success', 'executed', 'failure_category', 'message')}


def analyze(episode, episode_identity=None):
    identity = episode_identity or episode.name
    outcome = json.loads((episode/'episode_result.json').read_text())
    calls = jsonl(episode/'calls.jsonl')
    executed_cells = cells(episode)
    request_times = [(r['index'], r['t']) for r in jsonl(episode/'driver/requests.jsonl') if r.get('t') is not None]
    server_started = json.loads((episode/'result.json').read_text()).get('started_wall_time')
    schemas = {}
    manifest = episode/'capability_manifest.json'
    if manifest.exists():
        schemas = {c['name']: c['arguments_schema'] for c in json.loads(manifest.read_text())['capabilities']}
    failures, methods, python_errors = [], [], []
    for cell in executed_cells:
        if cell['python_error']:
            python_errors.append({key: cell[key] for key in ('request_index', 'tool_call_id', 'code', 'helper_code', 'python_error', 'raw_file', 'raw_line')}
                                 | {'episode_id': identity, 'host_success': outcome['host_final_success']})
    for i, call in enumerate(calls):
        linked, link_method, ambiguous = join_cell(call, executed_cells, request_times, server_started)
        methods.append(dict(method=call['name'], failed=call.get('success') is False,
                            mechanism=mechanism(call) if call.get('success') is False else None,
                            episode_id=identity, host_success=outcome['host_final_success'],
                            call_index=call['call_index'], t_s=call.get('t_s'), elapsed_s=call.get('elapsed_s'),
                            executed=call.get('executed'), consumes_action=call.get('consumes_action'),
                            request_index=linked['request_index'] if linked else None,
                            tool_call_id=linked['tool_call_id'] if linked else None,
                            raw_call_file=str(episode/'calls.jsonl'), raw_call_line=i+1,
                            join_method=link_method))
        if call.get('success') is not False:
            continue
        value = call.get('value') or {}
        schema = schemas.get(call['name'])
        validation_errors = []
        if schema:
            validation_errors = [dict(path=list(e.absolute_path), message=e.message)
                                 for e in Draft202012Validator(schema).iter_errors(call.get('arguments'))]
        if call['name'] in ('preview_motion', 'check_motion'):
            args = call.get('arguments') or {}
            target_schema = schemas.get(args.get('action'))
            if target_schema and isinstance(args.get('arguments'), dict):
                validation_errors.extend(dict(path=['arguments', *e.absolute_path], message=e.message,
                                              validation_scope=f"nested_action:{args['action']}")
                    for e in Draft202012Validator(target_schema).iter_errors(args['arguments']))
        next_physical = next((x for x in calls[i+1:] if x.get('consumes_action')), None)
        diagnostic = dict(episode_id=identity, host_success=outcome['host_final_success'],
            termination=outcome.get('termination_reason'), call_index=call['call_index'], method=call['name'],
            api_stage=stage(call), failure_mechanism=mechanism(call),
            failure_category=call.get('failure_category'), message=call.get('message'),
            failure_detail=value.get('evidence_error') or value.get('reason') or value.get('stopped_reason'),
            arguments=call.get('arguments'), schema_validation_errors=validation_errors,
            executed=call.get('executed'), value_was_null=call.get('value') is None,
            returned_value=call.get('value'), server_diagnostics=call.get('diagnostics'),
            t_s=call.get('t_s'), elapsed_s=call.get('elapsed_s'), consumes_action=call.get('consumes_action'),
            server_failure_stage=(call.get('diagnostics') or {}).get('failure_stage'),
            endpoint_reached=value.get('endpoint_reached'), position_residual_m=value.get('position_residual_m'),
            requested_displacement_m=value.get('requested_displacement_m'), actual_displacement_m=value.get('actual_displacement_m'),
            peak_force_n=value.get('max_observed_force_n'), force_limit_n=value.get('force_limit_n'),
            contact_detector=value.get('contact_detector'), receipt=call.get('receipt'),
            request_index=linked['request_index'] if linked else None,
            tool_call_id=linked['tool_call_id'] if linked else None,
            request_join_method=link_method, ambiguous_candidate_requests=ambiguous,
            code=linked['code'] if linked else None, helper_code=linked['helper_code'] if linked else None,
            python_error_same_cell=linked['python_error'] if linked else None,
            raw_code_file=linked['raw_file'] if linked else None, raw_code_line=linked['raw_line'] if linked else None,
            previous_calls=[compact(x) for x in calls[max(0, i-2):i]],
            next_calls=[compact(x) for x in calls[i+1:i+3]],
            next_physical_call=compact(next_physical) if next_physical else None,
            raw_call_file=str(episode/'calls.jsonl'), raw_call_line=i+1,
            schema_file=str(manifest), root_cause_review='unreviewed; observed local failure, not exclusive episode cause')
        failures.append(diagnostic)
    return failures, methods, python_errors


def dump_csv(path, records, fields):
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list))
                             else row.get(key) for key in fields})
    temp.replace(path)


def export(benchmark, cache, manifest_override=None, out_override=None, dest_override=None):
    if benchmark == 'libero':
        from summarize_batch import read_jobs, episode_dir
        output_root = out_override or ROOT/'runs/libero400'
        episodes = [episode_dir(output_root, job) for job in read_jobs(manifest_override or ROOT/'manifests/libero_400.tsv')]
    else:
        tasks = [json.loads(x) for x in (ROOT/'manifests/simplerenv_1400.jsonl').read_text().splitlines()]
        episodes = [ROOT/'runs/simplerenv1400'/t['name'] for t in tasks]
    for episode in episodes:
        cache_key = str(episode.resolve())
        if cache_key in cache or not (episode/'episode_result.json').is_file():
            continue
        try:
            outcome = json.loads((episode/'episode_result.json').read_text())
            if type(outcome.get('host_final_success')) is bool and outcome.get('termination_reason') not in ('infrastructure_error', 'dry_run'):
                identity = str(episode.relative_to(output_root)) if benchmark == 'libero' and out_override else None
                cache[cache_key] = analyze(episode, identity)
        except (OSError, ValueError):
            continue
    dest = dest_override or ROOT/'analysis'/f'{benchmark}-call-diagnostics'
    dest.mkdir(parents=True, exist_ok=True)
    failures = [row for value in cache.values() for row in value[0]]
    methods = [row for value in cache.values() for row in value[1]]
    python_errors = [row for value in cache.values() for row in value[2]]
    grouped = collections.defaultdict(list)
    for row in methods:
        grouped[row['method']].append(row)
    summary = [dict(method=name, calls=len(rows), failed_calls=sum(r['failed'] for r in rows),
                    failed_call_rate=sum(r['failed'] for r in rows)/len(rows),
                    affected_episodes=len({r['episode_id'] for r in rows if r['failed']}),
                    affected_host_failed_episodes=len({r['episode_id'] for r in rows if r['failed'] and not r['host_success']}),
                    affected_host_successful_episodes=len({r['episode_id'] for r in rows if r['failed'] and r['host_success']}),
                    mechanisms=dict(collections.Counter(r['mechanism'] for r in rows if r['failed'])))
               for name, rows in sorted(grouped.items())]
    dump_csv(dest/'method_summary.csv', summary, ['method', 'calls', 'failed_calls', 'failed_call_rate', 'affected_episodes',
             'affected_host_failed_episodes', 'affected_host_successful_episodes', 'mechanisms'])
    dump_csv(dest/'call_links.csv', methods, ['episode_id', 'host_success', 'call_index', 'method', 'failed',
             'mechanism', 'executed', 'consumes_action', 't_s', 'elapsed_s', 'request_index', 'tool_call_id',
             'join_method', 'raw_call_file', 'raw_call_line'])
    fields = ['episode_id', 'host_success', 'termination', 'call_index', 'method', 'api_stage', 'failure_mechanism',
              'failure_category', 'failure_detail', 'message', 'arguments', 'schema_validation_errors', 'executed', 'value_was_null',
              'server_failure_stage', 'endpoint_reached', 'position_residual_m', 'requested_displacement_m',
              'actual_displacement_m', 'peak_force_n', 'force_limit_n', 'request_index', 'tool_call_id',
              'request_join_method', 'ambiguous_candidate_requests', 'python_error_same_cell',
              'raw_call_file', 'raw_call_line', 'raw_code_file', 'raw_code_line', 'schema_file', 'root_cause_review']
    dump_csv(dest/'failure_calls.csv', failures, fields)
    dump_csv(dest/'python_errors.csv', python_errors, ['episode_id', 'host_success', 'request_index', 'tool_call_id',
             'python_error', 'code', 'helper_code', 'raw_file', 'raw_line'])
    target = dest/'failure_contexts.jsonl'
    temp = target.with_suffix('.jsonl.tmp')
    with temp.open('w') as stream:
        for row in failures:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    temp.replace(target)
    metadata = dict(updated_utc=datetime.now(timezone.utc).isoformat(), benchmark=benchmark, planned=len(episodes),
        completed=len(cache), calls=len(methods), failed_calls=len(failures), python_errors=len(python_errors),
        episodes_with_exact_cell_boundaries=len({r['episode_id'] for r in methods if r['join_method'] == 'exact_cell_boundary'}),
        all_call_join_counts=dict(collections.Counter(r['join_method'] for r in methods)),
        failure_call_join_counts=dict(collections.Counter(r['request_join_method'] for r in failures)),
        mechanism_counts=dict(collections.Counter(r['failure_mechanism'] for r in failures)),
        definitions=['api_stage is mapped from the API operation; it does not pretend to identify grasp/place intent.',
            'failure_mechanism describes the explicit returned error; categories are local events, not exclusive episode causes.',
            'Exact request joins use private cell call-index boundaries. Historical receipt links are labelled separately.',
            'Historical host-clock joins are approximate: same-host wall clocks, rounded call timestamps, no guessed joins within10ms of a response boundary.',
            'Ambiguous/unlinked calls have no guessed request index. Full arguments, return receipts, code and neighbors remain available.',
            'failure_contexts.jsonl retains full returned values and server diagnostics; call_links.csv includes successful calls for recovery analysis.',
            'python_error_same_cell is co-occurrence; error propagation requires code review before causal annotation.'])
    write_json(dest/'dataset_manifest.json', metadata)
    print(json.dumps({k: metadata[k] for k in ('benchmark', 'completed', 'failed_calls', 'failure_call_join_counts')}), flush=True)
    return len(cache) == len(episodes)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark', choices=('libero', 'simplerenv'))
    p.add_argument('--watch', action='store_true')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--dest', type=Path)
    args = p.parse_args()
    destination = args.dest or ROOT/'analysis'/f'{args.benchmark}-call-diagnostics'
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination/'export.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Diagnostic exporter already running')
    cache = {}
    while True:
        if export(args.benchmark, cache, args.manifest, args.out, args.dest) or not args.watch:
            break
        time.sleep(60)
