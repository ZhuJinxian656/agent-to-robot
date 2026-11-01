"""Extract observable output/code and tool-error patterns, retaining provenance.

Counters describe events, not adjudicated root causes. Responses are preferred over
the compacted final transcript; partial transcripts are explicitly labelled.
"""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(episode):
    result = json.loads((episode/'episode_result.json').read_text())
    responses = jsonl(episode/'driver/responses.jsonl')
    source = 'responses.jsonl' if responses else 'transcript.jsonl'
    items = [item for row in responses for item in row['items']] if responses else jsonl(episode/'driver/transcript.jsonl')
    request_index = [row['request'] for row in responses for _ in row['items']] if responses else [None]*len(items)
    code_items = [(i+1,item) for i,item in enumerate(items)
                  if item.get('type')=='custom_tool_call' and item.get('name')=='python']
    syntax_errors, branches, loops, functions, success_checks = 0,0,0,0,0
    signatures = collections.Counter()
    api_calls = collections.Counter()
    code_index = []
    for item_index,item in code_items:
        code=item.get('input',''); fingerprint=hashlib.sha256(code.encode()).hexdigest()
        entry=dict(item_index=item_index,request=request_index[item_index-1],call_id=item.get('call_id'),sha256=fingerprint,lines=len(code.splitlines()))
        try:
            tree=ast.parse(code)
            nodes=list(ast.walk(tree))
            signature=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
            signatures[signature]+=1
            branches+=sum(isinstance(n,ast.If) for n in nodes)
            loops+=sum(isinstance(n,(ast.For,ast.While)) for n in nodes)
            functions+=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in nodes)
            success_checks+=sum(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)
                                and n.slice.value=='success' for n in nodes)
            for n in nodes:
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='robot':
                    api_calls[n.func.attr]+=1
            entry.update(parseable=True,ast_sha256=signature)
        except SyntaxError as e:
            syntax_errors+=1;entry.update(parseable=False,error=str(e))
        code_index.append(entry)
    calls=jsonl(episode/'calls.jsonl')
    failed=[(i+1,c) for i,c in enumerate(calls) if c.get('success') is False]
    physical=[(i,c) for i,c in enumerate(calls) if c.get('consumes_action')]
    failure_continuations=[]
    for (index,call),(next_index,next_call) in zip(physical,physical[1:]):
        if call.get('success') is not False:
            continue
        same=(call['name']==next_call['name'] and
              json.dumps(call.get('arguments'),sort_keys=True)==json.dumps(next_call.get('arguments'),sort_keys=True))
        failure_continuations.append(dict(failed_call_index=call.get('call_index'),
            failed_name=call['name'],failed_action_executed=call.get('executed'),
            next_physical_call_index=next_call.get('call_index'),next_name=next_call['name'],
            same_name_and_arguments=same,next_success=next_call.get('success'),
            intervening_capabilities=[c['name'] for c in calls[index+1:next_index]]))
    audit=jsonl(episode/'hidden_success_audit.jsonl')
    successful_samples=[row for row in audit if row.get('success') is True]
    stable_start=None
    if result.get('host_final_success') is True:
        for row in reversed(audit):
            if row.get('success') is not True:
                break
            stable_start=row.get('call_index')
    post_success=[c for c in calls if stable_start is not None and c.get('call_index',0)>stable_start]
    outcome_timeline=dict(source='host-only hidden_success_audit.jsonl',available=bool(audit),
        sampling='after executed physical calls; no claim about unsampled intermediate states',
        first_success_call=successful_samples[0].get('call_index') if successful_samples else None,
        final_contiguous_success_start_call=stable_start,
        succeeded_earlier_but_failed_final=bool(successful_samples) and result.get('host_final_success') is False,
        calls_after_final_success_start=len(post_success),
        capability_names_after_final_success_start=dict(collections.Counter(c['name'] for c in post_success)))
    server_result=json.loads((episode/'result.json').read_text()) if (episode/'result.json').exists() else {}
    requests=jsonl(episode/'driver/requests.jsonl')
    stable_call=next((c for c in calls if stable_start is not None and c.get('call_index')==stable_start),None)
    anchor=server_result.get('started_wall_time')
    if stable_call and isinstance(anchor,(int,float)) and requests:
        success_at=anchor+stable_call['t_s']+stable_call['elapsed_s']
        after=[r for r in requests if r.get('t',0)>=success_at]
        finished_at=server_result.get('finished_wall_time')
        outcome_timeline.update(responses_after_final_success=len(after),
            response_indices_after_final_success=[r['index'] for r in after],
            generated_tokens_after_final_success=sum((r.get('usage') or {}).get('output_tokens',0) or 0 for r in after),
            seconds_after_final_success=finished_at-success_at if isinstance(finished_at,(int,float)) else None,
            clock_alignment='Response record time is after generation; success sample is approximated by physical-call start plus elapsed time on the server clock. Includes the finishing response.')
    reasoning=[item for item in items if item.get('type')=='reasoning']
    driver=result.get('driver') or result.get('codex') or {}
    usage=driver.get('usage') or {}
    runtime_audit=jsonl(episode/'driver/tool_audit.jsonl')
    python_ends=[r for r in runtime_audit if r.get('type')=='tool_end' and r.get('tool_name')=='python']
    runtime_errors=[r for r in python_ends if r.get('error')]
    footer=next((r for r in reversed(runtime_audit) if r.get('type')=='footer'),None)
    return dict(episode=str(episode),model=result.get('model'),host_success=result.get('host_final_success'),
                termination=result.get('termination_reason'),source=source,
                requests=driver.get('requests'),compactions=(driver.get('rollout') or {}).get('compactions'),
                input_tokens=usage.get('input_tokens'),output_tokens=usage.get('output_tokens'),
                reasoning_output_tokens=usage.get('reasoning_output_tokens'),
                python_runtime_executions=len(python_ends) if runtime_audit else None,
                tool_audit_complete=footer.get('complete') if footer else None,
                python_runtime_errors_by_class=dict(collections.Counter(r.get('error_class') for r in runtime_errors)),
                python_runtime_error_evidence=[dict(request=r.get('request_index'),tool_call_id=r.get('tool_call_id'),
                    parent_call_id=r.get('parent_call_id'),error_class=r.get('error_class'),
                    error=r.get('error'),environment_error=r.get('environment_error')) for r in runtime_errors],
                hidden_outcome_timeline=outcome_timeline,
                failed_physical_calls_followed_by_another=len(failure_continuations),
                exact_argument_retries_after_physical_failure=sum(c['same_name_and_arguments'] for c in failure_continuations),
                failure_continuations=failure_continuations,
                output_coverage='full per-request output' if responses else 'final transcript only; earlier history may be compacted',
                python_cells=len(code_items),syntax_error_cells=syntax_errors,
                if_statements=branches,loop_statements=loops,function_definitions=functions,
                success_subscripts=success_checks,robot_api_calls_in_code=dict(api_calls),
                exact_ast_repeat_excess=sum(n-1 for n in signatures.values()),
                reasoning_items=len(reasoning),
                reasoning_characters=sum(len(s.get('text','')) for item in reasoning for s in item.get('summary',[])),
                executed_capability_calls=len(calls),failed_capability_calls=len(failed),
                failed_capability_categories=dict(collections.Counter(c.get('failure_category') for _,c in failed)),
                failed_capability_names=dict(collections.Counter(c.get('name') for _,c in failed)),
                failure_evidence=[dict(line=i,call_index=c.get('call_index'),name=c.get('name'),
                                       category=c.get('failure_category'),receipt=c.get('receipt'),message=c.get('message')) for i,c in failed],
                code_index=code_index)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-label',required=True,help='e.g. local reproduction or upstream reference examples')
    args=p.parse_args()
    episodes=[p.parent for p in args.root.rglob('episode_result.json') if '.infra-' not in str(p)]
    rows,excluded=[],[]
    for episode in sorted(episodes):
        result=json.loads((episode/'episode_result.json').read_text())
        if result.get('termination_reason') in ('infrastructure_error','dry_run') or type(result.get('host_final_success')) is not bool:
            excluded.append(str(episode))
        else:
            rows.append(analyze(episode))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(source_label=args.source_label,episodes=rows,excluded_unscored=excluded,
        interpretation='Static code patterns and failed tool events are not causal failure annotations. Repeated code may be intentional.'),indent=2)+'\n')
    print(json.dumps({'source':args.source_label,'episodes':len(rows),'output':str(args.out)}))
