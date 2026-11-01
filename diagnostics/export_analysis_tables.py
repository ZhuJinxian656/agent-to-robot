"""Export completed manifest episodes without changing model-visible evaluation.

Raw logs remain authoritative. Failure events describe observations, not root causes.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from summarize_batch import ROOT, read_jobs, episode_dir
from analyze_trajectories import analyze, jsonl


FIELDS = {
    'episodes': 'episode_id suite task_id seed checkpoint host_success agent_finished strict_success false_finish missed_finish termination wall_s requests action_requests failed_actions input_tokens output_tokens reasoning_output_tokens compactions python_cells syntax_error_cells python_runtime_errors failed_capability_calls exact_argument_retries_after_physical_failure first_success_call final_contiguous_success_start_call responses_after_final_success seconds_after_final_success calls_after_final_success_start raw_directory',
    'requests': 'episode_id request_index status latency_s attempts input_tokens output_tokens cached_input_tokens reasoning_output_tokens t_epoch replayed raw_file raw_line response_file',
    'tool_calls': 'episode_id call_index name success executed consumes_action failure_category message receipt t_s elapsed_s arguments_json raw_file raw_line',
    'code_cells': 'episode_id request_index call_id item_index sha256 ast_sha256 lines parseable syntax_error raw_file',
    'failure_events': 'episode_id event_level category request_index call_index tool_call_id name message raw_file raw_line',
    'success_audit': 'episode_id action_index call_index capability environment_step success error raw_file raw_line',
}
FIELDS = {k: v.split() for k, v in FIELDS.items()}


def extract(job, episode, result):
    identity = f"{job['suite']}_{job['task_id']}_seed-{job['seed']}"
    a = analyze(episode)
    t = a['hidden_outcome_timeline']
    host, finished = result['host_final_success'], bool(result.get('agent_finished'))
    tables = {k: [] for k in FIELDS}
    row = dict(episode_id=identity, suite=job['suite'], task_id=int(job['task_id']),
               seed=int(job['seed']), checkpoint='YijiaFan/CaPtain-4B',
               host_success=host, agent_finished=finished, strict_success=host and finished,
               false_finish=finished and not host, missed_finish=host and not finished,
               termination=result.get('termination_reason'), wall_s=result.get('wall_time_s'),
               action_requests=result.get('action_requests'), failed_actions=result.get('failed_actions'),
               python_runtime_errors=len(a['python_runtime_error_evidence']), raw_directory=str(episode))
    for key in FIELDS['episodes']:
        if key in a:
            row[key] = a[key]
        elif key in t:
            row[key] = t[key]
    tables['episodes'].append(row)
    for line, r in enumerate(jsonl(episode/'driver/requests.jsonl'), 1):
        usage = r.get('usage') or {}
        tables['requests'].append(dict(episode_id=identity, request_index=r['index'],
            status=r.get('status'), latency_s=r.get('latency_s'), attempts=r.get('attempts'),
            input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
            cached_input_tokens=(usage.get('input_tokens_details') or {}).get('cached_tokens'),
            reasoning_output_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens'),
            t_epoch=r.get('t'), replayed=r.get('replayed'),
            raw_file=str(episode/'driver/requests.jsonl'), raw_line=line,
            response_file=str(episode/'driver/responses.jsonl')))
    for line, c in enumerate(jsonl(episode/'calls.jsonl'), 1):
        row = {k: c.get(k) for k in FIELDS['tool_calls'] if k in c}
        row.update(episode_id=identity, arguments_json=json.dumps(c.get('arguments'), ensure_ascii=False),
                   raw_file=str(episode/'calls.jsonl'), raw_line=line)
        tables['tool_calls'].append(row)
        if c.get('success') is False:
            tables['failure_events'].append(dict(episode_id=identity, event_level='capability',
                category=c.get('failure_category') or 'unclassified', call_index=c.get('call_index'),
                name=c.get('name'), message=c.get('message'), raw_file=row['raw_file'], raw_line=line))
    for c in a['code_index']:
        row = {k: c.get(k) for k in FIELDS['code_cells'] if k in c}
        row.update(episode_id=identity, request_index=c.get('request'), syntax_error=c.get('error'),
                   raw_file=str(episode/'driver'/a['source']))
        tables['code_cells'].append(row)
    for line, c in enumerate(jsonl(episode/'driver/tool_audit.jsonl'), 1):
        if c.get('type') == 'tool_end' and c.get('tool_name') == 'python' and c.get('error'):
            tables['failure_events'].append(dict(episode_id=identity, event_level='python',
                category=c.get('error_class') or 'unclassified', request_index=c.get('request_index'),
                tool_call_id=c.get('tool_call_id'), name='python', message=c.get('error'),
                raw_file=str(episode/'driver/tool_audit.jsonl'), raw_line=line))
    if not host or not finished:
        category = 'false_finish' if finished and not host else 'missed_finish' if host else 'host_failure'
        tables['failure_events'].append(dict(episode_id=identity, event_level='episode', category=category,
            message=result.get('termination_reason'), raw_file=str(episode/'episode_result.json')))
    for line, c in enumerate(jsonl(episode/'hidden_success_audit.jsonl'), 1):
        row = {k: c.get(k) for k in FIELDS['success_audit'] if k in c}
        row.update(episode_id=identity, raw_file=str(episode/'hidden_success_audit.jsonl'), raw_line=line)
        tables['success_audit'].append(row)
    return tables


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def export(args, cache):
    jobs = read_jobs(args.manifest)
    active, pending, excluded = [], [], []
    for job in jobs:
        episode = episode_dir(args.out, job)
        key = str(episode)
        try:
            if key not in cache:
                result = json.loads((episode/'episode_result.json').read_text())
                if result.get('termination_reason') in ('infrastructure_error', 'dry_run') or type(result.get('host_final_success')) is not bool:
                    excluded.append(key)
                    continue
                cache[key] = extract(job, episode, result)
            active.append(key)
        except (OSError, ValueError) as error:
            pending.append(dict(episode=key, reason=type(error).__name__))
    args.dest.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, fields in FIELDS.items():
        path = args.dest/(name+'.csv')
        tmp = path.with_suffix('.csv.tmp')
        counts[name] = 0
        with tmp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key in active:
                for row in cache[key][name]:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
                    counts[name] += 1
        tmp.replace(path)
    write_json(args.dest/'data_dictionary.json', dict(
        tables=FIELDS, join_keys={'all': 'episode_id', 'requests_to_code': 'episode_id + request_index',
            'tool_calls_to_success_audit': 'episode_id + call_index'},
        definitions={
            'seed': '官方初始化状态索引，11–20；不是十次独立采样的随机种子。',
            'host_success': '仿真器最终隐藏判据；只用于主机统计，不暴露给策略。',
            'strict_success': 'host_success 且 agent_finished。',
            'false_finish': '模型报告完成，但最终隐藏判据不满足。',
            'missed_finish': '最终隐藏判据满足，但模型未报告完成。',
            'failure_events': '分 capability/python/episode 层；同一事件可能跨层出现，不能相加当作失败轮数，也不是人工根因标签。',
            'python_runtime_errors': 'Python 工具记录的未捕获错误次数；包括语法错误，与 syntax_error_cells 可能重叠。',
            'exact_argument_retries_after_physical_failure': '失败物理调用后的下一物理调用名称和参数完全相同；不代表盲目重试。',
            'success_audit': '仅执行过的物理调用后采样；不覆盖运动中间状态。',
            'final_contiguous_success_start_call': '最终成功时，隐藏审计末尾连续成功样本的起点；不是精确的首次成功时刻。',
            'responses_after_final_success': '按服务器调用结束时间和请求记录时间近似对齐；包括最后完成回复。',
            'seconds_after_final_success': '上述采样起点至服务器结束的秒数；普通核验也计入，不自动标为失败。',
            't_s': '相对服务器启动的秒数；elapsed_s/latency_s/wall_s 均以秒计。',
            't_epoch': '模型请求记录时间，Unix 秒，生成后、工具执行前。',
            'raw_line': '原始 JSONL 从 1 开始的行号。',
            'blank': 'CSV 空值表示缺失或不适用；布尔值为 True/False。',
            'tokens': 'input_tokens 为每次请求输入累计，包含重复上下文；reasoning_output_tokens 是输出的子集，不额外相加。',
            'call_request_join': '能力调用没有精确 request_index，不推断跨表的调用到请求对应关系。',
        },
        raw_preserved='完整模型输出、代码、工具回执、图像和主机审计保留在 raw_directory；CSV 不复制大型图像/掩码。',
        exclusions='仅指定 400 轮 manifest 的完成且可评分结果；预检、参考示例、基础 Qwen、基础设施错误及其归档均不计成功率。'))
    snapshot = dict(updated_utc=datetime.now(timezone.utc).isoformat(), planned=len(jobs),
        completed=len(active), pending=pending, excluded_unscored=excluded, row_counts=counts,
        checkpoint='YijiaFan/CaPtain-4B', checkpoint_round=300,
        captain_commit='4bb86546196b80aaa2fa4bd0f9fa697aa7580e8e',
        harness_commit='de38d6d63b710caf356320a3c046b72b6d299514',
        raw_root=str(args.out), manifest=str(args.manifest), completed_only=True,
        paper_comparison='全部400轮为初始化11–20；11–16子集共240轮与论文初始化范围相同。',
        snapshot_note='各表原子替换；读取时检查此文件更新时间在读表前后相同。')
    write_json(args.dest/'dataset_manifest.json', snapshot)
    print(json.dumps(dict(completed=len(active), planned=len(jobs), row_counts=counts)), flush=True)
    return len(active) == len(jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'manifests/libero_400.tsv')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/libero400')
    parser.add_argument('--dest', type=Path, default=ROOT/'analysis/tables')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    cache = {}
    while True:
        complete = export(args, cache)
        if complete or not args.watch:
            break
        time.sleep(max(1, args.interval))
