"""Export a finished V2 rollout with causal labels, exact images and video timestamps.

Raw outcome/quality metadata is separate from model input. This exporter never splits a
large command or invents an intermediate observation. Each SFT row binds one actual request
to its actual teacher response and its actual native execution decision.
"""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

from contract import build_messages, canonical, digest, native_reply, parse

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def sufficient_translation(action,outcome,terminal_success_partial=False):
    if action['op']!='move' or terminal_success_partial:return True
    moved=outcome.get('moved_cm')
    return moved is not None and moved>=0.5*abs(action['amount'])

def export(run,out):
    manifest=read(run/'run.json')
    if manifest.get('phase')!='episode_exited' or not manifest.get('finished_at_utc'):
        raise ValueError('Episode must be finalized before successful SFT export')
    if not manifest.get('native_policy_source_unchanged'):
        raise ValueError('Native policy changed')
    summaries=list((run/'policy').rglob('episode_summary.json'))
    results=list((run/'native').rglob('_result.json'))
    if len(summaries)!=1 or len(results)!=1:raise ValueError('Missing unique native result')
    summary,result=read(summaries[0]),read(results[0])
    if (not summary['success'] or result.get('eval_time') != 1
        or len(result['details']) != 1
        or not all(x['success'] and x['layout_id']==manifest['case']['layout'] for x in result['details'].values())):
        raise ValueError('Native success not verified')
    quality=read(run/'quality_review.json') if (run/'quality_review.json').exists() else {}
    exclusions=set(quality.get('exclude_sft_decisions',[]))
    out.mkdir(parents=True,exist_ok=True)
    history=[];rows=[];index=[];dropped=[];motion_quality_exclusions=[]
    decisions=sorted((run/'policy').rglob('decision_*.json'))
    for dp in decisions:
        d=read(dp);i=d['decision'];folder=run/'current_session'/f'turn_{i:04d}'
        req=read(folder/'request.json');submitted=read(folder/'submitted_response.json')
        raw=read(folder/'response.json');bridge=read(folder/'bridge.json')
        source=read(run/'current_session/native_requests'/f'turn_{i:04d}.json')
        if sha(folder/'request.json')!=submitted['request_sha256'] or raw['text']!=submitted['text']:
            raise ValueError('Unbound or altered teacher response')
        mode=bridge['mode'];teacher=parse(raw['text'],mode)
        expected=build_messages(source['messages'],history,mode)
        if expected!=req['messages'] or digest(expected)!=bridge['model_messages_sha256']:
            raise ValueError('Training-serving input mismatch')
        if digest(source['messages'])!=bridge['native_request_sha256']:
            raise ValueError('Native source changed')
        if native_reply(teacher,mode)!=bridge['native_reply'] or canonical(bridge['native_reply'])!=d['response']:
            raise ValueError('Executed action differs from teacher label')
        history.append(teacher)
        before=int(re.search(r'^steps used: (\d+)$',d['prompt'],re.M)[1])
        # Match the same execution outcome representation as the original native exporter.
        execution=d.get('execution') or {}
        outcomes=execution.get('command_results',[]) if isinstance(execution,dict) else None
        # Exact structure is required, not silently guessed on an unfamiliar runtime.
        if not isinstance(outcomes,list):raise ValueError('Unknown execution structure')
        execution_ok=len(outcomes)==1 and all(x.get('ok') for x in outcomes)
        partial=execution.get('executed',0)<execution.get('requested',0)
        terminal_success_partial=(partial and i==summary['decisions'] and summary['success']
            and summary['termination_reason']=='native' and execution.get('executed',0)>0
            and execution.get('steps_used')==summary['steps_used'] and not execution.get('ik_failed'))
        execution_ok=execution_ok and (not partial or terminal_success_partial)
        # Native `ok` can permit almost no displacement for a 0.5 cm command.
        # Require measured movement for positive movement labels, except a move
        # intentionally interrupted by verified terminal success. This is an
        # export-quality threshold, not a modification of the native scorer.
        if execution_ok and teacher['action']['op']=='move' and not terminal_success_partial:
            moved=outcomes[0].get('moved_cm')
            requested=abs(teacher['action']['amount'])
            if not sufficient_translation(teacher['action'],outcomes[0]):
                execution_ok=False
                motion_quality_exclusions.append({'decision':i,'requested_cm':requested,'actual_cm':moved,
                    'reason':'missing_or_below_half_requested_displacement'})
        keep=i not in exclusions and not d.get('error') and not d.get('command_errors') and execution_ok
        index.append({'decision':i,'frame_before':before,'seconds_before':before/25,
                      'command':bridge['native_reply']['commands'][0],'sft_kept':keep,
                      'terminal_success_partial':terminal_success_partial,
                      'teacher_json':str(folder/'response.txt')})
        if not keep:
            dropped.append(i);continue
        saved=read(folder/'model_messages.json');messages=[];images=[]
        for mi,message in enumerate(saved):
            chunks=[]
            original=req['messages'][mi]['content']
            if isinstance(original,str):original=[{'type':'text','text':original}]
            if len(original)!=len(message['content']):raise ValueError('Message part count changed')
            for pi,part in enumerate(message['content']):
                if part['type']=='text':
                    if part['text']!=original[pi]['text']:raise ValueError('Text changed')
                    chunks.append(part['text']);continue
                src=folder/part['path']
                if sha(src)!=part['sha256']:raise ValueError('Image changed')
                if src.read_bytes()!=base64.b64decode(original[pi]['image_url']['url'].split(',',1)[1],validate=True):
                    raise ValueError('Image differs from actual actor request')
                dest=out/'images'/f"{sha(src)}.jpg";dest.parent.mkdir(exist_ok=True)
                if not dest.exists():shutil.copy2(src,dest)
                images.append(str(dest.relative_to(out)));chunks.append('<image>')
            messages.append({'role':message['role'],'content':''.join(chunks)})
        messages.append({'role':'assistant','content':raw['text']})
        rows.append({'id':f'{run.name}_turn_{i:04d}','episode_id':run.name,
                     'decision':i,'messages':messages,'images':images,
                     'interface_version':bridge['version'],'mode':mode,
                     'layout_sha256':manifest['layout_sha256'],
                     'source_request_sha256':sha(folder/'request.json')})
    if len(decisions)!=len(list((run/'current_session').glob('turn_*'))):
        raise ValueError('Decision/request count mismatch')
    # Only actor response content is supervised; outcome metadata never enters messages.
    (out/'all.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    with (out/'video_index.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
    report={'episode_id':run.name,'native_success':True,'raw_decisions':len(decisions),
            'sft_rows':len(rows),'excluded_decisions':dropped,'quality_review':quality,
            'motion_quality_exclusions':motion_quality_exclusions,
            'minimum_nonterminal_translation_fraction_for_positive_sft':0.5,
            'video_timebase_fps':25,'layout_sha256':manifest['layout_sha256'],
            'all_jsonl_sha256':sha(out/'all.jsonl'),'target_points':100,
            'target_is_not_a_padding_requirement':True,
            'video_integrity_verified':False,'visual_review_completed':False,
            'scope':'Causal input/action audit only; video integrity and visual review are separate required gates.'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();print(json.dumps(export(a.run.resolve(),a.output.resolve()),indent=2))
