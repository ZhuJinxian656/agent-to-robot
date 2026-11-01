"""Shared teacher/student interface. Pure encoding; never chooses a robot action."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re

VERSION = 'qwen-native-step-v2.0'
MOVE_CM = (-1, -0.5, 0.5, 1)
ROT_DEG = (-90, -75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90)
PHASES = ('observe', 'approach', 'align', 'descend', 'grasp', 'verify_grasp',
          'lift', 'transport', 'place', 'release', 'verify_goal', 'recover')
PRESETS = ('down', 'down15', 'down30', 'down45', 'down60', 'down75', 'forward')

def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}

def enum(values):
    return {'enum': list(values)}

ACTION_SCHEMA = {'oneOf': [
    obj({'arm': enum(('left','right')), 'op': enum(('move',)),
         'axis': enum(('x','y','z')), 'amount': enum(MOVE_CM)}),
    obj({'arm': enum(('left','right')), 'op': enum(('rotate',)),
         'axis': enum(('roll','pitch','yaw')), 'amount': enum(ROT_DEG)}),
    obj({'arm': enum(('left','right')), 'op': enum(('point',)), 'preset': enum(PRESETS)}),
    obj({'arm': enum(('left','right')), 'op': enum(('gripper',)), 'state': enum(('open','close'))}),
    obj({'arm': enum(('left','right')), 'op': enum(('home',))}),
    obj({'op': enum(('wait',))}),
]}

def schema(mode='reasoned'):
    if mode not in ('reasoned','action_only'):
        raise ValueError('Unknown output mode')
    fields = {} if mode == 'action_only' else {
        'phase': enum(PHASES),
        **{k: {'type':'string','minLength':1,'maxLength':160}
           for k in ('subgoal','evidence','expected_effect')}}
    return obj({**fields,'action': ACTION_SCHEMA})

def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key: '+key)
        out[key] = value
    return out

def parse(text, mode='reasoned'):
    result = json.loads(text, object_pairs_hook=_unique,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    validate(result, mode)
    return result

def validate(reply, mode='reasoned'):
    expected = schema(mode)['required']
    if not isinstance(reply,dict) or set(reply) != set(expected):
        raise ValueError('Wrong response fields')
    if mode == 'reasoned':
        if reply['phase'] not in PHASES:
            raise ValueError('Unknown phase')
        for k in ('subgoal','evidence','expected_effect'):
            if not isinstance(reply[k],str) or not 1 <= len(reply[k]) <= 160 or not reply[k].strip():
                raise ValueError('Invalid annotation: '+k)
    command(reply['action'])
    return reply

def command(action):
    """Exactly one native command, with no clipping, rounding, inference or fallback."""
    if not isinstance(action,dict):
        raise ValueError('Action must be an object')
    op = action.get('op')
    fields = {'move': {'arm','op','axis','amount'}, 'rotate': {'arm','op','axis','amount'},
              'point': {'arm','op','preset'}, 'gripper': {'arm','op','state'},
              'home': {'arm','op'}, 'wait': {'op'}}
    if op not in fields or set(action) != fields[op]:
        raise ValueError('Wrong action fields')
    if op == 'wait':
        return 'wait'
    arm = action['arm']
    if arm not in ('left','right'):
        raise ValueError('Invalid arm')
    if op in ('move','rotate'):
        value = action['amount']
        allowed = MOVE_CM if op == 'move' else ROT_DEG
        axes = ('x','y','z') if op == 'move' else ('roll','pitch','yaw')
        if (type(value) not in (int,float) or not math.isfinite(value)
                or value not in allowed or action['axis'] not in axes):
            raise ValueError('Invalid numeric action')
        return f"{arm} {op} {action['axis']} {value:+g}"
    if op == 'point':
        if action['preset'] not in PRESETS:
            raise ValueError('Invalid preset')
        return f"{arm} point {action['preset']}"
    if op == 'gripper':
        if action['state'] not in ('open','close'):
            raise ValueError('Invalid gripper state')
        return f"{arm} gripper {action['state']}"
    return f'{arm} home'

def native_reply(reply, mode='reasoned'):
    validate(reply,mode)
    return {'scene': reply.get('evidence',''), 'progress': reply.get('phase',''),
            'memory': reply.get('subgoal',''), 'plan': reply.get('expected_effect',''),
            'commands': [command(reply['action'])]}

BASE_PROMPT = '''Control the dual-arm ARX X5 in the supplied simulator. Use only these camera images,
reported robot state, and execution feedback. Select exactly ONE action, then observe again.
Images are head, left wrist, right wrist, in that order. World +x is right in the head view,
+y is away from the robot, +z is up. All translations are relative fingertip-center motion
in CENTIMETERS, not absolute coordinates. Rotation is about a WORLD axis through the
fingertip center in DEGREES. point presets tilt from down toward +y; forward points +y.
The fingertip center is 14.5 cm ahead of the wrist. Wrist white cross marks the fingertip
center; colored world-axis arrows span 5 cm with 1 cm ticks AT FINGERTIP DEPTH. Account for
parallax when an object is farther away. Read table x/y from object bases, not tops.
Left/right bases are approximately (-30,-45)/(30,-45) cm. Prefer the nearby arm. Approximate
x workspaces: left [-60,15], right [-15,60]. Downward poses reach roughly y<=5; forward poses
reach farther, roughly y<=20. These are approximate; observed planner feedback takes precedence.
Table height is given in the state. Keep fingertips at least 1 cm above it. Lift above nearby
objects before lateral travel; normally z>=table+12 cm. Keep grippers at least 15 cm apart.
Opening 1 is open; an empty closed gripper reads about 0.18. Larger opening is contact, not
proof of a secure grasp: inspect the object after a short test lift. Failed commands may move
part way: always use actual reported pose. After IK failure change pose/orientation rather
than repeating it. home uses a joint return and needs clearance; do not use it while holding.
Choose a graspable part and align its narrow width across the jaws. Use 0.5/1 cm corrections
near contact. This dense collection profile permits only 0.5 or 1 cm translation per turn.
Choose a useful physical correction every turn; finish when the native checker succeeds.
No hidden coordinates, invented successful
outcomes, or automatic grasp/placement routines are available. The native checker alone ends
successful episodes; there is no model-issued done action in this experimental contract.
Previous predictions are context, not verified facts. Current images/feedback take precedence.
'''

def system_prompt(mode='reasoned'):
    annotation = ('Use a short observable decision summary: phase, current subgoal, evidence in the'
                  ' supplied observation, expected_effect of the next action, and action. Evidence'
                  ' describes the present; expected_effect is a prediction, never a verified outcome.'
                  ' Each text is at most 160 characters. Do not output private deliberation or a long essay.'
                  if mode == 'reasoned' else 'Output only an action object under the action key.')
    return BASE_PROMPT + '\n' + annotation + '\nReturn JSON matching this schema:\n' + canonical(schema(mode))

def observation(native_messages):
    """Whitelist actor-visible current-turn input; omit demo and unbounded history."""
    current = native_messages[-1]
    if current['role'] != 'user' or not isinstance(current['content'],list):
        raise ValueError('Unexpected native current observation')
    images = [copy.deepcopy(p) for p in current['content'] if p['type']=='image_url']
    text = '\n'.join(p['text'] for p in current['content'] if p['type']=='text')
    if len(images) != 3 or not text.startswith('TASK: '):
        raise ValueError('Need task and exactly three native camera images')
    text = text.split('\n\nYOUR NOTES FROM PREVIOUS TURNS')[0].split('\n\nHISTORY')[0]
    if 'CURRENT STATE:' not in text or 'table top at z =' not in text:
        raise ValueError('Missing native robot state')
    return images, text

def build_messages(native_messages, history=(), mode='reasoned'):
    images, text = observation(native_messages)
    for reply in history:
        validate(reply, mode)
    recent = list(history)[-3:]
    if recent:
        text += '\n\nPREVIOUS OUTPUTS (predictions, not proof of execution):\n' + canonical(recent)
    return [{'role':'system','content':system_prompt(mode)},
            {'role':'user','content':images+[{'type':'text','text':text}]}]

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()
