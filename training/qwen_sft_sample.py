# Public preparation: machine-specific paths / addresses replaced with placeholders.
"""Prepare one exported row using the local Qwen processor, with loss masking.

No weights are loaded and no training is performed. A training loader can call
encode_sample(processor, row, dataset_dir); batch padding must also use -100
for labels. The reference demonstration's assistant reply is always masked.
"""
import argparse
import hashlib
import json
from pathlib import Path


def multimodal_messages(row, dataset_dir):
    images = iter(row['images'])
    result = []
    for message in row['messages']:
        parts = []
        chunks = message['content'].split('<image>')
        for i, chunk in enumerate(chunks):
            if chunk:
                parts.append({'type': 'text', 'text': chunk})
            if i < len(chunks) - 1:
                path = (Path(dataset_dir) / next(images)).resolve()
                assert path.is_file(), path
                parts.append({'type': 'image', 'image': str(path)})
        result.append({'role': message['role'], 'content': parts})
    assert next(images, None) is None
    return result


def encode_sample(processor, row, dataset_dir):
    messages = multimodal_messages(row, dataset_dir)
    options = dict(tokenize=True, return_dict=True, return_tensors='pt',
                   enable_thinking=False, processor_kwargs={'truncation': False})
    full = processor.apply_chat_template(messages, add_generation_prompt=False, **options)
    prefix = processor.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    ids, prefix_ids = full['input_ids'], prefix['input_ids']
    n = prefix_ids.shape[1]
    assert ids.shape[1] > n
    assert (ids[:, :n] == prefix_ids).all(), 'Template prefix mismatch; do not train with a guessed loss boundary'
    labels = ids.clone()
    labels[:, :n] = -100
    full['labels'] = labels
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset')
    ap.add_argument('--model', default='/path/to/workspace/models/Qwen3.5-2B')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--all', action='store_true', help='Validate every exported sample and report context lengths')
    ap.add_argument('--report', type=Path, help='Separate report path when validating another model processor')
    args = ap.parse_args()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    rows = [json.loads(line) for line in (args.dataset / 'all.jsonl').read_text().splitlines()]
    if args.all:
        checks = []
        for row in rows:
            encoded = encode_sample(processor, row, args.dataset)
            kept = encoded['labels'][encoded['labels'] != -100]
            target = processor.tokenizer.decode(kept)
            assert target.startswith(row['messages'][-1]['content']), row['id']
            checks.append({'id': row['id'], 'images': len(row['images']),
                           'input_tokens': encoded['input_ids'].numel(),
                           'supervised_tokens': kept.numel()})
        report = {'samples_checked': len(checks), 'min_input_tokens': min(x['input_tokens'] for x in checks),
                  'max_input_tokens': max(x['input_tokens'] for x in checks),
                  'prompt_and_demo_assistant_masked': True, 'truncation': False, 'samples': checks}
        report['all_jsonl_sha256'] = hashlib.sha256((args.dataset / 'all.jsonl').read_bytes()).hexdigest()
        report['model_processor'] = args.model
        report_path = args.report or args.dataset / 'qwen_processor_all_checks.json'
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))
        return
    row = rows[args.index]
    encoded = encode_sample(processor, row, args.dataset)
    kept = encoded['labels'][encoded['labels'] != -100]
    report = {'sample': row['id'], 'images': len(row['images']),
              'input_tokens': encoded['input_ids'].numel(), 'supervised_tokens': kept.numel(),
              'decoded_supervised_target': processor.tokenizer.decode(kept),
              'tensor_shapes': {k: list(v.shape) for k, v in encoded.items() if hasattr(v, 'shape')},
              'prompt_and_demo_assistant_masked': True, 'truncation': False}
    (args.dataset / 'qwen_processor_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
