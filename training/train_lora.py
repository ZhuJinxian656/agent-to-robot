"""Single-task multimodal LoRA; preserve all input tokens and supervise final answer.

Only supervised prediction positions are projected through the large vocabulary
head. This is the same masked causal cross entropy, without materializing logits
for the ~10K ignored prompt positions. No input or image is truncated.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

ROOT = Path(os.environ['LORA_RUN_DIR']).resolve()
MODEL = Path(os.environ['LORA_MODEL_DIR']).resolve()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def event(kind, **fields):
    row = {'at': now(), 'kind': kind, **fields}
    with (ROOT/'logs/metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)
    save(ROOT/'reports/training_status.json', row)


def rows(split):
    return [json.loads(line) for line in (ROOT/'dataset'/f'{split}.jsonl').read_text().splitlines()]


def prepare():
    import torch
    from transformers import AutoProcessor
    from qwen_sft_sample import encode_sample
    torch.set_num_threads(4)
    snapshot = json.loads((ROOT/'reports/dataset_snapshot.json').read_text())
    for name, sha in snapshot['source_files'].items():
        assert hashlib.sha256((ROOT/'dataset'/name).read_bytes()).hexdigest() == sha
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    checks = []
    all_rows = rows('all')
    for index, row in enumerate(all_rows):
        assert row['messages'][0]['role'] == 'system' and row['messages'][-1]['role'] == 'assistant'
        assert row['interface_version'] == 'qwen-native-step-v2.0'
        assert row['mode'] == 'reasoned'
        assert len(row['images']) == 3
        target = json.loads(row['messages'][-1]['content'])
        from contract import validate
        validate(target, 'reasoned')
        encoded = dict(encode_sample(processor, row, ROOT/'dataset'))
        tokens = encoded['input_ids'].shape[1]
        assert tokens <= 4096, 'Do not silently truncate inputs'
        kept = encoded['labels'][encoded['labels'] != -100]
        assert processor.tokenizer.decode(kept).startswith(row['messages'][-1]['content'])
        path = ROOT/'encoded'/f"{row['id']}.pt"
        torch.save(encoded, path)
        checks.append({'id':row['id'], 'input_tokens':tokens,
            'supervised_tokens':kept.numel(), 'images':3, 'encoded':str(path)})
        if (index+1) % 10 == 0:
            event('encoding', completed=index+1, total=len(all_rows))
    result = {'at':now(), 'samples':len(checks), 'min_tokens':min(x['input_tokens'] for x in checks),
        'max_tokens':max(x['input_tokens'] for x in checks), 'only_final_assistant_supervised':True,
        'thinking_enabled':False, 'input_truncation':False, 'checks':checks}
    save(ROOT/'reports/encoding_validation.json', result)
    event('encoding_complete', samples=len(checks), min_tokens=result['min_tokens'], max_tokens=result['max_tokens'])


def train():
    import torch
    import torch.nn.functional as F
    import transformers
    import peft
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, TaskType, get_peft_model

    seed = 20261029
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    # Bound only this training allocator; existing inference and simulator
    # processes are never signalled or modified by this trainer.
    budget_gib = float(os.environ.get('LORA_ALLOCATOR_GIB','14'))
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(budget_gib*1024**3/total, 0)
    validation = json.loads((ROOT/'reports/encoding_validation.json').read_text())
    assert validation['samples'] == len(rows('all')) and validation['max_tokens'] <= 4096

    # Check the causal shift and selective projection loss independently of
    # expensive image/model tensors, including ignored and final positions.
    toy_logits = torch.randn(1,9,17)
    toy_labels = torch.tensor([[-100,-100,-100,-100,4,8,3,2,6]])
    positions = torch.where(toy_labels[0,1:] != -100)[0]
    reference = F.cross_entropy(toy_logits[:,:-1].reshape(-1,17), toy_labels[:,1:].reshape(-1),ignore_index=-100)
    selected = F.cross_entropy(toy_logits[:,positions].reshape(-1,17),toy_labels[:,positions+1].reshape(-1))
    assert torch.allclose(reference,selected,atol=1e-7)

    event('loading_model', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], allocator_budget_gib=budget_gib)
    started = time.monotonic()
    base = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda:0')
    base.config.text_config.use_cache = False
    targets = [name for name, module in base.named_modules()
        if name.startswith('model.language_model.layers.') and isinstance(module,torch.nn.Linear)]
    assert targets and not any('visual' in name or 'lm_head' in name for name in targets)
    config = LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',
        task_type=TaskType.CAUSAL_LM,target_modules=targets)
    model = get_peft_model(base,config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    trainable = [(name,p.numel()) for name,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('.lora_' in name for name,_ in trainable)
    assert not any('visual' in name for name,_ in trainable)
    assert not any(p.requires_grad for name,p in model.named_parameters() if 'visual' in name)
    train_rows, val_rows = rows('train'), rows('validation')
    epochs, accumulation = 3, 8
    total_updates = math.ceil(len(train_rows)/accumulation)*epochs
    optimizer = torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    warmup = max(1, math.ceil(total_updates * .1))

    def lr_scale(step):
        if step < warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_updates-warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
    processor = AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    save(ROOT/'reports/training_config.json',{'at':now(),'model':str(MODEL),'model_revision':
        str(MODEL.resolve()),'dtype':'bfloat16','quantized':False,
        'epochs':epochs,'micro_batch_size':1,'gradient_accumulation':accumulation,
        'total_optimizer_updates':total_updates,'learning_rate':1e-4,'warmup_updates':warmup,
        'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,'targets':targets,
        'vision_frozen':True,'gradient_checkpointing':'non_reentrant','max_input_tokens':4096,
        'interface_version':'qwen-native-step-v2.0','mode':'reasoned',
        'dataset_snapshot':json.loads((ROOT/'reports/dataset_snapshot.json').read_text()),
        'thinking':False,'supervise_final_assistant_only':True,'input_truncation':False,
        'logits_only_at_supervised_prediction_positions':True,'loss_equivalence_test_passed':True,
        'trainable_parameters':sum(n for _,n in trainable),'total_parameters':sum(p.numel() for p in model.parameters()),
        'trainable_modules':trainable,'seed':seed,'versions':{'torch':torch.__version__,
        'transformers':transformers.__version__,'peft':peft.__version__},
        'model_load_seconds':time.monotonic()-started,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'allocator_budget_gib':budget_gib,'closed_loop_improvement_claimed':False})

    def loss_for(row):
        encoded = torch.load(ROOT/'encoded'/f"{row['id']}.pt",map_location='cpu',weights_only=True)
        labels = encoded.pop('labels').to('cuda:0')
        positions = torch.where(labels[0,1:] != -100)[0]
        assert positions.numel()>0
        batch = {key: value.to('cuda:0',dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                 for key,value in encoded.items()}
        output = model(**batch,use_cache=False,logits_to_keep=positions)
        targets = labels[:,positions+1]
        loss = F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),targets.reshape(-1))
        return loss,targets.numel()

    def evaluate(label):
        model.eval();total_loss=0.;tokens=0
        with torch.no_grad():
            for val_index, row in enumerate(val_rows):
                loss,count=loss_for(row)
                assert torch.isfinite(loss)
                total_loss += float(loss)*count;tokens += count
                if (val_index+1)%32==0:
                    event('validation_progress',label=label,completed=val_index+1,total=len(val_rows))
        value=total_loss/tokens
        event('validation',label=label,loss=value,supervised_tokens=tokens,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        model.train()
        return value

    def checkpoint(name,extra):
        directory=ROOT/'checkpoints'/name
        model.save_pretrained(directory,safe_serialization=True)
        processor.save_pretrained(directory)
        save(directory/'training_metadata.json',extra)
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate()},directory/'training_state.pt')
        assert (directory/'adapter_config.json').is_file() and (directory/'adapter_model.safetensors').is_file()

    baseline=evaluate('base_before_training')
    best=float('inf');update=0;micro=0;first_verified=False;start=time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1,epochs+1):
        order=list(train_rows);random.Random(seed+epoch).shuffle(order)
        save(ROOT/'reports'/f'epoch_{epoch}_order.json',[row['id'] for row in order])
        for offset in range(0,len(order),accumulation):
            group=order[offset:offset+accumulation]
            for row in group:
                before=time.monotonic();loss,_=loss_for(row)
                assert torch.isfinite(loss),row['id']
                (loss/len(group)).backward();torch.cuda.synchronize();micro+=1
                event('train_microbatch',epoch=epoch,microbatch=micro,planned_microbatches=len(train_rows)*epochs,
                    sample_id=row['id'],loss=float(loss.detach()),seconds=time.monotonic()-before,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    allocated_gib=torch.cuda.memory_allocated()/1024**3)
                del loss
            grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            assert float(grad_norm)>0,'No trainable gradient'
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);update+=1
            seconds=time.monotonic()-start
            event('optimizer_step',epoch=epoch,update=update,total_updates=total_updates,
                microbatches=micro,gradient_norm=float(grad_norm),learning_rate=scheduler.get_last_lr()[0],
                elapsed_training_seconds=seconds,remaining_training_seconds_estimate=
                seconds/micro*(len(train_rows)*epochs-micro))
            if not first_verified:
                changed={name:float(p.detach().abs().max()) for name,p in model.named_parameters()
                    if p.requires_grad and '.lora_B.' in name}
                assert max(changed.values())>0
                check={'at':now(),'finite_backward':True,'finite_optimizer_step':True,
                    'gradient_norm':float(grad_norm),'nonzero_lora_B':True,
                    'trainable_parameters':sum(n for _,n in trainable),
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'update':update}
                save(ROOT/'reports/first_optimizer_step_verified.json',check)
                checkpoint('first_step',check);first_verified=True
        value=evaluate(f'epoch_{epoch}')
        metadata={'epoch':epoch,'optimizer_updates':update,'validation_loss':value,'baseline_validation_loss':baseline}
        checkpoint(f'epoch_{epoch}',metadata)
        if value<best:
            best=value;checkpoint('best_validation',metadata)
    checkpoint('final',{'epoch':epochs,'optimizer_updates':update,'baseline_validation_loss':baseline,
        'best_validation_loss':best,'closed_loop_evaluation':'not yet performed'})
    event('complete',optimizer_updates=update,microbatches=micro,
        baseline_validation_loss=baseline,best_validation_loss=best,
        final_adapter=str(ROOT/'checkpoints/final'),best_adapter=str(ROOT/'checkpoints/best_validation'),
        training_seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        closed_loop_evaluation_performed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    try:
        prepare() if args.prepare else train()
    except BaseException as error:
        event('failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
