import os
import json
import random
import argparse
import tqdm
from datasets import load_dataset
import torch
import yaml
import jsonlines
from transformers import AutoTokenizer
from modeling_llada import LLaDAModelLM
from modeling_dream import DreamModelWrapper
from decoding import decoding_wino, decoding_faithful, decoding_original_saber, decoding_cover
import dataset_utils
from human_eval.evaluation import evaluate_functional_correctness
import tempfile
import numpy as np
from dataset_utils.eval_correctness_mbpp.evaluation import evaluate_functional_correctness as evaluate_functional_correctness_mbpp

def print_nv_info():
    # print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))
    # print(torch.cuda.memory_summary())
    print('free mem from mem_get_info', torch.cuda.mem_get_info())
    print('reserved_bytes.all.peak', torch.cuda.memory_stats()['reserved_bytes.all.peak'])

def set_seed(seed):
    """Seed every RNG the generation path can touch (gumbel sampling in decoding.py, dataset shuffles).
    Called once at startup and again right before each eval loop, so the loop is reproducible regardless
    of how much RNG the warm-up run consumed. Note: this does not force deterministic CUDA kernels."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_overrides(config, overrides):
    """Apply `--set a.b.c=value` overrides (values parsed as YAML) and tag config_name with them,
    so runs of a sweep write to distinct results files."""
    for item in overrides or []:
        key, sep, raw = item.partition('=')
        if not sep:
            raise ValueError(f"--set expects key=value, got '{item}'")
        value = yaml.safe_load(raw)
        node = config
        *parents, leaf = key.split('.')
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
        config['config_name'] += f"_{leaf}{raw}"


def apply_shard(dataset, config):
    """Slice dataset by shard. Returns (dataset, shard_offset) where shard_offset is the global index of dataset[0]."""
    shard = config.get('shard')
    num_shards = config.get('num_shards')
    if shard is None or num_shards is None:
        return dataset, 0
    total = len(dataset)
    s = (total * shard) // num_shards
    e = (total * (shard + 1)) // num_shards
    print(f"==> Shard {shard}/{num_shards}: indices [{s}:{e}] ({e - s} samples)")
    if hasattr(dataset, 'select'):
        dataset = dataset.select(range(s, e))
    else:
        dataset = dataset[s:e]
    return dataset, s

def main():
    parser = argparse.ArgumentParser(description="Unified Config-driven Evaluation Script for Language Models")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file (e.g., configs/llada/gsm8k/wino.yaml)")
    parser.add_argument("--set", nargs='+', default=None, metavar="KEY=VALUE", help="Override config values, e.g. --set generation_args.block_length=128 method_args.wino.threshold=0.7")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based). Splits dataset into --num-shards chunks and runs only this one.")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed. Overrides 'seed' in the config; appended to the results filename as _seed<N>.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    apply_overrides(config, args.set)
    if args.shard is not None and args.num_shards is not None:
        config['shard'] = args.shard
        config['num_shards'] = args.num_shards
    if args.seed is not None:
        config['seed'] = args.seed
    set_seed(config.get('seed'))
    print(f"==> Seed: {config.get('seed')}")
    print(f"==> Config: {args.config}")
    print(f"==> Loaded config for: {config['dataset_name']}")

    model_path = config['model_path']
    model_type = config.get('model_type', 'llada')
    print(f"==> Loading model ({model_type}): {model_path}")
    if config['method'] == 'cover':
        # COVER needs a model whose attention exposes the per-block seed-KV hooks the COVER generator
        # drives. Two separate backends: the KV-cache LLaDA, and the KV-cache Dream (whose custom
        # DreamAttention adds the shifted-logit off-diagonal corrections). Lazy imports keep the plain
        # decoding path from importing either model.
        if model_type == 'dream':
            from modeling_dream_cover import DreamModel as CoverDreamModel
            model = CoverDreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        else:
            from modeling_llada_kv_cover import LLaDAModelLM as CoverLLaDAModelLM
            model = CoverLLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    elif model_type == 'llada':
        model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    else:
        model = DreamModelWrapper.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if config['method'] == 'cover':
        # decoding_cover reads eos_token_id off this.
        model._cover_tokenizer = tokenizer
        # Selects the LLaDA vs Dream code path in decoding_cover without threading config through.
        model._cover_backend = 'dream' if model_type == 'dream' else 'llada'
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        if model_type == 'llada':
            mask_id = 126336
        else:
            raise ValueError(f"tokenizer.mask_token_id is None for model_type='{model_type}' — set it explicitly in the tokenizer config")
    print(f"==> mask_token_id: {mask_id} | pad_token_id: {tokenizer.pad_token_id}")
    run_single_task_evaluation(config, model, tokenizer, mask_id)


def get_generation_function(method_name):
    if method_name == 'wino': return decoding_wino
    elif method_name == 'faithful': return decoding_faithful
    elif method_name == 'original_saber': return decoding_original_saber
    elif method_name == 'cover': return decoding_cover
    else: raise ValueError(f"Unknown method: {method_name}")

def run_single_task_evaluation(config, model, tokenizer, mask_id):
    dataset_name = config['dataset_name']
    print(f"==> Running Single-Task Evaluation for {dataset_name}")
    dataset_cfg = config['dataset_config']
    doc_to_text_fn = getattr(dataset_utils, dataset_cfg['doc_to_text_fn'])
    extract_answer_fn = getattr(dataset_utils, dataset_cfg.get('extract_answer_fn')) if dataset_cfg.get('extract_answer_fn') else None
    is_correct_fn = getattr(dataset_utils, dataset_cfg.get('is_correct_fn')) if dataset_cfg.get('is_correct_fn') else None

    gen_cfg = config['generation_args']
    method_name = config['method']
    method_params = {**config.get('method_args', {}).get(method_name, {}), 'mask_id': mask_id}
    if config.get('model_type') == 'dream' and method_name != 'cover':
        method_params['logit_shift'] = True
    generation_fn = get_generation_function(method_name)
    print(f"==> Method: {method_name} | gen_args: {gen_cfg} | method_params: {method_params}")
    print(f"==> Loading dataset...")
    loader_type = dataset_cfg.get('data_loader', 'huggingface')
    dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
    
    if loader_type == 'huggingface':
        dataset_name_hf = dataset_cfg['load_dataset_args'].get('name')
        dataset = load_dataset(dataset_path, dataset_name_hf, trust_remote_code=True)[dataset_cfg['split']]
    elif dataset_name == 'mbpp':
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = list(loader_fn(dataset_path))
    # elif loader_type == 'custom_loader':
    else:
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = loader_fn(dataset_path)

    num_samples = config.get('num_samples')
    if num_samples is not None:
        print(f"==> Evaluating on the first {num_samples} samples.")
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(num_samples))
        else:
            dataset = dataset[:num_samples]
    dataset, shard_offset = apply_shard(dataset, config)

    # ---Warm-up---
    print("==> Performing warm-up run with one sample...")
    warmup_doc = dataset[0]
    trailing_prompt = ""
    if dataset_cfg['eval_mode'] in ['save_only', 'code_execution']:
        context, _, trailing_prompt = doc_to_text_fn(warmup_doc)
    else:
        context, _ = doc_to_text_fn(warmup_doc) 
    context = [{"role": "user", "content": ""}]
    prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt + dataset_cfg.get('prompt_suffix', '')
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    _ = generation_fn(model, input_ids, **gen_cfg, **method_params)   
    print("==> Warm-up complete.")
    # --- Warm-up complete ---

    set_seed(config.get('seed'))  # re-seed so the eval loop is independent of warm-up RNG use
    total_len = len(dataset)
    raw_outputs, correct_count, total_count, total_steps = [], 0, 0, 0
    remask_stats = {'remasked_positions': 0, 'reaccept_flipflop': 0, 'reaccept_diff_token': 0, 'oracle_nfe': 0}
    ff_all = [] if method_params.get('extra_ff_stats') else None
    trailing_prompt = ""
    for i in tqdm.tqdm(range(total_len), desc=f"Evaluating {dataset_name} with method '{method_name}'"):
        doc = dataset[i]
        
        gt_doc, gt_for_eval, trailing_prompt = None, None, ""
        if dataset_cfg['eval_mode'] in ['save_only', 'code_execution']:
            context, gt_doc, trailing_prompt = doc_to_text_fn(doc)
        else:
            context, gt_for_eval = doc_to_text_fn(doc)
        
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt + dataset_cfg.get('prompt_suffix', '')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        result = generation_fn(model, input_ids, **gen_cfg, **method_params)
        if len(result) == 3:
            gen_output, steps, remask_result = result
            for k in remask_stats:
                remask_stats[k] += remask_result.get(k, 0)
        else:
            gen_output, steps = result
        if isinstance(steps, list):
            steps = steps[0]
        gen_str = tokenizer.batch_decode(gen_output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        print_nv_info()

        total_steps += steps
        
        pred, is_correct = None, "N/A"
        result_item = {'index': i + shard_offset, 'full_response': gen_str, 'steps': steps}

        if dataset_cfg['eval_mode'] == 'binary':
            pred = extract_answer_fn(gen_str)
            is_correct = is_correct_fn(pred, gt_for_eval)
            if is_correct: correct_count += 1
            result_item.update({'prediction': pred, 'is_correct': is_correct})
        elif dataset_cfg['eval_mode'] == 'save_only':
            if dataset_name == 'mbpp':
                gen_str = f"```python\n" + gen_str
                gen_code = extract_answer_fn(gen_str, doc['entry_point'])
                result_item['completion'] = gen_code
            else:
                pred = extract_answer_fn(gen_str, doc)
                result_item['completion'] = pred
            if 'task_id' in gt_doc: result_item['task_id'] = gt_doc['task_id']
            
        raw_outputs.append(result_item)
        if ff_all is not None:
            ff_entry = {
                'dataset_idx': i,
                'is_correct': is_correct,
                'remasks': remask_result.get('ff_records', []) if len(result) == 3 else [],
            }
            if 'task_id' in result_item:
                ff_entry['task_id'] = result_item['task_id']
            ff_all.append(ff_entry)

    final_metrics = {}
    code_results_path = None
    output_formatter = config['dataset_config'].get('output_formatter', 'default')
    if output_formatter == 'humaneval':
        if evaluate_functional_correctness is None:
            print("Warning: 'human-eval' library not found. Skipping functional correctness evaluation.")
        else:
            print("\n==> Generations complete. Calling official evaluation script...")
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as temp_f:
                for item in raw_outputs:
                    temp_f.write(json.dumps(item) + "\n")
                    temp_file_path = temp_f.name
            if 'humaneval' in dataset_name:
                # Partial runs (num_samples / shard) only cover some problems; score just those.
                partial = config.get('num_samples') is not None or config.get('shard') is not None
                final_metrics = evaluate_functional_correctness(temp_file_path, ignore_incomplete=partial)
            elif 'mbpp' in dataset_name:
                problem_file = config['dataset_config'].get('problem_file', 'default')
                final_metrics = evaluate_functional_correctness_mbpp(temp_file_path,problem_file=problem_file,is_mbpp=True)
            code_results_path = temp_file_path + "_results.jsonl"
        os.unlink(temp_file_path)
    if any(v > 0 for v in remask_stats.values()):
        print(f"\n==> Remask stats over {total_len} samples: {remask_stats}")
    save_results(config, raw_outputs, correct_count, total_count, total_steps, total_len, external_metrics=final_metrics, ff_all=ff_all, code_results_path=code_results_path)


def save_results(config, raw_outputs, correct_count, total_count, total_steps, total_len, external_metrics, ff_all=None, code_results_path=None):
    eval_mode = config['dataset_config']['eval_mode']
    if external_metrics:
        accuracy = external_metrics.get('pass@1', next(iter(external_metrics.values()), "N/A"))
    else:
        accuracy = "N/A"
        if eval_mode == 'binary':
            accuracy = correct_count / total_len if total_len > 0 else 0

    avg_steps = total_steps / total_len if total_len > 0 else 0

    parts = [config['config_name']]
    if config.get('method_args', {}).get(config.get('method', ''), {}).get('equal_shadow_remask', False):
        parts.append('_esr')
    if config.get('method_args', {}).get(config.get('method', ''), {}).get('disable_max_accept', False):
        parts.append('_noma')
    max_remask_val = config.get('method_args', {}).get(config.get('method', ''), {}).get('max_remask')
    if max_remask_val is not None:
        parts.append(f'_mr{max_remask_val}')
    if config.get('num_samples') is not None:
        parts.append(f"_n{config['num_samples']}")
    if config.get('seed') is not None:
        parts.append(f"_seed{config['seed']}")
    if config.get('shard') is not None and config.get('num_shards') is not None:
        parts.append(f"_s{config['shard']}of{config['num_shards']}")
    jsonl_path = f"./results/{''.join(parts)}.jsonl"

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        jsonlines.Writer(f).write_all(raw_outputs)
    print(f"Results saved to {jsonl_path}")
    if ff_all is not None:
        ff_path = jsonl_path.replace(".jsonl", "_ff.jsonl")
        with open(ff_path, "w", encoding="utf-8") as f:
            for rec in ff_all:
                f.write(json.dumps(rec) + "\n")
        print(f"FF stats saved to {ff_path}")
    if code_results_path is not None and os.path.exists(code_results_path):
        import shutil
        dest = jsonl_path.replace(".jsonl", "_code_results.jsonl")
        shutil.copy(code_results_path, dest)
        os.unlink(code_results_path)
        print(f"Per-task code results saved to {dest}")
    print(f"Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"Accuracy: {accuracy}")
    print(f"Average Steps: {avg_steps:.2f}")

if __name__ == "__main__":
    main()
