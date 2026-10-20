import argparse

import torch
from transformers import AutoTokenizer

from modeling_llada import LLaDAModelLM
from modeling_dream import DreamModelWrapper
from decoding import decoding_wino

MODELS = {
    'llada': ('GSAI-ML/LLaDA-8B-Instruct', LLaDAModelLM),
    'dream': ('Dream-org/Dream-v0-Instruct-7B', DreamModelWrapper),
}


def main():
    parser = argparse.ArgumentParser(description="Generate one answer with Fast-dLLM (--threshold_back 0) or WINO.")
    parser.add_argument('--model', choices=MODELS, default='llada')
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.6, help="unmasking threshold (lambda1)")
    parser.add_argument('--threshold_back', type=float, default=0.8, help="remasking threshold (lambda2); 0 disables remasking (Fast-dLLM)")
    args = parser.parse_args()

    device = 'cuda:0'
    model_path, model_cls = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model_cls.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()

    prompt = 'Victor purchased a container of gumballs. He gave 4 to Todd, then he gave twice as many as he had given Todd to Alisha, and then he gave 5 less than four times as many to Bobby as he had given to Alisha. If Victor had 6 gumballs remaining, what is the total number of gumballs that Victor purchased?'
    m = [{"role": "user", "content": prompt + '\nPlease think step by step.'}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336  # LLaDA's tokenizer leaves it unset

    out, steps, stats = decoding_wino(model, input_ids, gen_length=args.gen_length, block_length=args.block_length,
                                      temperature=0., threshold=args.threshold, threshold_back=args.threshold_back,
                                      mask_id=mask_id, logit_shift=(args.model == 'dream'))
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print(f'\nSteps (NFEs): {steps} | remask stats: {stats}')


if __name__ == '__main__':
    main()
