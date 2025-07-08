#!/usr/bin/env python
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from utils_prompts import format_alpaca

if __name__=='__main__':
    parser = argparse.ArgumentParser("Preprocess Alpaca-GPT4 Korean dataset")
    parser.add_argument('--output', required=True, help='Output folder for processed.jsonl')
    args = parser.parse_args()

    ds = load_dataset('FreedomIntelligence/alpaca-gpt4-korean', split='train')
    tokenizer = AutoTokenizer.from_pretrained('unsloth/mistral-7b-v0.3-bnb-4bit')
    eos = tokenizer.eos_token

    ds = ds.map(lambda ex: format_alpaca(ex, eos), batched=True)
    ds.to_json(f"{args.output}/processed.jsonl", orient='records', lines=True)