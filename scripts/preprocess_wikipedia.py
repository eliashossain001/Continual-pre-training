#!/usr/bin/env python
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from utils_prompts import format_wiki

if __name__=='__main__':
    parser = argparse.ArgumentParser("Preprocess Wikipedia Korean dataset")
    parser.add_argument('--output', required=True, help='Output folder for processed.jsonl')
    args = parser.parse_args()

    # Load and sample
    ds = load_dataset('wikimedia/wikipedia', '20231101.ko', split='train')
    ds = ds.train_test_split(train_size=0.01, seed=42)['train']

    # Tokenizer for EOS
    tokenizer = AutoTokenizer.from_pretrained('unsloth/mistral-7b-v0.3-bnb-4bit')
    eos = tokenizer.eos_token

    # Format
    ds = ds.map(lambda ex: format_wiki(ex, eos), batched=True)
    ds.to_json(f"{args.output}/processed.jsonl", orient='records', lines=True)