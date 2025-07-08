#!/usr/bin/env python
import argparse
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TextStreamer, logging as hf_logging
from utils_prompts import WIKI_PROMPT, ALPACA_PROMPT

def main():
    # 1️⃣ Parse args
    parser = argparse.ArgumentParser("Fast inference with fine-tuned model")
    parser.add_argument("--model_dir", required=True,
                        help="Path to your LoRA-fine-tuned checkpoint folder")
    parser.add_argument("--prompt_type", choices=["wiki","alpaca"], default="alpaca")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="", help="Optional second part of prompt")
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    # 2️⃣ Silence HF warnings
    hf_logging.set_verbosity_error()

    # 3️⃣ Load model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_dir,
        load_in_4bit=False,
        dtype=torch.float16
    )
    FastLanguageModel.for_inference(model)

    # 4️⃣ Build the prompt
    if args.prompt_type == "wiki":
        full_prompt = WIKI_PROMPT.format(args.instruction, args.input)
    else:
        full_prompt = ALPACA_PROMPT.format(args.instruction, args.input)

    # 5️⃣ Tokenize
    inputs = tokenizer(full_prompt + tokenizer.eos_token,
                       return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    # 6️⃣ Generate with streaming disabled (we'll stream manually)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        use_cache=True
    )

    # 7️⃣ Slice off the prompt tokens and decode only the generated part
    gen_ids = outputs[0][prompt_len:]
    generated_text = tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print(generated_text)


if __name__ == "__main__":
    main()
