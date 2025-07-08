#!/usr/bin/env python
import argparse
import yaml
from pathlib import Path
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Instruction Finetuning with UnslothTrainer"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the InstFT config YAML (e.g. config/inst_ft.yaml)"
    )
    args = parser.parse_args()

    # 1️⃣ Load config
    cfg = yaml.safe_load(open(args.config, "r"))

    # 2️⃣ Resolve model directory: pick the latest checkpoint if parent folder given
    model_dir = Path(cfg["model"])
    if model_dir.is_dir():
        ckpts = sorted(
            [d for d in model_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[-1])
        )
        if ckpts:
            model_dir = ckpts[-1]

    # 3️⃣ Load the CPT-adapted base + tokenizer (includes LoRA adapter)
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(model_dir),
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["load_in_4bit"]
    )

    # ──────> **DO NOT** call get_peft_model here, since the adapter is already attached

    # 4️⃣ Load the preprocessed Alpaca-Korean dataset
    ds = load_dataset(
        "json",
        data_files={"train": "data/alpaca_kr/processed.jsonl"}
    )["train"]

    # 5️⃣ Build valid training args (cast numerics properly)
    training_args = UnslothTrainingArguments(
        per_device_train_batch_size = int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"]),
        max_steps = int(cfg["max_steps"]),
        warmup_steps = int(cfg["warmup_steps"]),
        learning_rate = float(cfg["learning_rate"]),
        embedding_learning_rate = float(cfg["embedding_learning_rate"]),
        logging_steps = int(cfg.get("logging_steps", 10)),
        optim = cfg.get("optim", "adamw_8bit"),
        weight_decay = float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type = cfg.get("lr_scheduler_type", "linear"),
        seed = int(cfg.get("random_state", 3407)),
        output_dir = cfg["output_dir"],
        report_to = cfg.get("report_to", "none")
    )

    # 6️⃣ Initialize and run the trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        dataset_text_field = "text",
        max_seq_length = cfg["max_seq_length"],
        dataset_num_proc = int(cfg.get("dataset_num_proc", 8)),
        args = training_args
    )
    trainer.train()

if __name__ == "__main__":
    main()
