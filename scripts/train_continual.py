#!/usr/bin/env python
import argparse
import yaml
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Continual Pretraining with UnslothTrainer")
    parser.add_argument(
        "--config", required=True,
        help="Path to the CPT config YAML (e.g. config/cpt.yaml)"
    )
    args = parser.parse_args()

    # Load hyperparameters from YAML
    cfg = yaml.safe_load(open(args.config, "r"))

    # 1️⃣ Load base model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg["model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["load_in_4bit"]
    )

    # 2️⃣ Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        target_modules=cfg["adapter_targets"],
        lora_alpha=cfg.get("lora_alpha", cfg["lora_rank"] * 2),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias="none",
        use_gradient_checkpointing=cfg["gradient_checkpointing"],
        random_state=cfg.get("random_state", 3407),
        use_rslora=True
    )

    # 3️⃣ Load preprocessed Wikipedia dataset
    ds = load_dataset(
        "json",
        data_files={"train": "data/wikipedia_ko/processed.jsonl"}
    )["train"]

    # 4️⃣ Build valid training arguments (cast numerics appropriately)
    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        max_steps=int(cfg["max_steps"]),
        warmup_steps=int(cfg["warmup_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        embedding_learning_rate=float(cfg["embedding_learning_rate"]),
        logging_steps=int(cfg.get("logging_steps", 10)),
        optim=cfg.get("optim", "adamw_8bit"),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        seed=int(cfg.get("random_state", 3407)),
        output_dir=cfg["output_dir"],
        report_to=cfg.get("report_to", "none")
    )

    # 5️⃣ Initialize and run the trainer
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        dataset_num_proc=int(cfg.get("dataset_num_proc", 2)),
        args=training_args
    )

    trainer.train()

if __name__ == "__main__":
    main()
