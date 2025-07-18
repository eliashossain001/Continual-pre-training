### Continual Pretraining and Instruction Finetuning Pipeline Using Mistral v0.3 (7B)


A complete end-to-end VS Code–friendly setup for:

* **Continual Pretraining** (CPT) on domain-specific corpora (e.g., Wikipedia Korean) using 4‑bit quantized LLMs + LoRA adapters.
* **Instruction Finetuning** (InstFT) on instruction–response datasets (e.g., Alpaca-GPT4 Korean) with minimal additional parameters.
* **Fast Inference** via optimized kernels and streaming support.
* **Deployment Packaging** to GGUF format for llama.cpp, Ollama, or vLLM.

---

## 📋 Prerequisites

* **GPU** with at least 8 GB VRAM (recommend >16 GB).
* **Python 3.10+**
* **VS Code** (optional, but recommended)
* **Poetry** or **pip** for environment management.

---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/eliashossain001/Continual-pre-training.git
   cd cpt-pipeline
   ```

2. **Create & activate a virtual environment**:

   ```bash
   python3 -m venv eliasenv
   source eliasenv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure VS Code** (optional):

   * Create a folder `.vscode/` at the project root.
   * Add `settings.json` with your interpreter path and exclusions.

---

## 💾 Data Preparation

### 1. Preprocess Korean Wikipedia

```bash
python scripts/preprocess_wikipedia.py \
  --output data/wikipedia_ko
```

* Samples 1% of the `wikimedia/wikipedia` Korean split.
* Outputs `data/wikipedia_ko/processed.jsonl`.

### 2. Preprocess Alpaca‑GPT4 Korean

```bash
python scripts/preprocess_alpaca.py \
  --output data/alpaca_kr
```

* Loads `FreedomIntelligence/alpaca-gpt4-korean`.
* Outputs `data/alpaca_kr/processed.jsonl`.

---

## 🏋️ Continual Pretraining

```bash
python scripts/train_continual.py \
  --config config/cpt.yaml
```

* Loads a 4‑bit base model (e.g. `unsloth/mistral-7b-v0.3-bnb-4bit`).
* Attaches LoRA adapters with CPT-specific targets.
* Trains for `max_steps` on the Wikipedia subset.
* Saves adapter checkpoints under `outputs/llama3-cpt/checkpoint-*`.

---

## 📝 Instruction Finetuning

```bash
python scripts/train_instruction.py \
  --config config/inst_ft.yaml
```

* Automatically picks the latest `outputs/llama3-cpt/checkpoint-*`.
* Attaches LoRA adapters (if not already) for instruction data.
* Fine-tunes on `data/alpaca_kr/processed.jsonl`.
* Saves under `outputs/llama3-inst/checkpoint-*`.

---

## 🚀 Inference

```bash
python scripts/inference.py \
  --model_dir outputs/llama3-inst/checkpoint-XXX \
  --prompt_type alpaca \
  --instruction "지구를 광범위하게 설명하세요." \
  --max_tokens 64
```

* Loads the adapter + base weights.
* Builds a prompt from the chosen template.
* Generates up to `max_tokens` new tokens.
* Prints *only* the model’s response (no prompt echo).

**GPU memory**:

```bash
python scripts/memory_stats.py
```

---

## 📦 Export to GGUF

```bash
python scripts/export_gguf.py \
  --base-model unsloth/llama-3-8b-bnb-4bit \
  --adapter-folder outputs/llama3-inst/checkpoint-XXX \
  --outfile llama3-inst.gguf
```

* Merges base + LoRA into a single `.gguf` file.
* Deploy with llama.cpp, Ollama, or vLLM on CPU/GPU.

---

## 🔧 Configuration

* **`config/cpt.yaml`**: Hyperparameters for continual pretraining.
* **`config/inst_ft.yaml`**: Hyperparameters for instruction finetuning.
* **`.vscode/settings.json`**: VS Code workspace settings.

---

## 📂 Directory Layout

```
cpt-pipeline/
├── data/                             # Raw & processed datasets
│   ├── wikipedia_ko/                 # Preprocessed Wiki JSONL
│   └── alpaca_kr/                    # Preprocessed Alpaca JSONL
├── scripts/                          # All pipeline scripts
├── config/                           # YAML configs
├── outputs/                          # Saved checkpoints & GGUF bundles
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files to ignore in VCS
└── README.md                         # This file
```

---


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![GitHub stars](https://img.shields.io/github/stars/eliashossain001/Domain-adaptive-llm-ft?style=social)](https://github.com/eliashossain001/Domain-adaptive-llm-ft)


## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD Student | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
