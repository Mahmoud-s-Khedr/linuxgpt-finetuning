# LinuxGPT fine-tuning

A learning experiment adapting **Qwen2.5-Coder-0.5B-Instruct** and **SmolLM2-360M-Instruct** with LoRA for Linux Q&A and exact shell-command suffix completion. This is Experiment B; [Experiment A](https://github.com/Mahmoud-s-Khedr/linuxgpt-from-scratch) implements and trains small models from scratch.

This repository works independently. It includes the shared SFT dataset builder, Hugging Face chat formatting, prompt loss masking, LoRA training, optional Colab automation, study notes and historical evaluation outputs. It contains no pretrained weights, adapters or training corpus.

## Setup

Python 3.10+ and Bash are required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The Transformers/PEFT versions are those used in the archived evaluation environment. For GPU training, install a CUDA-compatible PyTorch build. The base models download from Hugging Face when training starts.

## Data

Prepare your own cleaned sources under `data/sources/`, using [the documented input format](docs/DATA.md), then run:

```bash
python build_sft_data.py --data_dir data/sources --out_dir artifacts/sft_v2
```

Alternatively, provide existing `sft_train.jsonl` and `sft_val.jsonl` with `--sft_dir`. Each row contains `mode`, `prompt` and `completion`:

```json
{"mode":"complete","prompt":"<|complete|><|ctx|>files=report.txt<|cmd|>cat<|pred|>","completion":" report.txt<|end|>"}
```

The leading space in that completion is required: it separates `cat` from its filename. The publication version preserves it; the historical trainer stripped it. The archived adapters/results below predate that fix and have not been retrained to measure its effect.

The builder splits unique commands before slicing prefixes. That reduces exact command leakage; it does not establish a wholly uncontaminated benchmark. Batched Bash parsing is only a filter and does not guarantee individual syntax, valid flags or correct intent. Inspect `sft_meta.json`: the builder's minimum 200-item validation reservations can exhaust tiny input datasets.

## Fine-tune

```bash
python finetune_lora.py --model_id Qwen/Qwen2.5-Coder-0.5B-Instruct --sft_dir artifacts/sft_v2 --out_dir runs/qwen --max_steps 1000
python finetune_lora.py --model_id HuggingFaceTB/SmolLM2-360M-Instruct --sft_dir artifacts/sft_v2 --out_dir runs/smollm --max_steps 1000
```

Defaults use rank 16, alpha 32, dropout 0.05 and adapters on attention/MLP projection layers. Prompt labels are `-100`; the Hugging Face causal model performs the label shift internally. Answer-token losses still backpropagate through the prompt's contextual computation.

The trainer saves the adapter and tokenizer under `<out_dir>/best_lora_adapter/`, plus metrics and before/during-training probes. It does not save a merged base model. The Q&A CLI below loads the base model together with that adapter.

## Ask Linux questions from the CLI (Q&A)

Run from the repository root with your virtual environment activated. Train the adapters above first, or copy your saved adapter directories into this checkout. Model weights and adapters are not included in the repository. The base model downloads from Hugging Face on first use and is cached locally; subsequent questions run locally with no OpenRouter API key.

### Ask one question with Qwen

```bash
python cli.py --model qwen --adapter runs/qwen/best_lora_adapter "How do I find large files?"
```

### Ask one question with SmolLM2

```bash
python cli.py --model smollm --adapter runs/smollm/best_lora_adapter "What is the difference between chmod and chown?"
```

The question is quoted as one shell argument. The CLI prints an answer and does not execute any generated command. The adapter directory must contain `adapter_config.json`, adapter weights and the tokenizer saved by training. The base model is read from the adapter metadata.

`--adapter` is optional when using the default directories shown above:

```bash
python cli.py --model qwen "How do I check free disk space?"
python cli.py --model smollm --device cpu --max-tokens 128 "What does ls -a show?"
```

`--device auto` is the default: CUDA when available, otherwise CPU. Answers use greedy decoding; `--max-tokens` defaults to 256.

### Interactive Q&A

Keep the model loaded and ask multiple independent questions:

```bash
python cli.py --model qwen --chat
python cli.py --model smollm --chat
```

Type `quit` or `exit` to stop. Each question is answered independently; previous turns are not included as conversation history.

### Compare with the original base model

To ask the original instruction model without a LinuxGPT LoRA adapter:

```bash
python cli.py --model qwen --base "How do I find large files?"
python cli.py --model smollm --base --chat
```

This uses the pretrained model's behavior, not your fine-tuned adapter. Use `python cli.py --help` to see all Q&A options.

## Optional Colab automation

The launcher expects an authenticated installation of the separate `google-colab-cli` tool (0.8.0 in the original environment). Set `COLAB_BIN` if it is not on `PATH`; job wrappers accept `PYTHON_BIN`.

```bash
python run_colab_exp_b.py --model qwen --gpu L4
python run_colab_exp_b.py --model smollm --gpu L4
```

These commands allocate paid GPU sessions, upload prepared SFT data and download the adapter/metrics. Run from the repository root. Sessions are stopped in `finally` unless `--keep_alive` is specified; check your sessions after failures. Publication checks did not allocate new Colab GPUs or retrain the archived models.

## Historical evaluation

Greedy generation, FP32 CPU inference, four threads, the same frozen 138-case suite as Experiment A:

| Model | Q&A rubric hits /54 | Strict suffix matches /74 |
|---|---:|---:|
| Qwen base | 32 | 1 |
| Qwen + LinuxGPT LoRA | 32 | 9 |
| SmolLM2 base | 11 | 0 |
| SmolLM2 + LinuxGPT LoRA | 17 | 2 |

Fine-tuning improved suffix compliance but did not produce a reliable autocomplete system. Matching `docker compose` in one probe was insufficient evidence: the larger suite exposed failures on filename context, mid-flag cursors, spaces and quoting. The Qwen adapter's Q&A rubric score stayed unchanged.

```bash
python evaluation/score_saved.py
```

Saved responses, checkpoint/adapter hashes, model revisions and metadata are included under [evaluation](evaluation/README.md). Rubric scores are diagnostic matches, not verified factual accuracy. Tokenizers, prior training and sizes differ, so raw losses across these models are not a fair accuracy comparison.

## Study guides

- [Qwen2.5-Coder and LoRA](docs/QWEN25_CODER_LORA_GUIDE.md)
- [SmolLM2 and LoRA](docs/SMOLLM2_LORA_GUIDE.md)

These notes were written during the original experiments. Their line numbers, paths and selected probes describe that working session; use this README for standalone commands and the larger benchmark for performance claims. Base models and source data retain their respective terms; their weights and corpus are not redistributed here.
