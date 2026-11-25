# Fine-Tuning `Qwen2.5-Coder-0.5B-Instruct` with LoRA — Line-by-Line Study Guide

**Training script:** [`finetune_lora.py`](../finetune_lora.py)  
**Saved adapter:** `runs/qwen/best_lora_adapter/` (34 MB `safetensors`)  
**Metrics log:** `runs/qwen/metrics_lora.jsonl`  
**Training dataset:** Group-split, syntax-filtered `sft_v2` (40,000 train / 1,500 val examples sampled)

---

## 1. Why Start From `Qwen2.5-Coder-0.5B-Instruct`?

In Experiment A, our scratch `llama_50m` learned Linux syntax patterns, but couldn't answer basic plain-English questions like `"What is ls?"` or `"What is Arch Linux?"` reliably because 196M forum tokens isn't enough for general language comprehension.

`Qwen2.5-Coder-0.5B-Instruct` already completed **Stage 1 (General Language + Code Comprehension)** on trillions of tokens—and uses the exact same modern Transformer family you built from scratch in `models/llama.py`:
- **Architecture:** 24 layers, `n_embd = 896`, **GQA** (14 Q heads, 2 KV heads — 7× smaller KV cache!), **RoPE**, **RMSNorm**, **SwiGLU**
- **Parameters:** **494M** base params (502.8M with LoRA)
- **What it ALREADY knows zero-shot:** Fluent English, `ls`, `du`, `Arch Linux`, Bash concepts.
- **What it FAILS zero-shot (why we fine-tuned it!):** Inline shell autocomplete! Before LoRA:
  - `docker compose ` → `docker-compose.yml` ❌ *(repeated filename from context)*
  - `tar -xzf ` → `tar -xzf` ❌ *(echoed the prefix instead of completing it)*

---

## 2. How LoRA (Low-Rank Adaptation) Works — The Core Math

Updating all **494,032,768** parameters directly (full fine-tuning) requires huge optimizer memory, risks **catastrophic forgetting** of its pretrained English/code knowledge, and produces a 1 GB checkpoint every time.

**LoRA** freezes 100% of the base weights $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and attaches two tiny trainable matrices $A$ and $B$ in parallel:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ (initialized with random Gaussian noise)
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ (initialized to **all zeros**, so at step 0, $BA = 0$ and the model behaves *identically* to the base model!)
- **Rank `r = 16`** (`finetune_lora.py`, line 255): Controls capacity of the low-rank update.
- **Scaling `lora_alpha = 32`** (line 256): Scales the LoRA update by $\alpha / r = 32 / 16 = 2.0$.
- **Dropout `lora_dropout = 0.05`** (line 257): Regularizes the adapter input.

### Exact Target Modules (`finetune_lora.py`, line 260)
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention projections
    "gate_proj", "up_proj", "down_proj",      # SwiGLU MLP projections
]
```
Notice those 7 names are the **exact HuggingFace equivalents** of `wq, wk, wv, wo` and `w_gate, w_up, w_down` from your own `models/llama.py`!

### Parameter Savings (Exact Numbers from Our Run)
- **Frozen base parameters:** `494,032,768` (98.25%)
- **Trainable LoRA parameters:** **`8,798,208` (1.75%)**
- **Saved adapter on disk:** Only **34 MB** (`adapter_model.safetensors`) instead of ~1 GB!

---

## 3. Dual-Mode Prompt Formatting (`finetune_lora.py`, lines 39–75)

Unlike our scratch models (which invented custom tokens `<|qa|>` and `<|complete|>`), `Qwen2.5-Coder-0.5B-Instruct` was already trained on **ChatML** (`<|im_start|>system ... <|im_end|>`). Rather than fighting its native format, we adapt our two LinuxGPT modes cleanly into its chat template via two distinct system prompts:

1. **Q&A Mode (`SYSTEM_QA`, lines 39–42):**
   ```text
   <|im_start|>system
   You are LinuxGPT, a concise and accurate Linux CLI assistant...<|im_end|>
   <|im_start|>user
   What is ls?<|im_end|>
   <|im_start|>assistant
   `ls` is the `list` command...<|im_end|>
   ```
2. **Inline Autocomplete Mode (`SYSTEM_COMPLETE`, lines 43–47):**
   ```text
   <|im_start|>system
   You are LinuxGPT inline shell autocomplete. Output ONLY the remaining command suffix...<|im_end|>
   <|im_start|>user
   Context: files=docker-compose.yml
   Command prefix: docker compose <|im_end|>
   <|im_start|>assistant
   up -d<|im_end|>
   ```

`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` (line 75) automatically inserts Qwen's exact `<|im_start|>` / `<|im_end|>` tokens.

---

## 4. Exact Prompt Loss Masking (`PretrainedDualModeDataset`, lines 78–134)

```python
prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
comp_ids   = tokenizer.encode(comp, add_special_tokens=False) + [eos_id]

input_ids = prompt_ids + comp_ids
labels    = [-100] * len(prompt_ids) + comp_ids
```

**Important subtle difference vs. our scratch `train_sft.py`:**
- In our scratch `models/llama.py`, we manually shifted `x = full_ids[:-1]` and `y = full_ids[1:]`.
- In HuggingFace's `AutoModelForCausalLM`, `model(input_ids=ids, labels=lbl)` performs the 1-token causal shift **internally**! So `labels` aligns **1:1** with `input_ids`: every prompt token gets `-100`, every completion token (including `eos_id` = `<|im_end|>`) gets its real token ID, and padding tokens in `collate()` (line 145) also get `-100` plus `attention_mask = 0`.

---

## 5. Training Loop & Hyperparameters (`finetune_lora.py`, lines 265–325)

- **Hardware:** 1× NVIDIA L4 (24 GB VRAM), `bf16` AMP + TF32 matmuls (~20 minutes total)
- **Batching:** `batch_size = 8`, `grad_accum = 4` → **effective batch size = 32 examples/step**
- **Steps:** `1,000` steps = **32,000 examples seen** (mixed Q&A + syntax-filtered autocomplete)
- **Optimizer:** `AdamW` updating **only** `[p for p in model.parameters() if p.requires_grad]` (the 1.75% LoRA weights!)
- **LR Schedule:** Linear warmup (`80` steps) → Cosine decay from `2e-4` down to `2e-5` (higher than full fine-tuning because LoRA adapters start from zero!)
- **Gradient Clipping:** `clip_grad_norm_(..., 1.0)`
- **Train-mode safety:** Both `evaluate()` and `run_probes()` save `was_training = model.training` and restore `model.train()` in `finally:` (fixing the bug caught in our Experiment A review!).

---

## 6. Actual Results — Before vs. After LoRA

### Validation Loss Curve (`metrics_lora.jsonl`)
| Step | Val Loss | Learning Rate |
| :---: | :---: | :---: |
| 200 | 2.2642 | `1.93e-4` |
| 400 | 2.2157 | `1.51e-4` |
| 600 | 2.1830 | `9.17e-5` |
| 800 | 2.1719 | `4.02e-5` |
| **999** | **2.1652** | `2.00e-5` |

### Qualitative Before vs. After Probes
| Mode | Prompt | Zero-Shot Base Model | **After 1,000-Step LoRA** |
| :--- | :--- | :--- | :--- |
| **COMP** | `docker compose ` *(ctx: `files=docker-compose.yml`)* | `docker-compose.yml` ❌ | **`up -d`** ✅ |
| **COMP** | `tar -xzf ` | `tar -xzf` ❌ | **`file.tar.gz` / `"$1"`** ✅ |
| **Q&A** | `What is ls?` | Accurate explanation ✅ | Accurate explanation ✅ |
| **Q&A** | `What is Arch Linux?` | Accurate (`lightweight Linux distro...`) ✅ | Accurate (`community-driven Linux distro...`) ✅ |
| **Q&A** | `How do I find large files?` | `du` explanation ✅ | `du -h` with example ✅ |

---

## 7. How to Load & Merge for Local Inference

At inference time, you can either load the base model + 34 MB adapter dynamically, **or merge $W = W_0 + \frac{\alpha}{r}BA$ permanently** so inference has **zero LoRA overhead**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "experiments/B-finetuning/runs/qwen/best_lora_adapter")
merged = model.merge_and_unload()  # folds B @ A directly into W_0!
```
Once merged, it is a standard HuggingFace model that can be converted straight to **GGUF Q4_K_M** (~350 MB) for ultra-fast `llama.cpp` CPU inference!
