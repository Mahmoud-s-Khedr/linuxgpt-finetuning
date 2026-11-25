# Fine-Tuning `SmolLM2-360M-Instruct` with LoRA — Line-by-Line Study Guide

**Training script:** [`finetune_lora.py`](../finetune_lora.py)  
**Saved adapter:** `runs/smollm/best_lora_adapter/` (34 MB `safetensors`)  
**Metrics log:** `runs/smollm/metrics_lora.jsonl`  
**Training dataset:** Group-split, syntax-filtered `sft_v2` (40,000 train / 1,500 val examples sampled)

---

## 1. Why `SmolLM2-360M-Instruct`?

`SmolLM2-360M-Instruct` (HuggingFace, 2025) is one of the strongest ultra-compact open LLMs specifically engineered for **fast laptop-CPU inference**. It bridges the gap between your scratch **36.6M LLaMA** and **494M Qwen2.5-Coder**:

- **Architecture:** Pure LLaMA-style decoder — **32 layers**, `n_embd = 960`, **GQA** (15 Q heads, 5 KV heads — 3× smaller KV cache), **RoPE**, **RMSNorm**, **SwiGLU**, **49,152** vocab (vs. Qwen's 151,936 vocab → smaller embedding table & faster CPU softmax!).
- **Pretraining:** ~4 trillion tokens of curated web, textbooks, code, and math (SmolLM2 multi-stage data recipe).
- **Parameters:** **361.8M** base params (370.5M with LoRA) — **27% smaller** than `Qwen2.5-Coder-0.5B`.
- **What it FAILS zero-shot (why we fine-tuned it!):**
  Before LoRA, when asked to complete `tar -xzf `, SmolLM2 replied with **prose**:
  - `tar -xzf ` → `Extract the archive.` ❌ *(explained the command instead of completing it!)*

---

## 2. LoRA Configuration & Exact Parameter Math

Just like with Qwen, we freeze 100% of SmolLM2's 361.8M base weights $W_0$ and learn low-rank adapters $W = W_0 + \frac{\alpha}{r}BA$ on all 7 attention + SwiGLU projections across all **32 layers**:

```python
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,          # scaling factor = 32 / 16 = 2.0
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # GQA attention
        "gate_proj", "up_proj", "down_proj",      # SwiGLU MLP
    ],
)
```

### Exact Parameter Breakdown (from `logs/expb_smollm.log`)
- **Frozen base parameters:** `361,821,120` (97.66%)
- **Trainable LoRA parameters:** **`8,683,520` (2.34%)**
- **Saved adapter size:** **34 MB** (`adapter_model.safetensors`)

*(Notice why SmolLM2 has almost the same LoRA param count as Qwen despite being 27% smaller overall: SmolLM2 is **deeper** — 32 layers × `n_embd=960` vs. Qwen's 24 layers × `n_embd=896`, while Qwen spends a larger share of its total 494M params on its huge 151k-token multilingual embedding table!)*

---

## 3. Dual-Mode Data Pipeline & `-100` Loss Masking

`SmolLM2-360M-Instruct` also uses a ChatML-style template (`<|im_start|>system ... <|im_end|>`), so `finetune_lora.py` uses `tokenizer.apply_chat_template(...)` to format both modes identically to Qwen:

1. **Q&A Mode:** System prompt instructs concise Linux Q&A → user asks question → loss computed **only** on assistant response + `<|im_end|>`.
2. **Inline Autocomplete Mode:** System prompt instructs outputting **only** the remaining command suffix → user supplies `Context: ...` + `Command prefix: ...` → loss computed **only** on the suffix + `<|im_end|>`.

```python
input_ids = prompt_ids + comp_ids
labels    = [-100] * len(prompt_ids) + comp_ids  # prompt tokens masked!
```
Because HuggingFace's `LlamaForCausalLM` shifts `labels` internally by 1 token, masking `prompt_ids` with `-100` removes direct prompt-target loss. Answer losses still backpropagate through the context computation.

---

## 4. Training Setup & Convergence (`metrics_lora.jsonl`)

- **Hardware:** 1× NVIDIA L4 (24 GB VRAM), `bf16` AMP + TF32
- **Schedule:** `1,000` steps, effective batch `32` (`8 × 4` grad accum) = **32,000 examples seen**, cosine LR `2e-4 → 2e-5` with `80`-step warmup

| Step | Val Loss | Learning Rate |
| :---: | :---: | :---: |
| 200 | 2.2164 | `1.93e-4` |
| 400 | 2.1794 | `1.51e-4` |
| 600 | 2.1579 | `9.17e-5` |
| 800 | 2.1415 | `4.02e-5` |
| **999** | **2.1372** | `2.00e-5` |

---

## 5. Actual Results — Before vs. After LoRA (And a Crucial LLM Lesson!)

| Mode | Prompt | Zero-Shot `SmolLM2-360M` | **After 1,000-Step LoRA** |
| :--- | :--- | :--- | :--- |
| **COMP** | `docker compose ` *(ctx: `files=docker-compose.yml`)* | *(non-suffix)* ❌ | **`up -d`** ✅ |
| **COMP** | `tar -xzf ` | `Extract the archive.` ❌ *(prose!)* | **`/path/to/file.tar.gz`** ✅ *(clean shell suffix!)* |
| **Q&A** | `What is ls?` | `ls: List files and directories...` ✅ | `` `ls` is a command to list files... `` ✅ |
| **Q&A** | `How do I find large files?` | `du -h` list ✅ | `du -h path/to/file \| sort -h` ✅ |
| **Q&A** | `What is Arch Linux?` | `"based on the Debian project"` ❌ | `"based on the Debian system"` ❌ |

### Key Lesson You Just Observed Firsthand: **The Superficial Alignment Hypothesis (LIMA)**
Look at `tar -xzf ` vs. `What is Arch Linux?` above:
1. **What LoRA SFT IS amazing at (Behavior & Format):** In just 200–1,000 steps, LoRA completely transformed SmolLM2 from answering `tar -xzf ` with prose (`Extract the archive.`) into outputting **exact, clean shell completions** (`up -d`, `/path/to/file.tar.gz`)!
2. **What LoRA SFT does NOT easily overwrite (Pretrained Factual Myths):** SmolLM2's *base* model already believed *"Arch Linux is based on Debian."* Because our `sft_v2` sample didn't contain enough Arch Linux facts to overwrite that deep pretraining belief, it still repeated it—whereas `Qwen2.5-Coder-0.5B` (which was pretrained more heavily on technical/code docs) got Arch Linux right both before and after LoRA!
3. **Why `val_loss` alone doesn't tell the whole story:** SmolLM2 actually got a slightly lower SFT val loss (`2.1372`) than Qwen (`2.1652`)—largely because SmolLM2 has a **3× smaller vocabulary** (`49k` vs. `152k`, making next-token prediction over fewer classes mathematically lower entropy!)—even though Qwen had stronger factual Linux accuracy!

---

## 6. Head-to-Head Summary: `SmolLM2-360M` vs. `Qwen2.5-Coder-0.5B`

| Metric | `SmolLM2-360M-Instruct` LoRA | `Qwen2.5-Coder-0.5B-Instruct` LoRA |
| :--- | :---: | :---: |
| **Base Parameters** | **361.8M** *(faster on CPU!)* | 494.0M |
| **Layers / Dim** | 32 layers, `n_embd=960` | 24 layers, `n_embd=896` |
| **Vocab Size** | **49,152** *(faster CPU softmax)* | 151,936 |
| **LoRA Trainable Params** | 8.68M (2.34%) | 8.80M (1.75%) |
| **Adapter Disk Size** | 34 MB | 34 MB |
| **Best SFT Val Loss** | 2.1372 *(49k vocab)* | 2.1652 *(152k vocab)* |
| **Inline Autocomplete** | **`up -d`**, **`/path/to/file.tar.gz`** ✅ | **`up -d`**, `file.tar.gz` / `"$1"` ✅ |
| **Factual Linux Knowledge** | Good, occasional myths (*Arch=Debian*) | **Stronger** (*code/Linux-focused base*) |
| **Best Role in LinuxGPT** | **Ultra-fast CPU inline autocomplete** | **Best overall Q&A + autocomplete model** |
