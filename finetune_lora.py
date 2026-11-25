"""
Experiment B — Dual-Mode LoRA Fine-Tuning for Small Pretrained LLMs:
  - Qwen/Qwen2.5-Coder-0.5B-Instruct
  - HuggingFaceTB/SmolLM2-360M-Instruct

Trains on the group-split, syntax-filtered SFT v2 format shared with Experiment A
(`artifacts/sft_v2/sft_train.jsonl` & `sft_val.jsonl`), using:
  1. Exact prompt-token loss masking (`labels = -100` on prompt tokens)
  2. Dual-mode formatting:
     - Q&A:          Uses the model's native chat template (preserves fluent English!)
     - Autocomplete: Compact inline completion prompt with `<|ctx|>` + `<|cmd|>`
  3. LoRA (r=16, alpha=32) on all attention + MLP projections
  4. Saves the LoRA adapter and tokenizer + metrics JSONL + live dual-mode probes

Usage:
  python finetune_lora.py \
      --model_id Qwen/Qwen2.5-Coder-0.5B-Instruct \
      --sft_dir artifacts/sft_v2 \
      --out_dir runs/qwen25_coder_0.5b \
      --max_steps 1200
"""

from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


SYSTEM_QA = (
    "You are LinuxGPT, a concise and accurate Linux CLI assistant. "
    "Answer clearly with valid Linux commands. If a command or package does not exist, say so."
)
SYSTEM_COMPLETE = (
    "You are LinuxGPT inline shell autocomplete. "
    "Output ONLY the remaining command suffix to complete the user's partial Linux command. "
    "No explanation, no markdown, no extra text."
)


def parse_sft_v2_record(rec: dict) -> tuple[str, str, str]:
    """Extract (mode, user_text, assistant_text) from our unified sft_v2 JSONL."""
    mode = rec["mode"]
    prompt = rec["prompt"]
    comp = rec["completion"].removesuffix("<|end|>")
    if mode == "qa":
        comp = comp.strip()

    if mode == "qa":
        user_text = prompt.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0].strip()
    else:
        # Preserve <|ctx|> if present + <|cmd|> prefix cleanly
        if "<|ctx|>" in prompt:
            ctx = prompt.split("<|ctx|>", 1)[1].split("<|cmd|>", 1)[0].strip()
            cmd = prompt.split("<|cmd|>", 1)[1].split("<|pred|>", 1)[0]
            user_text = f"Context: {ctx}\nCommand prefix: {cmd}"
        else:
            cmd = prompt.split("<|cmd|>", 1)[1].split("<|pred|>", 1)[0]
            user_text = f"Command prefix: {cmd}"
    return mode, user_text, comp


def format_chat_prompt(tokenizer, mode: str, user_text: str) -> str:
    sys_msg = SYSTEM_QA if mode == "qa" else SYSTEM_COMPLETE
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class PretrainedDualModeDataset(Dataset):
    """Tokenizes dual-mode Q&A + Autocomplete records with exact -100 prompt masking."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        max_len: int = 512,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.samples: list[tuple[list[int], list[int]]] = []
        eos_id = tokenizer.eos_token_id

        with jsonl_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        if max_samples is not None and len(lines) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(lines)
            lines = lines[:max_samples]

        skipped = 0
        for line in lines:
            rec = json.loads(line)
            mode, user_text, comp = parse_sft_v2_record(rec)
            if not comp:
                skipped += 1
                continue

            prompt_str = format_chat_prompt(tokenizer, mode, user_text)
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            comp_ids = tokenizer.encode(comp, add_special_tokens=False) + [eos_id]

            if len(prompt_ids) + len(comp_ids) > max_len:
                max_p = max(64, max_len - 96)
                prompt_ids = prompt_ids[:max_p]
                comp_ids = comp_ids[: (max_len - len(prompt_ids))]

            if len(prompt_ids) < 4 or len(comp_ids) < 2:
                skipped += 1
                continue

            input_ids = prompt_ids + comp_ids
            # Standard HF CausalLM shifts labels internally, so labels align 1:1 with input_ids!
            labels = [-100] * len(prompt_ids) + comp_ids
            self.samples.append((input_ids, labels))

        print(f"Loaded {len(self.samples):,} examples from {jsonl_path.name} (skipped {skipped})", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def make_collate_fn(pad_id: int):
    def collate(batch: list[tuple[list[int], list[int]]]):
        max_len = max(len(ids) for ids, _ in batch)
        max_len = ((max_len + 7) // 8) * 8
        input_ids, labels, attn_mask = [], [], []
        for ids, lbl in batch:
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            labels.append(lbl + [-100] * pad_n)
            attn_mask.append([1] * len(ids) + [0] * pad_n)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
        )
    return collate


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, device: str, amp_ctx, max_batches: int = 30) -> float:
    was_training = model.training
    model.eval()
    try:
        losses = []
        for i, (ids, lbl, mask) in enumerate(val_loader):
            if i >= max_batches:
                break
            ids, lbl, mask = ids.to(device), lbl.to(device), mask.to(device)
            with amp_ctx:
                out = model(input_ids=ids, attention_mask=mask, labels=lbl)
            losses.append(out.loss.item())
        return sum(losses) / max(1, len(losses))
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def run_probes(model, tokenizer, device: str, tag: str):
    was_training = model.training
    model.eval()
    try:
        probes = [
            ("Q&A", "qa", "What is ls?", 64),
            ("Q&A", "qa", "How do I find large files in Linux?", 80),
            ("Q&A", "qa", "What is Arch Linux?", 64),
            ("COMP", "complete", "Context: files=docker-compose.yml\nCommand prefix: docker compose ", 16),
            ("COMP", "complete", "Command prefix: tar -xzf ", 16),
        ]
        print(f"  --- [{tag}] Live Dual-Mode Probes ---", flush=True)
        for lbl, mode, user_text, max_new in probes:
            prompt = format_chat_prompt(tokenizer, mode, user_text)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            gen_clean = gen.replace("\n", "\\n")
            print(f"    [{lbl}] {user_text[:45]:<45s} ==> {gen_clean[:120]}", flush=True)
    finally:
        if was_training:
            model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="HF model ID (e.g. Qwen/Qwen2.5-Coder-0.5B-Instruct)")
    parser.add_argument("--sft_dir", type=Path, default=Path("artifacts/sft_v2"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=40000)
    parser.add_argument("--max_val_samples", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        from contextlib import nullcontext
        amp_dtype = torch.float32
        amp_ctx = nullcontext()

    print(f"Loading tokenizer & base model: {args.model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Probe BEFORE fine-tuning so we have a clean zero-shot baseline!
    run_probes(model, tokenizer, device, f"{args.model_id.split('/')[-1]}-base")

    train_ds = PretrainedDualModeDataset(
        args.sft_dir / "sft_train.jsonl", tokenizer, max_len=args.max_len, max_samples=args.max_train_samples, seed=args.seed
    )
    val_ds = PretrainedDualModeDataset(
        args.sft_dir / "sft_val.jsonl", tokenizer, max_len=args.max_len, max_samples=args.max_val_samples, seed=args.seed
    )

    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    metrics_path = args.out_dir / "metrics_lora.jsonl"
    best_val = float("inf")
    data_iter = iter(train_loader)
    t_log = time.perf_counter()

    tag = args.model_id.split("/")[-1]
    print(f"\n=== LoRA Fine-Tuning: {tag} (steps={args.max_steps}, eff_batch={args.batch_size * args.grad_accum}) ===\n", flush=True)

    for step in range(args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                ids, lbl, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                ids, lbl, mask = next(data_iter)
            ids, lbl, mask = ids.to(device), lbl.to(device), mask.to(device)
            with amp_ctx:
                out = model(input_ids=ids, attention_mask=mask, labels=lbl)
                loss = out.loss / args.grad_accum
            accum_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            dt = time.perf_counter() - t_log
            print(f"[{tag}] step {step:4d}/{args.max_steps} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt:.1f}s", flush=True)
            t_log = time.perf_counter()

        if (step > 0 and step % args.eval_interval == 0) or step == args.max_steps - 1:
            val_loss = evaluate(model, val_loader, device, amp_ctx)
            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps({"step": step, "val_loss": round(val_loss, 4), "lr": lr}) + "\n")
            print(f"  >>> [{tag}] EVAL step {step}: val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                model.save_pretrained(args.out_dir / "best_lora_adapter")
                tokenizer.save_pretrained(args.out_dir / "best_lora_adapter")
            run_probes(model, tokenizer, device, tag)
            t_log = time.perf_counter()

    print(f"\nDone! Best val_loss={best_val:.4f} -> saved LoRA adapter to {args.out_dir / 'best_lora_adapter'}", flush=True)


if __name__ == "__main__":
    main()
