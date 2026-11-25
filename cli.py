"""Local Q&A for the Experiment B adapters, using their training chat template."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "smollm": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


def load_qa_model(adapter: Path | None, family: str, device: str):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or --device auto.")

    base_id = PeftConfig.from_pretrained(str(adapter)).base_model_name_or_path if adapter else MODEL_IDS[family]
    tokenizer = AutoTokenizer.from_pretrained(str(adapter) if adapter else base_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.float32 if device == "cpu" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    print(f"Loading {base_id} on {device}" + (f" with adapter {adapter}" if adapter else " (base model)"), file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, tokenizer


def answer_question(model, tokenizer, question: str, max_tokens: int) -> str:
    import torch
    from finetune_lora import format_chat_prompt

    # Match the SFT system instruction and model-native chat template exactly.
    prompt = format_chat_prompt(tokenizer, "qa", question)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_length = inputs["input_ids"].shape[1]
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and prompt_length + max_tokens > context_limit:
        raise ValueError("Question plus output allowance exceeds the model context; shorten the question or lower --max-tokens.")
    inputs = inputs.to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="LinuxGPT Experiment B: local Linux Q&A")
    parser.add_argument("question", nargs="?", help="Linux question, quoted as one argument")
    parser.add_argument("--model", choices=MODEL_IDS, default="qwen", help="Default adapter family or base model (default: qwen)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter", type=Path, help="Saved adapter/tokenizer directory; base model is read from its metadata")
    group.add_argument("--base", action="store_true", help="Use the original pretrained model without a LinuxGPT adapter")
    parser.add_argument("--chat", action="store_true", help="Keep the model loaded for independent Q&A turns; quit to exit")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated answer tokens (default: 256)")
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.chat and args.question is not None:
        parser.error("Use either a question or --chat")
    if not args.chat and (not args.question or not args.question.strip()):
        parser.error("Provide a question or use --chat")
    adapter = None if args.base else (args.adapter or ROOT / "runs" / args.model / "best_lora_adapter")
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        parser.error(f"No adapter_config.json in {adapter}. Train first, pass --adapter, or use --base.")
    try:
        model, tokenizer = load_qa_model(adapter, args.model, args.device)
    except ValueError as error:
        parser.error(str(error))
    if not args.chat:
        try:
            print(answer_question(model, tokenizer, args.question, args.max_tokens))
        except ValueError as error:
            parser.error(str(error))
        return
    print("LinuxGPT Q&A. Each question is independent. Type quit to exit.")
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() in {"quit", "exit", "/quit"}:
            break
        if not question.strip():
            continue
        try:
            print("LinuxGPT:", answer_question(model, tokenizer, question, args.max_tokens))
        except ValueError as error:
            print(error, file=sys.stderr)


if __name__ == "__main__":
    main()
