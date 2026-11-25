"""
Standalone single-VM Google Colab runner for Experiment B (LoRA fine-tuning of
Qwen2.5-Coder-0.5B-Instruct and SmolLM2-360M-Instruct on our clean SFT v2 dataset).

Why saving only the LoRA adapter is super fast & clean:
  - Base weights download directly from HuggingFace on Colab (~10s) and locally (~10s).
  - The trained LoRA adapter is only ~25–35 MB, so downloading it back from Colab
    takes ~3 seconds in 1-2 chunks!

Usage:
  python run_colab_exp_b.py --model qwen   --gpu L4
  python run_colab_exp_b.py --model smollm --gpu L4
"""

from __future__ import annotations
import argparse
import os
import subprocess
import tarfile
from pathlib import Path

COLAB_BIN = os.environ.get("COLAB_BIN", "colab")
CHUNK_BYTES = 25 * 1024 * 1024

MODEL_MAP = {
    "qwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "smollm": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


def run_cmd(cmd: list[str], log_file: Path) -> str:
    cmd_str = " ".join(cmd)
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n$ {cmd_str}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout)
        lf.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {cmd_str}\n{proc.stdout[-800:]}")
    return proc.stdout


def exec_checked(session: str, script_path: Path, code: str, timeout: int, log_file: Path) -> str:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    # Build sentinel dynamically so it never appears literally in traceback source lines!
    script_path.write_text(
        code.rstrip() + "\nprint('=== ' + 'STEP_OK' + ' ===', flush=True)\n",
        encoding="utf-8",
    )
    out = run_cmd([COLAB_BIN, "exec", "-s", session, "--timeout", str(timeout), "-f", str(script_path)], log_file)
    if "=== STEP_OK ===" not in out:
        raise RuntimeError(f"Remote step failed on {session}. Tail:\n{out[-1500:]}")
    return out


def ensure_sft_v2_chunks(bundle_path: Path, chunks_dir: Path) -> list[Path]:
    sft_dir = Path("artifacts/sft_v2")
    files = [
        (Path("finetune_lora.py"), "finetune_lora.py"),
        (sft_dir / "sft_train.jsonl", "sft_v2/sft_train.jsonl"),
        (sft_dir / "sft_val.jsonl", "sft_v2/sft_val.jsonl"),
    ]
    for src, _ in files:
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w:gz", compresslevel=3) as tar:
        for src, arc in files:
            tar.add(src, arcname=arc)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old in chunks_dir.glob("expb.part*"):
        old.unlink()
    chunks = []
    with bundle_path.open("rb") as f:
        idx = 0
        while True:
            buf = f.read(CHUNK_BYTES)
            if not buf:
                break
            cp = chunks_dir / f"expb.part{idx:02d}"
            cp.write_bytes(buf)
            chunks.append(cp)
            idx += 1
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "smollm"], required=True)
    parser.add_argument("--gpu", choices=["T4", "L4", "A100"], default="L4")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--keep_alive", action="store_true")
    args = parser.parse_args()

    model_id = MODEL_MAP[args.model]
    session = f"linuxgpt-expb-{args.model}"
    log_file = Path("logs") / f"expb_{args.model}.log"
    scripts_dir = Path("artifacts/remote_scripts")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(f"=== Starting {session} ({model_id} on {args.gpu}) ===\n", encoding="utf-8")

    chunks = ensure_sft_v2_chunks(Path("artifacts/expb_bundle.tar.gz"), Path("artifacts/chunks"))
    created = False

    try:
        print(f"[{session}] Creating {args.gpu} session (log: {log_file}) ...", flush=True)
        run_cmd([COLAB_BIN, "new", "-s", session, "--gpu", args.gpu], log_file)
        created = True

        print(f"[{session}] Uploading {len(chunks)} chunks ...", flush=True)
        for cp in chunks:
            run_cmd([COLAB_BIN, "upload", "-s", session, str(cp), cp.name], log_file)

        print(f"[{session}] Extracting & installing transformers + peft ...", flush=True)
        setup_py = """
import glob, os, subprocess, sys
candidates = [sorted(glob.glob('expb.part*')), sorted(glob.glob('/content/expb.part*')), sorted(glob.glob('/expb.part*'))]
parts = next((c for c in candidates if c), [])
assert parts, 'No expb.part* found!'
os.makedirs('/content/expb', exist_ok=True)
tar_p = '/content/expb_bundle.tar.gz'
with open(tar_p, 'wb') as out_f:
    for p in parts:
        with open(p, 'rb') as pf:
            out_f.write(pf.read())
        os.remove(p)
subprocess.run(['tar', '-xzf', tar_p, '-C', '/content/expb'], check=True)
os.remove(tar_p)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.24.1', 'peft==0.22.0', 'accelerate==1.16.0'], check=True)
import torch
print('GPU ready:', torch.cuda.get_device_name(0), flush=True)
"""
        exec_checked(session, scripts_dir / f"{args.model}_01_setup.py", setup_py, 300, log_file)

        print(f"[{session}] Fine-tuning {model_id} ({args.max_steps} steps on {args.gpu}) ...", flush=True)
        train_py = f"""
import subprocess, sys
cmd = [
    sys.executable, 'finetune_lora.py',
    '--model_id', '{model_id}',
    '--sft_dir', '/content/expb/sft_v2',
    '--out_dir', '/content/expb/runs/{args.model}',
    '--max_steps', '{args.max_steps}',
    '--batch_size', '8', '--grad_accum', '4',
]
print('Running:', ' '.join(cmd), flush=True)
proc = subprocess.Popen(cmd, cwd='/content/expb', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end='', flush=True)
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f'finetune_lora.py exited with rc={{rc}}')
"""
        exec_checked(session, scripts_dir / f"{args.model}_02_train.py", train_py, 0, log_file)

        print(f"[{session}] Packing LoRA adapter & metrics ...", flush=True)
        pack_py = f"""
import os, tarfile
tar_out = '/content/expb_{args.model}_results.tar.gz'
with tarfile.open(tar_out, 'w:gz') as tar:
    tar.add('/content/expb/runs/{args.model}', arcname='{args.model}')
chunk_size = 25 * 1024 * 1024
part_names = []
with open(tar_out, 'rb') as rf:
    idx = 0
    while True:
        buf = rf.read(chunk_size)
        if not buf:
            break
        pname = f'expb_res_{args.model}.part{{idx:02d}}'
        with open(f'/content/{{pname}}', 'wb') as wf:
            wf.write(buf)
        part_names.append(pname)
        idx += 1
print('RES_PARTS:', part_names, flush=True)
"""
        pack_out = exec_checked(session, scripts_dir / f"{args.model}_03_pack.py", pack_py, 300, log_file)
        parts_line = [ln for ln in pack_out.splitlines() if "RES_PARTS:" in ln][-1]
        part_names = eval(parts_line.split("RES_PARTS:", 1)[1].strip())

        local_out = Path("runs")
        local_out.mkdir(parents=True, exist_ok=True)
        local_parts = []
        for pname in part_names:
            lp = local_out / pname
            try:
                run_cmd([COLAB_BIN, "download", "-s", session, f"content/{pname}", str(lp)], log_file)
            except RuntimeError:
                run_cmd([COLAB_BIN, "download", "-s", session, pname, str(lp)], log_file)
            local_parts.append(lp)

        res_tar = local_out / f"expb_{args.model}_results.tar.gz"
        with res_tar.open("wb") as out_f:
            for lp in local_parts:
                out_f.write(lp.read_bytes())
                lp.unlink()
        with tarfile.open(res_tar, "r:gz") as tar:
            tar.extractall(local_out)
        res_tar.unlink()

        print(f"[{session}] COMPLETE! LoRA adapter & metrics saved to {local_out / args.model}", flush=True)
    finally:
        if created and not args.keep_alive:
            print(f"[{session}] Stopping Colab session ...", flush=True)
            try:
                run_cmd([COLAB_BIN, "stop", "-s", session], log_file)
            except Exception as e:
                print(f"[{session}] Warning: failed to stop session: {e}", flush=True)


if __name__ == "__main__":
    main()
