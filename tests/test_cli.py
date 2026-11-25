"""Offline CLI integration using a tiny local model and a real PEFT adapter."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        base_dir = Path(cls.tmp.name) / "base"
        cls.adapter = Path(cls.tmp.name) / "adapter"
        config = Qwen2Config(vocab_size=16, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=128, eos_token_id=2, pad_token_id=3)
        model = AutoModelForCausalLM.from_config(config)
        # Greedy ties select token 0, so the expected generated text is known.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.save_pretrained(base_dir)
        model = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True)
        adapter = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2,
                                                  lora_alpha=4, target_modules=["q_proj"]))
        adapter.save_pretrained(cls.adapter)
        backend = Tokenizer(models.WordLevel({"answer": 0, "[UNK]": 1, "[EOS]": 2,
                                               "[PAD]": 3, "system": 4, "user": 5,
                                               "assistant": 6}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                            eos_token="[EOS]", pad_token="[PAD]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(cls.adapter)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, *args, input=None):
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        return subprocess.run([sys.executable, str(ROOT / "cli.py"), *args],
                              input=input, text=True, capture_output=True, env=env, timeout=60)

    def test_question_loads_adapter_and_prints_only_generated_answer(self):
        result = self.run_cli("--adapter", str(self.adapter), "--device", "cpu", "--max-tokens", "2", "What is ls?")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "answer answer")

    def test_interactive_questions_and_quit(self):
        result = self.run_cli("--adapter", str(self.adapter), "--device", "cpu", "--max-tokens", "2",
                              "--chat", input="What is ls?\nWhat is pwd?\nquit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("LinuxGPT: answer answer"), 2)

    def test_missing_adapter_reports_training_prerequisite(self):
        result = self.run_cli("--adapter", str(Path(self.tmp.name) / "missing"), "What is ls?")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Train first", result.stderr)


if __name__ == "__main__":
    unittest.main()
