import json
from pathlib import Path
import tempfile
import unittest

from finetune_lora import PretrainedDualModeDataset, make_collate_fn, parse_sft_v2_record


class CharacterTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return 'USER:' + messages[-1]['content'] + '\nASSISTANT:'


def completion(prefix='cat', suffix=' report.txt'):
    return {'mode': 'complete',
            'prompt': '<|complete|><|ctx|>files=report.txt<|cmd|>' + prefix + '<|pred|>',
            'completion': suffix + '<|end|>'}


class ProtocolTests(unittest.TestCase):
    def test_cursor_spaces_and_partial_flag_are_preserved(self):
        mode, user, suffix = parse_sft_v2_record(completion())
        self.assertEqual(suffix, ' report.txt')
        self.assertTrue(user.endswith('Command prefix: cat'))
        _, user, suffix = parse_sft_v2_record(completion('cat ', '"report draft.txt"'))
        self.assertTrue(user.endswith('Command prefix: cat '))
        self.assertEqual(suffix, '"report draft.txt"')
        self.assertEqual(parse_sft_v2_record(completion('git log --on', 'eline'))[2], 'eline')

    def test_first_suffix_space_is_supervised_and_padding_is_not(self):
        tok = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sft.jsonl'
            path.write_text(json.dumps(completion()) + '\n')
            dataset = PretrainedDualModeDataset(path, tok)
        ids, labels = dataset[0]
        first = next(i for i, label in enumerate(labels) if label != -100)
        self.assertTrue(all(label == -100 for label in labels[:first]))
        self.assertEqual(labels[first], ord(' ') + 2)
        self.assertEqual(labels[first:], tok.encode(' report.txt') + [tok.eos_token_id])
        self.assertEqual(ids[first:], labels[first:])
        input_ids, batched_labels, mask = make_collate_fn(0)([(ids, labels), (ids[:-1], labels[:-1])])
        self.assertTrue((batched_labels[mask == 0] == -100).all())
        self.assertEqual(int(mask[1].sum()), len(ids) - 1)


if __name__ == '__main__':
    unittest.main()
