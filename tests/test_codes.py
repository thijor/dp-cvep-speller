"""Tests for code-sequence loading and key/code map construction."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tests._support import create_key2seq_and_code2key, load_cfg


def _n_keys(cfg):
    return sum(len(row) for row in cfg["speller"]["keys"]["keys_upper"])


def _raw_codes(cfg, phase):
    return np.loadtxt(
        f"{cfg['speller']['codes_dir']}/{cfg[phase]['codes_file']}", delimiter=","
    )


def _factor(cfg):
    return int(
        cfg["speller"]["screen"]["refresh_rate_hz"]
        / cfg["speller"]["presentation_rate_hz"]
    )


class CreateMapsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()

    def test_maps_have_one_entry_per_key(self):
        expected = [k for row in self.cfg["speller"]["keys"]["keys_upper"] for k in row]
        for phase in ("training", "online"):
            with self.subTest(phase=phase):
                key_to_seq, code_to_key = create_key2seq_and_code2key(self.cfg, phase)
                self.assertEqual(list(code_to_key.values()), expected)
                self.assertEqual(set(key_to_seq), set(expected))
                self.assertEqual(len(code_to_key), _n_keys(self.cfg))

    def test_code_to_key_indices_are_contiguous(self):
        _, code_to_key = create_key2seq_and_code2key(self.cfg, "training")
        self.assertEqual(sorted(code_to_key), list(range(_n_keys(self.cfg))))

    def test_sequences_round_trip_to_raw_codes(self):
        key_to_seq, code_to_key = create_key2seq_and_code2key(self.cfg, "training")
        raw = _raw_codes(self.cfg, "training")
        factor = _factor(self.cfg)
        for idx, key in code_to_key.items():
            with self.subTest(key=key):
                self.assertEqual(key_to_seq[key], np.repeat(raw[idx], factor).tolist())

    def test_presentation_rate_upsamples_sequences(self):
        self.cfg["speller"]["presentation_rate_hz"] = 30  # half of 60 -> factor 2
        key_to_seq, _ = create_key2seq_and_code2key(self.cfg, "training")
        raw = _raw_codes(self.cfg, "training")
        any_seq = next(iter(key_to_seq.values()))
        self.assertEqual(len(any_seq), raw.shape[1] * 2)

    def test_sequences_are_binary(self):
        key_to_seq, _ = create_key2seq_and_code2key(self.cfg, "training")
        for seq in key_to_seq.values():
            self.assertLessEqual(set(seq), {0.0, 1.0})


class SttSequenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()

    def test_stt_sequence_added_when_enabled(self):
        self.cfg["speller"]["stt"]["enabled"] = True
        key_to_seq, code_to_key = create_key2seq_and_code2key(self.cfg, "training")
        self.assertIn("stt", key_to_seq)
        self.assertNotIn("stt", code_to_key.values())
        self.assertEqual(key_to_seq["stt"][0], 1)
        self.assertEqual(set(key_to_seq["stt"][1:]), {0})

    def test_stt_absent_when_disabled(self):
        self.assertFalse(self.cfg["speller"]["stt"]["enabled"])
        key_to_seq, _ = create_key2seq_and_code2key(self.cfg, "training")
        self.assertNotIn("stt", key_to_seq)


class SubsetLayoutTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _write_layout(self, subset, layout, codes_file):
        path = self.tmp / "subset_layout.json"
        path.write_text(
            json.dumps({"subset": subset, "layout": layout, "codes_file": codes_file})
        )
        return path

    def test_subset_and_layout_applied_online(self):
        n_keys = _n_keys(self.cfg)
        subset = list(range(n_keys))
        layout = list(reversed(range(n_keys)))
        path = self._write_layout(subset, layout, self.cfg["online"]["codes_file"])
        self.cfg["decoder"]["decoder_subset_layout_file"] = str(path)

        key_to_seq, code_to_key = create_key2seq_and_code2key(self.cfg, "online")
        raw = _raw_codes(self.cfg, "online")
        factor = _factor(self.cfg)
        first_key = code_to_key[0]
        self.assertEqual(
            key_to_seq[first_key],
            np.repeat(raw[subset[layout[0]]], factor).tolist(),
        )

    def test_subset_layout_ignored_for_training(self):
        path = self._write_layout([0], [0], "does-not-match.txt")
        self.cfg["decoder"]["decoder_subset_layout_file"] = str(path)
        # Would raise on the codes_file assertion if applied; training skips it.
        _, code_to_key = create_key2seq_and_code2key(self.cfg, "training")
        self.assertEqual(len(code_to_key), _n_keys(self.cfg))

    def test_codes_file_mismatch_raises(self):
        path = self._write_layout([0], [0], "wrong_file.txt")
        self.cfg["decoder"]["decoder_subset_layout_file"] = str(path)
        with self.assertRaises(AssertionError):
            create_key2seq_and_code2key(self.cfg, "online")


if __name__ == "__main__":
    unittest.main()
