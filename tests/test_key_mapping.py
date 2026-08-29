"""Tests for the static key/character mappings."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._support import KEY_MAPPING, load_cfg, make_bare_speller


class KeyMappingTests(unittest.TestCase):
    def test_values_are_unique(self):
        values = list(KEY_MAPPING.values())
        self.assertEqual(len(values), len(set(values)))

    def test_covers_filename_unsafe_characters(self):
        for char in ["/", ":", "*", "?", '"', "<", ">", "|", "~", "\\"]:
            self.assertIn(char, KEY_MAPPING.values())

    def test_no_value_is_also_a_key(self):
        # Guards against a double-translation bug: applying KEY_MAPPING twice
        # must be idempotent, which only holds if no mapped char is itself a key.
        self.assertTrue(set(KEY_MAPPING.keys()).isdisjoint(set(KEY_MAPPING.values())))


class SetAllKeysTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()
        self.all_keys = make_bare_speller(self.cfg).set_all_keys(self.cfg)

    def test_mapping_is_bidirectional(self):
        self.assertEqual(self.all_keys["A"], "a")
        self.assertEqual(self.all_keys["a"], "A")
        self.assertEqual(self.all_keys["!"], "1")
        self.assertEqual(self.all_keys["1"], "!")
        self.assertEqual(self.all_keys["question"], "slash")
        self.assertEqual(self.all_keys["slash"], "question")

    def test_covers_every_configured_key(self):
        upper = self.cfg["speller"]["keys"]["keys_upper"]
        lower = self.cfg["speller"]["keys"]["keys_lower"]
        for row_u, row_l in zip(upper, lower):
            for ku, kl in zip(row_u, row_l):
                with self.subTest(upper=ku, lower=kl):
                    self.assertEqual(self.all_keys[ku], kl)
                    self.assertEqual(self.all_keys[kl], ku)

    def test_config_layout_rows_have_matching_lengths(self):
        upper = self.cfg["speller"]["keys"]["keys_upper"]
        lower = self.cfg["speller"]["keys"]["keys_lower"]
        self.assertEqual(len(upper), len(lower))
        for row_u, row_l in zip(upper, lower):
            self.assertEqual(len(row_u), len(row_l))


if __name__ == "__main__":
    unittest.main()
