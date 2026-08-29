"""Tests for the offline (n-gram) and online (LLM) autocomplete helpers.

The underlying engines are stubbed (see _support), so these tests only cover the
text-wrangling logic in the Speller wrappers.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._support import load_cfg, make_bare_speller, speller


class OfflineAutocompleteTests(unittest.TestCase):
    def setUp(self):
        self.spl = make_bare_speller(load_cfg())
        # Reset the shared autocomplete stub between tests.
        speller.autocomplete.predict.reset_mock(return_value=True)

    def test_returns_text_unchanged_after_space(self):
        # A trailing space means "wait for the next word" -> no prediction.
        self.assertEqual(self.spl.offline_autocomplete("hello "), "hello ")

    def test_single_word_is_capitalized(self):
        speller.autocomplete.predict.return_value = [["hello", 42]]
        self.assertEqual(self.spl.offline_autocomplete("hel"), "Hello")

    def test_multi_word_completes_last_word(self):
        speller.autocomplete.predict.return_value = [["world", 7]]
        self.assertEqual(self.spl.offline_autocomplete("hello wo"), "hello world")

    def test_falls_back_to_input_when_no_prediction(self):
        speller.autocomplete.predict.return_value = []  # no match, even for "the"
        self.assertEqual(self.spl.offline_autocomplete("zxq"), "zxq")


class OnlineAutocompleteTests(unittest.TestCase):
    def test_strips_engine_output(self):
        class _Resp:
            text = "  completed sentence  "

        class _Engine:
            def generate_content(self, text, generation_config=None):
                return _Resp()

        spl = make_bare_speller(load_cfg())
        spl.autocomplete_engine = _Engine()
        self.assertEqual(spl.online_autocomplete("some text"), "completed sentence")


if __name__ == "__main__":
    unittest.main()
