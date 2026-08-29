"""Tests for the decoding/selection handling logic.

These exercise ``handle_decoding_event`` (how a decoded key index becomes a
spelled character, a special action, and a highlight) and ``has_decoding_event``
(how numeric predictions are read off the decoder stream). The GUI/LSL side is
stubbed so only the application logic is under test.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._support import (
    FakeStreamWatcher,
    Speller,
    load_cfg,
    make_decode_speller,
    select,
)


class SpellingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()

    def test_letter_lowercase(self):
        spl = make_decode_speller(self.cfg, case_flag=False)
        select(spl, "A")
        self.assertEqual(spl._fields["text"], "a")

    def test_letter_uppercase(self):
        spl = make_decode_speller(self.cfg, case_flag=True)
        select(spl, "A")
        self.assertEqual(spl._fields["text"], "A")

    def test_space_key(self):
        spl = make_decode_speller(self.cfg, initial_text="hi")
        select(spl, "space")
        self.assertEqual(spl._fields["text"], "hi ")

    def test_backspace_key(self):
        spl = make_decode_speller(self.cfg, initial_text="hey")
        select(spl, "backspace")
        self.assertEqual(spl._fields["text"], "he")

    def test_clear_key(self):
        spl = make_decode_speller(self.cfg, initial_text="some text")
        select(spl, "clear")
        self.assertEqual(spl._fields["text"], "")

    def test_shift_toggles_case_without_typing(self):
        spl = make_decode_speller(self.cfg, case_flag=False, initial_text="x")
        select(spl, "shift")
        self.assertTrue(spl.case_flag)
        self.assertEqual(spl._fields["text"], "x")


class SpecialKeyRegressionTests(unittest.TestCase):
    """Special/punctuation keys used to raise KeyError because the highlight was
    keyed by the translated character instead of the upper-layout key name."""

    def setUp(self):
        self.cfg = load_cfg()

    def test_punctuation_lowercase_does_not_crash(self):
        cases = [("question", "/"), ("larger", "."), ("smaller", ",")]
        for key_name, expected_char in cases:
            with self.subTest(key=key_name):
                spl = make_decode_speller(self.cfg, case_flag=False)
                select(spl, key_name)  # must not raise
                self.assertEqual(spl._fields["text"], expected_char)
                self.assertIn(key_name, spl.highlights)
                self.assertEqual(spl.highlights[key_name], [0])

    def test_special_key_uppercase_does_not_crash(self):
        spl = make_decode_speller(self.cfg, case_flag=True)
        select(spl, "asterisk")  # "asterisk" -> "*"
        self.assertEqual(spl._fields["text"], "*")
        self.assertEqual(spl.highlights["asterisk"], [0])


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()

    def test_feedback_run_marker_uses_key_name(self):
        spl = make_decode_speller(self.cfg)
        select(spl, "backspace")
        self.assertEqual(len(spl.run_calls), 1)
        self.assertIn("key=backspace", spl.run_calls[0]["start_marker"])

    def test_highlight_restored_to_zero_after_selection(self):
        spl = make_decode_speller(self.cfg)
        select(spl, "A")
        self.assertTrue(all(v == [0] for v in spl.highlights.values()))

    def test_out_of_range_index_wraps(self):
        spl = make_decode_speller(self.cfg)
        n = len(spl.key_map)
        spl.last_selected_key_idx = n + 3  # beyond the number of keys
        spl.handle_decoding_event()  # must not raise
        self.assertEqual(spl.highlights[spl.key_map[3]], [0])

    def test_text2speech_key_triggers_speech(self):
        self.assertTrue(self.cfg["speller"]["text2speech"]["enabled"])
        spl = make_decode_speller(self.cfg, initial_text="hello")
        select(spl, "speaker")
        self.assertEqual(spl.tts_calls, ["hello"])
        self.assertFalse(spl.text2speech_flag)


class HasDecodingEventTests(unittest.TestCase):
    def _speller_with_stream(self, samples):
        spl = Speller.__new__(Speller)
        spl.last_selected_key_idx = None
        spl.decoder_sw = FakeStreamWatcher(samples)
        return spl

    def test_returns_last_nonnegative(self):
        spl = self._speller_with_stream([-1, -1, 5, -1, 3])
        self.assertTrue(spl.has_decoding_event())
        self.assertEqual(spl.last_selected_key_idx, 3)

    def test_ignores_all_negative(self):
        spl = self._speller_with_stream([-1, -1, -1])
        self.assertFalse(spl.has_decoding_event())
        self.assertIsNone(spl.last_selected_key_idx)

    def test_no_new_samples(self):
        spl = self._speller_with_stream([])
        self.assertFalse(spl.has_decoding_event())


if __name__ == "__main__":
    unittest.main()
