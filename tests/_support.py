"""Shared test support: import-time stubs, config loading, and helpers.

``cvep_speller.speller`` imports several heavy GUI/hardware dependencies at
import time (psychopy, pylsl, pyttsx3, google-generativeai, autocomplete,
dareplane_utils, fire). None are needed to exercise the pure application logic,
and most are impractical to install in CI, so this module registers lightweight
stand-ins in ``sys.modules`` *before* importing the speller module.

Every test module imports from here (never straight from ``cvep_speller`` at the
top level), which guarantees the stubs are installed first regardless of import
ordering.
"""

import copy
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import toml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make the project importable regardless of the discovery start/top dir.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_stub_modules() -> None:
    """Register fake modules for the uninstalled heavy dependencies (idempotent)."""

    if (
        "psychopy" in sys.modules
        and isinstance(sys.modules["psychopy"], types.ModuleType)
        and getattr(sys.modules["psychopy"], "_cvep_stub", False)
    ):
        return  # already installed

    # fire: only ``Fire`` is imported (and only used under __main__).
    fire = types.ModuleType("fire")
    fire.Fire = lambda *args, **kwargs: None
    sys.modules["fire"] = fire

    # psychopy: submodules are accessed via ``from psychopy import ...``.
    psychopy = types.ModuleType("psychopy")
    psychopy.__version__ = "0.0-test"
    psychopy._cvep_stub = True
    for sub in ("event", "misc", "monitors", "visual"):
        stub = MagicMock(name=f"psychopy.{sub}")
        setattr(psychopy, sub, stub)
        sys.modules[f"psychopy.{sub}"] = stub
    sys.modules["psychopy"] = psychopy

    # pylsl
    pylsl = types.ModuleType("pylsl")
    pylsl.StreamInfo = MagicMock(name="StreamInfo")
    pylsl.StreamOutlet = MagicMock(name="StreamOutlet")
    sys.modules["pylsl"] = pylsl

    # pyttsx3
    pyttsx3 = types.ModuleType("pyttsx3")
    pyttsx3.init = MagicMock(name="pyttsx3.init")
    sys.modules["pyttsx3"] = pyttsx3

    # autocomplete (n-gram engine). Tests configure predict()/load() as needed.
    autocomplete = types.ModuleType("autocomplete")
    autocomplete.load = MagicMock(name="autocomplete.load")
    autocomplete.predict = MagicMock(name="autocomplete.predict")
    sys.modules["autocomplete"] = autocomplete

    # google.generativeai -> imported as ``genai``. A MagicMock transparently
    # covers the nested ``genai.types.GenerationConfig`` access.
    google = types.ModuleType("google")
    genai = MagicMock(name="google.generativeai")
    google.generativeai = genai
    sys.modules["google"] = google
    sys.modules["google.generativeai"] = genai

    # dareplane_utils.logging.logger.get_logger -> a real stdlib logger.
    def get_logger(name="cvep-speller", add_console_handler=False, *args, **kwargs):
        return logging.getLogger(name)

    du = types.ModuleType("dareplane_utils")
    du_logging = types.ModuleType("dareplane_utils.logging")
    du_logging_logger = types.ModuleType("dareplane_utils.logging.logger")
    du_logging_logger.get_logger = get_logger
    du_logging.logger = du_logging_logger
    du.logging = du_logging

    du_sw = types.ModuleType("dareplane_utils.stream_watcher")
    du_sw_lsl = types.ModuleType("dareplane_utils.stream_watcher.lsl_stream_watcher")
    du_sw_lsl.StreamWatcher = MagicMock(name="StreamWatcher")
    du_sw.lsl_stream_watcher = du_sw_lsl
    du.stream_watcher = du_sw

    sys.modules["dareplane_utils"] = du
    sys.modules["dareplane_utils.logging"] = du_logging
    sys.modules["dareplane_utils.logging.logger"] = du_logging_logger
    sys.modules["dareplane_utils.stream_watcher"] = du_sw
    sys.modules["dareplane_utils.stream_watcher.lsl_stream_watcher"] = du_sw_lsl


_install_stub_modules()

# Safe to import the module under test now that the stubs are in place.
import cvep_speller.speller as speller  # noqa: E402
from cvep_speller.speller import (  # noqa: E402
    KEY_MAPPING,
    Speller,
    create_key2seq_and_code2key,
)

# Re-exported for the test modules so they import everything from here (which
# guarantees the stubs above are installed first).
__all__ = [
    "REPO_ROOT",
    "KEY_MAPPING",
    "Speller",
    "create_key2seq_and_code2key",
    "speller",
    "load_cfg",
    "make_bare_speller",
    "make_decode_speller",
    "index_of",
    "select",
    "FakeStreamWatcher",
]

_BASE_CFG = toml.load(REPO_ROOT / "configs" / "speller.toml")


def load_cfg() -> dict:
    """Return a fresh, path-resolved copy of the default speller config."""
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["speller"]["codes_dir"] = str(REPO_ROOT / "cvep_speller" / "codes")
    cfg["speller"]["images_dir"] = str(REPO_ROOT / "cvep_speller" / "images")
    return cfg


def make_bare_speller(cfg: dict) -> Speller:
    """A Speller instance with no __init__ run (no window/LSL needed)."""
    spl = Speller.__new__(Speller)
    spl.cfg = cfg
    return spl


def make_decode_speller(cfg, *, case_flag=False, initial_text="") -> Speller:
    """A Speller wired up so handle_decoding_event runs without a window.

    Text fields, feedback presentation (``run``), autocomplete, and TTS are
    replaced by in-memory stand-ins. Recorded calls are exposed on the instance
    as ``run_calls`` and ``tts_calls``.
    """
    spl = make_bare_speller(cfg)
    _, code_to_key = create_key2seq_and_code2key(cfg, "training")
    spl.key_map = code_to_key
    spl.all_keys = spl.set_all_keys(cfg)
    spl.case_flag = case_flag
    spl.next_autocomplete = ""
    spl.text2speech_flag = False
    spl.init_highlights_with_zero()

    spl._fields = {"text": initial_text, "autocomplete_text": ""}
    spl.get_text_field = lambda name: spl._fields[name]
    spl.set_text_field = lambda name, text: spl._fields.__setitem__(name, text)

    spl.run_calls = []
    spl.run = lambda **kwargs: spl.run_calls.append(kwargs)
    spl.start_autocomplete = lambda: None

    spl.tts_calls = []
    spl.text2speech = lambda text: spl.tts_calls.append(text)
    return spl


def index_of(code_to_key: dict, name: str) -> int:
    """The decoder index that maps to the given upper-layout key name."""
    return next(i for i, k in code_to_key.items() if k == name)


def select(spl: Speller, key_name: str) -> None:
    """Simulate the decoder selecting the key with the given key name."""
    spl.last_selected_key_idx = index_of(spl.key_map, key_name)
    spl.handle_decoding_event()


class FakeStreamWatcher:
    """Minimal stand-in for dareplane's StreamWatcher."""

    def __init__(self, samples):
        import numpy as np

        self._samples = np.asarray(samples, dtype=float)
        self.n_new = 0

    def update(self):
        self.n_new = len(self._samples)

    def unfold_buffer(self):
        return self._samples.reshape(-1, 1)
