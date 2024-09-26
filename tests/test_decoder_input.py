import threading
import time

import numpy as np
import pylsl
import pytest
import toml
from dareplane_utils.logging.logger import get_logger
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher

from cvep_speller.speller import create_key2seq_and_code2key, setup_speller

logger = get_logger("cvep-speller", add_console_handler=True)
logger.setLevel("DEBUG")

CFG = toml.load("./configs/speller.toml")


def provide_lsl_stream(stop_event: threading.Event, srate: float = 5):
    """Send a decoding signal every 1.5s looping over the alphabet"""
    # block print to stdout to suppress noise
    outlet = pylsl.StreamOutlet(
        pylsl.StreamInfo(
            CFG["run"]["online"]["decoder_lsl_stream_name"],
            "EEG",
            1,
            srate,
            "string",
            "cvep_decoder_stream",
        )
    )

    data = np.hstack(CFG["speller"]["keys"]["keys"]).flatten()

    isampl = 0
    ichar = 0
    nsent = 0
    tstart = time.time_ns()
    t_last_decode = time.time()
    while not stop_event.is_set():
        dt = time.time_ns() - tstart
        req_samples = int((dt / 1e9) * srate) - nsent
        if req_samples > 0:

            # logger.debug(f"Sending {req_samples=} samples")
            # a decoding signal
            if time.time() - t_last_decode > 1.0:
                t_last_decode = time.time()
                outlet.push_sample([f"speller_select {ichar}"])
                # logger.debug(f"Sent decoding signal for {data[ichar]=}, {ichar=}")

                ichar = (ichar + 1) % data.shape[0]

            else:
                outlet.push_chunk([[data[ichar]] * req_samples])
                nsent += req_samples
                isampl = (isampl + req_samples) % data.shape[0]  # wrap around

        time.sleep(1 / srate)


@pytest.fixture(scope="session")  # only create once for all tests
def spawn_lsl_data_stream() -> threading.Event:

    stop_event = threading.Event()
    stop_event.clear()
    th = threading.Thread(target=provide_lsl_stream, args=(stop_event,))
    th.start()

    yield stop_event

    # teardown
    stop_event.set()
    th.join()


def test_decoder_selection(spawn_lsl_data_stream):

    # Assert that the LSL stream is up and running
    n_tries = 50
    sw = StreamWatcher(CFG["run"]["online"]["decoder_lsl_stream_name"])
    sw.connect_to_stream()
    i_try = 0
    while sw.n_new < 1 and n_tries > 0:
        time.sleep(0.2)
        i_try += 1
        sw.update()
    if sw.n_new == 0:
        raise TimeoutError("LSL stream did not start in time")

    speller = setup_speller(CFG)
    speller.connect_to_decoder_lsl_stream()

    if speller.decoder_sw is None:
        breakpoint()

    key_to_seq, code_to_key = create_key2seq_and_code2key(CFG)
    speller.key_map = code_to_key

    decoded_idx = []
    for i in range(3):
        # sw.update()
        # sw.n_new = 0

        speller.run(
            sequences=key_to_seq,
            duration=2,  # the mockup decoder stream should deliver a signal every 1.5s
            start_marker="start",
            stop_marker="stop",
        )
        if speller.last_selected_key_idx is not None:
            decoded_idx.append(speller.last_selected_key_idx)

        # sw.update()
        # logger.info(f"Markers of this period: {sw.unfold_buffer()[-sw.n_new:]}")

    assert len(decoded_idx) == 3
    assert decoded_idx[0] != decoded_idx[1] != decoded_idx[2]

    # assert that the indeces are at most 6 appart (new maker every second from LSL = loop with 3 times 2s intervals)
    assert decoded_idx[0] + 6 >= decoded_idx[-1]
