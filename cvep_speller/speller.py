import random
import sys
import time
from pathlib import Path

import numpy as np
import psychopy
import toml
from dareplane_utils.logging.logger import get_logger
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from fire import Fire
from psychopy import event, misc, monitors, visual
from pylsl import StreamInfo, StreamOutlet

from cvep_speller.utils.logging import logger

# Windows does not allow / , : * ? " < > | ~ in file names (for the images)
KEY_MAPPING = {
    "slash": "/",
    "comma": ",",
    "colon": ":",
    "asterisk": "*",
    "question": "?",
    "quote": '"',
    "smaller": "<",
    "larger": ">",
    "bar": "|",
    "tilde": "~",
}


class Speller(object):
    """
    An object to create a speller with keys and text fields. Keys can alternate their background images according to
    specifically setup stimulation sequences.

    Parameters
    ----------
    screen_resolution: tuple[int, int]
        The screen resolution in pixels, provided as (width, height).
    width_cm: float
        The width of the screen in cm to compute pixels per degree.
    distance_cm: float
        The distance of the screen to the user in cm to compute pixels per degree.
    refresh_rate: int
        The screen refresh rate in Hz.
    cfg: dict
        config object containing context and paradigm configuration info as loaded
        from `./configs/speller.toml`.
    screen_id: int (default: 0)
        The screen number where to present the keyboard when multiple screens are used.
    background_color: tuple[float, float, float] (default: (0., 0., 0.)
        The keyboard's background color specified as list of RGB values.
    lsl_stream_name: str (default: "marker-stream")
        The name of the LSL stream to which markers of the keyboard are logged.
    quit_controls: list[str] (default: None)
        A list of keys that can be pressed to initiate quiting of the speller.
    full_screen: bool (default: True)
        Whether to present the speller in full screen mode.

    Attributes
    ----------
    keys: dict
        A dictionary of keys with a mapping of key name to a list of PsychoPy ImageStim.
    text_fields: dict
        A dictionary of text fields with a mapping of text field name to PsychoPy TextBox2.
    """

    keys: dict = dict()
    text_fields: dict = dict()

    def __init__(
        self,
        screen_resolution: tuple[int, int],
        width_cm: float,
        distance_cm: float,
        refresh_rate: int,
        cfg: dict,
        screen_id: int = 0,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        lsl_stream_name: str = "marker-stream",
        quit_controls: list[str] = None,
        full_screen: bool = True,
    ) -> None:
        self.cfg = cfg
        self.screen_resolution = screen_resolution
        self.full_screen = full_screen
        self.width_cm = width_cm
        self.distance_cm = distance_cm
        self.refresh_rate = refresh_rate
        self.quit_controls = quit_controls

        # Setup monitor
        self.monitor = monitors.Monitor(
            name="testMonitor", width=width_cm, distance=distance_cm
        )
        self.monitor.setSizePix(screen_resolution)

        # Setup window
        self.window = visual.Window(
            monitor=self.monitor,
            screen=screen_id,
            units="pix",
            size=screen_resolution,
            color=background_color,
            fullscr=full_screen,
            waitBlanking=False,
            allowGUI=False,
            # infoMsg="",
        )
        self.window.setMouseVisible(False)

        # Setup LSL stream
        info = StreamInfo(
            name=lsl_stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=lsl_stream_name,
        )
        self.outlet = StreamOutlet(info)

        self.last_selected_key_idx: int | None = None
        self.key_map: dict[int, str] = {}
        self.highlights: dict = {}
        self.decoder_sw = None

    def add_key(
        self,
        name: str,
        images: list,
        size: tuple[int, int],
        pos: tuple[int, int],
    ) -> None:
        """
        Add a key to the speller.

        Parameters
        ----------
        name: str
            The name of the key.
        images: list
            The list of images associated to the key. Note, index -1 is used for presenting feedback, and index -2 is
            used for cueing.
        size: tuple[int, int]
            The size of the key in pixels provided as (width, height).
        pos: tuple[int, int]
            The position of the key in pixels provided as (x, y).
        """
        assert name not in self.keys, "Adding a key with a name that already exists!"
        self.keys[name] = []
        for image in images:
            self.keys[name].append(
                visual.ImageStim(
                    win=self.window,
                    name=name,
                    image=image,
                    units="pix",
                    pos=pos,
                    size=size,
                    autoLog=False,
                )
            )

        # Set autoDraw to True for first default key to keep app visible
        self.keys[name][0].setAutoDraw(True)

    def add_text_field(
        self,
        name: str,
        text: str,
        size: tuple[int, int],
        pos: tuple[int, int],
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        text_color: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        alignment: str = "left",
    ) -> None:
        """
        Add a text field to the speller.

        Parameters
        ----------
        name: str
            The name of the text field.
        text: str
            The text to display on the text field.
        size: tuple[int, int]
            The size of the text field in pixels provided as (width, height).
        pos: tuple[int, int]
            The position of the text field in pixels provided as (x, y).
        background_color: tuple[float, float, float] (default: (0., 0., 0.))
            The background color of the text field  specified as list of RGB values.
        text_color: tuple[float, float, float] (default: (-1., -1., -1.))
            The text color of the text field  specified as list of RGB values.
        alignment: str (default: "left")
            The alignment of the text in the text field.
        """
        assert (
            name not in self.text_fields
        ), "Adding a text field with a name that already exists!"
        self.text_fields[name] = visual.TextBox2(
            win=self.window,
            name=name,
            text=text,
            units="pix",
            pos=pos,
            size=size,
            letterHeight=0.8 * size[1],
            fillColor=background_color,
            color=text_color,
            alignment=alignment,
            autoDraw=True,
            autoLog=False,
        )

    def connect_to_decoder_lsl_stream(self) -> None:
        name = self.cfg["run"]["online"]["decoder"]["lsl_stream_name"]
        logger.info(f'Connecting to decoder stream "{name=}".')
        self.decoder_sw = StreamWatcher(name=name)
        self.decoder_sw.connect_to_stream()

    def get_pixels_per_degree(
        self,
    ) -> float:
        """
        Get the pixels per degree of the screen.

        Returns
        -------
        ppd: float
            Pixels per degree of the screen.
        """
        return misc.deg2pix(degrees=1.0, monitor=self.monitor)

    def get_text_field(
        self,
        name: str,
    ) -> str:
        """
        Get the text of the text field.

        Parameters
        ----------
        name: str
            The name of the text field.

        Returns
        -------
        text: str
            The text of the text field.
        """
        assert (
            name in self.text_fields
        ), "Getting text of a text field with a name that does not exists!"
        return self.text_fields[name].text

    def set_text_field(
        self,
        name: str,
        text: str,
    ) -> None:
        """
        Set the text of a text field.

        Parameters
        ----------
        name: str
            The name of the text field.
        text: str
            The text of the text field.
        """
        assert (
            name in self.text_fields
        ), "Setting text of a text field with a name that does not exists!"
        self.text_fields[name].setText(text)
        self.window.flip()

    def log(
        self,
        marker: str,
        on_flip: bool = False,
    ) -> None:
        """
        Log a marker to the marker stream.

        Parameters
        ----------
        marker: str
            The marker to log.
        on_flip: bool (default: False)
            Whether to log on the next frame flip.
        """
        if on_flip:
            self.window.callOnFlip(self.outlet.push_sample, [marker])
        else:
            self.outlet.push_sample([marker])

    def run(
        self,
        sequences: dict,
        duration: float = None,
        start_marker: str = None,
        stop_marker: str = None,
    ) -> None:
        """
        Run a stimulation phase of the speller, which makes the keys flash according to specific sequences.

        Parameters
        ----------
        sequences: dict
            A dictionary containing the stimulus sequences per key.
        duration: float (default: None)
            The duration of the stimulation in seconds. If None, the duration of the first key in the dictionary is
            used.
        start_marker: str (default: None)
            The marker to log when stimulation starts. If None, no marker is logged.
        stop_marker: str (default: None)
            The marker to log when stimulation stops. If None, no marker is logged.
        """

        # Set number of frames
        if duration is None:
            n_frames = len(sequences[list(sequences.keys())[0]])
        else:
            n_frames = int(duration * self.refresh_rate)

        # Set autoDraw to False for full control
        for key in self.keys.values():
            key[0].setAutoDraw(False)

        # Send start marker
        if start_marker is not None:
            self.log(start_marker, on_flip=True)

        # Loop frame flips
        for i in range(n_frames):
            stime = time.time()

            # Check quiting
            if i % 60 == 0:
                if len(event.getKeys(keyList=self.quit_controls)) > 0:
                    self.quit()
                    break

            # Check selection marker
            if self.decoder_sw is not None:
                if self.has_decoding_event():
                    self.handle_decoding_event()
                    break

            # Present keys with color depending on code state
            for name, code in sequences.items():
                self.keys[name][code[i % len(code)]].draw()

            # Check if frame flip can happen within a frame
            etime = time.time() - stime
            if etime >= 1 / self.refresh_rate:
                logger.warn(f"Frame flip took too long ({etime:.6f}), dropping frames!")

            self.window.flip()
        else:
            logger.debug(f"All {n_frames=} shown.")

        # Send stop marker
        if stop_marker is not None:
            self.log(stop_marker)

        # Set autoDraw to True to keep speller visible
        for key in self.keys.values():
            key[0].setAutoDraw(True)

    def quit(
        self,
    ) -> None:
        """
        Quit the speller, close the window.
        """
        for key in self.keys.values():
            key[0].setAutoDraw(True)

        if self.window is not None:
            self.window.flip()
            self.window.setMouseVisible(True)
            self.window.close()

    def has_decoding_event(self) -> bool:
        """
        Check if the LSL stream contained a `speller_select <key_idx>` marker.
        """

        # disregard all previous data
        self.decoder_sw.n_new = 0
        self.decoder_sw.update()

        if self.decoder_sw.n_new != 0:
            prediction = self.decoder_sw.unfold_buffer()[
                -self.decoder_sw.n_new:
            ].flatten()

            logger.debug(f"Received: prediction={prediction}")

            # selections = [
            #     c
            #     for c in prediction
            #     if isinstance(c, str) and c.startswith("speller_select")
            # ]
            selections = prediction[prediction >= 0]

            if len(selections) > 0:
                # In case multiple decode markers arrive -> consider only last
                # self.last_selected_key_idx = int(
                #     selections[-1].replace("speller_select", "")
                # )
                self.last_selected_key_idx = int(selections[-1])
                logger.debug(f"Selection: {self.last_selected_key_idx}, {selections=}")
                return True

        return False

    def handle_decoding_event(self) -> None:
        # Decoding
        logger.info("Waiting for decoding")

        prediction = self.last_selected_key_idx
        prediction_key = self.key_map[prediction]
        logger.debug(
            f"Decoding: prediction={prediction} prediction_key={prediction_key}"
        )

        # Spelling
        text = self.get_text_field("text")
        symbol = prediction_key
        if symbol in KEY_MAPPING:
            symbol = KEY_MAPPING[symbol]
        if symbol == "<":
            text = text[:-1]
        elif symbol == "space":
            text = text + " "
        else:
            text += symbol
        self.set_text_field(name="text", text=text)
        logger.debug(f"Feedback: symbol={symbol} text={text}")

        # Feedback
        logger.info(f"Presenting feedback {prediction_key} ({prediction})")
        self.highlights[prediction_key] = [-1]
        self.run(
            sequences=self.highlights,
            duration=self.cfg["speller"]["timing"]["feedback_s"],
            start_marker=f'{self.cfg["speller"]["markers"]["feedback_start"]};label={prediction};key={prediction_key}',
            stop_marker=self.cfg["speller"]["markers"]["feedback_stop"],
        )
        self.highlights[prediction_key] = [0]

    def init_highlights_with_zero(self) -> None:
        # Setup highlights
        self.highlights = dict()
        for row in self.cfg["speller"]["keys"]["keys"]:
            for key in row:
                self.highlights[key] = [0]
        self.highlights["stt"] = [0]


def setup_speller(cfg: dict) -> Speller:

    # Setup speller
    speller = Speller(
        screen_resolution=cfg["speller"]["screen"]["resolution"],
        width_cm=cfg["speller"]["screen"]["width_cm"],
        distance_cm=cfg["speller"]["screen"]["distance_cm"],
        refresh_rate=cfg["speller"]["screen"]["refresh_rate_hz"],
        screen_id=cfg["speller"]["screen"]["id"],
        full_screen=cfg["speller"]["screen"]["full_screen"],
        background_color=cfg["speller"]["screen"]["background_color"],
        lsl_stream_name=cfg["run"]["lsl_stream_name"],
        quit_controls=cfg["speller"]["controls"]["quit"],
        cfg=cfg,
    )
    ppd = speller.get_pixels_per_degree()

    # Add stimulus timing tracker at left top of the screen
    x_pos = int(
        -cfg["speller"]["screen"]["resolution"][0] / 2
        + cfg["speller"]["stt"]["width_dva"] / 2 * ppd
    )
    y_pos = int(
        cfg["speller"]["screen"]["resolution"][1] / 2
        - cfg["speller"]["stt"]["height_dva"] / 2 * ppd
    )
    speller.add_key(
        name="stt",
        images=[
            Path(cfg["speller"]["images_dir"]) / f"{color}.png"
            for color in cfg["speller"]["stt"]["colors"]
        ],
        size=(
            int(cfg["speller"]["stt"]["width_dva"] * ppd),
            int(cfg["speller"]["stt"]["height_dva"] * ppd),
        ),
        pos=(x_pos, y_pos),
    )

    # Add text field at the top of the screen containing spelled text
    x_pos = int(cfg["speller"]["stt"]["width_dva"] * ppd / 2)
    y_pos = int(
        cfg["speller"]["screen"]["resolution"][1] / 2
        - cfg["speller"]["text_fields"]["height_dva"] * ppd / 2
    )
    speller.add_text_field(
        name="text",
        text="",
        size=(
            int(
                cfg["speller"]["screen"]["resolution"][0]
                - cfg["speller"]["stt"]["width_dva"] * ppd
            ),
            int(cfg["speller"]["text_fields"]["height_dva"] * ppd),
        ),
        pos=(x_pos, y_pos),
        background_color=cfg["speller"]["text_fields"]["background_color"],
    )

    # Add text field at the bottom of the screen containing system messages
    x_pos = 0
    y_pos = int(
        -cfg["speller"]["screen"]["resolution"][1] / 2
        + cfg["speller"]["text_fields"]["height_dva"] * ppd / 2
    )
    speller.add_text_field(
        name="messages",
        text="",
        size=(
            cfg["speller"]["screen"]["resolution"][0],
            int(cfg["speller"]["text_fields"]["height_dva"] * ppd),
        ),
        pos=(x_pos, y_pos),
        background_color=cfg["speller"]["text_fields"]["background_color"],
        alignment="center",
    )

    # Add keys
    for y in range(len(cfg["speller"]["keys"]["keys"])):
        for x in range(len(cfg["speller"]["keys"]["keys"][y])):
            x_pos = int(
                (x - len(cfg["speller"]["keys"]["keys"][y]) / 2 + 0.5)
                * (
                    cfg["speller"]["keys"]["width_dva"]
                    + cfg["speller"]["keys"]["space_dva"]
                )
                * ppd
            )
            y_pos = int(
                -(y - len(cfg["speller"]["keys"]["keys"]) / 2)
                * (
                    cfg["speller"]["keys"]["height_dva"]
                    + cfg["speller"]["keys"]["space_dva"]
                )
                * ppd
                - cfg["speller"]["text_fields"]["height_dva"] * ppd
            )
            if y == 0 or y == 1:
                x_pos -= int(0.5 * cfg["speller"]["keys"]["width_dva"] * ppd)
            if y == 3:
                x_pos += int(0.25 * cfg["speller"]["keys"]["width_dva"] * ppd)
            if y == 4:
                x_pos -= int(0.25 * cfg["speller"]["keys"]["width_dva"] * ppd)
            if cfg["speller"]["keys"]["keys"][y][x] == "space":
                images = [
                    Path(cfg["speller"]["images_dir"]) / f"{color}.png"
                    for color in cfg["speller"]["keys"]["colors"]
                ]
            else:
                images = [
                    Path(cfg["speller"]["images_dir"])
                    / f'{cfg["speller"]["keys"]["keys"][y][x]}_{color}.png'
                    for color in cfg["speller"]["keys"]["colors"]
                ]
            speller.add_key(
                name=cfg["speller"]["keys"]["keys"][y][x],
                images=images,
                size=(
                    int(cfg["speller"]["keys"]["width_dva"] * ppd),
                    int(cfg["speller"]["keys"]["height_dva"] * ppd),
                ),
                pos=(x_pos, y_pos),
            )

    speller.init_highlights_with_zero()

    return speller


def create_key2seq_and_code2key(cfg: dict) -> tuple[dict, dict]:

    # Setup code sequences
    codes = np.load(
        Path(cfg["speller"]["codes_dir"]) / Path(cfg["speller"]["codes_file"])
    )["codes"]
    codes = np.repeat(
        codes,
        int(
            cfg["speller"]["screen"]["refresh_rate_hz"]
            / cfg["speller"]["presentation_rate_hz"]
        ),
        axis=1,
    )
    key_to_sequence = dict()
    code_to_key = dict()
    i_code = 0
    for row in cfg["speller"]["keys"]["keys"]:
        for key in row:
            key_to_sequence[key] = codes[i_code, :].tolist()
            code_to_key[i_code] = key
            i_code += 1
    key_to_sequence["stt"] = [1] + [0] * int(
        (1 + cfg["speller"]["timing"]["trial_s"])
        * cfg["speller"]["screen"]["refresh_rate_hz"]
    )

    return key_to_sequence, code_to_key


def run_speller_paradigm(
    phase: str = "training",
    config_path: Path = Path("./configs/speller.toml"),  # relative to the project root
) -> int:
    """
    Run the speller in a particular phase (training or online).

    Parameters
    ----------
    phase: str (default: "training")
        The phase of the speller being either training or online. During training, the user is cued to attend to a
        random target key every trail. During online, the user attends to their target, while their EEG is decoded and
        the decoded key is used to perform an action (e.g., add a symbol to a sentence, backspace, etc.). In the online
        phase, the speller will continuously query an LSL marker stream to look for a potential decoding result from
        the decoder module. If the stream contains a marker `speller_select <key_idx>`, the speller will
        stop the presentation and will show the selected symbol.
    config_path: Path (default: "./configs/speller.toml")
        The path to the configuration file containing session specific hyperparameters for the speller setup.

    Returns
    -------
    flag: int
        Whether the process ran without errors or with.
    """
    cfg = toml.load(config_path)
    speller = setup_speller(cfg)

    if phase != "training":
        speller.connect_to_decoder_lsl_stream()

    key_to_sequence, code_to_key = create_key2seq_and_code2key(cfg)
    speller.key_map = code_to_key
    n_classes = len(code_to_key)

    # Wait to start run
    logger.info("Waiting for button press to start")
    speller.set_text_field(name="messages", text="Press button to start.")
    event.waitKeys(keyList=cfg["speller"]["controls"]["continue"])
    speller.set_text_field(name="messages", text="")

    # Log info
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    speller.log(marker=f"version;python={python_version};psychopy={psychopy.__version__}")
    speller.log(marker=f"setup;codes={key_to_sequence};labels={code_to_key}")

    # Start run
    logger.info("Starting")
    speller.log(marker="start_run")
    speller.set_text_field(name="messages", text="Starting...")
    speller.run(sequences=speller.highlights, duration=5.0)
    speller.set_text_field(name="messages", text="")

    # Loop trials
    n_trials = cfg["run"][phase]["n_trials"]
    for i_trial in range(n_trials):
        logger.info(f"Initiating trial {1 + i_trial}/{n_trials}")

        if phase == "training":
            # Set a random target
            target = random.randint(0, n_classes - 1)
            target_key = code_to_key[target]

            # Cue
            logger.info(f"Cueing {target_key} ({target})")
            speller.highlights[target_key] = [-2]
            speller.run(
                sequences=speller.highlights,
                duration=cfg["speller"]["timing"]["cue_s"],
                start_marker=f'{cfg["speller"]["markers"]["cue_start"]};label={target};key={target_key}',
                stop_marker=cfg["speller"]["markers"]["cue_stop"],
            )
            speller.highlights[target_key] = [0]

        # Trial
        logger.info("Starting stimulation")
        speller.run(
            sequences=key_to_sequence,
            duration=cfg["speller"]["timing"]["trial_s"],
            start_marker=f'{cfg["speller"]["markers"]["trial_start"]}',
            stop_marker=cfg["speller"]["markers"]["trial_stop"],
        )

        # Inter-trial time
        logger.info("Inter-trial interval")
        speller.run(
            sequences=speller.highlights,
            duration=cfg["speller"]["timing"]["iti_s"],
            start_marker=f'{cfg["speller"]["markers"]["iti_start"]}',
            stop_marker=cfg["speller"]["markers"]["iti_stop"],
        )

        if phase == "online" and speller.get_text_field("text").endswith(cfg["speller"]["quit_phrase"]):
            break

    # Wait to stop
    logger.info("Waiting for button press to stop")
    speller.set_text_field(name="messages", text="Press button to stop.")
    event.waitKeys(keyList=cfg["speller"]["controls"]["continue"])
    speller.set_text_field(name="messages", text="")

    # Stop run
    logger.info("Stopping")
    speller.log(marker="stop_run")
    speller.set_text_field(name="messages", text="Stopping...")
    speller.run(sequences=speller.highlights, duration=5.0)
    speller.set_text_field(name="messages", text="")
    speller.quit()

    return 0


# make this the cli starting point
def cli_run(
    phase: str = "training",
    config_path: Path = Path("./configs/speller.toml"),  # relative to the project root
    log_level: int = 30,
) -> None:
    """
    Run the speller in a particular phase (training or online).

    Parameters
    ----------
    phase: str (default: "training")
        The phase of the speller being either training or online. During training, the user is cued to attend to a
        random target key every trail. During online, the user attends to their target, while their EEG is decoded and
        the decoded key is used to perform an action (e.g., add a symbol to a sentence, backspace, etc.). In the online
        phase, the speller will continuously query an LSL marker stream to look for a potential decoding result from
        the decoder module. If the stream contains a marker `speller_select <key_idx>`, the speller will
        stop the presentation and will show the selected symbol.
    config_path: Path (default: "./configs/speller.toml")
        The path to the configuration file containing session specific hyperparameters for the speller setup.
    log_level : int (default: 30)
        The logging level to use.

    """

    # activate the console logging if started via CLI
    logger = get_logger("cvep-speller", add_console_handler=True)
    logger.setLevel(log_level)

    run_speller_paradigm(phase=phase, config_path=config_path)


if __name__ == "__main__":
    Fire(cli_run)
