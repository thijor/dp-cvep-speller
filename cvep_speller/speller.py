import json
import random

import numpy as np
from psychopy import visual, event, monitors, misc
from pylsl import StreamInfo, StreamOutlet

from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher


SCREEN = 0
SCREEN_SIZE = (1920, 1080)
SCREEN_WIDTH = 53.0
SCREEN_DISTANCE = 60.0
SCREEN_COLOR = (0, 0, 0)
SCREEN_FR = 60
SCREEN_PR = 60

STT_WIDTH = 2.2
STT_HEIGHT = 2.2

TEXT_FIELD_HEIGHT = 2.3

KEY_WIDTH = 3.0
KEY_HEIGHT = 3.0
KEY_SPACE = 0.5
KEY_COLORS = ["black", "white", "green", "blue"]
KEYS = [
    ["!", "@", "#", "$", "%", "^", "&", "asterisk", "(", ")", "_", "+"],  # 12
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="],  # 12
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]"],  # 12
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":", "quote", "bar"],  # 12
    ["tilde", "Z", "X", "C", "V", "B", "N", "M", "comma", ".", "question", "slash"],  # 12
    ["smaller", "space", "larger"]]  # 3
KEY_MAPPING = {  # Windows does not allow / , : * ? " < > | ~ in file names
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

KEYS_QUIT = ["q", "escape"]
KEYS_CONTINUE = ["c"]

CODE_FILE = "mgold_61_6521.npz"

TIME_CUE = 0.8
TIME_TRIAL = 4.2
TIME_FEEDBACK = 0.5
TIME_ITI = 0.5

N_TRAINING_TRIALS = 10
N_ONLINE_TRIALS = 100


class Keyboard(object):
    """
    An object to create a keyboard with keys and text fields. Keys can alternate their foreground images according to
    specifically setup stimulation sequences.

    Parameters
    ----------
    size: tuple[int, int]
        The screen resolution in pixels, provided as (width, height).
    width: float
        The width of the screen in cm to compute pixels per degree.
    distance: float
        The distance of the screen to the user in cm to compute pixels per degree.
    fr: int
        The screen refresh rate in Hz.
    screen: int (default: 0)
        The screen number where to present the keyboard when multiple screens are used.
    background_color: tuple[float, float, float] (default: (0., 0., 0.)
        The keyboard's background color specified as list of RGB values.

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
            size: tuple[int, int],
            width: float,
            distance: float,
            fr: int,
            screen: int = 0,
            background_color: tuple[float, float, float] = (0., 0., 0.),
    ) -> None:
        self.size = size
        self.width = width
        self.distance = distance
        self.fr = fr

        # Setup monitor
        self.monitor = monitors.Monitor(name="testMonitor", width=width, distance=distance)
        self.monitor.setSizePix(size)

        # Setup window
        self.window = visual.Window(
            monitor=self.monitor, screen=screen, units="pix", size=size, color=background_color, fullscr=True,
            waitBlanking=False, allowGUI=False, infoMsg=""
        )
        self.window.setMouseVisible(False)

        # Setup LSL stream
        info = StreamInfo(
            name="MarkerStream", type="Markers", channel_count=1, nominal_srate=0, channel_format="string",
            source_id="MarkerStream")
        self.outlet = StreamOutlet(info)

    def add_key(
            self,
            name: str,
            images: list,
            size: tuple[int, int],
            pos: tuple[int, int],
    ) -> None:
        """
        Add a key to the keyboard.

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
            self.keys[name].append(visual.ImageStim(
                win=self.window, name=name, image=image, units="pix", pos=pos, size=size, autoLog=False)
            )

        # Set autoDraw to True for first default key to keep app visible
        self.keys[name][0].setAutoDraw(True)

    def add_text_field(
            self,
            name: str,
            text: str,
            size: tuple[int, int],
            pos: tuple[int, int],
            background_color: tuple[float, float, float] = (0., 0., 0.),
            text_color: tuple[float, float, float] = (-1., -1., -1.),
            alignment: str = "left",
    ) -> None:
        """
        Add a text field to the keyboard.

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
        assert name not in self.text_fields, "Adding a text field with a name that already exists!"
        self.text_fields[name] = visual.TextBox2(
            win=self.window, name=name, text=text, units="pix", pos=pos, size=size, letterHeight=0.8*size[1],
            fillColor=background_color, color=text_color, alignment=alignment, autoDraw=True, autoLog=False)

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
        assert name in self.text_fields, "Getting text of a text field with a name that does not exists!"
        return self.text_fields[name].text

    def set_text_field(
            self,
            name,
            text,
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
        assert name in self.text_fields, "Setting text of a text field with a name that does not exists!"
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
            stop_marker: str = None
    ) -> None:
        """
        Run a stimulation phase of the keyboard, which makes the keys flash according to specific sequences.

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
            n_frames = int(duration * self.fr)

        # Set autoDraw to False for full control
        for key in self.keys.values():
            key[0].setAutoDraw(False)

        # Send start marker
        if start_marker is not None:
            self.log(start_marker, on_flip=True)

        # Loop frame flips
        for i in range(n_frames):

            # Check quiting
            if i % 60 == 0:
                if len(event.getKeys(keyList=KEYS_QUIT)) > 0:
                    self.quit()

            # Present keys with color depending on code state
            for name, code in sequences.items():
                self.keys[name][code[i % len(code)]].draw()
            self.window.flip()

        # Send stop marker
        if stop_marker is not None:
            self.log(stop_marker)

        # Set autoDraw to True to keep speller visible
        for key in self.keys.values():
            key[0].setAutoDraw(True)
        self.window.flip()

    def quit(
            self,
    ) -> None:
        """
        Quit the keyboard, close the window.
        """
        for key in self.keys.values():
            key[0].setAutoDraw(True)
        self.window.flip()
        self.window.setMouseVisible(True)
        self.window.close()


def run(phase):
    """
    Run the keyboard speller in a particular phase (training or online).

    Parameters
    ----------
    phase: str (default: "training")
        The phase of the speller being either training or online. During training, the user is cued to attend to a
        random target key every trail. During online, the user attends to their target, while their EEG is decoded and
        the decoded key is used to perform an action (e.g., add a symbol to a sentence, backspace, etc.).
    """
    # Setup keyboard
    keyboard = Keyboard(
        size=SCREEN_SIZE, width=SCREEN_WIDTH, distance=SCREEN_DISTANCE, fr=SCREEN_FR, screen=SCREEN,
        background_color=SCREEN_COLOR
    )
    ppd = keyboard.get_pixels_per_degree()

    # Add stimulus timing tracker at left top of the screen
    x_pos = int(-SCREEN_SIZE[0] / 2 + STT_WIDTH / 2 * ppd)
    y_pos = int(SCREEN_SIZE[1] / 2 - STT_HEIGHT / 2 * ppd)
    keyboard.add_key(
        name="stt", images=["images/black.png", "images/white.png"], size=(int(STT_WIDTH * ppd), int(STT_HEIGHT * ppd)),
        pos=(x_pos, y_pos)
    )

    # Add text field at the top of the screen containing spelled text
    x_pos = int(STT_WIDTH * ppd / 2)
    y_pos = int(SCREEN_SIZE[1] / 2 - TEXT_FIELD_HEIGHT * ppd / 2)
    keyboard.add_text_field(
        name="text", text="", size=(int(SCREEN_SIZE[0] - STT_WIDTH * ppd), int(TEXT_FIELD_HEIGHT * ppd)),
        pos=(x_pos, y_pos), background_color=(-0.05, -0.05, -0.05)
    )

    # Add text field at the bottom of the screen containing system messages
    x_pos = 0
    y_pos = int(-SCREEN_SIZE[1] / 2 + TEXT_FIELD_HEIGHT * ppd / 2)
    keyboard.add_text_field(
        name="messages", text="", size=(SCREEN_SIZE[0], int(TEXT_FIELD_HEIGHT * ppd)), pos=(x_pos, y_pos),
        background_color=(-0.05, -0.05, -0.05), alignment="center"
    )

    # Add keys
    for y in range(len(KEYS)):
        for x in range(len(KEYS[y])):
            x_pos = int((x - len(KEYS[y]) / 2 + 0.5) * (KEY_WIDTH + KEY_SPACE) * ppd)
            y_pos = int(-(y - len(KEYS) / 2) * (KEY_HEIGHT + KEY_SPACE) * ppd - TEXT_FIELD_HEIGHT * ppd)
            if y == 0 or y == 1:
                x_pos -= int(0.5 * KEY_WIDTH * ppd)
            if y == 3:
                x_pos += int(0.25 * KEY_WIDTH * ppd)
            if y == 4:
                x_pos -= int(0.25 * KEY_WIDTH * ppd)
            if KEYS[y][x] == "space":
                images = [f"images/{color}.png" for color in KEY_COLORS]
            else:
                images = [f"images/{KEYS[y][x]}_{color}.png" for color in KEY_COLORS]
            keyboard.add_key(
                name=KEYS[y][x], images=images,
                size=(int(KEY_WIDTH * ppd), int(KEY_HEIGHT * ppd)), pos=(x_pos, y_pos)
            )

    # Setup code sequences
    tmp = np.load(f"codes/{CODE_FILE}")["codes"]
    key_to_sequence = dict()
    code_to_key = dict()
    i_code = 0
    for row in KEYS:
        for key in row:
            key_to_sequence[key] = tmp[i_code, :].tolist()
            code_to_key[i_code] = key
            i_code += 1
    n_classes = i_code
    key_to_sequence["stt"] = [1] + [0] * int((1 + TIME_TRIAL) * SCREEN_FR)

    # Setup highlights
    highlights = dict()
    for row in KEYS:
        for key in row:
            highlights[key] = [0]
    highlights["stt"] = [0]

    # Log configuration
    keyboard.log(marker=json.dumps({"codes": key_to_sequence}))

    if phase == "training":
        n_trials = N_TRAINING_TRIALS
    elif phase == "online":
        n_trials = N_ONLINE_TRIALS
    else:
        raise Exception("Unknown phase:", phase)

    # connect to decoder stream
    if phase == "online":
        sw = StreamWatcher(name="decoder")
        sw.connect_to_stream()

    # Wait to start run
    keyboard.set_text_field(name="messages", text="Press button to start.")
    event.waitKeys(keyList=KEYS_CONTINUE)
    keyboard.set_text_field(name="messages", text="")

    # Start run
    keyboard.log(marker="start_run")
    keyboard.set_text_field(name="messages", text="Starting...")
    keyboard.run(sequences=highlights, duration=5.0)
    keyboard.set_text_field(name="messages", text="")

    # Loop trials
    for i_trial in range(n_trials):

        if phase == "training":
            # Set a random target
            target = random.randint(0, n_classes - 1)
            target_key = code_to_key[target]
            keyboard.log(json.dumps({"i_trial": i_trial, "target": target, "target_key": target_key}))

            # Cue
            highlights[target_key] = [-2]
            keyboard.run(
                sequences=highlights, duration=TIME_CUE, start_marker="start_cue", stop_marker="stop_cue"
            )
            highlights[target_key] = [0]

        # Trial
        keyboard.run(
            sequences=key_to_sequence, duration=TIME_TRIAL, start_marker="start_trial", stop_marker="stop_trial"
        )

        if phase == "online":
            # Decoding
            prediction = []
            while len(prediction) == 0:
                sw.update()
                if sw.n_new > 0:
                    prediction = sw.unfold_buffer()[-sw.n_new:] - 1
                    sw.n_new = 0
            prediction_key = code_to_key[prediction]
            keyboard.log(json.dumps({"i_trial": i_trial, "prediction": prediction, "prediction_key": prediction_key}))

            # Spelling
            text = keyboard.get_text_field("text")
            symbol = prediction_key
            if symbol in KEY_MAPPING:
                symbol = KEY_MAPPING[symbol]
            if symbol == "<":
                text = text[:-1]
            if symbol == "space":
                text = text + " "
            else:
                text += symbol
            keyboard.set_text_field(name="text", text=text)

            # Feedback
            highlights[prediction_key] = [-1]
            keyboard.run(
                sequences=highlights, duration=TIME_FEEDBACK, start_marker="start_feedback", stop_marker="stop_feedback"
            )
            highlights[prediction_key] = [0]

        # Inter-trial time
        keyboard.run(
            sequences=highlights, duration=TIME_ITI, start_marker="start_inter_trial", stop_marker="stop_inter_trial"
        )

    # Wait to stop
    keyboard.set_text_field(name="messages", text="Press button to stop.")
    event.waitKeys(keyList=KEYS_CONTINUE)
    keyboard.set_text_field(name="messages", text="")

    # Stop run
    keyboard.log(marker="stop_run")
    keyboard.set_text_field(name="messages", text="Stopping...")
    keyboard.run(sequences=highlights, duration=5.0)
    keyboard.set_text_field(name="messages", text="")
    keyboard.quit()

    return 0


if __name__ == "__main__":
    run(phase="online")
