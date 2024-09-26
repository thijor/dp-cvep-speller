from functools import partial

from dareplane_utils.default_server.server import DefaultServer
from fire import Fire

from cvep_speller.speller import run_speller_paradigm
from cvep_speller.utils.logging import logger


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10) -> int:
    logger.setLevel(loglevel)

    # Primary commands
    pcommand_map = {
        "TRAINING": partial(run_speller_paradigm, phase="training"),
        "ONLINE": partial(run_speller_paradigm, phase="online"),
    }

    # Setup server
    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="cvep_speller_server"
    )
    server.init_server()
    server.start_listening()

    return 0


if __name__ == "__main__":
    Fire(main)
