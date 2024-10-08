# Dareplane c-VEP Speller

This is a module for the [Dareplane](https://github.com/bsdlab/Dareplane) project. It provides a speller interface with individual symbols highlighted using the noise-tagging protocol, known to evoke the code-modulated visual evoked potential (c-VEP) in the EEG.

## Installation

To download the dp-cvep-speller, use:

	git clone https://github.com/thijor/dp-cvep-speller.git

Make sure that requirements are installed, ideally in a separate conda environment:

    conda create --name dp-cvep-speller python=3.10
    conda activate dp-cvep-speller
    pip install -r requirements.txt

## Getting started

To run the dp-cvep-speller module in isolation, use:

    python -m cvep_speller.speller.py

This will run a minimal example using defaults as specified in `configs/speller.toml`.

## Caveat of current unit tests:

- [ ] While the `test_decoder_selection` seems to work as intended, tests can fail due to a dynamic refresh rate of the screen (e.g., MacBooks)

- [ ] Add a checking function to ensure that the screen refresh rate is as specified in the config
