# Dareplane c-VEP Speller

This is a module compatible with the [Dareplane](https://bsdlab.github.io/Dareplane/main.html) platform. It provides a speller interface with individual symbols highlighted using the noise-tagging protocol, known to evoke the code-modulated visual evoked potential (c-VEP) in the EEG.

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

## Autocomplete

The default autocomplete functionality uses a local n-gram model. the online autocomplete requires a [Google AI authentication key](https://aistudio.google.com/app/apikey). This keys should be added to the config `speller.toml` via `speller.autocomplete.online`.

## Citation

If you use [Dareplane](https://bsdlab.github.io/Dareplane/main.html) or this model for your work, please cite both the following two references:
```bibtex
@article{dold2025,
    title = {Dareplane: a modular open-source software platform for {BCI} research with application in closed-loop deep brain stimulation},
    author = {Dold, Matthias and Pereira, Joana and Sajonz, Bastian and Coenen, Volker A and Thielen, Jordy and Janssen, Marcus L F and Tangermann, Michael},
    journal = {Journal of Neural Engineering},
    year = {2025},
    month = {mar},
    publisher = {IOP Publishing},
    volume = {22},
    number = {2},
    pages = {026029},
    doi = {10.1088/1741-2552/adbb20},
    url = {https://doi.org/10.1088/1741-2552/adbb20},
}
```
```bibtex
@article{thielen2021,
    title = {From full calibration to zero training for a code-modulated visual evoked potentials for brain--computer interface},
    author = {Thielen, Jordy and Marsman, Pieter and Farquhar, Jason and Desain, Peter},
    journal = {Journal of Neural Engineering},
    publisher = {IOP Publishing Ltd},
    volume = {18},
    number = {5},
    pages = {056007},
    year = {2021},
    doi = {10.1088/1741-2552/abecef},
    url = {https://doi.org/10.1088/1741-2552/abecef}
}
```
