# Dareplane c-VEP Speller

This is a module for the [Dareplane](https://github.com/bsdlab/Dareplane) project. It provides a speller interface with individual symbols highlighted using the noise-tagging protocol, known to evoke the code-modulated visual evoked potential (c-VEP) in the EEG. 

## Caveat of current unit tests:
- [ ] While the `test_decoder_selection` seems to work as intended, there can be test failures due to a dynamic refresh rate of the screen (e.g., MacBooks)

- [ ] Add a checking function to ensure that the screen refresh rate is as specified in the config
