# TF Inference
Use the TensorFlow Lite C API to run oject detection inferences on SDR JPG spectra.
- Uses the [stb library](https://github.com/georgeslabreche/stb) for image processing.

## Build
- Initialize and update the stb Git submodule: `git submodule init && git submodule update`.
- Compile with `make`. Can also compile for ARM architecture with `make TARGET=arm`.