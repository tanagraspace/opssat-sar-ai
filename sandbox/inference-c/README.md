# TF Inference
Use the TensorFlow Lite C API to run oject detection inferences on SDR JPG spectra.
- Uses the [stb library](https://github.com/georgeslabreche/stb) for image processing.
- Uses the [TensorFlow GitHub Repo (branch r2.9)](https://github.com/tensorflow/tensorflow/tree/r2.9) as the compiler's include path option.

## Build
- Initialize and update the stb Git submodule: `git submodule init && git submodule update`.
- Compile with `make`. Can also compile for ARM architecture with `make TARGET=arm`.

## Run
Super easy to run.

### Helps
```bash
$ ./inference -?
inference [options] ...
  --input    / -i        input image filename
  --model    / -m        tflite model filename
  --xsize    / -x        training input width
  --ysize    / -y        training input height
  --mean     / -n        input mean (optional)
  --std      / -s        input standard deviation (optional)
  --help     / -?        this information
```

### Inference
```bash
$ ./inference -i test/test_input.cf32.jpg -m test/97708c55_efficientdet_lite1.tflite -x 384 -y 384
```

## Note
If the input mean is set 0 and the input standard deviation is set to 255 then the image input's 0-255 RGB range is rescaled/normalized to 0-1. Using normalized inputs is preferred with respect to performance with tensor calculations. Only normalize inputs if the model was trained with normalized training data.