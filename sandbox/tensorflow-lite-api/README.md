# using Docker to build the tensorflow lite C API

Make sure Docker is installed. Build the image and run a container as shown below:
```
DOCKER_BUILDKIT=1 docker build . -t sepp-toolchain:latest
docker run -u 0 -it --rm --entrypoint /bin/bash --name sepp-toolchain -v /home/tom/git/opssat-sar-ai/sandbox/inference-c:/home/user/share sepp-toolchain
```

Inside the docker container run bazel and wait for the build to complete:
```
cd /home/user/tensorflow_src
bazel build --config=elinux_armhf -c opt //tensorflow/lite/c:tensorflowlite_c
INFO: Analyzed target //tensorflow/lite/c:tensorflowlite_c (85 packages loaded, 10514 targets configured).
INFO: Found 1 target...
Target //tensorflow/lite/c:tensorflowlite_c up-to-date:
  bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so
INFO: Elapsed time: 608.247s, Critical Path: 69.71s
INFO: 1074 processes: 189 internal, 885 local.
INFO: Build completed successfully, 1074 total actions
```

Targets will reside in:
```
root@65f920c1a935:/home/user/tensorflow_src# ls -lat
total 876
drwxr-xr-x 1 user user   4096 Jul 30 16:34 .
lrwxrwxrwx 1 root root    111 Jul 30 16:34 bazel-bin -> /root/.cache/bazel/_bazel_root/748994b1258cd50aea83eb1fdf8edfae/execroot/org_tensorflow/bazel-out/armhf-opt/bin
lrwxrwxrwx 1 root root     97 Jul 30 16:34 bazel-out -> /root/.cache/bazel/_bazel_root/748994b1258cd50aea83eb1fdf8edfae/execroot/org_tensorflow/bazel-out
lrwxrwxrwx 1 root root     87 Jul 30 16:34 bazel-tensorflow_src -> /root/.cache/bazel/_bazel_root/748994b1258cd50aea83eb1fdf8edfae/execroot/org_tensorflow
lrwxrwxrwx 1 root root    116 Jul 30 16:34 bazel-testlogs -> /root/.cache/bazel/_bazel_root/748994b1258cd50aea83eb1fdf8edfae/execroot/org_tensorflow/bazel-out/armhf-opt/testlogs
```

```libtensorflowlite_c.so``` can be found in:
```
root@65f920c1a935:/home/user/tensorflow_src/bazel-bin/tensorflow/lite/c# ls -lart
total 3368
-r-xr-xr-x  1 root root     168 Jul 30 16:34 libc_api_experimental.pic.lo-2.params
-r-xr-xr-x  1 root root   12090 Jul 30 16:34 libtensorflowlite_c.so-2.params
-r-xr-xr-x  1 root root     169 Jul 30 16:34 libc_api_without_op_resolver.pic.lo-2.params
-r-xr-xr-x  1 root root     132 Jul 30 16:34 libcommon.pic.lo-2.params
-r-xr-xr-x  1 root root    7232 Jul 30 16:35 libcommon.pic.lo
-r-xr-xr-x  1 root root   12060 Jul 30 16:35 libc_api_experimental.pic.lo
drwxr-xr-x  5 root root    4096 Jul 30 16:40 _objs
-r-xr-xr-x  1 root root   25358 Jul 30 16:40 libc_api_without_op_resolver.pic.lo
drwxr-xr-x 11 root root    4096 Jul 30 16:41 ..
drwxr-xr-x  3 root root    4096 Jul 30 16:41 .
-r-xr-xr-x  1 root root 3361892 Jul 30 16:41 libtensorflowlite_c.so
root@65f920c1a935:/home/user/tensorflow_src/bazel-bin/tensorflow/lite/c# file libtensorflowlite_c.so
libtensorflowlite_c.so: ELF 32-bit LSB shared object, ARM, EABI5 version 1 (GNU/Linux), dynamically linked, BuildID[md5/uuid]=dfc00899e974d7496866ca63281e30ea, stripped
```