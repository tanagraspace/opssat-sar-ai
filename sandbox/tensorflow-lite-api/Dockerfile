FROM ubuntu:16.04 AS builder

#SOURCE: https://gitlab.com/esa/NMF/ops-sat-sepp-ci-docker/-/blob/master/Dockerfile

RUN apt-get update && apt-get install -y \
        wget git-core \
        sudo diffstat \
        iproute2 iputils-ping \
        python3 python \ 
        gcc make gawk cpio g++ \
        language-pack-en-base locales \
        xz-utils bzip2

# Configure default user
COPY sudoers.user /etc/
RUN groupadd -g 70 user && \
    useradd -N -m -u 70 -g 70 user && \
    echo "user ALL=NOPASSWD:ALL" >> /etc/sudoers

USER user
# Install Poky eSDK toolchain
ADD http://downloads.yoctoproject.org/releases/yocto/yocto-2.4.4/toolchain/x86_64/poky-glibc-x86_64-core-image-minimal-cortexa8hf-neon-toolchain-ext-2.4.4.sh /tmp/sdk_installer.sh

RUN sudo chmod 755 /tmp/sdk_installer.sh
RUN /tmp/sdk_installer.sh -y -d ~/poky_sdk/

# Cleanup SDK installer
RUN sudo rm /tmp/sdk_installer.sh

ENV LANG=en_US.UTF-8

# Extend toolchain with libpng and ffmpeg
RUN echo "LICENSE_FLAGS_WHITELIST=\"commercial\"" >> ~/poky_sdk/conf/local.conf
RUN . ~/poky_sdk/environment-setup-cortexa8hf-neon-poky-linux-gnueabi; devtool sdk-install -s libpng ffmpeg

# Cleanup temp data - WARNING - SDK might not be able to build extra components properly afterwards
WORKDIR /home/user/poky_sdk
RUN rm -rf sstate-cache downloads tmp/work tmp/log tmp/stamps tmp/cache tmp/buildstats

FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
        wget git-core \
        sudo diffstat \
        python3 \
        python \
        make swig \
        gcc make gawk cpio g++ \
        language-pack-en-base locales \
        xz-utils bzip2 \
        iproute2 iputils-ping

ENV LANG=en_US.UTF-8

# Configure default user
COPY sudoers.user /etc/
RUN groupadd -g 70 user && \
    useradd -N -m -u 70 -g 70 user && \
    echo "user ALL=NOPASSWD:ALL" >> /etc/sudoers

USER user
# Copy SDK
COPY --from=builder /home/user/poky_sdk /home/user/poky_sdk

WORKDIR /home/user/poky_sdk
RUN touch conf/sanity.conf

# setup for installing Google's bazel tool
USER root
RUN sudo apt install -y apt-transport-https curl gnupg
RUN sudo curl -fsSL https://bazel.build/bazel-release.pub.gpg | sudo gpg --dearmor > bazel.gpg
RUN sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
RUN sudo apt update && sudo apt install -y bazel=5.0.0 python3-numpy

# get tensorflow v2.9
USER user
WORKDIR /home/user
RUN git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
WORKDIR /home/user/tensorflow_src
RUN git checkout r2.9
