FROM ubuntu:20.04

LABEL maintainer="Federico 'Larroca' La rocca - flarroca@fing.edu.uy"

#ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# else it will output an error about Gtk namespace not found
RUN apt-get install -y gir1.2-gtk-3.0

# to have add-apt-repository available
RUN apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:gnuradio/gnuradio-releases-3.9

# create user gnuario with sudo (and password gnuradio)
RUN apt-get install -y sudo
RUN useradd --create-home --shell /bin/bash -G sudo gnuradio
RUN echo 'gnuradio:gnuradio' | chpasswd

# I create a dir at home which I'll use to persist after the container is closed (need to change it's ownership)
RUN mkdir /home/gnuradio/persistent  && chown gnuradio /home/gnuradio/persistent

RUN apt-get update

RUN apt-get install -y gnuradio

# installing other packages needed for downloading and installing OOT modules
RUN apt-get install -y gnuradio-dev cmake git libboost-all-dev libcppunit-dev liblog4cpp5-dev python3-pygccxml pybind11-dev liborc-dev

# of course, nothing useful can be done without vim
RUN apt-get install -y vim pip


#USER gnuradio

WORKDIR /home/setup
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN git clone https://github.com/TomMladenov/renderfall.git
WORKDIR /home/setup/renderfall
RUN mkdir build && apt-get install -y libpng-dev
WORKDIR /home/setup/renderfall/build
RUN cmake .. && make && make install

ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3/dist-packages"

ARG GIT_VERSION
ARG BUILD
ENV GIT_VERSION=$GIT_VERSION
ENV BUILD=$BUILD

USER gnuradio
CMD gnuradio-companion
