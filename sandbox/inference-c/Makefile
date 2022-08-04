TARGET = dev

# Compiler options.
# FIXME: Use proper Poky cross-compilter toolchain for CC_ARM.
CC_DEV = gcc
CC_ARM = /usr/bin/arm-linux-gnueabihf-gcc

# Architecture.
ARCH_DEV = x86_64
ARCH_ARM = arm32

# Target compiler and architecture environment.
ifeq ($(TARGET),arm)
	CC = $(CC_ARM)
	ARCH = $(ARCH_ARM)
else
	CC = $(CC_DEV)
	ARCH = $(ARCH_DEV)
endif

# Header includes.
INCLUDEPATH = -Istb -Itensorflow

# Flags.
CFLAGS = -Wall -static -O3

# Library flags
LIBDIRFLAGS = -Ltst/$(ARCH)/tensorflowlite_c
LIBFLAGS = -ltensorflowlite_c -lm

# Source directory and files.
SOURCEDIR = src
HEADERS := $(wildcard $(SOURCEDIR)/*.h)
SOURCES := $(wildcard $(SOURCEDIR)/*.c)

# Target output.
BUILDTARGET = inference

# FIXME: Linking error when we use $(CFLAGS)
all:
#	$(CC) $(CFLAGS) $(INCLUDEPATH) $(LIBDIRFLAGS) $(HEADERS) $(SOURCES) -o $(BUILDTARGET) $(LIBFLAGS)
	$(CC) $(INCLUDEPATH) $(LIBDIRFLAGS) $(HEADERS) $(SOURCES) -o $(BUILDTARGET) $(LIBFLAGS)


clean:
	rm -f $(BUILDTARGET)