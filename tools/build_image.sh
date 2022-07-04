#!/bin/bash

version=$(git describe --tags --always)

if [[ $(git diff --stat) != '' ]]; then
  version="${version}-dirty"
else
  version="${version}-clean"
fi

build_date=$(date)

docker build --build-arg GIT_VERSION=$version --build-arg BUILD="$build_date" -t sar-ai-tools:latest -t sar-ai-tools:$version .

