#!/bin/bash
curdir=$(pwd)
mydir="${0%/*}"
version="rel-3.0.0"

cd "$mydir" || exit 1

echo "Cloning CaDiCaL"
git clone https://github.com/arminbiere/cadical.git cadical

echo "Entering CaDiCaL directory"
cd cadical || exit 1

echo "Checking out $version"
git checkout "$version" || exit 1

echo "Building CaDiCaL"
CFLAGS="-fPIC" CXXFLAGS="-fPIC" ./configure || exit 1

if command -v nproc >/dev/null 2>&1; then
  JOBS=$(nproc)
else
  JOBS=$(sysctl -n hw.ncpu)
fi

make -j"$JOBS" || exit 1

cd "$curdir" || exit 1