#!/bin/bash

if [ ! -z $1 ] && [ $1 == "--plugins" ]
then
    cd plugins && rm -rf build
    mkdir build && cd build
    cmake .. && make -j4  install
    cd ../../
    cd torch2trt/ops/dcn
    ./build.sh
    cd ../../../
    python setup.py install --plugins
else
    python setup.py install
fi
