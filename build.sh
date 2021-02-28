#!/bin/bash

if [ ! -z $1 ] && [ $1 == "--plugins" ]
then
    cd plugins && rm -rf build
    mkdir build && cd build
    cmake .. && make -j$(nproc) install
    cd ../../
    python setup.py install
else
    python setup.py install
fi
