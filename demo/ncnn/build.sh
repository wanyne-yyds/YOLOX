#!/bin/bash
rm -rf ./build
mkdir ./build
cd ./build
cmake ../cpp
make -j