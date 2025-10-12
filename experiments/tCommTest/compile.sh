#!/bin/bash
commBytes=(8 16 32 64 128 256)
rm -rf bin
mkdir -p bin 
for cb in ${commBytes[@]}; do
    cmake -B build --fresh -DCOMM_BYTES=$cb
    cmake --build build 
    cp "./build/tCommTest${cb}" bin/
done