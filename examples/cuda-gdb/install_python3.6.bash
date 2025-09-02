#!/bin/bash
mkdir tmp
cd tmp
wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz
tar xzf Python-3.6.15.tgz
cd Python-3.6.15

sudo ./configure --prefix=/opt/python3.6/ \
    --enable-optimizations \
    --enable-shared \
    --with-ensurepip=no
sudo make -j "$(grep -c ^processor /proc/cpuinfo)"
sudo make altinstall
cd ../..
sudo rm -rf tmp
