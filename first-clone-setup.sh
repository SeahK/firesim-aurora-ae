#!/usr/bin/env bash

set -e
set -o pipefail

sudo yum install autoconf

# build setup
./build-setup.sh
source sourceme-f1-manager.sh

RDIR=$(pwd)

cd deploy
firesim managerinit --platform f1

# run through elaboration flow to get chisel/sbt all setup
cd ../sim
echo "chisel/sbt setup"
make f1

cd $RDIR
cd target-design/chipyard/tests
rm -rf imagenet
unzip sample.zip
rm sample.zip

cd $RDIR
source sourceme-f1-manager.sh
cd target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests
./build.sh
rm -rf build/imagenet
cp -r ../../../../tests/imagenet build/imagenet

cd $RDIR
# build target software
cd sw/firesim-software
echo "Firemarshal setup"
./init-submodules.sh
marshal -v build br-base.json

cd $RDIR
