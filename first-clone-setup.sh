#!/usr/bin/env bash

set -e
set -o pipefail

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

# build target software
cd ../sw/firesim-software
echo "Firemarshal setup"
./init-submodules.sh
marshal -v build br-base.json

cd $RDIR
cd target_design/chipyard/generators/gemmini/software/gemmini-rocc-tests
echo "Building gemmini-rocc-tests benchmark"
./build.sh
cd $RDIR
