#!/usr/bin/env bash

set -e
set -o pipefail

RDIR=$(pwd)

cd $RDIR
echo "Generating workload images"
cd target-design/chipyard/generators/gemmini/software
./build-gemmini-workload.sh

cd $RDIR
