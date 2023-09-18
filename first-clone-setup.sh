#!/usr/bin/env bash

set -e
set -o pipefail

sudo yum install autoconf


#RDIR=$(git rev-parse --show-toplevel)

#git config --global protocol.file.allow always

RDIR=$(pwd)
cd ../ 
#wget -O gemmini-rocc-tests-moca-ae.zip https://zenodo.org/record/7456045/files/gemmini-rocc-tests-moca-ae.zip
#wget -O gemmini-moca-ae.zip https://zenodo.org/record/7456052/files/gemmini-moca-ae.zip
#wget -O chipyard-moca-ae.zip https://zenodo.org/record/7456073/files/chipyard-moca-ae.zip

unzip chipyard-aurora-ae.zip
unzip gemmini-aurora-ae.zip
unzip aurora-rocc-tests-ae.zip

cd $RDIR

echo "git submodule chipyard"
sed -i 's/https:\/\/github.com\/SeahK\/chipyard-aurora-ae/..\/chipyard-aurora-ae/g' .gitmodules
#git submodule set-url -- target-design/chipyard ../chipyard-aurora-ae
#git submodule sync
git submodule update --init target-design/chipyard
cd target-design/chipyard
#git checkout main

echo "git submodule gemmini"
sed -i 's/https:\/\/github.com\/SeahK\/gemmini-aurora-ae/..\/gemmini-aurora-ae/g' .gitmodules
#git submodule set-url -- generators/gemmini ../gemmini-aurora-ae
#git submodule sync
git submodule update --init generators/gemmini
cd generators/gemmini
#git checkout main

echo "git submodule rocc tests"
sed -i 's/https:\/\/github.com\/SeahK\/aurora-rocc-tests-ae/..\/aurora-rocc-tests-ae/g' .gitmodules
#git submodule set-url -- software/gemmini-rocc-tests ../aurora-rocc-tests-ae
#git submodule sync
git submodule update --init software/gemmini-rocc-tests 
cd software/gemmini-rocc-tests
#git checkout main

echo "git submodules configured manually from zip"

cd $RDIR

echo "running build script"
# build setup
./build-setup.sh
echo "finished running build script"
source sourceme-f1-manager.sh


cd deploy
firesim managerinit --platform f1
cp -r scripts/* results-workload/

# run through elaboration flow to get chisel/sbt all setup
cd ../sim
echo "chisel/sbt setup"
make f1

cd $RDIR
cd target-design/chipyard/generators/gemmini/software
rm -rf imagenet
unzip sample.zip
rm sample.zip

cd $RDIR
source sourceme-f1-manager.sh
cd target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests
./build.sh
rm -rf build/imagenet
cp -r ../imagenet build/imagenet

cd $RDIR
# build target software
cd sw/firesim-software
echo "Firemarshal setup"
./init-submodules.sh
marshal -v build br-base.json

cd $RDIR
