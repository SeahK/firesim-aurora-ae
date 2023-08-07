#!/bin/bash

python3 parse_xr.py $1/gemmini-tests-workload-xbar/uartlog 0
python3 parse_xr.py $1/gemmini-tests-workload-noc/uartlog 1
