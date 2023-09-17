#!/bin/bash

python3 parse_sla.py $1/gemmini-tests-workload-xbar/uartlog 0 0 &&
python3 parse_sla.py $1/gemmini-tests-workload-xbar/uartlog 1 0 &&
python3 parse_sla.py $1/gemmini-tests-workload-xbar/uartlog 2 0
python3 parse_sla.py $1/gemmini-tests-workload-noc/uartlog 0 1 &&
python3 parse_sla.py $1/gemmini-tests-workload-noc/uartlog 1 1 &&
python3 parse_sla.py $1/gemmini-tests-workload-noc/uartlog 2 1
