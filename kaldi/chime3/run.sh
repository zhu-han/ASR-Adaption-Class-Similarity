#!/bin/bash

set -e
set -u
set -o pipefail

chime3_data=/mnt/c/zhuhan/CHiME3

./run_init_chime3.sh --stage 0 $chime3_data  >log/log.log 2>&1
