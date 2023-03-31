#!/bin/sh
set -x

exp_dir=$1
config=$2

mkdir -p ${exp_dir}

export PYTHONPATH=.
python -u run/distill.py \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/distill-$(date +"%Y%m%d_%H%M").log