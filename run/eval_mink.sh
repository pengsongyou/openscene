set -x

exp_dir=$1
config=$2

mkdir -p ${exp_dir}

model_dir=${exp_dir}/model
result_dir=${exp_dir}/result_eval

export PYTHONPATH=.
python -u run/eval_mink.py \
  --config=${config} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best.pth.tar
  2>&1 | tee -a ${exp_dir}/eval-mink-$(date +"%Y%m%d_%H%M").log