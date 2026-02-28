PROJECT_ROOT="/mnt/petrelfs/zhengqiaoyu.p/SunYuze/comclip_copy"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# 将src目录添加到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
NGPUS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json
echo "1e4"
srun --partition=medai_p --mpi=pmi2 --gres=gpu:$NGPUS --quotatype=reserved -n1 --ntasks-per-node=1 --job-name=Comclip --kill-on-bad-exit=1 \
python /mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction_cleaned/src/visualization/attention_map/visualize_attention.py