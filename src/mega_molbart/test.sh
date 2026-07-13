PROJECT_ROOT="/mnt/petrelfs/zhengqiaoyu.p/SunYuze/comclip_copy"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# 将src目录添加到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
NGPUS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json
echo "timm"
srun --partition=medai_p --mpi=pmi2 --gres=gpu:$NGPUS --quotatype=spot -n1 --ntasks-per-node=1 --cpus-per-task=16 --job-name=Comclip --kill-on-bad-exit=1 \
python /mnt/petrelfs/zhengqiaoyu.p/SunYuze/final/clipmaedino/mega_molbart/STMencoder.py