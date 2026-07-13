PROJECT_ROOT="/mnt/petrelfs/zhengqiaoyu.p/SunYuze/comclip_copy"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# 将src目录添加到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
NGPUS=8
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json
echo "timm"
srun --partition=medai_p --mpi=pmi2 --gres=gpu:$NGPUS --quotatype=reserved -n1 --ntasks-per-node=1 --cpus-per-task=32 --job-name=Comclip --kill-on-bad-exit=1 \
torchrun --nproc_per_node=8 --master_port=2222 ./jointtrain.py \
    --train_csv_path ./data/Broad-270/compound_splits/all.csv \
    --val_csv_path ./data/ChEMBL-209/splits/val1.csv \
    --control_json_path ./data/control_data/control_cleaned.json \
    --petrel_conf_path /mnt/petrelfs/zhengqiaoyu.p/SunYuze/petreloss.conf \
    --batch_size 512 \
    --epochs 60 \
    --lr 1e-4 \
    --resize_short_side 780 \
    --save_dir ./pretrain_ckpt/setting2 \
    --pretrained \
    --molstm_ckpt ./MoleculeSTM/molecule_model.pth \
    --molstm_vocab ./MoleculeSTM/bart_vocab.txt \
    --lambda_clip 1.0 \
    --lambda_mae 0.0 \
    --lambda_dino 1.0 \
    --locals_per_global 4 \
    --dino_out_dim 65536 \