PROJECT_ROOT="/path/to/OpenPheno"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
PETREL_CONF="/path/to/petreloss.conf"
PRETRAIN_CKPT="${PROJECT_ROOT}/dir/OpenPheno_setting2.pt"
SAVE_DIR="${PROJECT_ROOT}/src/outputs/setting2_openpheno"
cd "${PROJECT_ROOT}/src"
NGPUS=4
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json
echo "1e4"
srun --partition=medai_p --mpi=pmi2 --gres=gpu:$NGPUS --quotatype=reserved -n1 --ntasks-per-node=1 --job-name=Comclip --kill-on-bad-exit=1 \
torchrun --nproc_per_node=4 --master_port=22222 ./downstream/train.py \
    --train_image_csv ./data/Broad-270/compound_splits/all.csv \
    --val_image_csv ./data/Broad-270/compound_splits/all.csv \
    --test_image_csv ./data/Broad-270/compound_splits/all.csv \
    --train_label_csv ./data/Broad-270/assay_splits/assay_train.csv \
    --val_label_csv ./data/Broad-270/assay_splits/assay_val.csv \
    --test_label_csv ./data/Broad-270/assay_splits/assay_test.csv \
    --assay_json ./data/Broad-270/assay_meta/assay_description.json \
    --resize_short_side 780 \
    --petrel_conf_path "${PETREL_CONF}" \
    --freeze_backbone \
    --batch_size 128 --epochs 10 \
    --save_dir "${SAVE_DIR}" --wandb_mode offline \
    --interval_step 40 \
    --test_on \
    --lr 3e-4 \
    --pretrain_ckpt "${PRETRAIN_CKPT}"
