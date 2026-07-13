# OpenPheno

**Phenotypic bioactivity prediction as open-set biological assay querying**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenPheno is a multimodal framework for bioactivity prediction from Cell Painting morphology, chemical structures, and natural-language assay descriptions. Instead of treating assays as fixed output indices, OpenPheno formulates prediction as an open-set assay-querying problem: a model can be asked about a new biological assay using its text description.

The repository contains the training and downstream evaluation code used in the paper, together with processed metadata/splits and example result files. Full reproduction requires access to the corresponding Cell Painting image objects and Petrel/S3-style image paths, so the scripts are intended as reference workflows and runnable templates once the image backend is configured.

## Highlights

- **Open-set assay querying:** predict activity for held-out assays from assay descriptions.
- **Multimodal pretraining:** Cell Painting images are aligned with molecular structures and stabilized with DINO-style replicate consistency.
- **Assay Query Network:** fuses image, molecule, and assay-description embeddings for downstream bioactivity prediction.
- **Benchmarks included:** Broad-270, ChEMBL-209, and EvE-24 metadata/splits.

## Repository Layout

```text
src/
  data/
    Broad-270/        # processed labels, assay metadata, compound/assay splits
    ChEMBL-209/       # processed labels, metadata, scaffold splits
    Eve-24/           # external EvE-24 label matrix and assay descriptions
    control_data/     # DMSO/control image metadata
  pretrain/
    train.py          # Stage-I multimodal pretraining
    scripts/          # example pretraining jobs
  downstream/
    train.py          # assay-query downstream training/evaluation
    OpenPheno_setting2.sh
  results/
    *.csv             # released prediction logits/results
    calculate_metrics.py
  MoleculeSTM/        # MoleculeSTM checkpoint/vocabulary expected by scripts
```

## Installation

The code was tested with Python 3.10 and PyTorch 2.4.1 + CUDA 11.8.

```bash
git clone https://github.com/Yuze-e20/OpenPheno.git
cd OpenPheno

conda create -n openpheno python=3.10 -y
conda activate openpheno

# Install a CUDA build of PyTorch suitable for your machine.
# Example for CUDA 11.8:
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# Install OpenPheno and Python dependencies.
pip install -e .
```

If you use Petrel/S3-backed Cell Painting images, edit the `PETREL_CONF` variable at the top of the example scripts:

```bash
PETREL_CONF="/path/to/petreloss.conf"
```

The training scripts assume that image CSVs contain five channel paths (`ch1_path` to `ch5_path`) and that those paths can be opened by the configured image backend.

## Pretrained Checkpoint

For downstream zero-shot evaluation, download the released OpenPheno checkpoint from ModelScope:

```bash
modelscope download \
  --model YuzeSun/OpenPheno \
  OpenPheno_setting2.pt \
  --local_dir ./dir
```

Then use the downloaded path as `--pretrain_ckpt`, or edit the `PRETRAIN_CKPT` variable in the example script:

```bash
src/downstream/OpenPheno_setting2.sh
```

## Example: Pretraining

An example ChEMBL-209 split-1 pretraining job is provided at:

```bash
src/pretrain/scripts/setting1split2091.sh
```

Run it from the repository root:

```bash
bash src/pretrain/scripts/setting1split2091.sh
```

The script launches `src/pretrain/train.py` with `torchrun`/`srun`. Before running on a new cluster, update the SLURM options if needed:

```bash
NGPUS=8
srun --partition=medai_p --gres=gpu:$NGPUS ...
```

Key inputs used by the pretraining script:

- `src/data/ChEMBL-209/splits/train1.csv`
- `src/data/ChEMBL-209/splits/val1.csv`
- `src/data/control_data/control_cleaned.json`
- `src/MoleculeSTM/molecule_model.pth`
- `src/MoleculeSTM/bart_vocab.txt`

## Example: Setting 2 Downstream Evaluation

Setting 2 evaluates known compounds on assays entirely withheld from downstream training. The demo script is:

```bash
src/downstream/OpenPheno_setting2.sh
```

After downloading the checkpoint, edit the path variables at the top of the script:

```bash
PROJECT_ROOT="/path/to/OpenPheno"
PETREL_CONF="/path/to/petreloss.conf"
PRETRAIN_CKPT="${PROJECT_ROOT}/dir/OpenPheno_setting2.pt"
```

Then run:

```bash
bash src/downstream/OpenPheno_setting2.sh
```

The script uses:

- training labels: `src/data/Broad-270/assay_splits/assay_train.csv`
- validation labels: `src/data/Broad-270/assay_splits/assay_val.csv`
- held-out test labels: `src/data/Broad-270/assay_splits/assay_test.csv`
- assay descriptions: `src/data/Broad-270/assay_meta/assay_description.json`

Outputs are written to `src/outputs/setting2_openpheno` by default.

## Recomputing Metrics

Released logits are stored in `src/results`. To recompute AUROC, AUPRC, F1, EF@5%, and bootstrap 95% confidence intervals:

```bash
python src/results/calculate_metrics.py
```

This writes:

```text
src/results/metrics/summary_metrics.csv
src/results/metrics/per_assay_*.csv
```

Single-file example:

```bash
python src/results/calculate_metrics.py \
  --file src/results/setting3_Eve-24.csv \
  --out_dir /tmp/openpheno_metrics
```

Metric conventions:

- Broad-270 closed-set Setting 1: median across five folds per assay, then mean across assays.
- ChEMBL-209 closed-set Setting 1: missing or invalid test-fold AUROC is filled with 0.5, following the original benchmark convention.
- Other settings: direct mean across assay-level metrics.
- F1 is computed by selecting the top-scoring fraction according to the hard-coded training positive rate used in the paper.
- EF@5% is computed from the top 5% ranked compounds for each assay.

## Data Notes

The repository includes processed labels, assay descriptions, split files, and example logits. The full Cell Painting image collection is too large to bundle directly. To run training or evaluation end-to-end, users must provide image access compatible with the paths in the CSV files and configure the `PETREL_CONF` path in the example scripts if using Petrel.

## Citation

If you use this repository, please cite the OpenPheno paper.

```bibtex
@article{openpheno,
  title   = {Phenotypic Bioactivity Prediction as Open-set Biological Assay Querying},
  author  = {Sun, Yuze and collaborators},
  journal = {TBD},
  year    = {2026}
}
```
