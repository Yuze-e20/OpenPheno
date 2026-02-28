# train_visiononly.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score  
from downstream.downstream_dataset import MultiLabelPairDataset
from downstream.model_tqn import DownstreamModelTQN 
from assay_utils import load_and_encode_assays
import wandb
from torch.amp import autocast, GradScaler

@dataclass
class DSConfig:
    train_image_csv: str
    val_image_csv: str
    train_label_csv: str
    val_label_csv: str 
    assay_json: str 
    pretrain_ckpt: str
    resume_ckpt: str
    resize_short_side: int
    test_on: bool = False
    out_dim: int = 54
    test_image_csv: Optional[str] = None
    test_label_csv: Optional[str] = None
    freeze_backbone: bool = True
    batch_size: int = 128
    num_workers: int = 8
    lr: float = 1e-4 
    weight_decay: float = 1e-4
    epochs: int = 10
    grad_clip: float = 3.0
    dropout: float = 0.1
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = 'cuda'
    save_dir: str = "./down_ckpts_tqn_residual"
    interval_step: int = 50
    seed: int = 42
    project: str = "comclip-tqn"
    run_name: Optional[str] = None
    log_id: Optional[str] = None
    wandb_mode: str = "disabled"

def setup_distributed(cfg: DSConfig):
    cfg.rank = int(os.environ["RANK"])
    cfg.world_size = int(os.environ["WORLD_SIZE"])
    cfg.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.local_rank)
    cfg.device = f"cuda:{cfg.local_rank}"
    dist.init_process_group(backend="nccl", init_method="env://")
    return cfg

def set_seed(cfg: DSConfig):
    random.seed(cfg.seed + cfg.rank)
    torch.manual_seed(cfg.seed + cfg.rank)
    torch.cuda.manual_seed_all(cfg.seed + cfg.rank)
    np.random.seed(cfg.seed + cfg.rank)

def load_and_broadcast_assay(assay_json: str, label_csv: str, device: torch.device, rank: int) -> Tuple[torch.Tensor, List[str]]:
    embeds = None
    assay_ids = []
    if rank == 0:
        print(f"Rank 0: Loading assays for {label_csv}...")
        embeds_cpu, assay_ids = load_and_encode_assays(assay_json, label_csv)
        embeds = embeds_cpu.to(device)
        print(f"Rank 0: Loaded shape: {embeds.shape}, Num IDs: {len(assay_ids)}")

    shape_tensor = torch.zeros(2, dtype=torch.long, device=device)
    if rank == 0:
        shape_tensor[0] = embeds.shape[0]
        shape_tensor[1] = embeds.shape[1]
    
    dist.broadcast(shape_tensor, src=0)
    
    if rank != 0:
        embeds = torch.zeros((shape_tensor[0], shape_tensor[1]), device=device)
    
    dist.broadcast(embeds, src=0)
    return embeds, assay_ids

def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss_all = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = loss_all * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom

def _safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x).clamp(1e-7, 1 - 1e-7)

def print_detailed_stats(epoch: int, step: int, train_loss: float, eval_loss: float, metrics: Dict[str, Any], best_t: float, prefix: str = "Val"):
    macro_auc = metrics["macro_auc"]
    auc_list = metrics["auc_list"]
    f1_stats = metrics["f1_stats"]
    
    thr_09, thr_08, thr_07, thr_05 = 0.9, 0.8, 0.7, 0.5
    cnt_09, cnt_08, cnt_07, cnt_05 = 0, 0, 0, 0
    
    valid_auc_count = 0
    for a in auc_list:
        if a == a:
            valid_auc_count += 1
            if a > thr_09: cnt_09 += 1
            if a > thr_08: cnt_08 += 1
            if a > thr_07: cnt_07 += 1
            if a > thr_05: cnt_05 += 1
            
    print(f"\n[Epoch {epoch} | Step {step}] [{prefix}] Train Loss={train_loss:.4f} | {prefix} Loss={eval_loss:.4f} | Macro AUROC={macro_auc:.4f}")
    
    f1_str_list = []
    for t, val in f1_stats.items():
        f1_str_list.append(f"Thr={t:.2f}: {val:.4f}")
    
    print(f"    >>> Macro F1s ({prefix}): " + " | ".join(f1_str_list))
    print(f"    >>> Selected F1 at Val Best Thr ({best_t:.2f}): {f1_stats.get(best_t, float('nan')):.4f}")
    print(f"    >>> AUROC Stats ({prefix}): >0.9: {cnt_09} | >0.8: {cnt_08} | >0.7: {cnt_07} | >0.5: {cnt_05} (Total Valid Tasks: {valid_auc_count})")

def compute_metrics_standard(y_true_nan: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    K = y_true_nan.shape[1]
    aucs = [float("nan")] * K
    thresholds = [round(t, 2) for t in np.arange(0.01, 0.51, 0.01)]
    f1_per_task = {t: [float("nan")] * K for t in thresholds}
    valid_auc_vals = []
    
    for k in range(K):
        yk = y_true_nan[:, k]
        pk = y_prob[:, k]
        m = ~np.isnan(yk)
        yk_v = yk[m]
        pk_v = pk[m]
        
        if len(yk_v) > 0 and len(np.unique(yk_v)) >= 2:
            try:
                auc = float(roc_auc_score(yk_v, pk_v))
                aucs[k] = auc
                valid_auc_vals.append(auc)
            except Exception:
                pass
        
        if len(yk_v) > 0:
            for t in thresholds:
                try:
                    pred_bin = (pk_v > t).astype(int)
                    f1 = float(f1_score(yk_v, pred_bin, zero_division=0))
                    f1_per_task[t][k] = f1
                except Exception:
                    pass

    macro_auc = float(np.mean(valid_auc_vals)) if len(valid_auc_vals) > 0 else float("nan")
    
    f1_stats = {}
    for t in thresholds:
        valid_f1s = [val for val in f1_per_task[t] if not np.isnan(val)]
        if len(valid_f1s) > 0:
            f1_stats[t] = float(np.mean(valid_f1s))
        else:
            f1_stats[t] = float("nan")
            
    return {
        "macro_auc": macro_auc,
        "auc_list": aucs,
        "f1_stats": f1_stats,
        "f1_per_task": f1_per_task
    }

def save_logits_to_csv(save_path: str, probs: np.ndarray, targets: np.ndarray, assay_ids: List[str]):
    rows, cols = np.where(~np.isnan(targets))
    if len(rows) == 0:
        print(f"  [Warning] No valid targets found to save logits to {save_path}")
        return

    valid_probs = probs[rows, cols]
    valid_labels = targets[rows, cols]
    assay_ids_arr = np.array(assay_ids)
    valid_assay_names = assay_ids_arr[cols]
    
    df = pd.DataFrame({
        "sample_idx": rows,
        "assay_id": valid_assay_names,
        "prob": valid_probs,
        "label": valid_labels.astype(int)
    })
    df.to_csv(save_path, index=False)
    print(f"  Saved Logits CSV to {save_path} (Rows: {len(df)})")

@torch.no_grad()
def evaluate_distributed(model: nn.Module, loader: DataLoader, cfg: DSConfig, assay_embeds: torch.Tensor, desc: str = "Val") -> Dict[str, Any]:
    device = torch.device(cfg.device)
    rank = cfg.rank
    local_probs, local_targets, local_masks = [], [], []
    local_loss_sum = 0.0
    local_samples = 0
    model.eval()
    
    it = loader if rank != 0 else tqdm(loader, desc=f"[{desc}]")
    for batch in it:
        images, smiles, targets, mask = batch
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        with autocast(device_type="cuda"):
            out = model(images) 
            loss = masked_bce_with_logits(out["logits"], targets, mask)
            probs = _safe_sigmoid(out["logits"])
            
        bs = images.size(0)
        local_loss_sum += float(loss) * bs
        local_samples += bs
        
        local_probs.append(probs.detach().cpu().numpy())
        local_targets.append(targets.detach().cpu().numpy())
        local_masks.append(mask.detach().cpu().numpy())

    t_loss = torch.tensor([local_loss_sum, local_samples], dtype=torch.float64, device=device)
    dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
    global_loss = (t_loss[0] / t_loss[1]).item() if t_loss[1] > 0 else float("inf")

    obj_local = None
    if len(local_probs) > 0:
        obj_local = (
            np.concatenate(local_probs, axis=0),
            np.concatenate(local_targets, axis=0),
            np.concatenate(local_masks, axis=0),
        )

    gathered = [None for _ in range(cfg.world_size)]
    dist.all_gather_object(gathered, obj_local)

    results = {
        "loss": global_loss,
        "macro_auc": float("nan"),
        "auc_list": [],
        "f1_stats": {},
        "f1_per_task": {},
        "raw_probs": None,
        "raw_targets": None
    }

    if rank == 0:
        ys, ts, ms = [], [], []
        for item in gathered:
            if item is None: continue
            y, t, m = item
            ys.append(y); ts.append(t); ms.append(m)
        if len(ys) > 0:
            y_prob = np.concatenate(ys, axis=0)
            y_true = np.concatenate(ts, axis=0)
            msk = np.concatenate(ms, axis=0)
            y_true_nan = y_true.copy().astype(np.float32)
            y_true_nan[msk < 0.5] = np.nan
            
            metrics_res = compute_metrics_standard(y_true_nan, y_prob)
            results.update(metrics_res)
            results["raw_probs"] = y_prob
            results["raw_targets"] = y_true_nan
        else:
            results["macro_auc"] = 0.0

    metrics_tensor = torch.tensor([results["macro_auc"]], dtype=torch.float64, device=device)
    dist.broadcast(metrics_tensor, src=0)
    results["macro_auc"] = float(metrics_tensor[0].item())

    return results

def run_eval_and_save(
    cfg, model, 
    val_loader, val_assay_embeds, val_assay_ids,
    test_loader, test_assay_embeds, test_assay_ids,
    epoch, global_step, current_train_loss, 
    best_loss, best_path
):
    rank = cfg.rank
    
    val_results = evaluate_distributed(
        model, val_loader, cfg, val_assay_embeds, desc="Val"
    )
    val_loss = val_results["loss"]
    
    test_results = None
    if cfg.test_on and test_loader is not None and test_assay_embeds is not None:
        test_results = evaluate_distributed(
            model, test_loader, cfg, test_assay_embeds, desc="Test"
        )

    model.train()

    if rank == 0:
        best_t = 0.5
        if val_results["f1_stats"]:
            valid_f1s = {k: v for k, v in val_results["f1_stats"].items() if not np.isnan(v)}
            if valid_f1s:
                best_t = max(valid_f1s, key=valid_f1s.get)

        print_detailed_stats(epoch, global_step, current_train_loss, val_loss, val_results, best_t, prefix="Val")
        if test_results:
            print_detailed_stats(epoch, global_step, current_train_loss, test_results["loss"], test_results, best_t, prefix="Test")

        log_dict = {
            "epoch": epoch,
            "train/loss": current_train_loss,
            "val/loss": val_loss,
            "val/auroc_macro": val_results["macro_auc"],
            f"val/f1_at_best_thr_{best_t:.2f}": val_results["f1_stats"].get(best_t, float('nan'))
        }
        for t, f1 in val_results["f1_stats"].items():
            log_dict[f"val/f1_thr_{t:.2f}"] = f1
            
        if test_results:
            log_dict["test/loss"] = test_results["loss"]
            log_dict["test/auroc_macro"] = test_results["macro_auc"]
            log_dict[f"test/f1_at_val_best_thr_{best_t:.2f}"] = test_results["f1_stats"].get(best_t, float('nan'))
            for t, f1 in test_results["f1_stats"].items():
                log_dict[f"test/f1_thr_{t:.2f}"] = f1
        
        wandb.log(log_dict, step=global_step)

        save_path = os.path.join(cfg.save_dir, f"epoch{epoch}_step{global_step}.pt")
        
        torch.save({
            "downstream_state": model.state_dict(),
            "cfg": asdict(cfg),
            "best_val_loss": best_loss,
        }, save_path)
        print(f"  Saved checkpoint to {save_path}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "downstream_state": model.state_dict(),
                "cfg": asdict(cfg),
                "best_val_loss": best_loss,
            }, best_path)
            print(f"  Saved BEST to {best_path} (Val Loss: {best_loss:.4f})")

        def save_set_results(results, assay_ids, prefix):
            auc_list = results["auc_list"]
            
            if len(assay_ids) == len(auc_list):
                data_dict = {
                    "assay_id": assay_ids,
                    "auroc": auc_list
                }
                df = pd.DataFrame(data_dict)
                
                f1_per_task = results.get("f1_per_task", {})
                if f1_per_task:
                    sorted_thresholds = sorted(f1_per_task.keys())
                    for t in sorted_thresholds:
                        f1_list = f1_per_task[t]
                        if len(f1_list) == len(assay_ids):
                            df[f"f1_thr_{t:.2f}"] = f1_list
                
                valid_aucs = [a for a in auc_list if not np.isnan(a)]
                count_valid = len(valid_aucs)
                mean_auroc = np.mean(valid_aucs) if count_valid > 0 else 0.0
                
                df["global_mean_auroc"] = mean_auroc
                df["count_valid_tasks"] = count_valid

                csv_path = os.path.join(cfg.save_dir, f"epoch{epoch}_step{global_step}_{prefix}_metrics.csv")
                df.to_csv(csv_path, index=False)
                print(f"  Saved {prefix.upper()} Detailed Metrics CSV to {csv_path}")
            
            if results["raw_probs"] is not None and len(assay_ids) == results["raw_probs"].shape[1]:
                logits_csv_path = os.path.join(cfg.save_dir, f"epoch{epoch}_step{global_step}_{prefix}_logits.csv")
                save_logits_to_csv(
                    logits_csv_path,
                    results["raw_probs"],
                    results["raw_targets"],
                    assay_ids
                )
            else:
                print(f"  [Warning] Skipping {prefix.upper()} Logits CSV: Data missing or shape mismatch.")
        
        save_set_results(val_results, val_assay_ids, "val")
        
        if test_results is not None and test_assay_ids is not None:
            save_set_results(test_results, test_assay_ids, "test")

    return best_loss

def get_params_groups(model, weight_decay=1e-4):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith(".bias") or len(param.shape) == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]

def train(cfg: DSConfig):
    local_rank = cfg.local_rank
    rank = cfg.rank
    world_size = cfg.world_size

    set_seed(cfg)
    device = torch.device(cfg.device)
    if rank == 0:
        print(f"DDP: world_size={world_size} | local_rank={local_rank} | device={device}")
        wandb.init(
            project=cfg.project,
            name=cfg.run_name,
            mode=cfg.wandb_mode,
            id=cfg.log_id,
            config=asdict(cfg),
        )

    train_assay_embeds, _ = load_and_broadcast_assay(cfg.assay_json, cfg.train_label_csv, device, rank)
    val_assay_embeds, val_assay_ids = load_and_broadcast_assay(cfg.assay_json, cfg.val_label_csv, device, rank)
    
    test_assay_embeds = None
    test_assay_ids = None
    if cfg.test_on and cfg.test_label_csv:
        test_assay_embeds, test_assay_ids = load_and_broadcast_assay(cfg.assay_json, cfg.test_label_csv, device, rank)

    train_ds = MultiLabelPairDataset(
        image_csv_path=cfg.train_image_csv,
        label_csv_path=cfg.train_label_csv, 
        resize_short_side=cfg.resize_short_side,
    )
    val_ds = MultiLabelPairDataset(
        image_csv_path=cfg.val_image_csv,
        label_csv_path=cfg.val_label_csv,   
        resize_short_side=cfg.resize_short_side,
        use_center_crop=True
    )

    test_loader = None
    if cfg.test_on and cfg.test_image_csv and cfg.test_label_csv:
        if rank == 0: print(f"Rank 0: Loading Test Dataset from {cfg.test_image_csv}...")
        test_ds = MultiLabelPairDataset(
            image_csv_path=cfg.test_image_csv,
            label_csv_path=cfg.test_label_csv, 
            resize_short_side=cfg.resize_short_side,
            use_center_crop=True
        )
        test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, sampler=test_sampler,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=False
        )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    
    model = DownstreamModelTQN(
        device=device,
        load_pretrained=True,
        ckpt_path=cfg.pretrain_ckpt,
        freeze_backbone=cfg.freeze_backbone,
        dropout=cfg.dropout,
        out_dim=cfg.out_dim
    ).to(device)
    model.forward = model.forward_vision
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if cfg.resume_ckpt is not None:
        if rank == 0:
            print(f"Rank 0: Resuming/Finetuning from {cfg.resume_ckpt}...")
        ckpt = torch.load(cfg.resume_ckpt, map_location=device, weights_only=True)
        missing, unexpected = model.module.load_state_dict(ckpt['downstream_state'], strict=True)
        if rank == 0:
            print(f"Rank 0: Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    param_groups = get_params_groups(model, weight_decay=cfg.weight_decay)
    optim = torch.optim.AdamW(param_groups, lr=cfg.lr)
    scaler = GradScaler()
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_loss = float('inf')
    best_path = os.path.join(cfg.save_dir, "best.pt")

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        model.train()
        epoch_loss_local, n_seen_local = 0.0, 0

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{cfg.epochs}") if rank == 0 else train_loader
        
        for batch in pbar:
            images, smiles, targets, mask = batch
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            optim.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                out = model(images) 
                loss = masked_bce_with_logits(out["logits"], targets, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            bs = images.size(0)
            epoch_loss_local += float(loss) * bs
            n_seen_local += bs
            global_step += 1

            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    "loss": f"{(epoch_loss_local / max(1, n_seen_local)):.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                })

            if global_step % cfg.interval_step == 0:
                t = torch.tensor([epoch_loss_local, n_seen_local], dtype=torch.float64, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                current_train_loss = (t[0] / t[1]).item() if t[1] > 0 else float("nan")
                
                best_loss = run_eval_and_save(
                    cfg, model.module, 
                    val_loader, val_assay_embeds, val_assay_ids,
                    test_loader, test_assay_embeds, test_assay_ids, 
                    epoch, global_step, current_train_loss, best_loss, best_path
                )

def parse_args() -> DSConfig:
    import argparse
    p = argparse.ArgumentParser(description="Downstream DDP with ComCLIP + TQN Residual")
    p.add_argument("--train_image_csv", type=str, required=True)
    p.add_argument("--val_image_csv", type=str, required=True)
    p.add_argument("--test_image_csv", type=str, default=None)
    p.add_argument("--train_label_csv", type=str, required=True)
    p.add_argument("--val_label_csv", type=str, required=True)
    p.add_argument("--test_label_csv", type=str, default=None)
    p.add_argument("--resize_short_side", type=int, default=780)
    p.add_argument("--assay_json", type=str, required=True)
    p.add_argument("--test_on", action="store_true")
    p.add_argument("--out_dim", type=int, default=54)
    p.add_argument("--pretrain_ckpt", type=str, default=None)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--freeze_backbone", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=3.0)
    p.add_argument("--interval_step", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="./down_ckpts_tqn_residual")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", type=str, default="comclip-tqn")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--log_id", type=str, default="test")
    p.add_argument("--wandb_mode", type=str, default="disabled")
    args = p.parse_args()
    cfg = DSConfig(**vars(args))
    return cfg

@record
def main():
    cfg = parse_args()
    cfg = setup_distributed(cfg)
    train(cfg)

if __name__ == "__main__":
    main()