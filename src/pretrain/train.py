import os
from PIL import Image
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from pretrain.dataset import MultiViewCompoundDataset
from pretrain.jointmodel import JointModel, cosine_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

@dataclass
class TrainConfig:
    train_csv_path: str = "/path/to/train_csv_path"
    val_csv_path: str = "/path/to/val_csv_path"
    control_json_path: str = "/path/to/control_json_path"
    batch_size: int = 64
    num_workers: int = 8
    global_crop_size: int = 224
    local_crop_size: int = 96
    locals_per_global: int = 0
    resize_short_side: int = 780
    in_channels: int = 5
    embed_dim: int = 512
    pretrained: bool = True
    mae_mask_ratio: float = 0.75
    molstm_ckpt: str = "/path/to/molecule_model.pth"
    molstm_vocab: str = "/path/to/bart_vocab.txt"
    dino_out_dim: int = 4096
    dino_use_bn_in_head: bool = True
    freeze_last_layer_epochs: int = 1
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = 'cuda'
    lr: float = 1e-4
    weight_decay: float = 0.01
    weight_decay_end: float = 0.01  
    min_lr: float = 0.0            
    warmup_epochs: int = 0         
    epochs: int = 10
    grad_clip: float = 3.0
    lambda_clip: float = 1.0
    lambda_mae: float = 1.0
    lambda_dino: float = 1.0
    project: str = "joint_clip_mae_dino"
    run_name: str = None
    wandb_mode: str = "offline"
    save_dir: str = "./checkpoints_joint_cmd"
    save_every: int = 0
    resume: str = None
    log_id: str = 'new'
    petrel_conf_path: str = None


def save_visualization(original, masked_vis, reconstructed, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, 'visualization')
    os.makedirs(save_dir, exist_ok=True)
    
    def tensor_to_image(t):
        t = t[:3].clamp(0, 1)
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    tensor_to_image(original).save(os.path.join(save_dir, f"epoch_{epoch:04d}_original.png"))
    tensor_to_image(masked_vis).save(os.path.join(save_dir, f"epoch_{epoch:04d}_masked.png"))
    tensor_to_image(reconstructed).save(os.path.join(save_dir, f"epoch_{epoch:04d}_reconstructed.png"))

def create_masked_visualization(original, mask, patch_size=16, gray_value=0.5):
    C, H, W = original.shape
    p = patch_size
    h = H // p
    w = W // p

    mask_2d = mask.view(h, w) 

    mask_full = mask_2d.unsqueeze(0).unsqueeze(0).float() 
    mask_full = torch.nn.functional.interpolate(
        mask_full,
        size=(H, W),
        mode='nearest'
    ) 
    mask_full = mask_full.squeeze(0).squeeze(0) 

    mask_full = mask_full.unsqueeze(0).expand(C, -1, -1) 

    vis = torch.where(mask_full > 0.5, torch.full_like(original, gray_value), original)

    return vis

def setup_distributed(cfg: TrainConfig):
    cfg.rank = int(os.environ["RANK"])
    cfg.world_size = int(os.environ["WORLD_SIZE"])
    cfg.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.local_rank)
    cfg.device = f"cuda:{cfg.local_rank}"
    dist.init_process_group(backend="nccl", init_method="env://")

    return cfg

def set_seed(cfg: TrainConfig, seed: int = 42):
    random.seed(seed + cfg.rank)
    torch.manual_seed(seed + cfg.rank)
    torch.cuda.manual_seed_all(seed + cfg.rank)
    np.random.seed(seed + cfg.rank)

def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "dino_student_head.last_layer" in n:
            p.grad = None


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def build_datasets(cfg: TrainConfig):
    train_ds = MultiViewCompoundDataset(
        csv_path=cfg.train_csv_path,
        control_json_path=cfg.control_json_path,
        global_crop_size=cfg.global_crop_size,
        local_crop_size=cfg.local_crop_size,
        locals_per_global=cfg.locals_per_global,
        resize_short_side=cfg.resize_short_side,
        petrel_conf_path=cfg.petrel_conf_path,
    )
    val_ds = MultiViewCompoundDataset(
        csv_path=cfg.val_csv_path,
        control_json_path=cfg.control_json_path,
        global_crop_size=cfg.global_crop_size,
        local_crop_size=cfg.local_crop_size,
        locals_per_global=cfg.locals_per_global,
        resize_short_side=cfg.resize_short_side,
        petrel_conf_path=cfg.petrel_conf_path,
    )
    return train_ds, val_ds


def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig, epoch: int) -> Dict[str, float]:
    device = torch.device(cfg.device)
    meters = {"clip": 0.0, "mae": 0.0, "dino": 0.0, "acc": 0.0, "all": 0.0}
    n_samples = 0
    with torch.no_grad():
        iterator = loader if cfg.rank != 0 else tqdm(loader, desc="Validate")

        for idx, batch in enumerate(iterator):
            treated = batch["treated_global1"].to(device, non_blocking=True)
            treated_another = batch["treated_global2"].to(device, non_blocking=True)
            control = batch["control_globals"].to(device, non_blocking=True)
            smiles = batch["smiles"]

            dino_images = [treated, treated_another]
            
            if "treated_locals" in batch:
                locals_stack = batch["treated_locals"].to(device, non_blocking=True)
                local_views = locals_stack.unbind(dim=1)
                dino_images.extend(local_views)
            
            if idx == 0 and cfg.lambda_mae > 0:
                vis_img = treated[0:1].to(device)
                with autocast(device_type="cuda"):
                    pred_vis, mask_vis, ids_restore_vis = model.vision.mae_forward(vis_img, mask_ratio=cfg.mae_mask_ratio)

                
                target = model.vision.patchify(vis_img)  
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                pred_pixels = pred_vis * (var + 1e-6).sqrt() + mean

                reconstructed = model.vision.unpatchify(pred_pixels)  

                original = vis_img[0].cpu() 
                masked_vis = create_masked_visualization(original=original, mask=mask_vis[0].cpu())

                save_visualization(
                    original, 
                    masked_vis, 
                    reconstructed[0].cpu(),  
                    epoch, 
                    cfg.save_dir
                )
                print(f"Visualization saved for epoch {epoch}")

            with autocast(device_type="cuda"):
                out = model(
                    treated_images=treated,
                    control_images=control,
                    smiles_list=smiles,
                    dino_images=dino_images if len(dino_images) >= 2 else None,
                    compute_clip=True,
                    compute_mae=True,
                    compute_dino=True,
                    current_epoch=epoch,
                )

            loss_clip = out.get("loss_clip", torch.tensor(0.0, device=device)).item()
            loss_mae = out.get("loss_mae", torch.tensor(0.0, device=device)).item()
            loss_dino = out.get("loss_dino", torch.tensor(0.0, device=device)).item()
            acc = out.get("acc", torch.tensor(0.0, device=device))
            loss = cfg.lambda_clip * loss_clip + cfg.lambda_mae * loss_mae + cfg.lambda_dino * loss_dino

            bs = treated.shape[0]
            meters["clip"] += loss_clip * bs
            meters["mae"] += loss_mae * bs
            meters["dino"] += loss_dino * bs
            meters["all"] += loss * bs
            meters["acc"] += acc * bs
            n_samples += bs

            if cfg.rank == 0:
                iterator.set_postfix({
                    "loss": f"{meters['all'] / n_samples:.4f}",
                    "clip": f"{meters['clip'] / n_samples:.4f}",
                    "mae": f"{meters['mae'] / n_samples:.4f}",
                    "dino": f"{meters['dino'] / n_samples:.4f}",
                    "acc": f"{meters['acc'] / n_samples:.4f}",
                })

    t = torch.tensor([meters["all"], meters["clip"], meters["mae"], meters["dino"], meters["acc"], n_samples], dtype=torch.float64, device=device)

    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total, clip, mae, dino, acc, n_samples = t.tolist()

    return {
        "val_loss": total / n_samples,
        "val_loss_clip": clip / n_samples,
        "val_loss_mae": mae / n_samples,
        "val_loss_dino": dino / n_samples,
        "val_acc": acc / n_samples,
    }


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    if cfg.rank == 0:
        print(f"λ: clip={cfg.lambda_clip}, mae={cfg.lambda_mae}, dino={cfg.lambda_dino}")
    set_seed(cfg, 42)

    if cfg.rank == 0:
        wandb.init(project=cfg.project, name=cfg.run_name, id=cfg.log_id, mode=cfg.wandb_mode, config=asdict(cfg))

    train_ds, val_ds = build_datasets(cfg)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=True, drop_last=False)
    local_batch = cfg.batch_size // cfg.world_size

    train_loader = DataLoader(
        train_ds,
        batch_size=local_batch,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=local_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = JointModel(
        in_channels=cfg.in_channels,
        img_size=cfg.global_crop_size,
        pretrained=cfg.pretrained,
        mae_mask_ratio=cfg.mae_mask_ratio,
        embed_dim=cfg.embed_dim,
        molstm_ckpt=cfg.molstm_ckpt,
        molstm_vocab=cfg.molstm_vocab,
        dino_out_dim=cfg.dino_out_dim,
        dino_use_bn_in_head=cfg.dino_use_bn_in_head,
        dino_ncrops=(2 + 2 * cfg.locals_per_global if cfg.locals_per_global > 0 else 2),
        total_epochs=cfg.epochs,
    ).to(device)

    model = DDP(model, device_ids=[cfg.local_rank], find_unused_parameters=False)
    params_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)

    niter_per_ep = len(train_loader)
    
    base_lr = cfg.lr * cfg.batch_size / 256.0
    
    lr_schedule = cosine_scheduler(
        base_value=base_lr,
        final_value=cfg.min_lr,
        epochs=cfg.epochs,
        niter_per_ep=niter_per_ep,
        warmup_epochs=cfg.warmup_epochs,
        start_warmup_value=0.0,
    )
    wd_schedule = cosine_scheduler(
        base_value=cfg.weight_decay,
        final_value=cfg.weight_decay_end,
        epochs=cfg.epochs,
        niter_per_ep=niter_per_ep,
        warmup_epochs=0,  
        start_warmup_value=cfg.weight_decay,
    )

    scaler = GradScaler()

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    best_epoch = -1

    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val_loss", best_val))
        best_epoch = int(ckpt.get("best_epoch", -1))
        if cfg.rank == 0:
            print(f"✅ Resume from {cfg.resume} | start_epoch={start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]") if cfg.rank == 0 else train_loader
        model.module.train()

        epoch_loss = epoch_clip = epoch_mae = epoch_dino = epoch_acc = 0.0
        n_seen = 0

        for it, batch in enumerate(pbar):

            treated = batch["treated_global1"].to(device, non_blocking=True)
            treated_another = batch["treated_global2"].to(device, non_blocking=True)
            control = batch["control_globals"].to(device, non_blocking=True)
            smiles = batch["smiles"]

            dino_images = [treated, treated_another]
            
            if "treated_locals" in batch:
                locals_stack = batch["treated_locals"].to(device, non_blocking=True)
                local_views = locals_stack.unbind(dim=1)
                dino_images.extend(local_views)

            it_global = epoch * niter_per_ep + it
            lr_now = float(lr_schedule[it_global])
            wd_now = float(wd_schedule[it_global])

            for pg_idx, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr_now
                if pg_idx == 0:
                    pg["weight_decay"] = wd_now

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                out = model(
                    treated_images=treated,
                    control_images=control,
                    smiles_list=smiles,
                    dino_images=dino_images,
                    compute_clip=True,
                    compute_mae=True,
                    compute_dino=True,
                    current_epoch=epoch,
                )
                loss_clip = out.get("loss_clip", torch.tensor(0.0, device=device))
                loss_mae = out.get("loss_mae", torch.tensor(0.0, device=device))
                loss_dino = out.get("loss_dino", torch.tensor(0.0, device=device))
                
                loss = cfg.lambda_clip * loss_clip + cfg.lambda_mae * loss_mae + cfg.lambda_dino * loss_dino
                acc = out.get("acc", torch.tensor(0.0, device=device))
                logit_scale = out.get('logit_scale', torch.tensor(0.0)).item()

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            cancel_gradients_last_layer(epoch, model.module, cfg.freeze_last_layer_epochs)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                niter_per_ep = len(train_loader)
                momentum = model.module.get_dino_momentum(
                    epoch=epoch, it_in_epoch=it, niter_per_ep=niter_per_ep, final_momentum=1.0
                )
                model.module.update_teacher(momentum)

            bs = treated.shape[0]
            epoch_loss += loss.item() * bs
            epoch_clip += loss_clip.item() * bs
            epoch_mae += loss_mae.item() * bs
            epoch_dino += loss_dino.item() * bs
            epoch_acc += acc * bs
            n_seen += bs

            if cfg.rank == 0:
                pbar.set_postfix({
                    "loss": f"{epoch_loss / n_seen:.4f}",
                    "clip": f"{epoch_clip / n_seen:.4f}",
                    "mae": f"{epoch_mae / n_seen:.4f}",
                    "dino": f"{epoch_dino / n_seen:.4f}",
                    "acc": f"{epoch_acc / n_seen:.4f}",
                    "lr": f"{lr_now:.2e}",
                    "wd": f"{wd_now:.2e}",
                    "T": f"{logit_scale:.2f}",
                })
                
                wandb.log({
                    "train_step_loss": loss.item(),
                    "train_step_loss_clip": loss_clip.item(),
                    "train_step_loss_mae": loss_mae.item(),
                    "train_step_loss_dino": loss_dino.item(),
                    "train_step_acc": acc,
                    "train_step_logit_scale": logit_scale,
                    "lr": lr_now,
                    "wd": wd_now,
                    "epoch": epoch + 1,
                    "step": it_global,
                })

        val_metrics = evaluate(model.module, val_loader, cfg, epoch)

        if cfg.rank == 0:
            tr_loss = epoch_loss / n_seen
            tr_clip = epoch_clip / n_seen
            tr_mae = epoch_mae / n_seen
            tr_dino = epoch_dino / n_seen
            tr_acc = epoch_acc / n_seen

            print(f"📊 [Epoch {epoch+1}] train loss={tr_loss:.4f} clip={tr_clip:.4f} mae={tr_mae:.4f} dino={tr_dino:.4f} acc={tr_acc:.4f} | "
                  f"val loss={val_metrics['val_loss']:.4f} clip={val_metrics['val_loss_clip']:.4f} mae={val_metrics['val_loss_mae']:.4f} dino={val_metrics['val_loss_dino']:.4f} acc={val_metrics['val_acc']:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                **val_metrics
            })

            current_val = val_metrics["val_loss"]
            if current_val < best_val:
                best_val = current_val
                best_epoch = epoch + 1
                os.makedirs(cfg.save_dir, exist_ok=True)
                best_path = os.path.join(cfg.save_dir, "best_joint_cmd.pt")
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch + 1,
                    "global_step": (epoch + 1) * niter_per_ep,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "val": val_metrics,
                }, best_path)
                print(f"💾 New best val={best_val:.4f} at epoch {epoch+1}. Saved: {best_path}")

            if cfg.save_every and ((epoch + 1) % cfg.save_every == 0):
                os.makedirs(cfg.save_dir, exist_ok=True)
                path = os.path.join(cfg.save_dir, f"epoch{epoch+1}.pt")
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch + 1,
                    "global_step": (epoch + 1) * niter_per_ep,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "val": val_metrics,
                }, path)
                print(f"💾 Saved checkpoint: {path}")

            os.makedirs(cfg.save_dir, exist_ok=True)
            path = os.path.join(cfg.save_dir, f"latest.pt")
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch + 1,
                "global_step": (epoch + 1) * niter_per_ep,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "val": val_metrics,
            }, path)
            print(f"💾 Saved checkpoint: {path}")


def parse_args() -> TrainConfig:
    import argparse
    parser = argparse.ArgumentParser(description="Joint Training: CLIP + MAE + DINO (teacher has independent backbone)")

    parser.add_argument("--train_csv_path", type=str, required=True)
    parser.add_argument("--val_csv_path", type=str, required=True)
    parser.add_argument("--control_json_path", type=str, required=True)
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--global_crop_size", type=int, default=224)
    parser.add_argument("--local_crop_size", type=int, default=96)
    parser.add_argument("--locals_per_global", type=int, default=4)
    parser.add_argument("--resize_short_side", type=int, default=780)

    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--mae_mask_ratio", type=float, default=0.6)

    parser.add_argument("--molstm_ckpt", type=str, default="/path/to/molecule_model.pth")
    parser.add_argument("--molstm_vocab", type=str, default="/path/to/bart_vocab.txt")

    parser.add_argument("--dino_out_dim", type=int, default=65536)
    parser.add_argument("--dino_use_bn_in_head", action="store_true", default=False)
    parser.add_argument("--freeze_last_layer_epochs", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)         
    parser.add_argument("--min_lr", type=float, default=1e-6)     
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--weight_decay_end", type=float, default=0.4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--lambda_clip", type=float, default=1.0)
    parser.add_argument("--lambda_mae", type=float, default=1.0)
    parser.add_argument("--lambda_dino", type=float, default=1.0)

    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_id", type=str, default=None)

    parser.add_argument("--petrel_conf_path", type=str, default=None)

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    
    return cfg

def main():
    cfg = parse_args()
    cfg = setup_distributed(cfg)
    train(cfg)

if __name__ == "__main__":
    main()