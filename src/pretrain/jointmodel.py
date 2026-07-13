# jointmodel.py
from typing import Any, Dict, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pretrain.vision_transformer import TimmViTUnified
from timm.layers import trunc_normal_
from mega_molbart.STMencoder import MolSTM_Extractor

class SMILESEncoder(nn.Module): 
    def __init__(self, ckpt_path: str, vocab_path: str):
        super().__init__()
        self.extractor = MolSTM_Extractor(ckpt_path=ckpt_path, vocab_path=vocab_path) 
    
        for p in self.extractor.parameters():
            p.requires_grad = False

    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        feats = self.extractor.forward(smiles_list) 
        return feats

class GateFusion(nn.Module): 
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, zi: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        gate_alpha = self.gate(torch.cat([zi, zc], dim=-1))
        fusion_embedding = gate_alpha * zi + (1 - gate_alpha) * zc
        vision_embedding = self.norm(fusion_embedding)
        return vision_embedding

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs) if warmup_teacher_temp_epochs > 0 else np.array([teacher_temp]),
            np.ones(max(0, nepochs - warmup_teacher_temp_epochs)) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch: int):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v, s in enumerate(student_out):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss = total_loss / n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0.0):
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * niter_per_ep)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * \
               (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class JointModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        img_size: int = 224,
        pretrained: bool = True,
        patch_size: int = 16,
        mae_mask_ratio: float = 0.75,
        embed_dim: int = 512,
        molstm_ckpt: str = "/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction_cleaned/MoleculeSTM/molecule_model.pth",
        molstm_vocab: str = "/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction_cleaned/MoleculeSTM/bart_vocab.txt",
        dino_out_dim: int = 65536,
        dino_use_bn_in_head: bool = False,
        dino_ncrops: int = 10,
        dino_student_temp: float = 0.1,
        dino_center_momentum: float = 0.9,
        dino_momentum_teacher: float = 0.996,
        dino_warmup_teacher_temp: float = 0.04,
        dino_teacher_temp: float = 0.04,
        dino_warmup_teacher_temp_epochs: int = 0,
        total_epochs: int = 100,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vision = TimmViTUnified(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            pretrained=pretrained,
        )
        self.embed_dim_backbone = self.vision.embed_dim
        self.mae_mask_ratio = mae_mask_ratio

        self.image_proj = nn.Linear(self.embed_dim_backbone, embed_dim)
        self.text_proj = nn.Linear(256, 512)

        self.fusion = GateFusion(dim=self.embed_dim_backbone)
        self.text_encoder = SMILESEncoder(ckpt_path=molstm_ckpt, vocab_path=molstm_vocab)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dino_student_head = DINOHead(
            in_dim=self.embed_dim_backbone,
            out_dim=dino_out_dim,
            use_bn=dino_use_bn_in_head,
        )
        self.dino_teacher_head = DINOHead(
            in_dim=self.embed_dim_backbone,
            out_dim=dino_out_dim,
            use_bn=dino_use_bn_in_head,
        )
        self.teacher_backbone = TimmViTUnified(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            pretrained=pretrained,
        )
        self._init_teacher()

        self.dino_loss_fn = DINOLoss(
            out_dim=dino_out_dim,
            ncrops=dino_ncrops,
            warmup_teacher_temp=dino_warmup_teacher_temp,
            teacher_temp=dino_teacher_temp,
            warmup_teacher_temp_epochs=dino_warmup_teacher_temp_epochs,
            nepochs=total_epochs,
            student_temp=dino_student_temp,
            center_momentum=dino_center_momentum,
        )
        self.dino_momentum_teacher = dino_momentum_teacher
        self.total_epochs = total_epochs

        

    def _init_teacher(self):
        self.teacher_backbone.load_state_dict(self.vision.state_dict(), strict=True)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        self.dino_teacher_head.load_state_dict(self.dino_student_head.state_dict(), strict=True)
        for p in self.dino_teacher_head.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def update_teacher(self, momentum: float):
        student_state = dict(self.vision.named_parameters())
        for name, pt in self.teacher_backbone.named_parameters():
            if name in student_state:
                ps = student_state[name]
                pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)
        student_buffers = dict(self.vision.named_buffers())
        for name, bt in self.teacher_backbone.named_buffers():
            if name in student_buffers:
                sb = student_buffers[name]
                if bt.dtype == sb.dtype and bt.shape == sb.shape:
                    bt.data.mul_(momentum).add_((1.0 - momentum) * sb.data)

        for ps, pt in zip(self.dino_student_head.parameters(), self.dino_teacher_head.parameters()):
            pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)

    def get_dino_momentum(self, epoch: int, it_in_epoch: int, niter_per_ep: int,
                          final_momentum: float = 1.0) -> float:
        schedule = cosine_scheduler(
            base_value=self.dino_momentum_teacher,
            final_value=final_momentum,
            epochs=self.total_epochs,
            niter_per_ep=niter_per_ep,
            warmup_epochs=0,
            start_warmup_value=self.dino_momentum_teacher,
        )
        idx = min(epoch * niter_per_ep + it_in_epoch, len(schedule) - 1)
        return float(schedule[idx])

    def encode_image_clip(self, images: torch.Tensor) -> torch.Tensor:
        zi = self.vision.clip_forward(images)
        return zi

    def encode_image_fused(self, treated: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        zi = self.encode_image_clip(treated)
        zc = self.encode_image_clip(control)
        zf = self.fusion(zi, zc)
        zf = self.image_proj(zf)
        zf = F.normalize(zf, dim=1, p=2)
        return zf

    def encode_text(self, smiles_list: List[str]) -> torch.Tensor:
        t = self.text_encoder(smiles_list)
        t = self.text_proj(t)
        t = F.normalize(t, dim=1, p=2)
        return t

    @staticmethod
    def top1_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return (pred == targets).float().mean().item()
   
    def encode_image_single(self, images: torch.Tensor) -> torch.Tensor:
        zi = self.vision.clip_forward(images)
        return zi

    def encode_text_single(self, smiles_list: List[str]) -> torch.Tensor:
        t = self.text_encoder(smiles_list)
        t = self.text_proj(t)
        return t

    def encode_text_only(self, smiles_list: List[str]) -> torch.Tensor:
        t = self.text_encoder(smiles_list)
        return t
    
    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        features = self.vision.forward_patches(images)
        z_seq = self.image_proj(features)
        return z_seq

    def encode_image_patches_nocls(self, images: torch.Tensor) -> torch.Tensor:
        features = self.vision.forward_patches_nocls(images)
        z_seq = self.image_proj(features)
        return z_seq

    def clip_loss(self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
        target = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        loss_i = F.cross_entropy(logits_per_image, target)
        loss_t = F.cross_entropy(logits_per_text, target)
        return 0.5 * (loss_i + loss_t)

    def forward(
        self,
        treated_images: torch.Tensor,
        control_images: torch.Tensor,
        smiles_list: List[str],
        dino_images: Optional[List[torch.Tensor]] = None,  
        *,
        compute_clip: bool = True,
        compute_mae: bool = True,
        compute_dino: bool = True,
        current_epoch: int = 0,
        
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if compute_clip:
            img_feats = self.encode_image_fused(treated_images, control_images)
            txt_feats = self.encode_text(smiles_list)
            logit_scale = self.logit_scale.exp().clamp(1/100.0, 100.0)
            logits_per_image = logit_scale * img_feats @ txt_feats.t()
            logits_per_text = logits_per_image.t()
            num_samples = logits_per_image.shape[0]
            
            targets = torch.arange(num_samples, device=logits_per_image.device)
            acc_i = self.top1_accuracy_from_logits(logits_per_image, targets)
            acc_t = self.top1_accuracy_from_logits(logits_per_text, targets)
            acc = (acc_i + acc_t) / 2

            loss_clip = self.clip_loss(logits_per_image, logits_per_text)
            out.update({
                "logit_scale": logit_scale,
                "loss_clip": loss_clip,
                "acc": acc,
            })

        if compute_mae:
            pred, mask, ids_restore = self.vision.mae_forward(treated_images, mask_ratio=self.mae_mask_ratio)
            loss_mae = self.vision.mae_loss(treated_images, pred, mask)
            out.update({
                "mae_pred": pred,
                "mae_mask": mask,
                "mae_ids_restore": ids_restore,
                "loss_mae": loss_mae,
            })

        if compute_dino:
            device = treated_images.device

            def build_idx_crops(x_list: List[torch.Tensor]) -> List[int]:
                sizes = torch.tensor([int(x.shape[-1]) for x in x_list], device=device)
                _, counts = torch.unique_consecutive(sizes, return_counts=True)
                idx = torch.cumsum(counts, dim=0).tolist()
                return [int(i) for i in idx]

            def student_backbone_forward(x_list: List[torch.Tensor]) -> torch.Tensor:
                idx_crops_local = build_idx_crops(x_list)
                outputs = []
                start = 0
                for end in idx_crops_local:
                    if end <= start:
                        continue  
                    x_cat = torch.cat(x_list[start:end], dim=0)
                    feats = self.vision.dino_forward(x_cat)
                    outputs.append(feats)
                    start = end
                return torch.cat(outputs, dim=0)

            def teacher_backbone_forward(x_list: List[torch.Tensor]) -> torch.Tensor:
                idx_crops_local = build_idx_crops(x_list)
                outputs = []
                start = 0
                for end in idx_crops_local:
                    if end <= start:
                        continue  
                    x_cat = torch.cat(x_list[start:end], dim=0)
                    feats = self.teacher_backbone.dino_forward(x_cat)
                    outputs.append(feats)
                    start = end
                return torch.cat(outputs, dim=0)

            student_feats = student_backbone_forward(dino_images)
            student_out = self.dino_student_head(student_feats)

            teacher_inputs = dino_images[:2]
            with torch.no_grad():
                teacher_feats = teacher_backbone_forward(teacher_inputs)
                teacher_out = self.dino_teacher_head(teacher_feats)

            loss_dino = self.dino_loss_fn(student_out, teacher_out, current_epoch)
            out.update({"loss_dino": loss_dino})

        return out

    