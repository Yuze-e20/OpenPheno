import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import os

from pretrain.jointmodel import JointModel

class MultiLabelHead(nn.Module):
    
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 270, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FrozenWrapper(nn.Module):
    def __init__(
        self,
        device: torch.device,
        load_pretrained: bool = True,
        ckpt_path: Optional[str] = None,
        freeze_backbone: bool = True, 
        dropout: float = 0.1,        
    ):
        super().__init__()
    
        self.model = JointModel(pretrained=True, dino_out_dim=65536) 

        if load_pretrained and ckpt_path is not None:
            print(f"Loading pretrained weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            
            state = ckpt['model'] if 'model' in ckpt else ckpt
            self.model.load_state_dict(state, strict=False)

        self.embed_dim = self.model.embed_dim
        
        for p in self.model.parameters():
                p.requires_grad = False
        if freeze_backbone:
            self.model.eval()
        else:
            self.model.train()
            for p in self.model.vision.parameters():
                p.requires_grad = True
            for p in self.model.image_proj.parameters():
                p.requires_grad = True

        self.model.to(device)
        self.device = device
        self.img_post_process = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout)
        )
        self.txt_post_process = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout)
        )

    def encode_image_patches(self, treated: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image_patches(treated) 
        features = self.img_post_process(features)
        return features

    def encode_image_cls(self, treated: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image_single(treated)
        return features

    def encode_text_cls(self, smiles_list: List[str]) -> torch.Tensor:
        txt_emb = self.model.encode_text_only(smiles_list)
        return txt_emb

    def encode_text(self, smiles_list: List[str]) -> torch.Tensor:
        txt_emb = self.model.encode_text_single(smiles_list) 
        txt_emb = self.txt_post_process(txt_emb)
        return txt_emb

class AssayQueryTransformer(nn.Module):
    def __init__(self, 
                 model_dim: int = 512, 
                 assay_input_dim: int = 768, 
                 num_decoder_layers: int = 2, 
                 nhead: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        
        self.assay_proj = nn.Sequential(
            nn.Linear(assay_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout) 
        )

        self.gate_proj = nn.Linear(model_dim * 2, model_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, 
            nhead=nhead, 
            dim_feedforward=2048, 
            dropout=dropout, 
            batch_first=True, 
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(model_dim))
        
        self.output_head = nn.Sequential(
            nn.Linear(model_dim, model_dim), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1)        
        )

    def forward(self, 
                assay_embeds: torch.Tensor, 
                kv_features: torch.Tensor, 
                structure_emb: torch.Tensor, 
                kv_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        B = kv_features.shape[0]
        N_tasks = assay_embeds.shape[0]
        D = self.model_dim

        
        static_feats = self.assay_proj(assay_embeds) 
        dynamic_feats = structure_emb               

        static_expanded = static_feats.unsqueeze(0).expand(B, -1, -1)    
        struct_expanded = dynamic_feats.unsqueeze(1).expand(-1, N_tasks, -1) 

        concat_feats = torch.cat([static_expanded, struct_expanded], dim=-1)
        alpha = torch.sigmoid(self.gate_proj(concat_feats))

        dynamic_queries = alpha * static_expanded + (1 - alpha) * struct_expanded

        fused_features = self.decoder(
            tgt=dynamic_queries, 
            memory=kv_features, 
            memory_key_padding_mask=kv_padding_mask
        )

        logits = self.output_head(fused_features).squeeze(-1) 
        
        return logits

class DownstreamModelTQN(nn.Module):
    def __init__(
        self,
        device: torch.device,
        load_pretrained: bool = True,
        ckpt_path: Optional[str] = None,
        freeze_backbone: bool = True,
        out_dim: int = 270,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        print(f"\n[DownstreamModelTQN] initialize TQN (Enhanced Proj & Head Version)...")
        
       
        self.backbone = FrozenWrapper(
            device=device,
            load_pretrained=load_pretrained,
            ckpt_path=ckpt_path,
            freeze_backbone=freeze_backbone,
            dropout=dropout 
        )
        D = self.backbone.embed_dim 
        
        
        self.tqn = AssayQueryTransformer(
            model_dim=D,
            num_decoder_layers=4, 
            nhead=8,
            dropout=dropout
        )

        self.head_vision = MultiLabelHead(in_dim=768, hidden=512, out_dim=out_dim, dropout=dropout)
        self.head_smiles = MultiLabelHead(in_dim=256, hidden=512, out_dim=out_dim, dropout=dropout)


    def forward_vision(self, treated_images: torch.Tensor) -> Dict[str, Any]:
        zi = self.backbone.encode_image_cls(treated_images)
        logits = self.head_vision(zi)
        return {
            "logits": logits,
        }

    def forward_smiles(self, smiles_list: List[str]) -> Dict[str, Any]: 
        zt = self.backbone.encode_text_cls(smiles_list)
        logits = self.head_smiles(zt)
        return {
            "logits": logits,
        }

    def forward(
        self,
        treated_images: torch.Tensor,    
        smiles_list: List[str], 
        assay_embeddings: torch.Tensor,          
    ) -> Dict[str, Any]:
        
        
        img_seq = self.backbone.encode_image_patches(treated_images)
        txt_emb = self.backbone.encode_text(smiles_list)
        
        logits = self.tqn(
            assay_embeds=assay_embeddings, 
            kv_features=img_seq,
            structure_emb=txt_emb 
        )
        
        return {
            "logits": logits,
            "img_seq": img_seq,
            "txt": txt_emb
        }