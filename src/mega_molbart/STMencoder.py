import sys
from rdkit import Chem
from pathlib import Path

import torch
import torch.nn as nn

# 假设这些包在他的环境中是可用的，保持原样
from mega_molbart.tokenizer import MolEncTokenizer
from mega_molbart.util import REGEX, DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH
from mega_molbart.decoder import DecodeSampler
from mega_molbart.megatron_bart import MegatronBART

class MolSTM_Extractor(nn.Module):
    def __init__(
        self,
        ckpt_path="/mnt/petrelfs/zhengqiaoyu.p/SunYuze/comclip_copy/molbart/molecule_model.pth",
        vocab_path="/mnt/petrelfs/zhengqiaoyu.p/SunYuze/comclip_copy/molbart/MolBART/bart_vocab.txt",
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.vocab_path = vocab_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = MolEncTokenizer.from_vocab_file(self.vocab_path, REGEX, DEFAULT_CHEM_TOKEN_START)
        self._load_model()

    def _load_model(self):
        state_dict = torch.load(self.ckpt_path, map_location="cpu")

        vocab_size = len(self.tokenizer)
        pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]

        d_model = state_dict["emb.weight"].shape[1]
        self.dim_model = d_model
        
        num_layers = max(int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.layers.")) + 1
        d_feedforward = state_dict["encoder.layers.0.fc1.weight"].shape[0]
        num_heads = 8
        self.max_seq_len = state_dict["pos_emb"].shape[0]

        sampler = DecodeSampler(self.tokenizer, max_seq_len=self.max_seq_len)
        self._model = MegatronBART(
            decode_sampler=sampler,
            pad_token_idx=pad_token_idx,
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            max_seq_len=self.max_seq_len,
            dropout=0.1,
        )

        missing, unexpected = self._model.load_state_dict(state_dict, strict=True)

    def _process_inputs(self, drugs):
        """内部辅助函数：处理SMILES到Tensor"""
        drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in drugs]
        
        tok = self.tokenizer.tokenize(drugs, pad=True)
        ids = self.tokenizer.convert_tokens_to_ids(tok['original_tokens'])

        encoder_input = torch.tensor(ids, dtype=torch.long).T                 # (seq_len, batch)
        encoder_pad_mask = torch.tensor(tok['masked_pad_masks'], dtype=torch.bool).T  # (seq_len, batch)
        
        device = next(self._model.parameters()).device
        encoder_input = encoder_input.to(device)
        encoder_pad_mask = encoder_pad_mask.to(device)
        
        # 截断到模型最大长度
        encoder_input = encoder_input[:self.max_seq_len] 
        encoder_pad_mask = encoder_pad_mask[:self.max_seq_len]
        return encoder_input, encoder_pad_mask

    def forward(self, drugs):
        '''
        原始方法：返回聚合后的向量 (Batch, Hidden)
        '''
        encoder_input, encoder_pad_mask = self._process_inputs(drugs)

        memory = self._model.encode({
            "encoder_input": encoder_input,
            "encoder_pad_mask": encoder_pad_mask,
        }) # seq_len * batch * d_model
        
        valid = (~encoder_pad_mask).float()
        weights = valid / (valid.sum(dim=0, keepdim=True) + 1e-9)
        weights = weights.unsqueeze(-1) # seq_len * batch * 1
        mol_vec = (memory.squeeze(1) * weights).sum(dim=0) # batch * d_model
        
        return mol_vec

    def forward_seq_nocls(self, drugs):
        '''
        [新增方法] 下游任务专用接口
        Returns:
            token_embeddings: (Batch, Seq_Len, Hidden) - 调整为 batch_first 方便下游使用
            pad_mask: (Batch, Seq_Len) - True 表示是 Padding，需要被 Mask 掉
        '''
        encoder_input, encoder_pad_mask = self._process_inputs(drugs)

        # memory shape: (seq_len, batch, d_model)
        memory = self._model.encode({
            "encoder_input": encoder_input,
            "encoder_pad_mask": encoder_pad_mask,
        }) 
        
        # 调整维度为 (batch, seq_len, d_model)
        token_embeddings = memory.permute(1, 0, 2)
        
        # 调整 mask 维度为 (batch, seq_len)
        # 注意：这里的 mask 为 True 代表是 Padding
        pad_mask = encoder_pad_mask.permute(1, 0)
        
        return token_embeddings, pad_mask

    def forward_seq(self, drugs):
        '''
        [修改后] 下游任务专用接口
        功能：
        1. 计算原始的序列 embeddings。
        2. 计算聚合后的 mol_vec (相当于 CLS token)。
        3. 将 mol_vec 拼接到序列的最前面。
        4. 对应修改 mask，给 CLS token 添加 False (非padding) 标记。
        
        Returns:
            final_embeddings: (Batch, 1 + Seq_Len, Hidden)
            final_mask: (Batch, 1 + Seq_Len)
        '''
        encoder_input, encoder_pad_mask = self._process_inputs(drugs)

        # memory shape: (seq_len, batch, d_model)
        memory = self._model.encode({
            "encoder_input": encoder_input,
            "encoder_pad_mask": encoder_pad_mask,
        }) 
        
        # --- 步骤 1: 计算聚合向量 (复用 forward 的逻辑) ---
        # valid shape: (seq_len, batch)
        valid = (~encoder_pad_mask).float()
        # weights shape: (seq_len, batch, 1)
        weights = valid / (valid.sum(dim=0, keepdim=True) + 1e-9)
        weights = weights.unsqueeze(-1) 
        
        # mol_vec shape: (batch, d_model)
        # 注意：这里直接用 memory * weights 即可，不需要 squeeze(1)，
        # 这样可以兼容 batch_size > 1 的情况
        mol_vec = (memory * weights).sum(dim=0) 
        
        # --- 步骤 2: 准备序列数据 ---
        # 调整 memory 维度为 (batch, seq_len, d_model)
        token_embeddings = memory.permute(1, 0, 2)
        
        # --- 步骤 3: 拼接 CLS Token ---
        # 将 mol_vec 扩充维度变成 (batch, 1, d_model) 以便拼接
        cls_token = mol_vec.unsqueeze(1)
        
        # 拼接: [CLS, Token1, Token2, ...] -> (batch, 1 + seq_len, d_model)
        final_embeddings = torch.cat([cls_token, token_embeddings], dim=1)
        
        # --- 步骤 4: 处理 Mask ---
        # 原始 mask: (batch, seq_len)
        pad_mask = encoder_pad_mask.permute(1, 0)
        
        # 创建 CLS 的 mask: (batch, 1)
        # 这里的 mask 定义是: True 代表 Pad (需要被忽略), False 代表有效内容。
        # CLS token 是有效内容，所以全是 False。
        cls_mask = torch.zeros(
            (pad_mask.shape[0], 1), 
            dtype=pad_mask.dtype, 
            device=pad_mask.device
        )
        
        # 拼接 mask: [False, Mask1, Mask2, ...] -> (batch, 1 + seq_len)
        final_mask = torch.cat([cls_mask, pad_mask], dim=1)
        
        return final_embeddings, final_mask