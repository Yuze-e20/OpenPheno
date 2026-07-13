import os
import argparse
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from downstream.downstreamdataset_new import MultiViewCompoundDataset
from downstream.model_tqn import DownstreamModelTQN
from downstream.assay_utils import load_and_encode_assays

SELECT_CH_R = 2
SELECT_CH_G = 3
SELECT_CH_B = 4
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")
class AttentionCatcher:
    def __init__(self, model):
        self.model = model
        self.captured_weights = []
        self._inject_hook()

    def _inject_hook(self):
        target_layer = self.model.tqn.decoder.layers[-1]
        self.original_forward = target_layer.multihead_attn.forward
        
        def custom_forward(*args, **kwargs):
            kwargs['need_weights'] = True
            attn_output, attn_weights = self.original_forward(*args, **kwargs)
            self.captured_weights.append(attn_weights.detach().cpu())
            return attn_output, attn_weights

        target_layer.multihead_attn.forward = custom_forward

    def get_last_weights(self):
        if self.captured_weights:
            return self.captured_weights[-1]
        return None

    def clear(self):
        self.captured_weights = []

def process_channel_norm(img_tensor, channel_idx):
    img = img_tensor[channel_idx].cpu().numpy()
    denom = img.max() - img.min()
    if denom == 0:
        return img - img.min()
    return (img - img.min()) / denom

def prepare_vis_data(img_tensor, attn_map_vector):
    idx_r = SELECT_CH_R - 1
    idx_g = SELECT_CH_G - 1
    idx_b = SELECT_CH_B - 1
    
    img_r = process_channel_norm(img_tensor, idx_r)
    img_g = process_channel_norm(img_tensor, idx_g)
    img_b = process_channel_norm(img_tensor, idx_b)
    
    rgb_img = np.stack([img_r, img_g, img_b], axis=2)
    
    patch_attn = attn_map_vector[1:]
    grid_size = int(np.sqrt(patch_attn.shape[0]))
    
    att_mat = patch_attn.reshape(grid_size, grid_size).numpy()
    
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    att_mat = cv2.resize(att_mat, (W, H), interpolation=cv2.INTER_CUBIC)
    
    denom = att_mat.max() - att_mat.min()
    if denom != 0:
        att_mat = (att_mat - att_mat.min()) / denom
    else:
        att_mat = att_mat - att_mat.min()
        
    return rgb_img, att_mat

def plot_single_row(save_path, rgb_img, att_mat, assay_id_str, smiles, show_title=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(rgb_img)
    if show_title:
        axes[0].set_title(f"Composite (R:{SELECT_CH_R}, G:{SELECT_CH_G}, B:{SELECT_CH_B})", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(att_mat, cmap='jet')
    if show_title:
        axes[1].set_title(f"Attn: {assay_id_str}", fontsize=12, color='red', fontweight='bold')
    axes[1].axis('off')
    
    if show_title:
        plt.suptitle(f"SMILES: {smiles[:60]}...", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

def plot_combined_rows(save_path, all_rows_data, assay_ids, smiles):
    num_assays = len(all_rows_data)
    fig, axes = plt.subplots(num_assays, 2, figsize=(10, 4 * num_assays))
    
    if num_assays == 1:
        axes = axes.reshape(1, -1)

    for row_idx in range(num_assays):
        rgb_img, a_mat = all_rows_data[row_idx]
        aid = assay_ids[row_idx]
        
        axes[row_idx, 0].imshow(rgb_img)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title("RGB Composite", fontsize=14)
        
        axes[row_idx, 1].imshow(a_mat, cmap='jet')
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].set_title(f"Assay: {aid}", fontsize=12, color='blue')

    plt.suptitle(f"Combined Visualization for {smiles[:50]}...", fontsize=16, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

def get_assay_list_from_csv(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header[1:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='path/to/downstream/ckpt')
    parser.add_argument("--image_csv", type=str, default='./data/Broad-270/compound_splits/all.csv')
    parser.add_argument("--assay_json", type=str, default='./data/Broad-270/assay_meta/assay_description.json')
    parser.add_argument("--label_csv", type=str, default='./data/Broad-270/label/compound_assay_matrix_270.csv')
    parser.add_argument("--output_dir", type=str, default="./visulization/attention_map/results")
    parser.add_argument("--petrel_conf", type=str, default='/mnt/petrelfs/zhengqiaoyu.p/SunYuze/petreloss.conf')
    parser.add_argument("--assay_id", type=str, default='215_685', help="Specific assay ID. If not provided, visualize all assays.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_assay_ids = get_assay_list_from_csv(args.label_csv)
    target_indices = []
    
    if args.assay_id and args.assay_id.strip() != "":
        try:
            idx = all_assay_ids.index(args.assay_id)
            target_indices.append((idx, args.assay_id))
            print(f"Targeting single assay: {args.assay_id}")
        except ValueError:
            print(f"Error: Assay ID {args.assay_id} not found in CSV.")
            return
    else:
        print(f"No specific assay ID provided. Targeting ALL {len(all_assay_ids)} assays.")
        for idx, aid in enumerate(all_assay_ids):
            target_indices.append((idx, aid))

    assay_embeds_cpu, _ = load_and_encode_assays(args.assay_json, args.label_csv)
    assay_embeds = assay_embeds_cpu.to(device)
    
    print(f"Loading model...")
    model = DownstreamModelTQN(device=device, load_pretrained=False, ckpt_path=None, freeze_backbone=True)
    model.to(device)
    model.eval()

    catcher = AttentionCatcher(model)

    dataset = MultiViewCompoundDataset(
        csv_path=args.image_csv,
        petrel_conf_path=args.petrel_conf,
        global_crop_size=224,
        resize_short_side=1040, 
        use_center_crop=False    
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Found {len(dataset)} images.")

    for i, batch in enumerate(dataloader):
        imgs, smiles_tuple, meta = batch
        smiles_raw = smiles_tuple[0]
        safe_smiles = "".join([c if c.isalnum() else "_" for c in smiles_raw])[:20]
        print(f"\nProcessing Img {i+1}/{len(dataset)}: {safe_smiles}")

        imgs_input = imgs.repeat(2, 1, 1, 1).to(device).float()
        smiles_input = list(smiles_tuple) * 2 
        
        catcher.clear()
        with torch.no_grad():
            _ = model(imgs_input, smiles_input, assay_embeds)
        
        attn_weights = catcher.get_last_weights() 
        if attn_weights is None: continue

        single_sample_attn = attn_weights[0] 

        compound_vis_data = [] 
        compound_assay_ids = []

        pbar = tqdm(target_indices, desc="Generating Maps", leave=False)
        for (assay_idx, assay_id_str) in pbar:
            specific_attn_vec = single_sample_attn[assay_idx, :]
            
            rgb_img, att_mat = prepare_vis_data(imgs[0], specific_attn_vec)
            
            save_name_single = f"img{i}_{safe_smiles}_assay_{assay_id_str}.svg"
            plot_single_row(
                os.path.join(args.output_dir, save_name_single),
                rgb_img,
                att_mat,
                assay_id_str,
                smiles_raw
            )
            
            compound_vis_data.append((rgb_img, att_mat))
            compound_assay_ids.append(assay_id_str)

        if len(compound_vis_data) > 0:
            if args.assay_id is None or args.assay_id.strip() == "":
                save_name_combined = f"img{i}_{safe_smiles}_ALL_ASSAYS_COMBINED.svg"
                plot_combined_rows(
                    os.path.join(args.output_dir, save_name_combined),
                    compound_vis_data,
                    compound_assay_ids,
                    smiles_raw
                )

    print("Done.")

if __name__ == "__main__":
    main()