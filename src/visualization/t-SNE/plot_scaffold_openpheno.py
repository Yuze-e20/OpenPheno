import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import List, Dict

from pretrain.dataset import _open_petrel, read_channels, FiveChannelAugmentation
from pretrain.jointmodel import JointModel


CONFIG = {
    "scaffold_bg_csv": "./data/rawdata/clustered_compounds.csv", 
    "scaffold_fg_csv": "./data/rawdata/scaffold_ten.csv",
    
    "image_csv": "./data/Broad-270/compound_splits/all.csv",
    "ckpt_path": "path/to/downstream/ckpt",
    "molstm_ckpt": "./MoleculeSTM/molecule_model.pth",
    "molstm_vocab": "./MoleculeSTM/bart_vocab.txt",
    "petrel_conf": '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/petreloss.conf',
    
    "batch_size": 32,
    "num_workers": 4,
    "img_size": 224,
    "resize_short_side": 780,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_plot": "./visualization/t-SNE/scaffold_openpheno"
}


class InferenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, petrel_conf=None, img_size=224, resize_short_side=780):
        self.data = dataframe.reset_index(drop=True)
        self.petrel = _open_petrel(petrel_conf)
        self.img_size = img_size
        
        self.aug = FiveChannelAugmentation(
            global_size=img_size, 
            resize_short_side=resize_short_side
        )
        
        self.channel_cols = ["ch1_path", "ch2_path", "ch3_path", "ch4_path", "ch5_path"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        paths = [row[c] for c in self.channel_cols]
        
        try:
            img_arr = read_channels(paths, self.petrel)
        except Exception as e:
            print(f"Error reading path for index {idx}: {e}")
            img_arr = np.zeros((5, self.img_size, self.img_size), dtype=np.float32)

        img_tensor = self.aug.preprocess_only(img_arr)
        
        _, h, w = img_tensor.shape
        dy = (h - self.img_size) // 2
        dx = (w - self.img_size) // 2
        
        if h >= self.img_size and w >= self.img_size:
            crop = img_tensor[:, dy:dy+self.img_size, dx:dx+self.img_size]
        else:
            import torch.nn.functional as F
            crop = F.interpolate(img_tensor.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear').squeeze(0)
        
        label = row['cluster_id'] 
        smiles = str(row['smiles'])
        
        source = row['data_source']
        
        return crop, label, smiles, source


def prepare_data(bg_csv, fg_csv, image_csv):
    print(f"Loading image paths from {image_csv}...")
    df_images = pd.read_csv(image_csv)
    df_images.columns = [c.strip() for c in df_images.columns]
    
    if 'Compound SMILES' in df_images.columns:
        df_images.rename(columns={'Compound SMILES': 'smiles'}, inplace=True)
    elif 'smiles' not in df_images.columns:
        raise KeyError(f"Cannot find SMILES column in image csv. Available columns: {df_images.columns}")
    
    df_images['smiles'] = df_images['smiles'].astype(str).str.strip()

    print(f"Loading BACKGROUND scaffold data from {bg_csv}...")
    df_bg = pd.read_csv(bg_csv)
    df_bg['smiles'] = df_bg['smiles'].astype(str).str.strip()
    
    df_bg_merged = pd.merge(df_bg, df_images, on='smiles', how='inner')
    df_bg_merged['data_source'] = 'background' 
    print(f"Background samples merged: {len(df_bg_merged)}")

    print(f"Loading FOREGROUND scaffold data from {fg_csv}...")
    df_fg = pd.read_csv(fg_csv)
    df_fg['smiles'] = df_fg['smiles'].astype(str).str.strip()
    
    df_fg_merged = pd.merge(df_fg, df_images, on='smiles', how='inner')
    df_fg_merged['data_source'] = 'foreground' 
    print(f"Foreground samples merged: {len(df_fg_merged)}")

    df_total = pd.concat([df_bg_merged, df_fg_merged], ignore_index=True)
    
    if len(df_total) == 0:
        raise ValueError("Merged dataframe is empty! Please check SMILES matching.")

    return df_total


def main():
    df_data = prepare_data(CONFIG['scaffold_bg_csv'], CONFIG['scaffold_fg_csv'], CONFIG['image_csv'])
    
    dataset = InferenceDataset(
        df_data, 
        petrel_conf=CONFIG['petrel_conf'],
        img_size=CONFIG['img_size'],
        resize_short_side=CONFIG['resize_short_side']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print("Initializing model...")
    model = JointModel(
        in_channels=5,
        pretrained=False,
        img_size=CONFIG['img_size'],
        molstm_ckpt=CONFIG['molstm_ckpt'],   
        molstm_vocab=CONFIG['molstm_vocab'], 
    )
    
    print(f"Loading checkpoint from {CONFIG['ckpt_path']}...")
    checkpoint = torch.load(CONFIG['ckpt_path'], map_location='cpu')
    
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded. Missing keys: {len(msg.missing_keys)}")
    
    model.to(CONFIG['device'])
    model.eval()
    
    features_list = []
    labels_list = []
    sources_list = [] 
    
    print("Extracting features...")
    with torch.no_grad():
        for imgs, labels, _, sources in tqdm(dataloader, desc="Inference"): 
            imgs = imgs.to(CONFIG['device'])
            
            feats = model.vision.clip_forward(imgs)
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy()) 
            sources_list.extend(sources)       
            
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    sources = np.array(sources_list)
    
    print(f"Total Feature shape: {features.shape}")
    
    print("Running t-SNE on combined data...")
    n_samples = features.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=perp)
    features_2d = tsne.fit_transform(features)
    
    plot_df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'Cluster ID': labels,
        'Source': sources
    })
    plot_df['Cluster ID'] = plot_df['Cluster ID'].astype(str)
    
    df_bg_plot = plot_df[plot_df['Source'] == 'background']
    df_fg_plot = plot_df[plot_df['Source'] == 'foreground']
    
    print(f"Plotting: {len(df_bg_plot)} background points, {len(df_fg_plot)} foreground points.")

    save_dir = CONFIG['output_plot']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Plotting combined map...")
    plt.figure(figsize=(12, 10))
    
    plt.scatter(
        df_bg_plot['x'], 
        df_bg_plot['y'], 
        c='#E0E0E0', 
        alpha=0.3, 
        s=40, 
        label='Background',
        zorder=0 
    )
    
    sns.scatterplot(
        data=df_fg_plot, 
        x='x', 
        y='y', 
        hue='Cluster ID', 
        palette='tab10', 
        s=60, 
        alpha=0.9,
        zorder=1
    )
    
    plt.title(f"t-SNE visualization (Foreground vs Background)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    
    if len(df_fg_plot['Cluster ID'].unique()) > 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster ID", ncol=2)
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster ID")
        
    plt.tight_layout()
    
    combined_save_path = os.path.join(save_dir, "combined_tsne.png")
    plt.savefig(combined_save_path, dpi=300)
    print(f"Combined plot saved to {combined_save_path}")
    plt.close() 

    x_min, x_max = plot_df['x'].min(), plot_df['x'].max()
    y_min, y_max = plot_df['y'].min(), plot_df['y'].max()
    
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    
    xlims = (x_min - margin_x, x_max + margin_x)
    ylims = (y_min - margin_y, y_max + margin_y)

    unique_fg_clusters = np.unique(df_fg_plot['Cluster ID'])
    
    print(f"Generating individual plots for {len(unique_fg_clusters)} foreground clusters...")
    
    for cls_id in tqdm(unique_fg_clusters, desc="Saving individual plots"):
        plt.figure(figsize=(12, 10))
        
        plt.scatter(
            df_bg_plot['x'], 
            df_bg_plot['y'], 
            c='#E0E0E0', 
            alpha=0.3, 
            s=40, 
            zorder=0
        )
        
        subset = df_fg_plot[df_fg_plot['Cluster ID'] == str(cls_id)]
        
        sns.scatterplot(
            data=subset, 
            x='x', 
            y='y', 
            s=60, 
            alpha=0.9,
            color='tab:blue', 
            zorder=1
        )
        
        plt.xlim(xlims)
        plt.ylim(ylims)
        
        plt.title(f"Cluster {cls_id} (Foreground) on Global Background")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        
        individual_save_path = os.path.join(save_dir, f"cluster_{cls_id}.png")
        plt.tight_layout()
        plt.savefig(individual_save_path, dpi=300)
        plt.close() 

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()