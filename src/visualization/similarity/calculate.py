import json
import pandas as pd
import torch
import numpy as np
import os
import warnings
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(os.getcwd()).resolve()
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

warnings.filterwarnings("ignore", message=".*font.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache.*", category=UserWarning)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Roboto', 'Arial', 'DejaVu Sans']


STYLE_COLORS = {
    'coral': '#EA9383',      
    'light_blue': '#A4C2E2', 
    'lavender': '#CEC8E0',   
    'yeast': '#E9DCBA',      
    'fungal': '#BDCCCB',    
}


TYPE_COLOR_MAP = {
    'bacterial': STYLE_COLORS['coral'],
    'cell': STYLE_COLORS['light_blue'],
    'biochem': STYLE_COLORS['lavender'],
    'yeast': STYLE_COLORS['yeast'],
    'fungal': STYLE_COLORS['fungal'],
    'other': STYLE_COLORS['yeast'],
}

def load_and_encode_assays(
    json_path: str, 
    label_csv_path: str, 
    model_name: str = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/Biolord',
    strict_check: bool = False
) -> Tuple[torch.Tensor, List[str]]:
    
    print(f"[Step] Reading Label CSV header from: {label_csv_path}")
    df_cols = pd.read_csv(label_csv_path, nrows=0).columns.tolist()
    task_ids = [str(c) for c in df_cols[1:]]

    print(f"[Step] Reading Assay JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        assay_data = json.load(f)
    
    id_to_summary = {}
    for item in assay_data:
        aid = str(item.get("id", ""))
        summary = item.get("summary", "")
        if aid:
            id_to_summary[aid] = summary
            
    aligned_summaries = []
    default_summary = "Unknown biological assay."

    for tid in task_ids:
        if tid in id_to_summary:
            aligned_summaries.append(id_to_summary[tid])
        else:
            aligned_summaries.append(default_summary)

    print(f"[Step] Encoding {len(aligned_summaries)} summaries...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(aligned_summaries, convert_to_tensor=True, show_progress_bar=True)
    
    return embeddings.cpu(), task_ids

def main():
    BASE_PATH = ""
    JSON_PATH = os.path.join(BASE_PATH, "data/Broad-270/assay_meta/assay_description.json")
    TRAIN_CSV_PATH = os.path.join(BASE_PATH, "data/Broad-270/assay_splits/fewshot/label_train.csv")
    TEST_CSV_PATH = os.path.join(BASE_PATH, "data/Broad-270/assay_splits/fewshot/label_test.csv")
    META_CSV_PATH = os.path.join(BASE_PATH, "visualization/similarity/meta_setting3.csv") 
    
    OUTPUT_DIR = Path(BASE_PATH) / "src/visualization/similarity"
    OUTPUT_CSV_PATH = OUTPUT_DIR / "setting3.csv"
    OUTPUT_PLOT_PATH = OUTPUT_DIR / "setting3.png"
    OUTPUT_PLOT_PDF = OUTPUT_DIR / "setting3.svg"

    for p in [JSON_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH, META_CSV_PATH]:
        if not os.path.exists(p):
            print(f"Error: File not found: {p}")
            return

    print("\n>>> Processing TRAIN set...")
    train_embeddings, train_ids = load_and_encode_assays(JSON_PATH, TRAIN_CSV_PATH)

    print("\n>>> Processing TEST set...")
    test_embeddings, test_ids = load_and_encode_assays(JSON_PATH, TEST_CSV_PATH)

    print("\n>>> Calculating Similarity...")
    sim_matrix = util.cos_sim(test_embeddings, train_embeddings)
    
    results = []
    for i, test_id in enumerate(test_ids):
        similarities = sim_matrix[i]
        max_sim_score, max_idx = torch.max(similarities, dim=0)
        best_train_id = train_ids[max_idx.item()]
        results.append({
            "assay_id": test_id,
            "max_similarity": max_sim_score.item(),
            "nearest_train_neighbor": best_train_id
        })
    
    df_sim = pd.DataFrame(results)

    print(f"\n>>> Merging Metadata...")
    df_meta = pd.read_csv(META_CSV_PATH, usecols=['assay_id', 'ASSAY_TYPE', 'auroc'])
    df_meta['assay_id'] = df_meta['assay_id'].astype(str)
    df_final = pd.merge(df_sim, df_meta, on='assay_id', how='inner')
    
    df_final['type_key'] = df_final['ASSAY_TYPE'].str.lower()
    df_final.loc[df_final['type_key'].str.contains('fungal|yeast'), 'type_key'] = 'fungal'

    if len(df_final) < 2:
        return

    spearman_r, spearman_p = stats.spearmanr(df_final['max_similarity'], df_final['auroc'])
    print(f"\n>>> Spearman Correlation: r={spearman_r:.4f}, p={spearman_p:.4e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n>>> Generating Plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))  
    fig.patch.set_facecolor('white')
    ax.set_facecolor("#f8f9fa")  
    assay_types = df_final['ASSAY_TYPE'].unique()
    
    for atype in assay_types:
        type_key = atype.lower()
        if 'fungal' in type_key:
            color = TYPE_COLOR_MAP['fungal']
        elif 'yeast' in type_key:
            color = TYPE_COLOR_MAP['yeast']
        elif type_key in TYPE_COLOR_MAP:
            color = TYPE_COLOR_MAP[type_key]
        else:
            color = TYPE_COLOR_MAP['other']

        subset = df_final[df_final['ASSAY_TYPE'] == atype]
        
        ax.scatter(
            subset['max_similarity'], 
            subset['auroc'], 
            c=color, 
            label=atype,
            s=100,           
            alpha=0.8,       
            edgecolors='black',
            linewidth=1.0,      
            zorder=5
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.4)
    ax.spines['left'].set_linewidth(1.4)

    ax.set_xlabel('Max Cosine Similarity', fontsize=16)
    ax.set_ylabel('AUROC', fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=18)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.35, 1.05)
    
    from matplotlib.patches import Patch

    unique_types = sorted(df_final['ASSAY_TYPE'].str.lower().unique())
    assay_type_colors = {
        'bacterial': TYPE_COLOR_MAP['bacterial'],
        'cell': TYPE_COLOR_MAP['cell'],
        'yeast': TYPE_COLOR_MAP['yeast'],
        'fungal': TYPE_COLOR_MAP['fungal'],
        'biochem': TYPE_COLOR_MAP['biochem'],
    }

    legend_elements = [
        Patch(
            facecolor=assay_type_colors.get(atype, TYPE_COLOR_MAP['other']),
            label=atype.capitalize(),
            alpha=0.85,
        )
        for atype in unique_types
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout()

    plt.savefig(OUTPUT_PLOT_PDF, bbox_inches="tight")
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight", dpi=300)
    
    print(f"   Saved plot to: {OUTPUT_PLOT_PATH}")
    print("\n================ PROCESS END ================")

if __name__ == "__main__":
    main()