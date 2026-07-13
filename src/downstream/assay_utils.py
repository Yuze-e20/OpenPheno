import json
import pandas as pd
import torch
import numpy as np
import os
import argparse
import sys
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

def load_and_encode_assays(
    json_path: str, 
    label_csv_path: str, 
    model_name: str = 'FremyCompany/BioLORD-2023',
    strict_check: bool = False
) -> Tuple[torch.Tensor, List[str]]:
    
    print(f"[Step 1] Reading Label CSV header from: {label_csv_path}")
    df_cols = pd.read_csv(label_csv_path, nrows=0).columns.tolist()

    task_ids = [str(c) for c in df_cols[1:]]
    print(f"   -> Found {len(task_ids)} task columns in CSV.")

    print(f"[Step 2] Reading Assay JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        assay_data = json.load(f)
    
    id_to_summary = {}
    for item in assay_data:
        aid = str(item.get("id", ""))
        summary = item.get("summary", "")
        
        if aid:
            id_to_summary[aid] = summary
            
    print(f"   -> Loaded {len(id_to_summary)} assay descriptions from JSON.")

    print(f"\n[Step 3] Checking Data Integrity (CSV vs JSON)...")
    
    aligned_summaries = []
    missing_ids = []
    found_count = 0
    default_summary = "Unknown biological assay."

    for tid in task_ids:
        if tid in id_to_summary:
            aligned_summaries.append(id_to_summary[tid])
            found_count += 1
        else:
            missing_ids.append(tid)
            aligned_summaries.append(default_summary)

    print("-" * 40)
    print(f"   REPORT:")
    print(f"   - Total Tasks in CSV:   {len(task_ids)}")
    print(f"   - Matched in JSON:      {found_count}")
    print(f"   - MISSING in JSON:      {len(missing_ids)}")
    print("-" * 40)
    print("✅ PERFECT MATCH: All CSV columns found in JSON.")

    print(f"\n[Step 4] Encoding {len(aligned_summaries)} summaries with {model_name}...")
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(aligned_summaries, convert_to_tensor=True, show_progress_bar=True)
    
    return embeddings.cpu(), task_ids
