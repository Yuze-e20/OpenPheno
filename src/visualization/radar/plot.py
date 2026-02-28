import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi
import os
import warnings
from pathlib import Path

ROOT = Path(os.getcwd()).resolve()
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

warnings.filterwarnings("ignore", message=".*font.*", category=UserWarning)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Roboto', 'Arial', 'DejaVu Sans']

COLORS = {
    'setting2': '#A4C2E2',  
    
    'setting3': '#EA9383',  

    'random': '#BDBDBD',    
 
    'fill_alpha': 0.25,
    
    'bg_color': '#f8f9fa',
    
    'grid_color': '#999999'
}

def process_data(csv_path, min_samples=0):
    if not os.path.exists(csv_path):
        print(f"Warning: File not found {csv_path}")
        return None

    df = pd.read_csv(csv_path)
     
    df['type_clean'] = df['ASSAY_TYPE'].str.lower().str.strip()
  
    df.loc[df['type_clean'].isin(['fungal', 'yeast']), 'type_clean'] = 'fungal & yeast'
    
    counts = df['type_clean'].value_counts()
    
    valid_types = counts[counts >= min_samples].index.tolist()
    df.loc[~df['type_clean'].isin(valid_types), 'type_clean'] = 'others'
    
    grouped = df.groupby('type_clean')['auroc'].mean().reset_index()

    def format_type(x):

        if x == 'fungal & yeast':
            return 'Fungal & Yeast'
        
        x = x.title()
        x = x.replace("Bacterial", "Bacterial")
        x = x.replace("Biochem", "Biochemical")
        return x

    grouped['type_display'] = grouped['type_clean'].apply(format_type)
    
    return grouped

def align_data(df_s2, df_s3):
    all_categories = set(df_s2['type_display']).union(set(df_s3['type_display']))
    
    priority_order = ['Cell', 'Biochemical', 'Bacterial', 'Fungal & Yeast', 'Others']
    
    sorted_cats = []
    for cat in priority_order:
        if cat in all_categories:
            sorted_cats.append(cat)
    for cat in sorted(list(all_categories)):
        if cat not in sorted_cats:
            sorted_cats.append(cat)
            
    values_s2 = []
    values_s3 = []
    
    for cat in sorted_cats:
        row2 = df_s2[df_s2['type_display'] == cat]
        val2 = row2['auroc'].values[0] if not row2.empty else 0.5 
        values_s2.append(val2)
 
        row3 = df_s3[df_s3['type_display'] == cat]
        val3 = row3['auroc'].values[0] if not row3.empty else 0.5
        values_s3.append(val3)
        
    return sorted_cats, values_s2, values_s3

def plot_radar(categories, values_s2, values_s3, output_path):
    
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] 
    
    values_s2 += values_s2[:1]
    values_s3 += values_s3[:1]
    baseline = [0.5] * (N + 1) 
    
    fig = plt.figure(figsize=(9, 9)) 
    fig.patch.set_facecolor('white')
    
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(COLORS['bg_color']) 
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1) 
    
    plt.xticks(angles[:-1], categories, size=12, color='#333333') 
    
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_rlabel_position(0)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9], ["0.5", "0.6", "0.7", "0.8", "0.9"], 
               color="#666666", size=11)
    plt.ylim(0.45, 0.95)
    
    ax.plot(angles, baseline, linewidth=1.5, linestyle='--', color=COLORS['random'], zorder=1)
    
    ax.plot(angles, values_s2, linewidth=2.5, linestyle='-', color=COLORS['setting2'], 
            marker='o', markersize=9, markeredgecolor='white', markeredgewidth=1.5,
            label='Known Compound, Novel Assay', zorder=3)
    ax.fill(angles, values_s2, color=COLORS['setting2'], alpha=COLORS['fill_alpha'])
    
    ax.plot(angles, values_s3, linewidth=2.5, linestyle='-', color=COLORS['setting3'], 
            marker='o', markersize=9, markeredgecolor='white', markeredgewidth=1.5,
            label='Novel Compound, Novel Assay', zorder=4)
    ax.fill(angles, values_s3, color=COLORS['setting3'], alpha=COLORS['fill_alpha'])
    
    ax.spines['polar'].set_visible(False)
    ax.grid(True, color=COLORS['grid_color'], linestyle='--', alpha=0.4)
    
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=False)
    
    plt.tight_layout()
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file).replace('.png', '.svg'), bbox_inches='tight')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"Saved radar chart to {output_path}")

def main():
    BASE_PATH = ""
    
    SETTING2_CSV = os.path.join(BASE_PATH, "./visualization/radar/meta_setting2.csv") 
    SETTING3_CSV = os.path.join(BASE_PATH, "./visualization/radar/meta_setting3.csv") 
    
    OUTPUT_PLOT = "./visulization/radar/radar_chart.png"

    print("Processing Setting 2 Data...")
    df_s2_grouped = process_data(SETTING2_CSV)
    
    print("Processing Setting 3 Data...")
    df_s3_grouped = process_data(SETTING3_CSV)
    
    if df_s2_grouped is None or df_s3_grouped is None:
        return

    print("Aligning Data...")
    cats, vals_s2, vals_s3 = align_data(df_s2_grouped, df_s3_grouped)
    
    print(f"Categories: {cats}")
    print(f"Setting 2 Values: {vals_s2}")
    print(f"Setting 3 Values: {vals_s3}")
    
    print("Plotting...")
    plot_radar(cats, vals_s2, vals_s3, OUTPUT_PLOT)

if __name__ == "__main__":
    main()