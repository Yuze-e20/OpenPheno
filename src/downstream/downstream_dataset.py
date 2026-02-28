import io
import csv
import time
import random
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from petrel_client.client import Client
import torch.nn.functional as F

def _open_petrel(conf_path: Optional[str] = None) -> Client:
    return Client(conf_path)

def load_tif_from_petrel(path: str, petrel: Client) -> np.ndarray:
    data = petrel.get(path)
    with io.BytesIO(data) as bio:
        arr = tifffile.imread(bio)
    return arr

def read_channels(paths: List[str], petrel: Client) -> np.ndarray:
    chans = [load_tif_from_petrel(p, petrel) for p in paths]
    arr = np.stack(chans, axis=0)
    return arr.astype(np.float32)

class FiveChannelAugmentation:
    
    def __init__(self, global_size=224, resize_short_side=1040, use_center_crop=False):
        self.global_size = global_size
        self.resize_short_side = resize_short_side
        self.use_center_crop = use_center_crop
    
    @staticmethod
    def _per_channel_minmax(arr: np.ndarray) -> np.ndarray:
        eps = 1e-6
        cmin = np.percentile(arr, 0.1, axis=(1, 2), keepdims=True)
        cmax = np.percentile(arr, 99.9, axis=(1, 2), keepdims=True)
        
        arr = np.clip(arr, cmin, cmax)
        
        denom = cmax - cmin + eps
        arr -= cmin
        arr /= denom
        return arr
    
    @staticmethod
    def _resize_short_side(arr: np.ndarray, short_side: int) -> torch.Tensor:
        C, H, W = arr.shape
        min_hw = min(H, W)
        
        x = torch.from_numpy(arr)
        
        if min_hw == short_side:
            return x
            
        scale = short_side / float(min_hw)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        
        x = x.unsqueeze(0) 
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x.squeeze(0)
    
    @staticmethod
    def _random_square_crop(t: torch.Tensor, crop_size: int) -> torch.Tensor:
        C, H, W = t.shape
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        return t[:, top:top+crop_size, left:left+crop_size]

    @staticmethod
    def _center_square_crop(t: torch.Tensor, crop_size: int) -> torch.Tensor:
        C, H, W = t.shape
        
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        return t[:, top:top+crop_size, left:left+crop_size]
    
    def augment_single_view(self, img_array: np.ndarray, crop_size: int):
        arr = self._per_channel_minmax(img_array.copy())
        
        arr = self._resize_short_side(arr, self.resize_short_side)
        
        if self.use_center_crop:
            arr = self._center_square_crop(arr, crop_size)
        else:
            arr = self._random_square_crop(arr, crop_size)
        
        return arr
    
    def __call__(self, img_array: np.ndarray):
        global_crop = self.augment_single_view(img_array, crop_size=self.global_size)
        
        return global_crop

@dataclass
class SampleRecord:
    smiles: str
    channel_paths: List[str]
    plate_id: str
    well: str

class MultiViewCompoundDataset(Dataset):
    
    def __init__(
        self,
        csv_path: str,
        petrel_conf_path: Optional[str] = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/petreloss.conf',
        global_crop_size: int = 224,
        resize_short_side: int = 780,
        use_center_crop: bool = False,
    ):
        super().__init__()
        
        self.records: List[SampleRecord] = []
        
        self.global_crop_size = int(global_crop_size)
        self.resize_short_side = int(resize_short_side)
        self.use_center_crop = use_center_crop
        
        self.petrel = _open_petrel(petrel_conf_path)
        
        self.augmentation = FiveChannelAugmentation(
            global_size=global_crop_size,
            resize_short_side=resize_short_side,
            use_center_crop=use_center_crop
        )
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
        
        smiles_col = fieldnames[0]
        plate_col = "Plate"
        well_col = "Well"
        
        channels_cols = ["ch1_path", "ch2_path", "ch3_path", "ch4_path", "ch5_path"]

        self.num_csv_channels = len(channels_cols)
        
        def _parse_plate_well(row: Dict[str, str]) -> Tuple[str, str]:
            plate = str(row.get(plate_col, "")).strip()
            well = str(row.get(well_col, "")).strip().upper()
            return plate, well
        
        for r in rows:
            smiles = (r.get(smiles_col) or "").strip()
            paths = []
            for c in channels_cols:
                p = r.get(c, "")
                paths.append(p)
            
            plate, well = _parse_plate_well(r)
            
            self.records.append(
                SampleRecord(smiles=smiles, channel_paths=paths, plate_id=str(plate), well=str(well))
            )
        
        self.unique_smiles = sorted({rec.smiles for rec in self.records})
        
        print(f"   - Number of samples: {len(self.records)}")
        print(f"   - Unique compounds: {len(self.unique_smiles)}")
        print(f"   - Resize short side: {self.resize_short_side}")
        print(f"   - Global size: {self.global_crop_size}x{self.global_crop_size}")
        print(f"   - Crop mode: {'CenterCrop' if self.use_center_crop else 'RandomCrop'}")
        
        self._n_raw = len(self.records)
    
    def __len__(self):
        return self._n_raw
    
    def _load_arr_for_record(self, rec: SampleRecord) -> np.ndarray:
        return read_channels(rec.channel_paths, self.petrel)
    
    def __getitem__(self, idx: int):
        rec = self.records[idx]
        
        treated_src = self._load_arr_for_record(rec)
       
        treated_global = self.augmentation(treated_src)
        
        smiles = rec.smiles
        
        meta: Dict[str, Any] = {
            "smiles": rec.smiles,
            "plate_id": rec.plate_id,
            "well": rec.well,
        }
        return treated_global, smiles, meta

class MultiLabelPairDataset(Dataset):
    def __init__(
        self,
        image_csv_path: str,
        label_csv_path: str,
        petrel_conf_path: str = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/petreloss.conf',
        global_crop_size: int = 224,
        resize_short_side: int = 780,
        use_center_crop: bool = False,
    ):
        super().__init__()
        t_all = time.time()
        t0 = time.time()
        print(f"Resize Short Side: {resize_short_side}, CenterCrop: {use_center_crop}")
        
        self.image_ds = MultiViewCompoundDataset(
            csv_path=image_csv_path,
            global_crop_size=global_crop_size,
            resize_short_side=resize_short_side,
            petrel_conf_path=petrel_conf_path,
            use_center_crop=use_center_crop,
        )
        print(f"[ImageDS] Init done: N={len(self.image_ds)} in {time.time()-t0:.2f}s")

        self.label_names, self.smiles_to_targets = self._load_label_table(label_csv_path)
        self.K = len(self.label_names)

        t1 = time.time()
        keep: List[int] = []
        recs: List[SampleRecord] = getattr(self.image_ds, "records", [])
        itr = tqdm(recs, desc="Build indices from image_csv")
        for i, rec in enumerate(itr):
            smi = rec.smiles
            if smi in self.smiles_to_targets:
                keep.append(i)
        self.keep_indices = keep
        print(f"[Filter] kept {len(self.keep_indices)}/{len(self.image_ds)} in {time.time()-t1:.2f}s")
        print(f"[Dataset] Ready: K={self.K}, final N={len(self.keep_indices)} | total {time.time()-t_all:.2f}s")

    def _load_label_table(self, csv_path: str):
        t0 = time.time()
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        header = rows[0]
        col_names = header[1:]

        smiles_to_targets: Dict[str, np.ndarray] = {}
        for r in rows[1:]:
            
            smiles = r[0]
            vals: List[float] = []
            for v in r[1:]:
                v = (v or "").strip()
                if v == "" or v.lower() == "nan":
                    vals.append(-1.0)
                else:
                    fv = float(v)
                    vals.append(1.0 if fv >= 0.5 else 0.0)
            smiles_to_targets[smiles] = np.array(vals, dtype=np.float32)
        print(f"[Labels] Loaded: {len(smiles_to_targets)} smiles, {len(col_names)} tasks in {time.time()-t0:.2f}s")
        return col_names, smiles_to_targets

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx: int):
        base_idx = self.keep_indices[idx]
        img, smiles, _ = self.image_ds[base_idx]
        arr = self.smiles_to_targets.get(smiles, None)
        targets = torch.from_numpy(arr)
        mask = (targets >= 0).float()
        targets = targets.clamp(min=0.0)
        return img, smiles, targets, mask