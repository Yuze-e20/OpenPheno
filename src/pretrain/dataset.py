import io
import csv
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
import random
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
    def __init__(self, global_size=224, local_size=96, locals_per_global=4, resize_short_side=1040):
        self.global_size = global_size
        self.local_size = local_size
        self.locals_per_global = locals_per_global
        self.resize_short_side = resize_short_side

    @staticmethod
    def _per_channel_minmax(arr: np.ndarray) -> np.ndarray:
        eps = 1e-6
        cmin = np.percentile(arr, 0.1, axis=(1, 2), keepdims=True).astype(np.float32)
        cmax = np.percentile(arr, 99.9, axis=(1, 2), keepdims=True).astype(np.float32)
        
        arr = np.clip(arr, cmin, cmax)
        
        denom = cmax - cmin + eps
        arr -= cmin
        arr /= denom
        return arr
    
    @staticmethod
    def _resize_short_side_to_tensor(arr: np.ndarray, short_side: int) -> torch.Tensor:
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
    def _random_square_crop_tensor(t: torch.Tensor, crop_size: int) -> torch.Tensor:
        C, H, W = t.shape
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        return t[:, top:top+crop_size, left:left+crop_size]

    def preprocess_only(self, img_array: np.ndarray) -> torch.Tensor:
        arr = self._per_channel_minmax(img_array)
        t_arr = self._resize_short_side_to_tensor(arr, self.resize_short_side)
        return t_arr

    def crops_only(self, t_arr: torch.Tensor):
        global_crop = self._random_square_crop_tensor(t_arr, crop_size=self.global_size)
        
        local_crops = []
        for _ in range(self.locals_per_global):
            lc = self._random_square_crop_tensor(t_arr, crop_size=self.local_size)
            local_crops.append(lc)
            
        return global_crop, local_crops

    def global_only(self, t_arr: torch.Tensor):
        global_crop = self._random_square_crop_tensor(t_arr, crop_size=self.global_size)
        return global_crop

    def __call__(self, img_array: np.ndarray):
        t_arr = self.preprocess_only(img_array)
        return self.crops_only(t_arr)

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
        control_json_path: Optional[str] = None,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        locals_per_global: int = 4,
        resize_short_side: int = 780,
        petrel_conf_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.records: List[SampleRecord] = []
        
        self.global_crop_size = int(global_crop_size)
        self.local_crop_size = int(local_crop_size)
        self.locals_per_global = int(locals_per_global)
        self.resize_short_side = int(resize_short_side)
        
        
        self.petrel = _open_petrel(petrel_conf_path)
        self.control_channel_order = ["Hoechst", "ERSyto", "ERSytoBleed", "Ph_golgi", "Mito"]
        
        self.augmentation = FiveChannelAugmentation(
            global_size=global_crop_size,
            local_size=local_crop_size,
            locals_per_global=locals_per_global,
            resize_short_side=resize_short_side
        )
        
        with open(csv_path, "r", newline="", encoding="utf-8", ) as f:
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
            smiles = r.get(smiles_col).strip()
            paths = []
            for c in channels_cols:
                p = r.get(c, "")
                paths.append(p)
            
            plate, well = _parse_plate_well(r)
            
            self.records.append(
                SampleRecord(smiles=smiles, channel_paths=paths, plate_id=str(plate), well=str(well))
            )
        self.unique_smiles = sorted({rec.smiles for rec in self.records})
        
        self.smiles_to_records_by_plate: Dict[str, Dict[str, List[SampleRecord]]] = {}
        
        for rec in self.records:
            d = self.smiles_to_records_by_plate.setdefault(rec.smiles, {})
            d.setdefault(rec.plate_id, []).append(rec)
        
        self.control_map: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.plate_to_control_wells: Dict[str, List[str]] = {}
        
        with open(control_json_path, "r", encoding="utf-8") as f:
            self.control_map = json.load(f)
        for plate, wells in self.control_map.items():
            avail = [w.upper() for w in wells.keys()]
            self.plate_to_control_wells[plate] = avail
        
        self._n_raw = len(self.records)

        print(f"   - samples: {len(self.records)}")
        print(f"   - unique compound: {len(self.unique_smiles)}")
        print(f"   - Resize: {self.resize_short_side}")
        print(f"   - global: {self.global_crop_size}×{self.global_crop_size}")
        print(f"   - local: {self.local_crop_size}×{self.local_crop_size}")
        print(f"   - local_num: {self.locals_per_global}")
        print(f"   - global_num: {self.locals_per_global * 2}")
    
    def __len__(self):
        return self._n_raw
    
    def _pick_random_control_paths(self, plate_id: str) -> Tuple[str, List[str]]:
        wells = self.plate_to_control_wells.get(plate_id, [])
        
        control_well = random.choice(wells)
        chan_map = self.control_map.get(plate_id, {}).get(control_well, {})
        ctrl_paths = []
        for ch_name in self.control_channel_order:
            p = chan_map.get(ch_name, "")
            ctrl_paths.append(p)
        
        return control_well, ctrl_paths
    
    def _pick_another_treated_record(self, rec: SampleRecord) -> Optional[SampleRecord]:
        by_plate = self.smiles_to_records_by_plate.get(rec.smiles, {}) # plate -> rec
        if len(by_plate) <= 1:
            return None
        other_plates = [p for p in by_plate.keys() if p != rec.plate_id]
        plate_sel = random.choice(other_plates)
        cand_list = by_plate.get(plate_sel, [])
        return random.choice(cand_list)
    
    def _load_arr_for_record(self, rec: SampleRecord) -> np.ndarray:
        return read_channels(rec.channel_paths, self.petrel)
    
    def __getitem__(self, idx: int):
        rec = self.records[idx]
        treated_src = self._load_arr_for_record(rec)
        treated_tensor = self.augmentation.preprocess_only(treated_src)
        treated_global1, locals1 = self.augmentation.crops_only(treated_tensor)
        another_rec = self._pick_another_treated_record(rec)
        
        if another_rec is None:
            another_fallback = True
            another_meta = {
                "another_plate_id": rec.plate_id,
                "another_well": rec.well,
                "another_paths": rec.channel_paths,
            }
            treated_global2, locals2 = self.augmentation.crops_only(treated_tensor)
            
        else:
            another_fallback = False
            another_src = self._load_arr_for_record(another_rec)
            another_tensor = self.augmentation.preprocess_only(another_src)
            
            another_meta = {
                "another_plate_id": another_rec.plate_id,
                "another_well": another_rec.well,
                "another_paths": another_rec.channel_paths,
            }
            treated_global2, locals2 = self.augmentation.crops_only(another_tensor)
        
        control_well, control_paths = self._pick_random_control_paths(rec.plate_id)
        control_src = read_channels(control_paths, self.petrel)
            
        control_tensor = self.augmentation.preprocess_only(control_src)
        control_global = self.augmentation.global_only(control_tensor)
        
        all_locals = locals1 + locals2
        treated_locals = torch.stack(all_locals, dim=0)
        smiles = rec.smiles
        
        meta: Dict[str, Any] = {
            "smiles": rec.smiles,
            "plate_id": rec.plate_id,
            "well": rec.well,
            "treated_paths": rec.channel_paths,
            "control_well": control_well,
            "control_paths": control_paths,
            "another_is_fallback": another_fallback,
        }
        meta.update(another_meta)
        
        return {
            "treated_global1": treated_global1,       
            "treated_global2": treated_global2,       
            "treated_locals": treated_locals,         
            "control_globals": control_global,        
            "smiles": rec.smiles,                    
            "meta": meta                              
        }
