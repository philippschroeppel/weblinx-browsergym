import json
from pathlib import Path
import os

import pandas as pd

os.environ['BROWSERGYM_WEBLINX_REGISTER_TRAIN'] = "False"  
os.environ['BROWSERGYM_WEBLINX_REGISTER_VALID'] = "False"
os.environ['BROWSERGYM_WEBLINX_REGISTER_TEST'] = "False"

import weblinx_browsergym

split_to_browsergym_split = {
    "train": "train",
    "valid": "valid",
    "test_iid": "test",
    "test_vis": "test_vis",
    "test_geo": "test_geo",
    "test_web": "test_web",
    "test_cat": "test_cat",
}

metadata_path = './metadata.json'
save_base_dir = './bg_wl_data'

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

dfs = []

for split, metadata_split in metadata.items():
    if split not in split_to_browsergym_split:
        print(f"Skipping split {split}")
        continue

    output_csv = []
    acceptable_tasks = set(weblinx_browsergym.list_tasks(split=split))
    for demo_id, demo_dict in metadata_split.items():
        for step_num, step_dict in demo_dict.items():
            # task_name,miniwob_category,comment,webgum_subset,browsergym_split
            task_name = f"weblinx.{demo_id}.{step_num}"

            if task_name not in acceptable_tasks:
                continue

            output_csv.append({
                "task_name": task_name,
                'demo_name': demo_id,
                'step': step_num,
                "split": split,
                "browsergym_split": split_to_browsergym_split[split],
            })

    print(f"{split}: Found {len(output_csv)} tasks")
    df = pd.DataFrame(output_csv)
    dfs.append(df)

    save_dir = Path(save_base_dir, f'browsergym/task_metadata/weblinx_{split}.csv')
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_dir, index=False)

# combine all splits
df = pd.concat(dfs)
save_dir = Path(save_base_dir, 'browsergym/task_metadata/weblinx_all.csv')
df.to_csv(save_dir, index=False)
print(f"Saved combined metadata to {save_dir}")