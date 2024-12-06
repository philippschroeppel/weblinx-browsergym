"""
The purpose of this script is to create a directory that contains
the demonstrations for only the files necessary to run agentlab on
the browsergym-weblinx environment. This is done to reduce the size
of the data that needs to be uploaded to huggingface hub.
"""

import os
from pathlib import Path
import json
import zipfile
import shutil
from copy import deepcopy


from tqdm.auto import tqdm

def verify_overlap_with_other_bboxes(bbox: dict, bboxes_list: list, iou_threshold=0.9):
    """
    Verify if a bounding box overlaps with other bounding boxes.

    Args:
        bbox: dict
            Bounding box dictionary.
        bboxes_list: list
            List of bounding box dictionaries.
        iou_threshold: float
            Intersection over union threshold for overlapping bounding boxes.

    Returns:
        bool: True if the bounding box overlaps with other bounding boxes, False otherwise.
    """
    for other_bbox in bboxes_list:
        x1, y1, w1, h1 = bbox
        x2, y2, w2, h2 = other_bbox

        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = w1 * h1
        boxBArea = w2 * h2

        iou = interArea / float(boxAArea + boxBArea - interArea)

        if iou > iou_threshold:
            return True

    return False

def sort_uid_by_area(extra_props: dict):
    """
    Sort the unique IDs in extra properties by area of the bounding box.

    Args:
        extra_props: dict
            Extra properties dictionary.

    Returns:
        list: List of unique IDs sorted by area of the bounding box.
    """
    uid_to_area = {}

    for uid, props in extra_props.items():
        if props["bbox"] is None:
            uid_to_area[uid] = 0
        else:
            x, y, w, h = props["bbox"]
            uid_to_area[uid] = w * h
    
    # sort ascending
    sorted_uids = list(sorted(uid_to_area, key=uid_to_area.get))
    return sorted_uids
    
def remove_overlapping_bboxes(extra_props: dict, iou_threshold=0.9, inplace=True):
    """
    Remove overlapping bounding boxes in extra properties.

    Args:
        extra_props: dict
            Extra properties dictionary.
        iou_threshold: float
            Intersection over union threshold for overlapping bounding boxes.

    Returns:
        dict: Extra properties dictionary with overlapping bounding boxes removed.
    """
    if not inplace:
        extra_props = deepcopy(extra_props)

    uids_sorted = sort_uid_by_area(extra_props)

    other_bboxes = []

    for uid in uids_sorted:
        props = extra_props[uid]

        original_som = props['set_of_marks']
        if original_som == 0:
            continue
    
        if props["bbox"] is None:
            continue

        is_overlapping = verify_overlap_with_other_bboxes(props["bbox"], other_bboxes, iou_threshold)
        if is_overlapping:
            props['set_of_marks'] = 0
        
        other_bboxes.append(props["bbox"])
        
    return extra_props

def preprocess_extra_props(extra_props, min_area=20, max_area=1_000_000):
    """
    This function ensures that extra properties will have set_of_marks and clickable properties set
    to 1 if they are not present in the extra properties.
    """
    for key in extra_props:
        if extra_props[key]["bbox"] is None or len(extra_props[key]["bbox"]) != 4:
            extra_props[key]["set_of_marks"] = 0
            extra_props[key]["clickable"] = 0
            continue
        
        if extra_props[key]["set_of_marks"] == 1:
            v = extra_props[key]["set_of_marks"]
            x, y, width, height = extra_props[key]["bbox"]
            # if area is less than 20, we set set_of_marks to 0, otherwise 1
            is_within_area = min_area <= width * height <= max_area
            extra_props[key]["set_of_marks"] = v if is_within_area else 0

    return extra_props

UPDATE_EXTRA_PROPS = True
override_existing = True
splits = [
    "train",
    "valid",
    "test_iid",
    "test_vis",
    "test_geo",
    "test_web",
    "test_cat",
]
dataset_dir = "wl_data"
output_dir = "bg_wl_data"
metadata_path = "./metadata.json"

dataset_dir = Path(dataset_dir)
output_dir = Path(output_dir)
demos_dir = dataset_dir / "demonstrations"
demos_output_dir = output_dir / "demonstrations"
demos_zip_dir = output_dir / "demonstrations_zip"

# Create the output directory
demos_output_dir.mkdir(exist_ok=True, parents=True)
demos_zip_dir.mkdir(exist_ok=True, parents=True)

# first, copy ./metadata.json to the output directory
shutil.copy(metadata_path, output_dir)
print(f"Copied {metadata_path} to {output_dir}")

# get the list of demo ids from metadata.json
with open(metadata_path, "r") as f:
    metadata = json.load(f)

for split in splits:
    demo_ids = list(metadata[split].keys())

    # compute number of demo steps in the test path
    num_demo_steps = sum(len(demo_dict) for demo_dict in metadata[split].values())
    print(f"Number of demo steps: {num_demo_steps}")

    # for each demo id, find it in demonstrations directory and copy replay.json, metadata.json,
    # form.json, to the output directory
    for demo_id in tqdm(demo_ids, desc="Copying replay.json, form.json, metadata.json"):
        demo_id = str(demo_id)
        replay_path = demos_dir / demo_id / "replay.json"
        metadata_path = demos_dir / demo_id / "metadata.json"
        form_path = demos_dir / demo_id / "form.json"

        demo_out_dir = output_dir / "demonstrations" / demo_id
        demo_out_dir.mkdir(parents=True, exist_ok=True)

        replay_path_output = demo_out_dir / "replay.json"
        meta_path_output = demo_out_dir / "metadata.json"
        form_path_output = demo_out_dir / "form.json"

        if not replay_path_output.exists():
            shutil.copy(replay_path, replay_path_output)
        if not meta_path_output.exists():
            shutil.copy(metadata_path, meta_path_output)
        if not form_path_output.exists():
            shutil.copy(form_path, form_path_output)

    for demo_id in tqdm(demo_ids, desc="Copying step files"):
        demo_dict = metadata[split][demo_id]

        for step_id, step_dict in demo_dict.items():
            step_num = int(step_id)

            if not step_dict["is_task"] or not step_dict["has_full_snapshot"]:
                continue

            bbox_path = demos_dir / step_dict["bbox_path"]
            dom_object_path = demos_dir / step_dict["dom_object_path"]
            axtree_path = demos_dir / step_dict["axtree_path"]
            extra_props_path = demos_dir / step_dict["extra_props_path"]
            screenshot_path = demos_dir / step_dict["screenshot_path"]
            html_path = demos_dir / step_dict["html_path"]

            extra_props_was_copied = False

            # for each of those files, copy them to the output directory unless they already exist
            for path in [
                bbox_path,
                dom_object_path,
                axtree_path,
                extra_props_path,
                screenshot_path,
                html_path,
            ]:
                if path is None:
                    continue

                path = Path(path)

                if not path.exists():
                    tqdm.write(f"File {path} does not exist")
                    continue

                path_out = output_dir / path.relative_to(dataset_dir)

                if not path_out.exists() or override_existing:
                    path_out.parent.mkdir(parents=True, exist_ok=True)
                    tqdm.write(f"Copying {path} to {path_out}")
                    shutil.copy(path, path_out)
                    extra_props_was_copied = True

            # for extra_props_path, we need to preprocess the file to remove overlapping bounding boxes
            if extra_props_was_copied and UPDATE_EXTRA_PROPS:
                tqdm.write(f"Processing {extra_props_path}")
                extra_props_path_out = output_dir / extra_props_path.relative_to(
                    dataset_dir
                )
                with open(extra_props_path_out, "r") as f:
                    extra_properties = json.load(f)

                extra_properties = preprocess_extra_props(
                    extra_properties, min_area=50, max_area=500_000
                )
                remove_overlapping_bboxes(
                    extra_properties, inplace=True, iou_threshold=0.75
                )

                with open(extra_props_path_out, "w") as f:
                    json.dump(extra_properties, f)

        # if the zip file doesn't exist, create it
        tqdm.write("Zipping the output directory...")
        zip_filename = demos_zip_dir / f"{demo_id}.zip"
        demo_dir = demos_output_dir / demo_id

        if not zip_filename.exists() or override_existing:
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the directory and add all files to the zip
                for root, dirs, files in os.walk(demo_dir):
                    for file in files:
                        file_path = Path(root) / file
                        # Add file to the zip, but keep the folder structure
                        zipf.write(file_path, file_path.relative_to(demo_dir))

# print number of files in the output directory (recursively)
num_files = sum(1 for _ in output_dir.rglob("*"))
print(f"Number of files in output directory: {num_files}")
