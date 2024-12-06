"""
This script will generate a metadata.json file that will be used to test the weblinx environment. It contains the following
structure:
{
    <split>: {
        <demo_id>: {
            <step_num>: {
                'type': <'chat' or 'browser'>,
                'screenshot_path': <path>,
                'bbox_path': <path>,
                'num_actions': <int>,
                'action_history': <list>,
                'html_path': <path>,
                'intent': <str>,
                'args': <dict>,
            }
        }
    }
}
"""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import weblinx as wl
import weblinx.processing as wlp
import weblinx.eval


def convert_html_path_to_playwright(html_path: Path, replace_with="axtrees") -> Path:
    # replace the directory "/pages/" with "/axtrees/" and the extension ".html" with ".json"
    # find the name of part that contains "pages" and replace it with "axtrees"
    parts = html_path.parts
    idx = parts.index("pages")
    parts = parts[:idx] + (replace_with,) + parts[idx + 1 :]
    # replace the extension ".html" with ".json"
    return Path(*parts).with_suffix(".json")


wl_data_dir = "wl_data"
uid_key = "data-webtasks-id"


results = {}

splits = [
    'train',
    'valid',
    'test_iid',
    'test_geo',
    'test_vis',
    'test_web',
    'test_cat',
]

for split in splits:
    demos = wl.load_demos_in_split(
        split=split,
        split_path=Path(wl_data_dir) / "splits.json",
        demo_base_dir=Path(wl_data_dir) / "demonstrations",
    )


    results[split] = {}

    valid_intents = {
        "click",
        "hover",
        "textInput",
        "load",
        "scroll",
        "tabcreate",
        "tabswitch",
        "tabremove",
        "submit",
    }

    intents_with_element = {"click", "hover", "textinput", "submit"}
    intents_without_args = {"tabcreate"}

    for demo in tqdm(demos, desc=f"Processing {split} demos"):
        demo_id = demo.name
        demo_start_time = demo.metadata["recordingStart"]
        replay = wl.Replay.from_demonstration(demo)

        results[split][demo_id] = {}
        action_history = []

        demo_name_path = Path(demo.name)
        last_screenshot = demo_name_path / "screenshots" / "screenshot-0-0.png"
        last_bbox = demo_name_path / "bboxes" / "bboxes-0.json"
        last_html = demo_name_path / "pages" / "page-0-0.html"

        last_url = None
        last_tab_id = None

        if not Path(demo.base_dir).joinpath(last_screenshot).exists():
            print(f"File not found: {last_screenshot}")
            last_screenshot = None

        if not Path(demo.base_dir).joinpath(last_bbox).exists():
            print(f"File not found: {last_bbox}")
            last_bbox = None

        if not Path(demo.base_dir).joinpath(last_html).exists():
            print(f"File not found: {last_html}")
            last_html = None

        tab_id_to_url = {}

        for i, turn in enumerate(replay):
            if turn.type == "browser" and turn.intent not in valid_intents:
                continue
            
            # if there's no screenshot and there's no last screenshot, skip, otherwise use last screenshot
            if not turn.has_screenshot():
                if last_screenshot is None:
                    continue
                else:
                    screenshot_path = last_screenshot
            else:
                # otherwise, use the current screenshot
                screenshot_path = Path(turn.get_screenshot_path())
                screenshot_path = screenshot_path.relative_to(turn.base_dir)

                last_screenshot = screenshot_path

            # if one of html or bbox is missing, skip since we want to avoid mismatch
            only_missing_html = not turn.has_html() and turn.has_bboxes()
            only_missing_bbox = turn.has_html() and not turn.has_bboxes()
            if only_missing_html or only_missing_bbox:
                continue

            # if missing both, take last ones unless they're missing
            missing_both = not turn.has_html() and not turn.has_bboxes()

            if missing_both:
                if last_html is None or last_bbox is None:
                    continue
                else:
                    html_path = last_html
                    bbox_path = last_bbox
            else:
                html_path = Path(turn.get_html_path())
                bbox_path = Path(turn.get_bboxes_path())

                html_path = html_path.relative_to(turn.base_dir)
                bbox_path = bbox_path.relative_to(turn.base_dir)

                last_html = html_path
                last_bbox = bbox_path

            base_dir = Path(turn.base_dir)
            # check if replacing html_path to get dom_object_path works, if not we mark it as none
            dom_obj_path = convert_html_path_to_playwright(
                html_path, replace_with="dom_snapshots"
            )
            axtree_path = convert_html_path_to_playwright(html_path, replace_with="axtrees")
            extra_props_path = convert_html_path_to_playwright(
                html_path, replace_with="extra_element_properties"
            )

            if not base_dir.joinpath(dom_obj_path).exists():
                dom_obj_path = None
            else:
                dom_obj_path = str(dom_obj_path)

            if not base_dir.joinpath(axtree_path).exists():
                axtree_path = None
            else:
                axtree_path = str(axtree_path)

            if not base_dir.joinpath(extra_props_path).exists():
                extra_props_path = None
            else:
                extra_props_path = str(extra_props_path)

            ref_action = wlp.outputs.extract_action_from_turn(turn)
            intent = ref_action["intent"]
            args = ref_action["args"]
            element = ref_action["element"]

            # find the tab_id by looking at the last tab_id if it's not provided
            if turn.tab_id is None:
                tab_id = last_tab_id
            elif turn.intent in ["tabcreate", "tabswitch"]:
                tab_id = turn.args["properties"]["tabId"]  # created/switched target tab id
                last_tab_id = tab_id
            else:
                tab_id = turn.tab_id
                last_tab_id = tab_id

            # find the correct url by looking at the last url if it's not provided
            if turn.url is None or turn.url == "":
                if intent == "tabcreate":
                    url = "about:blank"
                else:
                    url = tab_id_to_url.get(tab_id, last_url)
            else:
                url = turn.url
                last_url = url
                tab_id_to_url[tab_id] = url

            if intent in intents_with_element and element is None:
                # skip if the element is None
                continue

            if element is not None:
                # first, move the uid to the args, if it exists (if not, we skip)
                if uid_key not in element["attributes"]:
                    continue
                else:
                    args["uid"] = element["attributes"][uid_key]

                # second, check if the element has more than one attribute, if not we delete it
                if len(element) > 1 and len(element["attributes"]) > 1:
                    print("Element with more than one attribute:", element)
                else:
                    del element["attributes"]

                # for top, right, bottom, left, we round them to nearest decimal point
                for k in ["top", "right", "bottom", "left", "x", "y", "width", "height"]:
                    element["bbox"][k] = round(element["bbox"][k], 1)

            if len(args) == 0 and intent not in intents_without_args:
                print("Empty args:", ref_action)

            # for tabremove, we change "tab_id" to "target"
            if intent == "tabremove":
                args["target"] = args["tab_id"]
                del args["tab_id"]

            # for tabswitch, we change tab_id_from to origin and tab_id_to to target
            if intent == "tabswitch":
                args["origin"] = args["tab_id_from"]
                args["target"] = args["tab_id_to"]
                del args["tab_id_from"]
                del args["tab_id_to"]

            if intent == "textinput":
                args["value"] = args.pop("text")

            # mark a step as is_task if it's not a instructor utterance
            if intent == "say" and args["speaker"] == "instructor":
                is_task = False
            else:
                is_task = True

            has_full_snapshot = (
                axtree_path is not None
                and dom_obj_path is not None
                and extra_props_path is not None
            )

            # update action history
            action_history.append(ref_action)

            # get zoom level
            zoom = turn.zoom if turn.zoom is not None else 1.0

            d = {
                "intent": intent,
                "args": args,
                "is_task": is_task,
                "has_full_snapshot": has_full_snapshot,
                "timestamp": demo_start_time + turn.timestamp,
                "screenshot_path": str(screenshot_path),
                "bbox_path": str(bbox_path),
                "html_path": str(html_path),
                "tab": {"url": url, "id": tab_id},
                "zoom": zoom,
                
                # snapshot info from playwright starts here:
                "axtree_path": axtree_path,
                "dom_object_path": dom_obj_path,
                "extra_props_path": extra_props_path,
                "focused_element_uid": None,
                
                # demo-level metadata
                'user_sees_screen': demo.form['instructor_sees_screen'],
                'uses_ai_output': demo.form['uses_ai_generated_output'],
                'annotator_id': demo.form['annotator'],
                'upload_date': demo.form['upload_date'],
            }

            if element is not None:
                d["element"] = element

            results[split][demo_id][i] = d

        # update each with number of actions
        num_actions = len(action_history)
        for i in results[split][demo_id]:
            results[split][demo_id][i]["num_actions"] = num_actions

# now, we can save the results
with open("metadata.json", "w") as f:
    json.dump(results, f, indent=2)
