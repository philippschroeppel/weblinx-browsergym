import logging
from pathlib import Path
from socket import timeout
import os
import json
from textwrap import dedent
import signal

import browsergym.core.observation
from browsergym.core.observation import (
    extract_focused_element_bid,
    extract_dom_extra_properties,
    extract_dom_snapshot,
    extract_merged_axtree,
)
from PIL import Image
from playwright.sync_api import sync_playwright, Browser
from tqdm.auto import tqdm
import weblinx.utils


# Define a handler for the timeout
def timeout_handler(signum, frame):
    raise TimeoutError(
        "The function took too long to run, so it was terminated after the timeout."
    )


def run_with_timeout(timeout, function, *args, **kwargs):
    # Register the signal function handler
    signal.signal(signal.SIGALRM, timeout_handler)

    # Start the countdown for the timeout
    signal.alarm(timeout)
    try:
        # Call the function
        result = function(*args, **kwargs)
    finally:
        # Cancel the alarm after the function finishes
        signal.alarm(0)
    return result


def wrap_with_timeout(function, timeout):
    def wrapper(*args, **kwargs):
        return run_with_timeout(timeout=timeout, function=function, *args, **kwargs)

    return wrapper


def convert_temporary_id_format_to_bid(temp_id: str) -> str:
    """
    The temporary format is in the form of "BIDaaaaaaaaxbbbbxcccc" where a, b, c are hex digits.
    We convert it to the format "aaaaaaaa-bbbb-cccc" where a, b, c are hex digits, used by the original weblinx.
    The reason is that - is not allowed in the browsergym format, so we had to replace it with x, and prefix
    with a BID to signify that it is a browsergym id, so that we can find it later.
    """
    if not temp_id.startswith("BID"):
        raise ValueError(f"Invalid temporary id format: {temp_id}")

    temp_id = temp_id[3:]
    temp_id = temp_id.replace("x", "-")
    return temp_id


def is_temporary_id_format(temp_id: str) -> str:
    """
    Check if the id is in the temporary format.
    """
    if not temp_id.startswith("BID"):
        return False

    temp_id = temp_id[3:]
    # count number of x's
    if temp_id.count("x") != 2 or temp_id[8] != "x" or temp_id[13] != "x":
        return False

    # now, verify that the rest are hex digits
    for c in temp_id:
        if c not in "0123456789abcdefx":
            return False

    return True


def remap_dom_snapshot_bid(dom_snapshot: dict) -> dict:
    """
    Here, we look at all the dom_snapshot['strings'] and replace the temporary ids with the weblinx format.
    """
    for i, text in enumerate(dom_snapshot["strings"]):
        if is_temporary_id_format(text):
            dom_snapshot["strings"][i] = convert_temporary_id_format_to_bid(text)


def remap_axtree_bid(axtree: dict) -> dict:
    """
    Here, we look at all the axtree['nodes'] and replace the temporary ids with the weblinx format.
    """
    for node in axtree["nodes"]:
        if "browsergym_id" in node:
            node["browsergym_id"] = convert_temporary_id_format_to_bid(
                node["browsergym_id"]
            )


def remap_extra_props_bid(extra_props: dict) -> dict:
    """
    Here, we look at all the extra_props and replace the temporary ids with the weblinx format.
    """
    new_props = {}

    for key, value in extra_props.items():
        new_props[convert_temporary_id_format_to_bid(key)] = value

    return new_props


def compute_visibility(bbox: dict, screen_width, screen_height) -> str:
    # get proportions of the bounding box with respect to the screen
    x = bbox["x"]
    y = bbox["y"]
    width = bbox["width"]
    height = bbox["height"]

    # check if the bounding box is outside the screen
    if x + width < 0 or x > screen_width or y + height < 0 or y > screen_height:
        return 0

    # check if the bounding box is completely inside the screen
    if x >= 0 and y >= 0 and x + width <= screen_width and y + height <= screen_height:
        return 1

    # calculate the proportion of the bounding box that is inside the screen
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + width, screen_width)
    y2 = min(y + height, screen_height)

    area = (x2 - x1) * (y2 - y1)
    total_area = width * height
    if total_area <= 0:
        return 0
    
    vis = area / total_area

    return vis


def compute_iou(bbox1: dict, bbox2: dict) -> float:
    """
    Compute the intersection over union of two bounding boxes.
    """
    x1 = bbox1["x"]
    y1 = bbox1["y"]
    w1 = bbox1["width"]
    h1 = bbox1["height"]

    x2 = bbox2["x"]
    y2 = bbox2["y"]
    w2 = bbox2["width"]
    h2 = bbox2["height"]

    # get the coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = w1 * h1
    boxBArea = w2 * h2

    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def infer_set_of_marks(
    key: dict,
    bboxes: dict,
    existing_soms: dict,
    iou_threshold=0.9,
    area_px_threshold=25,
) -> bool:
    """
    This function will deduplicate the set of marks by checking if the exact bounding box already exists in the
    existing_soms. If it does, we return False, otherwise, we return True, and add the bounding box to the existing_soms.
    """
    bbox = bboxes[key]

    for k, val in existing_soms.items():
        # if iou is greater than the threshold, we return False
        if compute_iou(bbox, val) > iou_threshold:
            return False

        # now, if the area of the bounding box is less than the threshold, we return False
        if bbox["width"] * bbox["height"] < area_px_threshold:
            return False

    existing_soms[key] = bbox

    return True


def update_extra_props_with_bboxes(
    extra_props: dict, bboxes: dict, screen_width, screen_height
) -> dict:
    """
    Given the extra_props and bboxes, we update the bounding box in extra_props with the bounding box in bboxes.
    """
    for key, value in extra_props.items():
        existing_soms = {}
        if key in bboxes:
            b = bboxes[key]
            extra_props[key]["bbox"] = [b["x"], b["y"], b["width"], b["height"]]
            extra_props[key]["visibility"] = compute_visibility(
                b, screen_width=screen_width, screen_height=screen_height
            )
            extra_props[key]["set_of_marks"] = infer_set_of_marks(
                key, bboxes, existing_soms=existing_soms
            )
        else:
            extra_props[key]["visibility"] = 0
            extra_props[key]["set_of_marks"] = False


def get_snapshot_from_path(html_file_path, browser: Browser):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    page = browser.new_page()

    # Convert the HTML file path to an absolute path
    absolute_path = os.path.abspath(html_file_path)

    # # Navigate to the local HTML file
    # # Construct the file URL
    # file_url = "file://" + absolute_path.replace("\\", "/")
    # page.goto(file_url, wait_until="load", timeout=5000)

    # load file into string
    with open(absolute_path, "r") as f:
        html_str = f.read()

    page.set_content(html_str, wait_until="load", timeout=5000)

    # Wait for the page to load completely
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception as e:
        print("=" * 50)
        print(f"Error in loading page '{html_file_path}': {e}")
        print("-" * 50)
        print("Will try to get the accessibility tree anyway.")

    # first, replace all data-webtasks-id with bid, then inject attribute aria-roledescription=BID_<uid>
    page.evaluate(
        """
(function(){
function push_bid_to_attribute(bid, elem, attr){
    let original_content = "";
    if (elem.hasAttribute(attr)) {
        original_content = elem.getAttribute(attr);
    }
    let new_content = `browsergym_id_${bid} ${original_content}`
    elem.setAttribute(attr, new_content);
}

function replace_and_inject() {
    const elements = document.querySelectorAll('[data-webtasks-id]');
    for (let i = 0; i < elements.length; i++) {
        const elem = elements[i];
        let bid = elem.getAttribute('data-webtasks-id');
        // replace the - with _ in the bid
        bid = bid.replace(/-/g, 'x');
        bid = "BID" + bid;
        elem.setAttribute('bid', bid);
        elem.removeAttribute('data-webtasks-id');
        
        // fallback for generic nodes
        push_bid_to_attribute(bid, elem, 'aria-roledescription');
        push_bid_to_attribute(bid, elem, "aria-description");
    }
}

replace_and_inject();
})();
"""
    )

    dom_snapshot = extract_dom_snapshot(page, include_dom_rects=True)
    # include_dom_rects is set to false since the rendered bounding boxes are not accurate, given that
    # we don't have the process CSS of the page. We will use the bounding boxes from bboxes.json

    # get the accessibility tree
    # add a timeout to the function, if it takes too long, we will just skip it
    axtree_snapshot = extract_merged_axtree(page)
    extra_props = extract_dom_extra_properties(dom_snapshot=dom_snapshot)

    # remap the ids from the temporary format to the weblinx format
    remap_dom_snapshot_bid(dom_snapshot)
    remap_axtree_bid(axtree_snapshot)

    # extra_props will give bad bounding boxes, need to post-process with bboxes.json
    extra_props = remap_extra_props_bid(extra_props)

    # delete page
    page.close()

    return axtree_snapshot, dom_snapshot, extra_props


def main(base_dir="./wl_data", allowed_demo_ids=None, skipped_demo_ids=None):

    base_dir = Path(base_dir)
    # get all html files in base_dir/demonstrations/<demo_id>/pages/<name>.html and save them in
    # base_dir/demonstrations/<demo_id>/axtrees/<name>.json
    relaunch_browser_rate = 1
    total_so_far = 0

    file_lst = list(base_dir.glob("demonstrations/*/pages/*.html"))

    # filter based on allowed_demo_ids
    if allowed_demo_ids is not None:
        allowed_demo_ids = set(allowed_demo_ids)
        file_lst = [
            html_file_path
            for html_file_path in file_lst
            if html_file_path.parts[-3] in allowed_demo_ids
            and html_file_path.parts[-3] not in skipped_demo_ids
        ]

    # with sync_playwright() as p:
    # browser = p.chromium.launch()

    for i, html_path in enumerate(file_lst):
        print()
        prefix = f"[{i+1}/{len(file_lst)}]"
        # get the demo_id and name of the html file
        demo_id = html_path.parts[-3]

        name = html_path.stem
        demo_dir = base_dir / "demonstrations" / demo_id

        axtree_path = demo_dir / f"axtrees/{name}.json"
        dom_snap_path = demo_dir / f"dom_snapshots/{name}.json"
        extra_props_path = demo_dir / f"extra_element_properties/{name}.json"

        index, index2 = weblinx.utils.get_nums_from_path(html_path)
        bboxes_path = demo_dir / "bboxes" / f"bboxes-{index}.json"
        screenshot_path = (
            demo_dir / "screenshots" / f"screenshot-{index}-{index2}.png"
        )

        if screenshot_path.exists():
            # get the widht and height of the screenshot from the metadata without loading
            im = Image.open(screenshot_path)
            screen_width, screen_height = im.size
            im.close()
        else:
            print(
                f"{prefix} Screenshot does not exist in '{screenshot_path}', defaulting to 1366 x 768"
            )
            screen_width, screen_height = 1366, 768

        failed_path = axtree_path.parent / f"{name}-failed.json"

        if failed_path.exists():
            print(
                f"{prefix} Snapshot processing previously failed, see {failed_path}"
            )
            continue

        if not bboxes_path.exists():
            print(f"{prefix} Bounding box file does not exist in '{bboxes_path}'")
            continue

        if (
            axtree_path.exists()
            and dom_snap_path.exists()
            and extra_props_path.exists()
        ):
            print(f"{prefix} Accessibility tree already exists in '{axtree_path}'")
            print(f"{prefix} DOM snapshot already exists in '{dom_snap_path}'")
            print(
                f"{prefix} Extra elem props already exists in '{extra_props_path}'"
            )
            continue

        axtree_path.parent.mkdir(parents=True, exist_ok=True)
        dom_snap_path.parent.mkdir(parents=True, exist_ok=True)
        extra_props_path.parent.mkdir(parents=True, exist_ok=True)

        # total_so_far += 1
        # if total_so_far % relaunch_browser_rate == 0:
        #     print(f"{prefix} Closing browser after {total_so_far} pages")
        #     browser.close()
        #     print(f"{prefix} Relaunching browser after {total_so_far} pages")
        #     browser = p.chromium.launch()

        print(f"{prefix} Loading: {bboxes_path}")
        with open(bboxes_path, "r") as f:
            bboxes = json.load(f)

        with sync_playwright() as p:
            # now, get the screen width and height from the dataset
            print(f"{prefix} Launching browser")
            browser = p.chromium.launch()

            print(f"{prefix} Processing: {html_path}")

            try:
                get_snapshot_from_path_timeout = wrap_with_timeout(
                    get_snapshot_from_path, timeout=20
                )
                axtree_snap, dom_snap, extra_props = get_snapshot_from_path_timeout(
                    html_file_path=html_path, browser=browser
                )
            except Exception as e:
                print("=" * 50)
                print(f"{prefix} Error in getting accessibility tree for {html_path}: ")
                print(e)
                print("-" * 50)
                # save a "failed.json" file in the axtrees directory
                with open(failed_path, "w") as f:
                    json.dump({"error": str(e), "html_file_path": str(html_path)}, f)
                print(f"{prefix} Saved failed info json in: '{failed_path}'")
                browser.close()
                print(f"{prefix} Closed browser")
                continue

            else:
                update_extra_props_with_bboxes(
                    extra_props=extra_props,
                    bboxes=bboxes,
                    screen_width=screen_width,
                    screen_height=screen_height,
                )
                # in case of success, we print success message
                print(f"{prefix} Completed without errors!")

            # save the accessibility tree in axtree_file_path
            with open(axtree_path, "w") as f:
                json.dump(axtree_snap, f)
            print(f"{prefix} Saved axtree tree in: '{axtree_path}'")

            with open(dom_snap_path, "w") as f:
                json.dump(dom_snap, f)
            print(f"{prefix} Saved DOM snapshot in: '{dom_snap_path}'")

            with open(extra_props_path, "w") as f:
                json.dump(extra_props, f)
            print(f"{prefix} Saved extra props in: '{extra_props_path}'")

            browser.close()
            print(f"{prefix} Closed browser")

    # browser.close()


# Example usage
if __name__ == "__main__":
    import argparse
    import weblinx as wl

    # split = "test_vis"

    parser = argparse.ArgumentParser(
        description="Get snapshots for the web tasks dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s", "--split",
        type=str,
        default="test_iid",
        help="The split to get snapshots for.",
        choices=["train", "valid", "test_iid", "test_vis", "test_geo", "test_cat", "test_web"],
    )

    split = parser.parse_args().split

    # splits are:
    #   train: The set used for training
    #   valid: Demos from the same websites as the train set, used for training
    #   test_iid: Demos from the same websites as the train set
    #   test_vis: Demos where the instructor cannot see the navigator screen
    #   test_geo: Demos from different geographical locations from train set
    #   test_cat: Demos from new sub-categories (but same top-level category)
    #   test_web: Demos from websites not in the training set

    base_dir = "wl_data"
    wl_data_dir = "wl_data"

    skipped_demo_ids = {"oiiwawv", "bkseapp", "etzkmrj", "cdfkxtv", "kjcptgq"}

    demo_ids = wl.utils.load_demo_names_in_split(
        split=split,
        split_path=Path(base_dir, "splits.json"),
    )

    main(base_dir, allowed_demo_ids=demo_ids, skipped_demo_ids=skipped_demo_ids)
