"""
Utilities:
1. process_websites(num, dir): Process num websites and save the html and screenshot in the dir
2. single_turn(prompt, bot, img_path, save_path=None): Get the html code from a single screenshot
3. single_turn_exp(bot, img_dir, save_dir, prompt): Get the html code from multiple screenshots
4. multi_turn(bot, img_path, save_path, num_turns): Get the html code from a single screenshot with multiple turns
5. multi_turn_exp(bot, img_dir, save_dir, num_turns): Get the html code from multiple screenshots with multiple turns
6. dcgen(bot, img_path, save_path=None, max_depth=2): Get the html code from a single screenshot using DCGen
7. dcgen_exp(bot, img_dir, save_dir, max_depth=2): Get the html code from multiple screenshots using DCGen

Usage:
1. process_websites(10, "data"): Process 10 websites and save the html and screenshot in the "data" directory
2. bot = GPT4("../keys/key_self.txt", model="gpt-4o"): Load the bot
3. dcgen_exp(bot, "data/original/", "data/dcgen", 2): Get the html code from multiple screenshots using DCGen
4. multi_turn_exp(bot, "data/original/", "data/self_refine", 1): Get the html code from multiple screenshots with multiple turns
5. single_turn_exp(bot, "data/original/", "data/cot", prompt_cot): Get the html code from multiple screenshots
"""

import code
import os
import sys
import json
sys.path.append('..')
from utils import simplify_html, get_driver, take_screenshot, encode_image, GPT4, QwenVL_2, DCGenTrace, ImgSegmentation, Gemini, QwenVL, DCGenGrid, Claude
from utils_method2 import YOLOSegmentation
# from single_file import single_file

import pandas as pd
from multiprocessing import Process
from threading import Thread
from tqdm.auto import tqdm
from single_file import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import re
import time

REFUSAL_PHRASES = ["unable to view", "can't view", "cannot view", "I'm unable to", "I cannot see"]


def get_dir_list(dir, end=".png", exclude="placeholder"):
    if type(exclude) == str:
        filelist = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith(end) and exclude not in x]
    elif type(exclude) == list:
        filelist = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith(end) and all([e not in x for e in exclude])]
    return filelist


def process_website(url, id, dir):
    single_file(url, f"{dir}/{id}.html")
    simplify_html(f"{dir}/{id}.html", f"{dir}/{id}.html")
    driver = get_driver(file=f"{dir}/{id}.html")
    take_screenshot(driver, f"{dir}/{id}.png")
    driver.quit()


def process_websites(num, dir):
    data = pd.read_csv("url_list.csv")[:num]
    p_list = []
    for i in range(num):
        id = data.iloc[i]["id"]
        url = data.iloc[i]["url"]
        p = Process(target=process_website, args=(url, id, dir))
        p_list.append(p)
        p.start()
        if len(p_list) == 10:
            for p in p_list:
                p.join()
            p_list = []
    for p in p_list:
        p.join()


class Method2Grid(DCGenGrid):
    """
    DCGenGrid subclass for Method2.
    """

    _CTX_WIDTH = 1600  # max width used when scaling panels for the VLM
    _PANEL_H   = 1000  # both left and right panels scaled to this height
    _USE_BG_COLOR = False   # set False to use transparent body background, set True to sample and use dominant background color from the screenshot
    _USE_COMPOSITE_IMAGE = False   # set False to send only the segment crop to the VLM
    _USE_VLM_DEDUP = True   # set False to skip the VLM deduplication pre-pass

    def __init__(self, img_seg, prompt_seg, prompt_refine, artifact_dir=None):
        self._skip_ids = set()
        self._artifact_dir = artifact_dir or "."
        super().__init__(img_seg, prompt_seg, prompt_refine)

    # Flow-based HTML scaffold
    def get_html_template(self, output_file=None, verbose=False):
        """
        Generates a flow-based scaffold: vertically stacked <div> sections,
        each with width:100% and height equal to the segment's bbox height.

        Segments in self._skip_ids are omitted. self._skip_ids is empty at
        __init__ time (all segments included) and is populated by
        _vlm_dedup_pass() before the first real generation call.
        """
        page_w, page_h = self.img.size
        bg_color = (YOLOSegmentation._sample_background_color(self.img) if self._USE_BG_COLOR else "transparent")

        leaf_nodes = []
        def collect_leaves(node):
            if node["children"] == []:
                leaf_nodes.append(node)
            else:
                for child in node["children"]:
                    collect_leaves(child)
        collect_leaves(self.img_seg_tree)

        sections_html = ""
        for node in leaf_nodes:
            if node["id"] in self._skip_ids:
                continue
            _, y1, _, y2 = node["bbox"]
            seg_h = max(1, y2 - y1)
            sections_html += (
                f'        <div id="{node["id"]}" class="section" '
                f'style="min-height: {seg_h}px;"></div>\n'
            )

        html = f"""<!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <title>Page Layout</title>
                        <script src="https://cdn.tailwindcss.com"></script>
                        <style>
                            * {{
                                margin: 0;
                                padding: 0;
                                box-sizing: border-box;
                            }}
                            body {{
                                width: 100%;
                                background-color: {bg_color};
                            }}
                            .section {{
                                width: 100%;
                                overflow: hidden; 
                                position: relative; 
                            }}
                        </style>
                    </head>
                    <body>
                    {sections_html}
                    </body>
                    </html>"""

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    # Full pipeline override
    def generate_code(self, bot, multi_thread=True):
        """
        Override to inject the VLM dedup pre-pass and template rebuild
        before per-segment code generation.

        Order:
          1. VLM dedup pre-pass  → populates self._skip_ids
          2. Rebuild scaffold    → omits skipped segments
          3. Generate code dict  → parallel or sequential, skips duplicates
          4. Inject into scaffold
          5. Refinement VLM pass over assembled page
          6. bot.optimize picks best of refined vs raw
        """

        self._skip_ids = self._vlm_dedup_pass(bot) if self._USE_VLM_DEDUP else set()
        self.html_template = self.get_html_template()
        code_dict = self.generate_code_dict(bot, multi_thread)
        self.raw_code = self.code_substitution(self.html_template, code_dict)
        code = bot.try_ask(
            self.prompt_refine.replace("[CODE]", self.raw_code),
            encode_image(self.img),
            num_generations=1,
        )
        # Retry if the model refused to process the image
        if code is None or any(p.lower() in code.lower() for p in REFUSAL_PHRASES):
            print("[refine] model refused image — retrying with explicit instruction")
            code = bot.try_ask(
                "You MUST use the attached image. " + self.prompt_refine.replace("[CODE]", self.raw_code),
                encode_image(self.img),
                num_generations=1,
            )

        if code is None or any(p.lower() in code.lower() for p in REFUSAL_PHRASES):
            print("[refine] fallback to raw_code after retry still refused")
            code = self.raw_code

        pure_code = re.findall(r"```html([^`]+)```", code)
        if pure_code:
            code = pure_code[0]

        # self.code = bot.optimize([code, self.raw_code], self.img, showimg=False)
        self.code = code
        return self.code

    # VLM deduplication pre-pass
    def _vlm_dedup_pass(self, bot):
        """
        Single VLM call with a numbered annotated thumbnail to identify
        duplicate or redundant segments that geometric IoU dedup missed.

        Segments are numbered 1..N in display order (top to bottom).
        The VLM is asked which numbers to remove. Returns a set of node IDs
        (integers) to skip during code generation and scaffold construction.

        Does NOT use detection confidence — wider segments are preferred over
        narrower ones, but after x-snapping all segments are full-width, so
        the y-extent and visual content are the only signals.
        """
        leaf_nodes = []
        def collect_leaves(node):
            if node["children"] == []:
                leaf_nodes.append(node)
            else:
                for child in node["children"]:
                    collect_leaves(child)
        collect_leaves(self.img_seg_tree)

        if len(leaf_nodes) <= 1:
            return set()
        
        dbg = os.path.join(self._artifact_dir, "annotated_thumbnails")
        os.makedirs(dbg, exist_ok=True)

        # Build annotated thumbnail
        page_w, page_h = self.img.size
        scale = min(1.0, self._CTX_WIDTH / page_w, self._PANEL_H / page_h)
        ann_w = max(1, int(page_w * scale))
        ann_h = max(1, int(page_h * scale))
        annotated = self.img.copy().convert("RGB").resize((ann_w, ann_h), Image.LANCZOS)
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except OSError:
            try:
                font = ImageFont.load_default(size=18)
            except TypeError:
                font = ImageFont.load_default()

        # Draw numbered full-width horizontal bands
        id_to_label = {}  # node["id"] (int) → display label (str)
        for i, node in enumerate(leaf_nodes):
            _, y1, _, y2 = node["bbox"]
            sy1 = max(0, int(y1 * scale))
            sy2 = min(ann_h - 1, int(y2 * scale))
            label = str(i + 1)
            id_to_label[node["id"]] = label

            # Alternate badge placement to avoid stacking
            if i % 2 == 0:
                badge_x = 2
            else:
                badge_x = ann_w - 32  # right side
            draw.rectangle([badge_x, sy1 + 2, badge_x + 28, sy1 + 22], fill=(255, 0, 0))
            draw.text((badge_x + 2, sy1 + 3), label, fill=(255, 255, 255), font=font)

        annotated.save(f"{dbg}/annotated.png")

        prompt = (
            "You are shown a full webpage screenshot with numbered red horizontal bands. "
            "Each band corresponds to a detected layout section. "
            "Identify which band NUMBERS are duplicates (same visual content detected twice) "
            "or are redundant sub-regions of larger bands. "
            "We prefer larger bands over smaller sub-regions. "
            "Respond with ONLY a JSON array of the band numbers to REMOVE, e.g. [2, 5]. "
            "If there are no duplicates respond with []. "
            "DO NOT include any explanation or extra text."
        )

        try:
            response = bot.ask(prompt, encode_image(annotated))
            match = re.search(r'\[[\d,\s]*\]', response)
            if not match:
                return set()
            numbers_to_remove = json.loads(match.group())
            label_to_id = {v: k for k, v in id_to_label.items()}
            skip_ids = {
                label_to_id[str(int(n))]
                for n in numbers_to_remove
                if str(int(n)) in label_to_id
            }
            if skip_ids:
                print(f"[VLM dedup] removing band labels {numbers_to_remove} "
                      f"→ node IDs {skip_ids}")
            return skip_ids
        except Exception as exc:
            print(f"[VLM dedup] failed ({exc}), skipping VLM dedup pass")
            return set()

    # Composite context image — full-width horizontal band
    def _make_context_image(self, bbox):
        """
        Returns the image sent to the VLM for one segment.

        If _USE_COMPOSITE_IMAGE is True (default):
            LEFT  — full-page thumbnail with the band highlighted in red.
            RIGHT — the full-width band cropped at original resolution.
        If _USE_COMPOSITE_IMAGE is False:
            Returns only the full-width band crop (scaled to _PANEL_H).
        """
        _, y1, _, y2 = (int(v) for v in bbox)
        page_w, page_h = self.img.size
        tag = f"{y1}_{y2}"
        dbg = os.path.join(self._artifact_dir, "debug_ctx")
        os.makedirs(dbg, exist_ok=True)

        # Always crop the band first
        band = self.img.crop((0, y1, page_w, y2))
        band.save(f"{dbg}/crop_{tag}.png")

        # Scale band (used in both branches)
        right_scale = min(
            self._PANEL_H   / max(band.height, 1),
            self._CTX_WIDTH / max(band.width,  1),
            3.0,
        )
        right_w = max(1, int(band.width  * right_scale))
        right_h = max(1, int(band.height * right_scale))
        band_resized = band.resize((right_w, right_h), Image.LANCZOS)

        if not self._USE_COMPOSITE_IMAGE:
            band_resized.save(f"{dbg}/ctx_{tag}.png")
            return band_resized

        # Left panel: full page scaled to _PANEL_H
        full = self.img.copy()
        left_scale = min(self._PANEL_H / full.height, self._CTX_WIDTH / full.width)
        left_w = max(1, int(full.width  * left_scale))
        left_h = max(1, int(full.height * left_scale))
        full_small = full.resize((left_w, left_h), Image.LANCZOS)
        full_small.save(f"{dbg}/full_{tag}.png")

        draw = ImageDraw.Draw(full_small)
        sy1 = max(0, int(y1 * left_scale))
        sy2 = min(left_h - 1, int(y2 * left_scale))
        draw.rectangle([0, sy1, left_w - 1, sy2], outline=(255, 0, 0), width=3)

        gap = 20
        canvas_w = left_w + gap + right_w
        canvas_h = max(left_h, right_h)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (230, 230, 230))
        canvas.paste(full_small, (0, 0))
        canvas.paste(band_resized, (left_w + gap, 0))
        canvas.save(f"{dbg}/ctx_{tag}.png")
        return canvas

    # Leaf code generation — full-width, y-only constraints
    def _generate_leaf_code(self, bot, node):
        bbox = node["bbox"]
        _, y1, _, y2 = bbox
        seg_h = y2 - y1
        page_w, page_h = self.img.size

        ctx_img = self._make_context_image(bbox)

        dim_hint = (
            f"\nThis section which you should reproduce exactly is a full-width  "
            f"horizontal band: {page_w}px wide × {seg_h}px tall, "
            f"spanning y={y1} to y={y2} on a {page_w}×{page_h} page. "
            "The generated code will be placed inside a container with "
            f"width: 100% and height: {seg_h}px. "
            "The outermost element in your code MUST use width: 100%. "
            "Do NOT use fixed pixel widths, viewport units (vw/vh), "
            "or any width value exceeding 100%."
            "Pay special attention to the exact background color of this"
            "section — do not default to white if the section background is "
            "off-white or colored, approximate it as closely as possible."
        )

        if self._USE_COMPOSITE_IMAGE:
            prompt = self.prompt_seg + dim_hint
        else:
            # Single-image prompt: no LEFT/RIGHT panel description
            prompt = (
                "You are given a single image showing a full-width horizontal section of a webpage.\n\n"
                "Objective:\n"
                "Recreate this section exactly using HTML + Tailwind CSS.\n\n"
                "Implementation Rules:\n"
                "The outermost container must use width: 100%.\n"
                "Do NOT use: Fixed pixel widths, Viewport units (vw, vh) or Any width exceeding 100%.\n"
                "For Images: Replace all images with: placeholder.png at full width.\n"
                "Styling: Match colors, typography, spacing, and proportions as closely as possible.\n\n"
                "Output Requirements:\n"
                "- Return ONLY the inner HTML with Tailwind CSS\n"
                "- Do NOT include <html>, <head>, or <body> tags\n"
                "- Do NOT include any extra text or explanations\n"
                "- Fill in:\n"
                "<div>\n"
                "    Your code here\n"
                "</div>"
            ) + dim_hint

        code = bot.try_ask(prompt, encode_image(ctx_img), num_generations=1)

        # Retry if the model refused to process the image
        if code is None or any(p.lower() in code.lower() for p in REFUSAL_PHRASES):
            print("[leaf_node_code] model refused image — retrying with explicit instruction")
            code = bot.try_ask(
                "You MUST use the attached image. " + prompt,
                encode_image(ctx_img),
                num_generations=1,
            )

        return code.replace("```html", "").replace("```", "")

    # Sequential generation — respects self._skip_ids
    def _generate_code_dict(self, bot):
        code_dict = {}

        def _generate_code(node):
            if node["children"] == []:
                if node["id"] not in self._skip_ids:
                    code_dict[node["id"]] = self._generate_leaf_code(bot, node)
            else:
                for child in node["children"]:
                    _generate_code(child)

        _generate_code(self.img_seg_tree)
        return code_dict

    
    # Parallel generation — respects self._skip_ids
    def _generate_code_dict_parallel(self, bot):
        code_dict = {}
        leaf_nodes = []

        def collect_leaves(node):
            if node["children"] == []:
                leaf_nodes.append(node)
            else:
                for child in node["children"]:
                    collect_leaves(child)

        collect_leaves(self.img_seg_tree)

        active_nodes = [n for n in leaf_nodes if n["id"] not in self._skip_ids]

        with ThreadPoolExecutor() as executor:
            future_to_node = {
                executor.submit(self._generate_leaf_code, bot, node): node
                for node in active_nodes
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                code_dict[node["id"]] = future.result()

        return code_dict


# Prompts used by Method 2
prompt_method_2 = {
    "prompt_seg": (
        "You are given a single composite image consisting of two parts placed (stitched) side-by-side:\n"
        "Use your vision capability to analyze and interpret the image.\n"
        "LEFT: The full webpage with a red horizontal band highlights a specific section.\n"
        "RIGHT: A full-width, high-resolution crop of the highlighted section in the LEFT image.\n\n"
        "Objective:\n"
        "Recreate ONLY the section shown in the RIGHT image using HTML + Tailwind CSS, matching it as precisely as possible.\n\n"
        "How to Use Each Image:\n"
        "RIGHT image (PRIMARY SOURCE):\n"
        "This is what you should reproduce.\n"
        "Reproduce everything visible exactly, including but not limited to:\n"
        "Text content, Layout and structure, Spacing and alignment, Font sizes and styles, Colors  "
        "Backgrounds, Buttons, Cards, Images, and all other UI elements within the RIGHT image.\n"
        "LEFT image (CONTEXT ONLY):\n"
        "Use this only for reference to infer: Global color palette, Typography consistency, "
        "Layout relationships with surrounding sections.\n"
        "Important Clarifications:\n"
        "Please note, the red band is purely a visual marker and does NOT indicate actual background color.\n"
        "Therefore, do NOT apply red backgrounds unless they genuinely appear in the RIGHT image.\n\n"
        "Implementation Rules:\n"
        "The outermost container must use width: 100%.\n"
        "Do NOT use: Fixed pixel width, Viewport units (vw, vh) or Any width exceeding 100%.\n"
        "For Images: Replace all images with: placeholder.png at full width as shown in the RIGHT section\n"
        "Styling: Match colors, typography, spacing, and proportions as closely as possible\n\n"
        "Output Requirements:\n"
        "- Return ONLY the inner HTML with Tailwind CSS\n"
        "- Do NOT include <html>, <head>, or <body> tags\n"
        "- Do NOT include any extra text or explanations\n"
        "- Fill in:\n"
        "<div>\n"
        "    Your code here\n"
        "</div>"
    ),

    "prompt_refine": (
    "You are provided with a screenshot image of a webpage (attached). "
    "You MUST your vision capability to look at and use the attached image — do NOT say you cannot view images. "
    "I have a draft HTML file that APPROXIMATES the layout by stacking sections "
    "vertically — section positions may be inaccurate and there may be missing gaps. "
    "Use the prototype image as the ground truth, and correct the draft HTML to match, "
    "the prototype as closely as possible in terms of layout, colours, any missing or wrong element details. "
    "Return a single, complete, accurate HTML + Tailwind CSS file. "
    "Use 'placeholder.png' for images. "
    "Respond with the full HTML file. "
    "The current draft approximation is:\n\n[CODE]"
)
}


def method2_full(bot, img_path, save_path=None):
    """
    Method2 — full-image mode (no segmentation, single VLM call).
    """
    prompt = prompt_method_2["prompt_refine"].replace("[CODE]", "")
    return single_turn(prompt, bot, img_path, save_path)


def method2_segmented(bot, img_path, layout, save_path=None, multi_thread=True):
    """
    Method2 — segmented mode (YOLO-based)..

    Pipeline:
      1. YOLO inference → deduplicate → x-snap → sort → YOLOSegmentation
      2. Method2Grid: VLM dedup pre-pass → flow scaffold → parallel code gen
      3. Refinement VLM pass → bot.optimize picks best candidate
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_path, stem)

    if isinstance(layout, str) and layout.endswith(".pt"):
        img_seg = YOLOSegmentation.from_yolo_model(
            img_path,
            model_path=layout,
            save_dir=save_path,
            iou_threshold=0.7,
            containment_threshold=0.7,
        )
    else:
        img_seg = YOLOSegmentation(img_path, layout)

    grid = Method2Grid(
        img_seg,
        prompt_seg=prompt_method_2["prompt_seg"],
        prompt_refine=prompt_method_2["prompt_refine"],
        artifact_dir=save_path,   # save_path is already OUT_DIR/{stem} at this point
    )
    grid.generate_code(bot, multi_thread=multi_thread)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"{stem}.html"), "w", encoding="utf-8", errors="ignore") as fh:
            fh.write(grid.code)

    return grid.code


def method2_full_exp(bot, img_dir, save_dir, multi_thread=True):
    """Batch Method2 full-image mode over a directory of PNG screenshots."""
    filelist = get_dir_list(img_dir, exclude=["placeholder", "bbox"])
    os.makedirs(save_dir, exist_ok=True)

    def _run_one(file):
        save_path = os.path.join(
            save_dir, os.path.basename(file).replace(".png", ".html")
        )
        if os.path.exists(save_path):
            return
        try:
            method2_full(bot, file, save_path)
        except Exception as exc:
            print(f"[method2_full_exp] error on {file}: {exc}")

    if multi_thread:
        p_list = []
        for file in tqdm(filelist):
            p = Thread(target=_run_one, args=(file,))
            p.start()
            p_list.append(p)
            if len(p_list) == 5:
                for p in p_list:
                    p.join()
                p_list = []
        for p in p_list:
            p.join()
    else:
        for file in tqdm(filelist):
            _run_one(file)


def method2_segmented_exp(bot, img_dir, layout_dir, save_dir, multi_thread=True):
    """Batch Method2 segmented mode."""
    filelist = get_dir_list(img_dir, exclude=["placeholder", "bbox"])
    os.makedirs(save_dir, exist_ok=True)

    def _run_one(file):
        stem = os.path.splitext(os.path.basename(file))[0]
        layout_path = os.path.join(layout_dir, f"{stem}.json")
        save_path = os.path.join(save_dir, f"{stem}.html")

        if os.path.exists(save_path):
            return
        if not os.path.exists(layout_path):
            print(f"[method2_segmented_exp] no layout for {file}, skipping")
            return
        try:
            method2_segmented(bot, file, layout_path, save_path, multi_thread=False)
        except Exception as exc:
            print(f"[method2_segmented_exp] error on {file}: {exc}")

    if multi_thread:
        p_list = []
        for file in tqdm(filelist):
            p = Thread(target=_run_one, args=(file,))
            p.start()
            p_list.append(p)
            if len(p_list) == 5:
                for p in p_list:
                    p.join()
                p_list = []
        for p in p_list:
            p.join()
    else:
        for file in tqdm(filelist):
            _run_one(file)


prompt_direct = """Here is a prototype image of a webpage. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Use "placeholder.png" to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code."""
prompt_cot = """Here is a prototype image of a webpage. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Please think step by step by dividing the prototype image into multiple parts, write the code for each part, and combine them to form the final code. Use "placeholder.png" to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code."""
prompt_multi = """Here is a prototype image of a webpage. I have an HTML file for implementing a webpage but it has some missing or wrong elements that are different from the original webpage. Please compare the two webpages and revise the original HTML implementation. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Use "placeholder.png" to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code. The current implementation I have is: \n\n [CODE]"""

prompt_dcgen = {
    "prompt_leaf": """Here is a prototype image of a container. Please fill a single piece of HTML and tail-wind CSS code to reproduce exactly the given container. Use 'placeholder.png' to replace the images. Pay attention to things like size, text, and color of all the elements, as well as the background color and layout. Here is the code for you to fill in:
    <div>
    You code here
    </div>
    Respond with only the code inside the <div> tags.""",

    "prompt_root": """Here is a prototype image of a webpage. I have an draft HTML file that contains most of the elements and their correct positions, but it has *inaccurate background*, and some missing or wrong elements. Please compare the draft and the prototype image, then revise the draft implementation. Return a single piece of accurate HTML+tail-wind CSS code to reproduce the website. Use "placeholder.png" to replace the images. Respond with the content of the HTML+tail-wind CSS code. The current implementation I have is: \n\n [CODE]"""
}


def single_turn(prompt, bot, img_path, save_path=None):
    for i in range(3):
        try:
            html = bot.ask(prompt, encode_image(img_path))
            code = re.findall(r"```html([^`]+)```", html)
            if code:
                html = code[0]
            if len(html) < 10:
                raise Exception("No html code found")
            if save_path:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                with open(save_path, 'w', encoding="utf-8") as f:
                    f.write(html)
            return html
        except Exception as e:
            print(e)
            time.sleep(1)
    raise Exception("Failed to get html code")


def single_turn_exp(bot, img_dir, save_dir, prompt, multi_thread=True):
    filelist = get_dir_list(img_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if multi_thread:
        t_list = []
        for file in tqdm(filelist):
            save_path = f"{save_dir}/{file.split('/')[-1].replace('.png', '.html')}"
            if os.path.exists(save_path):
                continue
            t = Thread(target=single_turn, args=(prompt, bot, file, save_path))
            t.start()
            t_list.append(t)
            if len(t_list) == 10:
                for t in t_list:
                    t.join()
                t_list = []
        for t in t_list:
            t.join()
    else:
        for file in tqdm(filelist):
            save_path = f"{save_dir}/{file.split('/')[-1].replace('.png', '.html')}"
            if os.path.exists(save_path):
                continue
            single_turn(prompt, bot, file, save_path)


def multi_turn(bot, img_path, save_path, num_turns):
    initial_html = single_turn(prompt_direct, bot, img_path)
    for i in range(num_turns):
        prompt = prompt_multi.replace("[CODE]", initial_html)
        initial_html = single_turn(prompt, bot, img_path)
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(initial_html)
    return initial_html


def multi_turn_exp(bot, img_dir, save_dir, num_turns, multi_thread=True):
    filelist = get_dir_list(img_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if multi_thread:
        t_list = []
        for file in tqdm(filelist):
            save_path = f"{save_dir}/{file.split('/')[-1].replace('.png', '.html')}"
            if os.path.exists(save_path):
                continue
            t = Thread(target=multi_turn, args=(bot, file, save_path, num_turns))
            t.start()
            t_list.append(t)
            if len(t_list) == 10:
                for t in t_list:
                    t.join()
                t_list = []
        for t in t_list:
            t.join()
    else:
        for file in tqdm(filelist):
            save_path = f"{save_dir}/{file.split('/')[-1].replace('.png', '.html')}"
            if os.path.exists(save_path):
                continue
            multi_turn(bot, file, save_path, num_turns)


def dcgen(bot, img_path, save_path=None, max_depth=2, multi_thread=True, seg_params=None):
    print(f"Running DCGen for {img_path}")
    if not seg_params:
        img_seg = ImgSegmentation(img_path, max_depth=max_depth)
    else:
        img_seg = ImgSegmentation(img_path, **seg_params)

    dcgen_grid = DCGenGrid(img_seg, prompt_seg=prompt_dcgen["prompt_leaf"], prompt_refine=prompt_dcgen["prompt_root"])
    dcgen_grid.generate_code(bot, multi_thread=multi_thread)
    if save_path:
        with open(save_path, 'w', encoding="utf-8", errors="ignore") as f:
            f.write(dcgen_grid.code)
    return dcgen_grid.code


def dcgen_exp(bot, img_dir, save_dir, max_depth=2, multi_thread=True, seg_params=None):
    """img_dir should end with /"""
    filelist = get_dir_list(img_dir, exclude=["placeholder", "bbox"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if multi_thread:
        p_list = []
        for file in tqdm(filelist):
            save_path = f"{save_dir}/{file.split('/')[-1].replace('.png', '.html')}"
            if os.path.exists(save_path):
                continue
            p = Thread(target=dcgen, args=(bot, file, save_path, max_depth, multi_thread, seg_params))
            p.start()
            p_list.append(p)
            if len(p_list) == 5:
                for p in p_list:
                    p.join()
                p_list = []
        for p in p_list:
            p.join()
    else:
        for file in tqdm(filelist):
            save_path = f"{save_dir}/{file.split('/')[-1].replace('.png', '.html')}"
            if os.path.exists(save_path):
                continue
            try:
                dcgen(bot, file, save_path, max_depth, multi_thread=False)
            except:
                continue


def take_screenshots_for_dir(dir, replace=False):
    """dir should end with /"""
    filelist = get_dir_list(dir, end=".html")
    driver = get_driver(string="<html></html>")
    for file in tqdm(filelist):
        if os.path.exists(file.replace(".html", ".png")) and not replace:
            continue
        driver.get("file://" + os.path.abspath(file))
        take_screenshot(driver, file.replace(".html", ".png"))
    driver.quit()


def clean_html_for_dir(dir):
    filelist = get_dir_list(dir, end=".html")
    for file in tqdm(filelist):
        code = open(file, 'r', encoding="utf-8", errors="ignore").read()
        code = code.replace("overflow: auto;", "")
        with open(file, 'w', encoding="utf-8", errors="ignore") as f:
            f.write(code)


if __name__ == "__main__":
    bot = GPT4("../keys/key.txt", model="gpt-4o")

    image_path = "../data/demo_images/1.png"
    yolo_model_path = "../models/yolo11n_400_best.pt"
    save_path = "../data/test_demo/output"

    method2_segmented(bot=bot,
                       img_path=image_path,
                       layout=yolo_model_path,
                       save_path=save_path,
                       multi_thread=False,)