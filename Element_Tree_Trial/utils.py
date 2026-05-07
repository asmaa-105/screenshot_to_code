from typing import Union
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from skimage.metrics import structural_similarity as ssim
import os
from PIL import Image, ImageDraw, ImageEnhance 
from tqdm.auto import tqdm
import time
import re
import shutil
import base64
import io
from openai import OpenAI, AzureOpenAI
import numpy as np
import json
import anthropic
import random
import math
import cv2
import easyocr


def _is_llm_transient_error(err_msg: str) -> bool:
    msg = (err_msg or "").lower()
    transient_markers = [
        "503",
        "429",
        "unavailable",
        "resource_exhausted",
        "quota",
        "overloaded",
        "deadline exceeded",
        "timeout",
        "internal",
    ]
    return any(marker in msg for marker in transient_markers)


def llm_call_with_retry(func, max_retries=6, base_wait=2.0, max_wait=30.0, provider_name="LLM"):
    """
    Retry transient API failures with exponential backoff + jitter.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            msg = str(e)
            if not _is_llm_transient_error(msg):
                raise

            retry_idx = attempt + 1
            if retry_idx >= max_retries:
                break

            # Exponential backoff with small jitter to reduce synchronized retries.
            wait = min(max_wait, base_wait * (2 ** attempt)) + random.uniform(0, 0.75)
            print(f"{provider_name} overloaded. Retry {retry_idx}/{max_retries - 1} in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"{provider_name} failed after retries: {last_error}")

def take_screenshot(driver, filename):
    driver.save_full_page_screenshot(filename)

def get_driver(file=None, headless=True, string=None, window_size=(1920, 1080)):
    assert file or string, "You must provide a file or a string"
    options = Options()
    if headless:
        options.add_argument("-headless")
        driver = webdriver.Firefox(options=options)  # or use another driver
    else:
        driver = webdriver.Firefox(options=options)

    if not string:
        driver.get("file:///" + os.getcwd() + "/" + file)
    else:
        string = base64.b64encode(string.encode('utf-8')).decode()
        driver.get("data:text/html;base64," + string)

    driver.set_window_size(window_size[0], window_size[1])
    return driver


from playwright.sync_api import sync_playwright
import os
import base64

def take_screenshot_pw(page, filename=None):
    # Takes a full-page screenshot with Playwright
    if filename:
        page.screenshot(path=filename, full_page=True)
    else:
        return page.screenshot(full_page=True)  # Returns the screenshot as bytes if no filename is provided

def get_driver_pw(file=None, headless=True, string=None, window_size=(1920, 1080)):
    assert file or string, "You must provide a file or a string"
   
    p = sync_playwright().start()  # Start Playwright context manually
    browser = p.chromium.launch(headless=headless)
    page = browser.new_page()

    # If the user provides a file, load it, else load the HTML string
    if file:
        page.goto("file://" + os.getcwd() + "/" + file)
    else:
        string = base64.b64encode(string.encode('utf-8')).decode()
        page.goto("data:text/html;base64," + string)
    
    # Set the window size
    page.set_viewport_size({"width": window_size[0], "height": window_size[1]})
    
    return page, browser  # Return the page and browser objects


def get_placeholder_url():
    placeholder_path = os.path.join(os.path.dirname(__file__), "placeholder.png")

    if not os.path.exists(placeholder_path):
        return None

    with open(placeholder_path, "rb") as image_file:
        img_data = image_file.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

def get_placeholder(html):
    placeholder_url = get_placeholder_url()
    if placeholder_url:
        return html.replace("placeholder.png", placeholder_url)
    return html

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
class Bot:
    def __init__(self, key_path, patience=3) -> None:
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                self.key = f.read().replace("\n", "")
        else:
            self.key = key_path
        self.patience = patience
    
    def ask(self):
        raise NotImplementedError
    
    def attempt_ask_with_retries(self, question, image_encoding, verbose):
        for attempt in range(self.patience):
            try:
                return self.ask(question, image_encoding, verbose)  # Attempt to ask
            except Exception as e:
                if attempt < self.patience - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"All attempts failed for this generation: {e}")
                    return None  # Return None if all attempts fail

    
    def try_ask(self, question, image_encoding=None, verbose=False, num_generations=1, multithread=True):
        assert num_generations > 0, "num_generations must be greater than 0"
        if num_generations == 1:
            for i in range(self.patience):
                try:
                    return self.ask(question, image_encoding, verbose)
                except Exception as e:
                    print(e, "waiting for 5 seconds")
                    time.sleep(5)
            return None
        elif multithread:
            responses = []

            # Helper function to attempt 'self.ask' with retries

            # Using ThreadPoolExecutor to handle parallel execution
            with ThreadPoolExecutor() as executor:
                futures = []
                
                # Submit tasks to the executor (one task per generation)
                for i in range(num_generations):
                    futures.append(executor.submit(self.attempt_ask_with_retries, question, image_encoding, verbose))
                
                # Collect responses as they complete
                for future in as_completed(futures):
                    result = future.result()  # Get the result from the future
                    if result:  # Only append if we got a valid result (non-None)
                        responses.append(result)
                    else:
                        print(f"Generation {futures.index(future)} failed after {self.patience} attempts.")

            # print(f"Responses received: {len(responses)}")
        
        else:
            responses = []
            for i in range(num_generations):
                for j in range(self.patience):
                    try:
                        responses.append(self.ask(question, image_encoding, verbose))
                        break
                    except Exception as e:
                        print(e, "waiting for 5 seconds")
                        time.sleep(5)
        return self.optimize(responses, image_encoding) 


    def optimize(self, candidates, img, window_size=(1920, 1080), showimg=False):
        # print("Optimizing candidates...")
        # print([x[:20] for x in candidates])
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tailwind CSS Template</title>
            <!-- Tailwind CSS CDN Link -->
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body>
            [CODE]
        </body>
        </html>
        """
        with sync_playwright() as p:
            # Start Playwright context manually
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            min_mae = float('inf')
            if type(img) == str:
                img = Image.open(io.BytesIO(base64.b64decode(img)))
            img = img.convert("RGB")
            page.set_viewport_size({"width": img.size[0], "height": img.size[1]})
            # print("Image size:", np.array(img).shape)
            for candidate in candidates:
            # Set the content of the page to the candidate HTML
                code = re.findall(r"```html([^`]+)```", candidate)
                if code:
                    candidate = code[0]
                structure_penalty = html_structure_penalty(candidate)
                candidate = html_template.replace("[CODE]", candidate)
                page.set_content(get_placeholder(candidate))
                # Take a screenshot and get it in-memory
                screenshot_data = take_screenshot_pw(page)
                # Convert screenshot data to an image in memory
                screenshot_img = Image.open(io.BytesIO(screenshot_data)).convert("RGB").resize(img.size)
                # print("Screenshot size:", np.array(screenshot_img).shape)

                # img.show()
                # Calculate the mean absolute error (MAE) between the screenshot and the original image
                mae = np.mean(np.abs(np.array(screenshot_img) - np.array(img)))
                score = mae + structure_penalty
                # screenshot_img.show()
                # print(mae)
                # Track the best candidate based on MAE
                if score < min_mae:
                    min_mae = score
                    best_response = candidate

            # Return the best response
            return best_response


class Gemini(Bot):
    def __init__(self, key_path, patience=3) -> None:
        super().__init__(key_path, patience)
        import google.generativeai as genai
        GOOGLE_API_KEY = self.key
        genai.configure(api_key=GOOGLE_API_KEY)
        self.name = "gemini"
        self.file_count = 0
        self._genai = genai
        
    def ask(self, question, image_encoding=None, verbose=False):
        model = self._genai.GenerativeModel('gemini-2.5-flash')
        config = self._genai.types.GenerationConfig(temperature=0.2, max_output_tokens=10000)

        if verbose:
            print(f"##################{self.file_count}##################")
            print("question:\n", question)

        if image_encoding:
            img = base64.b64decode(image_encoding)
            img = Image.open(io.BytesIO(img))
            response = model.generate_content([question, img], request_options={"timeout": 3000}, generation_config=config) 
        else:    
            response = model.generate_content(question, request_options={"timeout": 3000}, generation_config=config)
        response.resolve()

        if verbose:
            print("####################################")
            print("response:\n", response.text)
            self.file_count += 1

        return response.text

class GPT4(Bot):
    def __init__(self, key_path, patience=3, model="gpt-4o") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        # self.client = AzureOpenAI(
        #             azure_endpoint="",
        #             api_key="",
        #             api_version=""
        #         )
        self.name="gpt4"
        self.model = model
        self.max_tokens = 10000
        
    def ask(self, question, image_encoding=None, verbose=False):
        
        if image_encoding:
            content =    {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_encoding}",
                },
                },
            ],
            }
        else:
            content = {"role": "user", "content": question}
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
         content
        ],
        max_tokens=self.max_tokens,
        temperature=0.2,
        seed=42,
        )
        response = response.choices[0].message.content
        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response)
            print("seed used: 42")
            # img = base64.b64decode(image_encoding)
            # img = Image.open(io.BytesIO(img))
            # img.show()
        return response

class QwenVL(GPT4):
    def __init__(self, key_path, model="qwen2.5-vl-72b-instruct", patience=3) -> None:
        super().__init__(key_path, patience, model)
        self.name = "qwenvl"
        self.client = OpenAI(api_key=self.key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.max_tokens = 8192


class Claude(Bot):
    def __init__(self, key_path, patience=3) -> None:
        super().__init__(key_path, patience)
        self.client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=self.key,
        )
        self.name = "claude"
        self.file_count = 0
        
    def ask(self, question, image_encoding=None, verbose=False):

        if image_encoding:
            content =   {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_encoding,
                        },
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ],
            }
        else:
            content = {"role": "user", "content": question}


        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=0.2,
            messages=[content],
        )
        response = message.content[0].text
        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response)

        return response


def load_element_tree(tree_or_path):
    if isinstance(tree_or_path, str):
        with open(tree_or_path, "r", encoding="utf-8") as f:
            tree = json.load(f)
    else:
        tree = tree_or_path
    return tree


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def sanitize_element_tree(tree, image_size=None):
    """
    Ensures:
    - image_size exists
    - bbox values are valid
    - missing parent => root
    - missing ids => auto-generated
    """
    tree = json.loads(json.dumps(tree))  # deep copy
    elements = tree.get("elements", [])

    if image_size is None:
        img_w = tree.get("image_size", {}).get("w")
        img_h = tree.get("image_size", {}).get("h")
    else:
        img_w, img_h = image_size

    if img_w is None or img_h is None:
        raise ValueError("image_size is required either in tree or via image_size arg")

    clean = []
    for i, e in enumerate(elements):
        bbox = e.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        bbox = clamp_bbox(bbox, img_w, img_h)
        if bbox is None:
            continue

        clean.append({
            "id": str(e.get("id", f"node_{i}")),
            "type": str(e.get("type", "unknown")).lower().strip(),
            "bbox": bbox,
            "text": e.get("text", "") or "",
            "parent": str(e.get("parent", "root")),
            "reading_index": int(e.get("reading_index", i)),
        })

    tree["image_size"] = {"w": img_w, "h": img_h}
    tree["elements"] = clean
    return tree


def element_tree_to_nested(tree):
    """
    Convert flat element list with parent ids into nested tree:
    {
      "id": "root",
      "bbox": [0,0,w,h],
      "children": [...]
    }
    """
    w = tree["image_size"]["w"]
    h = tree["image_size"]["h"]

    nodes = {
        "root": {
            "id": "root",
            "type": "root",
            "bbox": [0, 0, w, h],
            "text": "",
            "reading_index": -1,
            "children": []
        }
    }

    for e in tree["elements"]:
        nodes[e["id"]] = {
            "id": e["id"],
            "type": e["type"],
            "bbox": e["bbox"],
            "text": e["text"],
            "reading_index": e["reading_index"],
            "children": []
        }

    for e in tree["elements"]:
        parent_id = e.get("parent", "root")
        if parent_id not in nodes:
            parent_id = "root"
        nodes[parent_id]["children"].append(nodes[e["id"]])

    def sort_children(node):
        node["children"] = sorted(
            node["children"],
            key=lambda x: (x.get("reading_index", 10**9), x["bbox"][1], x["bbox"][0])
        )
        for child in node["children"]:
            sort_children(child)

    sort_children(nodes["root"])
    return nodes["root"]


def draw_element_tree_overlay(image, tree, output_path=None):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    out = image.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    for e in tree.get("elements", []):
        x1, y1, x2, y2 = e["bbox"]
        label = e["type"]
        if e.get("text"):
            label += f": {e['text'][:24]}"

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        tx, ty = x1 + 2, max(0, y1 - 14)
        draw.rectangle([tx, ty, tx + 8 + 7 * min(len(label), 30), ty + 14], fill="black")
        draw.text((tx + 2, ty), label[:30], fill="lime", font=font)

    if output_path:
        out.save(output_path)
    return out
    
def process_image_to_fixed_json_only(image_path, sample_out_dir, api_key, min_ocr_conf=0.4):
    """
    image -> raw tree -> OCR/NMS bbox fix -> fixed json
    """
    os.makedirs(sample_out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    raw_json_path = os.path.join(sample_out_dir, f"{base}_tree_raw.json")
    fixed_json_path = os.path.join(sample_out_dir, f"{base}_tree_fixed.json")

    if os.path.exists(fixed_json_path):
        return {
            "raw_json_path": raw_json_path if os.path.exists(raw_json_path) else None,
            "fixed_json_path": fixed_json_path,
        }

    tree_raw = get_element_tree_raw(image_path, api_key)
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(tree_raw, f, indent=2, ensure_ascii=False)

    tree_fixed = generate_fixed_element_tree_json(
        image_path,
        api_key=None,
        min_ocr_conf=min_ocr_conf,
        tree_raw=tree_raw
    )
    with open(fixed_json_path, "w", encoding="utf-8") as f:
        json.dump(tree_fixed, f, indent=2, ensure_ascii=False)

    return {
        "raw_json_path": raw_json_path,
        "fixed_json_path": fixed_json_path,
    }
    
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import base64
from tqdm.auto import tqdm
import os
from PIL import Image, ImageDraw, ImageChops

def num_of_nodes(driver, area="body", element=None):
    # number of nodes in body
    element = driver.find_element(By.TAG_NAME, area) if not element else element
    script = """
    function get_number_of_nodes(base) {
        var count = 0;
        var queue = [];
        queue.push(base);
        while (queue.length > 0) {
            var node = queue.shift();
            count += 1;
            var children = node.children;
            for (var i = 0; i < children.length; i++) {
                queue.push(children[i]);
            }
        }
        return count;
    }
    return get_number_of_nodes(arguments[0]);
    """
    return driver.execute_script(script, element)

measure_time = {
    "script": 0,
    "screenshot": 0,
    "comparison": 0,
    "open image": 0,
    "hash": 0,
}


import hashlib
import mmap

def compute_hash(image_path):
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        # Use memory-mapped file for efficient reading
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            hash_md5.update(mm)
    return hash_md5.hexdigest()

def are_different_fast(img1_path, img2_path):
    # a extremely fast algorithm to determine if two images are different,
    # only compare the size and the hash of the image
    return compute_hash(img1_path) != compute_hash(img2_path)

str2base64 = lambda s: base64.b64encode(s.encode('utf-8')).decode()

import time

def simplify_graphic(driver, element, progress_bar=None, img_name={"origin": "origin.png", "after": "after.png"}):
    """utility for simplify_html, simplify the html by removing elements that are not visible in the screenshot"""
    children = element.find_elements(By.XPATH, "./*")
    deletable = True
    # check childern
    if len(children) > 0:
        for child in children:
            deletable *= simplify_graphic(driver, child, progress_bar=progress_bar, img_name=img_name)
    # check itself
    
    if deletable:
        original_html = driver.execute_script("return arguments[0].outerHTML;", element)

        tick = time.time()
        driver.execute_script("""
            var element = arguments[0];
            var attrs = element.attributes;
            while(attrs.length > 0) {
                element.removeAttribute(attrs[0].name);
            }
            element.innerHTML = '';""", element)
        measure_time["script"] += time.time() - tick
        tick = time.time()
        driver.save_full_page_screenshot(img_name["after"])
        measure_time["screenshot"] += time.time() - tick
        tick = time.time()
        deletable = not are_different_fast(img_name["origin"], img_name["after"])
        measure_time["comparison"] += time.time() - tick

        if not deletable:
            # be careful with children vs child_node and assining outer html to element without parent
            driver.execute_script("arguments[0].outerHTML = arguments[1];", element, original_html)
        else:
            driver.execute_script("arguments[0].innerHTML = 'MockElement!';", element)
            # set visible to false
            driver.execute_script("arguments[0].style.display = 'none';", element)
    if progress_bar:
        progress_bar.update(1)

    return deletable
            
def simplify_html(fname, save_name, pbar=True, area="html", headless=True):
    """simplify the html file and save the result to save_name, return the compression rate of the html file after simplification"""
    # copy the fname as save_name
    
    driver = get_driver(file=fname, headless=headless)
    print("driver initialized")
    original_nodes = num_of_nodes(driver, area)
    bar = tqdm(total=original_nodes) if pbar else None
    compression_rate = 1
    driver.save_full_page_screenshot(f"{fname}_origin.png")
    try:
        simplify_graphic(driver, driver.find_element(By.TAG_NAME, area), progress_bar=bar, img_name={"origin": f"{fname}_origin.png", "after": f"{fname}_after.png"})
        elements = driver.find_elements(By.XPATH, "//*[text()='MockElement!']")

        # Iterate over the elements and remove them from the DOM
        for element in elements:
            driver.execute_script("""
                var elem = arguments[0];
                elem.parentNode.removeChild(elem);
            """, element)
        
        compression_rate = num_of_nodes(driver, area) / original_nodes
        with open(save_name, "w", encoding="utf-8") as f:
            f.write(driver.execute_script("return document.documentElement.outerHTML;"))
    except Exception as e:
        print(e, fname)
    # remove images
    driver.quit()

    os.remove(f"{fname}_origin.png")
    os.remove(f"{fname}_after.png")
    return compression_rate


# Function to encode the image in base64
def encode_image(image):
    if type(image) == str:
        try: 
            with open(image, "rb") as image_file:
                encoding = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(e)
            with open(image, "r", encoding="utf-8") as image_file:
                encoding = base64.b64encode(image_file.read()).decode('utf-8')
        return encoding
    
    else:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


from PIL import Image, ImageDraw, ImageFont
import random
class FakeBot(Bot):
    def __init__(self, key_path, patience=1) -> None:
        self.name = "FakeBot"
        pass
        
    def ask(self, question, image_encoding=None, verbose=False):
        print(question)
        if image_encoding:
            pass
            # img = base64.b64decode(image_encoding)
            # img = Image.open(io.BytesIO(img))
            # "The bounding box is: (xx, xx, xx, xx)"
            # bbox = re.findall(r"(\([\d]+, [\d]+, [\d]+, [\d]+\))", question)
            # draw = ImageDraw.Draw(img)
            # draw.rectangle(eval(bbox[0]), outline="red", width=5)
            # draw.text((10, 10), question, fill="green")
            # img.show()
            # if random.random() > 0.5:
            #     raise Exception("I am not able to do this")
        return f"```html \nxxxxxxxxxxxxxxxxxxx\n```"


from abc import ABC, abstractmethod
import random

class ImgNode(ABC):
    # self.img: the image of the node
    # self.bbox: the bounding box of the node
    # self.children: the children of the node

    @abstractmethod
    def get_img(self):
        pass

MODEL_ELEMENT_TREE = os.environ.get("ELEMENT_TREE_MODEL", "gpt-4o")

PROMPT_ELEMENT_TREE = """
Analyze this UI screenshot.

Return STRICTLY VALID JSON.

Return a JSON object with:
- image_size: {w, h}
- elements: list of {
    id: string,
    type: container | text | button | input | icon | image | unknown,
    bbox: [x1,y1,x2,y2] in pixel coordinates,
    text: string,
    parent: string (or "root"),
    reading_index: integer
}

IMPORTANT:
- bbox must be pixel coords for the screenshot size.
Return JSON only. No explanations.
"""


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def token_jaccard(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, len(ta | tb))


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(1, (x2 - x1) * (y2 - y1))

def bbox_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih

def bbox_contains_ratio(parent_bbox, child_bbox):
    inter = bbox_intersection_area(parent_bbox, child_bbox)
    return inter / float(max(1, bbox_area(child_bbox)))


def center_dist(a, b) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def nms_boxes(boxes, iou_thresh=0.25):
    # boxes: list of [x1,y1,x2,y2,score]
    if not boxes:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou < iou_thresh)[0]
        order = order[inds + 1]

    return boxes[keep].tolist()


def detect_ui_candidates(img_pil: Image.Image, min_conf=0.4):
    img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    reader = easyocr.Reader(['en'], gpu=True)
    ocr = reader.readtext(img_bgr)

    text_boxes = []
    for poly, text, conf in ocr:
        if conf < min_conf:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        text_boxes.append([x1, y1, x2, y2, float(conf), text])

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Lightweight detector branch 1: MSER regions (good for buttons/inputs/cards/text groups).
    try:
        mser = cv2.MSER_create(5, 120, max(800, int(0.18 * W * H)))
    except Exception:
        mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mser_boxes = []
    for pts in regions:
        x, y, ww, hh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        area = ww * hh
        if area < 1000 or ww < 20 or hh < 16:
            continue
        if area > 0.7 * W * H:
            continue
        aspect = ww / float(max(1, hh))
        if aspect > 20 or aspect < 0.08:
            continue
        mser_boxes.append([x, y, x + ww, y + hh, 0.58, ""])

    # Lightweight detector branch 2: connected components on gradient map.
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.convertScaleAbs(mag)
    _, grad_bin = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cc_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad_bin = cv2.morphologyEx(grad_bin, cv2.MORPH_CLOSE, cc_kernel, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(grad_bin, connectivity=8)
    cc_boxes = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 700 or ww < 18 or hh < 14:
            continue
        if area > 0.85 * W * H:
            continue
        cc_boxes.append([int(x), int(y), int(x + ww), int(y + hh), 0.54, ""])

    # Lightweight detector branch 3: explicit line/rectangle-like proposals.
    edges = cv2.Canny(blur, 60, 180)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, line_kernel, iterations=2)
    contours_edges, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes = []
    for c in contours_edges:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 1000 or ww < 22 or hh < 16:
            continue
        if area > 0.90 * W * H:
            continue
        if min(ww, hh) < 6:
            continue
        line_boxes.append([x, y, x + ww, y + hh, 0.52, ""])

    all_for_nms = [[b[0], b[1], b[2], b[3], b[4]] for b in text_boxes]
    all_for_nms.extend([[b[0], b[1], b[2], b[3], b[4]] for b in mser_boxes])
    all_for_nms.extend([[b[0], b[1], b[2], b[3], b[4]] for b in cc_boxes])
    all_for_nms.extend([[b[0], b[1], b[2], b[3], b[4]] for b in line_boxes])
    merged = nms_boxes(all_for_nms, iou_thresh=0.25)
    merged_xyxy = [[int(m[0]), int(m[1]), int(m[2]), int(m[3])] for m in merged]

    candidates = [
        {
            "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
            "text": b[5],
            "kind": "text",
        }
        for b in text_boxes
    ]

    for mb in merged_xyxy:
        dup = False
        for c in candidates:
            if bbox_iou(mb, c["bbox"]) > 0.85:
                dup = True
                break
        if not dup:
            candidates.append({"bbox": mb, "text": "", "kind": "shape"})

    out = []
    for c in candidates:
        cb = clamp_bbox(c["bbox"], W, H)
        if cb is None:
            continue
        out.append({"bbox": cb, "text": c["text"], "kind": c["kind"]})

    return out

def _rebuild_children_map(elements):
    children_map = {}
    id_set = {e["id"] for e in elements}
    for e in elements:
        parent = e.get("parent", "root")
        if parent not in id_set:
            parent = "root"
        children_map.setdefault(parent, []).append(e)
    return children_map

def _infer_layout_axis(siblings):
    if len(siblings) < 2:
        return "none"
    centers_x = [bbox_center(s["bbox"])[0] for s in siblings]
    centers_y = [bbox_center(s["bbox"])[1] for s in siblings]
    span_x = max(centers_x) - min(centers_x)
    span_y = max(centers_y) - min(centers_y)
    return "row" if span_x >= span_y else "column"

def _repair_tree_structure(tree):
    elements = tree.get("elements", [])
    if not elements:
        return tree

    by_id = {e["id"]: e for e in elements}
    image_box = [0, 0, tree["image_size"]["w"], tree["image_size"]["h"]]

    # Reassign parent when child is mostly outside its current parent.
    for e in elements:
        if e["id"] == "root":
            continue
        parent_id = e.get("parent", "root")
        parent_bbox = image_box if parent_id == "root" else by_id.get(parent_id, {}).get("bbox", image_box)
        if bbox_contains_ratio(parent_bbox, e["bbox"]) >= 0.6:
            continue

        best_parent = "root"
        best_area = bbox_area(image_box)
        for cand in elements:
            if cand["id"] == e["id"]:
                continue
            contain_ratio = bbox_contains_ratio(cand["bbox"], e["bbox"])
            if contain_ratio < 0.6:
                continue
            area = bbox_area(cand["bbox"])
            if area < best_area:
                best_area = area
                best_parent = cand["id"]
        e["parent"] = best_parent

    # Drop tiny container nodes that are unlikely to be meaningful groups.
    min_container_area = max(200, int(0.0006 * tree["image_size"]["w"] * tree["image_size"]["h"]))
    to_drop = set()
    for e in elements:
        if e.get("type") != "container":
            continue
        if bbox_area(e["bbox"]) < min_container_area:
            to_drop.add(e["id"])

    if to_drop:
        for e in elements:
            if e.get("parent") in to_drop:
                e["parent"] = "root"
        elements = [e for e in elements if e["id"] not in to_drop]
        tree["elements"] = elements
        by_id = {e["id"]: e for e in elements}

    # Remove strong sibling overlaps by promoting one sibling under the other.
    children_map = _rebuild_children_map(elements)
    for parent_id, siblings in list(children_map.items()):
        if len(siblings) < 2:
            continue
        for i in range(len(siblings)):
            a = siblings[i]
            for j in range(i + 1, len(siblings)):
                b = siblings[j]
                inter = bbox_intersection_area(a["bbox"], b["bbox"])
                small_area = float(max(1, min(bbox_area(a["bbox"]), bbox_area(b["bbox"]))))
                overlap_ratio = inter / small_area
                if overlap_ratio < 0.45:
                    continue
                a_contains_b = bbox_contains_ratio(a["bbox"], b["bbox"]) >= 0.65
                b_contains_a = bbox_contains_ratio(b["bbox"], a["bbox"]) >= 0.65
                if a_contains_b and bbox_area(a["bbox"]) >= bbox_area(b["bbox"]):
                    b["parent"] = a["id"]
                elif b_contains_a and bbox_area(b["bbox"]) >= bbox_area(a["bbox"]):
                    a["parent"] = b["id"]

    # Infer row/column ordering among siblings via reading_index.
    children_map = _rebuild_children_map(tree["elements"])
    for _, siblings in children_map.items():
        axis = _infer_layout_axis(siblings)
        if axis == "none":
            continue
        if axis == "row":
            ordered = sorted(siblings, key=lambda e: (bbox_center(e["bbox"])[0], bbox_center(e["bbox"])[1]))
        else:
            ordered = sorted(siblings, key=lambda e: (bbox_center(e["bbox"])[1], bbox_center(e["bbox"])[0]))
        for idx, node in enumerate(ordered):
            node["reading_index"] = idx

    return tree

def summarize_layout_groups(tree):
    elements = tree.get("elements", [])
    sections = []
    repeated = []
    children_map = _rebuild_children_map(elements)
    root_children = children_map.get("root", [])

    for node in sorted(root_children, key=lambda e: e.get("reading_index", 10**9))[:8]:
        axis = _infer_layout_axis(children_map.get(node["id"], []))
        sections.append({
            "id": node["id"],
            "type": node.get("type", "unknown"),
            "bbox": node.get("bbox", []),
            "layout_axis": axis,
        })

    # Basic repeated-card heuristic by sibling size similarity.
    for parent_id, siblings in children_map.items():
        if len(siblings) < 3:
            continue
        areas = [bbox_area(s["bbox"]) for s in siblings]
        avg_area = sum(areas) / len(areas)
        if avg_area <= 0:
            continue
        close = sum(1 for a in areas if abs(a - avg_area) / avg_area < 0.25)
        if close >= 3:
            repeated.append({"parent": parent_id, "count": close})

    type_hist = {}
    for e in elements:
        t = e.get("type", "unknown")
        type_hist[t] = type_hist.get(t, 0) + 1

    return {
        "num_elements": len(elements),
        "type_histogram": type_hist,
        "top_sections": sections,
        "repeated_groups": repeated[:6],
    }

def html_structure_penalty(html: str) -> float:
    lower = (html or "").lower()
    abs_count = lower.count("absolute")
    div_count = max(1, lower.count("<div"))
    # Penalize excessive absolute-position usage.
    abs_ratio = abs_count / div_count
    penalty = 0.0
    if abs_ratio > 0.55:
        penalty += (abs_ratio - 0.55) * 120.0
    # Penalize obvious missing major sections.
    if not any(tag in lower for tag in ["<header", "<main", "<section", "<footer", "nav"]):
        penalty += 12.0
    return penalty

def render_html_snippet_to_image(html_snippet, target_size):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Render Candidate</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body>
        [CODE]
    </body>
    </html>
    """
    page_html = html_template.replace("[CODE]", html_snippet or "")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": int(target_size[0]), "height": int(target_size[1])})
        page.set_content(get_placeholder(page_html))
        screenshot_data = take_screenshot_pw(page)
        browser.close()
    return Image.open(io.BytesIO(screenshot_data)).convert("RGB").resize(target_size)

def analyze_render_diff(target_img: Image.Image, rendered_img: Image.Image):
    target = np.array(target_img.convert("RGB"), dtype=np.float32)
    pred = np.array(rendered_img.convert("RGB"), dtype=np.float32)
    diff = np.abs(target - pred).mean(axis=2)
    mae = float(diff.mean())
    color_mae = float(np.mean(np.abs(target.mean(axis=(0, 1)) - pred.mean(axis=(0, 1)))))

    target_gray = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    pred_gray = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_edges = cv2.Canny(target_gray, 60, 180).astype(np.float32) / 255.0
    pred_edges = cv2.Canny(pred_gray, 60, 180).astype(np.float32) / 255.0
    edge_diff = float(np.mean(np.abs(target_edges - pred_edges)))

    h, w = diff.shape
    # Quadrant-level mismatch to produce actionable feedback.
    half_h, half_w = h // 2, w // 2
    regions = {
        "top_left": diff[:half_h, :half_w],
        "top_right": diff[:half_h, half_w:],
        "bottom_left": diff[half_h:, :half_w],
        "bottom_right": diff[half_h:, half_w:],
    }
    region_scores = {k: float(v.mean()) for k, v in regions.items()}
    worst_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    worst_region_names = [x[0] for x in worst_regions]

    threshold = max(22.0, mae * 1.2)
    mismatch_ratio = float((diff > threshold).sum()) / float(max(1, h * w))

    return {
        "mae": mae,
        "color_mae": color_mae,
        "edge_diff": edge_diff,
        "region_scores": region_scores,
        "worst_regions": worst_region_names,
        "mismatch_ratio": mismatch_ratio,
    }

def visual_fidelity_score(diff_info, html_code=""):
    # Lower is better.
    structure = html_structure_penalty(html_code)
    return (
        0.58 * float(diff_info.get("mae", 0.0))
        + 0.20 * float(diff_info.get("color_mae", 0.0))
        + 22.0 * float(diff_info.get("edge_diff", 0.0))
        + 28.0 * float(diff_info.get("mismatch_ratio", 0.0))
        + 0.55 * structure
    )

def build_repair_prompt(base_prompt, current_html, diff_info):
    region_text = ", ".join(diff_info.get("worst_regions", [])) or "mixed regions"
    region_scores = json.dumps(diff_info.get("region_scores", {}), indent=2)
    return f"""
{base_prompt}

You previously generated HTML, but rendered comparison shows mismatches.
Repair priorities:
- Focus first on these worst mismatching regions: {region_text}
- Improve section grouping and row/column alignment using flex/grid.
- Match screenshot-faithful spacing, typography, colors, border radii, and shadows.
- Preserve visible text and avoid collapsing unrelated content.
- Correct font family, font weight, and line-height to visually match the screenshot.
- Correct exact element placement/alignment (left/right/center) and inter-element spacing.
- Avoid excessive absolute positioning except overlays.

Render mismatch metrics:
- global_mae: {diff_info.get("mae", 0.0):.2f}
- color_mae: {diff_info.get("color_mae", 0.0):.2f}
- edge_diff: {diff_info.get("edge_diff", 0.0):.3f}
- mismatch_ratio: {diff_info.get("mismatch_ratio", 0.0):.3f}
- region_scores: {region_scores}

Current HTML to repair:
```html
{current_html}
```

Return only repaired HTML in ```html``` block.
"""

def build_skeleton_prompt(base_prompt, layout_summary, tree_json, img_size):
    return f"""
{base_prompt}

Stage-1 task (structure only):
Generate a semantic HTML skeleton that captures:
- major sections (header/main/section/footer),
- parent-child grouping,
- repeated list/card structures,
- row/column relations using flex/grid.

Important:
- Keep all visible text content.
- Use minimal styling classes only for structure.
- Avoid detailed colors/shadows/typography fine tuning in this stage.
- Avoid absolute positioning except for explicit overlays.

SCREENSHOT SIZE: {img_size[0]} x {img_size[1]}

DERIVED LAYOUT SUMMARY:
```json
{layout_summary}
```

ELEMENT TREE JSON:
```json
{tree_json}
```

Return only the stage-1 HTML in ```html``` block.
"""

def build_style_prompt(base_prompt, skeleton_html, layout_summary, tree_json, img_size):
    return f"""
{base_prompt}

Stage-2 task (style refinement):
You are given a structural skeleton. Preserve its section/group hierarchy and refine visual fidelity.

Requirements:
- Keep the same semantic layout hierarchy unless absolutely necessary.
- Improve spacing, typography scale, colors, borders, radius, and shadows to match screenshot.
- Match font family/weight/line-height and heading hierarchy to screenshot.
- Match background colors and contrast first, then component-level colors.
- Match element placement precisely (alignment, padding, margins, gaps).
- Keep flex/grid structure for responsive groups.
- Use absolute positioning only for isolated overlays.
- Do not delete major sections or merge unrelated text blocks.
- Use Tailwind classes (including arbitrary values like text-[14px], leading-[20px], px-[18px]) when needed for precise fidelity.

SCREENSHOT SIZE: {img_size[0]} x {img_size[1]}

DERIVED LAYOUT SUMMARY:
```json
{layout_summary}
```

ELEMENT TREE JSON:
```json
{tree_json}
```

Current skeleton HTML:
```html
{skeleton_html}
```

Return only refined final HTML in ```html``` block.
"""

def build_legacy_refine_prompt(current_html):
    return f"""Here is a prototype image of a webpage. I have an draft HTML file that contains most of the elements and their correct positions, but it has *inaccurate background*, and some missing or wrong elements. Please compare the draft and the prototype image, then revise the draft implementation. Return a single piece of accurate HTML+tail-wind CSS code to reproduce the website. Use "placeholder.png" to replace the images. Respond with the content of the HTML+tail-wind CSS code. The current implementation I have is:

{current_html}
"""

def force_placeholder_png(html):
    if not html:
        return html
    # Force every <img ... src="..."> to use placeholder.png.
    html = re.sub(
        r'(<img\b[^>]*\bsrc\s*=\s*)(["\'])(.*?)\2',
        r'\1"placeholder.png"',
        html,
        flags=re.IGNORECASE,
    )
    # Force CSS background-image url(...) references to placeholder.png.
    html = re.sub(
        r'background-image\s*:\s*url\(([^)]*)\)',
        'background-image: url("placeholder.png")',
        html,
        flags=re.IGNORECASE,
    )
    return html

def normalize_html_snippet(html):
    """
    Normalize model output into a clean body-ready snippet.
    Prevents nested full-document wrappers from hurting render fidelity.
    """
    if not html:
        return html

    text = html.strip()
    # Strip markdown fences if the model returns fenced code.
    text = re.sub(r"^```html\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If a full HTML document is returned, keep body content only.
    body_matches = re.findall(r"<body\b[^>]*>([\s\S]*?)</body>", text, flags=re.IGNORECASE)
    if body_matches:
        text = body_matches[-1].strip()
    else:
        # Fallback: drop doctype/html/head wrappers if present.
        text = re.sub(r"<!DOCTYPE[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?html\b[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<head\b[\s\S]*?</head>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?body\b[^>]*>", "", text, flags=re.IGNORECASE)
        text = text.strip()

    # Re-run body extraction once in case there are nested wrappers.
    nested_body = re.findall(r"<body\b[^>]*>([\s\S]*?)</body>", text, flags=re.IGNORECASE)
    if nested_body:
        text = nested_body[-1].strip()

    return text

_LAST_ELEMENT_TREE_CALL_TS = 0.0

def _extract_first_balanced_json_object(text: str):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def _normalize_json_like_text(raw_text: str):
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    block = _extract_first_balanced_json_object(text)
    if block:
        text = block
    text = text.replace("\ufeff", "")
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\bInfinity\b", "null", text)
    text = re.sub(r"\b-Infinity\b", "null", text)
    # Remove trailing commas before object/array close.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()

def _normalize_json_like_text_relaxed(raw_text: str):
    text = _normalize_json_like_text(raw_text)
    # Quote unquoted object keys: { key: ... } or , key: ...
    text = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)
    # Convert single-quoted strings to double-quoted strings (best-effort).
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    # Remove trailing commas again after transformations.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()

def _safe_json_loads_from_llm(raw_text: str):
    normalized = _normalize_json_like_text(raw_text)
    return json.loads(normalized)

def _safe_json_loads_from_llm_relaxed(raw_text: str):
    normalized = _normalize_json_like_text_relaxed(raw_text)
    return json.loads(normalized)

def _repair_json_with_llm(client, broken_text):
    repair_prompt = f"""
You will receive invalid JSON that should represent a UI element tree.
Rewrite it into STRICT valid JSON only.

Required top-level keys:
- image_size: object with w, h
- elements: list of objects with keys:
  id, type, bbox, text, parent, reading_index

Rules:
- Use double quotes for all keys/strings.
- No trailing commas.
- No comments.
- Output ONLY JSON.

Invalid JSON input:
```text
{broken_text}
```
"""
    _throttle_element_tree_calls(min_interval_sec=1.2)
    resp = llm_call_with_retry(
        lambda: client.chat.completions.create(
            model=MODEL_ELEMENT_TREE,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0,
            max_tokens=4000,
        ),
        provider_name="OpenAI",
    )
    content = resp.choices[0].message.content
    return content if isinstance(content, str) else str(content)


def _throttle_element_tree_calls(min_interval_sec=1.2):
    """
    Keep a minimum gap between element-tree model calls to avoid burst throttling.
    """
    global _LAST_ELEMENT_TREE_CALL_TS
    now = time.time()
    elapsed = now - _LAST_ELEMENT_TREE_CALL_TS
    if elapsed < min_interval_sec:
        time.sleep(min_interval_sec - elapsed)
    _LAST_ELEMENT_TREE_CALL_TS = time.time()


def get_element_tree_raw(image_path, api_key):
    client = OpenAI(api_key=api_key)

    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    img_b64 = encode_image(img)

    def _openai_request():
        _throttle_element_tree_calls(min_interval_sec=1.2)
        return client.chat.completions.create(
            model=MODEL_ELEMENT_TREE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_ELEMENT_TREE},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000,
        )

    resp = llm_call_with_retry(_openai_request, provider_name="OpenAI")

    content = resp.choices[0].message.content
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            else:
                chunks.append(str(item))
        text = "".join(chunks).strip()
    else:
        text = str(content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty response while JSON was requested.")
    try:
        tree = _safe_json_loads_from_llm(text)
    except Exception:
        # One extra retry is cheap and often resolves malformed edge outputs.
        resp_retry = llm_call_with_retry(_openai_request, provider_name="OpenAI")
        content_retry = resp_retry.choices[0].message.content
        retry_text = content_retry if isinstance(content_retry, str) else str(content_retry)
        try:
            tree = _safe_json_loads_from_llm(retry_text)
        except Exception:
            # Relaxed parser pass (unquoted keys / single-quote strings).
            try:
                tree = _safe_json_loads_from_llm_relaxed(retry_text)
            except Exception:
                # Final fallback: ask model to repair invalid JSON text.
                repaired = _repair_json_with_llm(client, retry_text)
                try:
                    tree = _safe_json_loads_from_llm(repaired)
                except Exception:
                    tree = _safe_json_loads_from_llm_relaxed(repaired)
    w, h = img.size
    tree["image_size"] = {"w": w, "h": h}
    return tree


def replace_only_bboxes_keep_gemini(tree: dict, candidates: list):
    elems = tree.get("elements", [])
    if not elems:
        return tree

    text_idxs = [i for i, c in enumerate(candidates) if c["kind"] == "text"]
    shape_idxs = [i for i, c in enumerate(candidates) if c["kind"] != "text"]
    all_idxs = list(range(len(candidates)))
    unused = set(all_idxs)

    for e in elems:
        g_bbox = e.get("bbox")
        if not g_bbox or len(g_bbox) != 4:
            continue

        g_type = (e.get("type") or "").lower().strip()
        g_text = normalize_text(e.get("text", ""))
        wants_text_pool = (g_type == "text") or bool(g_text)

        preferred_pool = text_idxs if wants_text_pool else shape_idxs
        fallback_pool = shape_idxs if wants_text_pool else text_idxs

        def best_match(pool):
            best_i, best_score = None, -1e18
            for i in pool:
                if i not in unused:
                    continue

                c = candidates[i]
                c_bbox = c["bbox"]

                iou_s = bbox_iou(g_bbox, c_bbox)
                dist_s = -center_dist(g_bbox, c_bbox) / 1200.0
                txt_s = token_jaccard(g_text, c["text"]) if g_text else 0.0

                if wants_text_pool:
                    score = 4.0 * txt_s + 1.8 * iou_s + dist_s
                else:
                    score = 2.4 * iou_s + dist_s + 0.2 * txt_s

                if score > best_score:
                    best_score = score
                    best_i = i
            return best_i

        best_i = best_match(preferred_pool)
        if best_i is None:
            best_i = best_match(fallback_pool)
        if best_i is None:
            best_i = best_match(all_idxs)

        if best_i is not None:
            e["bbox"] = candidates[best_i]["bbox"]
            unused.remove(best_i)

    tree = _repair_tree_structure(tree)
    return tree

def generate_fixed_element_tree_json(image_path, api_key=None, min_ocr_conf=0.4, tree_raw=None):
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    if tree_raw is None:
        if api_key is None:
            raise ValueError("api_key is required when tree_raw is not provided")
        tree_raw = get_element_tree_raw(img, api_key)

    candidates = detect_ui_candidates(img, min_conf=min_ocr_conf)
    tree_fixed = replace_only_bboxes_keep_gemini(tree_raw, candidates)
    return sanitize_element_tree(tree_fixed, image_size=img.size)


class DCGenElementTree:
    """
    Generate final HTML from:
    - original screenshot
    - fixed element-tree JSON
    """

    def __init__(self, image, element_tree, prompt_html=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        self.img = image
        self.tree = sanitize_element_tree(
            load_element_tree(element_tree),
            image_size=image.size
        )
        self.nested_tree = element_tree_to_nested(self.tree)
        self.prompt_html = prompt_html or self.default_prompt()
        self.code = None
        self.raw_response = None

    @staticmethod
    def default_prompt():
        return """
You are given:
1) a UI screenshot
2) a fixed JSON element tree with approximate layout hierarchy and element bounding boxes

Your task:
Generate the COMPLETE HTML for the ENTIRE page using Tailwind CSS.

Rules:
- Use HTML + Tailwind CSS utility classes.
- Return ONLY HTML wrapped in ```html ... ```
- Output the full page content, not just the first section.
- Make sure all opened tags are closed.
- The screenshot is the source of truth for visual grouping, style, spacing, colors, and typography.
- The JSON tree is approximate and should be used as a hint for text content, element identity, and approximate localization.
- If tree hierarchy conflicts with the screenshot, preserve the screenshot's visible grouping.
- Do not invent extra major elements not present in the tree.
- Build semantic sections/components first (header, hero, cards, lists, forms, footer) and then refine styles.
- Prefer flex/grid for rows, columns, and repeated structures.
- Use absolute positioning only for isolated overlays or decorative floating items.
- Text in the screenshot should appear in the HTML.
- If an element type is:
  - container: use div
  - text: use p/span/div
  - button: use button
  - input: use input or div styled like input
  - image: always use div/img with src="placeholder.png" (do not use real URLs or data URIs)
  - icon: use simple unicode/icon-like placeholder if needed
- The output should be ready to place inside <body>.

Do not:
- Flatten the entire page into one absolute-positioned wrapper.
- Merge unrelated texts into one paragraph.
- Ignore repeated card/list structure.
"""

    def build_question(self):
        tree_json = json.dumps(self.tree, indent=2, ensure_ascii=False)
        layout_summary = json.dumps(summarize_layout_groups(self.tree), indent=2, ensure_ascii=False)

        question = f"""
{self.prompt_html}

SCREENSHOT SIZE:
{self.img.size[0]} x {self.img.size[1]}

DERIVED LAYOUT SUMMARY:
```json
{layout_summary}```

ELEMENT TREE JSON:
```json
{tree_json}```
"""
        return question
    def _extract_html(self, response):
        pure_code = re.findall(r"```html\s*([\s\S]*?)```", response or "")
        html = pure_code[0].strip() if pure_code else (response or "").strip()
        return normalize_html_snippet(html)

    def _validate_html_candidate(self, html):
        if len(html) < 500:
            return False
        if html_structure_penalty(html) > 30:
            return False
        return True

    def _legacy_refine_once(self, bot, html, image_b64, num_generations):
        legacy_prompt = build_legacy_refine_prompt(html)
        legacy_response = bot.try_ask(
            legacy_prompt,
            image_b64,
            num_generations=max(1, num_generations // 2)
        )
        if legacy_response is None:
            return html
        legacy_html = self._extract_html(legacy_response)
        if self._validate_html_candidate(legacy_html):
            return legacy_html
        return html

    def generate_code(self, bot, num_generations=4, max_retries=3):
        tree_json = json.dumps(self.tree, indent=2, ensure_ascii=False)
        layout_summary = json.dumps(summarize_layout_groups(self.tree), indent=2, ensure_ascii=False)
        image_b64 = encode_image(self.img)

        skeleton_prompt = build_skeleton_prompt(
            self.prompt_html, layout_summary, tree_json, self.img.size
        )

        skeleton_html = None
        for attempt in range(max_retries):
            skeleton_resp = bot.try_ask(
                skeleton_prompt,
                image_b64,
                num_generations=max(2, num_generations // 2)
            )
            if skeleton_resp is None:
                print(f"Skeleton attempt {attempt + 1}: no response")
                continue
            skeleton_html = self._extract_html(skeleton_resp)
            if len(skeleton_html) >= 250:
                break

        if not skeleton_html:
            raise RuntimeError("Model failed to produce stage-1 skeleton HTML.")

        style_prompt = build_style_prompt(
            self.prompt_html, skeleton_html, layout_summary, tree_json, self.img.size
        )

        for attempt in range(max_retries):
            response = bot.try_ask(
                style_prompt,
                image_b64,
                num_generations=num_generations
            )
            if response is None:
                print(f"Style attempt {attempt + 1}: no response")
                continue

            self.raw_response = response
            html = self._extract_html(response)
            if not self._validate_html_candidate(html):
                print(f"Style attempt {attempt + 1}: candidate rejected, retrying...")
                continue

            # Keep the new two-stage flow, then run legacy prompt_root-style refinement.
            html = self._legacy_refine_once(bot, html, image_b64, num_generations)
            self.code = force_placeholder_png(normalize_html_snippet(html))
            return self.code

        raise RuntimeError("Model returned incomplete HTML after two-stage generation retries.")

    def repair_code_once(self, bot, html_code, num_generations=2):
        rendered = render_html_snippet_to_image(html_code, self.img.size)
        diff_info = analyze_render_diff(self.img, rendered)
        base_score = visual_fidelity_score(diff_info, html_code)

        # Skip repair if already close enough.
        if diff_info["mae"] < 18 and diff_info["mismatch_ratio"] < 0.12:
            return html_code, diff_info

        repair_question = build_repair_prompt(self.prompt_html, html_code, diff_info)
        response = bot.try_ask(
            repair_question,
            encode_image(self.img),
            num_generations=num_generations
        )
        if response is None:
            return html_code, diff_info

        pure_code = re.findall(r"```html\s*([\s\S]*?)```", response)
        repaired_html = pure_code[0].strip() if pure_code else response.strip()
        repaired_html = normalize_html_snippet(repaired_html)
        if len(repaired_html) < 250:
            return html_code, diff_info

        repaired_render = render_html_snippet_to_image(repaired_html, self.img.size)
        repaired_diff = analyze_render_diff(self.img, repaired_render)
        repaired_score = visual_fidelity_score(repaired_diff, repaired_html)

        if repaired_score <= base_score:
            return repaired_html, repaired_diff
        return html_code, diff_info

    def repair_code_multi_round(self, bot, html_code, max_rounds=3, num_generations=2, min_delta=0.6):
        current_html = html_code
        current_render = render_html_snippet_to_image(current_html, self.img.size)
        current_diff = analyze_render_diff(self.img, current_render)
        current_score = visual_fidelity_score(current_diff, current_html)
        best_html, best_diff, best_score = current_html, current_diff, current_score

        for round_idx in range(max_rounds):
            repaired_html, repaired_diff = self.repair_code_once(
                bot, current_html, num_generations=num_generations
            )
            repaired_score = visual_fidelity_score(repaired_diff, repaired_html)

            # Keep global best candidate.
            if repaired_score < best_score:
                best_html, best_diff, best_score = repaired_html, repaired_diff, repaired_score

            # Early stop if round does not improve enough.
            delta = current_score - repaired_score
            if delta < min_delta:
                break

            current_html, current_diff, current_score = repaired_html, repaired_diff, repaired_score

            # Already very close; no need for additional rounds.
            if (
                current_diff["mae"] < 14
                and current_diff["mismatch_ratio"] < 0.09
                and current_diff.get("color_mae", 99.0) < 10
                and current_diff.get("edge_diff", 1.0) < 0.12
            ):
                break

        return best_html, best_diff
    # def generate_code(self, bot, num_generations=1):
#     question = self.build_question()
#     response = bot.try_ask(
#         question,
#         encode_image(self.img),
#         num_generations=num_generations
#     )

#     if response is None:
#         raise RuntimeError("Model failed to generate HTML.")

#     self.raw_response = response

#     pure_code = re.findall(r"```html\s*([\s\S]*?)```", response)
#     if pure_code:
#         self.code = pure_code[0].strip()
#     else:
#         self.code = response.strip()

#     return self.code

def wrap_html_document(html_code):
      return f"""<!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>DCGen Output</title>
      <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body class="m-0 p-0">
  {html_code}
  </body>
  </html>
  """

def process_image_full_pipeline(image_path, sample_out_dir, api_key, bot, min_ocr_conf=0.4):
    """
    Full pipeline:
    image -> raw tree -> OCR/NMS bbox fix -> fixed json -> html
    """
    os.makedirs(sample_out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    raw_json_path = os.path.join(sample_out_dir, f"{base}_tree_raw.json")
    fixed_json_path = os.path.join(sample_out_dir, f"{base}_tree_fixed.json")
    html_path = os.path.join(sample_out_dir, "output.html")

    tree_raw = get_element_tree_raw(image_path, api_key)
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(tree_raw, f, indent=2, ensure_ascii=False)

    tree_fixed = generate_fixed_element_tree_json(
        image_path,
        api_key=None,
        min_ocr_conf=min_ocr_conf,
        tree_raw=tree_raw
    )
    with open(fixed_json_path, "w", encoding="utf-8") as f:
        json.dump(tree_fixed, f, indent=2, ensure_ascii=False)

    gen = DCGenElementTree(image_path, fixed_json_path)
    html_snippet = gen.generate_code(bot, num_generations=4)

    # Multi-round render-compare-repair with early stopping.
    repaired_html, _ = gen.repair_code_multi_round(
        bot,
        html_snippet,
        max_rounds=3,
        num_generations=2,
        min_delta=0.6
    )
    html_snippet = force_placeholder_png(repaired_html)
    full_html = wrap_html_document(html_snippet)

    placeholder_src = os.path.join(os.path.dirname(__file__), "placeholder.png")
    if os.path.exists(placeholder_src):
        shutil.copy2(placeholder_src, os.path.join(sample_out_dir, "placeholder.png"))

    with open(html_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(full_html)

    return {
        "raw_json_path": raw_json_path,
        "fixed_json_path": fixed_json_path,
        "html_path": html_path,
    }