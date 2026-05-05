import os, sys, json
sys.path.append('..')
from utils import GPT4, encode_image, get_driver, take_screenshot
from experiments import method2_segmented, get_dir_list
from evaluate import clip_experiment, code_sim_experiment
from playwright.sync_api import sync_playwright
from PIL import Image
from tqdm import tqdm
import numpy as np
import datetime
import time
import shutil


IMG_DIR      = "../data/111_eval_data"          # 111 reference PNGs
YOLO_MODEL   = "../models/yolo11n_400_best.pt"
OUT_DIR      = "../data/experiments/exp_1/method2_output"   # one sub-folder per image
RESULTS_DIR  = "../data/experiments/exp_1/results/exp_method2"
os.makedirs(RESULTS_DIR, exist_ok=True)


bot = GPT4("../keys/key.txt", model="gpt-4o")

#Generate HTML for all images
filelist = get_dir_list(IMG_DIR, exclude=["placeholder", "bbox"])
for img_path in tqdm(filelist, desc="Generating HTML"):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_html = os.path.join(OUT_DIR, stem, f"{stem}.html")
    if os.path.exists(out_html):
        continue  # already done — resumable
    try:
        # save_path is the BASE dir; method2_segmented appends /{stem}/ internally
        method2_segmented(     
            bot=bot,
            img_path=img_path,
            layout=YOLO_MODEL,
            save_path=OUT_DIR,         # ← pass base dir, NOT a .html path
            multi_thread=False,   # ← sequential segments: gentler on TPM
        )
    except Exception as e:
        print(f"[FAILED] {stem}: {e}")
    time.sleep(15)   # ← wait 15s between images to let TPM quota reset

# Screenshot all generated HTMLs (needed for CLIP)
def screenshot_all(out_dir, replace=False):
    html_files = []
    for stem in os.listdir(out_dir):
        html = os.path.join(out_dir, stem, f"{stem}.html")
        png  = os.path.join(out_dir, stem, f"{stem}.png")
        if os.path.isfile(html) and (replace or not os.path.exists(png)):
            html_files.append((html, png))

    driver = get_driver(string="<html></html>", window_size=(1920, 1080))
    for html_path, png_path in tqdm(html_files, desc="Screenshots"):
        try:
            driver.get("file://" + os.path.abspath(html_path))
            take_screenshot(driver, png_path)
        except Exception as e:
            print(f"Screenshot failed for {html_path}: {e}")
    driver.quit()

screenshot_all(OUT_DIR)

# Flatten generated PNGs into one dir for evaluate.py
FLAT_PNG_DIR  = os.path.join(RESULTS_DIR, "generated_pngs")
FLAT_HTML_DIR = os.path.join(RESULTS_DIR, "generated_htmls")
# Clear first to avoid stale _p/_p_1 artifacts from previous runs
if os.path.exists(FLAT_HTML_DIR):
    shutil.rmtree(FLAT_HTML_DIR)
if os.path.exists(FLAT_PNG_DIR):
    shutil.rmtree(FLAT_PNG_DIR)
os.makedirs(FLAT_PNG_DIR,  exist_ok=True)
os.makedirs(FLAT_HTML_DIR, exist_ok=True)

# build set of valid stems from the source image list
valid_stems = {os.path.splitext(os.path.basename(f))[0] for f in filelist}
for stem in os.listdir(OUT_DIR):
    if stem not in valid_stems:          # ← skip _p, _p_1 intermediate dirs
        continue
    html = os.path.join(OUT_DIR, stem, f"{stem}.html")
    png  = os.path.join(OUT_DIR, stem, f"{stem}.png")
    if os.path.isfile(html):
        shutil.copy2(html, os.path.join(FLAT_HTML_DIR, f"{stem}.html"))
    if os.path.isfile(png):
        shutil.copy2(png,  os.path.join(FLAT_PNG_DIR,  f"{stem}.png"))
        shutil.copy2(png, os.path.join(FLAT_HTML_DIR, f"{stem}.png")) 

# Run metrics 
test_dirs_png  = {"method2": FLAT_PNG_DIR}
test_dirs_html = {"method2": FLAT_HTML_DIR}

clip_results = clip_experiment(IMG_DIR, test_dirs_png,  os.path.join(RESULTS_DIR, "exp"))
code_results = code_sim_experiment(IMG_DIR, test_dirs_html, os.path.join(RESULTS_DIR, "exp"))

# absolute paths BEFORE chdir
VISUAL_JSON = os.path.join(RESULTS_DIR, "exp_visual.json")
abs_ref_html   = os.path.abspath(IMG_DIR)
abs_test_html  = os.path.abspath(FLAT_HTML_DIR)
abs_visual_out = os.path.abspath(VISUAL_JSON)

metric_dir = os.path.abspath("./metric/Design2Code")
sys.path.insert(0, os.path.abspath("./metric"))   # parent of Design2Code package

orig_cwd = os.getcwd()
try:
    os.chdir(metric_dir)                    # required by visual_score.py
    from Design2Code.metrics.multi_processing_eval import eval as d2c_eval
    d2c_eval(
        orig_reference_dir=abs_ref_html,
        test_dirs={"method2": abs_test_html},
        output_path=abs_visual_out,
    )
finally:
    os.chdir(orig_cwd) # restore CWD

# read back the written JSON
with open(VISUAL_JSON) as f:
    visual_results = json.load(f)

# Write summary log 
log_path = os.path.join(RESULTS_DIR, "summary.txt")
with open(log_path, "w", encoding="utf-8") as log:
    log.write(f"Experiment Summary\n")
    log.write(f"Run date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"Image dir: {IMG_DIR}\n")
    log.write(f"YOLO     : {YOLO_MODEL}\n")
    log.write(f"Output   : {OUT_DIR}\n")
    log.write("=" * 60 + "\n\n")

    log.write("CLIP Scores (cosine similarity, higher is better)\n")
    log.write("-" * 40 + "\n")
    for key, vals in clip_results.items():
        if vals:
            scores = list(vals.values())
            log.write(f"  Method : {key}\n")
            log.write(f"  N      : {len(scores)}\n")
            log.write(f"  Mean   : {sum(scores)/len(scores):.4f}\n")
            log.write(f"  Min    : {min(scores):.4f}\n")
            log.write(f"  Max    : {max(scores):.4f}\n")
            # per-file breakdown
            log.write(f"  Per-file:\n")
            for fname, score in sorted(vals.items()):
                log.write(f"    {fname:<30} {score:.4f}\n")
        else:
            log.write(f"  {key}: no results\n")
        log.write("\n")

    log.write("Code Similarity Scores (fuzz ratio 0-100, higher is better)\n")
    log.write("-" * 40 + "\n")
    for key, vals in code_results.items():
        if vals:
            scores = list(vals.values())
            log.write(f"  Method : {key}\n")
            log.write(f"  N      : {len(scores)}\n")
            log.write(f"  Mean   : {sum(scores)/len(scores):.2f}\n")
            log.write(f"  Min    : {min(scores):.2f}\n")
            log.write(f"  Max    : {max(scores):.2f}\n")
            log.write(f"  Per-file:\n")
            for fname, score in sorted(vals.items()):
                log.write(f"    {fname:<30} {score:.2f}\n")
        else:
            log.write(f"  {key}: no results\n")
        log.write("\n")


    SCORE_NAMES = ["Overall", "Block-Match", "Text", "Position", "Color", "CLIP(D2C)"]
    log.write("Design2Code Fine-Grained Scores (higher is better)\n")
    log.write("-" * 40 + "\n")
    for key, file_scores in visual_results.items():
        if not file_scores:
            log.write(f"  {key}: no results\n")
            continue
        arr = np.array(list(file_scores.values()))  # shape (N, 6)
        means = arr.mean(axis=0)
        log.write(f"  Method : {key}\n")
        log.write(f"  N      : {len(file_scores)}\n")
        for i, name in enumerate(SCORE_NAMES):
            log.write(f"  {name:<20}: {means[i]:.4f}\n")
        log.write(f"  Per-file (Overall | Block | Text | Pos | Color | CLIP):\n")
        for fname, scores in sorted(file_scores.items()):
            row = "  ".join(f"{s:.3f}" for s in scores)
            log.write(f"    {fname:<30} {row}\n")
        log.write("\n")

print(f"Summary written to {log_path}")