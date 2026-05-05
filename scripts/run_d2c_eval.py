# run_d2c_eval.py
# Standalone offline Design2Code evaluation.
# Requires run_experiment.py to have already completed Steps 1-3
# (HTML generation, screenshots, flattening into generated_htmls/).
import os, sys, json, re
import numpy as np


RESULTS_DIR   = "path/to/experiment/results"   # ← UPDATE THIS to your actual results dir
IMG_DIR       = "path/to/experiment/eval_data"    # ← UPDATE THIS to your actual image dir
FLAT_HTML_DIR = os.path.join(RESULTS_DIR, "generated_htmls")
VISUAL_JSON   = os.path.join(RESULTS_DIR, "exp_visual.json")


# Run Design2Code eval 
abs_ref_html   = os.path.abspath(IMG_DIR)
abs_test_html  = os.path.abspath(FLAT_HTML_DIR)
abs_visual_out = os.path.abspath(VISUAL_JSON)

metric_dir = os.path.abspath("./metric/Design2Code")
sys.path.insert(0, os.path.abspath("./metric"))

print(f"Reference HTMLs : {abs_ref_html}")
print(f"Generated HTMLs : {abs_test_html}")
print(f"Output JSON     : {abs_visual_out}")

orig_cwd = os.getcwd()
try:
    os.chdir(metric_dir)
    from Design2Code.metrics.multi_processing_eval import eval as d2c_eval
    d2c_eval(
        orig_reference_dir=abs_ref_html,
        test_dirs={"method2": abs_test_html},
        output_path=abs_visual_out,
    )
finally:
    os.chdir(orig_cwd)

# Write summary 
with open(VISUAL_JSON) as f:
    visual_results = json.load(f)

SCORE_NAMES = ["Overall", "Block-Match", "Text", "Position", "Color", "CLIP(D2C)"]
log_path = os.path.join(RESULTS_DIR, "summary_d2c.txt")

with open(log_path, "w", encoding="utf-8") as log:
    log.write("Design2Code Fine-Grained Scores (higher is better)\n")
    log.write("-" * 40 + "\n")
    for key, file_scores in visual_results.items():
        if not file_scores:
            log.write(f"  {key}: no results\n")
            continue
        arr   = np.array(list(file_scores.values()))
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