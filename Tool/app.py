from flask import Flask, render_template, request, jsonify
import re
import sys
import os
import tempfile
import base64
import importlib.util
from io import BytesIO
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_SCRIPTS = os.path.join(_HERE, '..', 'scripts')
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPTS)

from utils import ImgSegmentation, DCGenGrid, GPT4, encode_image
from utils_method2 import YOLOSegmentation
from experiments import Method2Grid, prompt_method_2

app = Flask(__name__)
_METHOD1_UTILS = None

_PROMPT_DIRECT = (
    'Here is a prototype image of a webpage. Return a single piece of HTML and '
    'Tailwind CSS code to reproduce exactly the website. Use "placeholder.png" to '
    'replace the images. Pay attention to things like size, text, position, and color '
    'of all the elements, as well as the overall layout. Respond with the content of '
    'the HTML + Tailwind CSS code.'
)

_PROMPT_DCGEN_LEAF = (
    "Here is a prototype image of a container. Please fill a single piece of HTML and "
    "Tailwind CSS code to reproduce exactly the given container. Use 'placeholder.png' "
    "to replace the images. Pay attention to things like size, text, and color of all "
    "the elements, as well as the background color and layout. Here is the code for you "
    "to fill in:\n    <div>\n    Your code here\n    </div>\n    Respond with only the "
    "code inside the <div> tags."
)

_PROMPT_DCGEN_ROOT = (
    "Here is a prototype image of a webpage. I have a draft HTML file that contains "
    "most of the elements and their correct positions, but it has *inaccurate background*, "
    "and some missing or wrong elements. Please compare the draft and the prototype image, "
    "then revise the draft implementation. Return a single piece of accurate HTML + "
    'Tailwind CSS code to reproduce the website. Use "placeholder.png" to replace the '
    "images. Respond with the content of the HTML + Tailwind CSS code. The current "
    "implementation I have is:\n\n[CODE]"
)


def _read_api_key(key_path):
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"API key file not found: {key_path}")
    with open(key_path, 'r', encoding='utf-8') as f:
        key = f.read().strip()
    if not key:
        raise ValueError(f"API key file is empty: {key_path}")
    return key


def _load_method1_utils():
    global _METHOD1_UTILS
    if _METHOD1_UTILS is not None:
        return _METHOD1_UTILS

    method1_utils_path = os.path.join(_ROOT, 'Element_Tree_Trial', 'utils.py')
    if not os.path.exists(method1_utils_path):
        raise FileNotFoundError(
            f"Method 1 utils not found at {method1_utils_path}"
        )

    spec = importlib.util.spec_from_file_location('element_tree_trial_utils', method1_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load Method 1 module spec.")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _METHOD1_UTILS = mod
    return _METHOD1_UTILS


@app.route('/')
def index():
    return render_template('interface.html')


@app.route('/example_image', methods=['POST'])
def example():
    with open('./static/example.png', 'rb') as f:
        image = f.read()
    base64_image = base64.b64encode(image).decode('utf-8')
    return jsonify({"image": f"data:image/png;base64,{base64_image}"})


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    image_base64 = data.get('image', '')
    method = data.get('method', 'direct')

    try:
        bot = GPT4(os.path.join(_ROOT, 'keys', 'key.txt'), model='gpt-4o')
        if not image_base64 or ',' not in image_base64:
            return jsonify({"error": "No valid image payload provided."}), 400
        img = Image.open(BytesIO(base64.b64decode(image_base64.split(',', 1)[1])))

        # Direct 
        if method == 'direct':
            response_text = bot.ask(_PROMPT_DIRECT, encode_image(img))
            pure = re.findall(r'```html([^`]+)```', response_text)
            html = pure[0] if pure else response_text
            return jsonify({"html": html})

        # DCGen 
        elif method == 'dcgen':
            depth       = int(data.get('dcgen_depth', 2))
            prompt_leaf  = data.get('prompt_leaf',  _PROMPT_DCGEN_LEAF)
            prompt_final = data.get('prompt_final', _PROMPT_DCGEN_ROOT)
            img_seg = ImgSegmentation(img, max_depth=depth)
            grid = DCGenGrid(img_seg, prompt_seg=prompt_leaf, prompt_refine=prompt_final)
            grid.generate_code(bot, multi_thread=True)
            return jsonify({"html": grid.code})

        # Method2
        elif method == 'method2':
            yolo_rel      = data.get('method2_yolo_model', 'models/yolo11n_400_best.pt')
            yolo_path     = yolo_rel if os.path.isabs(yolo_rel) else os.path.join(_ROOT, yolo_rel)
            use_composite = bool(data.get('method2_use_composite', False))
            use_vlm_dedup = bool(data.get('method2_use_vlm_dedup', True))
            use_bg_color  = bool(data.get('method2_use_bg_color',  False))

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name

            save_dir = tempfile.mkdtemp()
            try:
                img_seg = YOLOSegmentation.from_yolo_model(
                    tmp_path,
                    model_path=yolo_path,
                    save_dir=save_dir,
                    iou_threshold=0.7,
                    containment_threshold=0.7,
                )
                # Apply toggles as class-level overrides before instantiation
                Method2Grid._USE_BG_COLOR        = use_bg_color
                Method2Grid._USE_COMPOSITE_IMAGE = use_composite
                Method2Grid._USE_VLM_DEDUP       = use_vlm_dedup

                grid = Method2Grid(
                    img_seg,
                    prompt_seg=prompt_method_2['prompt_seg'],
                    prompt_refine=prompt_method_2['prompt_refine'],
                    artifact_dir=save_dir,
                )
                grid.generate_code(bot, multi_thread=True)
                return jsonify({"html": grid.code})
            finally:
                os.unlink(tmp_path)

        # Method1 (Element Tree Trial)
        elif method == 'method1':
            method1_min_ocr_conf = float(data.get('method1_min_ocr_conf', 0.4))
            method1_utils = _load_method1_utils()
            key_path = os.path.join(_ROOT, 'keys', 'key.txt')
            api_key = _read_api_key(key_path)
            method1_bot = method1_utils.GPT4(key_path, model='gpt-4o')

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name

            sample_out_dir = tempfile.mkdtemp()
            try:
                outputs = method1_utils.process_image_full_pipeline(
                    image_path=tmp_path,
                    sample_out_dir=sample_out_dir,
                    api_key=api_key,
                    bot=method1_bot,
                    min_ocr_conf=method1_min_ocr_conf,
                )
                html_path = outputs.get("html_path")
                if not html_path or not os.path.exists(html_path):
                    raise RuntimeError("Method 1 did not produce output HTML.")
                with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html = f.read()
                return jsonify({"html": html})
            finally:
                os.unlink(tmp_path)

        else:
            return jsonify({"error": f"Unknown method: {method}"}), 400

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)