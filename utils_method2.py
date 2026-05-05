import json
import os
from collections import Counter          
from typing import Union, List

from PIL import Image


class YOLOSegmentation:
    """
    Wraps a YOLO layout.json so that DCGenGrid (utils.py) can be reused

    Constructors
    ------------
    YOLOSegmentation(img, layout)
        Build from a pre-computed layout.json file path or already-parsed dict.

    YOLOSegmentation.from_yolo_model(img, model_path, ...)
        Run a finetuned YOLO model on *img* and build the instance on the fly.
        Requires `ultralytics` to be installed.
    """

    def __init__(
        self,
        img: Union[str, Image.Image],
        layout: Union[str, dict],
        deduplicate: bool = True,
        iou_threshold: float = 0.7,
        containment_threshold: float = 0.7,
    ) -> None:
        """
        Parameters
        ----------
        img    : path to the full-page PNG screenshot, or a PIL Image.
        layout : path to layout.json, or the already-parsed dict.
        """
        if isinstance(img, str):
            img = Image.open(img)
        self.img = img  
        self.bbox = (0, 0, self.img.size[0], self.img.size[1]) 

        if isinstance(layout, str):
            with open(layout, "r", encoding="utf-8") as fh:
                layout = json.load(fh)
        self.layout = layout

        if deduplicate:
            self.layout["segments"] = self.deduplicate_segments(
                self.layout.get("segments", []),
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
                image_width=self.layout.get("image_width", 0),
                image_height=self.layout.get("image_height", 0),
            )

    # run YOLO inference and build the instance
    @classmethod
    def from_yolo_model(
        cls, # class
        img: Union[str, Image.Image],
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        containment_threshold: float = 0.7,
        save_dir: str = None,
    ) -> "YOLOSegmentation":
        """
        Run a finetuned YOLO model on *img* and return a YOLOSegmentation.

        Parameters
        ----------
        img            : full-page screenshot path or PIL Image.
        model_path     : path to the finetuned YOLO .pt weights file.
        conf_threshold : minimum confidence to keep a detection.
        save_dir       : if given, each deduplicated segment crop is saved as a
                         PNG and layout.json is written inside this directory.
                         The raw (pre-deduplication) crops are also saved to
                         save_dir/pre_dedup/ for inspection.

        Returns
        -------
        YOLOSegmentation ready to be passed directly to DCGenGrid.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for from_yolo_model(). "
                "Install it with:  pip install ultralytics"
            )

        img_path = None
        if isinstance(img, str):
            img_path = img
            pil_img = Image.open(img_path)
        else:
            pil_img = img

        model = YOLO(model_path)
        results = model(pil_img, conf=conf_threshold, iou=iou_threshold)[0]

        w, h = pil_img.size
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        raw_segments = []
        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            # clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            seg_id = f"segment_{idx + 1}"
            seg_filename = f"{seg_id}.png"

            raw_segments.append({
                "id": seg_id,
                "file": seg_filename,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
            })

        # save pre-dedup crops for inspection
        if save_dir:
            pre_dedup_dir = os.path.join(save_dir, "segments", "pre_dedup")
            os.makedirs(pre_dedup_dir, exist_ok=True)

            for seg in raw_segments:
                x1, y1, x2, y2 = seg["bbox_xyxy"]
                # raw YOLO crop (existing)
                pil_img.crop((x1, y1, x2, y2)).save(
                    os.path.join(pre_dedup_dir, seg["file"])
                )
            raw_layout = {
                "image_name": os.path.basename(img_path) if img_path else "image.png",
                "image_width": w,
                "image_height": h,
                "segments": raw_segments,
            }
            with open(os.path.join(pre_dedup_dir, "layout_raw.json"), "w", encoding="utf-8") as fh:
                json.dump(raw_layout, fh, indent=4)

        iou_threshold_for_dedup = 0.7 
        # Deduplicate
        deduped = cls.deduplicate_segments(
            raw_segments,
            iou_threshold=iou_threshold_for_dedup,
            containment_threshold=containment_threshold,
            image_width=w,
            image_height=h,
        )

        # sort top-to-bottom, left-to-right, then rename
        # segment_1 is always the topmost segment (smallest y1).
        # Visual placement in DCGenGrid is driven entirely by bbox_xyxy
        # coordinates (absolute %), so the sort does not change rendering —
        # it only makes the debug naming consistent with reading order.
        deduped.sort(key=lambda s: (s["bbox_xyxy"][1], s["bbox_xyxy"][0]))
        for i, seg in enumerate(deduped):
            seg["id"] = f"segment_{i + 1}"
            seg["file"] = f"segment_{i + 1}.png"

        layout = {
            "image_name": os.path.basename(img_path) if img_path else "image.png",
            "image_width": w,
            "image_height": h,
            "segments": deduped,
        }

        if save_dir:
            dedup_dir = os.path.join(save_dir, "segments", "dedup")
            snapped_dir = os.path.join(save_dir,  "segments", "snapped")
            os.makedirs(dedup_dir, exist_ok=True)
            os.makedirs(snapped_dir, exist_ok=True)
            for seg in deduped:
                x1, y1, x2, y2 = seg["bbox_xyxy"]
                pil_img.crop((x1, y1, x2, y2)).save(
                    os.path.join(dedup_dir, seg["file"])
                )
                pil_img.crop((0, y1, w, y2)).save(
                os.path.join(snapped_dir, seg["file"])
                )
            with open(os.path.join(save_dir, "segments", "layout.json"), "w", encoding="utf-8") as fh:
                json.dump(layout, fh, indent=4)

        return cls(pil_img, layout, deduplicate=False)  # already deduped


    # Deduplication logic
    @staticmethod
    def deduplicate_segments(
        segments: List[dict],
        iou_threshold: float = 0.7,
        containment_threshold: float = 0.7,
        image_width: int = 0,
        image_height: int = 0,
        min_area_ratio: float = 0.005,
    ) -> List[dict]:
        """
        Remove overlapping/duplicate detections.

        """
        if not segments:
            return segments

        # minimum-area filter 
        if image_width > 0 and image_height > 0:
            min_area = image_width * image_height * min_area_ratio
            segments = [
                s for s in segments
                if (s["bbox_xyxy"][2] - s["bbox_xyxy"][0])
                   * (s["bbox_xyxy"][3] - s["bbox_xyxy"][1]) >= min_area
            ]

        if len(segments) <= 1:
            return segments

        def area(seg):
            x1, y1, x2, y2 = seg["bbox_xyxy"]
            return max(0, x2 - x1) * max(0, y2 - y1)

        def intersection(a, b):
            ax1, ay1, ax2, ay2 = a["bbox_xyxy"]
            bx1, by1, bx2, by2 = b["bbox_xyxy"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0
            return (ix2 - ix1) * (iy2 - iy1)

        # Sort largest-first so large boxes act as anchors
        indexed = sorted(enumerate(segments), key=lambda x: area(x[1]), reverse=True)
        suppressed = set()

        for rank_i, (orig_i, seg_i) in enumerate(indexed):
            if orig_i in suppressed:
                continue
            for rank_j, (orig_j, seg_j) in enumerate(indexed):
                if rank_j <= rank_i or orig_j in suppressed:
                    continue

                inter = intersection(seg_i, seg_j)
                if inter == 0:
                    continue

                area_i = area(seg_i)
                area_j = area(seg_j)
                union = area_i + area_j - inter
                iou_val = inter / union if union > 0 else 0
                # seg_i is always the larger (sorted largest-first)
                containment_val = inter / area_j if area_j > 0 else 0

                if iou_val > iou_threshold:
                    # Near-duplicate: keep the LARGER box (seg_i).
                    # Confidence is intentionally ignored — it is not a
                    # reliable indicator of segmentation quality.
                    suppressed.add(orig_j)
                elif containment_val > containment_threshold:
                    # Smaller box mostly inside larger: suppress smaller.
                    suppressed.add(orig_j)

        return [seg for orig_i, seg in enumerate(segments) if orig_i not in suppressed]


    # Background colour estimation (used by Method2Grid)
    @staticmethod
    def _sample_background_color(img: Image.Image) -> str:
        """
        Estimate the dominant background colour of a full-page screenshot by
        sampling pixels along all four edges (top row, bottom row, left column,
        right column).  Returns a CSS rgb() string.
        """
        rgb_img = img.convert("RGB")
        w, h = rgb_img.size
        step_x = max(1, w // 20)
        step_y = max(1, h // 20)
        pixels = []
        for x in range(0, w, step_x):
            pixels.append(rgb_img.getpixel((x, 0)))
            pixels.append(rgb_img.getpixel((x, h - 1)))
        for y in range(0, h, step_y):
            pixels.append(rgb_img.getpixel((0, y)))
            pixels.append(rgb_img.getpixel((w - 1, y)))
        r, g, b = Counter(pixels).most_common(1)[0][0]
        return f"rgb({r}, {g}, {b})"


    # DCGenGrid-compatible interface - this is diffrent from the original DCGenGrid
    def to_json_tree(self) -> dict:
        """
        Returns a bbox-tree dict compatible with DCGenGrid.assign_seg_tree_id().

        x-coordinates are snapped to full page width
        (x1=0, x2=page_w) so every segment becomes a full-width horizontal
        band. This aligns the scaffold and VLM input with actual webpage flow.

        After snapping, any two bands whose y-ranges overlap by more than 70%
        are deduplicated (keeping the first/topmost). This catches cases where
        two YOLO boxes had different x-extents (low original IoU) but become
        near-identical after x-snapping.

        Tree format (depth-1):
        {
            "bbox": [0, 0, page_w, page_h],
            "children": [
                {"bbox": [0, y1, page_w, y2], "children": []},
                ...
            ]
        }
        """
        w, h = self.img.size
        snapped = []
        for seg in self.layout.get("segments", []):
            x1, y1, x2, y2 = seg["bbox_xyxy"]
            y1 = max(0, int(y1))
            y2 = min(h, int(y2))
            if y2 > y1:
                snapped.append({"bbox": [0, y1, w, y2], "children": []})

        # Sort by y1 (defensive — from_yolo_model sorts, but layout.json path may not)
        snapped.sort(key=lambda c: c["bbox"][1])

        # Remove y-range near-duplicates created by x-snapping
        filtered = []
        for candidate in snapped:
            cy1, cy2 = candidate["bbox"][1], candidate["bbox"][3]
            ch = cy2 - cy1
            dominated = False
            for existing in filtered:
                ey1, ey2 = existing["bbox"][1], existing["bbox"][3]
                overlap = max(0, min(cy2, ey2) - max(cy1, ey1))
                if ch > 0 and overlap / ch > 0.7:
                    dominated = True
                    break
            if not dominated:
                filtered.append(candidate)

        # Fill coverage gaps between YOLO segments
        # gap_filled = []
        # prev_y2 = 0
        # for child in filtered:
        #     cy1 = child["bbox"][1]
        #     if cy1 > prev_y2 + 2:          # gap of more than 2px → insert filler
        #         gap_filled.append({
        #             "bbox": [0, prev_y2, w, cy1],
        #             "children": [],
        #             "_is_gap": True,        # flag: no VLM call needed
        #         })
        #     gap_filled.append(child)
        #     prev_y2 = child["bbox"][3]
        # # Fill any gap at the bottom of the page
        # if prev_y2 < h - 2:
        #     gap_filled.append({
        #         "bbox": [0, prev_y2, w, h],
        #         "children": [],
        #         "_is_gap": True,
        #     })
        return {"bbox": [0, 0, w, h], "children": filtered}

