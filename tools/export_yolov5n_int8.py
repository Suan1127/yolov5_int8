#!/usr/bin/env python3
"""
Export YOLOv5n Conv weights as INT8 + scale_w from the start.

- *.conv.weight: symmetric int8 (scale_w = max(|min|,|max|)/127), saved to weights_int8.bin
- model.24.m.0/1/2.weight (Detect head 1x1 conv): 동일 방식으로 포함 → C에서 int8 detect head 사용 가능
- Bias/BN 등 나머지는 기존 float weights.bin 유지 (동일 경로에 두거나 별도)
- 결과: weights_int8.bin, scales_int8.json (key 순서 + scale_w + shape)
- C 쪽에서는 float weight 버퍼와 weight minmax 없이 q_weight, scale_w만 로드하면 됨.

포맷 맵은 inference에서 쓰는 것과 동일해야 함:
- fused 사용 중이면: weights_map_fused.json
- unfused 사용 중이면: weights_map_unfused.json

사용법 (fused 기준):
  python tools/export_yolov5n_int8.py --format-map weights/yolov5n/weights_map_fused.json
  (weights_fused.bin 과 같은 디렉에 weights_int8.bin, scales_int8.json 생성)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict

# third_party/yolov5 (ultralytics yolov5) 로 모델 로드 → torch.load(weights_only) 회피
YOLOV5_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "yolov5"
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

try:
    from models.experimental import attempt_load
except ImportError as e:
    print(f"Error: {e}. Run: pip install -r third_party/yolov5/requirements.txt")
    attempt_load = None


def symmetric_scale_from_minmax(min_val: float, max_val: float) -> float:
    """scale = max(|min|, |max|) / 127"""
    a = max(abs(min_val), abs(max_val))
    if a < 1e-9:
        a = 1e-9
    return float(a) / 127.0


def quantize_float_to_int8(arr: np.ndarray, scale: float) -> np.ndarray:
    """q = round(x / scale), clamp [-127, 127]"""
    q = np.round(arr.astype(np.float64) / scale).astype(np.int32)
    q = np.clip(q, -127, 127)
    return q.astype(np.int8)


# Detect head 1x1 conv weights (이름에 .conv 없음; C 쪽 model.24.m.0/1/2.weight 로 로드)
DETECT_HEAD_WEIGHT_KEYS = ("model.24.m.0.weight", "model.24.m.1.weight", "model.24.m.2.weight")


def load_format_map(format_map_path: str) -> list:
    """Return list of (key, shape) in map order: *.conv.weight + detect head model.24.m.*.weight."""
    with open(format_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for k, v in data.items():
        is_conv_weight = k.endswith(".conv.weight")
        is_detect_head = k in DETECT_HEAD_WEIGHT_KEYS
        if not (is_conv_weight or is_detect_head):
            continue
        shape = tuple(v["shape"])
        out.append((k, shape))
    return out


def export_int8_weights(model_pt_path: str, format_map_path: str, output_dir: str, fuse: bool = True) -> None:
    if attempt_load is None:
        raise RuntimeError("YOLOv5 attempt_load not found. Install third_party/yolov5 and requirements.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key_order = load_format_map(format_map_path)
    if not key_order:
        raise ValueError("No *.conv.weight keys in format map")

    # third_party/yolov5 로 로드 (torch.load weights_only 이슈 없음). fused 사용 시 fuse=True
    print(f"Loading model: {model_pt_path} (fuse={fuse})")
    model = attempt_load(model_pt_path, device="cpu", inplace=False, fuse=fuse)
    state = model.state_dict()

    int8_chunks = []
    order = []
    scales = {}
    shapes = {}

    for key, expected_shape in key_order:
        if key not in state:
            continue
        w = state[key].detach().cpu().numpy().astype(np.float32)
        if w.shape != expected_shape:
            continue
        min_val, max_val = float(w.min()), float(w.max())
        scale_w = symmetric_scale_from_minmax(min_val, max_val)
        q = quantize_float_to_int8(w, scale_w)
        int8_chunks.append(q.tobytes())
        order.append(key)
        scales[key] = scale_w
        shapes[key] = list(w.shape)

    weights_int8 = b"".join(int8_chunks)
    int8_path = output_dir / "weights_int8.bin"
    with open(int8_path, "wb") as f:
        f.write(weights_int8)
    print(f"Saved {int8_path} ({len(weights_int8)} bytes, {len(order)} conv layers)")

    meta = {"order": order, "scales": scales, "shapes": shapes}
    meta_path = output_dir / "scales_int8.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv5n conv weights as int8 + scale_w")
    parser.add_argument("model", nargs="?", default="yolov5n.pt", help="Path to yolov5n.pt")
    parser.add_argument("--format-map", type=str, required=True,
                        help="Inference와 동일한 map: fused면 weights_map_fused.json, unfused면 weights_map_unfused.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same dir as format-map)")
    parser.add_argument("--no-fuse", action="store_true",
                        help="Load unfused (use with weights_map_unfused.json)")
    args = parser.parse_args()

    format_map_path = Path(args.format_map)
    if not format_map_path.exists():
        print(f"Error: format map not found: {format_map_path}")
        return 1
    output_dir = args.output or str(format_map_path.parent)
    export_int8_weights(args.model, str(format_map_path), output_dir, fuse=not args.no_fuse)
    return 0


if __name__ == "__main__":
    exit(main() or 0)
