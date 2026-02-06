#!/usr/bin/env python3
"""
Calibration 입력 생성: inference와 100% 동일한 전처리 사용.

- 전처리: tools/preprocess.py의 preprocess_image() 재사용
  (letterbox 640x640, pad=114, BGR→RGB, /255, NCHW float32)
- 입력: data/coco/val2017/*.jpg
- 출력: data/calibration/inputs/*.bin + list.txt (경로 목록)

사용법:
  python tools/export_calibration_inputs.py
  python tools/export_calibration_inputs.py --max 100   # 처음 100장만
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 기준으로 preprocess 모듈 로드 (동일 전처리 = inference와 100% 일치)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.preprocess import preprocess_image, save_tensor
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("preprocess", PROJECT_ROOT / "tools" / "preprocess.py")
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)
    preprocess_image = preprocess.preprocess_image
    save_tensor = preprocess.save_tensor


def main():
    parser = argparse.ArgumentParser(description="Export calibration inputs (same preprocess as inference)")
    parser.add_argument("--input-dir", type=str, default="data/coco/val2017",
                        help="Input images directory")
    parser.add_argument("--output-dir", type=str, default="data/calibration/inputs",
                        help="Output directory for .bin tensors")
    parser.add_argument("--size", type=int, default=640, help="Letterbox size (must match inference)")
    parser.add_argument("--max", type=int, default=None, help="Max number of images (default: all)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if args.max is not None:
        images = images[: args.max]

    if not images:
        print(f"No images in {input_dir}")
        return 1

    list_path = output_dir / "list.txt"
    list_lines = []

    for img_path in images:
        try:
            tensor, _ = preprocess_image(str(img_path), args.size)
            stem = img_path.stem
            bin_path = output_dir / f"{stem}.bin"
            save_tensor(tensor, str(bin_path))
            list_lines.append(f"{stem}.bin\n")
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")
            continue

    with open(list_path, "w") as f:
        f.writelines(list_lines)

    print(f"Calibration inputs: {len(list_lines)} tensors -> {output_dir}")
    print(f"List file: {list_path}")
    return 0


if __name__ == "__main__":
    exit(main())
