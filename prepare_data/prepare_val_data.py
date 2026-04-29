"""
Stage3 验证集预处理（与 prepare_train_data 相同 schema）：
    - lmms-lab/TextCaps → split ``val``
    - derek-thomas/ScienceQA → split ``validation``
    - lmms-lab/VQAv2 → split ``testdev``

用法：
    python data2/prepare_val_data.py [--out_dir ./stage3_val_data] [--seed 42]

需在 ``data2`` 同级目录运行，或以 ``python path/to/prepare_val_data.py`` 方式运行（以便导入 prepare_train_data）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from prepare_train_data import IMAGE_PLACEHOLDER, sample_scienceqa, sample_textcaps, sample_vqav2

# HF 侧 split 名称（与数据集卡片一致）
SCIENCEQA_SPLIT = "validation"
TEXTCAPS_SPLIT = "val"
VQAV2_SPLIT = "testdev"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage3 验证集：TextCaps(val) + ScienceQA(validation) + VQAv2(testdev)，写出 chat.json / meta.json",
    )
    parser.add_argument("--out_dir", type=str, default="./stage3_val_data", help="输出根目录（含 images/）")
    parser.add_argument("--seed", type=int, default=42, help="各数据源打乱及合并后 shuffle 的随机种子")
    parser.add_argument(
        "--scienceqa-use-caption",
        action="store_true",
        help="ScienceQA Context 中并入图像 caption（官方字段 caption）",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不打乱合并后的样本顺序",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    rng = random.Random(args.seed)

    all_records: List[Dict[str, Any]] = []
    all_records.extend(
        sample_scienceqa(
            None,
            image_dir,
            rng,
            use_caption=args.scienceqa_use_caption,
            split=SCIENCEQA_SPLIT,
        )
    )
    all_records.extend(sample_textcaps(None, image_dir, rng, split=TEXTCAPS_SPLIT))
    all_records.extend(sample_vqav2(None, image_dir, rng, split=VQAV2_SPLIT))

    if not args.no_shuffle:
        rng.shuffle(all_records)

    chat_path = os.path.join(out_dir, "chat.json")
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    print(f"[done] wrote {len(all_records)} records -> {chat_path}")

    counts: Dict[str, int] = {}
    for r in all_records:
        counts[r["source"]] = counts.get(r["source"], 0) + 1

    meta = {
        "split_set": "validation",
        "total": len(all_records),
        "counts_by_source": counts,
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "scienceqa_use_caption": args.scienceqa_use_caption,
        "schema": "conversations=[{role, content}], image placeholder embedded",
        "image_placeholder": IMAGE_PLACEHOLDER,
        "datasets": {
            "scienceqa": {"hf_id": "derek-thomas/ScienceQA", "split": SCIENCEQA_SPLIT},
            "textcaps": {"hf_id": "lmms-lab/TextCaps", "split": TEXTCAPS_SPLIT},
            "vqav2": {"hf_id": "lmms-lab/VQAv2", "split": VQAV2_SPLIT},
        },
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote meta -> {meta_path}")


if __name__ == "__main__":
    main()
