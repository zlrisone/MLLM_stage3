"""
Stage3 数据预处理：
    1) 下载 ScienceQA、TextCaps、TextVQA（lmms-lab/textvqa）数据集
    2) 统一转成 {role, content} 形式的 conversations，image 占位符直接嵌入 content
    3) 图像落盘为本地 jpg
    4) 写出 chat.json 与 meta.json

用法：
    python data2/prepare_train_data.py [--out_dir ./stage3_train_data] [--seed 42]

VQA 子集来自 ScienceQA（derek-thomas/ScienceQA），user 文本为官方 QCM 段，
assistant 为 LEA：`Answer: {lecture} {solution} The answer is X.`（与
lupantech/ScienceQA `base_prompt.py` 中 prompt_format=QCM-LEA、few-shot 示范一致）

输出样本 schema:
    {
      "id": "...",
      "image": "<source>/<name>.jpg",   # 相对 image_root
      "source": "caption | vqa",
      "conversations": [
          {"role": "user",      "content": "<|vision_start|><|image_pad|><|vision_end|>\\nQuestion: ..."},
          {"role": "assistant", "content": "Answer: ... The answer is A."}
      ]
    }
"""
import os
import io
import json
import random
import argparse
from typing import List, Dict, Any, Optional

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

OPTION_LABELS_DEFAULT = ["A", "B", "C", "D", "E"]


SHORT_CAPTION_PROMPTS = [
    "Describe the image briefly.",
    "Provide a short caption for this image.",
    "What is shown in this image?",
    "Give a concise description of the image.",
    "Summarize this image in one sentence.",
]

def with_image(text: str) -> str:
    """在 user 文本中嵌入图像占位符（单图场景）。
    若含 <image>，仅替换为占位符（不再额外前缀一条）；否则在开头嵌入占位符。"""
    if "<image>" in text:
        return text.replace("<image>", IMAGE_PLACEHOLDER)
    if not text:
        return IMAGE_PLACEHOLDER
    return f"{IMAGE_PLACEHOLDER}\n{text}"


def save_pil_image(img: Image.Image, out_path: str, quality: int = 90) -> None:
    if img.mode != "RGB":
        img = img.convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "JPEG", quality=quality)


def _decode_image(raw) -> Optional[Image.Image]:
    """HF datasets 里的 image 字段可能是 PIL.Image 或 dict {bytes, path}。"""
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw
    if isinstance(raw, dict):
        if raw.get("bytes") is not None:
            return Image.open(io.BytesIO(raw["bytes"]))
        if raw.get("path") is not None and os.path.exists(raw["path"]):
            return Image.open(raw["path"])
    return None


# -----------------------------------------------------------------------------
# ScienceQA —— QCM-LEA（与 lupantech/ScienceQA models/base_prompt.py 一致）
# -----------------------------------------------------------------------------
def _scienceqa_context(row: Dict[str, Any], use_caption: bool) -> str:
    hint = (row.get("hint") or "").strip()
    img_ctx = ""
    if use_caption:
        img_ctx = (row.get("caption") or "").strip()
    merged = " ".join(p for p in (hint, img_ctx) if p).strip()
    return merged if merged else "N/A"


def _normalize_choices(raw) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else None
    return None


def _choice_line(choices: List[str], option_labels: List[str]) -> str:
    parts = []
    for i, c in enumerate(choices):
        if i >= len(option_labels):
            break
        parts.append(f"({option_labels[i]}) {c}")
    return " ".join(parts)


def _normalize_answer_idx(row: Dict[str, Any], n_choices: int) -> Optional[int]:
    a = row.get("answer")
    if a is None:
        return None
    if isinstance(a, bool):  # bool 是 int 子类，需排除
        return None
    try:
        i = int(a)
    except (TypeError, ValueError):
        return None
    return i if 0 <= i < n_choices else None


def format_qcm_user(
    row: Dict[str, Any],
    option_labels: List[str],
    use_caption: bool,
) -> Optional[str]:
    """QCM：Question / Context / Options（无 Answer: 行）。"""
    q = (row.get("question") or "").strip()
    if not q:
        return None
    choices = _normalize_choices(row.get("choices"))
    if not choices:
        return None
    ctx = _scienceqa_context(row, use_caption)
    opts = _choice_line(choices, option_labels)
    return f"Question: {q}\nContext: {ctx}\nOptions: {opts}\n"


def format_lea_assistant(row: Dict[str, Any], answer_letter: str) -> str:
    """LEA：Answer: {lecture} {solution} The answer is X."""
    lec = (row.get("lecture") or "").strip()
    sol = (row.get("solution") or "").strip()
    mid = " ".join(p for p in (lec, sol) if p).strip()
    if mid:
        return f"Answer: {mid} The answer is {answer_letter}."
    return f"Answer: The answer is {answer_letter}."


def sample_scienceqa(
    num_samples: Optional[int],
    image_dir: str,
    rng: random.Random,
    option_labels: Optional[List[str]] = None,
    use_caption: bool = False,
    split: str = "train",
) -> List[Dict[str, Any]]:
    print(f"[scienceqa] loading dataset split={split!r} ...")
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    print(f"[scienceqa] total rows: {len(ds)}")
    labels = option_labels or OPTION_LABELS_DEFAULT[:]

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    results: List[Dict[str, Any]] = []
    sample_idx = 0
    for idx in tqdm(indices, desc=f"scienceqa[{split}]"):
        if num_samples is not None and len(results) >= num_samples:
            break
        row = ds[idx]
        question = row.get("question")
        if not question:
            continue
        img = _decode_image(row.get("image"))
        if img is None:
            continue

        choices = _normalize_choices(row.get("choices"))
        if not choices:
            continue
        ans_idx = _normalize_answer_idx(row, len(choices))
        if ans_idx is None:
            continue
        letter = labels[ans_idx]

        image_rel = f"scienceqa/{sample_idx:07d}.jpg"
        image_abs = os.path.join(image_dir, image_rel)
        if not os.path.exists(image_abs):
            try:
                save_pil_image(img, image_abs)
            except Exception as e:
                print(f"[scienceqa] save image failed idx={idx}: {e}")
                continue

        user_text = format_qcm_user(row, labels, use_caption)
        if user_text is None:
            continue
        assistant_text = format_lea_assistant(row, letter)

        results.append({
            "id": f"scienceqa_{sample_idx:07d}",
            "image": image_rel,
            "source": "scienceQA",
            "conversations": [
                {"role": "user", "content": with_image(user_text)},
                {"role": "assistant", "content": assistant_text},
            ],
        })
        sample_idx += 1
    print(f"[scienceqa] sampled {len(results)} records")
    return results


# -----------------------------------------------------------------------------
# TextCaps (short caption)
# -----------------------------------------------------------------------------
def sample_textcaps(
    num_samples: Optional[int],
    image_dir: str,
    rng: random.Random,
    split: str = "train",
) -> List[Dict[str, Any]]:
    print(f"[textcaps] loading dataset split={split!r} ...")
    ds = load_dataset("lmms-lab/TextCaps", split=split)
    print(f"[textcaps] total rows: {len(ds)}")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    results: List[Dict[str, Any]] = []
    sample_idx = 0
    for idx in tqdm(indices, desc=f"textcaps[{split}]"):
        if num_samples is not None and len(results) >= num_samples:
            break
        row = ds[idx]
        captions = row.get("caption_str") or row.get("reference_strs") or []
        if not captions:
            continue
        img = _decode_image(row.get("image"))
        if img is None:
            continue

        image_id = row.get("image_id") or f"tc_{idx}"
        image_rel = f"textcaps/{image_id}.jpg"
        image_abs = os.path.join(image_dir, image_rel)
        if not os.path.exists(image_abs):
            try:
                save_pil_image(img, image_abs)
            except Exception as e:
                print(f"[textcaps] save image failed idx={idx}: {e}")
                continue

        used = captions[: min(3, len(captions))]
        for cap in used:
            if num_samples is not None and len(results) >= num_samples:
                break
            cap = cap.strip()
            if not cap:
                continue
            prompt = rng.choice(SHORT_CAPTION_PROMPTS)
            results.append({
                "id": f"textcaps_{sample_idx:07d}",
                "image": image_rel,
                "source": "caption",
                "conversations": [
                    {"role": "user", "content": with_image(prompt)},
                    {"role": "assistant", "content": cap},
                ],
            })
            sample_idx += 1
    print(f"[textcaps] sampled {len(results)} records")
    return results


def _majority_answer(answers) -> Optional[str]:
    """从多条标注中取众数答案；支持 list[str] 或 list[dict]（含 answer 等键）。"""
    if not answers:
        return None
    texts: List[str] = []
    for item in answers:
        if isinstance(item, dict):
            s = item.get("answer")
            if s is None:
                s = item.get("text") or item.get("raw")
        else:
            s = item
        if s is None:
            continue
        t = str(s).strip()
        if t:
            texts.append(t)
    if not texts:
        return None
    return Counter(texts).most_common(1)[0][0]

def sample_textqa(
    num_samples: Optional[int],
    image_dir: str,
    rng: random.Random,
    split: str = "train",
) -> List[Dict[str, Any]]:
    print(f"[textvqa] loading dataset split={split!r} ...")
    ds = load_dataset("lmms-lab/textvqa", split=split)
    print(f"[textvqa] total rows: {len(ds)}")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    results: List[Dict[str, Any]] = []
    sample_idx = 0
    for idx in tqdm(indices, desc=f"textvqa[{split}]"):
        if num_samples is not None and len(results) >= num_samples:
            break
        row = ds[idx]
        question = row.get("question")
        if not question:
            continue
        img = _decode_image(row.get("image"))
        if img is None:
            continue
        answers = row.get("answers")
        answer = _majority_answer(answers)
        if not answer:
            continue
        image_rel = f"textqa/{sample_idx:07d}.jpg"
        image_abs = os.path.join(image_dir, image_rel)
        if not os.path.exists(image_abs):
            try:
                save_pil_image(img, image_abs)
            except Exception as e:
                print(f"[textqa] save image failed idx={idx}: {e}")
                continue
        results.append({
            "id": f"textqa_{sample_idx:07d}",
            "image": image_rel,
            "source": "textqa",
            "conversations": [
                {"role": "user", "content": with_image(question)},
                {"role": "assistant", "content": answer},
            ],
        })
        sample_idx += 1
    print(f"[textqa] sampled {len(results)} records")
    return results


# -----------------------------------------------------------------------------
# main：ScienceQA + TextCaps + TextVQA 全量导出，不写配置文件、不做配比裁剪
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage3 训练数据：ScienceQA + TextCaps + TextVQA，全量可用样本，写出 chat.json / meta.json",
    )
    parser.add_argument("--out_dir", type=str, default="./stage3_train_data", help="输出根目录（含 images/）")
    parser.add_argument("--seed", type=int, default=42, help="各数据集行顺序打乱及合并后 shuffle 的随机种子")
    parser.add_argument(
        "--scienceqa-use-caption",
        action="store_true",
        help="ScienceQA Context 中并入图像 caption（官方字段 caption）",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不打乱合并后的样本顺序（默认为 scienceqa→textcaps→textvqa 合并后再 shuffle）",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    rng = random.Random(args.seed)

    all_records: List[Dict[str, Any]] = []
    all_records.extend(sample_scienceqa(None, image_dir, rng, use_caption=args.scienceqa_use_caption))
    all_records.extend(sample_textcaps(None, image_dir, rng))
    all_records.extend(sample_textqa(None, image_dir, rng))

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
        "total": len(all_records),
        "counts_by_source": counts,
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "scienceqa_use_caption": args.scienceqa_use_caption,
        "schema": "conversations=[{role, content}], image placeholder embedded",
        "image_placeholder": IMAGE_PLACEHOLDER,
        "datasets": {
            "scienceqa": {"hf_id": "derek-thomas/ScienceQA", "split": "train"},
            "textcaps": {"hf_id": "lmms-lab/TextCaps", "split": "train"},
            "textvqa": {"hf_id": "lmms-lab/textvqa", "split": "train"},
        },
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote meta -> {meta_path}")


if __name__ == "__main__":
    main()
