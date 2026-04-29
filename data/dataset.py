"""
从 HuggingFace Hub 加载 Stage3 数据集（例如 ``your/repo_id``，列为 ``image`` / ``conversations`` / ``source``），
构造 batch 的方式与 ``caption_dataset.MultiModalInstructDataset`` 一致：

  - ``tokenizer.apply_chat_template`` 组装文本与 assistant mask
  - train：整段监督；eval/test：去掉最后一轮 assistant，``add_generation_prompt=True`` 续写

前提：仓库由 ``prepare_data/upload_data.py`` 推送，或与之下列 schema 对齐：

    - ``image``: ``datasets.Image`` → 索引后为 ``PIL.Image.Image``
    - ``conversations``: ``[{"role","content"}, ...]``
    - ``source``: ``str``

用法示例::

    from transformers import AutoProcessor, AutoTokenizer

    from data.dataset import HubMultiModalInstructDataset, MultiModalCollator, build_hf_dataloaders

    train_loader, val_loader, test_loader = build_hf_dataloaders(
        repo_id="your/repo_id",
        vision_model_name="google/siglip2-base-patch16-224",
        qwen_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from transformers import AutoProcessor, AutoTokenizer

from data.caption_dataset import SYSTEM_PROMPT, MultiModalCollator, MultiModalInstructDataset


def _pil_rgb(img: Any) -> Image.Image:
    """HF Image 列可能是 PIL、numpy(H,W,C) 等。"""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    if hasattr(img, "convert"):
        return img.convert("RGB")
    raise TypeError(f"unsupported image type: {type(img)}")


def _normalize_conversations(conv: Any) -> List[Dict[str, str]]:
    """HF Arrow 可能给出 Arrow Struct / dict-like。"""
    if conv is None:
        return []
    out: List[Dict[str, str]] = []
    for m in conv:
        if hasattr(m, "as_py"):
            m = m.as_py()
        if not isinstance(m, dict):
            role = str(getattr(m, "role", ""))
            content = str(getattr(m, "content", ""))
        else:
            role = str(m.get("role", ""))
            content = str(m.get("content", ""))
        out.append({"role": role, "content": content})
    return out


class HubMultiModalInstructDataset(MultiModalInstructDataset):
    """数据来自 ``datasets.Dataset``（Hub），其余逻辑继承 ``MultiModalInstructDataset``。"""

    def __init__(
        self,
        hf_dataset: HFDataset,
        processor,
        tokenizer,
        max_length: int = 1024,
        mode: str = "train",
        system_prompt: str | None = None,
        *,
        hub_repo_id: str | None = None,
    ):
        """
        Args:
            hf_dataset: 已通过 ``load_dataset(..., split=...)`` 得到的 map-style Dataset。
            hub_repo_id: 可选，仅用于 ``image_path`` 占位字符串便于日志。
            system_prompt: 默认沿用 ``caption_dataset.SYSTEM_PROMPT``。
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert mode in ("train", "eval", "test")
        self.mode = mode
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self._hub_repo_id = hub_repo_id or ""

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = _pil_rgb(item["image"])
        conversations = _normalize_conversations(item["conversations"])

        full_text, assistant_spans, prompt_text, last_reference = self._build_prompt(conversations)

        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        hub_tag = self._hub_repo_id or "hf_dataset"
        image_path = f"{hub_tag}:{idx}"

        if self.mode == "train":
            enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            offsets = enc["offset_mapping"].squeeze(0).tolist()

            labels = torch.full_like(input_ids, fill_value=-100)
            for tok_idx, (c_start, c_end) in enumerate(offsets):
                if c_end <= c_start:
                    continue
                for s_start, s_end in assistant_spans:
                    if c_start >= s_start and c_end <= s_end:
                        labels[tok_idx] = input_ids[tok_idx]
                        break

            if (labels != -100).sum().item() == 0:
                tail = max(1, input_ids.size(0) // 4)
                labels[-tail:] = input_ids[-tail:]

            reference = ""
            if conversations and conversations[-1].get("role") == "assistant":
                reference = (conversations[-1].get("content") or "").strip()

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt_len": 0,
                "reference": reference,
                "image_path": image_path,
                "source": item.get("source", "") or "",
            }

        enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": input_ids.size(0),
            "reference": last_reference,
            "image_path": image_path,
            "source": item.get("source", "") or "",
        }


def load_hf_stage3_dataset_dict(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
) -> DatasetDict:
    """加载包含 ``train`` / ``validation`` / ``test`` 的 Hub 数据集。"""
    kwargs: Dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    ds = load_dataset(repo_id, **kwargs)
    if not isinstance(ds, DatasetDict):
        raise ValueError(f"期望 DatasetDict（含 train/validation/test），得到 {type(ds)}")
    missing = [s for s in ("train", "validation", "test") if s not in ds]
    if missing:
        raise KeyError(f"仓库缺少 split: {missing}；现有 keys={list(ds.keys())}")
    return ds


def build_hf_dataloaders(
    repo_id: str,
    vision_model_name: str,
    qwen_model_name: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 1024,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train / val / test DataLoader（数据来自 Hub ``repo_id``）。"""
    ds_dict = load_hf_stage3_dataset_dict(
        repo_id,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    processor = AutoProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = HubMultiModalInstructDataset(
        ds_dict["train"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="train",
        hub_repo_id=repo_id,
    )
    val_ds = HubMultiModalInstructDataset(
        ds_dict["validation"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="eval",
        hub_repo_id=repo_id,
    )
    test_ds = HubMultiModalInstructDataset(
        ds_dict["test"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="test",
        hub_repo_id=repo_id,
    )

    collator = MultiModalCollator(tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test: Hub Stage3 dataset")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--split", type=str, default="train", choices=("train", "validation", "test"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds_dict = load_hf_stage3_dataset_dict(args.repo_id, revision=args.revision)
    hf_ds = ds_dict[args.split]

    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mode = "train" if args.split == "train" else ("eval" if args.split == "validation" else "test")
    ds = HubMultiModalInstructDataset(
        hf_ds,
        processor=processor,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode=mode,
        hub_repo_id=args.repo_id,
    )

    idx = min(max(0, args.index), len(ds) - 1)

    raw = hf_ds[idx]
    print("\n--- Raw HF row ---")
    print(f"  source         : {raw.get('source', '')!r}")
    rim = raw["image"]
    if hasattr(rim, "size"):
        print(f"  image          : PIL size={rim.size}, mode={getattr(rim, 'mode', '?')}")
    conv_raw = raw["conversations"]
    print(f"  conversations  : {len(conv_raw)} turns")
    for ti, turn in enumerate(conv_raw[:6]):
        if hasattr(turn, "as_py"):
            turn = turn.as_py()
        role = turn["role"] if isinstance(turn, dict) else getattr(turn, "role", "?")
        content = (turn["content"] if isinstance(turn, dict) else getattr(turn, "content", "")) or ""
        preview = content[:160].replace("\n", "\\n")
        if len(content) > 160:
            preview += "..."
        print(f"    [{ti}] {role}: {preview}")

    print("\n--- HubMultiModalInstructDataset[idx] ---")
    print(f"[smoke] repo={args.repo_id} split={args.split} mode={mode} len={len(ds)} index={idx}")
    sample = ds[idx]
    print(f"  pixel_values : {tuple(sample['pixel_values'].shape)} dtype={sample['pixel_values'].dtype}")
    print(f"  input_ids    : {tuple(sample['input_ids'].shape)}")
    print(f"  attention_mask sum : {int(sample['attention_mask'].sum().item())}")
    if mode == "train":
        n_lab = int((sample["labels"] != -100).sum().item())
        print(f"  labels (non -100): {n_lab}")
    else:
        print(f"  prompt_len   : {sample['prompt_len']}")
    print(f"  reference (preview): {(sample['reference'] or '')[:200]}{'...' if len(sample.get('reference') or '') > 200 else ''}")
    print(f"  source       : {sample['source']!r}")
    print(f"  image_path   : {sample['image_path']}")

    text_preview = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    wrap = 600
    print(f"  decoded[:{wrap}]:\n    {text_preview[:wrap]}{'...' if len(text_preview) > wrap else ''}")
"""
from data.dataset import build_hf_dataloaders
train_loader, val_loader, test_loader = build_hf_dataloaders(
    repo_id="your/repo_id",
    vision_model_name="google/siglip2-base-patch16-224",
    qwen_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    batch_size=16,
    num_workers=2,
    max_length=1024,
)
cd stage3   # 项目根目录
python -m dataset --repo-id Lris47/MLLM3-textcaps-scienceqa-vqav2 --split train --index 0
"""