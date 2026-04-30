"""
Stage3 多模态指令数据集：仅从 HuggingFace Hub 加载。

固定默认仓库：``Lris47/MLLMstage3-textcaps-scienceqa-textvqa``（列 ``image`` / ``conversations`` / ``source``），
与 ``prepare_data/upload_data.py`` 推送的 schema 一致。

- ``tokenizer.apply_chat_template`` 组装文本；训练时对多轮 **每一轮 assistant** 用字符 span → token mask 监督。
- eval/test：去掉最后一轮 assistant，``add_generation_prompt=True`` 得到续写前缀。

Collator：若样本已带 ``labels``（训练集），则只做 padding；否则用 ``prompt_len`` 掩码（推理 split）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset


# ---------------------------------------------------------------------------
# 约定常量
# ---------------------------------------------------------------------------
DEFAULT_HF_REPO_ID = "Lris47/MLLMstage3-textcaps-scienceqa-textvqa"

SYSTEM_PROMPT = "You are a helpful assistant."


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


class MultiModalInstructDataset(Dataset):
    """数据来自 ``datasets.Dataset``（Hub map-style）。"""

    def __init__(
        self,
        hf_dataset: HFDataset,
        processor,
        tokenizer,
        max_length: int = 1024,
        mode: str = "train",
        system_prompt: str | None = None,
        *,
        hub_repo_id: str = "",
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert mode in ("train", "eval", "test")
        self.mode = mode
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self._hub_repo_id = hub_repo_id

    def __len__(self) -> int:
        return len(self.dataset)

    def _prepend_system(self, conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not conversations:
            return conversations
        if conversations[0].get("role") == "system":
            return conversations
        return [{"role": "system", "content": self.system_prompt}] + conversations

    def _build_prompt(
        self, conversations: List[Dict[str, str]]
    ) -> Tuple[str, List[Tuple[int, int]], str, Any]:
        """
        Returns:
            full_text: 完整对话渲染串
            assistant_spans: assistant **content** 在 full_text 中的 [start, end) 字符区间
            prompt_text: eval/test 用的续写前缀（最后一轮 assistant 已去掉 + generation prompt）
            last_reference: 最后一个 assistant 原文（评测参考）；test 下可能为 None
        """
        msgs = self._prepend_system(conversations)
        if not msgs:
            return "", [], "", ""

        full_text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

        assistant_spans: List[Tuple[int, int]] = []
        pos = 0
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            content = m.get("content") or ""
            if not content:
                continue
            idx = full_text.find(content, pos)
            if idx < 0:
                idx = full_text.find(content.strip(), pos)
            if idx >= 0:
                assistant_spans.append((idx, idx + len(content)))
                pos = idx + len(content)
            else:
                pos += 1

        if msgs[-1].get("role") == "assistant":
            last_content = msgs[-1].get("content") or ""
            if self.mode == "test":
                last_reference = (
                    last_content if last_content not in ("", "None", None) else None
                )
            else:
                last_reference = last_content
            msgs_trunc = msgs[:-1]
        else:
            last_reference = ""
            msgs_trunc = msgs

        prompt_text = self.tokenizer.apply_chat_template(
            msgs_trunc, tokenize=False, add_generation_prompt=True
        )

        return full_text, assistant_spans, prompt_text, last_reference

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = _pil_rgb(item["image"])
        conversations = _normalize_conversations(item["conversations"])

        full_text, assistant_spans, prompt_text, last_reference = self._build_prompt(
            conversations
        )

        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        hub_tag = self._hub_repo_id or DEFAULT_HF_REPO_ID
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


class MultiModalCollator:
    """Padding；若样本含与 ``input_ids`` 等长的 ``labels``（训练），则沿用；否则按 ``prompt_len`` 构造 mask。"""

    def __init__(self, tokenizer, max_length: int | None = None, padding_side: str = "right"):
        assert padding_side in ("right", "left")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
        references = [x["reference"] for x in batch]
        sources = [x.get("source", "") or "" for x in batch]
        image_paths = [x.get("image_path", "") or "" for x in batch]

        input_ids_list = [x["input_ids"] for x in batch]
        attn_list = [x["attention_mask"] for x in batch]
        prompt_lens = [int(x["prompt_len"]) for x in batch]

        pre_labels = [x.get("labels") for x in batch]
        use_precomputed = all(
            isinstance(lb, torch.Tensor)
            and lb.dtype == input_ids_list[i].dtype
            and lb.shape[0] == input_ids_list[i].shape[0]
            for i, lb in enumerate(pre_labels)
        )

        max_seq_len = max(x.size(0) for x in input_ids_list)
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)

        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for i, (ids, attn, prompt_len) in enumerate(
            zip(input_ids_list, attn_list, prompt_lens)
        ):
            ids = ids[:max_seq_len]
            attn = attn[:max_seq_len]
            prompt_len = min(prompt_len, max_seq_len)

            if use_precomputed:
                lb = pre_labels[i][:max_seq_len]
                labels_core = lb.clone()
            else:
                labels_core = ids.clone()
                labels_core[:prompt_len] = -100

            pad_len = max_seq_len - ids.size(0)

            pad_ids = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            pad_mask = torch.zeros(pad_len, dtype=attn.dtype)
            pad_labels = torch.full((pad_len,), -100, dtype=ids.dtype)

            if self.padding_side == "right":
                padded_ids = torch.cat([ids, pad_ids])
                padded_mask = torch.cat([attn, pad_mask])
                labels = (
                    torch.cat([labels_core, pad_labels]) if pad_len > 0 else labels_core
                )
            else:
                padded_ids = torch.cat([pad_ids, ids])
                padded_mask = torch.cat([pad_mask, attn])
                labels = (
                    torch.cat([pad_labels, labels_core]) if pad_len > 0 else labels_core
                )

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            padded_labels.append(labels)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "attention_mask": torch.stack(padded_attention_mask, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "references": references,
            "sources": sources,
            "image_paths": image_paths,
        }


def load_hf_stage3_dataset_dict(
    repo_id: str | None = None,
    *,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
) -> DatasetDict:
    """加载包含 ``train`` / ``validation`` / ``test`` 的 Hub 数据集。"""
    rid = (repo_id or "").strip() or DEFAULT_HF_REPO_ID
    kwargs: Dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    ds = load_dataset(rid, **kwargs)
    if not isinstance(ds, DatasetDict):
        raise ValueError(
            f"期望 DatasetDict（含 train/validation/test），得到 {type(ds)}"
        )
    missing = [s for s in ("train", "validation", "test") if s not in ds]
    if missing:
        raise KeyError(f"仓库缺少 split: {missing}；现有 keys={list(ds.keys())}")
    return ds


def build_dataloaders(
    vision_model_name: str,
    qwen_model_name: str,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 1024,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train / val / test DataLoader（数据来自 Hub；``repo_id`` 默认 `DEFAULT_HF_REPO_ID`）。"""
    ds_dict = load_hf_stage3_dataset_dict(
        repo_id,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    rid = (repo_id or "").strip() or DEFAULT_HF_REPO_ID

    processor = AutoProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = MultiModalInstructDataset(
        ds_dict["train"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="train",
        hub_repo_id=rid,
    )
    val_ds = MultiModalInstructDataset(
        ds_dict["validation"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="eval",
        hub_repo_id=rid,
    )
    test_ds = MultiModalInstructDataset(
        ds_dict["test"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="test",
        hub_repo_id=rid,
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


# 与历史代码兼容
build_hf_dataloaders = build_dataloaders

# 历史名称（曾指 Hub 子类）
HubMultiModalInstructDataset = MultiModalInstructDataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test: Hub Stage3 dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help=f"默认 {DEFAULT_HF_REPO_ID}",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--vision_model_name",
        type=str,
        default="google/siglip2-base-patch16-224",
    )
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds_dict = load_hf_stage3_dataset_dict(args.repo_id, revision=args.revision)
    hf_ds = ds_dict[args.split]

    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mode = (
        "train"
        if args.split == "train"
        else ("eval" if args.split == "validation" else "test")
    )
    ds = MultiModalInstructDataset(
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
        content = (
            turn["content"] if isinstance(turn, dict) else getattr(turn, "content", "")
        ) or ""
        preview = content[:160].replace("\n", "\\n")
        if len(content) > 160:
            preview += "..."
        print(f"    [{ti}] {role}: {preview}")

    print("\n--- MultiModalInstructDataset[idx] ---")
    print(
        f"[smoke] repo={args.repo_id} split={args.split} mode={mode} len={len(ds)} index={idx}"
    )
    sample = ds[idx]
    print(
        f"  pixel_values : {tuple(sample['pixel_values'].shape)} dtype={sample['pixel_values'].dtype}"
    )
    print(f"  input_ids    : {tuple(sample['input_ids'].shape)}")
    print(f"  attention_mask sum : {int(sample['attention_mask'].sum().item())}")
    if mode == "train":
        n_lab = int((sample["labels"] != -100).sum().item())
        print(f"  labels (non -100): {n_lab}")
    else:
        print(f"  prompt_len   : {sample['prompt_len']}")
    ref = sample.get("reference") or ""
    print(
        f"  reference (preview): {str(ref)[:200]}{'...' if len(str(ref)) > 200 else ''}"
    )
    print(f"  source       : {sample['source']!r}")
    print(f"  image_path   : {sample['image_path']}")

    text_preview = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    wrap = 600
    print(
        f"  decoded[:{wrap}]:\n    {text_preview[:wrap]}{'...' if len(text_preview) > wrap else ''}"
    )
