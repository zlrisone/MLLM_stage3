"""
Stage3 Dataset / Collator / DataLoader。

样本 schema（由 prepare_stage3_data.py 产出）::

    {
      "id": "...",
      "image": "<source>/<name>.jpg",     # 相对 image_root
      "source": "caption | long_caption | vqa",
      "conversations": [
          {"role": "user",      "content": "<|vision_start|><|image_pad|><|vision_end|>\\n..."},
          {"role": "assistant", "content": "..."},
          ...
      ]
    }

本模块约定：
  - 使用 `tokenizer.apply_chat_template(...)` 拼 prompt，对多轮天然支持
  - labels 在 Dataset 内部生成（assistant 段之外填 -100），collator 只做 padding
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer


SYSTEM_PROMPT = "You are a helpful assistant."
IM_END = "<|im_end|>"


class MultiModalInstructDataset(Dataset):
    """多模态（多轮）指令数据集。"""

    def __init__(
        self,
        json_path: str,
        image_root: str,
        processor,
        tokenizer,
        max_length: int = 1024,
        mode: str = "train",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert mode in ("train", "eval", "test")
        self.mode = mode
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.dataset)

    # -------------------------------------------------------------------------
    # prompt 构造（基于 tokenizer.apply_chat_template）
    # -------------------------------------------------------------------------
    def _prepend_system(self, conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not conversations:
            return conversations
        if conversations[0].get("role") == "system":
            return conversations
        return [{"role": "system", "content": self.system_prompt}] + conversations

    def _build_prompt(
        self, conversations: List[Dict[str, str]]
    ) -> Tuple[str, List[Tuple[int, int]], str, str]:
        """
        使用 `apply_chat_template` 组装 prompt。

        返回：
            full_text        : 完整渲染后的字符串（train 模式 tokenize 这个）
            assistant_spans  : full_text 中每个 assistant reply + 末尾 <|im_end|> 的
                               字符区间 [start, end)，用于 label mask
            prompt_text      : eval/test 模式下给模型续写的 prompt（到最后一个
                               assistant 之前 + `<|im_start|>assistant\\n`）
            last_reference   : 最后一个 assistant 的原文（评测参考）
        """
        msgs = self._prepend_system(conversations)

        full_text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )

        # 定位每个 assistant reply 在 full_text 中的 [start, end)。
        # 策略：对每个 assistant 消息，在 full_text 中找其 content 的子串位置，
        # 并把区间扩到紧随其后的 "<|im_end|>"（包含末尾的 stop token，便于训练学到停止）。
        assistant_spans: List[Tuple[int, int]] = []
        search_from = 0
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            pos = full_text.find(content, search_from)
            if pos < 0:
                # content 在 chat template 里可能被 strip/escape；退而求其次：
                # 找 "<|im_start|>assistant\n" 段落，把整段视为 assistant span
                tag = "<|im_start|>assistant\n"
                tag_pos = full_text.find(tag, search_from)
                if tag_pos < 0:
                    continue
                start = tag_pos + len(tag)
                end = full_text.find(IM_END, start)
                end = end + len(IM_END) if end >= 0 else len(full_text)
                assistant_spans.append((start, end))
                search_from = end
                continue

            end = pos + len(content)
            im_end_pos = full_text.find(IM_END, end)
            if im_end_pos >= 0 and im_end_pos - end <= 8:
                end = im_end_pos + len(IM_END)
            assistant_spans.append((pos, end))
            search_from = end

        # eval / test：去掉最后一个 assistant，让模型续写
        last_reference = ""
        prompt_text = ""
        if conversations and conversations[-1].get("role") == "assistant":
            last_reference = (conversations[-1].get("content") or "").strip()
            msgs_wo_last = self._prepend_system(conversations[:-1])
        else:
            msgs_wo_last = msgs
        prompt_text = self.tokenizer.apply_chat_template(
            msgs_wo_last, tokenize=False, add_generation_prompt=True,
        )

        return full_text, assistant_spans, prompt_text, last_reference

    # -------------------------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        item = self.dataset[idx]

        image_rel = item["image"]
        image_path = os.path.join(self.image_root, image_rel)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")

        conversations = item["conversations"]
        full_text, assistant_spans, prompt_text, last_reference = \
            self._build_prompt(conversations)

        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

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

            # 极端兜底：整段被截断导致没有任何 assistant token 留下
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
                "source": item.get("source", ""),
            }

        # eval / test
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
            "source": item.get("source", ""),
        }


# -----------------------------------------------------------------------------
# Collator
# -----------------------------------------------------------------------------
class MultiModalCollator:
    """
    只负责 padding：input_ids / attention_mask / labels。
    labels 已在 Dataset 内预先生成（非 assistant 段 = -100），这里补 -100 到 max_len。
    """

    def __init__(self, tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
        references = [x["reference"] for x in batch]
        sources = [x.get("source", "") for x in batch]
        image_paths = [x.get("image_path", "") for x in batch]

        input_ids_list = [x["input_ids"] for x in batch]
        attn_list = [x["attention_mask"] for x in batch]
        labels_list = [x["labels"] for x in batch]

        max_seq_len = max(x.size(0) for x in input_ids_list)
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)

        pad_id = self.tokenizer.pad_token_id

        padded_ids, padded_attn, padded_labels = [], [], []
        for ids, attn, lbls in zip(input_ids_list, attn_list, labels_list):
            ids = ids[:max_seq_len]
            attn = attn[:max_seq_len]
            lbls = lbls[:max_seq_len]

            pad_len = max_seq_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
                attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
                lbls = torch.cat([lbls, torch.full((pad_len,), -100, dtype=lbls.dtype)])

            padded_ids.append(ids)
            padded_attn.append(attn)
            padded_labels.append(lbls)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(padded_ids, dim=0),
            "attention_mask": torch.stack(padded_attn, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "references": references,
            "sources": sources,
            "image_paths": image_paths,
        }


def _resolve_image_root(json_path: str, shared_root: Optional[str]) -> str:
    """解析图像根目录。

    - ``shared_root`` 非空：三套 split 共用（兼容单一 ``stage3_data/images`` 目录）。
    - 否则：``<json_path 所在目录>/images``，与 prepare_train_data / prepare_val_data 等分目录产物一致。
    """
    if shared_root:
        return os.path.abspath(shared_root)
    return os.path.join(os.path.dirname(os.path.abspath(json_path)), "images")


# -----------------------------------------------------------------------------
# build_dataloaders
# -----------------------------------------------------------------------------
def build_dataloaders(
    vision_model_name: str,
    qwen_model_name: str,
    train_json: str,
    val_json: str,
    test_json: str,
    image_root: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 1024,
):
    processor = AutoProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    shared = (image_root or "").strip() or None

    train_ds = MultiModalInstructDataset(
        json_path=train_json,
        image_root=_resolve_image_root(train_json, shared),
        processor=processor, tokenizer=tokenizer,
        max_length=max_length, mode="train",
    )
    val_ds = MultiModalInstructDataset(
        json_path=val_json,
        image_root=_resolve_image_root(val_json, shared),
        processor=processor, tokenizer=tokenizer,
        max_length=max_length, mode="eval",
    )
    test_ds = MultiModalInstructDataset(
        json_path=test_json,
        image_root=_resolve_image_root(test_json, shared),
        processor=processor, tokenizer=tokenizer,
        max_length=max_length, mode="test",
    )

    collator = MultiModalCollator(tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collator, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collator, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="./stage3_data/train.json")
    parser.add_argument("--val_json", type=str, default="./stage3_data/val.json")
    parser.add_argument("--test_json", type=str, default="./stage3_data/test.json")
    parser.add_argument("--image_root", type=str, default="", help="可选；空则从各 json 同级目录下 images/ 解析")
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()
    shared_img = (args.image_root or "").strip() or None

    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 60)
    print("[smoke] Dataset mode=train")
    print("=" * 60)
    ds_train = MultiModalInstructDataset(
        json_path=args.train_json,
        image_root=_resolve_image_root(args.train_json, shared_img),
        processor=processor, tokenizer=tokenizer,
        max_length=args.max_length, mode="train",
    )
    print(f"[smoke] train size = {len(ds_train)}")

    rng = random.Random(0)
    sample_idx = rng.sample(range(len(ds_train)), k=min(args.num_samples, len(ds_train)))
    for i in sample_idx:
        s = ds_train[i]
        num_label_tokens = int((s["labels"] != -100).sum().item())
        print(f"\n--- sample {i} (source={s['source']}) ---")
        print(f"  input_ids shape   : {tuple(s['input_ids'].shape)}")
        print(f"  attention_mask sum: {int(s['attention_mask'].sum().item())}")
        print(f"  labels (non -100) : {num_label_tokens}")
        print(f"  reference        : {s['reference'][:80]}{'...' if len(s['reference'])>80 else ''}")
        print(f"  image_path       : {s['image_path']}")

    print("\n" + "=" * 60)
    print("[smoke] Collator batch")
    print("=" * 60)
    collator = MultiModalCollator(tokenizer, max_length=args.max_length)
    batch = collator([ds_train[i] for i in sample_idx[: args.batch_size]])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} len={len(v)}")

    print("\n" + "=" * 60)
    print("[smoke] Dataset mode=test (eval)")
    print("=" * 60)
    if os.path.exists(args.test_json):
        ds_test = MultiModalInstructDataset(
            json_path=args.test_json,
            image_root=_resolve_image_root(args.test_json, shared_img),
            processor=processor, tokenizer=tokenizer,
            max_length=args.max_length, mode="test",
        )
        print(f"[smoke] test size = {len(ds_test)}")
        s = ds_test[0]
        print(f"  prompt_len      : {s['prompt_len']}")
        print(f"  reference       : {s['reference'][:80]}{'...' if len(s['reference'])>80 else ''}")
