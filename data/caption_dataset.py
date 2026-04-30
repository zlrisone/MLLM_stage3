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
    ) -> Tuple[str, str, str]:
        """
        使用 `apply_chat_template` 组装 prompt。

        返回：
            full_text        : 完整渲染后的字符串（train 模式 tokenize 这个）
            prompt_text      : eval/test 模式下给模型续写的 prompt（到最后一个
                               assistant 之前 + `<|im_start|>assistant\\n`）
            last_reference   : 最后一个 assistant 的原文（评测参考）
        """
        msgs = self._prepend_system(conversations)

        full_text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = full_text.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"   
        content = msgs[-1].get("content","")
        if self.mode == "test":
            last_reference =  content if content not in ("", "None", None) else None  
        else:
            last_reference = content

        return full_text, prompt_text, last_reference

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
        full_text, prompt_text, last_reference = self._build_prompt(conversations)

        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        if self.mode == "train":
            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)
        else:
            input_ids = prompt_enc["input_ids"].squeeze(0)
            attention_mask = prompt_enc["attention_mask"].squeeze(0)
        prompt_len = prompt_enc["input_ids"].size(1)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": prompt_len,
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

    def __init__(self, tokenizer, max_length: int = None, padding_side: str = "right"):
        """
        padding_side:
            - "right"：训练/计算 loss 时使用，labels 用右 padding 对齐。
            - "left" ：batch 生成时必须使用，否则 causal LM 的新 token 会被追加到 pad 之后，
                       导致短 prompt 的样本产出乱码（常见现象：prompt tail 泄漏，比如输出里冒出 "assistant"）。
        """
        assert padding_side in ("right", "left")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
        references = [x["reference"] for x in batch]

        input_ids_list = [x["input_ids"] for x in batch]
        attn_list = [x["attention_mask"] for x in batch]
        prompt_lens = [x["prompt_len"] for x in batch]

        max_seq_len = max(x.size(0) for x in input_ids_list)
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)

        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for ids, attn, prompt_len in zip(input_ids_list, attn_list, prompt_lens):
            ids = ids[:max_seq_len]
            attn = attn[:max_seq_len]
            prompt_len = min(prompt_len, max_seq_len)

            pad_len = max_seq_len - ids.size(0)

            pad_ids = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            pad_mask = torch.zeros(pad_len, dtype=attn.dtype)
            pad_labels = torch.full((pad_len,), -100, dtype=ids.dtype)

            labels_core = ids.clone()
            labels_core[:prompt_len] = -100
            
            if self.padding_side == "right":
                padded_ids = torch.cat([ids, pad_ids])
                padded_mask = torch.cat([attn, pad_mask])
                labels = torch.cat([labels_core, pad_labels]) if pad_len > 0 else labels_core
            else:  # left padding —— 生成专用
                padded_ids = torch.cat([pad_ids, ids])
                padded_mask = torch.cat([pad_mask, attn])
                labels = torch.cat([pad_labels, labels_core]) if pad_len > 0 else labels_core
            
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            padded_labels.append(labels)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "attention_mask": torch.stack(padded_attention_mask, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "references": references,
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
