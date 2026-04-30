"""
Stage3 推理 demo：展示训练后模型在 caption / long_caption / 多轮 VQA 上的生成效果。

支持两种模式：

A) 数据集采样模式（默认）
   从 val / test split 里顺序取若干样本，打印"Question / Reference / Prediction"：
   python demo.py --config ./config/training_stage3.yaml \
                  --checkpoint ./outputs/checkpoint-best.pt \
                  --split test --max_samples 6 --source all

B) 自定义图像模式
   给定单张本地图片 + 自由 prompt，直接走一次 generate：
   python demo.py --config ./config/training_stage3.yaml \
                  --checkpoint ./outputs/checkpoint-best.pt \
                  --image ./stage3_data/images/textcaps/xxx.jpg \
                  --prompt "Describe this image in detail."

检查点不指定时，自动从 save_dir 选：best > latest。
"""

import argparse
import os
from collections import defaultdict
from typing import List, Optional

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from data.dataset import build_dataloaders
from models.multimodal_model import create_multimodal_model
from utils.checkpoint import create_checkpoint_manager


IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "You are a helpful assistant."


# -----------------------------------------------------------------------------
# 打印工具
# -----------------------------------------------------------------------------
def _bar(ch: str = "=", width: int = 80) -> str:
    return ch * width


def _print_sample(
    idx: int, source: str, image_path: str,
    question: str, reference: str, prediction: str,
):
    print(_bar("="))
    print(f"Sample #{idx}  |  source = {source}  |  image = {image_path}")
    print(_bar("-"))
    print("Question / Prompt:")
    print(question)
    print(_bar("-"))
    print("Reference:")
    print(reference or "<none>")
    print(_bar("-"))
    print("Prediction:")
    print(prediction)
    print(_bar("="))
    print()


# -----------------------------------------------------------------------------
# 模式 A：从 dataloader 采样
# -----------------------------------------------------------------------------
@torch.no_grad()
def demo_from_dataloader(
    model,
    dataloader,
    tokenizer,
    device: torch.device,
    max_samples: int = 6,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    do_sample: bool = False,
    source_filter: Optional[str] = None,
    per_source_cap: Optional[int] = None,
):
    """
    Args:
        source_filter:  只看某一类数据（"caption" / "long_caption" / "vqa"），
                        None 或 "all" 不过滤
        per_source_cap: 每个 source 最多展示多少条；None 则不限
    """
    model.eval()

    shown = 0
    per_source_shown = defaultdict(int)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        # stage3 的 generate() 基于 inputs_embeds，返回的 ids 只含新 token，无需切片
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )

        pred_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        pred_texts = [t.strip() for t in pred_texts]

        questions = tokenizer.batch_decode(
            input_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        refs = list(batch.get("references", [""] * len(pred_texts)))
        sources = list(batch.get("sources", [""] * len(pred_texts)))
        image_paths = list(batch.get("image_paths", [""] * len(pred_texts)))

        for q, ref, pred, src, img in zip(questions, refs, pred_texts, sources, image_paths):
            if source_filter and source_filter != "all" and src != source_filter:
                continue
            if per_source_cap is not None and per_source_shown[src] >= per_source_cap:
                continue

            _print_sample(
                idx=shown, source=src or "unknown", image_path=img,
                question=q.strip(), reference=ref, prediction=pred,
            )
            shown += 1
            per_source_shown[src] += 1
            if shown >= max_samples:
                return


# -----------------------------------------------------------------------------
# 模式 B：自定义图像 + prompt
# -----------------------------------------------------------------------------
@torch.no_grad()
def demo_from_image(
    model,
    processor,
    tokenizer,
    device: torch.device,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    do_sample: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # 用户消息里嵌入图像占位符（与训练对齐）
    user_content = f"{IMAGE_PLACEHOLDER}\n{prompt.strip()}" if prompt.strip() \
        else IMAGE_PLACEHOLDER
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    generated_ids = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    pred = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    _print_sample(
        idx=0, source="custom", image_path=image_path,
        question=prompt, reference="", prediction=pred,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _resolve_checkpoint(config, args) -> Optional[str]:
    if args.checkpoint:
        return args.checkpoint
    ckpt_mgr = create_checkpoint_manager(
        output_dir=config.get("save_dir", "./outputs"),
        max_checkpoints=int(config.get("max_checkpoints", 5)),
        save_trainable_only=bool(config.get("save_trainable_only", True)),
    )
    return ckpt_mgr.find_best_checkpoint() or ckpt_mgr.find_latest_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="Stage3 Demo Inference")
    parser.add_argument("--config", type=str, default="./config/training_stage3.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径；不指定则自动选 best > latest")

    # 模式 A：dataloader 采样
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=6)
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "caption", "long_caption", "vqa"])
    parser.add_argument("--per_source_cap", type=int, default=None,
                        help="每种 source 至多展示多少条（source=all 时更有用）")

    # 模式 B：单图
    parser.add_argument("--image", type=str, default=None,
                        help="给定后切换到自定义单图模式")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device = {device}")

    # 模型
    model = create_multimodal_model(config)
    model.to(device)

    # 检查点
    ckpt_path = _resolve_checkpoint(config, args)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt_mgr = create_checkpoint_manager(
            output_dir=config.get("save_dir", "./outputs"),
            max_checkpoints=int(config.get("max_checkpoints", 5)),
            save_trainable_only=bool(config.get("save_trainable_only", True)),
        )
        ckpt_mgr.load_checkpoint(ckpt_path, model)
    else:
        print("[warn] no checkpoint found, running with randomly-initialized projector.")

    ds_cfg = config["dataset"]
    gen_cfg = config.get("generation", {})
    max_new_tokens = args.max_new_tokens or gen_cfg.get("max_new_tokens", 256)
    num_beams = args.num_beams or gen_cfg.get("num_beams", 1)
    do_sample = args.do_sample or bool(gen_cfg.get("do_sample", False))

    # ---------------- 模式 B：单图 ----------------
    if args.image is not None:
        processor = AutoProcessor.from_pretrained(ds_cfg["vision_model_name"])
        tokenizer = AutoTokenizer.from_pretrained(ds_cfg["qwen_model_name"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        demo_from_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            image_path=args.image,
            prompt=args.prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        return

    # ---------------- 模式 A：dataloader ----------------
    train_loader, val_loader, test_loader = build_dataloaders(
        vision_model_name=ds_cfg["vision_model_name"],
        qwen_model_name=ds_cfg["qwen_model_name"],
        repo_id=ds_cfg.get("hf_repo_id"),
        batch_size=ds_cfg["batch_size"],
        num_workers=ds_cfg.get("num_workers", 2),
        max_length=ds_cfg.get("max_length", 1024),
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    tokenizer = AutoTokenizer.from_pretrained(ds_cfg["qwen_model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    demo_from_dataloader(
        model=model,
        dataloader=loader,
        tokenizer=tokenizer,
        device=device,
        max_samples=args.max_samples,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        source_filter=args.source,
        per_source_cap=args.per_source_cap,
    )


if __name__ == "__main__":
    main()
