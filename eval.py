"""
Stage3 生成式评测：caption / long_caption / 多轮 VQA 最后一轮。

- 使用 `MultimodalModel.generate`（基于 inputs_embeds），返回的 generated_ids
  只包含新生成 token，无需做 prompt_len 切片
- 整体指标 + 按 source 分组指标（caption / long_caption / vqa）

命令行用法：
    python eval.py --config ./config/training_stage3.yaml --resume ./outputs/checkpoint-best.pt
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.dataset import build_dataloaders
from models.multimodal_model import create_multimodal_model
from utils.checkpoint import create_checkpoint_manager
from utils.LM_metrics import evaluate_caption


# -----------------------------------------------------------------------------
# 生成 + 指标
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    do_sample: bool = False,
    save_json_path: Optional[str] = None,
    log_to_wandb: bool = False,
    prefix: str = "test",
    by_source: bool = True,
) -> Dict[str, Any]:
    """
    批量生成并做指标评测。

    Returns:
        metrics dict：包含整体 BLEU/ROUGE/CIDEr/METEOR，
        可选 per-source 细分指标（前缀 "by_source/<source>/<metric>"）
    """
    model.eval()

    all_predictions: List[str] = []
    all_references: List[Union[str, List[str]]] = []
    all_sources: List[str] = []
    prediction_records: List[Dict[str, Any]] = []

    pbar = tqdm(
        dataloader,
        desc=f"Evaluate {prefix}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        # inputs_embeds 路径：HF generate 只返回新生成的 token
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )

        pred_texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        pred_texts = [t.strip() for t in pred_texts]

        refs = list(batch.get("references", []))
        sources = list(batch.get("sources", ["" for _ in pred_texts]))
        image_paths = list(batch.get("image_paths", ["" for _ in pred_texts]))

        if len(pred_texts) != len(refs):
            raise ValueError(
                f"size mismatch: preds={len(pred_texts)}, refs={len(refs)}"
            )

        for pred, ref, src, img in zip(pred_texts, refs, sources, image_paths):
            all_predictions.append(pred)
            all_references.append(ref)
            all_sources.append(src)
            prediction_records.append({
                "sample_id": len(prediction_records),
                "source": src,
                "image_path": img,
                "prediction": pred,
                "reference": ref,
            })

    # --- 整体指标 ---
    metrics = evaluate_caption(all_predictions, all_references)

    # --- 按 source 分组指标 ---
    by_source_metrics: Dict[str, Dict[str, float]] = {}
    if by_source and any(s for s in all_sources):
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(all_sources):
            groups[s or "unknown"].append(i)

        for src, idxs in groups.items():
            if len(idxs) < 2:
                # pycocoevalcap 对样本数极少时不稳定
                continue
            sub_preds = [all_predictions[i] for i in idxs]
            sub_refs = [all_references[i] for i in idxs]
            try:
                sub_metrics = evaluate_caption(sub_preds, sub_refs)
            except Exception as e:
                print(f"[metrics] source={src} failed: {e}")
                continue
            clean = {k: v for k, v in sub_metrics.items()
                     if k != "per_sample" and isinstance(v, (int, float))}
            clean["num_samples"] = len(idxs)
            by_source_metrics[src] = clean

    # --- 保存详细预测 / 指标 ---
    if save_json_path is not None:
        save_path = Path(save_json_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        summary_metrics = {
            k: v for k, v in metrics.items()
            if k != "per_sample" and isinstance(v, (int, float))
        }
        save_obj = {
            "metrics": summary_metrics,
            "by_source": by_source_metrics,
            "num_samples": len(all_predictions),
            "predictions": prediction_records,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_obj, f, ensure_ascii=False, indent=2)
        print(f"[eval] results saved -> {save_json_path}")

    # --- wandb 日志 ---
    if log_to_wandb:
        try:
            import wandb
            log_dict: Dict[str, Any] = {}
            for k, v in metrics.items():
                if k == "per_sample" or not isinstance(v, (int, float)):
                    continue
                log_dict[f"{prefix}/{k}"] = v
            for src, sub in by_source_metrics.items():
                for k, v in sub.items():
                    log_dict[f"{prefix}/by_source/{src}/{k}"] = v
            if log_dict:
                wandb.log(log_dict)
        except Exception as e:
            print(f"[eval] wandb log failed: {e}")

    metrics["by_source"] = by_source_metrics
    return metrics


def print_caption_metrics(metrics: Dict[str, Any], prefix: str = "Eval"):
    keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]

    msg = [f"[{prefix}]"]
    for k in keys:
        if k in metrics:
            msg.append(f"{k}={metrics[k]:.4f}")
    print(" | ".join(msg))

    by_source = metrics.get("by_source", {})
    for src, sub in by_source.items():
        sub_msg = [f"  ↳ {src} (N={sub.get('num_samples', '?')})"]
        for k in keys:
            if k in sub:
                sub_msg.append(f"{k}={sub[k]:.4f}")
        print(" | ".join(sub_msg))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _pick_checkpoint_path(config: Dict[str, Any], args) -> Optional[str]:
    """优先级：--resume > best > latest"""
    if args.resume:
        return args.resume

    ckpt_mgr = create_checkpoint_manager(
        output_dir=config.get("save_dir", "./outputs"),
        max_checkpoints=int(config.get("max_checkpoints", 5)),
        save_trainable_only=bool(config.get("save_trainable_only", True)),
    )
    if args.use_latest:
        return ckpt_mgr.find_latest_checkpoint()
    return ckpt_mgr.find_best_checkpoint() or ckpt_mgr.find_latest_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="Stage3 Evaluation: generation metrics")
    parser.add_argument("--config", type=str, default="./config/training_stage3.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="指定检查点路径；不指定则优先 best，其次 latest")
    parser.add_argument("--use_latest", action="store_true",
                        help="强制使用 latest 检查点（覆盖 best 优先）")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test",
                        help="在哪个 split 上评测")
    parser.add_argument("--save_json", type=str, default=None,
                        help="预测明细输出 json 路径；默认 <save_dir>/<split>_predictions.json")
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

    # 加载检查点
    ckpt_path = _pick_checkpoint_path(config, args)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("[warn] no checkpoint found, evaluating with randomly-initialized projector.")
    else:
        ckpt_mgr = create_checkpoint_manager(
            output_dir=config.get("save_dir", "./outputs"),
            max_checkpoints=int(config.get("max_checkpoints", 5)),
            save_trainable_only=bool(config.get("save_trainable_only", True)),
        )
        ckpt_mgr.load_checkpoint(ckpt_path, model)

    # 数据
    ds_cfg = config["dataset"]
    train_loader, val_loader, test_loader = build_dataloaders(
        vision_model_name=ds_cfg["vision_model_name"],
        qwen_model_name=ds_cfg["qwen_model_name"],
        repo_id=ds_cfg.get("hf_repo_id"),
        batch_size=ds_cfg["batch_size"],
        num_workers=ds_cfg.get("num_workers", 2),
        max_length=ds_cfg.get("max_length", 1024),
    )
    loader = val_loader if args.split == "val" else test_loader
    print(f"[data] split={args.split} | batches={len(loader)}")

    tokenizer = AutoTokenizer.from_pretrained(ds_cfg["qwen_model_name"], use_fast=True)

    gen_cfg = config.get("generation", {})
    max_new_tokens = args.max_new_tokens or gen_cfg.get("max_new_tokens", 512)
    num_beams = args.num_beams or gen_cfg.get("num_beams", 1)
    do_sample = args.do_sample or bool(gen_cfg.get("do_sample", False))

    save_json = args.save_json or os.path.join(
        config.get("save_dir", "./outputs"),
        f"{args.split}_predictions.json",
    )

    metrics = evaluate_generation(
        model=model,
        dataloader=loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        save_json_path=save_json,
        log_to_wandb=False,
        prefix=args.split,
        by_source=True,
    )
    print_caption_metrics(metrics, prefix=args.split.capitalize())


if __name__ == "__main__":
    main()
