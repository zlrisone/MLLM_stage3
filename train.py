"""
Stage3 多模态指令微调训练脚本。

- 仅 Projector（以及可选 LoRA）可训练，vision encoder / LLM 冻结
- 支持 gradient_accumulation_steps
- 支持 cosine + warmup 调度
- 使用 CheckpointManager 做至多 N 个检查点的滚动保存，以及 latest / best 维护
- 支持从 --resume <path> 或 --auto_resume 恢复训练
- 训练结束可选地用 eval.py 跑测试集生成（若 eval.py 为空则跳过）
"""

import argparse
import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.dataset import build_dataloaders
from models.multimodal_model import create_multimodal_model
from utils.checkpoint import create_checkpoint_manager

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# -----------------------------------------------------------------------------
# 工具
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, opt_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"[model] trainable params: {n_train:,} / total: {n_total:,} "
          f"({100.0 * n_train / max(n_total, 1):.3f}%)")

    name = opt_cfg.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            trainable,
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: Dict[str, Any],
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    name = (sched_cfg.get("name") or "").lower()
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))

    if name == "cosine":
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if name in ("linear",):
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - warmup_steps)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return None


# -----------------------------------------------------------------------------
# validate
# -----------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Val   Epoch {epoch + 1:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        if loss is None:
            continue

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_loss / num_batches:.4f}",
        )

    return total_loss / max(num_batches, 1)


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: Dict[str, Any],
    ckpt_mgr,
    start_epoch: int = 0,
    start_global_step: int = 0,
    best_val_loss: float = float("inf"),
):
    num_epochs = int(config["epochs"])
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    logging_steps = int(config.get("logging_steps", 10))
    eval_steps = int(config.get("eval_steps", 500))
    save_steps = int(config.get("save_steps", 1000))

    use_wandb = bool(config.get("use_wandb", False)) and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.get("project_name", "multimodal-caption"),
            name=config.get("run_name", "stage3"),
            config=config,
            mode=config.get("wandb_mode", "online"),
            resume="allow",
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/global_step")
        wandb.define_metric("eval/*", step_metric="eval/global_step")

    global_step = start_global_step

    for epoch in range(start_epoch, num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1:02d}",
            leave=True,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            if loss is None:
                continue

            loss_for_log = loss.detach().float().item()
            (loss / grad_accum).backward()

            # 只有在 accumulate 满 grad_accum 步时才真正更新
            do_update = ((step + 1) % grad_accum == 0) or (step + 1 == len(train_loader))
            grad_norm = None
            if do_update:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_grad_norm,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += loss_for_log
            num_batches += 1
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / num_batches
            pbar.set_postfix(
                loss=f"{loss_for_log:.4f}",
                avg=f"{avg_loss:.4f}",
                lr=f"{current_lr:.2e}",
                step=global_step,
            )

            if do_update and logging_steps > 0 and global_step % logging_steps == 0:
                log_dict = {
                    "train/global_step": global_step,
                    "train/epoch": epoch + 1,
                    "train/loss": loss_for_log,
                    "train/avg_loss": avg_loss,
                    "train/lr": current_lr,
                }
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = float(grad_norm)
                if use_wandb:
                    wandb.log(log_dict)

            # step 级验证
            if (
                do_update
                and eval_steps > 0
                and global_step % eval_steps == 0
                and global_step > 0
            ):
                val_loss = validate(model, val_loader, device, epoch)
                print(f"[eval @ step {global_step}] val_loss={val_loss:.4f}")
                if use_wandb:
                    wandb.log({
                        "eval/global_step": global_step,
                        "eval/epoch": epoch + 1,
                        "eval/loss": val_loss,
                    })

                is_best = val_loss < best_val_loss - 1e-4
                if is_best:
                    best_val_loss = val_loss
                    if use_wandb:
                        wandb.log({
                            "eval/global_step": global_step,
                            "eval/best_val_loss": best_val_loss,
                        })

                if save_steps > 0:
                    ckpt_mgr.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        val_loss=val_loss,
                        config=config,
                        is_best=is_best,
                    )
                model.train()

            # step 级纯保存（与 eval 解耦，防止 eval_steps 不整除 save_steps 时丢点）
            if (
                do_update
                and save_steps > 0
                and global_step % save_steps == 0
                and global_step > 0
                and (eval_steps <= 0 or global_step % eval_steps != 0)
            ):
                ckpt_mgr.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    val_loss=float("nan"),
                    config=config,
                    is_best=False,
                )

        # epoch 末验证 + 保存
        epoch_avg_loss = total_loss / max(num_batches, 1)
        val_loss = validate(model, val_loader, device, epoch)
        print(
            f"[epoch {epoch + 1}/{num_epochs}] "
            f"train_avg_loss={epoch_avg_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if use_wandb:
            wandb.log({
                "eval/global_step": global_step,
                "eval/epoch": epoch + 1,
                "eval/epoch_train_avg_loss": epoch_avg_loss,
                "eval/loss": val_loss,
            })

        is_best = val_loss < best_val_loss - 1e-4
        if is_best:
            best_val_loss = val_loss
            if use_wandb:
                wandb.log({
                    "eval/global_step": global_step,
                    "eval/best_val_loss": best_val_loss,
                })

        ckpt_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            val_loss=val_loss,
            config=config,
            is_best=is_best,
            extra={"end_of_epoch": True},
        )

    # 最终权重（只含模型）
    ckpt_mgr.save_final_model(model, config)

    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["final_global_step"] = global_step

    return best_val_loss, global_step


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage3 Training: Multimodal Instruction Tuning")
    parser.add_argument("--config", type=str, default="./config/training_stage3.yaml")
    parser.add_argument("--resume", type=str, default=None, help="从指定检查点恢复")
    parser.add_argument("--auto_resume", action="store_true",
                        help="自动从 save_dir 下的 latest 检查点恢复（若存在）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("dataset", {}).get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device = {device}")

    # 1) 模型
    model = create_multimodal_model(config)
    model.to(device)

    # 2) 数据
    ds_cfg = config["dataset"]
    train_loader, val_loader, test_loader = build_dataloaders(
        vision_model_name=ds_cfg["vision_model_name"],
        qwen_model_name=ds_cfg["qwen_model_name"],
        repo_id=ds_cfg.get("hf_repo_id"),
        batch_size=ds_cfg["batch_size"],
        num_workers=ds_cfg.get("num_workers", 2),
        max_length=ds_cfg.get("max_length", 1024),
    )
    print(f"[data] train batches={len(train_loader)} | "
          f"val batches={len(val_loader)} | test batches={len(test_loader)}")

    # 3) 优化器 / 调度器
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    num_epochs = int(config["epochs"])
    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    num_training_steps = num_update_steps_per_epoch * num_epochs

    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}), num_training_steps)

    # 4) checkpoint manager
    ckpt_mgr = create_checkpoint_manager(
        output_dir=config.get("save_dir", "./outputs"),
        max_checkpoints=int(config.get("max_checkpoints", 5)),
        save_trainable_only=bool(config.get("save_trainable_only", True)),
    )

    # 5) resume
    start_epoch = 0
    start_global_step = 0
    best_val_loss = float("inf")

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = ckpt_mgr.find_latest_checkpoint()
        if resume_path:
            print(f"[resume] auto-resume from {resume_path}")

    if resume_path:
        info = ckpt_mgr.load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = int(info.get("epoch", 0))
        start_global_step = int(info.get("global_step", 0))
        best_val_loss = float(info.get("val_loss", float("inf")))
        # 如果是 epoch 末尾的 ckpt，下一次从下一个 epoch 开始
        if info.get("end_of_epoch", False):
            start_epoch += 1

    # 6) train
    best_val_loss, final_step = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        ckpt_mgr=ckpt_mgr,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        best_val_loss=best_val_loss,
    )
    print(f"[done] training finished. best_val_loss={best_val_loss:.6f} | "
          f"final_step={final_step}")

    # 7) 可选：测试集生成评估（依赖 eval.py 是否已实现）
    try:
        from eval import evaluate_generation, print_caption_metrics  # type: ignore
    except (ImportError, AttributeError):
        print("[test] eval.py not implemented yet, skip test-set evaluation.")
        if _WANDB_AVAILABLE and bool(config.get("use_wandb", False)):
            wandb.finish()
        return

    tokenizer = AutoTokenizer.from_pretrained(ds_cfg["qwen_model_name"], use_fast=True)
    gen_cfg = config.get("generation", {})
    save_json_path = os.path.join(
        config.get("save_dir", "./outputs"), "test_predictions.json"
    )

    test_metrics = evaluate_generation(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=gen_cfg.get("max_new_tokens", 512),
        num_beams=gen_cfg.get("num_beams", 1),
        do_sample=gen_cfg.get("do_sample", False),
        save_json_path=save_json_path,
        log_to_wandb=bool(config.get("use_wandb", False)) and _WANDB_AVAILABLE,
        prefix="test",
    )
    print_caption_metrics(test_metrics, prefix="Test")

    if _WANDB_AVAILABLE and bool(config.get("use_wandb", False)):
        wandb.finish()


if __name__ == "__main__":
    main()
