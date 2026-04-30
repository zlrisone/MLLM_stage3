"""
检查点管理模块（Stage3）

Stage3 中 Vision Encoder 和 LLM 默认是冻结的，仅 Projector（以及可选的 LoRA 参数）
需要训练。为节省磁盘空间，默认只保存可训练参数；加载时使用 strict=False。
"""

import json
import torch
import torch.nn as nn

from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    只导出 requires_grad=True 的参数（Projector / LoRA 等）
    """
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    full_state = model.state_dict()
    return {k: v for k, v in full_state.items() if k in trainable_names}


class CheckpointManager:
    """
    检查点管理器

    - 按 step 命名：checkpoint-epoch{E:03d}-step{S:06d}.pt
    - 自动维护 latest / best 软副本（直接写文件，跨平台可用）
    - 至多保留 max_checkpoints 个 step 检查点，超出按 mtime 淘汰最旧的
    """

    STEP_PATTERN = "checkpoint-epoch*-step*.pt"
    LATEST_NAME = "checkpoint-latest.pt"
    BEST_NAME = "checkpoint-best.pt"

    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 5,
        save_trainable_only: bool = True,
    ):
        """
        Args:
            output_dir:          检查点保存目录
            max_checkpoints:     最多保留的 step 检查点数量（latest/best 不计入）
            save_trainable_only: 是否只保存 requires_grad=True 的参数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_trainable_only = save_trainable_only

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        val_loss: float,
        config: Dict[str, Any],
        is_best: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        保存一个检查点，并同步更新 latest（以及当 is_best=True 时更新 best）

        Returns:
            step 检查点的绝对路径
        """
        if filename is None:
            filename = f"checkpoint-epoch{epoch:03d}-step{step:06d}.pt"

        ckpt_path = self.output_dir / filename

        if self.save_trainable_only:
            model_state = _get_trainable_state_dict(model)
        else:
            model_state = model.state_dict()

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": step,
            "val_loss": val_loss,
            "model_state_dict": model_state,
            "trainable_only": self.save_trainable_only,
            "config": config,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if extra is not None:
            checkpoint.update(extra)

        torch.save(checkpoint, ckpt_path)

        config_path = self.output_dir / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except TypeError:
            # 配置中可能存在非 JSON 可序列化对象时，安静跳过
            pass

        self._update_copy(ckpt_path, self.LATEST_NAME)
        if is_best:
            self._update_copy(ckpt_path, self.BEST_NAME)

        print(f"[ckpt] saved: {ckpt_path} (val_loss={val_loss:.4f}, best={is_best})")

        self._cleanup_old_checkpoints()

        return str(ckpt_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cpu",
        strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        加载检查点到给定的模型 / 优化器 / 调度器

        Args:
            strict: None 表示根据 checkpoint 中的 trainable_only 自动决定：
                    仅保存了可训练参数时用 strict=False；全量保存时用 strict=True
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        trainable_only = checkpoint.get("trainable_only", False)
        if strict is None:
            strict = not trainable_only

        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=strict
        )
        if not strict:
            # 只保存了 projector/LoRA 的情况下，frozen 的 vision/llm 参数属于 missing，正常现象
            if unexpected:
                print(f"[ckpt] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(
            f"[ckpt] loaded: {checkpoint_path} | "
            f"epoch={checkpoint.get('epoch')}, "
            f"step={checkpoint.get('global_step')}, "
            f"val_loss={checkpoint.get('val_loss', float('nan')):.4f}"
        )

        return checkpoint

    def find_latest_checkpoint(self) -> Optional[str]:
        latest_path = self.output_dir / self.LATEST_NAME
        if latest_path.exists():
            return str(latest_path)

        files = self._list_step_checkpoints()
        if not files:
            return None
        return str(files[0])

    def find_best_checkpoint(self) -> Optional[str]:
        best_path = self.output_dir / self.BEST_NAME
        if best_path.exists():
            return str(best_path)
        return None

    def save_final_model(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        filename: str = "final_model.pt",
    ) -> str:
        """
        仅保存模型权重（不含 optimizer / scheduler），用于部署或下游使用
        """
        final_path = self.output_dir / filename
        model_state = (
            _get_trainable_state_dict(model)
            if self.save_trainable_only
            else model.state_dict()
        )
        torch.save(
            {
                "model_state_dict": model_state,
                "trainable_only": self.save_trainable_only,
                "config": config,
            },
            final_path,
        )
        print(f"[ckpt] final model saved: {final_path}")
        return str(final_path)

    def _list_step_checkpoints(self) -> List[Path]:
        """按修改时间从新到旧排序，仅包含 step 形式的检查点"""
        files = list(self.output_dir.glob(self.STEP_PATTERN))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def _update_copy(self, src: Path, dst_name: str):
        """
        将 src 复制到 output_dir/dst_name（例如 latest/best）。
        采用读写方式而非 symlink，Windows 与部分文件系统也可用。
        """
        dst = self.output_dir / dst_name
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            # 复制字节流
            with open(src, "rb") as fr, open(dst, "wb") as fw:
                while True:
                    chunk = fr.read(1024 * 1024)
                    if not chunk:
                        break
                    fw.write(chunk)
        except Exception as e:
            print(f"[ckpt] failed to update {dst_name}: {e}")

    def _cleanup_old_checkpoints(self):
        """按 mtime 淘汰最旧的 step 检查点，直到数量 <= max_checkpoints"""
        if self.max_checkpoints <= 0:
            return

        files = self._list_step_checkpoints()
        if len(files) <= self.max_checkpoints:
            return

        for old in files[self.max_checkpoints:]:
            try:
                old.unlink()
                print(f"[ckpt] removed old checkpoint: {old.name}")
            except Exception as e:
                print(f"[ckpt] failed to remove {old}: {e}")


def create_checkpoint_manager(
    output_dir: str,
    max_checkpoints: int = 5,
    save_trainable_only: bool = True,
) -> CheckpointManager:
    return CheckpointManager(
        output_dir=output_dir,
        max_checkpoints=max_checkpoints,
        save_trainable_only=save_trainable_only,
    )
