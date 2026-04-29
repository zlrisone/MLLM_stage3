"""
将本地 prepare_* 脚本生成的 train / val / test 三套数据一并上传到 HuggingFace Hub。

约定（与 prepare_train_data / prepare_val_data / prepare_test_data 一致）：
    每个根目录下包含 ``chat.json`` 与 ``images/``，``chat.json`` 里每条样本的 ``image``
    为相对 ``images/`` 的路径；``conversations`` 为 ``[{"role","content"}, ...]``。

上传为一个 ``DatasetDict``，split 名称：
    ``train`` | ``validation`` | ``test``
（本地验证集目录对应 Hub 上的 ``validation`` split）

用法（在 ``stage3`` 项目根目录执行示例）：
    python prepare_data/upload_data.py \\
        --repo-id YOUR_USERNAME/stage3_mm \\
        --train-dir ./stage3_train_data \\
        --val-dir ./stage3_val_data \\
        --test-dir ./stage3_test_data

需已登录 HuggingFace CLI（``huggingface-cli login``）或设置环境变量 ``HF_TOKEN``。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

DEFAULT_TRAIN_DIR = "./stage3_train_data"
DEFAULT_VAL_DIR = "./stage3_val_data"
DEFAULT_TEST_DIR = "./stage3_test_data"


def load_chat_records(root: str) -> List[Dict[str, Any]]:
    path = os.path.join(os.path.abspath(root), "chat.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 chat.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层应为 JSON 数组")
    return data


def records_to_hub_rows(records: List[Dict[str, Any]], image_root: str) -> List[Dict[str, Any]]:
    """image_root 一般为 ``{root}/images``。"""
    rows: List[Dict[str, Any]] = []
    for r in records:
        rel = r.get("image")
        if not rel:
            raise ValueError(f"样本缺少 image 字段: id={r.get('id')}")
        abs_path = os.path.join(image_root, rel)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"图像不存在: {abs_path} (id={r.get('id')})")
        rows.append({
            "id": r["id"],
            "source": r.get("source", ""),
            "image": abs_path,
            "conversations": r["conversations"],
        })
    return rows


def push_datasetdict_to_hub(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    repo_id: str,
    *,
    private: bool,
    token: str | None,
    max_shard_size: str | None,
) -> None:
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence, Value

    features = Features({
        "id": Value("string"),
        "source": Value("string"),
        "image": HFImage(),
        "conversations": Sequence({
            "role": Value("string"),
            "content": Value("string"),
        }),
    })

    def _ds(rows: List[Dict[str, Any]], split_name: str):
        if not rows:
            print(f"[warn] split {split_name!r} 为空，仍将创建空表（可能对 Hub 评测脚本不便）")
        return Dataset.from_list(rows, features=features)

    dsd = DatasetDict({
        "train": _ds(train_rows, "train"),
        "validation": _ds(val_rows, "validation"),
        "test": _ds(test_rows, "test"),
    })

    kwargs: Dict[str, Any] = {"private": private}
    if token:
        kwargs["token"] = token
    if max_shard_size:
        kwargs["max_shard_size"] = max_shard_size

    print(f"[upload] push_to_hub repo_id={repo_id!r} private={private} ...")
    dsd.push_to_hub(repo_id, **kwargs)
    print(f"[upload] done -> https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将本地 train/val/test 三套 Stage3 数据上传到 HuggingFace Hub（单个 DatasetDict）",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hub 数据集仓库 ID，例如 username/stage3_mm_data",
    )
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR, help="训练集根目录（含 chat.json、images/）")
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR, help="验证集根目录")
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR, help="测试集根目录")
    parser.add_argument(
        "--public",
        action="store_true",
        help="公开数据集（默认上传到私有仓库）",
    )
    parser.add_argument("--token", type=str, default=None, help="HF token；默认读环境变量或已 login 缓存")
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default=None,
        help="可选，如 ``500MB``，大库可限分片体积",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅加载并校验文件，不执行 push_to_hub",
    )
    args = parser.parse_args()

    private = not args.public

    train_root = os.path.abspath(args.train_dir)
    val_root = os.path.abspath(args.val_dir)
    test_root = os.path.abspath(args.test_dir)

    print(f"[load] train_dir  = {train_root}")
    print(f"[load] val_dir    = {val_root}")
    print(f"[load] test_dir   = {test_root}")

    train_recs = load_chat_records(train_root)
    val_recs = load_chat_records(val_root)
    test_recs = load_chat_records(test_root)

    train_rows = records_to_hub_rows(train_recs, os.path.join(train_root, "images"))
    val_rows = records_to_hub_rows(val_recs, os.path.join(val_root, "images"))
    test_rows = records_to_hub_rows(test_recs, os.path.join(test_root, "images"))

    print(f"[load] counts train={len(train_rows)} validation={len(val_rows)} test={len(test_rows)}")

    if args.dry_run:
        print("[dry-run] skip push_to_hub")
        return

    push_datasetdict_to_hub(
        train_rows,
        val_rows,
        test_rows,
        args.repo_id,
        private=private,
        token=args.token,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
