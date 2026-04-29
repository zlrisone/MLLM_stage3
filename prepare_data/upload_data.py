"""
将本地 prepare_* 脚本生成的 train / val / test 三套数据一并上传到 HuggingFace Hub。

约定（与 prepare_train_data / prepare_val_data / prepare_test_data 一致）：
    每个根目录下包含 ``chat.json`` 与 ``images/``。
    **上传到 Hub 的每条样本**在 TextCaps 三列基础上增加 ``id``，共四列：
    - ``id``：字符串，优先取 ``chat.json`` 中样本的 ``id``；缺失则生成为 ``idx_<序号>``
    - ``image``：``datasets.Image``，``load_dataset`` 后用下标访问为 ``PIL.Image.Image``
    - ``conversations``：对话列表 ``[{"role","content"}, ...]``（多模态指令与回复）
    - ``source``：字符串，样本来源标签（如 ``caption`` / ``vqa``）

上传为一个 ``DatasetDict``，split 名称：
    ``train`` | ``validation`` | ``test``
（本地验证集目录对应 Hub 上的 ``validation`` split）

存储格式：
    ``DatasetDict.push_to_hub`` 以 **Parquet 分片** 写入 Hub。默认 ``embed_external_files=True``
    将图像字节写入 Parquet，下载后可直接得到解码后的 ``PIL.Image``。

用法（在 ``stage3`` 项目根目录执行示例）：
    python upload_data.py \
        --repo-id Lris47/MLLMstage3-textcaps-scienceqa-textvqa \
        --train-dir ./stage3_train_data \
        --val-dir ./stage3_val_data \
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


def _normalize_conversation_messages(raw: Any, sample_id: Any) -> List[Dict[str, str]]:
    """Hub 列 ``List({role, content})`` 要求每条消息为 ``dict``；部分 JSON 可能为 ``[role, content]``。"""
    if raw is None:
        raise ValueError(f"样本缺少 conversations: id={sample_id}")
    if not isinstance(raw, list):
        raise TypeError(f"conversations 应为 list，got {type(raw)!r} id={sample_id}")
    out: List[Dict[str, str]] = []
    for i, msg in enumerate(raw):
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role is None and "from" in msg:
                role = msg.get("from")
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            role, content = msg[0], msg[1]
        else:
            raise TypeError(
                f"conversations[{i}] 须为 dict 或 [role, content] 二元组，"
                f"got {type(msg)!r} id={sample_id}"
            )
        out.append({
            "role": "" if role is None else str(role),
            "content": "" if content is None else str(content),
        })
    return out


def records_to_hub_rows(records: List[Dict[str, Any]], image_root: str) -> List[Dict[str, Any]]:
    """构造上传行：id / image / conversations / source。

    ``image`` 此处为本地绝对路径字符串，交给 ``datasets.Image`` 编码；推送 Parquet 并
    ``embed_external_files=True`` 时，Hub 端为内嵌图像；``load_dataset`` 后列为
    ``PIL.Image.Image``。
    """
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        raw_id = r.get("id")
        sample_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else f"idx_{i:09d}"
        rel = r.get("image")
        if not rel:
            raise ValueError(f"样本缺少 image 字段: id={sample_id}")
        abs_path = os.path.join(image_root, rel)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"图像不存在: {abs_path} (id={sample_id})")
        rows.append({
            "id": sample_id,
            "image": abs_path,
            "conversations": _normalize_conversation_messages(r.get("conversations"), sample_id),
            "source": r.get("source", "") or "",
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
    embed_external_files: bool,
    num_proc: int | None,
    commit_message: str | None,
) -> None:
    """调用 ``DatasetDict.push_to_hub``：Hub 侧为 Parquet；列顺序 image / conversations / source。"""
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, List, Value

    # 须用 List(struct)，勿用 Sequence({...})：后者在 HF datasets 中会展开成
    # ``{role: List(...), content: List(...)}``（按列对齐），与 list[dict] 样本不兼容。
    features = Features({
        "id": Value("string"),
        "image": HFImage(),
        "conversations": List({
            "role": Value("string"),
            "content": Value("string"),
        }),
        "source": Value("string"),
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

    kwargs: Dict[str, Any] = {
        "private": private,
        # HF Datasets：push_to_hub 即上传 Parquet；以下为 Parquet 写出/嵌入行为
        "embed_external_files": embed_external_files,
    }
    if token:
        kwargs["token"] = token
    # 不传时使用库默认（通常为 500MB）；传入则控制单个 Parquet 分片体积
    if max_shard_size:
        kwargs["max_shard_size"] = max_shard_size
    if num_proc is not None and num_proc > 0:
        kwargs["num_proc"] = num_proc
    if commit_message:
        kwargs["commit_message"] = commit_message

    print(
        f"[upload] push_to_hub (Parquet shards) repo_id={repo_id!r} "
        f"private={private} embed_external_files={embed_external_files} ..."
    )
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
        help="单个 Parquet 分片最大体积（如 ``500MB``）；不传则由 datasets 默认（通常为 500MB）",
    )
    parser.add_argument(
        "--no-embed-external-files",
        action="store_true",
        help="不把图像字节写入 Parquet（仅保留路径类信息；Hub 离线克隆后可能无法直接看图）",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="写入/嵌入 Parquet 时的并行进程数（大数据集可选）",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Hub 提交说明（可选）",
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
        embed_external_files=not args.no_embed_external_files,
        num_proc=args.num_proc,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
