"""
Caption / 生成式文本指标计算。

依赖：
    - pycocoevalcap （BLEU / ROUGE-L / CIDEr）
    - nltk         （METEOR；首次运行会自动下载 wordnet / omw-1.4）

接口：
    evaluate_caption(predictions: list[str], references: list[str | list[str]]) -> dict
"""

import json
from typing import Any, Dict, List, Union

import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


# 确保 METEOR 所需词表可用；已下载过则 no-op
for _pkg in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


# -----------------------------------------------------------------------------
# 输入校验 / 格式转换
# -----------------------------------------------------------------------------
def _validate_inputs(
    predictions: List[str],
    references: List[Union[str, List[str]]],
) -> None:
    if not isinstance(predictions, list) or not isinstance(references, list):
        raise TypeError("predictions and references must both be lists.")

    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: len(predictions)={len(predictions)} "
            f"!= len(references)={len(references)}"
        )

    for i, pred in enumerate(predictions):
        if not isinstance(pred, str):
            raise TypeError(f"predictions[{i}] must be str, got {type(pred)}")

    for i, ref in enumerate(references):
        if isinstance(ref, str):
            continue
        if isinstance(ref, list):
            if len(ref) == 0:
                raise ValueError(f"references[{i}] is an empty list.")
            for j, r in enumerate(ref):
                if not isinstance(r, str):
                    raise TypeError(
                        f"references[{i}][{j}] must be str, got {type(r)}"
                    )
        else:
            raise TypeError(
                f"references[{i}] must be str or list[str], got {type(ref)}"
            )


def _to_coco_format(
    predictions: List[str],
    references: List[Union[str, List[str]]],
):
    """
    转 pycocoevalcap 格式：
        refs:  {id: [ref1, ref2, ...]}
        cands: {id: [pred]}
    pycocoevalcap 要求候选 / 参考都非空，这里兜底把空串替换成单空格。
    """
    refs: Dict[int, List[str]] = {}
    cands: Dict[int, List[str]] = {}

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if isinstance(ref, str):
            ref_list = [ref]
        else:
            ref_list = list(ref)

        ref_list = [(r if r.strip() else " ") for r in ref_list]
        pred_safe = pred if pred.strip() else " "

        refs[i] = ref_list
        cands[i] = [pred_safe]

    return refs, cands


def _simple_tokenize(text: str) -> List[str]:
    return text.strip().lower().split()


def _compute_meteor(
    predictions: List[str],
    references: List[Union[str, List[str]]],
):
    meteor_scores: List[float] = []

    for pred, ref in zip(predictions, references):
        pred_tokens = _simple_tokenize(pred)

        if isinstance(ref, str):
            ref_tokens = [_simple_tokenize(ref)]
        else:
            ref_tokens = [_simple_tokenize(r) for r in ref]

        if not pred_tokens or not any(ref_tokens):
            meteor_scores.append(0.0)
            continue

        try:
            score = meteor_score(ref_tokens, pred_tokens)
        except Exception:
            score = 0.0
        meteor_scores.append(float(score))

    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    return meteor_avg, meteor_scores


# -----------------------------------------------------------------------------
# main entry
# -----------------------------------------------------------------------------
def evaluate_caption(
    predictions: List[str],
    references: List[Union[str, List[str]]],
) -> Dict[str, Any]:
    """
    计算 caption 指标：BLEU-1~4 / ROUGE-L / CIDEr / METEOR。

    Args:
        predictions: list[str]，长度 N
        references:  list[str] 或 list[list[str]]，长度 N

    Returns:
        {
            "BLEU-1": ..., "BLEU-2": ..., "BLEU-3": ..., "BLEU-4": ...,
            "METEOR": ..., "ROUGE-L": ..., "CIDEr": ...,
            "per_sample": {metric: list[float]}
        }
    """
    _validate_inputs(predictions, references)
    refs, cands = _to_coco_format(predictions, references)

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    results: Dict[str, Any] = {}
    per_sample: Dict[str, Any] = {}

    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(refs, cands)
        except Exception as e:
            print(f"[metrics] {method} failed: {e}")
            if isinstance(method, list):
                for m in method:
                    results[m] = 0.0
                    per_sample[m] = [0.0] * len(predictions)
            else:
                results[method] = 0.0
                per_sample[method] = [0.0] * len(predictions)
            continue

        if isinstance(method, list):
            for m, s, ss in zip(method, score, scores):
                results[m] = float(s)
                per_sample[m] = [float(x) for x in ss]
        else:
            results[method] = float(score)
            per_sample[method] = [float(x) for x in scores]

    meteor_avg, meteor_scores = _compute_meteor(predictions, references)
    results["METEOR"] = meteor_avg
    per_sample["METEOR"] = meteor_scores

    results["per_sample"] = per_sample
    return results


if __name__ == "__main__":
    predictions = [
        "a cat sitting on a sofa",
        "a dog running in the grass",
    ]
    references = [
        "a cat is sitting on the couch",
        "a dog runs through a grassy field",
    ]
    results = evaluate_caption(predictions, references)

    print(json.dumps(
        {k: v for k, v in results.items() if k != "per_sample"},
        ensure_ascii=False, indent=2,
    ))
