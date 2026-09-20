#!/usr/bin/env python3
"""Append learned FR and NR perceptual metrics to selective-evaluation results."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS = (
    "dists",
    "pieapp",
    "lpips-vgg",
    "clipiqa",
    "maniqa",
    "musiq",
    "liqe",
    "nima",
)
BASE_METRICS = ("psnr", "ssim", "lpips")
GROUP_FIELDS = ("source_prompt", "remove", "preserve")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default=PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / "selective_control"
        / "metrics_per_image.csv",
        help="Per-image CSV produced by eval_selective_control.py",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "outputs" / "evaluation" / "selective_perceptual",
        help="Destination for extended CSV and JSON reports",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        metavar="NAME",
        help="pyiqa metric names",
    )
    parser.add_argument(
        "--torch-home",
        help="Persistent Torch cache root; PyTorch appends hub/ below this directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu, cuda, or cuda:1",
    )
    parser.add_argument(
        "--max-images-per-group",
        type=int,
        help="Optional smoke-test limit applied independently to each selective direction",
    )
    return parser.parse_args(argv)


def read_rows(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Per-image metric CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {*GROUP_FIELDS, "image", "target_path", "output_path"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    return rows


def limit_rows_per_group(
    rows: Sequence[dict[str, object]], limit: int | None
) -> list[dict[str, object]]:
    if limit is None:
        return [dict(row) for row in rows]
    if limit <= 0:
        raise ValueError("--max-images-per-group must be positive")
    counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
    selected: list[dict[str, object]] = []
    for row in rows:
        key = tuple(str(row[field]) for field in GROUP_FIELDS)
        if counts[key] >= limit:
            continue
        selected.append(dict(row))
        counts[key] += 1
    return selected


def resolve_image_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def summarize_rows(
    rows: Sequence[dict[str, object]], metric_names: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in GROUP_FIELDS)
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for key, group in grouped.items():
        item: dict[str, object] = dict(zip(GROUP_FIELDS, key))
        item["images"] = len(group)
        for metric_name in metric_names:
            item[f"mean_{metric_name}"] = sum(
                float(row[metric_name]) for row in group
            ) / len(group)
        summary.append(item)
    return summary


def _scalar(value: object) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return _scalar(value[0])
    return float(value)


def _available_existing_metrics(rows: Sequence[dict[str, object]]) -> list[str]:
    return [name for name in BASE_METRICS if name in rows[0]]


def write_reports(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    completed_metrics: Sequence[str],
    metadata: Sequence[dict[str, object]],
    errors: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = _available_existing_metrics(rows) + list(completed_metrics)
    summary = summarize_rows(rows, metric_names)

    detail_path = output_dir / "metrics_per_image_extended.csv"
    summary_path = output_dir / "metrics_summary_extended.csv"
    json_path = output_dir / "metrics_extended.json"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    overall = {
        f"mean_{metric_name}": sum(float(row[metric_name]) for row in rows) / len(rows)
        for metric_name in metric_names
    }
    report = {
        "images": len(rows),
        "completed_metrics": list(completed_metrics),
        "metric_metadata": list(metadata),
        "errors": errors,
        "summary": summary,
        "overall": overall,
        "per_image": list(rows),
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _metric_metadata(name: str, mode: str, model: object) -> dict[str, object]:
    score_range = getattr(model, "score_range", None)
    return {
        "name": name,
        "mode": mode,
        "lower_better": bool(getattr(model, "lower_better", False)),
        "score_range": None if score_range is None else str(score_range),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.torch_home:
        torch_home = Path(args.torch_home).expanduser()
        os.environ["TORCH_HOME"] = str(torch_home)
        if not (torch_home / "hub").is_dir():
            raise FileNotFoundError(f"Torch hub cache not found: {torch_home / 'hub'}")

    try:
        import pyiqa
    except ImportError as error:
        raise RuntimeError("pyiqa is not installed; run: pip install pyiqa") from error

    rows = limit_rows_per_group(read_rows(args.input_csv), args.max_images_per_group)
    requested = list(dict.fromkeys(args.metrics))
    all_models = set(pyiqa.list_models())
    unknown = [name for name in requested if name not in all_models]
    if unknown:
        raise ValueError(f"Unknown pyiqa metric(s): {unknown}")
    fr_models = set(pyiqa.list_models(metric_mode="FR"))
    nr_models = set(pyiqa.list_models(metric_mode="NR"))
    unsupported = [name for name in requested if name not in fr_models | nr_models]
    if unsupported:
        raise ValueError(f"Only scalar FR/NR metrics are supported, got: {unsupported}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} was requested, but CUDA is unavailable")

    output_dir = Path(args.output_dir)
    completed: list[str] = []
    metadata: list[dict[str, object]] = []
    errors: dict[str, str] = {}
    total_metrics = len(requested)
    total_images = len(rows)
    for metric_index, metric_name in enumerate(requested, start=1):
        mode = "FR" if metric_name in fr_models else "NR"
        print(
            f"[{metric_index}/{total_metrics}] loading {metric_name} ({mode})",
            flush=True,
        )
        metric = None
        try:
            metric = pyiqa.create_metric(metric_name, device=device)
            values: list[float] = []
            with torch.inference_mode():
                for image_index, row in enumerate(rows, start=1):
                    output_path = resolve_image_path(row["output_path"])
                    if mode == "FR":
                        target_path = resolve_image_path(row["target_path"])
                        score = metric(str(output_path), str(target_path))
                    else:
                        score = metric(str(output_path))
                    value = _scalar(score)
                    values.append(value)
                    print(
                        f"  [{image_index}/{total_images}] {row['source_prompt']} "
                        f"remove={row['remove']} {metric_name}={value:.6f}",
                        flush=True,
                    )
            for row, value in zip(rows, values):
                row[metric_name] = value
            completed.append(metric_name)
            metadata.append(_metric_metadata(metric_name, mode, metric))
            write_reports(output_dir, rows, completed, metadata, errors)
            print(f"completed: {metric_name}", flush=True)
        except Exception as error:  # Keep completed metrics and continue with independent models.
            errors[metric_name] = f"{type(error).__name__}: {error}"
            print(f"failed: {metric_name}: {errors[metric_name]}", flush=True)
            if completed:
                write_reports(output_dir, rows, completed, metadata, errors)
        finally:
            del metric
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not completed:
        raise RuntimeError(f"All requested metrics failed: {errors}")
    write_reports(output_dir, rows, completed, metadata, errors)
    result = {
        "images": len(rows),
        "completed_metrics": completed,
        "errors": errors,
        "output_dir": str(output_dir),
    }
    print(f"completed metrics: {', '.join(completed)}", flush=True)
    if errors:
        print(f"failed metrics: {errors}", flush=True)
    print(f"saved: {output_dir / 'metrics_summary_extended.csv'}", flush=True)
    print(f"saved: {output_dir / 'metrics_per_image_extended.csv'}", flush=True)
    print(f"saved: {output_dir / 'metrics_extended.json'}", flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
