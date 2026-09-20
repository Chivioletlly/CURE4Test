#!/usr/bin/env python3
"""Run and evaluate all ten two-factor selective-restoration directions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from skimage.metrics import structural_similarity

from cure.inference_utils import IMAGE_SUFFIXES, load_image, load_runtime, save_image
from cure.metrics import psnr


PROJECT_ROOT = Path(__file__).resolve().parent
TWO_FACTOR_PROMPTS = (
    "low_haze",
    "low_rain",
    "low_snow",
    "haze_rain",
    "haze_snow",
)


@dataclass(frozen=True)
class SelectiveJob:
    source_prompt: str
    remove: str
    preserve: str
    source: Path
    target: Path
    destination: Path


def resolve_dataset_roots(data_root: str | Path) -> tuple[Path, Path]:
    """Accept either the CCDD-11 root or its half_test directory."""

    root = Path(data_root).expanduser()
    candidates = (root / "half_test", root)
    for candidate in candidates:
        main_root = candidate / "main_data"
        sub_root = candidate / "sub_data"
        if main_root.is_dir() and sub_root.is_dir():
            return main_root, sub_root
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find main_data and sub_data below --data-root; searched: {searched}"
    )


def discover_jobs(
    data_root: str | Path,
    output_dir: str | Path,
    max_images: int | None = None,
    *,
    strict_pairs: bool = False,
) -> list[SelectiveJob]:
    if max_images is not None and max_images <= 0:
        raise ValueError("--max-images must be positive")

    main_root, sub_root = resolve_dataset_roots(data_root)
    image_root = Path(output_dir) / "images"
    jobs: list[SelectiveJob] = []
    unpaired_sources: list[tuple[Path, tuple[Path, ...]]] = []

    for source_prompt in TWO_FACTOR_PROMPTS:
        source_dir = main_root / source_prompt
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source degradation directory: {source_dir}")
        sources = sorted(
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not sources:
            raise ValueError(f"No input images found under {source_dir}")

        first, second = source_prompt.split("_")
        directions = ((first, second), (second, first))
        paired_sources: list[tuple[Path, dict[str, Path]]] = []
        for source in sources:
            stem = source.stem
            targets = {
                remove: sub_root / source_prompt / stem / f"{stem}_{preserve}_.png"
                for remove, preserve in directions
            }
            missing = tuple(path for path in targets.values() if not path.is_file())
            if missing:
                unpaired_sources.append((source, missing))
            else:
                paired_sources.append((source, targets))

        if max_images is not None:
            paired_sources = paired_sources[:max_images]
        if not paired_sources:
            raise ValueError(f"No fully paired source/target images found for {source_prompt}")

        for remove, preserve in directions:
            group_root = image_root / source_prompt / f"remove_{remove}"
            for source, targets in paired_sources:
                jobs.append(
                    SelectiveJob(
                        source_prompt=source_prompt,
                        remove=remove,
                        preserve=preserve,
                        source=source,
                        target=targets[remove],
                        destination=group_root / source.name,
                    )
                )

    if unpaired_sources:
        preview = "\n".join(
            f"  - {source} (missing: {', '.join(str(path) for path in missing)})"
            for source, missing in unpaired_sources[:10]
        )
        suffix = "\n  ..." if len(unpaired_sources) > 10 else ""
        message = (
            f"Found {len(unpaired_sources)} source image(s) without complete selective targets:\n"
            f"{preview}{suffix}"
        )
        if strict_pairs:
            raise FileNotFoundError(message)
        print(f"warning: {message}\nSkipping unpaired source image(s).", file=sys.stderr)
    return jobs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="CCDD-11 root containing half_test, or the half_test directory itself",
    )
    parser.add_argument(
        "--checkpoint",
        default=PROJECT_ROOT / "checkpoints" / "CURE_restorer.tar",
        help="CURE restorer checkpoint",
    )
    parser.add_argument(
        "--embedder-checkpoint",
        default=PROJECT_ROOT / "checkpoints" / "OneRestore_embedder.tar",
        help="OneRestore prompt-embedder checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "outputs" / "evaluation" / "selective_control",
        help="Root for restored images and metric reports",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu, cuda, or cuda:1",
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
        help="LPIPS backbone (default: alex)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional maximum number of inputs per source degradation (smoke testing)",
    )
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="Fail instead of skipping source images whose selective targets are missing",
    )
    return parser.parse_args(argv)


def _ssim(target: torch.Tensor, output: torch.Tensor) -> float:
    target_array = target.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_array = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return float(
        structural_similarity(
            target_array,
            output_array,
            data_range=1.0,
            channel_axis=-1,
        )
    )


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _load_lpips(backbone: str, device: torch.device):
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS is not installed. Run: pip install 'lpips>=0.1.4'"
        ) from error
    return lpips.LPIPS(net=backbone).to(device).eval()


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    jobs = discover_jobs(
        args.data_root,
        output_dir,
        args.max_images,
        strict_pairs=args.strict_pairs,
    )
    restorer, encoder, device = load_runtime(
        args.checkpoint,
        args.embedder_checkpoint,
        args.device,
    )
    lpips_metric = _load_lpips(args.lpips_net, device)
    embeddings = {
        factor: encoder([factor]) for factor in ("low", "haze", "rain", "snow")
    }

    rows: list[dict[str, object]] = []
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        source = load_image(job.source, device)
        target = load_image(job.target, device)
        if source.shape != target.shape:
            raise ValueError(
                f"Source/target shape mismatch: {job.source} {tuple(source.shape)} vs "
                f"{job.target} {tuple(target.shape)}"
            )

        output = restorer(source, embeddings[job.remove]).clamp(0, 1)
        save_image(output, job.destination)
        lpips_value = float(
            lpips_metric(output.mul(2).sub(1), target.mul(2).sub(1)).mean().item()
        )
        row: dict[str, object] = {
            "source_prompt": job.source_prompt,
            "remove": job.remove,
            "preserve": job.preserve,
            "image": job.source.name,
            "psnr": psnr(target, output),
            "ssim": _ssim(target, output),
            "lpips": lpips_value,
            "source_path": str(job.source),
            "target_path": str(job.target),
            "output_path": str(job.destination),
        }
        rows.append(row)
        print(
            f"[{index}/{total}] {job.source_prompt} remove={job.remove} "
            f"PSNR={row['psnr']:.4f} SSIM={row['ssim']:.6f} "
            f"LPIPS={row['lpips']:.6f}",
            flush=True,
        )

    summary: list[dict[str, object]] = []
    for source_prompt in TWO_FACTOR_PROMPTS:
        for remove in source_prompt.split("_"):
            group = [
                row
                for row in rows
                if row["source_prompt"] == source_prompt and row["remove"] == remove
            ]
            summary.append(
                {
                    "source_prompt": source_prompt,
                    "remove": remove,
                    "preserve": group[0]["preserve"],
                    "images": len(group),
                    "mean_psnr": _mean(group, "psnr"),
                    "mean_ssim": _mean(group, "ssim"),
                    "mean_lpips": _mean(group, "lpips"),
                }
            )

    overall = {
        "images": len(rows),
        "mean_psnr": _mean(rows, "psnr"),
        "mean_ssim": _mean(rows, "ssim"),
        "mean_lpips": _mean(rows, "lpips"),
    }
    result: dict[str, object] = {
        "data_root": str(args.data_root),
        "checkpoint": str(args.checkpoint),
        "embedder_checkpoint": str(args.embedder_checkpoint),
        "lpips_backbone": args.lpips_net,
        "summary": summary,
        "overall": overall,
        "per_image": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    summary_path = output_dir / "metrics_summary.csv"
    detail_path = output_dir / "metrics_per_image.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nsource_prompt  remove  preserve  images  mean_PSNR  mean_SSIM  mean_LPIPS")
    print("-------------  ------  --------  ------  ---------  ---------  ----------")
    for row in summary:
        print(
            f"{row['source_prompt']:<13}  {row['remove']:<6}  {row['preserve']:<8}  "
            f"{row['images']:>6}  {row['mean_psnr']:>9.4f}  "
            f"{row['mean_ssim']:>9.6f}  {row['mean_lpips']:>10.6f}"
        )
    print(
        f"overall: images={overall['images']} PSNR={overall['mean_psnr']:.4f} "
        f"SSIM={overall['mean_ssim']:.6f} LPIPS={overall['mean_lpips']:.6f}"
    )
    print(f"saved: {json_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {detail_path}")
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
