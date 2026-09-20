import csv
from pathlib import Path

from eval_perceptual_metrics import limit_rows_per_group, read_rows, summarize_rows


def test_read_limit_and_summarize_perceptual_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics_per_image.csv"
    rows = []
    for group in ("low_haze", "low_rain"):
        first, second = group.split("_")
        for index in range(2):
            rows.append(
                {
                    "source_prompt": group,
                    "remove": first,
                    "preserve": second,
                    "image": f"sample_{index}.png",
                    "psnr": str(30 + index),
                    "ssim": str(0.9 + index * 0.01),
                    "lpips": str(0.1 - index * 0.01),
                    "target_path": str(tmp_path / f"target_{group}_{index}.png"),
                    "output_path": str(tmp_path / f"output_{group}_{index}.png"),
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = read_rows(csv_path)
    selected = limit_rows_per_group(loaded, 1)
    assert len(selected) == 2
    selected[0]["dists"] = 0.2
    selected[1]["dists"] = 0.4

    summary = summarize_rows(selected, ("psnr", "dists"))
    assert len(summary) == 2
    assert summary[0]["mean_psnr"] == 30
    assert summary[0]["mean_dists"] == 0.2
