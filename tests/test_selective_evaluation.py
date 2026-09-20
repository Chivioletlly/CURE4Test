from pathlib import Path

from eval_selective_control import TWO_FACTOR_PROMPTS, discover_jobs, resolve_dataset_roots


def make_dataset(root: Path, image_count: int = 2) -> None:
    split = root / "half_test"
    for prompt in TWO_FACTOR_PROMPTS:
        source_dir = split / "main_data" / prompt
        source_dir.mkdir(parents=True)
        first, second = prompt.split("_")
        for index in range(image_count):
            name = f"sample_{index}"
            (source_dir / f"{name}.png").touch()
            target_dir = split / "sub_data" / prompt / name
            target_dir.mkdir(parents=True)
            (target_dir / f"{name}_{first}_.png").touch()
            (target_dir / f"{name}_{second}_.png").touch()


def test_dataset_root_accepts_ccdd_root_or_half_test(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    expected = (
        tmp_path / "half_test" / "main_data",
        tmp_path / "half_test" / "sub_data",
    )
    assert resolve_dataset_roots(tmp_path) == expected
    assert resolve_dataset_roots(tmp_path / "half_test") == expected


def test_discover_jobs_builds_ten_selective_directions(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    jobs = discover_jobs(tmp_path, tmp_path / "outputs", max_images=1)

    assert len(jobs) == 10
    low_haze_remove_low = next(
        job for job in jobs if job.source_prompt == "low_haze" and job.remove == "low"
    )
    assert low_haze_remove_low.preserve == "haze"
    assert low_haze_remove_low.target.name == "sample_0_haze_.png"
    assert low_haze_remove_low.destination == (
        tmp_path / "outputs" / "images" / "low_haze" / "remove_low" / "sample_0.png"
    )
