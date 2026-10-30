"""Convert Dehais2024 and Thielen2021 into the compact benchmark format."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvep_decoder import convert_dehais2024_subject, convert_thielen2021_subject


def convert_dehais(source: Path, output: Path) -> None:
    paths = sorted(
        source.glob("*.set"),
        key=lambda path: int(path.stem.split("_", 1)[0].lstrip("P")),
    )
    if len(paths) != 24:
        raise FileNotFoundError("expected 24 Ricker-condition EEGLAB .set files")
    for path in paths:
        subject = int(path.stem.split("_", 1)[0].lstrip("P"))
        destination = output / "dehais2024" / f"subject-{subject:02d}.npz"
        convert_dehais2024_subject(path, destination)
        print(f"Dehais subject {subject:02d}", flush=True)


def convert_thielen(output: Path, raw_cache: Path) -> None:
    for subject in range(1, 31):
        destination = output / "thielen2021" / f"subject-{subject:02d}.npz"
        convert_thielen2021_subject(subject, destination, data_root=raw_cache)
        print(f"Thielen subject {subject:02d}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--dehais-ricker-dir",
        type=Path,
        help="Extracted Zenodo Dry_Ricker directory",
    )
    parser.add_argument("--thielen", action="store_true", help="Download all 30 subjects")
    parser.add_argument("--raw-cache", type=Path, default=Path("data/moabb-download-cache"))
    args = parser.parse_args()
    if args.dehais_ricker_dir is None and not args.thielen:
        parser.error("select --dehais-ricker-dir and/or --thielen")
    if args.dehais_ricker_dir is not None:
        convert_dehais(args.dehais_ricker_dir, args.data_root)
    if args.thielen:
        convert_thielen(args.data_root, args.raw_cache)


if __name__ == "__main__":
    main()
