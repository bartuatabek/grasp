"""Convert one verified Figshare NBRS archive to compact benchmark files."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path

from cvep_decoder.datasets import NBRS_OCCIPITAL_8, convert_nbrs_subject

ARCHIVES = {
    "1-10": ("c97d9d43972e22dafef63b2d6ce3c58f", range(1, 11)),
    "11-20": ("ee6da4ca6b1c023c98bf557deea5037c", range(11, 21)),
    "21-30": ("1e68b4c2f14612ba39428190050244ee", range(21, 31)),
    "31-40": ("31b39d520ae3fdf77a846c5efd752bf1", range(31, 41)),
    "41-50": ("7591569828ea2e04f733ffca18ea1401", range(41, 51)),
    "51-60": ("61c931df9e17259852fd3104e28e6036", range(51, 61)),
    "61-70": ("843273f072dec27d0b908a4ec0375ee8", range(61, 71)),
    "71-80": ("5454184b00ba0d99a88aa67c49ab3e58", range(71, 81)),
    "81-90": ("97d70834e78945b726c5e0b3c2ae84de", range(81, 91)),
    "91-100": ("fc97b1a6438ba1087dc3d80f90c8dc34", range(91, 101)),
}


def file_md5(path: Path) -> str:
    """Hash a large archive without loading it into memory."""
    digest = hashlib.md5()  # noqa: S324 - verifies published provenance only
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--group", required=True, choices=ARCHIVES)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--montage",
        choices=("posterior8", "full59"),
        default="posterior8",
        help="Retain the predefined posterior eight or all 59 recorded EEG channels.",
    )
    args = parser.parse_args()

    expected_md5, subjects = ARCHIVES[args.group]
    observed_md5 = file_md5(args.archive)
    if observed_md5 != expected_md5:
        raise ValueError(f"archive MD5 mismatch: {observed_md5} != {expected_md5}")

    codebook = args.data_root / "nbrs_meta" / "NBRS-15.mat"
    if not codebook.exists():
        codebook = args.data_root / "nbrs15" / "subject-01.npz"
    if not codebook.exists():
        raise FileNotFoundError(
            "NBRS codebook not found; prepare posterior8 subject 1 or extract NBRS-15.mat"
        )
    channels = args.data_root / "nbrs_meta" / "channels.loc"
    selected_channels = None if args.montage == "full59" else NBRS_OCCIPITAL_8
    output_folder = "nbrs15_full59" if args.montage == "full59" else "nbrs15"
    with (
        zipfile.ZipFile(args.archive) as archive,
        tempfile.TemporaryDirectory(prefix="nbrs-convert-") as temporary,
    ):
        root = Path(temporary)
        for subject in subjects:
            member = f"S{subject}/NBRS-15/data.mat"
            archive.extract(member, root)
            output = args.data_root / output_folder / f"subject-{subject:02d}.npz"
            convert_nbrs_subject(
                root / member,
                codebook,
                channels,
                output,
                channels=selected_channels,
            )
            print(f"converted {member} -> {output}", flush=True)


if __name__ == "__main__":
    main()
