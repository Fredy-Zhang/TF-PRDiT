#!/usr/bin/env python3
"""Download the X2CT-GAN preprocessed LIDC-IDRI archive for TF-PRDiT.

This project expects the HDF5 dataset released with the X2CT-GAN GitHub
project. That release already applies the X2CT bed/table stripping, so the
TF-PRDiT dataloader can read the extracted files directly.
"""

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


def infer_filename_from_url(url: str, fallback: str = "LIDC-HDF5-256.zip") -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or fallback


def session_bin(retries: int = 5, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def download_file(url: str, output_path: Path, retries: int = 5) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    session = session_bin(retries=retries)

    with session.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        with open(tmp_path, "wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
            leave=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    pbar.update(len(chunk))

    os.replace(tmp_path, output_path)
    return output_path


def extract_archive(archive_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive_path.suffixes).lower()

    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(outdir)
    elif suffixes.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        with tarfile.open(archive_path) as archive:
            archive.extractall(outdir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return outdir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the X2CT-GAN preprocessed LIDC-IDRI HDF5 dataset."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Dataset archive URL from the official X2CT-GAN GitHub instructions.",
    )
    parser.add_argument(
        "--out",
        default="./data",
        help="Parent output directory for the extracted LIDC-HDF5-256 folder.",
    )
    parser.add_argument("--archive-name", default=None, help="Optional local archive filename.")
    parser.add_argument("--no-extract", action="store_true", help="Download archive only.")
    parser.add_argument(
        "--delete-archive",
        action="store_true",
        help="Delete the archive after successful extraction.",
    )
    parser.add_argument("--retries", type=int, default=5, help="Retry count per request.")
    args = parser.parse_args()

    outdir = Path(args.out)
    archive_name = args.archive_name or infer_filename_from_url(args.url)
    archive_path = outdir / archive_name

    print("[*] Downloading the X2CT-GAN preprocessed LIDC-IDRI dataset.")
    print("[*] The extracted folder should be used as data.target_path in the LIDC configs.")
    print(f"[*] URL: {args.url}")
    print(f"[*] Archive: {archive_path}")

    download_file(args.url, archive_path, retries=args.retries)

    if not args.no_extract:
        print(f"[*] Extracting to: {outdir}")
        extract_archive(archive_path, outdir)
        if args.delete_archive:
            archive_path.unlink(missing_ok=True)

    print("[*] Done.")
    print("[*] Expected config values:")
    print(f"    data.path: {outdir}")
    print('    data.target_path: "LIDC-HDF5-256"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
