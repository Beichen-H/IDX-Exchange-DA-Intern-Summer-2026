"""One-time merge utility for historical monthly CRMLS CSV exports.

This script folds monthly files such as crmls_listed_202602.csv and
crmls_sold_202602.csv into the production ledgers consumed by the incremental
pipelines:

    ../csv/listings.csv
    ../csv/sold.csv
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from listings_pipeline import FIELDS as LISTINGS_FIELDS  # noqa: E402
from sold_pipeline import FIELDS as SOLD_FIELDS  # noqa: E402


def atomic_write_csv(frame: pd.DataFrame, target_path: Path) -> None:
    """Write a CSV atomically so a failed write cannot corrupt the target."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            frame.to_csv(temp_file, index=False)
        os.replace(temp_path, target_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _read_monthly_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {path.name}", flush=True)
        return pd.DataFrame()


def _validate_listing_keys(frame: pd.DataFrame, source_label: str) -> None:
    if frame.empty:
        return
    if "ListingKey" not in frame.columns:
        raise ValueError(f"{source_label} is missing required ListingKey column.")
    blank_keys = frame["ListingKey"].isna() | (frame["ListingKey"].astype(str).str.strip() == "")
    if blank_keys.any():
        raise ValueError(f"{source_label} contains blank ListingKey values.")


def normalize_fields(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    """Align a frame to the production field order, adding missing fields."""
    return frame.reindex(columns=list(fields))


def merge_monthly_files(
    *,
    csv_dir: Path,
    prefix: str,
    target_path: Path,
    fields: Sequence[str],
) -> dict[str, int]:
    """Merge monthly CRMLS CSVs into one normalized, de-duplicated ledger."""
    csv_dir = Path(csv_dir)
    target_path = Path(target_path)
    monthly_files = sorted(csv_dir.glob(f"{prefix}*.csv"))

    print(
        f"Scanning {csv_dir} for {prefix}*.csv: found {len(monthly_files)} files.",
        flush=True,
    )

    frames: list[pd.DataFrame] = []
    total_input_rows = 0
    for path in monthly_files:
        frame = _read_monthly_csv(path)
        if frame.empty:
            print(f"{path.name}: 0 rows.", flush=True)
            continue
        _validate_listing_keys(frame, path.name)
        total_input_rows += len(frame)
        frames.append(frame)
        print(f"{path.name}: {len(frame)} rows.", flush=True)

    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged["ListingKey"] = merged["ListingKey"].astype(str).str.strip()
        before_dedup = len(merged)
        merged = merged.drop_duplicates(subset=["ListingKey"], keep="last")
        merged = normalize_fields(merged, fields)
    else:
        before_dedup = 0
        merged = pd.DataFrame(columns=list(fields))

    atomic_write_csv(merged, target_path)
    print(
        f"Wrote {target_path}: {len(merged)} rows "
        f"({before_dedup - len(merged)} duplicates removed).",
        flush=True,
    )

    return {
        "input_files": len(monthly_files),
        "input_rows": total_input_rows,
        "output_rows": len(merged),
        "duplicates_removed": before_dedup - len(merged),
    }


def main() -> int:
    print("Starting one-time CRMLS historical CSV merge.", flush=True)
    try:
        listings_stats = merge_monthly_files(
            csv_dir=CSV_DIR,
            prefix="CRMLSListing",
            target_path=CSV_DIR / "listings.csv",
            fields=LISTINGS_FIELDS,
        )
        sold_stats = merge_monthly_files(
            csv_dir=CSV_DIR,
            prefix="CRMLSSold",
            target_path=CSV_DIR / "sold.csv",
            fields=SOLD_FIELDS,
        )
    except Exception as exc:
        print(f"Initial merge failed: {exc}", flush=True)
        return 1

    print(
        "Initial merge complete: "
        f"listings={listings_stats['output_rows']} rows, "
        f"sold={sold_stats['output_rows']} rows.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
