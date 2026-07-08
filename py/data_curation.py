"""Curate merged CRMLS ledgers into lightweight Tableau-ready datasets.

This script intentionally does not modify the raw merged ledgers produced by
initial_merge.py or the incremental pipelines. It reads:

    ../csv/listings.csv
    ../csv/sold.csv

and writes new production-facing curated files:

    ../csv/listings_curated.csv
    ../csv/sold_curated.csv
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"

INPUT_OUTPUT_PAIRS = (
    (CSV_DIR / "listings.csv", CSV_DIR / "listings_curated.csv"),
    (CSV_DIR / "sold.csv", CSV_DIR / "sold_curated.csv"),
)

MISSING_THRESHOLD = 0.90

REQUIRED_FIELDS = (
    "ListingKey",
    "ListingContractDate",
    "CloseDate",
    "OriginalListPrice",
    "ClosePrice",
    "City",
    "PostalCode",
)

# Dashboard-oriented whitelist. Keep this compact: market trend dimensions,
# competitive pricing metrics, status/type segmentation, location, and basic
# physical property attributes. Large free-text fields such as UnparsedAddress,
# directions, remarks, descriptions, school narratives, etc. are deliberately
# excluded to keep Tableau extracts small and fast.
DASHBOARD_FIELDS = (
    "ListingKey",
    "ListingContractDate",
    "CloseDate",
    "OriginalListPrice",
    "ClosePrice",
    "City",
    "PostalCode",
    "ListPrice",
    "MlsStatus",
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "StateOrProvince",
    "MLSAreaMajor",
    "DaysOnMarket",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LivingArea",
    "BuildingAreaTotal",
    "LotSizeSquareFeet",
    "LotSizeAcres",
    "YearBuilt",
    "Latitude",
    "Longitude",
    "GarageSpaces",
    "ParkingTotal",
    "AssociationFee",
    "ListOfficeName",
    "BuyerOfficeName",
)


def atomic_write_csv(frame: pd.DataFrame, target_path: Path) -> None:
    """Write a CSV atomically so failures never leave a partial output file."""
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


def calculate_missing_rates(frame: pd.DataFrame) -> pd.Series:
    """Return each column's missing-value rate."""
    if frame.empty:
        return pd.Series(0.0, index=frame.columns, dtype="float64")
    return frame.isna().mean().sort_values(ascending=False)


def find_sparse_columns(frame: pd.DataFrame) -> dict[str, float]:
    """Find non-required columns whose missing rate is strictly greater than 90%."""
    required = set(REQUIRED_FIELDS)
    missing_rates = calculate_missing_rates(frame)
    sparse = missing_rates[
        (missing_rates > MISSING_THRESHOLD) & (~missing_rates.index.isin(required))
    ]
    return {column: float(rate) for column, rate in sparse.items()}


def _ordered_curated_columns(existing_columns: Sequence[str]) -> list[str]:
    existing = set(existing_columns)
    columns = list(REQUIRED_FIELDS)
    columns.extend(
        column
        for column in DASHBOARD_FIELDS
        if column not in REQUIRED_FIELDS and column in existing
    )
    return columns


def curate_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Drop sparse columns, keep dashboard fields, and protect required fields."""
    working = frame.copy()
    for field in REQUIRED_FIELDS:
        if field not in working.columns:
            working[field] = pd.NA

    sparse_columns = find_sparse_columns(working)
    if sparse_columns:
        working = working.drop(columns=list(sparse_columns))

    curated_columns = [
        column
        for column in _ordered_curated_columns(working.columns)
        if column in working.columns
    ]
    curated = working.reindex(columns=curated_columns)
    return curated, sparse_columns


def print_sparse_columns(dropped_columns: Mapping[str, float], label: str) -> None:
    if not dropped_columns:
        print(f"{label}: no columns exceeded {MISSING_THRESHOLD:.0%} missingness.", flush=True)
        return

    print(f"{label}: dropped columns with missing rate > {MISSING_THRESHOLD:.0%}:", flush=True)
    for column, rate in sorted(dropped_columns.items(), key=lambda item: (-item[1], item[0])):
        print(f"  - {column}: {rate:.2%}", flush=True)


def process_dataset(input_path: Path, output_path: Path) -> dict[str, int]:
    """Curate one merged ledger into one lightweight output CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"Reading {input_path}", flush=True)
    frame = pd.read_csv(input_path, low_memory=False)
    input_rows = len(frame)
    input_columns = len(frame.columns)

    curated, dropped_columns = curate_dataframe(frame)
    print_sparse_columns(dropped_columns, input_path.name)

    atomic_write_csv(curated, output_path)
    print(
        f"Wrote {output_path}: {len(curated)} rows, "
        f"{len(curated.columns)} columns "
        f"(from {input_rows} rows, {input_columns} columns).",
        flush=True,
    )

    return {
        "input_rows": input_rows,
        "input_columns": input_columns,
        "output_rows": len(curated),
        "output_columns": len(curated.columns),
        "sparse_columns_dropped": len(dropped_columns),
    }


def main() -> int:
    print("Starting CRMLS data curation.", flush=True)
    try:
        for input_path, output_path in INPUT_OUTPUT_PAIRS:
            process_dataset(input_path, output_path)
    except Exception as exc:
        print(f"Data curation failed: {exc}", flush=True)
        return 1

    print("CRMLS data curation complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
