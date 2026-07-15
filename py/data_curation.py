"""Curate merged CRMLS ledgers into lightweight Tableau-ready datasets.

This script intentionally does not modify the raw merged ledgers produced by
initial_merge.py or the incremental pipelines. It reads:

    ../csv/listings.csv
    ../csv/sold.csv

and writes new production-facing curated files:

    ../csv/listings_curated.csv
    ../csv/sold_curated.csv

It then materializes Tableau-ready terminal datasets:

    ../csv/crmls_unified_wide_table.csv
    ../csv/crmls_monthly_market_metrics.csv
"""

from __future__ import annotations

import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"

INPUT_OUTPUT_PAIRS = (
    (CSV_DIR / "listings.csv", CSV_DIR / "listings_curated.csv"),
    (CSV_DIR / "sold.csv", CSV_DIR / "sold_curated.csv"),
)
LISTINGS_CURATED_PATH = CSV_DIR / "listings_curated.csv"
SOLD_CURATED_PATH = CSV_DIR / "sold_curated.csv"
UNIFIED_WIDE_TABLE_PATH = CSV_DIR / "crmls_unified_wide_table.csv"
MONTHLY_MARKET_METRICS_PATH = CSV_DIR / "crmls_monthly_market_metrics.csv"
FRED_MORTGAGE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
YEAR_MONTH_COLUMN = "year_month"
MORTGAGE_RATE_COLUMN = "rate_30yr_fixed"
FRED_DATE_COLUMN = "observation_date"
FRED_RATE_COLUMN = "MORTGAGE30US"
FRED_TIMEOUT_SECONDS = 30

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
    YEAR_MONTH_COLUMN,
    MORTGAGE_RATE_COLUMN,
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


def fetch_monthly_mortgage_rates(
    source: str | Path = FRED_MORTGAGE_URL,
    *,
    timeout: int = FRED_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch FRED weekly mortgage rates and resample them to monthly means."""
    source_text = str(source)
    if source_text.lower().startswith(("http://", "https://")):
        client = session or requests.Session()
        response = client.get(source_text, timeout=timeout)
        response.raise_for_status()
        weekly = pd.read_csv(StringIO(response.text))
    else:
        weekly = pd.read_csv(source)

    required_columns = {FRED_DATE_COLUMN, FRED_RATE_COLUMN}
    missing_columns = required_columns.difference(weekly.columns)
    if missing_columns:
        raise ValueError(
            "FRED mortgage CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    weekly[FRED_DATE_COLUMN] = pd.to_datetime(weekly[FRED_DATE_COLUMN], errors="coerce")
    weekly[FRED_RATE_COLUMN] = pd.to_numeric(weekly[FRED_RATE_COLUMN], errors="coerce")
    weekly = weekly.dropna(subset=[FRED_DATE_COLUMN]).sort_values(FRED_DATE_COLUMN)
    if weekly.empty:
        raise ValueError("FRED mortgage CSV contains no valid observation dates.")

    monthly = (
        weekly.set_index(FRED_DATE_COLUMN)[FRED_RATE_COLUMN]
        .resample("MS")
        .mean()
        .dropna()
        .rename(MORTGAGE_RATE_COLUMN)
        .reset_index()
    )
    monthly[YEAR_MONTH_COLUMN] = (
        monthly[FRED_DATE_COLUMN].dt.to_period("M").astype("string")
    )
    return monthly[[YEAR_MONTH_COLUMN, MORTGAGE_RATE_COLUMN]]


def enrich_with_mortgage_rates(
    frame: pd.DataFrame,
    *,
    date_column: str,
    monthly_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Left join monthly mortgage rates using a YYYY-MM key derived from a date column."""
    if date_column not in frame.columns:
        raise ValueError(f"Dataset is missing date column: {date_column}")
    required_rate_columns = {YEAR_MONTH_COLUMN, MORTGAGE_RATE_COLUMN}
    missing_rate_columns = required_rate_columns.difference(monthly_rates.columns)
    if missing_rate_columns:
        raise ValueError(
            "Monthly mortgage rates are missing columns: "
            + ", ".join(sorted(missing_rate_columns))
        )

    working = frame.drop(
        columns=[YEAR_MONTH_COLUMN, MORTGAGE_RATE_COLUMN], errors="ignore"
    ).copy()
    parsed_dates = pd.to_datetime(working[date_column], errors="coerce")
    working[YEAR_MONTH_COLUMN] = parsed_dates.dt.to_period("M").astype("string")

    rates = monthly_rates[[YEAR_MONTH_COLUMN, MORTGAGE_RATE_COLUMN]].copy()
    rates[YEAR_MONTH_COLUMN] = rates[YEAR_MONTH_COLUMN].astype("string")
    rates[MORTGAGE_RATE_COLUMN] = pd.to_numeric(
        rates[MORTGAGE_RATE_COLUMN], errors="coerce"
    )
    rates = rates.drop_duplicates(subset=[YEAR_MONTH_COLUMN], keep="last")
    return working.merge(rates, how="left", on=YEAR_MONTH_COLUMN, validate="many_to_one")


def print_rate_null_check(frame: pd.DataFrame, label: str) -> int:
    """Print and return the number of unmatched mortgage-rate values."""
    null_count = int(frame[MORTGAGE_RATE_COLUMN].isna().sum())
    print(
        f"{label}: null {MORTGAGE_RATE_COLUMN} values = {null_count} "
        f"of {len(frame)} rows.",
        flush=True,
    )
    return null_count


def process_dataset(
    input_path: Path,
    output_path: Path,
    *,
    date_column: str | None = None,
    monthly_rates: pd.DataFrame | None = None,
) -> dict[str, int]:
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
    if monthly_rates is not None:
        if date_column is None:
            raise ValueError("date_column is required when monthly_rates is provided.")
        curated = enrich_with_mortgage_rates(
            curated,
            date_column=date_column,
            monthly_rates=monthly_rates,
        )
        print_rate_null_check(curated, output_path.name)

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


def build_unified_wide_table(listings: pd.DataFrame, sold: pd.DataFrame) -> pd.DataFrame:
    """Left join curated listings to curated sold records by ListingKey."""
    if "ListingKey" not in listings.columns:
        raise ValueError("listings_curated.csv is missing ListingKey.")
    if "ListingKey" not in sold.columns:
        raise ValueError("sold_curated.csv is missing ListingKey.")

    listings_working = listings.copy()
    sold_working = sold.copy()
    listings_working["ListingKey"] = listings_working["ListingKey"].astype(str).str.strip()
    sold_working["ListingKey"] = sold_working["ListingKey"].astype(str).str.strip()
    sold_working = sold_working.drop_duplicates(subset=["ListingKey"], keep="last")

    wide_table = listings_working.merge(
        sold_working,
        how="left",
        on="ListingKey",
        suffixes=("", "_sold"),
        validate="one_to_one",
    )
    if MORTGAGE_RATE_COLUMN not in wide_table.columns:
        sold_rate_column = f"{MORTGAGE_RATE_COLUMN}_sold"
        if sold_rate_column in wide_table.columns:
            wide_table[MORTGAGE_RATE_COLUMN] = wide_table[sold_rate_column]
        else:
            wide_table[MORTGAGE_RATE_COLUMN] = pd.NA
    return wide_table


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _numeric_series(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    column = _first_existing_column(frame, candidates)
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _datetime_series(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    column = _first_existing_column(frame, candidates)
    if column is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce")


def build_monthly_market_metrics(wide_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a Tableau-friendly monthly market snapshot."""
    if "ListingKey" not in wide_table.columns:
        raise ValueError("wide table is missing ListingKey.")

    working = pd.DataFrame(index=wide_table.index)
    listing_dates = _datetime_series(wide_table, ("ListingContractDate", "ListingContractDate_sold"))
    close_dates = _datetime_series(wide_table, ("CloseDate_sold", "CloseDate"))

    working["Month"] = listing_dates.dt.to_period("M").astype("string")
    working["City"] = wide_table.get("City", pd.Series(pd.NA, index=wide_table.index))
    working["PostalCode"] = wide_table.get("PostalCode", pd.Series(pd.NA, index=wide_table.index))
    working["City"] = working["City"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    working["PostalCode"] = (
        working["PostalCode"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )
    working["ListingKey"] = wide_table["ListingKey"]
    working["OriginalListPrice"] = _numeric_series(
        wide_table, ("OriginalListPrice", "OriginalListPrice_sold")
    )
    working["ClosePrice"] = _numeric_series(wide_table, ("ClosePrice_sold", "ClosePrice"))
    working[MORTGAGE_RATE_COLUMN] = _numeric_series(
        wide_table, (MORTGAGE_RATE_COLUMN, f"{MORTGAGE_RATE_COLUMN}_sold")
    )
    working["SoldFlag"] = close_dates.notna().astype(int)
    working["AbsorptionDays"] = (close_dates - listing_dates).dt.days
    working.loc[close_dates.isna() | listing_dates.isna(), "AbsorptionDays"] = pd.NA

    metrics = (
        working.groupby(["Month", "City", "PostalCode"], dropna=False)
        .agg(
            rate_30yr_fixed=(MORTGAGE_RATE_COLUMN, "mean"),
            CountOfListings=("ListingKey", "count"),
            CountOfSold=("SoldFlag", "sum"),
            MeanOriginalListPrice=("OriginalListPrice", "mean"),
            MeanClosePrice=("ClosePrice", "mean"),
            MeanAbsorptionDays=("AbsorptionDays", "mean"),
        )
        .reset_index()
        .sort_values(["Month", "City", "PostalCode"], na_position="last")
        .reset_index(drop=True)
    )

    return metrics[
        [
            "Month",
            "City",
            "PostalCode",
            MORTGAGE_RATE_COLUMN,
            "CountOfListings",
            "CountOfSold",
            "MeanOriginalListPrice",
            "MeanClosePrice",
            "MeanAbsorptionDays",
        ]
    ]


def process_tableau_outputs(
    *,
    listings_path: Path = LISTINGS_CURATED_PATH,
    sold_path: Path = SOLD_CURATED_PATH,
    wide_output_path: Path = UNIFIED_WIDE_TABLE_PATH,
    metrics_output_path: Path = MONTHLY_MARKET_METRICS_PATH,
) -> dict[str, int]:
    """Materialize Tableau terminal datasets from curated listings and sold CSVs."""
    listings_path = Path(listings_path)
    sold_path = Path(sold_path)
    if not listings_path.exists():
        raise FileNotFoundError(f"Curated listings CSV not found: {listings_path}")
    if not sold_path.exists():
        raise FileNotFoundError(f"Curated sold CSV not found: {sold_path}")

    print(f"Reading curated listings: {listings_path}", flush=True)
    listings = pd.read_csv(listings_path, low_memory=False)
    print(f"Reading curated sold: {sold_path}", flush=True)
    sold = pd.read_csv(sold_path, low_memory=False)

    wide_table = build_unified_wide_table(listings, sold)
    atomic_write_csv(wide_table, wide_output_path)
    print(
        f"Wrote {wide_output_path}: {len(wide_table)} rows, {len(wide_table.columns)} columns.",
        flush=True,
    )

    metrics = build_monthly_market_metrics(wide_table)
    atomic_write_csv(metrics, metrics_output_path)
    print(
        f"Wrote {metrics_output_path}: {len(metrics)} rows, {len(metrics.columns)} columns.",
        flush=True,
    )

    return {
        "wide_rows": len(wide_table),
        "wide_columns": len(wide_table.columns),
        "metrics_rows": len(metrics),
        "metrics_columns": len(metrics.columns),
    }


def main() -> int:
    print("Starting CRMLS data curation.", flush=True)
    try:
        print(f"Fetching FRED mortgage rates: {FRED_MORTGAGE_URL}", flush=True)
        monthly_rates = fetch_monthly_mortgage_rates()
        print(f"Loaded {len(monthly_rates)} monthly mortgage-rate observations.", flush=True)
        date_columns = {
            LISTINGS_CURATED_PATH: "ListingContractDate",
            SOLD_CURATED_PATH: "CloseDate",
        }
        for input_path, output_path in INPUT_OUTPUT_PAIRS:
            process_dataset(
                input_path,
                output_path,
                date_column=date_columns[output_path],
                monthly_rates=monthly_rates,
            )
        process_tableau_outputs()
    except Exception as exc:
        print(f"Data curation failed: {exc}", flush=True)
        return 1

    print("CRMLS data curation complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
