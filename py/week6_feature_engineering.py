"""Build Week 6 CRMLS market features and segmented Tableau-ready outputs.

Default inputs (first compatible file wins):

    ../csv/sold_curated.csv
    ../csv/sold.csv

Outputs:

    ../csv/week6_engineered_sold.csv
    ../csv/week6_sample_output.csv
    ../csv/week6_segment_summary.csv

California school-district boundaries are downloaded once and cached under
``../csv/reference/``. The source layer intentionally contains overlapping
elementary, high, and unified districts, so this script preserves those
district types in separate columns instead of duplicating property rows.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"

DEFAULT_INPUT_PATHS = (
    CSV_DIR / "sold_curated.csv",
    CSV_DIR / "sold.csv",
)
DEFAULT_OUTPUT_PATH = CSV_DIR / "week6_engineered_sold.csv"
DEFAULT_SAMPLE_OUTPUT_PATH = CSV_DIR / "week6_sample_output.csv"
DEFAULT_SEGMENT_OUTPUT_PATH = CSV_DIR / "week6_segment_summary.csv"
DEFAULT_DISTRICT_PATH = (
    CSV_DIR
    / "reference"
    / "california_school_district_areas_2024_25.geojson"
)

SCHOOL_DISTRICT_GEOJSON_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/"
    "b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
)
DOWNLOAD_TIMEOUT = (15, 180)
DEFAULT_SPATIAL_CHUNK_SIZE = 100_000

CORE_REQUIRED_COLUMNS = (
    "ListingKey",
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "DaysOnMarket",
    "CloseDate",
    "ListingContractDate",
    "PurchaseContractDate",
    "Latitude",
    "Longitude",
)

SEGMENT_DIMENSIONS = (
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "MLSAreaMajor",
    "ListOfficeName",
    "BuyerOfficeName",
    "SchoolDistrictName",
)

ENGINEERED_COLUMNS = (
    "PriceRatio",
    "CloseToOriginalListRatio",
    "PricePerSqFt",
    "DaysOnMarket",
    "Year",
    "Month",
    "YrMo",
    "ListingToContractDays",
    "ContractToCloseDays",
)

SCHOOL_DISTRICT_COLUMNS = (
    "SchoolDistrictName",
    "SchoolDistrictCode",
    "UnifiedSchoolDistrict",
    "ElementarySchoolDistrict",
    "HighSchoolDistrict",
    "SchoolDistrictMatchCount",
)

OUTPUT_BASE_COLUMNS = (
    "ListingKey",
    "ListingId",
    "ListingContractDate",
    "PurchaseContractDate",
    "CloseDate",
    "OriginalListPrice",
    "ListPrice",
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "MLSAreaMajor",
    "City",
    "PostalCode",
    "ListOfficeName",
    "BuyerOfficeName",
    "Latitude",
    "Longitude",
    "rate_30yr_fixed",
)


def atomic_write_csv(frame: pd.DataFrame, target_path: Path) -> None:
    """Atomically replace a CSV so interrupted writes cannot corrupt output."""
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


def download_reference_file(
    url: str,
    target_path: Path,
    *,
    session: requests.Session | None = None,
) -> Path:
    """Download a reference file atomically and return its local path."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    temp_path: Path | None = None

    try:
        with client.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        temp_file.write(block)
        os.replace(temp_path, target_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return target_path


def find_compatible_input(explicit_path: Path | None = None) -> Path:
    """Choose the first existing input that contains every core Week 6 field."""
    candidates = (Path(explicit_path),) if explicit_path is not None else DEFAULT_INPUT_PATHS
    incompatibilities: list[str] = []

    for path in candidates:
        if not path.exists():
            incompatibilities.append(f"{path}: file not found")
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(CORE_REQUIRED_COLUMNS).difference(columns))
        if not missing:
            return path
        incompatibilities.append(f"{path}: missing {', '.join(missing)}")

    details = "; ".join(incompatibilities)
    raise ValueError(f"No compatible Week 6 input CSV was found. {details}")


def read_week6_source(input_path: Path) -> pd.DataFrame:
    """Read only columns used by features, spatial enrichment, and segments."""
    available = list(pd.read_csv(input_path, nrows=0).columns)
    requested = {
        *CORE_REQUIRED_COLUMNS,
        *OUTPUT_BASE_COLUMNS,
        *SEGMENT_DIMENSIONS,
    }
    use_columns = [column for column in available if column in requested]
    return pd.read_csv(input_path, usecols=use_columns, low_memory=False)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide only by finite positive denominators."""
    result = numerator.div(denominator)
    invalid = denominator.isna() | denominator.le(0)
    return result.mask(invalid).replace([float("inf"), float("-inf")], pd.NA)


def engineer_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create every market metric required by the Week 6 handbook."""
    missing = sorted(set(CORE_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")

    working = frame.copy().reset_index(drop=True)
    close_price = _numeric(working, "ClosePrice")
    original_list_price = _numeric(working, "OriginalListPrice")
    living_area = _numeric(working, "LivingArea")
    close_date = pd.to_datetime(working["CloseDate"], errors="coerce", utc=True)
    listing_date = pd.to_datetime(
        working["ListingContractDate"], errors="coerce", utc=True
    )
    purchase_date = pd.to_datetime(
        working["PurchaseContractDate"], errors="coerce", utc=True
    )

    price_ratio = _safe_ratio(close_price, original_list_price)
    working["PriceRatio"] = price_ratio
    # The handbook lists both names with the same formula. Retain both so
    # Tableau authors can use the terminology expected by each dashboard.
    working["CloseToOriginalListRatio"] = price_ratio
    working["PricePerSqFt"] = _safe_ratio(close_price, living_area)
    working["DaysOnMarket"] = _numeric(working, "DaysOnMarket")
    working["Year"] = close_date.dt.year.astype("Int64")
    working["Month"] = close_date.dt.month.astype("Int64")
    working["YrMo"] = (
        close_date.dt.tz_localize(None).dt.to_period("M").astype("string")
    )
    listing_to_contract = (purchase_date - listing_date).dt.days
    contract_to_close = (close_date - purchase_date).dt.days
    working["ListingToContractDays"] = listing_to_contract.mask(
        listing_to_contract.lt(0)
    ).astype("Int64")
    working["ContractToCloseDays"] = contract_to_close.mask(
        contract_to_close.lt(0)
    ).astype("Int64")
    return working


def load_school_districts(district_path: Path) -> list[dict]:
    """Load the CDE GeoJSON into a compact polygon structure."""
    try:
        with Path(district_path).open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read school-district GeoJSON: {exc}") from exc

    if payload.get("type") != "FeatureCollection":
        raise ValueError("School-district file is not a GeoJSON FeatureCollection.")

    districts: list[dict] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            continue
        if not polygons:
            continue

        all_longitudes = [
            point[0]
            for polygon in polygons
            for ring in polygon
            for point in ring
        ]
        all_latitudes = [
            point[1]
            for polygon in polygons
            for ring in polygon
            for point in ring
        ]
        if not all_longitudes or not all_latitudes:
            continue
        districts.append(
            {
                "DistrictName": properties.get("DistrictName"),
                "DistrictType": properties.get("DistrictType"),
                "CDCode": properties.get("CDCode"),
                "polygons": polygons,
                "bounds": (
                    min(all_longitudes),
                    min(all_latitudes),
                    max(all_longitudes),
                    max(all_latitudes),
                ),
            }
        )

    if not districts:
        raise ValueError("School-district GeoJSON contains no usable polygons.")
    return districts


def _points_in_district(points, district: dict):
    """Vectorized point-in-(multi)polygon test, including interior holes."""
    import numpy as np
    from matplotlib.path import Path as PlotPath

    inside_district = np.zeros(len(points), dtype=bool)
    for polygon in district["polygons"]:
        if not polygon:
            continue
        inside_polygon = PlotPath(
            np.asarray(polygon[0], dtype="float64")
        ).contains_points(points, radius=1e-10)
        for hole in polygon[1:]:
            inside_polygon &= ~PlotPath(
                np.asarray(hole, dtype="float64")
            ).contains_points(points, radius=-1e-10)
        inside_district |= inside_polygon
    return inside_district


def _joined_values(values) -> str | pd.NA:
    cleaned = sorted(
        {str(value).strip() for value in values if pd.notna(value) and str(value).strip()}
    )
    return " | ".join(cleaned) if cleaned else pd.NA


def enrich_school_districts(
    frame: pd.DataFrame,
    districts: list[dict],
    *,
    chunk_size: int = DEFAULT_SPATIAL_CHUNK_SIZE,
) -> pd.DataFrame:
    """Point-in-polygon enrich properties without multiplying property rows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    import numpy as np

    working = frame.copy().reset_index(drop=True)
    for column in SCHOOL_DISTRICT_COLUMNS:
        working[column] = pd.NA

    latitude = _numeric(working, "Latitude")
    longitude = _numeric(working, "Longitude")
    valid = (
        latitude.between(32.0, 42.5)
        & longitude.between(-125.0, -113.0)
    )
    valid_rows = working.index[valid]
    if valid_rows.empty:
        working["SchoolDistrictMatchCount"] = working[
            "SchoolDistrictMatchCount"
        ].astype("Int64")
        return working

    row_matches: dict[int, list[tuple[object, object, object]]] = {}
    for start in range(0, len(valid_rows), chunk_size):
        row_ids = valid_rows[start : start + chunk_size].to_numpy()
        points = np.column_stack(
            (
                longitude.loc[row_ids].to_numpy(dtype="float64"),
                latitude.loc[row_ids].to_numpy(dtype="float64"),
            )
        )
        for district in districts:
            min_x, min_y, max_x, max_y = district["bounds"]
            candidates = (
                (points[:, 0] >= min_x)
                & (points[:, 0] <= max_x)
                & (points[:, 1] >= min_y)
                & (points[:, 1] <= max_y)
            )
            if not candidates.any():
                continue
            candidate_positions = np.flatnonzero(candidates)
            inside = _points_in_district(
                points[candidate_positions],
                district,
            )
            for position in candidate_positions[inside]:
                row_id = int(row_ids[position])
                row_matches.setdefault(row_id, []).append(
                    (
                        district["DistrictName"],
                        district["DistrictType"],
                        district["CDCode"],
                    )
                )
        print(
            f"School-district spatial match: "
            f"{min(start + chunk_size, len(valid_rows)):,}/{len(valid_rows):,} points.",
            flush=True,
        )

    for row_id, matches in row_matches.items():
        names = [match[0] for match in matches]
        types = [str(match[1]).lower() if pd.notna(match[1]) else "" for match in matches]
        working.at[row_id, "SchoolDistrictName"] = _joined_values(names)
        working.at[row_id, "SchoolDistrictCode"] = _joined_values(
            match[2] for match in matches
        )
        working.at[row_id, "SchoolDistrictMatchCount"] = len(
            {name for name in names if pd.notna(name)}
        )
        working.at[row_id, "UnifiedSchoolDistrict"] = _joined_values(
            match[0] for match, district_type in zip(matches, types)
            if "unified" in district_type
        )
        working.at[row_id, "ElementarySchoolDistrict"] = _joined_values(
            match[0] for match, district_type in zip(matches, types)
            if "elementary" in district_type
        )
        working.at[row_id, "HighSchoolDistrict"] = _joined_values(
            match[0] for match, district_type in zip(matches, types)
            if "high" in district_type or "secondary" in district_type
        )
    working["SchoolDistrictMatchCount"] = pd.to_numeric(
        working["SchoolDistrictMatchCount"], errors="coerce"
    ).astype("Int64")
    return working


def build_segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Stack summary statistics for each requested market segment dimension."""
    summaries: list[pd.DataFrame] = []
    aggregations = {
        "ListingsSold": ("ListingKey", "nunique"),
        "MeanClosePrice": ("ClosePrice", "mean"),
        "MedianClosePrice": ("ClosePrice", "median"),
        "MeanPriceRatio": ("PriceRatio", "mean"),
        "MeanPricePerSqFt": ("PricePerSqFt", "mean"),
        "MeanDaysOnMarket": ("DaysOnMarket", "mean"),
        "MeanListingToContractDays": ("ListingToContractDays", "mean"),
        "MeanContractToCloseDays": ("ContractToCloseDays", "mean"),
    }

    for dimension in SEGMENT_DIMENSIONS:
        if dimension not in frame.columns:
            continue
        metric_columns = sorted(
            {
                source_column
                for source_column, _ in aggregations.values()
            }
        )
        working = frame[[dimension, *metric_columns]].copy()
        working["_SegmentValue"] = (
            working[dimension]
            .astype("string")
            .fillna("Unknown")
            .str.strip()
            .replace("", "Unknown")
        )
        summary = (
            working.groupby("_SegmentValue", dropna=False)
            .agg(**aggregations)
            .reset_index()
            .rename(columns={"_SegmentValue": "SegmentValue"})
        )
        summary.insert(0, "SegmentDimension", dimension)
        summaries.append(summary)

    if not summaries:
        raise ValueError("No configured segment dimensions exist in the input data.")
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["SegmentDimension", "ListingsSold", "SegmentValue"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def select_engineered_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        *OUTPUT_BASE_COLUMNS,
        *ENGINEERED_COLUMNS,
        *SCHOOL_DISTRICT_COLUMNS,
    ]
    columns = list(dict.fromkeys(column for column in ordered if column in frame.columns))
    return frame.loc[:, columns]


def build_sample_output(frame: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive.")
    sample_columns = [
        "ListingKey",
        "ClosePrice",
        "OriginalListPrice",
        "LivingArea",
        *ENGINEERED_COLUMNS,
        *SCHOOL_DISTRICT_COLUMNS,
    ]
    sample_columns = list(
        dict.fromkeys(column for column in sample_columns if column in frame.columns)
    )
    populated_metrics = [
        column for column in ENGINEERED_COLUMNS if column in frame.columns
    ]
    candidates = frame.copy()
    candidates["_PopulatedMetricCount"] = candidates[populated_metrics].notna().sum(
        axis=1
    )
    candidates = candidates.sort_values(
        "_PopulatedMetricCount",
        ascending=False,
        kind="stable",
    )
    return candidates.loc[:, sample_columns].head(sample_rows).reset_index(drop=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Week 6 CRMLS engineered features and segment metrics."
    )
    parser.add_argument("--input", type=Path, help="Override the default sold input CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT_PATH
    )
    parser.add_argument(
        "--segment-output", type=Path, default=DEFAULT_SEGMENT_OUTPUT_PATH
    )
    parser.add_argument(
        "--school-district-file",
        type=Path,
        help="Use an existing local school-district vector file.",
    )
    parser.add_argument(
        "--refresh-school-districts",
        action="store_true",
        help="Redownload the cached CDE 2024-25 school-district GeoJSON.",
    )
    parser.add_argument(
        "--skip-school-districts",
        action="store_true",
        help="Create market features without the spatial enrichment.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_SPATIAL_CHUNK_SIZE,
        help="Number of property points per spatial-join chunk.",
    )
    parser.add_argument("--sample-rows", type=int, default=25)
    return parser


def run(args: argparse.Namespace) -> dict[str, int]:
    input_path = find_compatible_input(args.input)
    print(f"Reading Week 6 source: {input_path}", flush=True)
    source = read_week6_source(input_path)
    print(f"Loaded {len(source):,} rows and {len(source.columns)} columns.", flush=True)

    engineered = engineer_market_features(source)

    if args.skip_school_districts:
        print("School-district enrichment skipped by command-line option.", flush=True)
        for column in SCHOOL_DISTRICT_COLUMNS:
            engineered[column] = pd.NA
    else:
        district_path = args.school_district_file or DEFAULT_DISTRICT_PATH
        district_path = Path(district_path)
        if args.refresh_school_districts or not district_path.exists():
            print(
                f"Downloading CDE school-district boundaries to {district_path}",
                flush=True,
            )
            download_reference_file(SCHOOL_DISTRICT_GEOJSON_URL, district_path)
        print(f"Loading school-district polygons: {district_path}", flush=True)
        districts = load_school_districts(district_path)
        print(f"Loaded {len(districts):,} district polygon features.", flush=True)
        engineered = enrich_school_districts(
            engineered,
            districts,
            chunk_size=args.chunk_size,
        )
        matched = int(engineered["SchoolDistrictName"].notna().sum())
        print(
            f"School-district match coverage: {matched:,}/{len(engineered):,} "
            f"({matched / len(engineered):.2%})."
            if len(engineered)
            else "School-district match coverage: no property rows.",
            flush=True,
        )

    output = select_engineered_output_columns(engineered)
    sample = build_sample_output(output, args.sample_rows)
    summary = build_segment_summary(output)

    atomic_write_csv(output, args.output)
    atomic_write_csv(sample, args.sample_output)
    atomic_write_csv(summary, args.segment_output)

    print(
        f"Wrote engineered dataset: {args.output} "
        f"({len(output):,} rows, {len(output.columns)} columns).",
        flush=True,
    )
    print(f"Wrote sample output: {args.sample_output}", flush=True)
    print(sample.to_string(index=False), flush=True)
    print(f"Wrote segmented summary: {args.segment_output}", flush=True)
    print(summary.head(20).to_string(index=False), flush=True)

    return {
        "engineered_rows": len(output),
        "engineered_columns": len(output.columns),
        "sample_rows": len(sample),
        "segment_rows": len(summary),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    print("Starting Week 6 feature engineering.", flush=True)
    try:
        run(args)
    except Exception as exc:
        print(f"Week 6 feature engineering failed: {exc}", flush=True)
        return 1
    print("Week 6 feature engineering complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
