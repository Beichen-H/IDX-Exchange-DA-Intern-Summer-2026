"""Week 7 IQR outlier detection and data-quality pipeline.

The pipeline preserves every source record in a flagged audit dataset and
creates a separate filtered analysis dataset. IQR flags determine statistical
exclusions; 1st/99th percentile flags are informational review signals.

Default input priority:

    ../csv/week6_engineered_sold.csv
    ../csv/sold_curated.csv
    ../csv/sold.csv

Default outputs:

    ../csv/week7_flagged_sold.csv
    ../csv/week7_filtered_sold.csv
    ../csv/week7_outlier_comparison.csv
    ../reports/week7_outlier_comparison.md
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_INPUT_PATHS = (
    CSV_DIR / "week6_engineered_sold.csv",
    CSV_DIR / "sold_curated.csv",
    CSV_DIR / "sold.csv",
)
DEFAULT_FLAGGED_OUTPUT_PATH = CSV_DIR / "week7_flagged_sold.csv"
DEFAULT_FILTERED_OUTPUT_PATH = CSV_DIR / "week7_filtered_sold.csv"
DEFAULT_COMPARISON_OUTPUT_PATH = CSV_DIR / "week7_outlier_comparison.csv"
DEFAULT_REPORT_OUTPUT_PATH = REPORTS_DIR / "week7_outlier_comparison.md"

IQR_MULTIPLIER = 1.5
LOW_PERCENTILE = 0.01
HIGH_PERCENTILE = 0.99

REQUIRED_SOURCE_COLUMNS = (
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "DaysOnMarket",
)

# The first three fields satisfy the explicit Week 7 deliverable. The two
# derived metrics are included because the handbook also identifies them as
# capable of distorting market averages.
OUTLIER_FIELDS = (
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "PricePerSqFt",
    "CloseToOriginalListRatio",
)

FIELD_PREFIXES = {
    "ClosePrice": "close_price",
    "LivingArea": "living_area",
    "DaysOnMarket": "days_on_market",
    "PricePerSqFt": "price_per_sq_ft",
    "CloseToOriginalListRatio": "close_to_original_list_ratio",
}


def atomic_write_csv(frame: pd.DataFrame, target_path: Path) -> None:
    """Atomically replace a CSV without exposing a partially written file."""
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


def atomic_write_text(content: str, target_path: Path) -> None:
    """Atomically replace a UTF-8 text report."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        os.replace(temp_path, target_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def find_compatible_input(explicit_path: Path | None = None) -> Path:
    """Return the first existing CSV containing the required source fields."""
    candidates = (Path(explicit_path),) if explicit_path else DEFAULT_INPUT_PATHS
    incompatibilities: list[str] = []

    for path in candidates:
        if not path.exists():
            incompatibilities.append(f"{path}: file not found")
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(REQUIRED_SOURCE_COLUMNS).difference(columns))
        if not missing:
            return path
        incompatibilities.append(f"{path}: missing {', '.join(missing)}")

    raise ValueError(
        "No compatible Week 7 input CSV was found. " + "; ".join(incompatibilities)
    )


def _safe_positive_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Return a ratio only when both values are numeric and denominator > 0."""
    ratio = numerator.div(denominator)
    invalid = numerator.isna() | denominator.isna() | denominator.le(0)
    return ratio.mask(invalid).replace([float("inf"), float("-inf")], pd.NA)


def prepare_numeric_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw numerics and derive Week 6 ratios when they are absent."""
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing columns: {', '.join(missing)}")

    working = frame.copy().reset_index(drop=True)
    for column in REQUIRED_SOURCE_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    derived_ppsf = _safe_positive_ratio(
        working["ClosePrice"], working["LivingArea"]
    )
    if "PricePerSqFt" in working.columns:
        supplied_ppsf = pd.to_numeric(working["PricePerSqFt"], errors="coerce")
        working["PricePerSqFt"] = supplied_ppsf.fillna(derived_ppsf)
    else:
        working["PricePerSqFt"] = derived_ppsf

    derived_close_ratio = _safe_positive_ratio(
        working["ClosePrice"], working["OriginalListPrice"]
    )
    if "CloseToOriginalListRatio" in working.columns:
        supplied_ratio = pd.to_numeric(
            working["CloseToOriginalListRatio"], errors="coerce"
        )
        working["CloseToOriginalListRatio"] = supplied_ratio.fillna(
            derived_close_ratio
        )
    elif "PriceRatio" in working.columns:
        supplied_ratio = pd.to_numeric(working["PriceRatio"], errors="coerce")
        working["CloseToOriginalListRatio"] = supplied_ratio.fillna(
            derived_close_ratio
        )
    else:
        working["CloseToOriginalListRatio"] = derived_close_ratio

    return working


def calculate_iqr_statistics(
    series: pd.Series,
    *,
    multiplier: float = IQR_MULTIPLIER,
    low_percentile: float = LOW_PERCENTILE,
    high_percentile: float = HIGH_PERCENTILE,
) -> dict[str, float]:
    """Calculate percentile and IQR thresholds from a non-empty numeric series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Cannot calculate outlier thresholds from an empty series.")

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    return {
        "P01": float(clean.quantile(low_percentile)),
        "Q1": q1,
        "MedianBefore": float(clean.median()),
        "Q3": q3,
        "P99": float(clean.quantile(high_percentile)),
        "IQR": iqr,
        "LowerBound": q1 - multiplier * iqr,
        "UpperBound": q3 + multiplier * iqr,
    }


def _append_reason(
    reasons: pd.Series,
    mask: pd.Series,
    label: str,
) -> None:
    """Append an audit reason to selected rows without Python row iteration."""
    selected = mask.fillna(False)
    if not selected.any():
        return
    separators = reasons.loc[selected].map(lambda value: "; " if value else "")
    reasons.loc[selected] = reasons.loc[selected] + separators + label


def apply_outlier_detection(
    frame: pd.DataFrame,
    *,
    multiplier: float = IQR_MULTIPLIER,
    low_percentile: float = LOW_PERCENTILE,
    high_percentile: float = HIGH_PERCENTILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag business-rule and statistical outliers, then create a clean copy."""
    if multiplier <= 0:
        raise ValueError("IQR multiplier must be positive.")
    if not 0 <= low_percentile < high_percentile <= 1:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 1.")

    flagged = prepare_numeric_metrics(frame)

    # Business-rule flags are deliberately separate from statistical flags.
    business_masks = {
        "invalid_close_price_flag": flagged["ClosePrice"].isna()
        | flagged["ClosePrice"].le(0),
        "invalid_original_list_price_flag": flagged["OriginalListPrice"].isna()
        | flagged["OriginalListPrice"].le(0),
        "invalid_living_area_flag": flagged["LivingArea"].isna()
        | flagged["LivingArea"].le(0),
        "invalid_days_on_market_flag": flagged["DaysOnMarket"].isna()
        | flagged["DaysOnMarket"].lt(0),
        "invalid_price_per_sq_ft_flag": flagged["PricePerSqFt"].isna()
        | flagged["PricePerSqFt"].le(0),
        "invalid_close_to_original_list_ratio_flag": flagged[
            "CloseToOriginalListRatio"
        ].isna()
        | flagged["CloseToOriginalListRatio"].le(0),
    }
    for column, mask in business_masks.items():
        flagged[column] = mask.astype(bool)

    business_columns = list(business_masks)
    flagged["business_rule_invalid_flag"] = flagged[business_columns].any(axis=1)
    reasons = pd.Series("", index=flagged.index, dtype="string")
    for column, mask in business_masks.items():
        _append_reason(reasons, mask, column.removesuffix("_flag"))

    summary_rows: list[dict[str, object]] = []
    iqr_flag_columns: list[str] = []
    percentile_flag_columns: list[str] = []

    for field in OUTLIER_FIELDS:
        prefix = FIELD_PREFIXES[field]
        iqr_flag = f"{prefix}_iqr_outlier_flag"
        percentile_flag = f"{prefix}_percentile_extreme_flag"
        iqr_flag_columns.append(iqr_flag)
        percentile_flag_columns.append(percentile_flag)

        # Thresholds are learned only from rows that pass deterministic
        # business rules, preventing impossible values from shifting quartiles.
        population = flagged.loc[
            ~flagged["business_rule_invalid_flag"], field
        ].dropna()
        if population.empty:
            flagged[iqr_flag] = False
            flagged[percentile_flag] = False
            summary_rows.append(
                {
                    "Metric": field,
                    "NonNullBusinessValidCount": 0,
                    "P01": pd.NA,
                    "Q1": pd.NA,
                    "MedianBefore": pd.NA,
                    "Q3": pd.NA,
                    "P99": pd.NA,
                    "IQR": pd.NA,
                    "LowerBound": pd.NA,
                    "UpperBound": pd.NA,
                    "IqrOutlierCount": 0,
                    "PercentileExtremeCount": 0,
                }
            )
            continue

        statistics = calculate_iqr_statistics(
            population,
            multiplier=multiplier,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )
        values = flagged[field]
        valid_for_statistics = ~flagged["business_rule_invalid_flag"] & values.notna()
        iqr_mask = valid_for_statistics & (
            values.lt(statistics["LowerBound"])
            | values.gt(statistics["UpperBound"])
        )
        percentile_mask = valid_for_statistics & (
            values.lt(statistics["P01"]) | values.gt(statistics["P99"])
        )
        flagged[iqr_flag] = iqr_mask.astype(bool)
        flagged[percentile_flag] = percentile_mask.astype(bool)
        _append_reason(reasons, iqr_mask, f"{prefix}_iqr_outlier")

        summary_rows.append(
            {
                "Metric": field,
                "NonNullBusinessValidCount": int(population.size),
                **statistics,
                "IqrOutlierCount": int(iqr_mask.sum()),
                "PercentileExtremeCount": int(percentile_mask.sum()),
            }
        )

    flagged["any_iqr_outlier_flag"] = flagged[iqr_flag_columns].any(axis=1)
    flagged["any_percentile_extreme_flag"] = flagged[
        percentile_flag_columns
    ].any(axis=1)
    flagged["analysis_exclusion_flag"] = (
        flagged["business_rule_invalid_flag"] | flagged["any_iqr_outlier_flag"]
    )
    flagged["analysis_exclusion_reason"] = reasons.replace("", pd.NA)

    # The source is never overwritten or reduced. Filtering creates a distinct
    # analysis-ready copy as required by the handbook.
    filtered = flagged.loc[~flagged["analysis_exclusion_flag"]].copy()
    comparison = pd.DataFrame(summary_rows)
    comparison["MedianAfter"] = comparison["Metric"].map(
        {field: filtered[field].median() for field in OUTLIER_FIELDS}
    )
    comparison["MedianChange"] = (
        comparison["MedianAfter"] - comparison["MedianBefore"]
    )
    return flagged, filtered, comparison


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.4f}"


def build_comparison_report(
    *,
    input_path: Path,
    flagged: pd.DataFrame,
    filtered: pd.DataFrame,
    comparison: pd.DataFrame,
    multiplier: float,
    low_percentile: float,
    high_percentile: float,
) -> str:
    """Create the written before/after comparison required by Week 7."""
    before_rows = len(flagged)
    after_rows = len(filtered)
    removed_rows = before_rows - after_rows
    removed_rate = removed_rows / before_rows if before_rows else 0.0
    business_invalid = int(flagged["business_rule_invalid_flag"].sum())
    iqr_outliers = int(flagged["any_iqr_outlier_flag"].sum())
    percentile_extremes = int(flagged["any_percentile_extreme_flag"].sum())

    lines = [
        "# Week 7 Outlier Detection — Before/After Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Source: `{input_path}`",
        "",
        "## Method",
        "",
        f"IQR bounds use `Q1 - {multiplier:g} × IQR` and "
        f"`Q3 + {multiplier:g} × IQR`. "
        f"Percentile review flags use the {low_percentile:.0%} and "
        f"{high_percentile:.0%} cutoffs but do not independently remove rows.",
        "",
        "## Dataset Size",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Full flagged rows | {before_rows:,} |",
        f"| Clean filtered rows | {after_rows:,} |",
        f"| Excluded rows | {removed_rows:,} |",
        f"| Excluded percentage | {removed_rate:.2%} |",
        f"| Business-rule invalid rows | {business_invalid:,} |",
        f"| Rows with at least one IQR outlier | {iqr_outliers:,} |",
        f"| Rows with percentile review flags | {percentile_extremes:,} |",
        "",
        "## Median Comparison",
        "",
        "| Metric | Median Before | Median After | Change | IQR Outliers |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.Metric} | {_format_number(row.MedianBefore)} | "
            f"{_format_number(row.MedianAfter)} | "
            f"{_format_number(row.MedianChange)} | {row.IqrOutlierCount:,} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The flagged dataset retains every source record and adds audit columns.",
            "- The filtered dataset excludes deterministic business-rule failures and "
            "IQR outliers; it does not overwrite the source or flagged dataset.",
            "- Percentile flags identify records for analyst review and are not, by "
            "themselves, deletion criteria.",
            "- Median changes should be reviewed alongside excluded-row counts before "
            "using the filtered dataset in Tableau.",
            "",
        ]
    )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flag Week 7 IQR outliers and create a filtered analysis copy."
    )
    parser.add_argument("--input", type=Path, help="Override the default input CSV.")
    parser.add_argument(
        "--flagged-output", type=Path, default=DEFAULT_FLAGGED_OUTPUT_PATH
    )
    parser.add_argument(
        "--filtered-output", type=Path, default=DEFAULT_FILTERED_OUTPUT_PATH
    )
    parser.add_argument(
        "--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT_PATH
    )
    parser.add_argument(
        "--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT_PATH
    )
    parser.add_argument("--iqr-multiplier", type=float, default=IQR_MULTIPLIER)
    parser.add_argument("--low-percentile", type=float, default=LOW_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=HIGH_PERCENTILE)
    return parser


def run(args: argparse.Namespace) -> dict[str, int]:
    input_path = find_compatible_input(args.input)
    print(f"Reading Week 7 source: {input_path}", flush=True)
    source = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(source):,} rows and {len(source.columns)} columns.", flush=True)

    flagged, filtered, comparison = apply_outlier_detection(
        source,
        multiplier=args.iqr_multiplier,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    report = build_comparison_report(
        input_path=input_path,
        flagged=flagged,
        filtered=filtered,
        comparison=comparison,
        multiplier=args.iqr_multiplier,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )

    atomic_write_csv(flagged, args.flagged_output)
    atomic_write_csv(filtered, args.filtered_output)
    atomic_write_csv(comparison, args.comparison_output)
    atomic_write_text(report, args.report_output)

    removed = len(flagged) - len(filtered)
    print(
        f"Wrote full flagged dataset: {args.flagged_output} ({len(flagged):,} rows).",
        flush=True,
    )
    print(
        f"Wrote clean filtered dataset: {args.filtered_output} "
        f"({len(filtered):,} rows; {removed:,} excluded).",
        flush=True,
    )
    print(f"Wrote comparison table: {args.comparison_output}", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"Wrote written comparison: {args.report_output}", flush=True)

    return {
        "flagged_rows": len(flagged),
        "filtered_rows": len(filtered),
        "excluded_rows": removed,
        "comparison_rows": len(comparison),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    print("Starting Week 7 outlier detection.", flush=True)
    try:
        run(args)
    except Exception as exc:
        print(f"Week 7 outlier detection failed: {exc}", flush=True)
        return 1
    print("Week 7 outlier detection complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
