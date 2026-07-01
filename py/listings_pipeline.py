"""Incrementally synchronize CRMLS listings into ../csv/listings.csv."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import requests


# Existing CRMLS endpoints retained from crmls_listed.py.
url = "https://api-trestle.corelogic.com/trestle/odata/Property"
auth_endpoint = "https://idxexchange.com/internal-api/trestle_token.php?key=IDXEXCHANGE2026_CHANGE_THIS"

SELECT_FIELDS = (
    "OriginalListPrice, ListingKey,CloseDate,ClosePrice,ListAgentFirstName,"
    "ListAgentLastName,Latitude,Longitude,UnparsedAddress,PropertyType,LivingArea,"
    "ListPrice,DaysOnMarket,ListOfficeName,BuyerOfficeName,CoListOfficeName,"
    "ListAgentFullName,CoListAgentFirstName,CoListAgentLastName,BuyerAgentMlsId,"
    "BuyerAgentFirstName,BuyerAgentLastName,FireplacesTotal,AssociationFeeFrequency,"
    "AboveGradeFinishedArea,ListingKeyNumeric,MLSAreaMajor,TaxAnnualAmount,"
    "CountyOrParish,PropertyType,MlsStatus,ElementarySchool,ListAgentFirstName,"
    "AttachedGarageYN,ParkingTotal,BuilderName,PropertySubType,LotSizeAcres,"
    "SubdivisionName,BuyerOfficeAOR,YearBuilt,DaysOnMarket,StreetNumberNumeric,"
    "LivingArea,ListingId,BathroomsTotalInteger,City,TaxYear,BuildingAreaTotal,"
    "BedroomsTotal,ContractStatusChangeDate,Longitude,ElementarySchoolDistrict,"
    "CoBuyerAgentFirstName,PurchaseContractDate,ListingContractDate,"
    "BelowGradeFinishedArea,BusinessType,Latitude,ListPrice,StateOrProvince,"
    "CoveredSpaces,MiddleOrJuniorSchool,FireplaceYN,Stories,HighSchool,Levels,"
    "ListAgentLastName,CloseDate,LotSizeDimensions,LotSizeArea,MainLevelBedrooms,"
    "NewConstructionYN,GarageSpaces,HighSchoolDistrict,PostalCode,BuyerOfficeName,"
    "AssociationFee,LotSizeSquareFeet,MiddleOrJuniorSchoolDistrict,UnparsedAddress"
)

FIELDS = list(
    dict.fromkeys(field.strip() for field in SELECT_FIELDS.split(",") if field.strip())
)
DATE_COLUMN = "ListingContractDate"
CSV_PATH = (Path(__file__).resolve().parent / ".." / "csv" / "listings.csv").resolve()
FALLBACK_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
AUTH_TIMEOUT = (10, 30)
API_TIMEOUT = (10, 60)


def format_odata_timestamp(value: datetime) -> str:
    """Return a UTC OData timestamp with millisecond precision."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _max_date_from_frame(frame: pd.DataFrame) -> datetime:
    if DATE_COLUMN not in frame.columns or frame.empty:
        return FALLBACK_START
    parsed = pd.to_datetime(
        frame[DATE_COLUMN], format="mixed", utc=True, errors="coerce"
    )
    maximum = parsed.max()
    if pd.isna(maximum):
        return FALLBACK_START
    return maximum.to_pydatetime()


def get_start_time(csv_path: Path = CSV_PATH) -> datetime:
    """Find the inclusive incremental lower bound from a local CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return FALLBACK_START
    try:
        frame = pd.read_csv(csv_path, usecols=[DATE_COLUMN], low_memory=False)
    except (pd.errors.EmptyDataError, ValueError):
        return FALLBACK_START
    return _max_date_from_frame(frame)


def build_filter(start: datetime, end: datetime) -> str:
    return (
        f"{DATE_COLUMN} ge {format_odata_timestamp(start)} and "
        f"{DATE_COLUMN} lt {format_odata_timestamp(end)}"
    )


def get_access_token(session: requests.Session) -> str:
    response = session.get(auth_endpoint, timeout=AUTH_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("Token endpoint returned a non-object JSON response")
    token = payload.get("access_token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Token endpoint response does not contain a valid access_token")
    return token.strip()


def fetch_all_records(
    session: requests.Session,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Fetch every OData page, following @odata.nextLink to completion."""
    records: list[dict[str, Any]] = []
    current_url = url
    current_params: Mapping[str, Any] | None = params
    seen_urls: set[str] = set()
    page_number = 0

    while True:
        if current_url in seen_urls:
            raise ValueError(f"Repeated @odata.nextLink detected: {current_url}")
        seen_urls.add(current_url)
        page_number += 1

        response = session.get(
            current_url,
            params=current_params,
            headers=dict(headers),
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"API page {page_number} returned non-object JSON")

        observations = payload.get("value", [])
        if not isinstance(observations, list):
            raise ValueError(f"API page {page_number} has a non-list value field")
        if any(not isinstance(item, Mapping) for item in observations):
            raise ValueError(f"API page {page_number} contains a non-object record")

        records.extend(dict(item) for item in observations)
        print(
            f"Fetched page {page_number}: {len(observations)} records "
            f"({len(records)} total).",
            flush=True,
        )

        next_link = payload.get("@odata.nextLink")
        if not next_link:
            break
        if not isinstance(next_link, str):
            raise ValueError("@odata.nextLink must be a string")
        current_url = next_link
        current_params = None

    return records


def _validate_listing_keys(frame: pd.DataFrame, source: str) -> None:
    if frame.empty:
        return
    if "ListingKey" not in frame.columns:
        raise ValueError(f"{source} data is missing required ListingKey")
    invalid = frame["ListingKey"].isna() | frame["ListingKey"].astype(str).str.strip().eq("")
    if invalid.any():
        raise ValueError(f"{source} data contains missing or blank ListingKey values")


def merge_records(
    existing: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Append API rows and keep the newest representation of each ListingKey."""
    existing = existing.copy()
    incoming = pd.DataFrame.from_records(records)
    _validate_listing_keys(existing, "Existing CSV")
    _validate_listing_keys(incoming, "API")

    existing = existing.reindex(columns=FIELDS)
    incoming = incoming.reindex(columns=FIELDS)
    combined = pd.concat([existing, incoming], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=FIELDS)

    combined["ListingKey"] = combined["ListingKey"].astype("string").str.strip()
    return combined.drop_duplicates(subset=["ListingKey"], keep="last").reset_index(drop=True)


def atomic_write_csv(frame: pd.DataFrame, target: Path = CSV_PATH) -> None:
    """Write beside the destination, then atomically replace it."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.reindex(columns=FIELDS).to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
        )
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=FIELDS)
    try:
        return pd.read_csv(csv_path, dtype={"ListingKey": "string"}, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FIELDS)


def run_pipeline(
    session: requests.Session | None = None,
    now: datetime | None = None,
    csv_path: Path = CSV_PATH,
) -> int:
    """Run one complete listings synchronization and return the saved row count."""
    csv_path = Path(csv_path)
    existing_file = csv_path.exists()
    existing = _load_existing(csv_path)
    start = _max_date_from_frame(existing)
    end = now or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    end = end.astimezone(timezone.utc)
    if start > end:
        raise ValueError("Local maximum ListingContractDate is later than current UTC time")

    print(
        f"Listings window: {format_odata_timestamp(start)} <= {DATE_COLUMN} < "
        f"{format_odata_timestamp(end)}",
        flush=True,
    )

    http = session or requests.Session()
    token = get_access_token(http)
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "$select": SELECT_FIELDS,
        "$filter": build_filter(start, end),
        "$top": 1000,
    }
    records = fetch_all_records(http, headers, params)

    if not records and existing_file:
        print("No new listing records; existing CSV was left unchanged.", flush=True)
        return len(existing)

    merged = merge_records(existing, records)
    removed = len(existing) + len(records) - len(merged)
    atomic_write_csv(merged, csv_path)
    print(
        f"Listings saved to {csv_path}: {len(merged)} rows; "
        f"{removed} duplicate rows removed.",
        flush=True,
    )
    return len(merged)


def main() -> int:
    try:
        run_pipeline()
        return 0
    except Exception as exc:
        print(f"Listings pipeline failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
