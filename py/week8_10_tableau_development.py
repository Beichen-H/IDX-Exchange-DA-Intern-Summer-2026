"""Build the Weeks 8-10 Tableau dashboard deliverables.

This script prepares two Residential-only dashboard data sets from the monthly
CRMLS CSV exports and packages two Tableau workbooks:

* ``tableau/market_analysis.twbx``
* ``tableau/competitive_analysis.twbx``

The source CSV files are never modified.  All generated files are written
atomically so that a failed run cannot replace a previously valid deliverable.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "csv"
TABLEAU_DIR = PROJECT_ROOT / "tableau"

START_DATE = pd.Timestamp("2024-01-01")
RESIDENTIAL_PROPERTY_TYPE = "Residential"
CLOSED_STATUS = "Closed"

MARKET_DATA_FILE = TABLEAU_DIR / "market_dashboard_data.csv"
COMPETITIVE_DATA_FILE = TABLEAU_DIR / "competitive_dashboard_data.csv"
MARKET_HYPER_FILE = TABLEAU_DIR / "market_dashboard_data.hyper"
COMPETITIVE_HYPER_FILE = TABLEAU_DIR / "competitive_dashboard_data.hyper"


MARKET_COLUMNS = (
    "ListingKey",
    "ListingContractDate",
    "CloseDate",
    "OriginalListPrice",
    "ClosePrice",
    "City",
    "PostalCode",
    "MlsStatus",
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "DaysOnMarket",
    "LivingArea",
)

COMPETITIVE_COLUMNS = (
    *MARKET_COLUMNS,
    "ListAgentFirstName",
    "ListAgentLastName",
    "ListAgentFullName",
    "ListOfficeName",
    "BuyerOfficeName",
    "Latitude",
    "Longitude",
)

FILTER_FIELDS = ("City", "CountyOrParish", "PostalCode", "PropertySubType")

NUMERIC_FIELDS = (
    "ClosePrice",
    "OriginalListPrice",
    "DaysOnMarket",
    "CloseToOriginalListRatio",
    "PricePerSquareFoot",
    "Latitude",
    "Longitude",
    "UnitsSold",
)


@dataclass(frozen=True)
class SheetSpec:
    """Declarative description of one Tableau worksheet/dashboard."""

    name: str
    row_field: str
    row_aggregation: str
    column_field: str
    mark: str
    record_type: str | None = None
    top_n_field: str | None = None
    top_n: int | None = None
    color_field: str | None = None
    color_aggregation: str = "sum"
    geographic: bool = False


MARKET_SHEETS = (
    SheetSpec(
        "Monthly Median Close Price",
        "ClosePrice",
        "median",
        "Month",
        "Line",
        record_type="Closed Sale",
    ),
    SheetSpec(
        "Average Days on Market",
        "DaysOnMarket",
        "average",
        "Month",
        "Line",
        record_type="Closed Sale",
    ),
    SheetSpec(
        "Average Close-to-Original-List Ratio",
        "CloseToOriginalListRatio",
        "average",
        "Month",
        "Line",
        record_type="Closed Sale",
    ),
    SheetSpec(
        "New Listings",
        "ListingKey",
        "count-distinct",
        "Month",
        "Bar",
        record_type="New Listing",
    ),
    SheetSpec(
        "Closed Sales",
        "ListingKey",
        "count-distinct",
        "Month",
        "Bar",
        record_type="Closed Sale",
    ),
    SheetSpec(
        "Market Pulse - Price per Square Foot",
        "PricePerSquareFoot",
        "average",
        "Month",
        "Line",
        record_type="Closed Sale",
        color_field="PropertySubType",
        color_aggregation="none",
    ),
)

COMPETITIVE_SHEETS = (
    SheetSpec(
        "Top 100 Listing Agents",
        "ListAgentFullName",
        "none",
        "ClosePrice",
        "Bar",
        top_n_field="ListAgentFullName",
        top_n=100,
        color_field="UnitsSold",
        color_aggregation="sum",
    ),
    SheetSpec(
        "Top 100 Listing Offices",
        "ListOfficeName",
        "none",
        "ClosePrice",
        "Bar",
        top_n_field="ListOfficeName",
        top_n=100,
        color_field="UnitsSold",
        color_aggregation="sum",
    ),
    SheetSpec(
        "Zip Code Median Close Price Heat Map",
        "Latitude",
        "average",
        "Longitude",
        "Circle",
        color_field="ClosePrice",
        color_aggregation="median",
        geographic=True,
    ),
    SheetSpec(
        "Zip Code Homes Sold Heat Map",
        "Latitude",
        "average",
        "Longitude",
        "Circle",
        color_field="UnitsSold",
        color_aggregation="sum",
        geographic=True,
    ),
    SheetSpec(
        "Competitive Market Share - Office Volume vs Units",
        "UnitsSold",
        "sum",
        "ClosePrice",
        "Circle",
        color_field="ListOfficeName",
        color_aggregation="none",
    ),
)


def log(message: str) -> None:
    """Print an unbuffered progress message."""

    print(message, flush=True)


def _month_from_name(path: Path) -> str:
    match = re.search(r"(20\d{4})", path.stem)
    if not match:
        raise ValueError(f"Could not derive YYYYMM from {path.name}")
    return match.group(1)


def discover_listing_files(csv_dir: Path = CSV_DIR) -> list[Path]:
    """Return canonical CRMLS monthly listing exports from 2024 onward."""

    pattern = re.compile(r"^CRMLSListing(20\d{4})\.csv$", re.IGNORECASE)
    files = [path for path in csv_dir.glob("*.csv") if pattern.match(path.name)]
    return sorted(path for path in files if _month_from_name(path) >= "202401")


def discover_sold_files(csv_dir: Path = CSV_DIR) -> list[Path]:
    """Return one preferred CRMLS sold export per month from 2024 onward.

    Some historical months contain both a base file and a ``_filled`` file.
    The filled file is preferred because it is the corrected version.
    """

    pattern = re.compile(
        r"^CRMLSSold(?P<month>20\d{4})(?P<filled>_filled)?\.csv$",
        re.IGNORECASE,
    )
    preferred: dict[str, Path] = {}
    for path in csv_dir.glob("*.csv"):
        match = pattern.match(path.name)
        if not match or match.group("month") < "202401":
            continue
        month = match.group("month")
        if month not in preferred or match.group("filled"):
            preferred[month] = path
    return [preferred[month] for month in sorted(preferred)]


def _read_monthly_files(paths: Sequence[Path], wanted: Iterable[str]) -> pd.DataFrame:
    """Read only needed columns and concatenate all monthly files."""

    wanted_set = set(wanted)
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        log(f"Reading {path.name} ({index}/{len(paths)})...")
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in wanted_set,
            dtype={"ListingKey": "string", "PostalCode": "string"},
            low_memory=False,
        )
        # Older exports do not always contain every requested field.
        for column in wanted_set.difference(frame.columns):
            frame[column] = pd.NA
        frames.append(frame[list(wanted_set)])

    if not frames:
        raise FileNotFoundError("No matching monthly CRMLS files were found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _clean_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dashboard dimensions without changing source files."""

    result = frame.copy()
    for column in (
        "City",
        "CountyOrParish",
        "PostalCode",
        "PropertyType",
        "PropertySubType",
        "MlsStatus",
        "ListAgentFirstName",
        "ListAgentLastName",
        "ListAgentFullName",
        "ListOfficeName",
        "BuyerOfficeName",
    ):
        if column not in result:
            continue
        result[column] = result[column].astype("string").str.strip()

    for column in ("City", "CountyOrParish", "PostalCode", "PropertySubType"):
        if column in result:
            result[column] = result[column].fillna("Unknown").replace("", "Unknown")

    if "PostalCode" in result:
        # CRMLS occasionally emits ZIP+4 or a numeric-looking ZIP. Tableau's
        # geographic role works most reliably with a five-character string.
        result["PostalCode"] = (
            result["PostalCode"]
            .str.replace(r"\.0$", "", regex=True)
            .str.extract(r"(\d{5})", expand=False)
            .fillna("Unknown")
        )
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator.div(denominator.where(denominator > 0))


def prepare_sold_data(paths: Sequence[Path]) -> pd.DataFrame:
    """Create the Residential, closed-sale detail table."""

    sold = _read_monthly_files(paths, COMPETITIVE_COLUMNS)
    sold = _clean_dimensions(sold)
    sold["CloseDate"] = pd.to_datetime(sold["CloseDate"], errors="coerce")
    sold["ListingContractDate"] = pd.to_datetime(
        sold["ListingContractDate"], errors="coerce"
    )

    sold = sold.loc[
        sold["PropertyType"].eq(RESIDENTIAL_PROPERTY_TYPE)
        & sold["MlsStatus"].eq(CLOSED_STATUS)
        & sold["CloseDate"].ge(START_DATE)
    ].copy()
    sold = sold.dropna(subset=["ListingKey", "CloseDate"])
    sold = sold.drop_duplicates(subset="ListingKey", keep="last")

    for column in (
        "OriginalListPrice",
        "ClosePrice",
        "DaysOnMarket",
        "Latitude",
        "Longitude",
    ):
        sold[column] = pd.to_numeric(sold[column], errors="coerce")

    missing_agent = sold["ListAgentFullName"].isna() | sold["ListAgentFullName"].eq("")
    reconstructed = (
        sold["ListAgentFirstName"].fillna("")
        + " "
        + sold["ListAgentLastName"].fillna("")
    ).str.strip()
    sold.loc[missing_agent, "ListAgentFullName"] = reconstructed.loc[missing_agent]
    sold["ListAgentFullName"] = sold["ListAgentFullName"].fillna("Unknown").replace("", "Unknown")
    sold["ListOfficeName"] = sold["ListOfficeName"].fillna("Unknown").replace("", "Unknown")
    sold["BuyerOfficeName"] = sold["BuyerOfficeName"].fillna("Unknown").replace("", "Unknown")

    sold["Month"] = sold["CloseDate"].dt.to_period("M").dt.to_timestamp()
    sold["CloseToOriginalListRatio"] = _safe_ratio(
        sold["ClosePrice"], sold["OriginalListPrice"]
    )
    sold["UnitsSold"] = 1
    return sold


def prepare_listing_data(paths: Sequence[Path]) -> pd.DataFrame:
    """Create Residential new-listing events from the monthly listing files."""

    listings = _read_monthly_files(paths, MARKET_COLUMNS)
    listings = _clean_dimensions(listings)
    listings["ListingContractDate"] = pd.to_datetime(
        listings["ListingContractDate"], errors="coerce"
    )
    listings = listings.loc[
        listings["PropertyType"].eq(RESIDENTIAL_PROPERTY_TYPE)
        & listings["ListingContractDate"].ge(START_DATE)
    ].copy()
    listings = listings.dropna(subset=["ListingKey", "ListingContractDate"])
    listings = listings.drop_duplicates(subset="ListingKey", keep="last")
    listings["Month"] = listings["ListingContractDate"].dt.to_period("M").dt.to_timestamp()
    return listings


def build_market_data(listings: pd.DataFrame, sold: pd.DataFrame) -> pd.DataFrame:
    """Union listing and sale events at property-event grain."""

    columns = [
        "ListingKey",
        "EventDate",
        "Month",
        "RecordType",
        "City",
        "CountyOrParish",
        "PostalCode",
        "PropertySubType",
        "ClosePrice",
        "OriginalListPrice",
        "DaysOnMarket",
        "CloseToOriginalListRatio",
        "PricePerSquareFoot",
    ]

    listing_events = listings[
        [
            "ListingKey",
            "ListingContractDate",
            "Month",
            "City",
            "CountyOrParish",
            "PostalCode",
            "PropertySubType",
        ]
    ].copy()
    listing_events = listing_events.rename(columns={"ListingContractDate": "EventDate"})
    listing_events["RecordType"] = "New Listing"
    for column in (
        "ClosePrice",
        "OriginalListPrice",
        "DaysOnMarket",
        "CloseToOriginalListRatio",
        "PricePerSquareFoot",
    ):
        listing_events[column] = pd.NA

    sale_events = sold[
        [
            "ListingKey",
            "CloseDate",
            "Month",
            "City",
            "CountyOrParish",
            "PostalCode",
            "PropertySubType",
            "ClosePrice",
            "OriginalListPrice",
            "DaysOnMarket",
            "CloseToOriginalListRatio",
        ]
    ].copy()
    sale_events = sale_events.rename(columns={"CloseDate": "EventDate"})
    sale_events["RecordType"] = "Closed Sale"
    # LivingArea is loaded from the sold source only for PPSF; add it here if
    # the source contained it, otherwise PPSF remains null and Tableau safely
    # excludes those records from AVG.
    living_area = pd.to_numeric(sold.get("LivingArea"), errors="coerce")
    sale_events["PricePerSquareFoot"] = pd.to_numeric(
        sale_events["ClosePrice"], errors="coerce"
    ).div(living_area.where(living_area > 0))

    market = pd.concat(
        [listing_events[columns], sale_events[columns]],
        ignore_index=True,
        sort=False,
    )
    return market.sort_values(["Month", "RecordType", "ListingKey"], kind="stable")


def build_competitive_data(sold: pd.DataFrame) -> pd.DataFrame:
    """Select the row-level fields needed by competitive dashboards."""

    columns = [
        "ListingKey",
        "CloseDate",
        "Month",
        "City",
        "CountyOrParish",
        "PostalCode",
        "PropertySubType",
        "ClosePrice",
        "ListAgentFullName",
        "ListOfficeName",
        "BuyerOfficeName",
        "Latitude",
        "Longitude",
        "UnitsSold",
    ]
    competitive = sold[columns].copy()
    close_price = pd.to_numeric(competitive["ClosePrice"], errors="coerce")
    top_agents = (
        competitive.assign(_ClosePrice=close_price)
        .groupby("ListAgentFullName", dropna=True)["_ClosePrice"]
        .sum()
        .nlargest(100)
        .index
    )
    top_offices = (
        competitive.assign(_ClosePrice=close_price)
        .groupby("ListOfficeName", dropna=True)["_ClosePrice"]
        .sum()
        .nlargest(100)
        .index
    )
    competitive["Top100Agent"] = competitive["ListAgentFullName"].isin(
        top_agents
    ).map({True: "Yes", False: "No"})
    competitive["Top100Office"] = competitive["ListOfficeName"].isin(
        top_offices
    ).map({True: "Yes", False: "No"})
    return competitive.sort_values(["Month", "ListingKey"], kind="stable")


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Write a CSV and atomically replace the destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv.tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", date_format="%Y-%m-%d")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _tableau_type(series: pd.Series) -> tuple[str, str, str]:
    """Return Tableau datatype, role and type attributes."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date", "dimension", "ordinal"
    if pd.api.types.is_numeric_dtype(series):
        return "real", "measure", "quantitative"
    return "string", "dimension", "nominal"


def normalize_tableau_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce stable physical types before writing CSV, Hyper, and TWB XML."""

    normalized = frame.copy()
    for field in NUMERIC_FIELDS:
        if field in normalized.columns:
            normalized[field] = pd.to_numeric(normalized[field], errors="coerce")
    for field in ("EventDate", "CloseDate", "Month"):
        if field in normalized.columns:
            normalized[field] = pd.to_datetime(normalized[field], errors="coerce")
    return normalized


def _hyper_sql_type(series: pd.Series):
    """Map a pandas series to the matching Tableau Hyper SQL type."""

    from tableauhyperapi import SqlType

    if pd.api.types.is_datetime64_any_dtype(series):
        return SqlType.date()
    if pd.api.types.is_numeric_dtype(series):
        return SqlType.double()
    return SqlType.text()


def atomic_write_hyper(frame: pd.DataFrame, csv_path: Path, destination: Path) -> None:
    """Create a Tableau Hyper extract atomically from an already-written CSV."""

    try:
        from tableauhyperapi import (
            Connection,
            CreateMode,
            HyperProcess,
            TableDefinition,
            TableName,
            Telemetry,
            escape_string_literal,
        )
    except ImportError as exc:
        raise RuntimeError(
            "tableauhyperapi is required; install it with "
            "`uv pip install --python .venv/Scripts/python.exe tableauhyperapi`."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp.hyper")
    temporary.unlink(missing_ok=True)
    table_name = TableName("Extract", "Extract")
    columns = [
        TableDefinition.Column(name, _hyper_sql_type(frame[name]))
        for name in frame.columns
    ]
    try:
        with HyperProcess(
            Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
            parameters={"log_config": ""},
        ) as process:
            with Connection(
                process.endpoint,
                str(temporary),
                CreateMode.CREATE_AND_REPLACE,
            ) as connection:
                connection.catalog.create_schema("Extract")
                connection.catalog.create_table(TableDefinition(table_name, columns))
                connection.execute_command(
                    f"COPY {table_name} FROM {escape_string_literal(str(csv_path))} "
                    "WITH (FORMAT CSV, HEADER 1, NULL '')"
                )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _field_token(datasource: str, field: str, aggregation: str) -> str:
    """Create Tableau's internal field token for a shelf or encoding."""

    codes = {
        "none": ("none", "nk"),
        "sum": ("sum", "qk"),
        "average": ("avg", "qk"),
        "median": ("med", "qk"),
        "count-distinct": ("ctd", "qk"),
        "month": ("my", "ok"),
    }
    prefix, suffix = codes[aggregation]
    return f"[{datasource}].[{prefix}:{field}:{suffix}]"


def _add_datasource(
    root: ET.Element,
    frame: pd.DataFrame,
    datasource_name: str,
    caption: str,
    filename: str,
    hyper_filename: str,
) -> None:
    datasources = root.find("datasources")
    if datasources is None:
        datasources = ET.SubElement(root, "datasources")
    datasource = ET.SubElement(
        datasources,
        "datasource",
        {"caption": caption, "inline": "true", "name": datasource_name, "version": "18.1"},
    )
    # Tableau's datasource schema is order-sensitive: connection must precede
    # aliases, columns, layout and semantic-values.
    connection = ET.SubElement(datasource, "connection", {"class": "federated"})
    named_connections = ET.SubElement(connection, "named-connections")
    named = ET.SubElement(
        named_connections,
        "named-connection",
        {"caption": filename, "name": f"textscan.{datasource_name}"},
    )
    ET.SubElement(
        named,
        "connection",
        {
            "class": "textscan",
            "directory": "Data",
            "filename": filename,
            "password": "",
            "server": "",
        },
    )
    relation = ET.SubElement(
        connection,
        "relation",
        {
            "connection": f"textscan.{datasource_name}",
            "name": filename,
            "table": f"[{Path(filename).stem}#csv]",
            "type": "table",
        },
    )
    relation_columns = ET.SubElement(
        relation,
        "columns",
        {
            "character-set": "UTF-8",
            "header": "yes",
            "locale": "en_US",
            "separator": ",",
        },
    )
    for ordinal, name in enumerate(frame.columns):
        datatype, _, _ = _tableau_type(frame[name])
        ET.SubElement(
            relation_columns,
            "column",
            {"datatype": datatype, "name": name, "ordinal": str(ordinal)},
        )

    ET.SubElement(datasource, "aliases", {"enabled": "yes"})
    for name in frame.columns:
        datatype, role, field_type = _tableau_type(frame[name])
        attributes = {
            "datatype": datatype,
            "name": f"[{name}]",
            "role": role,
            "type": field_type,
        }
        if name == "PostalCode":
            attributes["semantic-role"] = "[ZipCode].[Name]"
        elif name == "City":
            attributes["semantic-role"] = "[City].[Name]"
        elif name == "CountyOrParish":
            attributes["semantic-role"] = "[County].[Name]"
        ET.SubElement(datasource, "column", attributes)

    extract = ET.SubElement(
        datasource,
        "extract",
        {
            "count": str(len(frame)),
            "enabled": "true",
            "object-id": "",
            "units": "records",
            "user-specific": "false",
        },
    )
    extract_connection = ET.SubElement(
        extract,
        "connection",
        {
            "access_mode": "readonly",
            "author-locale": "en_US",
            "class": "hyper",
            "dbname": f"Data/Extracts/{hyper_filename}",
            "default-settings": "hyper",
            "schema": "Extract",
            "sslmode": "",
            "tablename": "Extract",
            "username": "tableau_internal_user",
        },
    )
    ET.SubElement(
        extract_connection,
        "relation",
        {"name": "Extract", "table": "[Extract].[Extract]", "type": "table"},
    )
    metadata_records = ET.SubElement(extract_connection, "metadata-records")
    for ordinal, name in enumerate(frame.columns):
        datatype, _, _ = _tableau_type(frame[name])
        remote_type = {"date": "133", "real": "5", "string": "129"}[datatype]
        aggregation = "Year" if datatype == "date" else (
            "Sum" if datatype == "real" else "Count"
        )
        record = ET.SubElement(metadata_records, "metadata-record", {"class": "column"})
        ET.SubElement(record, "remote-name").text = name
        ET.SubElement(record, "remote-type").text = remote_type
        ET.SubElement(record, "local-name").text = f"[{name}]"
        ET.SubElement(record, "parent-name").text = "[Extract]"
        ET.SubElement(record, "remote-alias").text = name
        ET.SubElement(record, "ordinal").text = str(ordinal)
        ET.SubElement(record, "family").text = filename
        ET.SubElement(record, "local-type").text = datatype
        ET.SubElement(record, "aggregation").text = aggregation
        ET.SubElement(record, "contains-null").text = "true"
    ET.SubElement(
        datasource,
        "layout",
        {
            "dim-ordering": "alphabetic",
            "measure-ordering": "alphabetic",
            "show-structure": "true",
        },
    )
    semantic_values = ET.SubElement(datasource, "semantic-values")
    ET.SubElement(
        semantic_values,
        "semantic-value",
        {"key": "[Country].[Name]", "value": '"United States"'},
    )


def _add_filters(view: ET.Element, datasource_name: str, record_type: str | None) -> None:
    for field in FILTER_FIELDS:
        token = _field_token(datasource_name, field, "none")
        filter_element = ET.SubElement(view, "filter", {"class": "categorical", "column": token})
        ET.SubElement(
            filter_element,
            "groupfilter",
            {"function": "level-members", "level": f"[none:{field}:nk]"},
        )
    if record_type:
        token = _field_token(datasource_name, "RecordType", "none")
        filter_element = ET.SubElement(
            view, "filter", {"class": "categorical", "column": token}
        )
        ET.SubElement(
            filter_element,
            "groupfilter",
            {
                "function": "member",
                "level": "[none:RecordType:nk]",
                "member": f'"{record_type}"',
            },
        )


def _instance_derivation(aggregation: str) -> str:
    return {
        "none": "None",
        "sum": "Sum",
        "average": "Avg",
        "median": "Median",
        "count-distinct": "CountD",
        "month": "Month",
    }[aggregation]


def _add_dependency_field(
    dependencies: ET.Element,
    frame: pd.DataFrame,
    field: str,
    aggregation: str,
) -> None:
    """Add a Tableau column and its aggregation/date instance once."""

    datatype, role, field_type = _tableau_type(frame[field])
    column_name = f"[{field}]"
    if not any(child.get("name") == column_name for child in dependencies.findall("column")):
        attributes = {
            "datatype": datatype,
            "name": column_name,
            "role": role,
            "type": field_type,
        }
        if field == "PostalCode":
            attributes["semantic-role"] = "[ZipCode].[Name]"
        ET.SubElement(dependencies, "column", attributes)

    instance_name = _field_token("", field, aggregation).replace("[].", "")
    if any(child.get("name") == instance_name for child in dependencies.findall("column-instance")):
        return
    if aggregation == "none":
        instance_type = "nominal"
    elif aggregation == "month":
        instance_type = "ordinal"
    else:
        instance_type = "quantitative"
    ET.SubElement(
        dependencies,
        "column-instance",
        {
            "column": column_name,
            "derivation": _instance_derivation(aggregation),
            "name": instance_name,
            "pivot": "key",
            "type": instance_type,
        },
    )


def _add_worksheet(
    worksheets: ET.Element,
    spec: SheetSpec,
    worksheet_name: str,
    datasource_name: str,
    datasource_caption: str,
    frame: pd.DataFrame,
) -> None:
    worksheet = ET.SubElement(worksheets, "worksheet", {"name": worksheet_name})
    table = ET.SubElement(worksheet, "table")
    view = ET.SubElement(table, "view")
    datasources = ET.SubElement(view, "datasources")
    ET.SubElement(
        datasources,
        "datasource",
        {"caption": datasource_caption, "name": datasource_name},
    )
    dependencies = ET.SubElement(
        view, "datasource-dependencies", {"datasource": datasource_name}
    )
    for field in FILTER_FIELDS:
        _add_dependency_field(dependencies, frame, field, "none")
    if spec.record_type:
        _add_dependency_field(dependencies, frame, "RecordType", "none")
    top_flag = None
    if spec.top_n_field == "ListAgentFullName":
        top_flag = "Top100Agent"
    elif spec.top_n_field == "ListOfficeName":
        top_flag = "Top100Office"
    if top_flag:
        _add_dependency_field(dependencies, frame, top_flag, "none")

    row_aggregation = spec.row_aggregation
    if spec.column_field == "Month":
        column_aggregation = "month"
    elif spec.column_field in {"ListAgentFullName", "ListOfficeName"}:
        column_aggregation = "none"
    elif spec.column_field in {"Latitude", "Longitude"}:
        column_aggregation = "average"
    else:
        column_aggregation = "sum"
    _add_dependency_field(dependencies, frame, spec.row_field, row_aggregation)
    _add_dependency_field(dependencies, frame, spec.column_field, column_aggregation)
    if spec.color_field:
        _add_dependency_field(
            dependencies, frame, spec.color_field, spec.color_aggregation
        )
    if spec.geographic:
        _add_dependency_field(dependencies, frame, "PostalCode", "none")
        _add_dependency_field(dependencies, frame, "Month", "month")

    _add_filters(view, datasource_name, spec.record_type)
    if top_flag:
        token = _field_token(datasource_name, top_flag, "none")
        top_filter = ET.SubElement(
            view, "filter", {"class": "categorical", "column": token}
        )
        ET.SubElement(
            top_filter,
            "groupfilter",
            {
                "function": "member",
                "level": f"[none:{top_flag}:nk]",
                "member": '"Yes"',
            },
        )
    if spec.geographic:
        month_token = _field_token(datasource_name, "Month", "month")
        month_filter = ET.SubElement(
            view,
            "filter",
            {"class": "categorical", "column": month_token},
        )
        ET.SubElement(
            month_filter,
            "groupfilter",
            {"function": "level-members", "level": "[my:Month:ok]"},
        )

    ET.SubElement(view, "aggregation", {"value": "true"})
    ET.SubElement(table, "style")
    panes = ET.SubElement(table, "panes")
    pane = ET.SubElement(panes, "pane", {"selection-relaxation-option": "selection-relaxation-allow"})
    pane_view = ET.SubElement(pane, "view")
    ET.SubElement(pane_view, "breakdown", {"value": "auto"})
    ET.SubElement(pane, "mark", {"class": spec.mark})
    encodings = ET.SubElement(pane, "encodings")

    if spec.color_field:
        ET.SubElement(
            encodings,
            "color",
            {
                "column": _field_token(
                    datasource_name, spec.color_field, spec.color_aggregation
                )
            },
        )
    if spec.geographic:
        ET.SubElement(
            encodings,
            "lod",
            {"column": _field_token(datasource_name, "PostalCode", "none")},
        )

    if spec.column_field == "Month":
        column_token = _field_token(datasource_name, "Month", "month")
    else:
        column_token = _field_token(
            datasource_name,
            spec.column_field,
            column_aggregation,
        )
    row_token = _field_token(datasource_name, spec.row_field, spec.row_aggregation)
    ET.SubElement(table, "rows").text = row_token
    ET.SubElement(table, "cols").text = column_token
    ET.SubElement(
        worksheet,
        "simple-id",
        {"uuid": "{" + str(uuid.uuid4()).upper() + "}"},
    )


def _add_dashboard(
    dashboards: ET.Element,
    dashboard_name: str,
    worksheet_name: str,
    datasource_name: str,
    filter_fields: Sequence[str],
) -> None:
    dashboard = ET.SubElement(
        dashboards,
        "dashboard",
        {"name": dashboard_name},
    )
    ET.SubElement(dashboard, "style")
    ET.SubElement(
        dashboard,
        "size",
        {"maxheight": "900", "maxwidth": "1440", "minheight": "600", "minwidth": "900"},
    )
    zones = ET.SubElement(dashboard, "zones")
    root_zone = ET.SubElement(
        zones,
        "zone",
        {
            "h": "100000",
            "id": "4",
            "type-v2": "layout-basic",
            "w": "100000",
            "x": "0",
            "y": "0",
        },
    )
    flow = ET.SubElement(
        root_zone,
        "zone",
        {
            "h": "97154",
            "id": "7",
            "param": "horz",
            "type-v2": "layout-flow",
            "w": "97538",
            "x": "1231",
            "y": "1423",
        },
    )
    chart_container = ET.SubElement(
        flow,
        "zone",
        {
            "h": "97154",
            "id": "5",
            "type-v2": "layout-basic",
            "w": "76000",
            "x": "1231",
            "y": "1423",
        },
    )
    chart_zone = ET.SubElement(
        chart_container,
        "zone",
        {
            "h": "97154",
            "id": "3",
            "name": worksheet_name,
            "w": "76000",
            "x": "1231",
            "y": "1423",
        },
    )
    _add_zone_style(chart_zone, margin="4")
    filter_flow = ET.SubElement(
        flow,
        "zone",
        {
            "fixed-size": "260",
            "h": "97154",
            "id": "6",
            "is-fixed": "true",
            "param": "vert",
            "type-v2": "layout-flow",
            "w": "21538",
            "x": "77231",
            "y": "1423",
        },
    )
    zone_id = 2
    zone_height = max(10000, 90000 // max(1, len(filter_fields)))
    for index, field in enumerate(filter_fields):
        filter_aggregation = "month" if field == "Month" else "none"
        filter_zone = ET.SubElement(
            filter_flow,
            "zone",
            {
                "h": str(zone_height),
                "id": str(20 + zone_id),
                "mode": "checkdropdown",
                "name": worksheet_name,
                "param": _field_token(datasource_name, field, filter_aggregation),
                "type-v2": "filter",
                "values": "relevant",
                "w": "21538",
                "x": "77231",
                "y": str(1423 + index * zone_height),
            },
        )
        _add_zone_style(filter_zone, margin="4")
        zone_id += 1
    _add_zone_style(root_zone, margin="8")
    ET.SubElement(
        dashboard,
        "simple-id",
        {"uuid": "{" + str(uuid.uuid4()).upper() + "}"},
    )


def _add_zone_style(zone: ET.Element, margin: str) -> None:
    style = ET.SubElement(zone, "zone-style")
    ET.SubElement(style, "format", {"attr": "border-color", "value": "#000000"})
    ET.SubElement(style, "format", {"attr": "border-style", "value": "none"})
    ET.SubElement(style, "format", {"attr": "border-width", "value": "0"})
    ET.SubElement(style, "format", {"attr": "margin", "value": margin})


def build_workbook_xml(
    frame: pd.DataFrame,
    workbook_title: str,
    datasource_name: str,
    datasource_caption: str,
    data_filename: str,
    hyper_filename: str,
    sheets: Sequence[SheetSpec],
    competitive: bool = False,
) -> bytes:
    """Build a Tableau workbook XML document."""

    root = ET.Element(
        "workbook",
        {
            "original-version": "18.1",
            "source-build": "2026.2.0 (20262.26.0603.1643)",
            "source-platform": "win",
            "version": "18.1",
            "xmlns:user": "http://www.tableausoftware.com/xml/user",
        },
    )
    manifest = ET.SubElement(root, "document-format-change-manifest")
    for name in (
        "AnimationOnByDefault",
        "MarkAnimation",
        "ObjectModelEncapsulateLegacy",
        "ObjectModelExtractV2",
        "ObjectModelTableType",
        "SchemaViewerObjectModel",
        "SheetIdentifierTracking",
        "VConnDownstreamExtractsWithWarnings",
        "WindowsPersistSimpleIdentifiers",
    ):
        ET.SubElement(manifest, name)
    preferences = ET.SubElement(root, "preferences")
    ET.SubElement(preferences, "preference", {"name": "ui.encoding.shelf.height", "value": "24"})
    ET.SubElement(preferences, "preference", {"name": "ui.shelf.height", "value": "26"})
    ET.SubElement(root, "datasources")
    _add_datasource(
        root,
        frame,
        datasource_name,
        datasource_caption,
        data_filename,
        hyper_filename,
    )
    worksheets = ET.SubElement(root, "worksheets")
    for spec in sheets:
        _add_worksheet(
            worksheets,
            spec,
            f"WS - {spec.name}",
            datasource_name,
            datasource_caption,
            frame,
        )
    dashboards = ET.SubElement(root, "dashboards")
    for spec in sheets:
        dashboard_filters = list(FILTER_FIELDS)
        if competitive and "Heat Map" in spec.name:
            dashboard_filters.insert(0, "Month")
        _add_dashboard(
            dashboards,
            spec.name,
            f"WS - {spec.name}",
            datasource_name,
            dashboard_filters,
        )

    windows = ET.SubElement(root, "windows", {"source-height": "30"})
    for spec in sheets:
        window = ET.SubElement(
            windows,
            "window",
            {"class": "worksheet", "name": f"WS - {spec.name}"},
        )
        cards = ET.SubElement(window, "cards")
        left = ET.SubElement(cards, "edge", {"name": "left"})
        strip = ET.SubElement(left, "strip", {"size": "160"})
        ET.SubElement(strip, "card", {"type": "pages"})
        ET.SubElement(strip, "card", {"type": "filters"})
        ET.SubElement(strip, "card", {"type": "marks"})
        top = ET.SubElement(cards, "edge", {"name": "top"})
        top_columns = ET.SubElement(top, "strip", {"size": "2147483647"})
        ET.SubElement(top_columns, "card", {"type": "columns"})
        top_rows = ET.SubElement(top, "strip", {"size": "2147483647"})
        ET.SubElement(top_rows, "card", {"type": "rows"})
        top_title = ET.SubElement(top, "strip", {"size": "31"})
        ET.SubElement(top_title, "card", {"type": "title"})
        ET.SubElement(
            window,
            "simple-id",
            {"uuid": "{" + str(uuid.uuid4()).upper() + "}"},
        )
    for spec in sheets:
        window = ET.SubElement(
            windows,
            "window",
            {"class": "dashboard", "maximized": "true", "name": spec.name},
        )
        viewpoints = ET.SubElement(window, "viewpoints")
        viewpoint = ET.SubElement(
            viewpoints, "viewpoint", {"name": f"WS - {spec.name}"}
        )
        ET.SubElement(viewpoint, "zoom", {"type": "entire-view"})
        ET.SubElement(window, "active", {"id": "-1"})
        ET.SubElement(
            window,
            "simple-id",
            {"uuid": "{" + str(uuid.uuid4()).upper() + "}"},
        )

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    comment = f"<!-- {workbook_title}: generated by week8_10_tableau_development.py -->\n".encode()
    declaration_end = xml_body.find(b"?>") + 2
    return xml_body[:declaration_end] + b"\n" + comment + xml_body[declaration_end + 1 :]


def atomic_write_twbx(
    workbook_xml: bytes,
    workbook_name: str,
    data_path: Path,
    hyper_path: Path,
    destination: Path,
) -> None:
    """Package a TWB, source CSV, and Hyper extract into an atomic TWBX."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr(workbook_name, workbook_xml)
            archive.write(data_path, arcname=f"Data/{data_path.name}")
            archive.write(
                hyper_path,
                arcname=f"Data/Extracts/{hyper_path.name}",
            )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_twbx(
    path: Path,
    workbook_name: str,
    data_filename: str,
    hyper_filename: str,
) -> None:
    """Validate package entries, workbook XML, source CSV, and Hyper extract."""

    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path.name} is not a valid ZIP/TWBX archive")
    with zipfile.ZipFile(path) as archive:
        archive.testzip()
        required = {
            workbook_name,
            f"Data/{data_filename}",
            f"Data/Extracts/{hyper_filename}",
        }
        missing = required.difference(archive.namelist())
        if missing:
            raise ValueError(f"{path.name} is missing package entries: {sorted(missing)}")
        root = ET.fromstring(archive.read(workbook_name))
        if root.tag != "workbook":
            raise ValueError(f"{workbook_name} does not contain a Tableau workbook root")
        if not root.findall("./dashboards/dashboard"):
            raise ValueError(f"{workbook_name} contains no dashboards")


def write_readme(
    market_rows: int,
    competitive_rows: int,
    latest_month: pd.Timestamp,
) -> None:
    """Write concise usage and acceptance-test notes."""

    content = f"""# Weeks 8-10 Tableau Deliverables

Generated from Residential CRMLS records dated January 2024 through
{latest_month:%B %Y}.

## Workbooks

- `market_analysis.twbx` — {market_rows:,} listing/sale event rows and six dashboards.
- `competitive_analysis.twbx` — {competitive_rows:,} closed-sale rows and five dashboards.

Both packages embed their source CSV and a Tableau Hyper extract. Open each
`.twbx` directly in Tableau Desktop or Tableau Public. If Tableau prompts to
upgrade the workbook, accept the prompt and save it in the installed version.

All required views expose City, CountyOrParish, PostalCode, and PropertySubType
filters. Competitive heat maps additionally expose Month. PostalCode is assigned
Tableau's ZIP Code geographic role, while valid CRMLS latitude/longitude values
are retained for map marks.

## Rebuild

From the repository root on Windows:

```powershell
.venv\\Scripts\\python.exe py\\week8_10_tableau_development.py
```

The script reads monthly source exports without modifying them and atomically
replaces only the generated dashboard data and workbook packages.
"""
    destination = TABLEAU_DIR / "README.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)


def main() -> int:
    """Build data sources, package both workbooks, and validate outputs."""

    try:
        TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
        listing_files = discover_listing_files()
        sold_files = discover_sold_files()
        if not listing_files or not sold_files:
            raise FileNotFoundError(
                "Expected CRMLSListingYYYYMM.csv and CRMLSSoldYYYYMM.csv files in csv/."
            )

        log(
            f"Found {len(listing_files)} listing months and "
            f"{len(sold_files)} sold months."
        )
        listings = prepare_listing_data(listing_files)
        sold = prepare_sold_data(sold_files)
        market = normalize_tableau_types(build_market_data(listings, sold))
        competitive = normalize_tableau_types(build_competitive_data(sold))

        log(f"Writing {len(market):,} market dashboard rows...")
        atomic_write_csv(market, MARKET_DATA_FILE)
        log(f"Writing {len(competitive):,} competitive dashboard rows...")
        atomic_write_csv(competitive, COMPETITIVE_DATA_FILE)
        log("Creating Tableau Hyper extracts...")
        atomic_write_hyper(market, MARKET_DATA_FILE, MARKET_HYPER_FILE)
        atomic_write_hyper(
            competitive,
            COMPETITIVE_DATA_FILE,
            COMPETITIVE_HYPER_FILE,
        )

        market_xml = build_workbook_xml(
            market,
            "Market Analysis",
            "federated.market",
            "Market Dashboard Data",
            MARKET_DATA_FILE.name,
            MARKET_HYPER_FILE.name,
            MARKET_SHEETS,
        )
        competitive_xml = build_workbook_xml(
            competitive,
            "Competitive Analysis",
            "federated.competitive",
            "Competitive Dashboard Data",
            COMPETITIVE_DATA_FILE.name,
            COMPETITIVE_HYPER_FILE.name,
            COMPETITIVE_SHEETS,
            competitive=True,
        )

        market_package = TABLEAU_DIR / "market_analysis.twbx"
        competitive_package = TABLEAU_DIR / "competitive_analysis.twbx"
        log("Packaging market_analysis.twbx...")
        atomic_write_twbx(
            market_xml,
            "market_analysis.twb",
            MARKET_DATA_FILE,
            MARKET_HYPER_FILE,
            market_package,
        )
        log("Packaging competitive_analysis.twbx...")
        atomic_write_twbx(
            competitive_xml,
            "competitive_analysis.twb",
            COMPETITIVE_DATA_FILE,
            COMPETITIVE_HYPER_FILE,
            competitive_package,
        )

        validate_twbx(
            market_package,
            "market_analysis.twb",
            MARKET_DATA_FILE.name,
            MARKET_HYPER_FILE.name,
        )
        validate_twbx(
            competitive_package,
            "competitive_analysis.twb",
            COMPETITIVE_DATA_FILE.name,
            COMPETITIVE_HYPER_FILE.name,
        )
        latest_month = max(market["Month"].max(), competitive["Month"].max())
        write_readme(len(market), len(competitive), latest_month)

        log("Validation passed: both TWBX archives contain valid XML and data.")
        log(f"Created: {market_package}")
        log(f"Created: {competitive_package}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, pd.errors.ParserError) as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
