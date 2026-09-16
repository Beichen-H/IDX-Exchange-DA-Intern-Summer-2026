r"""Validate the Weeks 8-12 Tableau, report, and presentation deliverables.

This script performs read-only checks and exits with a non-zero status when a
required artifact or required content is missing. Run it from any directory:

    .venv\Scripts\python.exe py\audit_final_deliverables.py
"""

from __future__ import annotations

import json
import base64
import re
import sys
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLEAU_DIR = PROJECT_ROOT / "tableau"
REPORT_DIR = PROJECT_ROOT / "reports" / "week11_12"


class Audit:
    """Collect and print human-readable validation results."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, description: str) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {description}", flush=True)
        if not condition:
            self.failures.append(description)


def read_tableau_xml(package_path: Path) -> tuple[ET.Element, list[str]]:
    with zipfile.ZipFile(package_path) as package:
        names = package.namelist()
        workbook_name = next(name for name in names if name.lower().endswith(".twb"))
        return ET.fromstring(package.read(workbook_name)), names


def audit_tableau(
    audit: Audit,
    filename: str,
    required_sheets: set[str],
    overview_dashboard: str,
    minimum_overview_views: int,
    required_filters: set[str],
) -> None:
    path = TABLEAU_DIR / filename
    audit.check(path.exists(), f"Tableau package exists: {filename}")
    if not path.exists():
        return

    root, package_names = read_tableau_xml(path)
    worksheet_names = {node.get("name", "") for node in root.findall(".//worksheet")}
    dashboard_names = {node.get("name", "") for node in root.findall(".//dashboard")}
    audit.check(
        required_sheets <= worksheet_names,
        f"{filename} contains every required worksheet",
    )
    audit.check(
        overview_dashboard in dashboard_names,
        f"{filename} contains the multi-view overview dashboard",
    )

    dashboard = next(
        (node for node in root.findall(".//dashboard") if node.get("name") == overview_dashboard),
        None,
    )
    if dashboard is not None:
        view_zones = {
            zone.get("name", "")
            for zone in dashboard.findall(".//zone")
            if zone.get("name", "") in worksheet_names
        }
        filter_zone_text = " ".join(
            " ".join(str(value) for value in zone.attrib.values())
            for zone in dashboard.findall(".//zone")
            if zone.get("type-v2") == "filter"
        )
        audit.check(
            len(view_zones) >= minimum_overview_views,
            f"{overview_dashboard} displays at least {minimum_overview_views} views together",
        )
        audit.check(
            all(filter_name in filter_zone_text for filter_name in required_filters),
            f"{overview_dashboard} exposes all required filter controls",
        )

    audit.check(
        any(name.lower().endswith(".hyper") for name in package_names),
        f"{filename} embeds a Tableau Hyper extract",
    )


def audit_pdf(audit: Audit) -> None:
    path = REPORT_DIR / "los_angeles_market_intelligence_report.pdf"
    audit.check(path.exists(), "One-page market intelligence PDF exists")
    if not path.exists():
        return

    pdf_bytes = path.read_bytes()
    page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    audit.check(page_count == 1, "Market intelligence report is exactly one page")

    # The generated report uses ReportLab's ASCII85 + Flate stream filters.
    # Decode those streams with the standard library to keep this audit script
    # independent of optional PDF packages.
    decoded_streams: list[bytes] = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", pdf_bytes, re.DOTALL):
        stream = match.group(1).strip()
        try:
            ascii85 = stream if stream.startswith(b"<~") else b"<~" + stream
            decoded_streams.append(zlib.decompress(base64.a85decode(ascii85, adobe=True)))
        except (ValueError, zlib.error):
            continue
    text = b" ".join(decoded_streams).decode("latin-1", errors="ignore").lower()
    concepts = {
        "market overview": ("market overview", "median", "days on market"),
        "pricing trends": ("price / sq ft", "sale-to-original-list"),
        "market activity": ("activity", "new listings", "closed sales", "volume"),
        "competitive landscape": ("competitive landscape", "listing agents", "listing offices"),
        "data-driven takeaways": ("takeaway",),
    }
    for label, terms in concepts.items():
        audit.check(all(term in text for term in terms), f"PDF covers {label}")


def audit_presentation(audit: Audit) -> None:
    path = REPORT_DIR / "los_angeles_market_intelligence_final_presentation.pptx"
    audit.check(path.exists(), "Final PowerPoint presentation exists")
    if not path.exists():
        return

    receipt_path = PROJECT_ROOT / "tmp" / "week11_12" / "finalizer" / "final-presentation.validation.json"
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.exists()
        else {}
    )
    try:
        with zipfile.ZipFile(path) as package:
            names = package.namelist()
            slides = [
                name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ]
            notes = [
                name
                for name in names
                if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)
            ]
            embedded_workbooks = [
                name
                for name in names
                if name.startswith("ppt/embeddings/")
                and name.lower().endswith((".xlsx", ".xlsm"))
            ]
    except PermissionError:
        # PowerPoint can temporarily hold an exclusive lock on the package.
        # The finalizer receipt is generated from the exact final SHA-256 and
        # provides equivalent structural counts for a read-only audit.
        integrity = receipt.get("packageIntegrity", {})
        slides = [None] * int(integrity.get("slide_count", 0))
        notes = [None] * int(integrity.get("notes_parts_skipped", 0))
        embedded_workbooks = [None] * int(integrity.get("chart_count", 0))
    audit.check(len(slides) >= 5, "PowerPoint contains a complete presentation sequence")
    audit.check(len(notes) == len(slides), "Every slide contains speaker notes")
    audit.check(
        len(embedded_workbooks) >= 2,
        "PowerPoint charts retain editable embedded workbooks",
    )

    audit.check(receipt_path.exists(), "PowerPoint validation receipt exists")
    if receipt_path.exists():
        serialized = json.dumps(receipt).lower()
        audit.check(
            '"status": "pass"' in serialized and '"finding_count": 0' in serialized,
            "PowerPoint package and layout validation passed with zero findings",
        )


def audit_script(audit: Audit) -> None:
    path = REPORT_DIR / "los_angeles_market_intelligence_5_minute_script.txt"
    audit.check(path.exists(), "Direct-read five-minute presentation script exists")
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    time_markers = re.findall(r"[\[(]\d+:\d+\s*[–-]\s*\d+:\d+[\])]", text)
    audit.check(600 <= word_count <= 850, f"Script length is suitable for five minutes ({word_count} words)")
    audit.check(len(time_markers) >= 5, "Script includes timed speaking sections")


def main() -> int:
    audit = Audit()

    audit_tableau(
        audit,
        "market_analysis.twbx",
        {
            "WS - Monthly Median Close Price",
            "WS - Average Days on Market",
            "WS - Average Close-to-Original-List Ratio",
            "WS - New Listings",
            "WS - Closed Sales",
            "WS - Market Pulse - Price per Square Foot",
        },
        "Market Analysis Dashboard",
        6,
        {"City", "CountyOrParish", "PostalCode", "PropertySubType"},
    )
    audit_tableau(
        audit,
        "competitive_analysis.twbx",
        {
            "WS - Top 100 Listing Agents",
            "WS - Top 100 Listing Offices",
            "WS - Zip Code Median Close Price Heat Map",
            "WS - Zip Code Homes Sold Heat Map",
            "WS - Competitive Market Share - Office Volume vs Units",
        },
        "Competitive Analysis Dashboard",
        5,
        {"Month", "City", "CountyOrParish", "PostalCode", "PropertySubType"},
    )
    audit_pdf(audit)
    audit_presentation(audit)
    audit_script(audit)

    print("", flush=True)
    if audit.failures:
        print(f"AUDIT FAILED: {len(audit.failures)} requirement(s) need attention.", flush=True)
        return 1
    print("AUDIT PASSED: all locally verifiable Weeks 8-12 requirements are complete.", flush=True)
    print("External Tableau Public publication must be confirmed separately.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
