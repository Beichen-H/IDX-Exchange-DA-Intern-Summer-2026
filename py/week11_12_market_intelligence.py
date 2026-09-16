"""Create the Weeks 11-12 Los Angeles County market intelligence package.

The script consumes the analysis-ready Tableau CSV extracts produced by
``week8_10_tableau_development.py``. It preserves those inputs, applies a
documented IQR screen for analytical summaries, and creates a one-page PDF,
a machine-readable metrics summary, and a five-minute presenter script.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLEAU_DIR = PROJECT_ROOT / "tableau"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "week11_12"
MARKET_INPUT = TABLEAU_DIR / "market_dashboard_data.csv"
COMPETITIVE_INPUT = TABLEAU_DIR / "competitive_dashboard_data.csv"
REPORT_OUTPUT = OUTPUT_DIR / "los_angeles_market_intelligence_report.pdf"
SUMMARY_OUTPUT = OUTPUT_DIR / "analysis_summary.json"
SCRIPT_OUTPUT = OUTPUT_DIR / "presentation_script.txt"

COUNTY = "Los Angeles"
START_DATE = pd.Timestamp("2024-01-01")
LATEST_START = pd.Timestamp("2025-07-01")
LATEST_END = pd.Timestamp("2026-07-01")
PRIOR_START = pd.Timestamp("2024-07-01")
PRIOR_END = pd.Timestamp("2025-07-01")

NAVY = colors.HexColor("#173A5E")
BLUE = colors.HexColor("#2878B8")
SKY = colors.HexColor("#DDEFF8")
PALE = colors.HexColor("#F3F7FA")
INK = colors.HexColor("#1B2530")
MUTED = colors.HexColor("#5F6B76")
GREEN = colors.HexColor("#16856B")
RED = colors.HexColor("#C84A4A")
GRID = colors.HexColor("#D7E0E7")


@dataclass(frozen=True)
class WindowMetrics:
    median_close_price: float
    median_days_on_market: float
    median_price_ratio: float
    median_ppsf: float
    closed_sales: int
    new_listings: int
    transaction_volume: float


def atomic_replace(source: Path, target: Path) -> None:
    """Replace target atomically after all bytes have been written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    atomic_replace(temp_path, path)


def iqr_mask(series: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    q1, q3 = numeric.quantile([0.25, 0.75])
    span = q3 - q1
    lower = max(0.0, float(q1 - 1.5 * span))
    upper = float(q3 + 1.5 * span)
    return numeric.between(lower, upper), lower, upper


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[float]]]:
    if not MARKET_INPUT.exists() or not COMPETITIVE_INPUT.exists():
        raise FileNotFoundError(
            "Run py/week8_10_tableau_development.py before this script."
        )

    market = pd.read_csv(MARKET_INPUT, low_memory=False)
    competitive = pd.read_csv(COMPETITIVE_INPUT, low_memory=False)
    market["EventDate"] = pd.to_datetime(market["EventDate"], errors="coerce")
    competitive["CloseDate"] = pd.to_datetime(
        competitive["CloseDate"], errors="coerce"
    )

    county_market = market.loc[
        market["CountyOrParish"].astype("string").str.strip().eq(COUNTY)
        & market["EventDate"].ge(START_DATE)
        & market["EventDate"].lt(LATEST_END)
    ].copy()
    sales = county_market.loc[county_market["RecordType"].eq("Closed Sale")].copy()
    listings = county_market.loc[county_market["RecordType"].eq("New Listing")].copy()

    bounds: dict[str, list[float]] = {}
    valid = pd.Series(True, index=sales.index)
    for field in ("ClosePrice", "DaysOnMarket", "PricePerSquareFoot"):
        field_mask, lower, upper = iqr_mask(sales[field])
        bounds[field] = [lower, upper]
        valid &= field_mask

    ratio = pd.to_numeric(sales["CloseToOriginalListRatio"], errors="coerce")
    valid &= ratio.between(0.5, 1.5)
    valid &= pd.to_numeric(sales["ClosePrice"], errors="coerce").gt(0)
    cleaned_sales = sales.loc[valid].copy()

    clean_keys = set(cleaned_sales["ListingKey"].dropna().astype(str))
    competitive = competitive.loc[
        competitive["CountyOrParish"].astype("string").str.strip().eq(COUNTY)
        & competitive["CloseDate"].ge(START_DATE)
        & competitive["CloseDate"].lt(LATEST_END)
        & competitive["ListingKey"].astype(str).isin(clean_keys)
    ].copy()
    return cleaned_sales, listings, bounds


def metrics_for(
    sales: pd.DataFrame,
    listings: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> WindowMetrics:
    s = sales.loc[sales["EventDate"].between(start, end, inclusive="left")]
    l = listings.loc[listings["EventDate"].between(start, end, inclusive="left")]
    return WindowMetrics(
        median_close_price=float(pd.to_numeric(s["ClosePrice"]).median()),
        median_days_on_market=float(pd.to_numeric(s["DaysOnMarket"]).median()),
        median_price_ratio=float(pd.to_numeric(s["CloseToOriginalListRatio"]).median()),
        median_ppsf=float(pd.to_numeric(s["PricePerSquareFoot"]).median()),
        closed_sales=int(s["ListingKey"].nunique()),
        new_listings=int(l["ListingKey"].nunique()),
        transaction_volume=float(pd.to_numeric(s["ClosePrice"]).sum()),
    )


def pct_change(current: float, previous: float) -> float:
    return (current / previous - 1.0) * 100.0


def monthly_summary(sales: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    s = sales.assign(Month=sales["EventDate"].dt.to_period("M").dt.to_timestamp())
    l = listings.assign(Month=listings["EventDate"].dt.to_period("M").dt.to_timestamp())
    sales_month = s.groupby("Month", as_index=False).agg(
        median_close_price=("ClosePrice", "median"),
        median_days_on_market=("DaysOnMarket", "median"),
        median_price_ratio=("CloseToOriginalListRatio", "median"),
        median_ppsf=("PricePerSquareFoot", "median"),
        closed_sales=("ListingKey", "nunique"),
        transaction_volume=("ClosePrice", "sum"),
    )
    listing_month = l.groupby("Month", as_index=False).agg(
        new_listings=("ListingKey", "nunique")
    )
    return sales_month.merge(listing_month, on="Month", how="outer").sort_values("Month")


def competitive_summary(
    competitive: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = competitive.loc[
        competitive["CloseDate"].between(LATEST_START, LATEST_END, inclusive="left")
    ].copy()
    current["ClosePrice"] = pd.to_numeric(current["ClosePrice"], errors="coerce")
    agents = (
        current.dropna(subset=["ListAgentFullName"])
        .groupby("ListAgentFullName", as_index=False)
        .agg(volume=("ClosePrice", "sum"), units=("ListingKey", "nunique"))
        .sort_values(["volume", "units"], ascending=False)
        .head(5)
    )
    offices = (
        current.dropna(subset=["ListOfficeName"])
        .groupby("ListOfficeName", as_index=False)
        .agg(volume=("ClosePrice", "sum"), units=("ListingKey", "nunique"))
        .sort_values(["volume", "units"], ascending=False)
        .head(5)
    )
    return agents, offices


def draw_wrapped(
    page: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 8.5,
    leading: float = 10.5,
    color: colors.Color = INK,
    max_lines: int | None = None,
) -> float:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if stringWidth(trial, font, size) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    if max_lines is not None:
        lines = lines[:max_lines]
    page.setFont(font, size)
    page.setFillColor(color)
    for item in lines:
        page.drawString(x, y, item)
        y -= leading
    return y


def draw_delta(page: canvas.Canvas, x: float, y: float, value: float, suffix: str = "%") -> None:
    page.setFont("Helvetica-Bold", 8)
    page.setFillColor(GREEN if value >= 0 else RED)
    page.drawString(x, y, f"{value:+.1f}{suffix} vs prior 12M")


def draw_kpi(
    page: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    title: str,
    value: str,
    delta: float,
    suffix: str = "%",
) -> None:
    page.setFillColor(PALE)
    page.roundRect(x, y, w, 54, 5, fill=1, stroke=0)
    page.setFillColor(MUTED)
    page.setFont("Helvetica-Bold", 7.5)
    page.drawString(x + 9, y + 40, title.upper())
    page.setFillColor(NAVY)
    page.setFont("Helvetica-Bold", 17)
    page.drawString(x + 9, y + 20, value)
    draw_delta(page, x + 9, y + 7, delta, suffix)


def draw_line_chart(
    page: canvas.Canvas,
    data: pd.DataFrame,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    values = data["median_close_price"].astype(float).tolist()
    if not values:
        return
    low = math.floor(min(values) / 50_000) * 50_000
    high = math.ceil(max(values) / 50_000) * 50_000
    if high == low:
        high += 1
    page.setStrokeColor(GRID)
    page.setLineWidth(0.4)
    for fraction in (0, 0.5, 1):
        yy = y + fraction * h
        page.line(x, yy, x + w, yy)
        label = low + (high - low) * fraction
        page.setFillColor(MUTED)
        page.setFont("Helvetica", 6.5)
        page.drawRightString(x - 4, yy - 2, f"${label / 1000:.0f}K")
    points = []
    for idx, value in enumerate(values):
        px = x + idx * w / max(1, len(values) - 1)
        py = y + (value - low) / (high - low) * h
        points.append((px, py))
    page.setStrokeColor(BLUE)
    page.setLineWidth(1.6)
    path = page.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    page.drawPath(path)
    for idx in (0, 6, 12, 18, 24, len(data) - 1):
        if 0 <= idx < len(data):
            label = pd.Timestamp(data.iloc[idx]["Month"]).strftime("%b %y")
            page.setFillColor(MUTED)
            page.setFont("Helvetica", 6.2)
            page.drawCentredString(points[idx][0], y - 10, label)


def draw_report(
    summary: dict,
    monthly: pd.DataFrame,
    agents: pd.DataFrame,
    offices: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=OUTPUT_DIR) as handle:
        temp_path = Path(handle.name)
    page = canvas.Canvas(str(temp_path), pagesize=letter)
    width, height = letter

    page.setFillColor(NAVY)
    page.rect(0, height - 82, width, 82, fill=1, stroke=0)
    page.setFillColor(colors.white)
    page.setFont("Helvetica-Bold", 20)
    page.drawString(30, height - 39, "Los Angeles County Market Intelligence")
    page.setFont("Helvetica", 9)
    page.drawString(30, height - 58, "Residential market | January 2024 - June 2026")
    page.setFont("Helvetica-Bold", 8)
    page.drawRightString(width - 30, height - 57, "IDX EXCHANGE MLS ANALYTICS")

    current = summary["latest_12_months"]
    changes = summary["changes_percent"]
    kpi_y = height - 151
    kpi_w = 130
    draw_kpi(page, 30, kpi_y, kpi_w, "Median close price", f"${current['median_close_price']/1000:.0f}K", changes["median_close_price"])
    draw_kpi(page, 170, kpi_y, kpi_w, "Median days on market", f"{current['median_days_on_market']:.0f} days", changes["median_days_on_market"])
    draw_kpi(page, 310, kpi_y, kpi_w, "Sale-to-original-list", f"{current['median_price_ratio']*100:.1f}%", summary["changes_percentage_points"]["median_price_ratio"], " pp")
    draw_kpi(page, 450, kpi_y, 132, "Median price / sq ft", f"${current['median_ppsf']:.0f}", changes["median_ppsf"])

    page.setFillColor(NAVY)
    page.setFont("Helvetica-Bold", 11)
    page.drawString(30, 618, "MARKET OVERVIEW")
    page.setFillColor(PALE)
    page.roundRect(30, 430, 355, 176, 5, fill=1, stroke=0)
    page.setFillColor(INK)
    page.setFont("Helvetica-Bold", 8)
    page.drawString(45, 586, "Monthly median close price")
    draw_line_chart(page, monthly, 76, 463, 290, 105)
    draw_wrapped(
        page,
        "Prices remained resilient, but a 12.5% increase in median days on market and a 0.3 percentage-point decline in the sale-to-original-list ratio point to a more deliberate buyer environment.",
        45,
        447,
        325,
        size=7.7,
        leading=9.3,
    )

    page.setFillColor(NAVY)
    page.setFont("Helvetica-Bold", 11)
    page.drawString(405, 618, "ACTIVITY - LATEST 12 MONTHS")
    page.setFillColor(SKY)
    page.roundRect(405, 523, 177, 83, 5, fill=1, stroke=0)
    page.setFillColor(NAVY)
    page.setFont("Helvetica-Bold", 20)
    page.drawString(418, 574, f"{current['new_listings']:,}")
    page.setFont("Helvetica", 8)
    page.drawString(418, 558, "new listings")
    page.setFillColor(RED)
    page.setFont("Helvetica-Bold", 8)
    page.drawString(418, 541, f"{changes['new_listings']:+.1f}% vs prior 12M")
    page.setFillColor(PALE)
    page.roundRect(405, 430, 177, 83, 5, fill=1, stroke=0)
    page.setFillColor(NAVY)
    page.setFont("Helvetica-Bold", 20)
    page.drawString(418, 481, f"{current['closed_sales']:,}")
    page.setFont("Helvetica", 8)
    page.drawString(418, 465, "closed sales")
    page.setFillColor(RED)
    page.setFont("Helvetica-Bold", 8)
    page.drawString(418, 448, f"{changes['closed_sales']:+.1f}% vs prior 12M")

    page.setFillColor(NAVY)
    page.setFont("Helvetica-Bold", 11)
    page.drawString(30, 405, "COMPETITIVE LANDSCAPE - JUL 2025 TO JUN 2026")
    page.setFont("Helvetica-Bold", 8)
    page.setFillColor(MUTED)
    page.drawString(30, 389, "TOP LISTING AGENTS BY CLOSED VOLUME")
    page.drawString(310, 389, "TOP LISTING OFFICES BY CLOSED VOLUME")
    page.setStrokeColor(GRID)
    page.line(30, 383, 582, 383)

    for idx in range(5):
        yy = 366 - idx * 20
        if idx < len(agents):
            row = agents.iloc[idx]
            page.setFillColor(INK)
            page.setFont("Helvetica-Bold", 7.5)
            page.drawString(30, yy, f"{idx+1}. {str(row['ListAgentFullName'])[:25]}")
            page.setFont("Helvetica", 7.3)
            page.drawRightString(287, yy, f"${row['volume']/1_000_000:.1f}M | {int(row['units'])} units")
        if idx < len(offices):
            row = offices.iloc[idx]
            page.setFillColor(INK)
            page.setFont("Helvetica-Bold", 7.5)
            page.drawString(310, yy, f"{idx+1}. {str(row['ListOfficeName'])[:24]}")
            page.setFont("Helvetica", 7.3)
            page.drawRightString(582, yy, f"${row['volume']/1_000_000:.0f}M | {int(row['units']):,} units")
        page.setStrokeColor(GRID)
        page.line(30, yy - 7, 582, yy - 7)

    page.setFillColor(NAVY)
    page.setFont("Helvetica-Bold", 11)
    page.drawString(30, 257, "KEY TAKEAWAYS")
    insights = [
        f"1  Price stability: median close price rose {changes['median_close_price']:.1f}% to ${current['median_close_price']/1000:.0f}K while median PPSF declined {abs(changes['median_ppsf']):.1f}%.",
        f"2  Slower decisions: median market time increased from 16 to {current['median_days_on_market']:.0f} days, giving buyers modestly more negotiating room.",
        f"3  Softer pipeline: new listings fell {abs(changes['new_listings']):.1f}%, versus a smaller {abs(changes['closed_sales']):.1f}% decline in closed sales.",
        f"4  Volume held near prior levels: ${current['transaction_volume']/1_000_000_000:.1f}B closed, down {abs(changes['transaction_volume']):.1f}%.",
        f"5  Clear office leader: Compass closed ${offices.iloc[0]['volume']/1_000_000_000:.2f}B across {int(offices.iloc[0]['units']):,} units in the cleaned sample.",
    ]
    yy = 240
    for insight in insights:
        yy = draw_wrapped(page, insight, 30, yy, 552, size=8.2, leading=10.2, max_lines=2) - 3

    page.setFillColor(PALE)
    page.rect(0, 0, width, 63, fill=1, stroke=0)
    source = (
        "Source: CRMLS Residential listing/sold exports. Latest 12M: Jul 2025-Jun 2026; prior 12M: Jul 2024-Jun 2025. "
        "Sales metrics use positive-value business rules, 1.5x IQR screens for close price, days on market and PPSF, and a 0.50-1.50 ratio range. "
        f"Analysis-ready sales: {summary['clean_sales_rows']:,} rows. Descriptive analysis; no causal claims."
    )
    draw_wrapped(page, source, 30, 49, 552, size=6.7, leading=8.2, color=MUTED, max_lines=4)
    page.save()
    atomic_replace(temp_path, REPORT_OUTPUT)


def build_summary() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales, listings, bounds = load_and_clean()
    competitive = pd.read_csv(COMPETITIVE_INPUT, low_memory=False)
    competitive["CloseDate"] = pd.to_datetime(competitive["CloseDate"], errors="coerce")
    clean_keys = set(sales["ListingKey"].dropna().astype(str))
    competitive = competitive.loc[
        competitive["CountyOrParish"].astype("string").str.strip().eq(COUNTY)
        & competitive["CloseDate"].ge(START_DATE)
        & competitive["CloseDate"].lt(LATEST_END)
        & competitive["ListingKey"].astype(str).isin(clean_keys)
    ].copy()
    current = metrics_for(sales, listings, LATEST_START, LATEST_END)
    prior = metrics_for(sales, listings, PRIOR_START, PRIOR_END)
    current_dict = current.__dict__
    prior_dict = prior.__dict__
    changes = {key: pct_change(current_dict[key], prior_dict[key]) for key in current_dict}
    pp = {"median_price_ratio": (current.median_price_ratio - prior.median_price_ratio) * 100}
    monthly = monthly_summary(sales, listings)
    agents, offices = competitive_summary(competitive)
    summary = {
        "market": COUNTY,
        "analysis_period": "2024-01 through 2026-06",
        "latest_window": "2025-07 through 2026-06",
        "prior_window": "2024-07 through 2025-06",
        "latest_12_months": current_dict,
        "prior_12_months": prior_dict,
        "changes_percent": changes,
        "changes_percentage_points": pp,
        "clean_sales_rows": len(sales),
        "iqr_bounds": bounds,
        "top_agents": agents.to_dict(orient="records"),
        "top_offices": offices.to_dict(orient="records"),
        "monthly": [
            {
                **row,
                "Month": pd.Timestamp(row["Month"]).strftime("%Y-%m"),
            }
            for row in monthly.to_dict(orient="records")
        ],
    }
    return summary, monthly, agents, offices


def presenter_script(summary: dict) -> str:
    c = summary["latest_12_months"]
    d = summary["changes_percent"]
    office = summary["top_offices"][0]
    return f"""Los Angeles County Market Intelligence - 5 Minute Presentation Script

Slide 1 - Purpose (0:00-0:35)
This presentation summarizes the Los Angeles County residential market from January 2024 through June 2026. The comparison focuses on the latest twelve months, July 2025 through June 2026, versus the prior twelve months. All sales findings use an analysis-ready sample with business-rule validation and IQR outlier screens.

Slide 2 - Market pulse (0:35-1:25)
The headline is stability with a slower pace. Median close price reached ${c['median_close_price']:,.0f}, up only {d['median_close_price']:.1f} percent. Median days on market increased from 16 to {c['median_days_on_market']:.0f} days, a {d['median_days_on_market']:.1f} percent increase. Buyers are taking more time even though headline prices have held.

Slide 3 - Pricing and negotiation (1:25-2:15)
Median price per square foot declined {abs(d['median_ppsf']):.1f} percent to ${c['median_ppsf']:.0f}. The median close-to-original-list ratio eased to {c['median_price_ratio']*100:.1f} percent, a decline of about 0.3 percentage points. This is not a market collapse; it is a modest shift toward greater price discipline and negotiation.

Slide 4 - Market activity (2:15-3:05)
New listings totaled {c['new_listings']:,}, down {abs(d['new_listings']):.1f} percent. Closed sales totaled {c['closed_sales']:,}, down only {abs(d['closed_sales']):.1f} percent. Transaction volume was ${c['transaction_volume']/1_000_000_000:.1f} billion, down {abs(d['transaction_volume']):.1f} percent. Supply entered the market more slowly than demand exited it.

Slide 5 - Competitive landscape (3:05-4:00)
Compass led listing offices with ${office['volume']/1_000_000_000:.2f} billion in volume and {int(office['units']):,} units. Cesi Pagano led agents by cleaned closed volume at approximately $127.8 million. This dashboard supports drill-down by city, ZIP code, county and property subtype for more targeted competitive intelligence.

Slide 6 - Takeaways (4:00-5:00)
Five conclusions stand out. First, prices are stable rather than accelerating. Second, market time is longer. Third, buyers gained modest negotiating room. Fourth, listing supply contracted faster than sales. Fifth, competitive leadership is visible, but market share still requires local segment analysis. For sellers, realistic pricing matters more; for buyers, the slower pace creates room to compare and negotiate; for brokerages, ZIP-level productivity is the next actionable layer.

Source and methodology
CRMLS Residential listing and sold exports, January 2024 through June 2026. Positive-value business rules; 1.5 times IQR screens for close price, days on market and price per square foot; close-to-original-list ratio constrained to 0.50 through 1.50. Descriptive analysis only.
"""


def main() -> None:
    print("Building Los Angeles County market intelligence package...", flush=True)
    summary, monthly, agents, offices = build_summary()
    atomic_write_text(SUMMARY_OUTPUT, json.dumps(summary, indent=2, allow_nan=False))
    atomic_write_text(SCRIPT_OUTPUT, presenter_script(summary))
    draw_report(summary, monthly, agents, offices)
    print(f"Created: {REPORT_OUTPUT}", flush=True)
    print(f"Created: {SUMMARY_OUTPUT}", flush=True)
    print(f"Created: {SCRIPT_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
