# Weeks 8-10 Tableau Deliverables

Generated from Residential CRMLS records dated January 2024 through
June 2026.

## Workbooks

- `market_analysis.twbx` — 1,062,971 listing/sale event rows, one multi-view
  overview dashboard, and six detailed dashboards.
- `competitive_analysis.twbx` — 447,663 closed-sale rows, one
  multi-view overview dashboard, and five detailed dashboards.

## Published dashboards

- [Los Angeles County Market Analysis](https://public.tableau.com/views/market_analysis_17895254525090/MarketAnalysisDashboard?:showVizHome=no)
- [Los Angeles County Competitive Analysis](https://public.tableau.com/views/competitive_analysis_17895255168460/CompetitiveAnalysisDashboard?:showVizHome=no)

Both packages embed their source CSV and a Tableau Hyper extract. Open each
`.twbx` directly in Tableau Desktop or Tableau Public. If Tableau prompts to
upgrade the workbook, accept the prompt and save it in the installed version.

The overview dashboards display several worksheets simultaneously in tiled
horizontal/vertical containers with a shared filter rail. All required views
expose City, CountyOrParish, PostalCode, and PropertySubType filters.
Competitive views additionally expose Month. PostalCode is assigned Tableau's
ZIP Code geographic role, while valid CRMLS latitude/longitude values are
retained for map marks.

## Rebuild

From the repository root on Windows:

```powershell
.venv\Scripts\python.exe py\week8_10_tableau_development.py
```

The script reads monthly source exports without modifying them and atomically
replaces only the generated dashboard data and workbook packages.
