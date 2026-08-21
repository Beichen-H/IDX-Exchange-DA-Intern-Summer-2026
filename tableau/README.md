# Weeks 8-10 Tableau Deliverables

Generated from Residential CRMLS records dated January 2024 through
June 2026.

## Workbooks

- `market_analysis.twbx` — 1,062,971 listing/sale event rows and six dashboards.
- `competitive_analysis.twbx` — 447,663 closed-sale rows and five dashboards.

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
.venv\Scripts\python.exe py\week8_10_tableau_development.py
```

The script reads monthly source exports without modifying them and atomically
replaces only the generated dashboard data and workbook packages.
