import ast
import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {
    "listings": ROOT / "py" / "listings_pipeline.py",
    "sold": ROOT / "py" / "sold_pipeline.py",
}
INITIAL_MERGE = ROOT / "py" / "initial_merge.py"
DATA_CURATION = ROOT / "py" / "data_curation.py"


def load_pipeline(name):
    path = PIPELINES[name]
    if not path.exists():
        raise AssertionError(f"Missing pipeline module: {path}")
    spec = importlib.util.spec_from_file_location(f"{name}_pipeline_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_initial_merge():
    if not INITIAL_MERGE.exists():
        raise AssertionError(f"Missing initial merge module: {INITIAL_MERGE}")
    spec = importlib.util.spec_from_file_location("initial_merge_under_test", INITIAL_MERGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data_curation():
    if not DATA_CURATION.exists():
        raise AssertionError(f"Missing data curation module: {DATA_CURATION}")
    spec = importlib.util.spec_from_file_location("data_curation_under_test", DATA_CURATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class PipelineContractTests(unittest.TestCase):
    def test_missing_file_uses_2024_utc_fallback(self):
        for name in PIPELINES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                module = load_pipeline(name)
                result = module.get_start_time(Path(temp_dir) / "missing.csv")
                self.assertEqual(
                    result,
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                )

    def test_existing_file_uses_maximum_valid_incremental_date(self):
        cases = {
            "listings": "ListingContractDate",
            "sold": "CloseDate",
        }
        for name, date_column in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                module = load_pipeline(name)
                path = Path(temp_dir) / "data.csv"
                pd.DataFrame(
                    {
                        "ListingKey": ["a", "b", "c"],
                        date_column: [
                            "2025-02-01T02:00:00Z",
                            "invalid",
                            "2025-02-03T04:05:06.789Z",
                        ],
                    }
                ).to_csv(path, index=False)
                self.assertEqual(
                    module.get_start_time(path),
                    datetime(2025, 2, 3, 4, 5, 6, 789000, tzinfo=timezone.utc),
                )

    def test_filters_use_inclusive_lower_and_exclusive_upper_bounds(self):
        start = datetime(2025, 1, 2, 3, 4, 5, 6000, tzinfo=timezone.utc)
        end = datetime(2025, 1, 9, 10, 11, 12, 13000, tzinfo=timezone.utc)
        listings = load_pipeline("listings").build_filter(start, end)
        sold = load_pipeline("sold").build_filter(start, end)
        self.assertEqual(
            listings,
            "ListingContractDate ge 2025-01-02T03:04:05.006Z and "
            "ListingContractDate lt 2025-01-09T10:11:12.013Z",
        )
        self.assertEqual(
            sold,
            "MlsStatus eq 'Closed' and CloseDate ge 2025-01-02T03:04:05.006Z "
            "and CloseDate lt 2025-01-09T10:11:12.013Z",
        )

    def test_fetch_follows_next_link_without_reusing_initial_params(self):
        for name in PIPELINES:
            with self.subTest(name=name):
                module = load_pipeline(name)
                session = FakeSession(
                    [
                        FakeResponse(
                            {
                                "value": [{"ListingKey": "one"}],
                                "@odata.nextLink": "https://example.test/page-2",
                            }
                        ),
                        FakeResponse({"value": [{"ListingKey": "two"}]}),
                    ]
                )
                params = {"$top": 1000, "$filter": "date filter"}
                records = module.fetch_all_records(
                    session,
                    {"Authorization": "Bearer token"},
                    params,
                )
                self.assertEqual([row["ListingKey"] for row in records], ["one", "two"])
                self.assertEqual(session.calls[0][0], module.url)
                self.assertEqual(session.calls[0][1]["params"], params)
                self.assertEqual(session.calls[1][0], "https://example.test/page-2")
                self.assertIsNone(session.calls[1][1]["params"])

    def test_merge_keeps_latest_api_representation(self):
        for name in PIPELINES:
            with self.subTest(name=name):
                module = load_pipeline(name)
                existing = pd.DataFrame(
                    [{"ListingKey": "same", "ListPrice": 100}, {"ListingKey": "old", "ListPrice": 50}]
                )
                merged = module.merge_records(
                    existing,
                    [
                        {"ListingKey": "same", "ListPrice": 125},
                        {"ListingKey": "new", "ListPrice": 75},
                    ],
                )
                by_key = merged.set_index("ListingKey")
                self.assertEqual(by_key.loc["same", "ListPrice"], 125)
                self.assertEqual(set(by_key.index), {"same", "old", "new"})

    def test_merge_rejects_missing_or_blank_listing_keys(self):
        for name in PIPELINES:
            module = load_pipeline(name)
            for invalid in ({"ListPrice": 1}, {"ListingKey": "  ", "ListPrice": 1}):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "ListingKey"):
                        module.merge_records(pd.DataFrame(), [invalid])

    def test_atomic_write_preserves_existing_file_when_replace_fails(self):
        for name in PIPELINES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                module = load_pipeline(name)
                target = Path(temp_dir) / "data.csv"
                original = b"ListingKey,ListPrice\nold,100\n"
                target.write_bytes(original)
                frame = pd.DataFrame([{"ListingKey": "new", "ListPrice": 200}])
                with mock.patch.object(module.os, "replace", side_effect=OSError("blocked")):
                    with self.assertRaisesRegex(OSError, "blocked"):
                        module.atomic_write_csv(frame, target)
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_retrieval_failure_never_overwrites_existing_csv(self):
        date_columns = {
            "listings": "ListingContractDate",
            "sold": "CloseDate",
        }
        for name, date_column in date_columns.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                module = load_pipeline(name)
                target = Path(temp_dir) / "data.csv"
                pd.DataFrame(
                    [{"ListingKey": "safe", date_column: "2025-01-01T00:00:00Z"}]
                ).to_csv(target, index=False)
                original = target.read_bytes()
                session = FakeSession(
                    [
                        FakeResponse({"access_token": "token"}),
                        requests.Timeout("network timeout"),
                    ]
                )
                with self.assertRaises(requests.Timeout):
                    module.run_pipeline(
                        session=session,
                        now=datetime(2025, 1, 8, tzinfo=timezone.utc),
                        csv_path=target,
                    )
                self.assertEqual(target.read_bytes(), original)

    def test_first_run_can_create_header_only_csv(self):
        for name in PIPELINES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                module = load_pipeline(name)
                target = Path(temp_dir) / "data.csv"
                module.atomic_write_csv(pd.DataFrame(columns=module.FIELDS), target)
                loaded = pd.read_csv(target)
                self.assertEqual(list(loaded.columns), module.FIELDS)
                self.assertTrue(loaded.empty)

    def test_every_print_call_sets_flush_true(self):
        paths = {**PIPELINES, "initial_merge": INITIAL_MERGE, "data_curation": DATA_CURATION}
        for name, path in paths.items():
            with self.subTest(name=name):
                if not path.exists():
                    self.fail(f"Missing pipeline module: {path}")
                tree = ast.parse(path.read_text(encoding="utf-8"))
                print_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ]
                self.assertGreater(len(print_calls), 0)
                for call in print_calls:
                    flush = next((kw.value for kw in call.keywords if kw.arg == "flush"), None)
                    self.assertIsInstance(flush, ast.Constant)
                    self.assertIs(flush.value, True)


class InitialMergeTests(unittest.TestCase):
    def test_monthly_files_merge_deduplicate_and_normalize_columns(self):
        initial_merge = load_initial_merge()
        listings = load_pipeline("listings")
        sold = load_pipeline("sold")

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_dir = Path(temp_dir) / "csv"
            csv_dir.mkdir()

            pd.DataFrame(
                [
                    {"ListingKey": "a", "ListPrice": 100, "UnexpectedColumn": "drop-me"},
                    {"ListingKey": "b", "ListPrice": 200},
                ]
            ).to_csv(csv_dir / "crmls_listed_202602.csv", index=False)
            pd.DataFrame(
                [
                    {"ListingKey": "b", "ListPrice": 250},
                    {"ListingKey": "c", "ListPrice": 300},
                ]
            ).to_csv(csv_dir / "crmls_listed_202603.csv", index=False)
            pd.DataFrame([{"ListingKey": "ignore", "ListPrice": 999}]).to_csv(
                csv_dir / "not_crmls_listed_202603.csv", index=False
            )

            pd.DataFrame(
                [
                    {"ListingKey": "s1", "ClosePrice": 10},
                    {"ListingKey": "s2", "ClosePrice": 20},
                ]
            ).to_csv(csv_dir / "crmls_sold_202602.csv", index=False)
            pd.DataFrame(
                [
                    {"ListingKey": "s2", "ClosePrice": 25},
                    {"ListingKey": "s3", "ClosePrice": 30},
                ]
            ).to_csv(csv_dir / "crmls_sold_202603.csv", index=False)

            listed_stats = initial_merge.merge_monthly_files(
                csv_dir=csv_dir,
                prefix="crmls_listed_",
                target_path=csv_dir / "listings.csv",
                fields=listings.FIELDS,
            )
            sold_stats = initial_merge.merge_monthly_files(
                csv_dir=csv_dir,
                prefix="crmls_sold_",
                target_path=csv_dir / "sold.csv",
                fields=sold.FIELDS,
            )

            self.assertEqual(listed_stats["input_files"], 2)
            self.assertEqual(sold_stats["input_files"], 2)

            merged_listings = pd.read_csv(csv_dir / "listings.csv")
            self.assertEqual(list(merged_listings.columns), listings.FIELDS)
            listed_by_key = merged_listings.set_index("ListingKey")
            self.assertEqual(set(listed_by_key.index), {"a", "b", "c"})
            self.assertEqual(listed_by_key.loc["b", "ListPrice"], 250)

            merged_sold = pd.read_csv(csv_dir / "sold.csv")
            self.assertEqual(list(merged_sold.columns), sold.FIELDS)
            sold_by_key = merged_sold.set_index("ListingKey")
            self.assertEqual(set(sold_by_key.index), {"s1", "s2", "s3"})
            self.assertEqual(sold_by_key.loc["s2", "ClosePrice"], 25)

    def test_missing_monthly_files_creates_header_only_target(self):
        initial_merge = load_initial_merge()
        listings = load_pipeline("listings")

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_dir = Path(temp_dir) / "csv"
            csv_dir.mkdir()

            stats = initial_merge.merge_monthly_files(
                csv_dir=csv_dir,
                prefix="crmls_listed_",
                target_path=csv_dir / "listings.csv",
                fields=listings.FIELDS,
            )

            self.assertEqual(stats["input_files"], 0)
            merged = pd.read_csv(csv_dir / "listings.csv")
            self.assertEqual(list(merged.columns), listings.FIELDS)
            self.assertTrue(merged.empty)


class DataCurationTests(unittest.TestCase):
    def test_curate_dataframe_drops_sparse_and_text_columns_but_keeps_required_fields(self):
        data_curation = load_data_curation()
        frame = pd.DataFrame(
            {
                "ListingKey": ["a", "b", "c", "d", "e"],
                "ListingContractDate": ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"],
                "CloseDate": [None, None, None, None, None],
                "OriginalListPrice": [100, 110, 120, 130, 140],
                "ClosePrice": [None, 105, 115, 125, 135],
                "City": ["Irvine", "Irvine", "Tustin", "Orange", "Orange"],
                "PostalCode": ["92602", "92603", "92780", "92866", "92867"],
                "ListPrice": [101, 111, 121, 131, 141],
                "DaysOnMarket": [1, 2, 3, 4, 5],
                "Latitude": [33.7, 33.8, 33.9, 34.0, 34.1],
                "HugeRemarks": ["long text"] * 5,
                "SparseNoise": [None, None, None, None, None],
            }
        )

        curated, dropped = data_curation.curate_dataframe(frame)

        self.assertIn("CloseDate", curated.columns)
        self.assertNotIn("HugeRemarks", curated.columns)
        self.assertNotIn("SparseNoise", curated.columns)
        self.assertIn("SparseNoise", dropped)
        self.assertEqual(dropped["SparseNoise"], 1.0)
        self.assertEqual(
            list(curated.columns)[:7],
            [
                "ListingKey",
                "ListingContractDate",
                "CloseDate",
                "OriginalListPrice",
                "ClosePrice",
                "City",
                "PostalCode",
            ],
        )

    def test_process_dataset_writes_new_curated_file_without_overwriting_source(self):
        data_curation = load_data_curation()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "listings.csv"
            output_path = Path(temp_dir) / "listings_curated.csv"
            source = pd.DataFrame(
                {
                    "ListingKey": ["a", "b"],
                    "ListingContractDate": ["2026-05-01", "2026-05-02"],
                    "CloseDate": [None, None],
                    "OriginalListPrice": [100, 200],
                    "ClosePrice": [95, 190],
                    "City": ["Irvine", "Tustin"],
                    "PostalCode": ["92602", "92780"],
                    "UnparsedAddress": ["123 Very Long Street", "456 Very Long Street"],
                }
            )
            source.to_csv(input_path, index=False)
            original_bytes = input_path.read_bytes()

            stats = data_curation.process_dataset(input_path, output_path)

            self.assertEqual(input_path.read_bytes(), original_bytes)
            self.assertTrue(output_path.exists())
            curated = pd.read_csv(output_path)
            self.assertEqual(stats["output_rows"], 2)
            self.assertNotIn("UnparsedAddress", curated.columns)
            self.assertIn("ListingKey", curated.columns)

    def test_unified_wide_table_left_joins_sold_with_conflict_suffixes(self):
        data_curation = load_data_curation()
        listings = pd.DataFrame(
            {
                "ListingKey": ["a", "b"],
                "ListingContractDate": ["2026-05-01", "2026-05-02"],
                "City": ["Irvine", "Tustin"],
                "PostalCode": ["92602", "92780"],
                "OriginalListPrice": [100, 200],
            }
        )
        sold = pd.DataFrame(
            {
                "ListingKey": ["a"],
                "CloseDate": ["2026-05-10"],
                "ClosePrice": [110],
                "City": ["Irvine Sold"],
            }
        )

        wide = data_curation.build_unified_wide_table(listings, sold)

        self.assertEqual(list(wide["ListingKey"]), ["a", "b"])
        self.assertIn("City_sold", wide.columns)
        self.assertEqual(wide.loc[0, "City"], "Irvine")
        self.assertEqual(wide.loc[0, "City_sold"], "Irvine Sold")
        self.assertEqual(wide.loc[0, "ClosePrice"], 110)
        self.assertTrue(pd.isna(wide.loc[1, "ClosePrice"]))

    def test_monthly_market_metrics_aggregate_for_tableau(self):
        data_curation = load_data_curation()
        wide = pd.DataFrame(
            {
                "ListingKey": ["a", "b", "c"],
                "ListingContractDate": ["2026-05-01", "2026-05-15", "2026-06-01"],
                "CloseDate_sold": ["2026-05-10", None, "2026-06-20"],
                "City": ["Irvine", "Irvine", "Tustin"],
                "PostalCode": ["92602", "92602", "92780"],
                "OriginalListPrice": [100.0, 300.0, 500.0],
                "ClosePrice_sold": [110.0, None, 520.0],
            }
        )

        metrics = data_curation.build_monthly_market_metrics(wide)

        self.assertEqual(
            list(metrics.columns),
            [
                "Month",
                "City",
                "PostalCode",
                "CountOfListings",
                "CountOfSold",
                "MeanOriginalListPrice",
                "MeanClosePrice",
                "MeanAbsorptionDays",
            ],
        )
        irvine = metrics[
            (metrics["Month"] == "2026-05")
            & (metrics["City"] == "Irvine")
            & (metrics["PostalCode"] == "92602")
        ].iloc[0]
        self.assertEqual(irvine["CountOfListings"], 2)
        self.assertEqual(irvine["CountOfSold"], 1)
        self.assertEqual(irvine["MeanOriginalListPrice"], 200.0)
        self.assertEqual(irvine["MeanClosePrice"], 110.0)
        self.assertEqual(irvine["MeanAbsorptionDays"], 9.0)

    def test_tableau_outputs_are_written_without_overwriting_curated_inputs(self):
        data_curation = load_data_curation()

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_dir = Path(temp_dir)
            listings_path = csv_dir / "listings_curated.csv"
            sold_path = csv_dir / "sold_curated.csv"
            wide_path = csv_dir / "crmls_unified_wide_table.csv"
            metrics_path = csv_dir / "crmls_monthly_market_metrics.csv"

            pd.DataFrame(
                {
                    "ListingKey": ["a", "b"],
                    "ListingContractDate": ["2026-05-01", "2026-05-15"],
                    "City": ["Irvine", "Irvine"],
                    "PostalCode": ["92602", "92602"],
                    "OriginalListPrice": [100.0, 300.0],
                }
            ).to_csv(listings_path, index=False)
            pd.DataFrame(
                {
                    "ListingKey": ["a"],
                    "CloseDate": ["2026-05-10"],
                    "ClosePrice": [110.0],
                }
            ).to_csv(sold_path, index=False)
            original_listings = listings_path.read_bytes()
            original_sold = sold_path.read_bytes()

            stats = data_curation.process_tableau_outputs(
                listings_path=listings_path,
                sold_path=sold_path,
                wide_output_path=wide_path,
                metrics_output_path=metrics_path,
            )

            self.assertEqual(listings_path.read_bytes(), original_listings)
            self.assertEqual(sold_path.read_bytes(), original_sold)
            self.assertTrue(wide_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertEqual(stats["wide_rows"], 2)
            self.assertEqual(stats["metrics_rows"], 1)


if __name__ == "__main__":
    unittest.main()
