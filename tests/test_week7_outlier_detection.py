import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "py" / "week7_outlier_detection.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "week7_outlier_detection_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_frame() -> pd.DataFrame:
    normal_prices = [480_000, 490_000, 500_000, 510_000, 520_000, 530_000, 540_000, 550_000]
    return pd.DataFrame(
        {
            "ListingKey": [f"listing-{index}" for index in range(10)],
            "ClosePrice": [*normal_prices, 50_000_000, 0],
            "OriginalListPrice": [500_000] * 10,
            "LivingArea": [1_500, 1_525, 1_550, 1_575, 1_600, 1_625, 1_650, 1_675, 20_000, 1_500],
            "DaysOnMarket": [10, 12, 14, 16, 18, 20, 22, 24, 400, -1],
        }
    )


class Week7OutlierDetectionTests(unittest.TestCase):
    def test_flags_outliers_without_removing_rows_from_full_dataset(self):
        module = load_module()

        flagged, filtered, comparison = module.apply_outlier_detection(source_frame())

        self.assertEqual(len(flagged), 10)
        self.assertTrue(flagged.loc[8, "close_price_iqr_outlier_flag"])
        self.assertTrue(flagged.loc[8, "any_iqr_outlier_flag"])
        self.assertTrue(flagged.loc[9, "business_rule_invalid_flag"])
        self.assertTrue(flagged.loc[9, "analysis_exclusion_flag"])
        self.assertLess(len(filtered), len(flagged))
        self.assertNotIn("listing-8", set(filtered["ListingKey"]))
        self.assertNotIn("listing-9", set(filtered["ListingKey"]))
        self.assertEqual(set(comparison["Metric"]), set(module.OUTLIER_FIELDS))

    def test_derives_price_per_square_foot_and_close_ratio(self):
        module = load_module()

        prepared = module.prepare_numeric_metrics(source_frame())

        self.assertAlmostEqual(prepared.loc[0, "PricePerSqFt"], 320.0)
        self.assertAlmostEqual(
            prepared.loc[0, "CloseToOriginalListRatio"], 0.96
        )
        self.assertTrue(pd.isna(prepared.loc[9, "PricePerSqFt"]) or prepared.loc[9, "PricePerSqFt"] == 0)

    def test_percentile_flags_are_review_signals_not_independent_exclusions(self):
        module = load_module()
        frame = source_frame().iloc[:8].copy()

        flagged, filtered, _ = module.apply_outlier_detection(frame)

        percentile_only = flagged[
            flagged["any_percentile_extreme_flag"]
            & ~flagged["any_iqr_outlier_flag"]
            & ~flagged["business_rule_invalid_flag"]
        ]
        self.assertGreater(len(percentile_only), 0)
        self.assertTrue(
            set(percentile_only["ListingKey"]).issubset(set(filtered["ListingKey"]))
        )

    def test_written_report_compares_size_and_medians(self):
        module = load_module()
        flagged, filtered, comparison = module.apply_outlier_detection(source_frame())

        report = module.build_comparison_report(
            input_path=Path("sold.csv"),
            flagged=flagged,
            filtered=filtered,
            comparison=comparison,
            multiplier=1.5,
            low_percentile=0.01,
            high_percentile=0.99,
        )

        self.assertIn("Dataset Size", report)
        self.assertIn("Median Comparison", report)
        self.assertIn("Full flagged rows", report)
        self.assertIn("Clean filtered rows", report)


if __name__ == "__main__":
    unittest.main()
