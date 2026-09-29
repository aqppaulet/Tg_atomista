import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tellurite_tg.data import (
    _parse_composition,
    file_md5,
    load_external_dataset,
)
from tellurite_tg.evaluation import regression_metrics


class PipelineUnitTests(unittest.TestCase):
    def test_parse_composition_uses_mol_and_weight_fields(self) -> None:
        mol, weight = _parse_composition(
            "\x7fTeO2\x7f159.60\x7f80.0\x7f70.0"
            "\x7fZnO\x7f81.38\x7f20.0\x7f30.0\x7f"
        )
        self.assertEqual(mol, {"TeO2": 70.0, "ZnO": 30.0})
        self.assertEqual(weight, {"TeO2": 80.0, "ZnO": 20.0})

    def test_regression_metrics_for_perfect_prediction(self) -> None:
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(metrics["MAE_C"], 0.0)
        self.assertEqual(metrics["RMSE_C"], 0.0)
        self.assertEqual(metrics["R2"], 1.0)

    def test_file_md5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "known.txt"
            source.write_bytes(b"abc")
            self.assertEqual(
                file_md5(source), "900150983cd24fb0d6963f7d28e17f72"
            )

    def test_external_composition_must_be_inside_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "external.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "inside",
                        "TeO2": 70.0,
                        "ZnO": 15.0,
                        "La2O3": 5.0,
                        "WO3": 10.0,
                    },
                    {
                        "sample_id": "outside",
                        "TeO2": 74.0,
                        "ZnO": 9.0,
                        "La2O3": 5.0,
                        "WO3": 12.0,
                    },
                ]
            ).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "outside"):
                load_external_dataset(
                    source,
                    ["TeO2", "ZnO", "La2O3", "WO3", "Other"],
                )

    def test_external_general_tellurite_domain_does_not_require_zinc_or_lanthanum(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "external.csv"
            pd.DataFrame(
                [{"sample_id": "tellurite", "TeO2": 70.0, "Na2O": 30.0}]
            ).to_csv(source, index=False)
            features, _, sample_ids = load_external_dataset(
                source,
                ["TeO2", "Na2O", "Other"],
                enforce_component_domain=False,
            )
            self.assertEqual(sample_ids.iloc[0], "tellurite")
            self.assertAlmostEqual(features.loc[0, "TeO2"], 70.0)


if __name__ == "__main__":
    unittest.main()
