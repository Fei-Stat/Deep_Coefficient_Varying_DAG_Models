"""Targeted checks: chronology/scalers, zero news, edge orientation and export support.

Run: python check_news_stock.py
These checks do not claim statistical graph recovery.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.NewsStockDAG import NewsStockDAG, is_dag, to_return_units
from models.TransMIL_regression import DynamicDAG
from news_stock_config import EXCLUDED_STOCKS
from news_stock_data import load_prepared, prepare_data


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.dates = np.asarray([f"2020-01-{day:02d}" for day in range(1, 13)])
        self.columns = [f"S{i:02d}" for i in range(99)] + list(EXCLUDED_STOCKS)
        rates = np.arange(1, 12)[:, None] * np.linspace(0.001, 0.003, 104)[None, :]
        self.prices = 100 * np.exp(np.vstack((np.zeros(104), np.cumsum(rates, axis=0))))
        self.news_dates = np.delete(self.dates, 3)
        self.news = np.ones((len(self.news_dates), 25, 50), dtype=np.float32)
        self.news[1, 0] = 0

    def prepare(self, directory, prices=None):
        output = Path(directory) / "data.npz"
        with patch("news_stock_data.read_daily_matrix", return_value=(
                self.dates, self.columns, self.prices if prices is None else prices)), \
             patch("news_stock_data.read_news", return_value=(self.news_dates, self.news)), \
             patch("news_stock_data.file_sha256", return_value="test"):
            prepare_data("prices.xlsx", "news.xlsx", output, 0.6, 0.2)
        return load_prepared(output)

    def test_difference_precedes_join_and_zeros_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.prepare(tmp)
        row = int(np.flatnonzero(data["dates"] == "2020-01-05")[0])
        expected = np.log(self.prices[4, :99] / self.prices[3, :99])
        np.testing.assert_allclose(data["log_returns"][row], expected)
        self.assertEqual(data["previous_dates"][row], "2020-01-04")
        self.assertEqual(len(data["stock_ids"]), 99)
        self.assertTrue(np.all(data["news"][0, 0] == 0))
        idx = data["train_indices"]
        np.testing.assert_allclose(data["Y"][idx].mean(0), 0, atol=2e-7)
        np.testing.assert_allclose(data["Y"][idx].std(0), 1, atol=2e-7)

    def test_future_price_change_cannot_change_training_scaler(self):
        future_prices = self.prices.copy()
        future_prices[-1] *= 2
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = self.prepare(a), self.prepare(b, future_prices)
        np.testing.assert_array_equal(first["mean"], second["mean"])
        np.testing.assert_array_equal(first["scale"], second["scale"])
        np.testing.assert_array_equal(first["Y"][first["train_indices"]],
                                      second["Y"][second["train_indices"]])

    def test_missing_retained_stock_is_not_silently_filled(self):
        broken = self.prices.copy()
        broken[3, 0] = np.nan
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            self.prepare(tmp, broken)

    def test_duplicate_dates_rejected_by_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare(tmp)
            path = Path(tmp) / "data.npz"
            with np.load(path, allow_pickle=False) as original:
                arrays = {key: original[key] for key in original.files}
            arrays["dates"][1] = arrays["dates"][0]
            np.savez_compressed(path, **arrays)
            with self.assertRaises(ValueError):
                load_prepared(path)


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = NewsStockDAG(n_stocks=4, hidden_dim=8, context_dim=16)

    def test_original_core_and_news_only_inference(self):
        self.assertIsInstance(self.model.dag, DynamicDAG)
        result = self.model(torch.zeros(3, 25, 50))
        self.assertEqual(result["beta"].shape, (3, 4, 4))
        self.assertEqual(result["context"].shape, (3, 16))
        self.assertTrue(torch.isfinite(result["beta"]).all())
        torch.testing.assert_close(torch.diagonal(result["beta"], dim1=1, dim2=2),
                                   torch.zeros(3, 4))

    def test_fixed_mask_remains_acyclic_after_optimizer_step(self):
        info = self.model.freeze_graph(0.0)
        self.assertGreater(info["selected_edges"], 0)
        self.assertTrue(is_dag(self.model.dag.structural_mask))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        news, y = torch.randn(5, 25, 50), torch.randn(5, 4)
        out = self.model(news, y, lambda_group=0.001)
        out["loss"].backward()
        optimizer.step()
        self.model.dag.zero_forbidden_edges()
        out = self.model(news)
        forbidden = 1 - self.model.dag.structural_mask
        torch.testing.assert_close(out["beta"] * forbidden, torch.zeros(5, 4, 4))
        self.assertTrue(is_dag(self.model.dag.structural_mask))

    def test_original_return_units_reproduce_same_equations(self):
        news = torch.randn(5, 25, 50)
        y = torch.randn(5, 4, dtype=torch.float64)
        mean = torch.tensor([0.001, -0.002, 0.003, 0.004], dtype=torch.float64)
        scale = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
        out = self.model(news)
        beta, intercept = out["beta"].double(), out["intercept"].double()
        raw_beta, raw_intercept = to_return_units(beta, intercept, mean, scale)
        fitted_std = torch.einsum("bi,bij->bj", y, beta) + intercept
        raw_y = mean + scale * y
        fitted_raw = torch.einsum("bi,bij->bj", raw_y, raw_beta) + raw_intercept
        torch.testing.assert_close(fitted_raw, mean + scale * fitted_std)

    def test_state_round_trip_preserves_fixed_graph(self):
        self.model.freeze_graph(0.0)
        clone = NewsStockDAG(**self.model.model_config)
        clone.load_state_dict(self.model.state_dict())
        news = torch.randn(2, 25, 50)
        torch.testing.assert_close(self.model(news)["beta"], clone(news)["beta"])
        self.assertTrue(is_dag(clone.dag.structural_mask))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
