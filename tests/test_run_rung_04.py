from __future__ import annotations

import unittest

import pandas as pd

from scripts.correlation_training import run_rung_04 as module


class RunRung04Tests(unittest.TestCase):
    def test_dense_history_may_cover_an_unbalanced_evaluation_panel(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        history = pd.DataFrame(
            [
                {"stock": stock, "forecast_date": forecast_date}
                for forecast_date in dates
                for stock in ("A", "B")
            ]
        )
        evaluation = history.drop(index=[1]).reset_index(drop=True)
        self.assertTrue(
            module.validate_dcc_history_panel(evaluation, history, dates)
        )

    def test_history_missing_a_stock_date_is_rejected(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        history = pd.DataFrame(
            [
                {"stock": stock, "forecast_date": forecast_date}
                for forecast_date in dates
                for stock in ("A", "B")
            ]
        )
        evaluation = history.copy()
        with self.assertRaisesRegex(ValueError, "dense"):
            module.validate_dcc_history_panel(
                evaluation,
                history.drop(index=[1]).reset_index(drop=True),
                dates,
            )

    def test_history_missing_a_whole_official_session_is_rejected(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        complete = pd.DataFrame(
            [
                {"stock": stock, "forecast_date": forecast_date}
                for forecast_date in dates
                for stock in ("A", "B")
            ]
        )
        missing_session = complete[
            complete["forecast_date"].ne(dates[1])
        ].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "official-session"):
            module.validate_dcc_history_panel(
                missing_session, missing_session, dates
            )


if __name__ == "__main__":
    unittest.main()
