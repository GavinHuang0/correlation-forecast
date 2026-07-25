from __future__ import annotations

import unittest

import numpy as np

from scripts.correlation_training import dcc


class DccTests(unittest.TestCase):
    def test_fitted_parameters_are_stationary_and_forecasts_are_bounded(self):
        rng = np.random.default_rng(19)
        covariance = np.array([[1.0, 0.55], [0.55, 1.0]])
        returns = rng.multivariate_normal([0, 0], covariance, size=320) / 100
        left = dcc.fit_garch(returns[:, 0])
        right = dcc.fit_garch(returns[:, 1])
        left_h, left_e, left_z = dcc.filter_garch(returns[:, 0], left)
        right_h, right_e, right_z = dcc.filter_garch(returns[:, 1], right)
        fitted_dcc = dcc.fit_dcc(np.column_stack([left_z, right_z]))
        _, q_after = dcc.filter_dcc(
            np.column_stack([left_z, right_z]), fitted_dcc
        )
        one, five = dcc.correlation_forecasts(
            left_h[-1],
            right_h[-1],
            left_e[-1],
            right_e[-1],
            q_after[-1],
            left,
            right,
            fitted_dcc,
        )
        self.assertLess(left.alpha + left.beta, dcc.MAX_PERSISTENCE)
        self.assertLess(right.alpha + right.beta, dcc.MAX_PERSISTENCE)
        self.assertLess(fitted_dcc.a + fitted_dcc.b, dcc.MAX_PERSISTENCE)
        self.assertTrue(-1 <= one <= 1)
        self.assertTrue(-1 <= five <= 1)

    def test_missing_observation_uses_expected_state_update(self):
        parameters = dcc.GarchParameters(
            omega=0.1,
            alpha=0.1,
            beta=0.8,
            mean=0.0,
            scale=1.0,
            converged=True,
            objective=0.0,
            iterations=1,
        )
        variance, residual, standardized = dcc.filter_garch(
            np.array([0.2, np.nan, 0.1]), parameters
        )
        expected = parameters.omega + (
            parameters.alpha + parameters.beta
        ) * variance[1]
        self.assertAlmostEqual(variance[2], expected)
        self.assertTrue(np.isnan(standardized[1]))
        self.assertTrue(np.isnan(residual[1]))


if __name__ == "__main__":
    unittest.main()
