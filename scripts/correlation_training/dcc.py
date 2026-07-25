"""Small, auditable bivariate Gaussian GARCH(1,1)-DCC(1,1) implementation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize


MAX_PERSISTENCE = 0.999
VARIANCE_FLOOR = 1e-10


@dataclass(frozen=True)
class GarchParameters:
    omega: float
    alpha: float
    beta: float
    mean: float
    scale: float
    converged: bool
    objective: float
    iterations: int


@dataclass(frozen=True)
class DccParameters:
    a: float
    b: float
    qbar_00: float
    qbar_01: float
    qbar_11: float
    converged: bool
    objective: float
    iterations: int

    @property
    def qbar(self) -> np.ndarray:
        return np.array(
            [[self.qbar_00, self.qbar_01], [self.qbar_01, self.qbar_11]],
            dtype=float,
        )


def _persistence_from_unconstrained(
    first: float, second: float
) -> tuple[float, float]:
    values = np.exp(np.clip([0.0, first, second], -30, 30))
    denominator = float(values.sum())
    return (
        MAX_PERSISTENCE * float(values[1]) / denominator,
        MAX_PERSISTENCE * float(values[2]) / denominator,
    )


def _unconstrained_from_persistence(
    first: float, second: float
) -> tuple[float, float]:
    slack = MAX_PERSISTENCE - first - second
    if min(first, second, slack) <= 0:
        raise ValueError("Persistence starting values must be interior")
    return math.log(first / slack), math.log(second / slack)


def filter_garch(
    values: np.ndarray,
    parameters: GarchParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter conditional variance; missing returns receive expected updates."""

    values = np.asarray(values, dtype=float)
    scaled = values * parameters.scale
    residuals = scaled - parameters.mean
    unconditional = parameters.omega / max(
        1 - parameters.alpha - parameters.beta, 1e-8
    )
    variance = np.full(len(values), np.nan, dtype=float)
    standardized = np.full(len(values), np.nan, dtype=float)
    current = max(unconditional, VARIANCE_FLOOR)
    for position, residual in enumerate(residuals):
        variance[position] = current
        if np.isfinite(residual):
            standardized[position] = residual / math.sqrt(current)
            current = (
                parameters.omega
                + parameters.alpha * residual * residual
                + parameters.beta * current
            )
        else:
            current = parameters.omega + (
                parameters.alpha + parameters.beta
            ) * current
        current = max(float(current), VARIANCE_FLOOR)
    return variance, residuals, standardized


def fit_garch(
    returns: np.ndarray,
    *,
    scale: float = 100.0,
) -> GarchParameters:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 250:
        raise ValueError("GARCH requires at least 250 finite daily returns")
    scaled = values * scale
    mean = float(scaled.mean())
    residuals = scaled - mean
    sample_variance = max(float(np.var(residuals)), 1e-6)

    def objective(theta: np.ndarray) -> float:
        omega = math.exp(float(theta[0]))
        alpha, beta = _persistence_from_unconstrained(
            float(theta[1]), float(theta[2])
        )
        current = sample_variance
        likelihood = 0.0
        for residual in residuals:
            current = max(current, VARIANCE_FLOOR)
            likelihood += math.log(current) + residual * residual / current
            current = omega + alpha * residual * residual + beta * current
        value = 0.5 * likelihood
        return float(value) if np.isfinite(value) else 1e100

    candidates = []
    for alpha_start, beta_start in ((0.05, 0.90), (0.10, 0.80), (0.03, 0.95)):
        first, second = _unconstrained_from_persistence(
            alpha_start, beta_start
        )
        theta = np.array(
            [
                math.log(
                    max(
                        sample_variance
                        * (1 - alpha_start - beta_start),
                        1e-8,
                    )
                ),
                first,
                second,
            ]
        )
        result = minimize(
            objective,
            theta,
            method="L-BFGS-B",
            bounds=[(-20, 10), (-15, 15), (-15, 15)],
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        candidates.append(result)
    converged = [item for item in candidates if item.success and np.isfinite(item.fun)]
    if not converged:
        raise RuntimeError(
            "GARCH optimization did not converge: "
            + "; ".join(str(item.message) for item in candidates)
        )
    result = min(converged, key=lambda item: float(item.fun))
    omega = math.exp(float(result.x[0]))
    alpha, beta = _persistence_from_unconstrained(
        float(result.x[1]), float(result.x[2])
    )
    if not 0 <= alpha and 0 <= beta and alpha + beta < MAX_PERSISTENCE:
        raise AssertionError("Invalid fitted GARCH persistence")
    return GarchParameters(
        omega=omega,
        alpha=alpha,
        beta=beta,
        mean=mean,
        scale=scale,
        converged=True,
        objective=float(result.fun),
        iterations=int(result.nit),
    )


def normalize_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    diagonal = np.diag(q)
    if q.shape != (2, 2) or not np.isfinite(q).all() or (diagonal <= 0).any():
        raise ValueError("DCC Q is invalid")
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    correlation = q / denominator
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    if np.linalg.eigvalsh(correlation).min() <= 0:
        raise ValueError("DCC correlation is not positive definite")
    return correlation


def fit_dcc(standardized_residuals: np.ndarray) -> DccParameters:
    values = np.asarray(standardized_residuals, dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 250:
        raise ValueError("DCC requires at least 250 paired residuals")
    qbar = np.cov(values, rowvar=False, bias=True)
    qbar = (qbar + qbar.T) / 2
    eigenvalues = np.linalg.eigvalsh(qbar)
    if eigenvalues.min() <= 1e-8:
        qbar += np.eye(2) * (1e-8 - eigenvalues.min() + 1e-8)

    def objective(theta: np.ndarray) -> float:
        a, b = _persistence_from_unconstrained(
            float(theta[0]), float(theta[1])
        )
        q = qbar.copy()
        likelihood = 0.0
        for position, residual in enumerate(values):
            if position:
                previous = values[position - 1]
                q = (
                    (1 - a - b) * qbar
                    + a * np.outer(previous, previous)
                    + b * q
                )
            try:
                correlation = normalize_q(q)
                sign, logdet = np.linalg.slogdet(correlation)
                if sign <= 0:
                    return 1e100
                likelihood += logdet + float(
                    residual @ np.linalg.solve(correlation, residual)
                )
            except (ValueError, np.linalg.LinAlgError):
                return 1e100
        value = 0.5 * likelihood
        return float(value) if np.isfinite(value) else 1e100

    candidates = []
    for a_start, b_start in ((0.03, 0.95), (0.05, 0.90), (0.10, 0.80)):
        first, second = _unconstrained_from_persistence(a_start, b_start)
        result = minimize(
            objective,
            np.array([first, second]),
            method="L-BFGS-B",
            bounds=[(-15, 15), (-15, 15)],
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        candidates.append(result)
    converged = [item for item in candidates if item.success and np.isfinite(item.fun)]
    if not converged:
        raise RuntimeError(
            "DCC optimization did not converge: "
            + "; ".join(str(item.message) for item in candidates)
        )
    result = min(converged, key=lambda item: float(item.fun))
    a, b = _persistence_from_unconstrained(
        float(result.x[0]), float(result.x[1])
    )
    if not 0 <= a and 0 <= b and a + b < MAX_PERSISTENCE:
        raise AssertionError("Invalid fitted DCC persistence")
    return DccParameters(
        a=a,
        b=b,
        qbar_00=float(qbar[0, 0]),
        qbar_01=float(qbar[0, 1]),
        qbar_11=float(qbar[1, 1]),
        converged=True,
        objective=float(result.fun),
        iterations=int(result.nit),
    )


def filter_dcc(
    standardized_residuals: np.ndarray,
    parameters: DccParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional Q and post-observation Q for every date."""

    values = np.asarray(standardized_residuals, dtype=float)
    qbar = parameters.qbar
    conditional = np.empty((len(values), 2, 2), dtype=float)
    after = np.empty((len(values), 2, 2), dtype=float)
    q = qbar.copy()
    for position, residual in enumerate(values):
        conditional[position] = q
        correlation = normalize_q(q)
        if np.isfinite(residual).all():
            shock = np.outer(residual, residual)
        else:
            shock = correlation
        q = (
            (1 - parameters.a - parameters.b) * qbar
            + parameters.a * shock
            + parameters.b * q
        )
        normalize_q(q)
        after[position] = q
    return conditional, after


def next_variance(
    conditional_variance: float,
    residual: float,
    parameters: GarchParameters,
) -> float:
    if np.isfinite(residual):
        value = (
            parameters.omega
            + parameters.alpha * residual * residual
            + parameters.beta * conditional_variance
        )
    else:
        value = parameters.omega + (
            parameters.alpha + parameters.beta
        ) * conditional_variance
    return max(float(value), VARIANCE_FLOOR)


def correlation_forecasts(
    left_variance: float,
    right_variance: float,
    left_residual: float,
    right_residual: float,
    q_next: np.ndarray,
    left_parameters: GarchParameters,
    right_parameters: GarchParameters,
    dcc_parameters: DccParameters,
    *,
    horizon: int = 5,
) -> tuple[float, float]:
    """Return one-step and component-aggregated multi-step correlations."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    h_left = next_variance(
        left_variance, left_residual, left_parameters
    )
    h_right = next_variance(
        right_variance, right_residual, right_parameters
    )
    q = np.asarray(q_next, dtype=float)
    covariance_sum = 0.0
    left_sum = 0.0
    right_sum = 0.0
    first_correlation = np.nan
    for step in range(horizon):
        correlation_matrix = normalize_q(q)
        correlation = float(correlation_matrix[0, 1])
        if step == 0:
            first_correlation = correlation
        covariance_sum += correlation * math.sqrt(h_left * h_right)
        left_sum += h_left
        right_sum += h_right
        if step + 1 < horizon:
            h_left = left_parameters.omega + (
                left_parameters.alpha + left_parameters.beta
            ) * h_left
            h_right = right_parameters.omega + (
                right_parameters.alpha + right_parameters.beta
            ) * h_right
            q = (
                (1 - dcc_parameters.a - dcc_parameters.b)
                * dcc_parameters.qbar
                + dcc_parameters.a * correlation_matrix
                + dcc_parameters.b * q
            )
    multi = covariance_sum / math.sqrt(left_sum * right_sum)
    return float(np.clip(first_correlation, -1, 1)), float(
        np.clip(multi, -1, 1)
    )


def parameter_record(
    left: GarchParameters,
    right: GarchParameters,
    dcc: DccParameters,
) -> dict[str, object]:
    return {
        "left_garch": asdict(left),
        "right_garch": asdict(right),
        "dcc": asdict(dcc),
    }
