"""Bayesian idle/control detection from EEG and shared target evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal, special, stats


@dataclass(frozen=True, slots=True)
class IdleGateConfig:
    """Configuration for the binary idle/control stage."""

    sample_rate: float
    threshold: float = 0.50
    forgetting: float = 0.995
    evidence_weight: float = 0.35
    cycle_duration_s: float | None = None


@dataclass(slots=True)
class _NormalInverseGamma:
    mean0: float = 0.0
    kappa0: float = 0.25
    alpha0: float = 2.0
    beta0: float = 1.0
    count: float = 0.0
    total: float = 0.0
    total_square: float = 0.0

    def update(
        self,
        values: NDArray[np.float64] | float,
        *,
        forgetting: float = 1.0,
    ) -> None:
        if not 0 < forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        samples = np.atleast_1d(np.asarray(values, dtype=np.float64))
        if not np.all(np.isfinite(samples)):
            raise ValueError("features must be finite")
        self.count *= forgetting
        self.total *= forgetting
        self.total_square *= forgetting
        self.count += float(len(samples))
        self.total += float(samples.sum())
        self.total_square += float(samples @ samples)

    def log_predictive(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        kappa = self.kappa0 + self.count
        mean = (self.kappa0 * self.mean0 + self.total) / kappa
        alpha = self.alpha0 + 0.5 * self.count
        beta = self.beta0 + 0.5 * (
            self.total_square + self.kappa0 * self.mean0**2 - kappa * mean**2
        )
        scale = np.sqrt(max(float(beta), 1e-9) * (kappa + 1.0) / (alpha * kappa))
        return stats.t.logpdf(values, df=2.0 * alpha, loc=mean, scale=scale)


class BayesianIdleGate:
    """Diagonal Student-t classifier for idle versus evoked control.

    The gate uses spectral EEG features and three summaries of the *same*
    target-evidence vector consumed by the target selector. It therefore adds
    a binary decision, not another target classifier.
    """

    _bands = ((1.0, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 30.0))

    def __init__(self, config: IdleGateConfig):
        if config.sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if not 0 <= config.threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        if not 0 < config.forgetting <= 1:
            raise ValueError("forgetting must lie in (0, 1]")
        if config.evidence_weight <= 0:
            raise ValueError("evidence weight must be positive")
        self.config = config
        self._models: list[list[_NormalInverseGamma]] | None = None
        self._location: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None
        self._uses_target_scores: bool | None = None

    @staticmethod
    def _as_trials(eeg: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(eeg, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3 or not np.all(np.isfinite(values)):
            raise ValueError("EEG must be finite (trials, samples, channels)")
        return values

    def features(
        self,
        eeg: NDArray[np.float64],
        target_scores: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Extract EEG features and optional target-evidence summaries."""
        values = self._as_trials(eeg)
        frequencies, psd = signal.welch(
            values,
            fs=self.config.sample_rate,
            nperseg=min(256, values.shape[1]),
            axis=1,
        )
        powers = []
        for low, high in self._bands:
            selected = (frequencies >= low) & (frequencies < high)
            powers.append(np.log(np.maximum(psd[:, selected, :].mean(axis=1), 1e-12)))
        if target_scores is None:
            amplitude = np.log(np.maximum(values.std(axis=(1, 2)), 1e-12))
            line_length = np.log(
                np.maximum(np.abs(np.diff(values, axis=1)).mean(axis=(1, 2)), 1e-12)
            )
            repeatability = np.zeros(len(values), dtype=np.float64)
            if self.config.cycle_duration_s is not None:
                cycle_samples = round(self.config.cycle_duration_s * self.config.sample_rate)
                n_cycles = values.shape[1] // cycle_samples
                if n_cycles >= 2:
                    cycles = np.stack(
                        [
                            values[
                                :,
                                index * cycle_samples : (index + 1) * cycle_samples,
                            ]
                            for index in range(n_cycles)
                        ],
                        axis=1,
                    )
                    cycles -= cycles.mean(axis=2, keepdims=True)
                    first = cycles[:, :-1].reshape(len(values), n_cycles - 1, -1)
                    second = cycles[:, 1:].reshape(len(values), n_cycles - 1, -1)
                    numerator = np.sum(first * second, axis=2)
                    denominator = (
                        np.linalg.norm(first, axis=2) * np.linalg.norm(second, axis=2) + 1e-12
                    )
                    repeatability = np.mean(numerator / denominator, axis=1)
            evidence = np.column_stack([amplitude, line_length, repeatability])
        else:
            scores = np.asarray(target_scores, dtype=np.float64)
            if scores.ndim == 1:
                scores = scores[None, ...]
            if (
                scores.ndim != 2
                or len(scores) != len(values)
                or scores.shape[1] < 2
                or not np.all(np.isfinite(scores))
            ):
                raise ValueError("target scores must be finite (trials, targets)")
            centered = scores - scores.mean(axis=1, keepdims=True)
            normalized = centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
            ordered = np.sort(normalized, axis=1)
            evidence = np.column_stack(
                [
                    ordered[:, -1],
                    ordered[:, -1] - ordered[:, -2],
                    scores.std(axis=1),
                ]
            )
        return np.column_stack([*(item.reshape(len(values), -1) for item in powers), evidence])

    def fit(
        self,
        eeg: NDArray[np.float64],
        target_scores: NDArray[np.float64] | None,
        is_control: NDArray[np.bool_],
    ) -> BayesianIdleGate:
        """Fit posterior-predictive idle and control feature models."""
        features = self.features(eeg, target_scores)
        return self.fit_features(
            features,
            is_control,
            uses_target_scores=target_scores is not None,
        )

    def fit_features(
        self,
        features: NDArray[np.float64],
        is_control: NDArray[np.bool_],
        *,
        uses_target_scores: bool,
    ) -> BayesianIdleGate:
        """Fit from precomputed features for population-prior training."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("features must be a finite two-dimensional array")
        self._uses_target_scores = bool(uses_target_scores)
        labels = np.asarray(is_control, dtype=np.int64)
        if labels.shape != (len(features),) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("calibration requires both idle and control")
        self._location = np.median(features, axis=0)
        self._scale = np.maximum(np.std(features, axis=0), 1e-6)
        standardized = (features - self._location) / self._scale
        self._models = [[_NormalInverseGamma() for _ in range(features.shape[1])] for _ in range(2)]
        for class_id in (0, 1):
            rows = standardized[labels == class_id]
            for feature_id, model in enumerate(self._models[class_id]):
                model.update(rows[:, feature_id])
        return self

    def _standardize(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._location is None or self._scale is None:
            raise RuntimeError("idle gate has not been fitted")
        return (features - self._location) / self._scale

    def predict_proba(
        self,
        eeg: NDArray[np.float64],
        target_scores: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return columns ``[P(idle), P(control)]``."""
        if self._models is None:
            raise RuntimeError("idle gate has not been fitted")
        if (target_scores is not None) != self._uses_target_scores:
            raise ValueError("prediction must use the same EEG-only or score-assisted mode as fit")
        return self.predict_proba_features(self.features(eeg, target_scores))

    def predict_proba_features(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return probabilities from already extracted gate features."""
        features = self._standardize(np.asarray(features, dtype=np.float64))
        log_probability = np.empty((len(features), 2), dtype=np.float64)
        for class_id in (0, 1):
            evidence = np.column_stack(
                [
                    model.log_predictive(features[:, feature_id])
                    for feature_id, model in enumerate(self._models[class_id])
                ]
            ).sum(axis=1)
            log_probability[:, class_id] = np.log(0.5) + self.config.evidence_weight * evidence
        return special.softmax(log_probability, axis=1)

    def predict(
        self,
        eeg: NDArray[np.float64],
        target_scores: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """Return ``True`` for intentional-control windows."""
        return self.predict_proba(eeg, target_scores)[:, 1] >= self.config.threshold

    def feedback(
        self,
        eeg: NDArray[np.float64],
        target_scores: NDArray[np.float64] | None = None,
        *,
        is_control: bool,
    ) -> None:
        """Re-learn one confirmed idle/control example with forgetting."""
        if self._models is None:
            raise RuntimeError("idle gate has not been fitted")
        row = self._standardize(self.features(eeg, target_scores)).ravel()
        for value, model in zip(row, self._models[int(is_control)], strict=True):
            model.update(value, forgetting=self.config.forgetting)
