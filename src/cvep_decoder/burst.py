"""Subject-specific event-response decoding with Bayesian target evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import special


@dataclass(frozen=True, slots=True)
class BurstEvidenceConfig:
    """Configuration for the two-stage event-to-target decoder."""

    sample_rate: float
    presentation_rate: float = 60.0
    response_window_s: float = 0.25
    spatial_filters: int = 4
    covariance_estimator: str = "lwf"
    random_seed: int = 2024
    score: Literal["bayes", "correlation"] = "bayes"
    posterior_temperature: float = 1.0
    replay_trials: int = 120


@dataclass(frozen=True, slots=True)
class TargetEvidenceState:
    """One incremental Bayesian target update."""

    posterior: NDArray[np.float64]
    target: int
    confidence: float
    elapsed_s: float


class BayesianTargetAccumulator:
    """Constant-memory streaming target posterior."""

    def __init__(self, decoder: BurstEvidenceDecoder):
        if decoder.model is None or decoder.codebook is None or decoder.n_channels is None:
            raise RuntimeError("decoder has not been fitted")
        self.decoder = decoder
        _, self.codes = decoder._prepare_codebook(decoder.codebook)
        self.log_evidence = np.zeros(len(decoder.codebook), dtype=np.float64)
        self.frame_index = 0

    def update(self, response_epoch: NDArray[np.float64]) -> TargetEvidenceState:
        """Consume one event-aligned response epoch."""
        values = np.asarray(response_epoch, dtype=np.float64)
        expected_samples = round(
            self.decoder.config.response_window_s * self.decoder.config.sample_rate
        )
        if values.shape != (expected_samples, self.decoder.n_channels):
            raise ValueError(
                f"response epoch must have shape ({expected_samples}, {self.decoder.n_channels})"
            )
        probability = float(self.decoder.model.predict_proba(values.T[None, ...])[0, 1])
        probability = float(np.clip(probability, 1e-5, 1 - 1e-5))
        code = self.codes[:, self.frame_index % self.codes.shape[1]]
        self.log_evidence += code * np.log(probability)
        self.log_evidence += (1 - code) * np.log1p(-probability)
        posterior = special.softmax(self.log_evidence / self.decoder.config.posterior_temperature)
        target = int(np.argmax(posterior))
        elapsed = (
            self.decoder.config.response_window_s
            + self.frame_index / self.decoder.config.presentation_rate
        )
        self.frame_index += 1
        return TargetEvidenceState(
            posterior=posterior.copy(),
            target=target,
            confidence=float(posterior[target]),
            elapsed_s=elapsed,
        )

    def reset(self) -> None:
        """Start a new selection without reallocating the model."""
        self.log_evidence.fill(0)
        self.frame_index = 0


class BurstEvidenceDecoder:
    """Infer event responses, then accumulate evidence over arbitrary codes.

    The event-response classifier is fitted independently for each subject.
    It learns only two reusable EEG responses (event and no event), so
    calibration need not contain a separate template for every target.  At
    inference, the calibrated event probabilities are evaluated under every
    supplied code.  The default score is the Bernoulli log likelihood:

    ``sum(code * log(p_event) + (1-code) * log(1-p_event))``.

    Consequently, the model has one target variable and one evidence path; it
    is not a vote among unrelated target classifiers.
    """

    def __init__(self, config: BurstEvidenceConfig):
        if config.sample_rate <= 0 or config.presentation_rate <= 0:
            raise ValueError("sample and presentation rates must be positive")
        if config.response_window_s <= 0 or config.spatial_filters < 1:
            raise ValueError("response window and spatial-filter count must be positive")
        if config.posterior_temperature <= 0:
            raise ValueError("posterior temperature must be positive")
        if config.replay_trials < 1:
            raise ValueError("replay trial count must be positive")
        self.config = config
        self.model: Any | None = None
        self.codebook: NDArray[np.float64] | None = None
        self.frame_starts: NDArray[np.int64] | None = None
        self.frame_codes: NDArray[np.float64] | None = None
        self.n_channels: int | None = None
        self._calibration_eeg: NDArray[np.float64] | None = None
        self._calibration_labels: NDArray[np.int64] | None = None

    def _prepare_codebook(
        self, codebook: NDArray[np.float64], n_samples: int | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        traces = np.asarray(codebook, dtype=np.float64)
        if traces.ndim != 2 or traces.shape[1] < 2:
            raise ValueError("codebook must have target-by-sample shape")
        if n_samples is None:
            n_samples = traces.shape[1]
        duration_s = n_samples / self.config.sample_rate
        n_frames = int(np.floor(duration_s * self.config.presentation_rate + 1e-9))
        starts = np.rint(
            np.arange(n_frames) * self.config.sample_rate / self.config.presentation_rate
        ).astype(np.int64)
        sampled = traces[:, starts % traces.shape[1]]
        # Use the midpoint for true binary codes. A median threshold fails for
        # imbalanced binary sequences: when ones are the majority, median=1 and
        # every frame becomes dark. Median remains useful for continuous codes.
        unique_counts = np.asarray([len(np.unique(row)) for row in traces])
        binary_rows = unique_counts <= 2
        threshold = np.median(traces, axis=1, keepdims=True)
        if np.any(binary_rows):
            binary_values = np.asarray(
                [
                    (np.min(row) + np.max(row)) / 2.0
                    for row in traces[binary_rows]
                ],
                dtype=np.float64,
            )
            threshold[binary_rows, 0] = binary_values
        frame_states = (sampled > threshold).astype(np.float64)
        # The Dehais burst classifier predicts the onset of a bright burst,
        # not whether a frame is currently bright. Consecutive bright frames
        # therefore form one event at their initial dark-to-bright transition.
        previous = np.concatenate(
            (np.zeros((len(frame_states), 1)), frame_states[:, :-1]),
            axis=1,
        )
        event_codes = ((frame_states > 0.5) & (previous <= 0.5)).astype(np.float64)
        return starts, event_codes

    def fit(
        self,
        eeg: NDArray[np.float64],
        codebook: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> BurstEvidenceDecoder:
        """Fit a subject-specific event detector from any target subset."""
        try:
            from pyriemann.estimation import XdawnCovariances
            from pyriemann.tangentspace import TangentSpace
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:  # pragma: no cover - benchmark dependency
            raise RuntimeError("event-response decoding requires the benchmark extra") from exc

        values = np.asarray(eeg, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        traces = np.asarray(codebook, dtype=np.float64)
        if values.ndim != 3 or len(values) != len(targets) or len(values) == 0:
            raise ValueError("EEG must be non-empty trials-by-samples-by-channels")
        if not np.all(np.isfinite(values)):
            raise ValueError("EEG contains non-finite values")
        if np.any(targets < 0) or np.any(targets >= len(traces)):
            raise ValueError("target label outside the codebook")

        starts, frame_codes = self._prepare_codebook(traces, values.shape[1])
        window = round(self.config.response_window_s * self.config.sample_rate)
        starts = starts[starts + window <= values.shape[1]]
        if len(starts) < 2:
            raise ValueError("EEG is too short for an event-response window")
        frame_codes = frame_codes[:, : len(starts)]

        positive_pairs = [
            (trial_index, int(start))
            for trial_index, target in enumerate(targets)
            for start, is_event in zip(starts, frame_codes[target], strict=True)
            if is_event
        ]
        if not positive_pairs:
            raise ValueError("calibration must contain both event and no-event frames")
        rng = np.random.default_rng(self.config.random_seed)
        positive_by_trial = [
            {start for index, start in positive_pairs if index == trial_index}
            for trial_index in range(len(values))
        ]
        negative_pairs = [
            (trial_index, start)
            for trial_index in range(len(values))
            for start in range(values.shape[1] - window + 1)
            if start not in positive_by_trial[trial_index]
        ]
        if len(negative_pairs) < len(positive_pairs):
            raise ValueError("calibration does not contain enough no-event windows")
        selected_negative = rng.choice(len(negative_pairs), len(positive_pairs), replace=False)
        pairs = positive_pairs + [negative_pairs[index] for index in selected_negative]
        event_labels = np.concatenate(
            (
                np.ones(len(positive_pairs), dtype=int),
                np.zeros(len(positive_pairs), dtype=int),
            )
        )
        order = rng.permutation(len(pairs))
        epochs = np.stack(
            [
                values[pairs[index][0], pairs[index][1] : pairs[index][1] + window].T
                for index in order
            ]
        )
        event_labels = event_labels[order]

        self.model = make_pipeline(
            XdawnCovariances(
                nfilter=self.config.spatial_filters,
                classes=[1],
                estimator=self.config.covariance_estimator,
                xdawn_estimator=self.config.covariance_estimator,
            ),
            TangentSpace(metric="riemann"),
            LinearDiscriminantAnalysis(),
        )
        self.model.fit(epochs, event_labels)
        self.codebook = traces.copy()
        self.frame_starts, self.frame_codes = self._prepare_codebook(traces, values.shape[1])
        self.n_channels = values.shape[2]
        self._calibration_eeg = values[-self.config.replay_trials :].copy()
        self._calibration_labels = targets[-self.config.replay_trials :].copy()
        return self

    def event_probabilities(
        self, eeg: NDArray[np.float64], duration_s: float | None = None
    ) -> NDArray[np.float64]:
        """Return causal event probabilities at presentation-frame times."""
        if self.model is None or self.frame_starts is None or self.n_channels is None:
            raise RuntimeError("decoder has not been fitted")
        values = np.asarray(eeg, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[2] != self.n_channels:
            raise ValueError("EEG channel count differs from calibration")
        available = values.shape[1]
        if duration_s is not None:
            available = min(available, round(duration_s * self.config.sample_rate))
        window = round(self.config.response_window_s * self.config.sample_rate)
        starts = self.frame_starts[self.frame_starts + window <= available]
        if len(starts) < 2:
            raise ValueError("decision window contains fewer than two response epochs")
        epochs = np.concatenate(
            [np.stack([trial[start : start + window].T for start in starts]) for trial in values]
        )
        probability = self.model.predict_proba(epochs)[:, 1]
        return probability.reshape(len(values), len(starts))

    def decision_function(
        self, eeg: NDArray[np.float64], duration_s: float | None = None
    ) -> NDArray[np.float64]:
        """Return one accumulated score for every target."""
        if self.frame_codes is None:
            raise RuntimeError("decoder has not been fitted")
        probability = self.event_probabilities(eeg, duration_s)
        codes = self.frame_codes[:, : probability.shape[1]]
        if self.config.score == "bayes":
            p = np.clip(probability, 1e-5, 1 - 1e-5)
            return np.log(p) @ codes.T + np.log1p(-p) @ (1 - codes).T
        centered_probability = probability - probability.mean(axis=1, keepdims=True)
        centered_codes = codes - codes.mean(axis=1, keepdims=True)
        return (centered_probability @ centered_codes.T) / (
            np.maximum(np.linalg.norm(centered_probability, axis=1, keepdims=True), 1e-12)
            * np.maximum(np.linalg.norm(centered_codes, axis=1)[None, :], 1e-12)
        )

    def predict_proba(
        self, eeg: NDArray[np.float64], duration_s: float | None = None
    ) -> NDArray[np.float64]:
        """Return normalized target probabilities."""
        scores = self.decision_function(eeg, duration_s)
        return special.softmax(scores / self.config.posterior_temperature, axis=1)

    def predict(
        self, eeg: NDArray[np.float64], duration_s: float | None = None
    ) -> NDArray[np.int64]:
        """Return maximum-posterior targets."""
        return self.decision_function(eeg, duration_s).argmax(axis=1)

    def start_stream(self) -> BayesianTargetAccumulator:
        """Create an independent constant-memory evidence accumulator."""
        return BayesianTargetAccumulator(self)

    def partial_fit(
        self,
        eeg: NDArray[np.float64],
        confirmed_labels: NDArray[np.int64],
    ) -> BurstEvidenceDecoder:
        """Re-fit from a bounded replay buffer using confirmed labels only."""
        if (
            self.codebook is None
            or self._calibration_eeg is None
            or self._calibration_labels is None
        ):
            raise RuntimeError("decoder has not been fitted")
        values = np.asarray(eeg, dtype=np.float64)
        targets = np.asarray(confirmed_labels, dtype=np.int64)
        if values.ndim == 2:
            values = values[None]
        if (
            values.ndim != 3
            or len(values) != len(targets)
            or values.shape[2] != self.n_channels
            or np.any(targets < 0)
            or np.any(targets >= len(self.codebook))
        ):
            raise ValueError("confirmed EEG or labels differ from calibration")
        common_samples = min(self._calibration_eeg.shape[1], values.shape[1])
        replay_eeg = np.concatenate(
            (
                self._calibration_eeg[:, :common_samples],
                values[:, :common_samples],
            ),
            axis=0,
        )
        replay_labels = np.concatenate((self._calibration_labels, targets), axis=0)
        replay_eeg = replay_eeg[-self.config.replay_trials :]
        replay_labels = replay_labels[-self.config.replay_trials :]
        return self.fit(replay_eeg, self.codebook, replay_labels)
