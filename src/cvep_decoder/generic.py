"""Generic c-VEP decoding learned from an invariant representation bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, signal, special

from .burst import BurstEvidenceConfig, BurstEvidenceDecoder
from .fbcca import NarrowBandFBCCA, NarrowBandFBCCAConfig
from .idle import BayesianIdleGate, IdleGateConfig


@dataclass(frozen=True, slots=True)
class GenericDecoderState:
    """One sequential decision from the generic decoder."""

    target: int | None
    elapsed_s: float
    estimated_accuracy: float
    control_probability: float
    committed: bool
    reason: Literal["collecting", "calibrated_stop", "final", "idle"]
    scores: NDArray[np.float64]


class _ConstantCalibrator:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        probability = np.full(len(features), self.probability)
        return np.column_stack((1 - probability, probability))


class GenericCvepDecoder:
    """Learn a participant-specific decoder without a paradigm setting.

    ``fit`` accepts exactly EEG, sampling rate, codebook, and labels. Every
    participant receives the same representation bank:

    - state,
    - contrast,
    - signed difference,
    - on transition,
    - off transition,
    - a long temporal-lag response,
    - normalized spectral components, and
    - transition-event likelihood.

    Response models are first evaluated out of fold on inferred chronological
    calibration blocks. A second block-held-out stacking layer learns fusion
    weights, and its doubly out-of-fold scores calibrate one sequential
    correctness threshold. Dataset identity is never provided or inferred.
    """

    REPRESENTATIONS = (
        "state",
        "contrast",
        "signed_difference",
        "on_transition",
        "off_transition",
        "temporal_lags",
        "spectral_components",
        "transition_likelihood",
    )
    _RCCA_SPECS = (
        ("state", "stimulus", 0.30),
        ("contrast", "contrast", 0.20),
        ("signed_difference", "difference", 0.10),
        ("on_transition", "rise", 0.25),
        ("off_transition", "fall", 0.25),
        ("temporal_lags", "stimulus", 0.45),
    )
    _RCCA_REGULARIZATION = {
        "state": 1e-3,
        "contrast": 1e-5,
        "signed_difference": 1e-2,
        "on_transition": 1e-3,
        "off_transition": 1e-3,
        "temporal_lags": 1e-3,
    }
    _IDLE_ANALYSIS_S = 0.75

    def __init__(self):
        self.sample_rate: float | None = None
        self.codebook: NDArray[np.float64] | None = None
        self.n_channels: int | None = None
        self.models: dict[str, Any] = {}
        self.weights = np.full(
            len(self.REPRESENTATIONS), 1 / len(self.REPRESENTATIONS)
        )
        self.decision_times_s: tuple[float, ...] = ()
        self.stop_threshold = 1.0
        self.stop_calibrator: Any | None = None
        self.idle_gate: BayesianIdleGate | None = None
        self._winner: int | None = None
        self._winner_count = 0
        self._codebook_changed = False
        self._preprocess_sos: NDArray[np.float64] | None = None
        self._response_sos: NDArray[np.float64] | None = None
        self._feedback_count = 0
        self.oof_accuracy: float | None = None

    @staticmethod
    def _trials(eeg: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(eeg, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or not np.all(np.isfinite(values)):
            raise ValueError("EEG must be finite (trials, samples, channels)")
        return values

    @staticmethod
    def _normalize(scores: NDArray[np.float64]) -> NDArray[np.float64]:
        centered = scores - scores.mean(axis=1, keepdims=True)
        return centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-8)

    @staticmethod
    def _transform_correlation(scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the fixed Fisher transform before cross-candidate scaling."""
        return np.arctanh(np.clip(scores, -0.999, 0.999))

    @staticmethod
    def _infer_blocks(labels: NDArray[np.int64]) -> NDArray[np.int64]:
        """Infer chronological calibration blocks without extra metadata."""
        labels = np.asarray(labels, dtype=np.int64)
        counts = np.bincount(labels)
        present = counts[counts > 0]
        if len(present) and present.min() >= 3:
            occurrence: dict[int, int] = {}
            groups = np.empty(len(labels), dtype=np.int64)
            for index, label in enumerate(labels):
                groups[index] = occurrence.get(int(label), 0)
                occurrence[int(label)] = groups[index] + 1
            if len(np.unique(groups)) >= 3:
                return groups
        n_groups = min(5, max(3, len(labels) // 2))
        return np.minimum(
            np.arange(len(labels)) * n_groups // len(labels),
            n_groups - 1,
        )

    @staticmethod
    def _decision_grid(n_samples: int, sample_rate: float) -> tuple[float, ...]:
        fractions = np.asarray((0.25, 0.40, 0.55, 0.70, 0.85, 1.0))
        samples = np.rint(fractions * n_samples).astype(int)
        samples = np.maximum(samples, round(0.50 * sample_rate))
        samples = np.unique(np.minimum(samples, n_samples))
        return tuple(float(sample / sample_rate) for sample in samples)

    @staticmethod
    def _shared_passband(sample_rate: float) -> tuple[float, float]:
        """Return the same broad normalized passband for every input codebook."""
        return 1.0, 0.45 * sample_rate

    @staticmethod
    def _response_passband(
        codebook: NDArray[np.float64],
        sample_rate: float,
    ) -> tuple[float, float]:
        """Apply one code-spectrum formula to every spatiotemporal response."""
        centered = codebook - codebook.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
        power = power.mean(axis=0)
        frequencies = np.fft.rfftfreq(codebook.shape[1], 1 / sample_rate)
        cumulative = np.cumsum(power)
        cumulative /= max(cumulative[-1], 1e-12)
        low_energy = float(frequencies[np.searchsorted(cumulative, 0.05)])
        high_energy = float(frequencies[np.searchsorted(cumulative, 0.95)])
        if high_energy - low_energy < 0.12 * sample_rate:
            low = max(1.0, low_energy - 5.0)
            high = min(0.45 * sample_rate, high_energy + 0.22 * sample_rate)
            return low, high
        return GenericCvepDecoder._shared_passband(sample_rate)

    def _preprocess(self, eeg: NDArray[np.float64]) -> NDArray[np.float64]:
        values = self._trials(eeg)
        if self._preprocess_sos is None:
            raise RuntimeError("decoder preprocessing has not been fitted")
        return self._filter_trials(self._preprocess_sos, values)

    @staticmethod
    def _filter_trials(
        sos: NDArray[np.float64],
        eeg: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Causally filter independent trials with a stable initial state."""
        initial = signal.sosfilt_zi(sos)
        zi = initial[:, None, :, None] * eeg[:, 0, :][None, :, None, :]
        return signal.sosfilt(sos, eeg, axis=1, zi=zi)[0]

    def _response_preprocess(
        self,
        eeg: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._response_sos is None:
            raise RuntimeError("response preprocessing has not been fitted")
        return self._filter_trials(self._response_sos, eeg)

    def _idle_preprocess(
        self,
        eeg: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Causally filter a rolling window and retain its settled tail."""
        values = self._trials(eeg)
        if self._response_sos is None:
            raise RuntimeError("response preprocessing has not been fitted")
        filtered = signal.sosfilt(self._response_sos, values, axis=1)
        analysis_samples = round(self._IDLE_ANALYSIS_S * float(self.sample_rate))
        return filtered[:, -analysis_samples:]

    @staticmethod
    def _new_rcca(
        codebook: NDArray[np.float64],
        sample_rate: float,
        event: str,
        response_length_s: float,
        regularization: float,
        *,
        running: bool,
    ) -> Any:
        try:
            from pyntbci.classifiers import rCCA
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("generic decoding requires pyntbci") from exc
        return rCCA(
            stimulus=codebook,
            fs=int(round(sample_rate)),
            event=event,
            onset_event=False,
            encoding_length=response_length_s,
            gamma_x=regularization,
            gamma_m=regularization,
            score_metric="correlation",
            running=running,
        )

    @classmethod
    def _fit_models(
        cls,
        eeg: NDArray[np.float64],
        sample_rate: float,
        codebook: NDArray[np.float64],
        labels: NDArray[np.int64],
        *,
        running: bool,
    ) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for name, event, response_length in cls._RCCA_SPECS:
            model = cls._new_rcca(
                codebook,
                sample_rate,
                event,
                response_length,
                cls._RCCA_REGULARIZATION[name],
                running=running,
            )
            model.fit(eeg.transpose(0, 2, 1), labels)
            models[name] = model
        models["spectral_components"] = NarrowBandFBCCA(
            NarrowBandFBCCAConfig.generic(sample_rate)
        ).fit(codebook, n_channels=eeg.shape[2])
        models["transition_likelihood"] = BurstEvidenceDecoder(
            BurstEvidenceConfig(
                sample_rate=sample_rate,
                presentation_rate=60.0,
                response_window_s=0.25,
                spatial_filters=4,
                score="bayes",
            )
        ).fit(eeg, codebook, labels)
        return models

    @classmethod
    def _score_models(
        cls,
        models: dict[str, Any],
        response_eeg: NDArray[np.float64],
        spectral_eeg: NDArray[np.float64],
        duration_s: float,
        sample_rate: float,
        *,
        active: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        stop = min(response_eeg.shape[1], round(duration_s * sample_rate))
        response_values = response_eeg[:, :stop]
        spectral_values = spectral_eeg[:, :stop]
        scores = []
        for index, name in enumerate(cls.REPRESENTATIONS):
            if active is not None and not active[index]:
                scores.append(None)
                continue
            if name == "spectral_components":
                score = models[name].decision_function(spectral_values)
            elif name == "transition_likelihood":
                score = models[name].decision_function(response_values, duration_s)
            else:
                score = np.asarray(
                    models[name].decision_function(
                        response_values.transpose(0, 2, 1)
                    ),
                    dtype=np.float64,
                )
                if score.ndim == 3:
                    score = score.mean(axis=-1)
                score = cls._transform_correlation(score)
            scores.append(cls._normalize(score))
        if active is None:
            return np.stack(scores, axis=1)
        template = next(score for score in scores if score is not None)
        return np.stack(
            [
                np.zeros_like(template) if score is None else score
                for score in scores
            ],
            axis=1,
        )

    @staticmethod
    def _fit_weights(
        view_scores: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        n_views = view_scores.shape[1]
        rows = np.arange(len(labels))

        def loss(weights: NDArray[np.float64]) -> float:
            fused = np.einsum("nvt,v->nt", view_scores, weights)
            likelihood = special.logsumexp(fused, axis=1) - fused[rows, labels]
            return float(likelihood.mean() + 1e-3 * np.sum(weights**2))

        result = optimize.minimize(
            loss,
            np.full(n_views, 1 / n_views),
            method="SLSQP",
            # A single noisy calibration fold must not collapse the ensemble
            # onto one view. The universal cap is part of the stacking model,
            # not a paradigm-specific weight.
            bounds=[(0.0, 0.55)] * n_views,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return np.full(n_views, 1 / n_views)
        weights = np.maximum(result.x, 0)
        return weights / weights.sum()

    @staticmethod
    def _fuse(
        view_scores: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.einsum("...vt,v->...t", view_scores, weights)

    def _nested_oof(
        self,
        response_eeg: NDArray[np.float64],
        spectral_eeg: NDArray[np.float64],
        codebook: NDArray[np.float64],
        labels: NDArray[np.int64],
        groups: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_times = len(self.decision_times_s)
        base = np.empty(
            (
                len(response_eeg),
                n_times,
                len(self.REPRESENTATIONS),
                len(codebook),
            ),
            dtype=np.float64,
        )
        for group in np.unique(groups):
            train = groups != group
            validation = groups == group
            models = self._fit_models(
                response_eeg[train],
                float(self.sample_rate),
                codebook,
                labels[train],
                running=False,
            )
            for time_index, duration in enumerate(self.decision_times_s):
                base[validation, time_index] = self._score_models(
                    models,
                    response_eeg[validation],
                    spectral_eeg[validation],
                    duration,
                    float(self.sample_rate),
                )

        meta = np.empty(
            (len(response_eeg), n_times, len(codebook)),
            dtype=np.float64,
        )
        for group in np.unique(groups):
            train = groups != group
            validation = groups == group
            fold_weights = self._fit_weights(base[train, -1], labels[train])
            meta[validation] = self._fuse(base[validation], fold_weights)
        final_weights = self._fit_weights(base[:, -1], labels)
        return meta, final_weights

    @staticmethod
    def _stopping_features(
        scores: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        order = np.argsort(scores, axis=-1)
        winner = order[..., -1]
        margin = np.take_along_axis(scores, order[..., -1:], axis=-1)[..., 0]
        margin -= np.take_along_axis(scores, order[..., -2:-1], axis=-1)[..., 0]
        normalized_time = times / times[-1]
        time_feature = np.broadcast_to(normalized_time, margin.shape)
        features = np.column_stack(
            (
                margin.ravel(),
                time_feature.ravel(),
                (margin * np.sqrt(time_feature)).ravel(),
            )
        )
        return features, winner

    def _fit_stopping(
        self,
        oof_scores: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("generic stopping calibration requires scikit-learn") from exc
        times = np.asarray(self.decision_times_s)
        features, winner = self._stopping_features(oof_scores, times)
        correct = (winner == labels[:, None]).ravel().astype(int)
        if len(np.unique(correct)) == 1:
            self.stop_calibrator = _ConstantCalibrator(float(correct[0]))
        else:
            self.stop_calibrator = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=2024,
            ).fit(features, correct)
        probability = self.stop_calibrator.predict_proba(features)[:, 1]
        probability = probability.reshape(len(labels), len(times))
        final_accuracy = float(np.mean(winner[:, -1] == labels))
        candidates = np.unique(
            np.concatenate(
                (
                    np.asarray((0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975)),
                    probability.ravel(),
                )
            )
        )
        candidates = candidates[candidates >= 0.80]
        candidates = np.append(candidates, 1.01)
        best: tuple[float, float, float] | None = None
        best_threshold = 1.0
        for threshold in candidates:
            prediction, duration = self._stop_oof(
                winner,
                probability,
                times,
                float(threshold),
            )
            accuracy = float(np.mean(prediction == labels))
            if accuracy + 1e-12 < final_accuracy - 0.01:
                continue
            mean_duration = float(duration.mean())
            objective = accuracy / np.sqrt(mean_duration)
            candidate = (objective, accuracy, -mean_duration)
            if best is None or candidate > best:
                best = candidate
                best_threshold = float(threshold)
        self.stop_threshold = best_threshold
        self.oof_accuracy = final_accuracy

    @staticmethod
    def _stop_oof(
        winners: NDArray[np.int64],
        probability: NDArray[np.float64],
        times: NDArray[np.float64],
        threshold: float,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        prediction = winners[:, -1].copy()
        duration = np.full(len(winners), times[-1], dtype=np.float64)
        for trial in range(len(winners)):
            for index in range(1, len(times)):
                if (
                    winners[trial, index] == winners[trial, index - 1]
                    and probability[trial, index] >= threshold
                ):
                    prediction[trial] = winners[trial, index]
                    duration[trial] = times[index]
                    break
        return prediction, duration

    def fit(
        self,
        eeg: NDArray[np.float64],
        sample_rate: float,
        codebook: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> GenericCvepDecoder:
        """Fit from the only four required inputs."""
        values = self._trials(eeg)
        traces = np.asarray(codebook, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if values.shape[2] < 1:
            raise ValueError("EEG must contain at least one channel")
        if traces.ndim != 2 or len(traces) < 2 or not np.all(np.isfinite(traces)):
            raise ValueError("codebook must contain at least two finite target traces")
        if targets.shape != (len(values),) or np.any(targets < 0) or np.any(targets >= len(traces)):
            raise ValueError("labels must identify one codebook row per EEG trial")
        if len(values) < 6:
            raise ValueError("nested calibration requires at least six trials")
        self.sample_rate = float(sample_rate)
        self.codebook = traces.copy()
        self.n_channels = values.shape[2]
        passband = self._shared_passband(sample_rate)
        self._preprocess_sos = signal.butter(
            4,
            passband,
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        response_passband = self._response_passband(traces, sample_rate)
        self._response_sos = signal.cheby1(
            4,
            1,
            response_passband,
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        spectral_values = self._preprocess(values)
        response_values = self._response_preprocess(values)
        self.decision_times_s = self._decision_grid(
            response_values.shape[1],
            sample_rate,
        )
        groups = self._infer_blocks(targets)
        oof_scores, self.weights = self._nested_oof(
            response_values,
            spectral_values,
            traces,
            targets,
            groups,
        )
        self._fit_stopping(oof_scores, targets)
        self.models = self._fit_models(
            response_values,
            sample_rate,
            traces,
            targets,
            running=True,
        )
        self.idle_gate = None
        self._codebook_changed = False
        self._feedback_count = 0
        self.reset()
        return self

    def view_scores(
        self,
        eeg: NDArray[np.float64],
        duration_s: float | None = None,
    ) -> NDArray[np.float64]:
        """Return normalized evidence for all invariant representations."""
        if self.sample_rate is None or not self.models:
            raise RuntimeError("decoder has not been fitted")
        raw_values = self._trials(eeg)
        spectral_values = self._preprocess(raw_values)
        response_values = self._response_preprocess(raw_values)
        duration = (
            response_values.shape[1] / self.sample_rate
            if duration_s is None
            else duration_s
        )
        return self._score_models(
            self.models,
            response_values,
            spectral_values,
            duration,
            self.sample_rate,
        )

    def decision_function(
        self,
        eeg: NDArray[np.float64],
        duration_s: float | None = None,
    ) -> NDArray[np.float64]:
        """Return participant-weighted evidence for every active code."""
        if self.sample_rate is None or not self.models:
            raise RuntimeError("decoder has not been fitted")
        raw_values = self._trials(eeg)
        spectral_values = self._preprocess(raw_values)
        response_values = self._response_preprocess(raw_values)
        duration = (
            response_values.shape[1] / self.sample_rate
            if duration_s is None
            else duration_s
        )
        views = self._score_models(
            self.models,
            response_values,
            spectral_values,
            duration,
            self.sample_rate,
            active=self.weights > 1e-6,
        )
        return self._fuse(views, self.weights)

    def fit_idle(
        self,
        idle_eeg: NDArray[np.float64],
        control_eeg: NDArray[np.float64],
    ) -> GenericCvepDecoder:
        """Fit the optional idle gate without changing the target decoder."""
        if self.sample_rate is None or self.n_channels is None:
            raise RuntimeError("decoder has not been fitted")
        idle = self._idle_preprocess(idle_eeg)
        control = self._idle_preprocess(control_eeg)
        if idle.shape[1:] != control.shape[1:] or idle.shape[2] != self.n_channels:
            raise ValueError("idle and control windows must have equal channel shapes")
        combined = np.concatenate((idle, control))
        labels = np.concatenate(
            (np.zeros(len(idle), dtype=bool), np.ones(len(control), dtype=bool))
        )
        self.idle_gate = BayesianIdleGate(
            IdleGateConfig(sample_rate=self.sample_rate)
        ).fit(combined, None, labels)
        return self

    def idle_probability(
        self,
        eeg: NDArray[np.float64],
        duration_s: float | None = None,
    ) -> NDArray[np.float64]:
        """Return the fitted gate's probability that each window is control."""
        if self.sample_rate is None or self.idle_gate is None:
            raise RuntimeError("idle gate has not been fitted")
        values = self._idle_preprocess(eeg)
        duration = values.shape[1] / self.sample_rate if duration_s is None else duration_s
        stop = min(values.shape[1], round(duration * self.sample_rate))
        return self.idle_gate.predict_proba(values[:, :stop], None)[:, 1]

    def _estimated_accuracy(
        self,
        scores: NDArray[np.float64],
        elapsed_s: float,
    ) -> float:
        if self.stop_calibrator is None or not self.decision_times_s:
            raise RuntimeError("stopping rule has not been calibrated")
        order = np.argsort(scores)
        margin = float(scores[order[-1]] - scores[order[-2]])
        normalized_time = min(elapsed_s / self.decision_times_s[-1], 3.0)
        features = np.asarray(
            [[margin, normalized_time, margin * np.sqrt(normalized_time)]]
        )
        return float(self.stop_calibrator.predict_proba(features)[0, 1])

    def reset(self) -> None:
        """Start a new decision."""
        self._winner = None
        self._winner_count = 0

    def update(
        self,
        eeg: NDArray[np.float64],
        *,
        elapsed_s: float | None = None,
        final: bool = False,
    ) -> GenericDecoderState:
        """Select, continue observing, or force a choice when ``final``."""
        if self.sample_rate is None:
            raise RuntimeError("decoder has not been fitted")
        values = self._trials(eeg)
        if len(values) != 1:
            raise ValueError("sequential update expects one trial")
        elapsed = values.shape[1] / self.sample_rate if elapsed_s is None else elapsed_s
        scores = self.decision_function(values, elapsed)[0]
        control_probability = 1.0
        if self.idle_gate is not None:
            control_probability = float(self.idle_probability(values, elapsed)[0])
            if control_probability < self.idle_gate.config.threshold:
                self.reset()
                return GenericDecoderState(
                    None,
                    elapsed,
                    0.0,
                    control_probability,
                    False,
                    "idle",
                    scores,
                )
        target = int(np.argmax(scores))
        self._winner_count = self._winner_count + 1 if target == self._winner else 1
        self._winner = target
        estimated_accuracy = self._estimated_accuracy(scores, elapsed)
        if self._winner_count >= 2 and estimated_accuracy >= self.stop_threshold:
            return GenericDecoderState(
                target,
                elapsed,
                estimated_accuracy,
                control_probability,
                True,
                "calibrated_stop",
                scores,
            )
        if final:
            return GenericDecoderState(
                target,
                elapsed,
                estimated_accuracy,
                control_probability,
                True,
                "final",
                scores,
            )
        return GenericDecoderState(
            None,
            elapsed,
            estimated_accuracy,
            control_probability,
            False,
            "collecting",
            scores,
        )

    def set_codebook(self, codebook: NDArray[np.float64]) -> GenericCvepDecoder:
        """Change target identity/count while preserving fitted response filters."""
        if self.sample_rate is None or self.n_channels is None or not self.models:
            raise RuntimeError("decoder has not been fitted")
        traces = np.asarray(codebook, dtype=np.float64)
        if traces.ndim != 2 or len(traces) < 2 or not np.all(np.isfinite(traces)):
            raise ValueError("codebook must contain at least two finite target traces")
        for name, _, _ in self._RCCA_SPECS:
            self.models[name].set_stimulus(traces)
        self.models["spectral_components"].fit(traces, n_channels=self.n_channels)
        transition = self.models["transition_likelihood"]
        transition.codebook = traces.copy()
        transition.frame_starts, transition.frame_codes = transition._prepare_codebook(
            traces,
            max(
                traces.shape[1],
                round(self.decision_times_s[-1] * self.sample_rate),
            ),
        )
        self.codebook = traces.copy()
        self._codebook_changed = True
        self.reset()
        return self

    def feedback(
        self,
        eeg: NDArray[np.float64],
        predicted_target: int,
        *,
        corrected_target: int | None = None,
        was_control: bool = True,
    ) -> GenericCvepDecoder:
        """Update the same response family from explicit post-selection feedback."""
        if self.codebook is None or self.sample_rate is None or not self.models:
            raise RuntimeError("decoder has not been fitted")
        raw_values = self._trials(eeg)
        if self.idle_gate is not None:
            idle_values = self._idle_preprocess(raw_values)
            self.idle_gate.feedback(idle_values, None, is_control=was_control)
        if not was_control:
            return self
        values = self._response_preprocess(raw_values)
        label = predicted_target if corrected_target is None else corrected_target
        if not 0 <= label < len(self.codebook):
            raise ValueError("feedback target is outside the active codebook")
        target = np.asarray([label], dtype=np.int64)
        for name, _, _ in self._RCCA_SPECS:
            self.models[name].fit(values.transpose(0, 2, 1), target)
        self._feedback_count += 1
        if not self._codebook_changed and self._feedback_count % 8 == 0:
            self.models["transition_likelihood"].partial_fit(values, target)
        return self
