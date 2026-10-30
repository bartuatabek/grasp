"""Calibration-free filter-bank CCA for narrow-band c-VEP codes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal


@dataclass(frozen=True, slots=True)
class NarrowBandFBCCAConfig:
    """Configuration matching the NBRS reference-template construction."""

    sample_rate: float
    bands: tuple[tuple[float, float], ...]
    lag_offsets_s: tuple[float, ...]
    weight_exponent: float
    weight_offset: float
    regularization: float = 1e-6

    @classmethod
    def generic(cls, sample_rate: float) -> NarrowBandFBCCAConfig:
        """Sampling-rate-normalized filter bank used for every code family."""
        return cls(
            sample_rate=sample_rate,
            bands=(
                (0.04 * sample_rate, 0.32 * sample_rate),
                (0.10 * sample_rate, 0.32 * sample_rate),
                (0.16 * sample_rate, 0.32 * sample_rate),
            ),
            lag_offsets_s=(0.0, 2 / 120),
            weight_exponent=3.0,
            weight_offset=0.5,
        )

    @classmethod
    def nbrs15(cls, sample_rate: float = 250) -> NarrowBandFBCCAConfig:
        """Return the published NBRS-15 filter-bank settings."""
        return cls(
            sample_rate=sample_rate,
            bands=((10, 80), (25, 80), (40, 80)),
            lag_offsets_s=(0.0, 2 / 120),
            weight_exponent=3.0,
            weight_offset=0.5,
        )

    @classmethod
    def nbrs8(cls, sample_rate: float = 250) -> NarrowBandFBCCAConfig:
        """Return the published NBRS-8 filter-bank settings."""
        return cls(
            sample_rate=sample_rate,
            bands=((6, 50), (14, 50), (22, 50)),
            lag_offsets_s=(0.0, 1 / 120, 2 / 120, 3 / 120),
            weight_exponent=2.5,
            weight_offset=0.25,
        )


class NarrowBandFBCCA:
    """Score arbitrary codebooks without subject- or target-specific fitting.

    Each target reference contains latency-shifted versions of its code and all
    unique second-order Hadamard products. CCA evidence is computed in each
    causal pass-band and combined with the published filter-bank weighting
    form. The decoder is target-count agnostic; the retained benchmark uses
    only the 40 biologically presented NBRS codes.
    """

    def __init__(self, config: NarrowBandFBCCAConfig):
        if config.sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if not config.bands or not config.lag_offsets_s:
            raise ValueError("bands and lag offsets cannot be empty")
        if not 0 <= config.regularization < 1:
            raise ValueError("regularization must lie in [0, 1)")
        self.config = config
        self.codebook: NDArray[np.float64] | None = None
        self.n_channels: int | None = None
        self._filters = tuple(
            signal.cheby1(
                4,
                1,
                band,
                btype="bandpass",
                fs=config.sample_rate,
                output="sos",
            )
            for band in config.bands
        )

    def fit(
        self,
        codebook: NDArray[np.float64],
        *,
        n_channels: int | None = None,
    ) -> NarrowBandFBCCA:
        """Register a codebook; no EEG calibration data are consumed."""
        traces = np.asarray(codebook, dtype=np.float64)
        if (
            traces.ndim != 2
            or not traces.shape[0]
            or not traces.shape[1]
            or not np.all(np.isfinite(traces))
        ):
            raise ValueError("codebook must be a finite target-by-sample array")
        if n_channels is not None and n_channels < 1:
            raise ValueError("channel count must be positive")
        self.codebook = traces
        self.n_channels = n_channels
        return self

    @staticmethod
    def _tile(codebook: NDArray[np.float64], n_samples: int) -> NDArray[np.float64]:
        repeats = int(np.ceil(n_samples / codebook.shape[1]))
        return np.tile(codebook, (1, repeats))[:, :n_samples]

    def _reference(self, n_samples: int) -> NDArray[np.float64]:
        if self.codebook is None:
            raise RuntimeError("decoder has not been fitted")
        traces = self._tile(self.codebook, n_samples)
        lagged = []
        for offset_s in self.config.lag_offsets_s:
            delay = round(offset_s * self.config.sample_rate)
            shifted = np.zeros_like(traces)
            if delay == 0:
                shifted[:] = traces
            else:
                shifted[:, delay:] = traces[:, :-delay]
            lagged.append(shifted)
        basic = np.stack(lagged, axis=2)
        second_order = [
            basic[:, :, first] * basic[:, :, second]
            for first in range(basic.shape[2])
            for second in range(first, basic.shape[2])
        ]
        reference = np.concatenate((basic, np.stack(second_order, axis=2)), axis=2)
        reference -= reference.mean(axis=1, keepdims=True)
        reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
        return reference

    def _cca_scores(
        self,
        eeg: NDArray[np.float64],
        reference: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the largest regularized canonical correlation in batches."""
        centered = eeg - eeg.mean(axis=1, keepdims=True)
        n_trials, n_samples, n_channels = centered.shape
        n_targets, _, n_features = reference.shape
        covariance_x = np.einsum("ntc,ntd->ncd", centered, centered)
        scale_x = np.trace(covariance_x, axis1=1, axis2=2) / n_channels
        covariance_x += (
            self.config.regularization * scale_x[:, None, None] * np.eye(n_channels)[None]
        )
        eigenvalue_x, eigenvector_x = np.linalg.eigh(covariance_x)
        whiten_x = np.einsum(
            "nci,ni,nji->ncj",
            eigenvector_x,
            1 / np.sqrt(np.maximum(eigenvalue_x, 1e-12)),
            eigenvector_x,
        )
        covariance_r = np.einsum("ktf,ktg->kfg", reference, reference)
        scale_r = np.trace(covariance_r, axis1=1, axis2=2) / n_features
        covariance_r += (
            self.config.regularization * scale_r[:, None, None] * np.eye(n_features)[None]
        )
        eigenvalue_r, eigenvector_r = np.linalg.eigh(covariance_r)
        whiten_r = np.einsum(
            "kfi,ki,kji->kfj",
            eigenvector_r,
            1 / np.sqrt(np.maximum(eigenvalue_r, 1e-12)),
            eigenvector_r,
        )
        # optimize=True picks a BLAS-backed contraction order for these two
        # products. It is the dominant cost of a GRASP fit and is ~20x faster
        # than the naive order; results agree to ~1e-15 relative.
        cross = np.einsum("ntc,ktf->nkcf", centered, reference, optimize=True)
        canonical = np.einsum(
            "nac,nkcf,kfb->nkab", whiten_x, cross, whiten_r, optimize=True
        )
        singular_values = np.linalg.svd(canonical, compute_uv=False)
        return singular_values[:, :, 0] / max(n_samples - 1, 1)

    def decision_function(self, eeg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibration-free scores for every target in the codebook."""
        values = np.asarray(eeg, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or not np.all(np.isfinite(values)):
            raise ValueError("EEG must be a finite trial-by-sample-by-channel array")
        if self.n_channels is not None and values.shape[2] != self.n_channels:
            raise ValueError("EEG channel count differs from the configured count")
        filtered_bands = [signal.sosfilt(sos, values, axis=1) for sos in self._filters]
        return self.decision_function_bands(filtered_bands)

    def decision_function_bands(self, eeg_bands: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Score causally prefiltered bands with preserved filter history."""
        if len(eeg_bands) != len(self._filters):
            raise ValueError("prefiltered band count differs from configuration")
        values = [np.asarray(band, dtype=np.float64) for band in eeg_bands]
        if any(band.ndim != 3 for band in values):
            raise ValueError("each band must have trial-by-sample-by-channel shape")
        if self.n_channels is not None and any(band.shape[2] != self.n_channels for band in values):
            raise ValueError("EEG channel count differs from the configured count")
        n_samples = values[0].shape[1]
        if any(band.shape[:2] != values[0].shape[:2] for band in values):
            raise ValueError("prefiltered bands must share trials and samples")
        reference = self._reference(n_samples)
        scores = np.zeros((len(values[0]), len(reference)), dtype=np.float64)
        for index, filtered in enumerate(values, start=1):
            correlation = self._cca_scores(filtered, reference)
            weight = index ** (-self.config.weight_exponent)
            weight += self.config.weight_offset
            scores += weight * correlation**2
        return scores

    def predict(self, eeg: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return the maximum-evidence target index."""
        return self.decision_function(eeg).argmax(axis=1)
