"""Cached fixtures for the GRASP demo notebook.

A GRASP fit costs seconds, so the demo does not fit anything while you are
driving it. For each dataset this module fits one decoder once, evaluates its
eight alphabets on every held-out trial across a grid of observation windows,
and stores the resulting tensor. Everything the notebook shows is then a slice
or a weighted sum of that tensor, which keeps the controls instant and means
the notebook itself never imports the decoder.

Delete `demo/cache/*.npz` to force a rebuild.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULT_CACHE = DATA / "cache"
DEMO_CACHE = Path(__file__).resolve().parent / "cache"

DATASETS = ("Ricker Texture", "Narrow-band Random", "Modulated Gold", "Binary Sequence Set")

# Names used by the analysis notebook, the manuscript and the cached result CSVs.
INTERNAL_NAME = {
    "Ricker Texture": "Dehais 2024",
    "Narrow-band Random": "NBRS-15",
    "Modulated Gold": "Thielen 2021",
    "Binary Sequence Set": "Thielen 2025",
}
CITATION = {
    "Ricker Texture": "Dehais et al. 2024",
    "Narrow-band Random": "Zheng et al. 2024",
    "Modulated Gold": "Thielen et al. 2021",
    "Binary Sequence Set": "Thielen 2025",
}
BASELINE_NAME = {
    "Ricker Texture": "xDAWN-TS-LDA",
    "Narrow-band Random": "FBCCA",
    "Modulated Gold": "n-train rCCA",
    "Binary Sequence Set": "rCCA",
}
# Observation window at which GRASP and the published baseline are compared.
MATCHED_WINDOW = {"Ricker Texture": 1.87, "Narrow-band Random": 1.60,
                  "Modulated Gold": 2.10, "Binary Sequence Set": 4.20}
# Non-stimulus time between selections, taken from each source protocol.
OVERHEAD = {"Ricker Texture": 0.0, "Narrow-band Random": 0.5,
            "Modulated Gold": 1.0, "Binary Sequence Set": 0.8}

ALPHABETS = ("state", "contrast", "signed_difference", "on_transition",
             "off_transition", "temporal_lags", "spectral_components",
             "transition_likelihood")
ALPHABET_LABEL = {
    "state": "state", "contrast": "contrast", "signed_difference": "signed difference",
    "on_transition": "on-transition", "off_transition": "off-transition",
    "temporal_lags": "long-lag state", "spectral_components": "filter bank",
    "transition_likelihood": "transition likelihood",
}
ALPHABET_NOTE = {
    "state": "The code itself. One response per sustained level.",
    "contrast": "Any change in the code, regardless of direction.",
    "signed_difference": "Change with its sign kept.",
    "on_transition": "Only the rising edges.",
    "off_transition": "Only the falling edges.",
    "temporal_lags": "The state again, over a longer response window.",
    "spectral_components": "Latency-shifted copies of the code and their products. Uses no EEG.",
    "transition_likelihood": "One event and one no-event response, accumulated frame by frame.",
}
RESPONSE_ALPHABETS = ALPHABETS[:6]   # the six scored by regularised CCA

SOURCE = {
    "Ricker Texture": DATA / "dehais2024" / "subject-07.npz",
    "Narrow-band Random": DATA / "nbrs15" / "subject-13.npz",
    "Modulated Gold": DATA / "thielen2021" / "subject-09.npz",
    "Binary Sequence Set": DATA / "thielen2025" / "sub-01",
}


def _slug(dataset: str) -> str:
    return INTERNAL_NAME[dataset].replace(" ", "").replace("-", "").lower()


def cache_path(dataset: str) -> Path:
    return DEMO_CACHE / f"{_slug(dataset)}.npz"


# --------------------------------------------------------------------------- #
# the object the notebook actually talks to
# --------------------------------------------------------------------------- #
@dataclass
class Fixture:
    """Precomputed evidence for one participant on one dataset."""

    dataset: str
    times: np.ndarray            # (T,) observation windows, seconds
    is_decision: np.ndarray      # (T,) True where the decoder may emit a selection
    views: np.ndarray            # (trials, T, 8, targets) per-alphabet evidence
    eeg: np.ndarray              # (trials, samples, channels) held-out recordings
    response_sos: np.ndarray     # filter the decoder applies before scoring
    weights: np.ndarray          # (8,) fitted fusion weights, sum to 1, capped at 0.55
    truth: np.ndarray            # (trials,)
    codebook: np.ndarray         # (targets, code samples)
    sample_rate: float
    tau_reference: float         # trial length the stopping rule normalises against
    stop_threshold: float
    stop_coef: np.ndarray        # (3,) logistic coefficients on [margin, tau, margin*sqrt(tau)]
    stop_intercept: float
    responses: dict              # alphabet -> (samples,) temporal response r_v
    filters: dict                # alphabet -> (channels,) spatial filter w_v
    events: dict                 # alphabet -> (event name, onset_event, encoding length)
    subject: str
    n_calibration_trials: int
    calibration_seconds: float
    swap_views: np.ndarray | None = None
    swap_truth: np.ndarray | None = None
    swap_codebook: np.ndarray | None = None
    swap_family: str | None = None

    # -- shape helpers ----------------------------------------------------- #
    @property
    def n_targets(self) -> int:
        return self.codebook.shape[0]

    @property
    def n_trials(self) -> int:
        return self.views.shape[0]

    @property
    def n_channels(self) -> int:
        return self.eeg.shape[2]

    @property
    def n_samples(self) -> int:
        return self.eeg.shape[1]

    @property
    def lead_alphabet(self) -> str:
        """The response alphabet this participant leaned on most."""
        ranked = [(self.weights[i], a) for i, a in enumerate(ALPHABETS) if a in self.filters]
        return max(ranked)[1] if ranked else RESPONSE_ALPHABETS[0]

    def band_limited(self) -> np.ndarray:
        """The trials as the decoder sees them, through its own response filter."""
        values = np.asarray(self.eeg, float)
        initial = signal.sosfilt_zi(self.response_sos)
        zi = initial[:, None, :, None] * values[:, 0, :][None, :, None, :]
        return signal.sosfilt(self.response_sos, values, axis=1, zi=zi)[0]

    def project(self, alphabet: str | None = None) -> np.ndarray:
        """Spatially filter every trial down to the one signal the model reads."""
        name = alphabet or self.lead_alphabet
        return self.band_limited() @ self.filters[name]

    @property
    def trial_seconds(self) -> float:
        return float(self.times[-1])

    @property
    def chance(self) -> float:
        return 100.0 / self.n_targets

    # -- the decoder, re-expressed over the stored tensor ------------------ #
    def fuse(self, views: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Weighted sum over alphabets: S(c, t) of equation (3)."""
        w = self.weights if weights is None else np.asarray(weights, float)
        return np.einsum("...vc,v->...c", views, w)

    def stopping_probability(self, margin: np.ndarray, times: np.ndarray) -> np.ndarray:
        """q(t) of equation (4) from the top-two margin and the elapsed fraction."""
        tau = np.minimum(np.asarray(times, float) / self.tau_reference, 3.0)
        tau = np.broadcast_to(tau, np.shape(margin))
        z = (self.stop_intercept
             + self.stop_coef[0] * margin
             + self.stop_coef[1] * tau
             + self.stop_coef[2] * margin * np.sqrt(tau))
        return 1.0 / (1.0 + np.exp(-z))

    def trial_trace(self, trial: int, weights: np.ndarray | None = None,
                    views: np.ndarray | None = None) -> dict:
        """Everything the live-decoding panel needs for one held-out trial."""
        source = self.views if views is None else views
        fused = self.fuse(source[trial], weights)              # (T, targets)
        order = np.argsort(fused, axis=1)
        winner = order[:, -1]
        margin = (np.take_along_axis(fused, order[:, -1:], axis=1)[:, 0]
                  - np.take_along_axis(fused, order[:, -2:-1], axis=1)[:, 0])
        q = self.stopping_probability(margin, self.times)

        # A selection is emitted at the first decision time where the leader has
        # repeated and q passes the threshold; otherwise the final leader stands.
        decision_idx = np.flatnonzero(self.is_decision)
        emitted_at, forced = len(self.times) - 1, True
        for position, k in enumerate(decision_idx):
            if position == 0:
                continue
            previous = decision_idx[position - 1]
            if winner[k] == winner[previous] and q[k] >= self.stop_threshold:
                emitted_at, forced = int(k), False
                break
        return {"fused": fused, "winner": winner, "margin": margin, "q": q,
                "emitted_at": emitted_at, "forced": forced,
                "selection": int(winner[emitted_at]),
                "selection_time": float(self.times[emitted_at])}

    def accuracy_curve(self, weights: np.ndarray | None = None,
                       views: np.ndarray | None = None,
                       truth: np.ndarray | None = None) -> np.ndarray:
        """Held-out accuracy at every observation window, as a percentage."""
        source = self.views if views is None else views
        target = self.truth if truth is None else truth
        predicted = self.fuse(source, weights).argmax(axis=2)   # (trials, T)
        return 100.0 * (predicted == target[:, None]).mean(axis=0)


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def _notch_car(epochs: np.ndarray, sample_rate: float) -> np.ndarray:
    values = np.asarray(epochs, float).copy()
    values -= values.mean(axis=2, keepdims=True)
    b, a = signal.iirnotch(50, 30, sample_rate)
    return signal.sosfilt(signal.tf2sos(b, a), values, axis=1)


def _unique_codebook(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes, labels = [], []
    for row in V:
        match = next((i for i, x in enumerate(codes) if np.array_equal(row, x)), None)
        if match is None:
            match = len(codes)
            codes.append(row.copy())
        labels.append(match)
    return np.stack(codes), np.asarray(labels, int)


def _prepare(dataset: str):
    """Fit one decoder and return it with its held-out trials."""
    sys.path.insert(0, str(ROOT / "src"))
    from cvep_decoder import GRASPDecoder

    if dataset == "Ricker Texture":
        z = np.load(SOURCE[dataset], allow_pickle=False)
        eeg = z["eeg"].astype(float)
        labels, runs = z["labels"].astype(int), z["runs"].astype(int)
        codebook, fs = z["codebook"].astype(float), 500.0
        cal, test = runs < 8, runs >= 8
        decoder = GRASPDecoder().fit(eeg[cal], fs, codebook, labels[cal])
        return (decoder, eeg[test], labels[test], codebook, fs, "subject 07",
                int(cal.sum()), float(cal.sum() * eeg.shape[1] / fs), None)

    if dataset == "Narrow-band Random":
        z = np.load(SOURCE[dataset], allow_pickle=False)
        fs = 250.0
        referenced = _notch_car(z["epoch_eeg"].astype(float), fs)
        start = 250 + round(0.14 * 250)
        eeg = referenced[:, start:start + 4 * 250]
        labels, runs = z["labels"].astype(int), z["runs"].astype(int)
        codebook = z["codebook"].astype(float)
        cal, test = (runs == 0) & (labels < 10), runs > 0
        decoder = GRASPDecoder().fit(eeg[cal], fs, codebook, labels[cal])
        return (decoder, eeg[test], labels[test], codebook, fs, "subject 13",
                int(cal.sum()), float(cal.sum() * 4.0), None)

    if dataset == "Modulated Gold":
        z = np.load(SOURCE[dataset], allow_pickle=False)
        eeg = z["eeg"].astype(float)
        labels, runs = z["labels"].astype(int), z["runs"].astype(int)
        codebook, fs = z["codebook"].astype(float), 240.0
        # Target 4 repeats another code in the released stimulus matrix.
        keep = labels != 4
        labels = labels[keep] - (labels[keep] > 4)
        eeg, runs = eeg[keep], runs[keep]
        codebook = np.delete(codebook, 4, axis=0)
        available = [1, 2, 3, 4]
        index = np.asarray([
            np.flatnonzero((runs == available[t % 4]) & (labels == t))[0]
            for t in range(len(codebook))])
        test = runs == 0
        decoder = GRASPDecoder().fit(eeg[index, :504], fs, codebook, labels[index])
        return (decoder, eeg[test, :504], labels[test], codebook, fs, "subject 09",
                len(index), float(len(index) * 504 / fs), None)

    if dataset == "Binary Sequence Set":
        folder = SOURCE[dataset]
        # Of the 80 ordered cross-family pairs, this one's 26-participant mean (79.0 %)
        # sits closest to the cross-family average (79.3 %), so the single-participant
        # demonstration is representative rather than a best or worst case.
        donor, receiver = "golay", "m-mseq"
        loaded = {}
        for name, condition in (("donor", donor), ("receiver", receiver)):
            z = np.load(folder / f"sub-01_cvep_{condition}.npz", allow_pickle=False)
            X = np.transpose(z["X"].astype(float), (0, 2, 1))
            codes, y = _unique_codebook(z["V"].astype(float))
            loaded[name] = (X, codes, y, float(z["fs"]))
        Xd, codes_d, yd, fs = loaded["donor"]
        Xr, codes_r, yr, _ = loaded["receiver"]
        folds = np.arange(32) * 4 // 32
        train, test = folds != 0, folds == 0
        decoder = GRASPDecoder().fit(Xd[train], fs, codes_d, yd[train])
        swap = {"codebook": codes_r, "eeg": Xr[test], "truth": yr[test], "family": receiver}
        return (decoder, Xd[test], yd[test], codes_d, fs, "sub-01",
                int(train.sum()), float(train.sum() * Xd.shape[1] / fs), swap)

    raise ValueError(f"unknown dataset {dataset!r}")


def _time_grid(decoder, trial_seconds: float, codebook: np.ndarray, fs: float):
    """The decoder's own decision times, plus display points between them."""
    decision = list(decoder.decision_times_s)
    step = min(0.25, 0.5 * codebook.shape[1] / fs)
    t = (math.floor(decision[-1] / step) + 1) * step
    while t <= trial_seconds + 1e-9:
        decision.append(round(t, 6))
        t += step
    decision = sorted(set(decision))
    # The shortest usable window is the decoder's first decision time; below it the
    # transition-likelihood alphabet has too few response epochs to score.
    display = np.round(np.arange(decision[0], trial_seconds + 1e-9, 0.1), 6)
    times = np.unique(np.concatenate([np.asarray(decision), display]))
    times = times[times <= trial_seconds + 1e-9]
    is_decision = np.isin(times, np.asarray(decision))
    return times, is_decision


def _score_grid(decoder, eeg: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(trials, T, alphabets, targets) evidence, evaluated once per window."""
    return np.stack([decoder.view_scores(eeg, float(t)) for t in times],
                    axis=1).astype(np.float32)


def build(dataset: str, verbose: bool = True) -> Path:
    decoder, eeg, truth, codebook, fs, subject, n_cal, cal_s, swap = _prepare(dataset)
    trial_seconds = eeg.shape[1] / fs
    times, is_decision = _time_grid(decoder, trial_seconds, codebook, fs)
    if verbose:
        print(f"{dataset}: scoring {len(eeg)} trials at {len(times)} windows", flush=True)

    payload = {
        "dataset": dataset,
        "times": times.astype(np.float64),
        "is_decision": is_decision,
        "views": _score_grid(decoder, eeg, times),
        "eeg": np.asarray(eeg, np.float32),
        # The same second-order sections the decoder applies before scoring, so the
        # walkthrough can show the signal the model actually reads.
        "response_sos": np.asarray(decoder._response_sos, float),
        "weights": np.asarray(decoder.weights, float),
        "truth": np.asarray(truth, int),
        "codebook": np.asarray(codebook, np.float32),
        "sample_rate": float(fs),
        "tau_reference": float(decoder.decision_times_s[-1]),
        "stop_threshold": float(decoder.stop_threshold),
        "stop_coef": np.asarray(decoder.stop_calibrator.coef_, float).ravel(),
        "stop_intercept": float(np.asarray(decoder.stop_calibrator.intercept_).ravel()[0]),
        "subject": subject,
        "n_calibration_trials": int(n_cal),
        "calibration_seconds": float(cal_s),
    }
    for name in ALPHABETS:
        model = decoder.models[name]
        if hasattr(model, "r_"):
            payload[f"response__{name}"] = np.asarray(model.r_, float).ravel()
            payload[f"filter__{name}"] = np.asarray(model.w_, float).ravel()
            payload[f"event__{name}"] = np.array(
                [str(model.event), str(bool(model.onset_event)), str(model.encoding_length)])

    if swap is not None:
        # The same fitted decoder, told to expect a codebook from another family.
        decoder.set_codebook(swap["codebook"])
        payload["swap_views"] = _score_grid(decoder, swap["eeg"], times)
        payload["swap_truth"] = np.asarray(swap["truth"], int)
        payload["swap_codebook"] = np.asarray(swap["codebook"], np.float32)
        payload["swap_family"] = swap["family"]

    DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(dataset)
    np.savez_compressed(path, **payload)
    if verbose:
        print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return path


def load(dataset: str, rebuild: bool = False) -> Fixture:
    path = cache_path(dataset)
    if rebuild or not path.exists() or path.stat().st_size == 0:
        build(dataset)
    z = np.load(path, allow_pickle=False)
    responses, filters, events = {}, {}, {}
    for name in ALPHABETS:
        if f"response__{name}" in z:
            responses[name] = z[f"response__{name}"]
            filters[name] = z[f"filter__{name}"]
            event, onset, length = z[f"event__{name}"]
            events[name] = (str(event), onset == "True", float(length))
    return Fixture(
        dataset=str(z["dataset"]),
        times=z["times"], is_decision=z["is_decision"], views=z["views"],
        eeg=z["eeg"], response_sos=z["response_sos"],
        weights=z["weights"], truth=z["truth"], codebook=z["codebook"],
        sample_rate=float(z["sample_rate"]), tau_reference=float(z["tau_reference"]),
        stop_threshold=float(z["stop_threshold"]), stop_coef=z["stop_coef"],
        stop_intercept=float(z["stop_intercept"]),
        responses=responses, filters=filters, events=events,
        subject=str(z["subject"]), n_calibration_trials=int(z["n_calibration_trials"]),
        calibration_seconds=float(z["calibration_seconds"]),
        swap_views=z["swap_views"] if "swap_views" in z else None,
        swap_truth=z["swap_truth"] if "swap_truth" in z else None,
        swap_codebook=z["swap_codebook"] if "swap_codebook" in z else None,
        swap_family=str(z["swap_family"]) if "swap_family" in z else None,
    )


def ready(dataset: str) -> bool:
    """True when the dataset can be shown, from cache or from the recording."""
    path = cache_path(dataset)
    if path.exists() and path.stat().st_size > 0:
        return True
    return SOURCE[dataset].exists()


def event_train(codebook: np.ndarray, alphabet: str, events: dict) -> tuple[np.ndarray, tuple]:
    """Decompose codes into the event trains one alphabet predicts."""
    from pyntbci.utilities import event_matrix

    event, onset, _ = events[alphabet]
    matrix, names = event_matrix(np.asarray(codebook, float), event, onset)
    return np.asarray(matrix), tuple(names)


# --------------------------------------------------------------------------- #
# feature sets for the two-dimensional views
# --------------------------------------------------------------------------- #
FEATURE_SETS = {
    "Event-locked epochs": (
        "Short windows of the band-limited, spatially filtered signal, cut at rising "
        "edges, at falling edges, and where the code is not changing. Expect a blob: a "
        "single epoch of single-trial EEG is mostly noise. The probe score above runs "
        "from a little over chance on the narrow-band and binary-sequence codes to flat "
        "chance on the other two. That is the honest picture of what one event is worth, "
        "and it is why the decoder integrates thousands of them across a trial rather "
        "than classifying any one."
    ),
    "Candidate evidence": (
        "One point per (trial, candidate code): the eight alphabet scores at the "
        "matched window, just before they are fused. This is the space the decision is "
        "actually taken in, and the attended code separates from its competitors almost "
        "perfectly. Note the classes are very unbalanced, one attended code against "
        "n-1 competitors, so compare the probe score with the majority baseline rather "
        "than with 50 %."
    ),
    "Whole-trial responses": (
        "One point per held-out trial, the band-limited and spatially filtered signal "
        "over the whole trial, coloured by which code was attended. Where there are at "
        "least two trials per target the separation is strong, which is the same fact "
        "the one-trial-per-target baselines rely on. Modulated Gold and the binary "
        "sequence set have a single held-out trial per code, so nothing can be measured "
        "there and the scatter is decoration."
    ),
}


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    centred = values - values.mean(axis=1, keepdims=True)
    scale = np.maximum(centred.std(axis=1, keepdims=True), 1e-12)
    return centred / scale


def _event_epochs(fixture: "Fixture", epoch_s: float = 0.3, cap: int = 2400, seed: int = 0):
    """Cut the filtered signal at rising edges, falling edges and flat stretches."""
    rng = np.random.default_rng(seed)
    projected = fixture.project()
    width = max(4, int(epoch_s * fixture.sample_rate))
    rows, labels = [], []
    for trial in range(fixture.n_trials):
        signal_1d = projected[trial]
        code = np.asarray(fixture.codebook[int(fixture.truth[trial])], float)
        code = code[: len(signal_1d)]
        slope = np.diff(code, prepend=code[0])
        # Quantile cuts rather than a fixed threshold, so this also works for the
        # narrow-band codes, which are smooth rather than binary.
        high, low = np.quantile(slope, 0.9), np.quantile(slope, 0.1)
        flat = np.quantile(np.abs(slope), 0.25)
        groups = (
            (np.flatnonzero(slope >= high), 0),
            (np.flatnonzero(slope <= low), 1),
            (np.flatnonzero(np.abs(slope) <= flat), 2),
        )
        for positions, label in groups:
            positions = positions[positions + width <= len(signal_1d)]
            if len(positions) > 40:
                positions = rng.choice(positions, 40, replace=False)
            for start in positions:
                rows.append(signal_1d[start : start + width])
                labels.append(label)
    features = _zscore(np.asarray(rows))
    labels = np.asarray(labels, int)
    if len(features) > cap:
        keep = rng.choice(len(features), cap, replace=False)
        features, labels = features[keep], labels[keep]
    return features, labels, ("rising edge", "falling edge", "no transition"), "event"


def _candidate_evidence(fixture: "Fixture", window: float | None = None):
    """The eight alphabet scores for every candidate code on every trial."""
    target_window = MATCHED_WINDOW[fixture.dataset] if window is None else window
    index = int(np.argmin(np.abs(fixture.times - target_window)))
    block = fixture.views[:, index]                       # (trials, 8, targets)
    features = block.transpose(0, 2, 1).reshape(-1, block.shape[1]).astype(float)
    attended = (np.arange(fixture.n_targets)[None, :] == fixture.truth[:, None])
    return features, attended.ravel().astype(int), ("competing code", "attended code"), "attended"


def _trial_responses(fixture: "Fixture"):
    return (_zscore(fixture.project()), np.asarray(fixture.truth, int), None, "target")


def embedding_inputs(fixture: "Fixture", feature_set: str):
    """Return (features, labels, legend, kind) for one of the FEATURE_SETS."""
    if feature_set == "Event-locked epochs":
        return _event_epochs(fixture)
    if feature_set == "Candidate evidence":
        return _candidate_evidence(fixture)
    if feature_set == "Whole-trial responses":
        return _trial_responses(fixture)
    raise ValueError(f"unknown feature set {feature_set!r}")


# --------------------------------------------------------------------------- #
# embedding methods, detected at import time
# --------------------------------------------------------------------------- #
def _sklearn_embedders() -> dict:
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, Isomap, SpectralEmbedding, TSNE

    def tsne(features, seed):
        perplexity = float(min(30, max(5, (len(features) - 1) / 3)))
        return TSNE(n_components=2, random_state=seed, init="pca",
                    perplexity=perplexity).fit_transform(features)

    return {
        "PCA": lambda features, seed: PCA(n_components=2, random_state=seed).fit_transform(features),
        "t-SNE": tsne,
        "Isomap": lambda features, seed: Isomap(
            n_components=2, n_neighbors=int(min(10, max(2, len(features) - 1)))
        ).fit_transform(features),
        "Spectral embedding": lambda features, seed: SpectralEmbedding(
            n_components=2, random_state=seed,
            n_neighbors=int(min(10, max(2, len(features) - 1))),
        ).fit_transform(features),
        "MDS": lambda features, seed: MDS(
            n_components=2, random_state=seed, normalized_stress="auto"
        ).fit_transform(features),
    }


def _optional_embedders() -> dict:
    """Whatever extra libraries happen to be installed."""
    found = {}
    try:
        import umap

        found["UMAP"] = lambda features, seed: umap.UMAP(
            n_components=2, random_state=seed,
            n_neighbors=int(min(15, max(2, len(features) - 1))),
        ).fit_transform(features)
    except Exception:
        pass
    try:
        import pacmap

        found["PaCMAP"] = lambda features, seed: pacmap.PaCMAP(
            n_components=2, random_state=seed
        ).fit_transform(np.asarray(features, np.float32))
        if hasattr(pacmap, "LocalMAP"):
            found["LocalMAP"] = lambda features, seed: pacmap.LocalMAP(
                n_components=2, random_state=seed
            ).fit_transform(np.asarray(features, np.float32))
    except Exception:
        pass
    try:
        import trimap

        found["TriMap"] = lambda features, seed: trimap.TRIMAP(
            n_dims=2
        ).fit_transform(np.asarray(features, np.float32))
    except Exception:
        pass
    try:
        from openTSNE import TSNE as OpenTSNE

        found["openTSNE"] = lambda features, seed: np.asarray(
            OpenTSNE(n_components=2, random_state=seed).fit(np.asarray(features, float))
        )
    except Exception:
        pass
    try:
        import cne

        found["CNE"] = lambda features, seed: cne.CNE(
            n_components=2, seed=seed
        ).fit_transform(np.asarray(features, np.float32))
    except Exception:
        pass
    return found


# Register your own here as name -> callable(features, seed) -> (n, 2) array.
EXTRA_EMBEDDERS: dict = {}


def embedders() -> dict:
    found = _sklearn_embedders()
    found.update(_optional_embedders())
    found.update(EXTRA_EMBEDDERS)
    return found


def embed(features: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    available = embedders()
    if method not in available:
        raise KeyError(f"{method!r} is not installed; have {sorted(available)}")
    values = np.asarray(features, float)
    # Every method here is scale sensitive, so put the axes on equal footing first.
    values = values - values.mean(axis=0, keepdims=True)
    values = values / np.maximum(values.std(axis=0, keepdims=True), 1e-12)
    return np.asarray(available[method](values, seed), float)[:, :2]


def separability(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """How well a linear rule separates the classes, against always guessing the
    largest one. A scatter plot can only suggest structure; this measures it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    majority = float(counts.max() / len(labels))
    folds = int(min(5, counts.min()))
    if folds < 2 or len(labels) < 20:
        return {"majority": majority, "probe": None, "folds": 0}
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=400, random_state=seed))
    scores = cross_val_score(
        model, np.asarray(features, float), labels,
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
    )
    return {"majority": majority, "probe": float(scores.mean()), "folds": folds}


# --------------------------------------------------------------------------- #
# the decoding walkthrough
# --------------------------------------------------------------------------- #
def predicted_response(fixture: "Fixture", target: int, alphabet: str,
                       n_samples: int | None = None) -> np.ndarray:
    """Convolve a candidate's event train with the fitted response for an alphabet.

    This is the r_v^T M_v(c) side of equation (1): what the decoder expects to see
    if the participant is looking at `target`.
    """
    matrix, _ = event_train(fixture.codebook[target : target + 1], alphabet, fixture.events)
    response = np.asarray(fixture.responses[alphabet], float)
    length = fixture.n_samples if n_samples is None else int(n_samples)
    per_event = len(response) // max(1, matrix.shape[1])
    prediction = np.zeros(length + len(response))
    for index in range(matrix.shape[1]):
        kernel = response[index * per_event : (index + 1) * per_event]
        if not len(kernel):
            continue
        prediction[:length + len(kernel) - 1] += np.convolve(
            matrix[0, index, :length], kernel)[: length + len(kernel) - 1]
    return prediction[:length]


def walkthrough(fixture: "Fixture", trial: int, window: float,
                alphabet: str | None = None) -> dict:
    """Every intermediate quantity for one trial, in the order the decoder makes them."""
    name = alphabet or fixture.lead_alphabet
    index = int(np.argmin(np.abs(fixture.times - window)))
    n = int(min(fixture.n_samples, round(window * fixture.sample_rate)))

    raw = np.asarray(fixture.eeg[trial, :n], float)
    spatial = np.asarray(fixture.filters[name], float)
    measured = fixture.band_limited()[trial, :n] @ spatial

    attended = int(fixture.truth[trial])
    per_alphabet = fixture.views[trial, index]                 # (8, targets)
    fused = fixture.fuse(per_alphabet)
    leader = int(np.argmax(fused))
    runner_up = int(np.argsort(fused)[-2])

    return {
        "alphabet": name,
        "window": float(fixture.times[index]),
        "n_samples": n,
        "raw": raw,
        "spatial_filter": spatial,
        "measured": measured,
        "attended": attended,
        "leader": leader,
        "runner_up": runner_up,
        "predicted_attended": predicted_response(fixture, attended, name, n),
        "predicted_runner_up": predicted_response(fixture, runner_up, name, n),
        "per_alphabet_attended": per_alphabet[:, attended],
        "per_alphabet_leader": per_alphabet[:, leader],
        "per_alphabet_runner_up": per_alphabet[:, runner_up],
        "fused": fused,
        "correct": leader == attended,
    }


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(DATASETS)
    for name in wanted:
        build(name)
