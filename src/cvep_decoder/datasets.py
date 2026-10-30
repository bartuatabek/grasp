"""Event-aligned conversion of public c-VEP datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import signal

NBRS_OCCIPITAL_8 = ("POz", "PO3", "PO4", "PO7", "PO8", "O1", "Oz", "O2")
DEHAIS_OCCIPITAL_8 = ("PO7", "O1", "Oz", "O2", "PO8", "PO3", "POz", "PO4")


def convert_dehais2024_subject(
    source: str | Path,
    output: str | Path,
    *,
    sample_rate: int = 500,
    visual_latency_s: float = 0.08,
) -> Path:
    """Convert one Ricker-texture Dehais2024 recording.

    This implements the protocol reported by Dehais et al.: all eight dry EEG
    channels, 1--25 Hz continuous filtering, POz rereferencing, an 80 ms
    display-to-EEG alignment correction, and 2.2 s stimulus epochs.  Trial
    labels and the five actually displayed codes are read from the EEGLAB
    annotations rather than reconstructed.
    """
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise RuntimeError("conversion requires the benchmark dependencies") from exc

    if sample_rate != 500:
        raise ValueError("the matched Dehais2024 conversion preserves the native 500 Hz")
    raw = mne.io.read_raw_eeglab(source, preload=True, verbose=False)
    if len(raw.ch_names) != 8 or int(round(raw.info["sfreq"])) != 500:
        raise ValueError("expected eight-channel Dehais EEG sampled at 500 Hz")

    trial_events = [
        (float(onset), str(description))
        for onset, description in zip(
            raw.annotations.onset, raw.annotations.description, strict=True
        )
        if len(str(description)) >= 134
        and str(description)[-2] == "_"
        and set(str(description)[:132]) <= {"0", "1"}
    ]
    if len(trial_events) != 75:
        raise ValueError(f"expected 75 Dehais trials, found {len(trial_events)}")

    # The EEGLAB file stores generic channel names, but the recording order is
    # fixed and documented in the paper.  Preserve that order before applying
    # the paper's POz reference (index 6).
    raw.rename_channels(dict(zip(raw.ch_names, DEHAIS_OCCIPITAL_8, strict=True)))
    raw.filter(l_freq=1.0, h_freq=25.0, verbose=False)
    data = raw.get_data().T
    data = data - data[:, [DEHAIS_OCCIPITAL_8.index("POz")]]
    if sample_rate != int(round(raw.info["sfreq"])):
        data = signal.resample_poly(data, sample_rate, int(round(raw.info["sfreq"])), axis=0)

    n_samples = round(2.2 * sample_rate)
    latency_samples = round(visual_latency_s * sample_rate)
    epochs: list[NDArray[np.float64]] = []
    labels: list[int] = []
    codes: dict[int, NDArray[np.float64]] = {}
    for onset, description in trial_events:
        label = int(description[-1]) - 1
        code = np.fromiter((int(bit) for bit in description[:132]), dtype=np.float64)
        codes.setdefault(label, code)
        start = round(onset * sample_rate) + latency_samples
        stop = start + n_samples
        if stop > len(data):
            raise ValueError("stimulus epoch extends beyond the EEG recording")
        epochs.append(data[start:stop])
        labels.append(label)
    code_frames = np.stack([codes[index] for index in range(5)])
    frame_index = np.floor(np.arange(n_samples) * 60 / sample_rate).astype(int)
    codebook = code_frames[:, frame_index]

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subject_text = Path(source).stem.split("_", 1)[0].lstrip("P")
    np.savez_compressed(
        destination,
        eeg=np.asarray(epochs, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int16),
        runs=np.repeat(np.arange(15, dtype=np.int8), 5),
        codebook=codebook.astype(np.float32),
        channels=np.asarray(DEHAIS_OCCIPITAL_8),
        sample_rate=np.asarray(sample_rate),
        code_cycle_s=np.asarray(2.2),
        visual_latency_s=np.asarray(visual_latency_s),
        subject=np.asarray(int(subject_text)),
        dataset=np.asarray("Dehais2024-Ricker"),
        preprocessing=np.asarray(
            "continuous zero-phase 1-25 Hz; POz reference; native 500 Hz; 80 ms latency correction"
        ),
    )
    return destination


def convert_nbrs_subject(
    source: str | Path,
    code_source: str | Path,
    channel_source: str | Path,
    output: str | Path,
    *,
    channels: tuple[str, ...] | None = NBRS_OCCIPITAL_8,
    sample_rate: int = 250,
    visual_latency_s: float = 0.14,
) -> Path:
    """Convert one subject from the public 40-target NBRS dataset.

    The source epochs contain one second before stimulation, four seconds of
    stimulation, and one second after stimulation. The converter retains the
    unfiltered active EEG and the pre-stimulus segment so filtering choices can
    be evaluated without reconverting. Target labels and block identifiers are
    generated directly from the documented matrix axes.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("conversion requires scipy") from exc

    if sample_rate != 250:
        raise ValueError("the public NBRS data are distributed at 250 Hz")
    raw = np.asarray(loadmat(source)["data"], dtype=np.float32)
    if raw.ndim != 4 or raw.shape[0] != 59 or raw.shape[2:] != (3, 40):
        raise ValueError("expected NBRS data with shape (59, samples, 3 blocks, 40 targets)")
    channel_labels = tuple(
        line.split()[-1] for line in Path(channel_source).read_text().splitlines() if line.strip()
    )
    if len(channel_labels) != raw.shape[0]:
        raise ValueError("channel metadata do not match the EEG matrix")
    selected = channel_labels if channels is None else channels
    missing = sorted(set(selected) - set(channel_labels))
    if missing:
        raise ValueError(f"source data are missing channels: {missing}")
    indices = [channel_labels.index(name) for name in selected]

    onset = sample_rate
    active_start = onset + round(visual_latency_s * sample_rate)
    active_samples = 4 * sample_rate
    if active_start + active_samples > raw.shape[1]:
        raise ValueError("source epoch is too short for the requested latency")

    # Matrix order is channels × samples × blocks × targets. Trials are stored
    # block-major so chronological leave-one-block-out evaluation is explicit.
    epochs = raw[indices].transpose(2, 3, 1, 0).reshape(3 * 40, raw.shape[1], len(indices))

    code_path = Path(code_source)
    if code_path.suffix == ".npz":
        with np.load(code_path, allow_pickle=False) as code_data:
            codebook = np.asarray(code_data["codebook"], dtype=np.float64)
    else:
        code = np.asarray(loadmat(code_path)["code"], dtype=np.float64)
        if code.ndim != 2 or code.shape[0] < 40 or code.shape[1] != 480:
            raise ValueError("expected an NBRS codebook with at least 40 × 480")
        codebook = signal.resample_poly(code[:40], 25, 12, axis=1)
    if codebook.shape[0] != 40:
        raise ValueError("expected the 40 biologically presented NBRS codes")
    codebook = codebook[:, :active_samples]

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subject_name = Path(source).parents[1].name
    paradigm = Path(source).parent.name
    np.savez_compressed(
        destination,
        epoch_eeg=epochs.astype(np.float32),
        labels=np.tile(np.arange(40, dtype=np.int16), 3),
        runs=np.repeat(np.arange(3, dtype=np.int8), 40),
        codebook=codebook.astype(np.float32),
        channels=np.asarray(selected),
        sample_rate=np.asarray(sample_rate),
        code_cycle_s=np.asarray(1.0),
        visual_latency_s=np.asarray(visual_latency_s),
        subject=np.asarray(subject_name),
        dataset=np.asarray(f"Zheng2024-{paradigm}"),
        preprocessing=np.asarray("raw 250 Hz six-second epochs; no filter"),
    )
    return destination


def load_nbrs_subject(
    path: str | Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Load active EEG, idle EEG, labels, block IDs, and codebook."""
    with np.load(path, allow_pickle=False) as data:
        sample_rate = int(data["sample_rate"])
        latency = float(data["visual_latency_s"])
        epochs = data["epoch_eeg"].astype(np.float64)
        onset = sample_rate
        start = onset + round(latency * sample_rate)
        return (
            epochs[:, start : start + 4 * sample_rate],
            epochs[:, :onset],
            data["labels"].astype(np.int64),
            data["runs"].astype(np.int64),
            data["codebook"].astype(np.float64),
        )


def convert_thielen2021_subject(
    subject: int,
    output: str | Path,
    data_root: str | Path | None = None,
) -> Path:
    """Convert one 20-target Thielen2021 subject at the verified 240 Hz.

    The output matches the public PyntBCI example representation: five code
    cycles (10.5 s) per trial, eight recorded EEG channels, and one 2.1 s code
    cycle. Keeping 240 Hz is important because duration events occupy four or
    eight samples; reducing to 120 Hz materially degraded decoding in our
    alignment check.
    """
    try:
        import mne
        from moabb.datasets import Thielen2021
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise RuntimeError("conversion requires the benchmark dependencies") from exc

    dataset = Thielen2021()
    if data_root is not None:
        mne.set_config(
            "MNE_DATASETS_THIELEN2021_PATH",
            str(Path(data_root).resolve()),
            set_env=True,
        )
    paths = dataset.data_path(subject)
    runs = dataset.get_data(subjects=[subject])[subject]["0"]
    eeg_runs: list[NDArray[np.float32]] = []
    label_runs: list[NDArray[np.int64]] = []
    run_ids: list[NDArray[np.int8]] = []
    duration_s = 10.5
    sample_rate = 240
    n_samples = round(duration_s * sample_rate)

    for run_id, raw in enumerate(runs.values()):
        trial_events = mne.find_events(raw, stim_channel="stim_trial", verbose=False)
        labels = trial_events[:, 2].astype(np.int64) - 200
        continuous = raw.copy().pick("eeg")
        continuous.filter(l_freq=2.0, h_freq=30.0, verbose=False)
        events = np.column_stack(
            [trial_events[:, 0], np.zeros(len(trial_events), dtype=int), labels]
        )
        epochs = mne.Epochs(
            continuous,
            events,
            tmin=0,
            tmax=duration_s,
            baseline=None,
            preload=True,
            verbose=False,
        )
        epochs.resample(sample_rate, verbose=False)
        eeg_runs.append(epochs.get_data()[:, :, :n_samples].transpose(0, 2, 1).astype(np.float32))
        label_runs.append(labels)
        run_ids.append(np.full(len(labels), run_id, dtype=np.int8))

    raw_codes = loadmat(paths[-2])["codes"].T
    codebook = np.repeat(raw_codes, sample_rate // 60, axis=1)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        eeg=np.concatenate(eeg_runs),
        labels=np.concatenate(label_runs).astype(np.int16),
        runs=np.concatenate(run_ids),
        codebook=codebook.astype(np.float32),
        channels=np.asarray(["Fpz", "T7", "O1", "POz", "Oz", "Iz", "O2", "T8"]),
        sample_rate=np.asarray(sample_rate),
        subject=np.asarray(subject),
        dataset=np.asarray("Thielen2021"),
        preprocessing=np.asarray(
            "continuous MNE zero-phase 2-30 Hz; resample 240 Hz; first 10.5 s"
        ),
    )
    return destination
