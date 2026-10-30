# GRASP-cVEP

**Generic Representation-Adaptive Sequential Prediction for c-VEP decoding**

GRASP identifies the visual target a person is attending to from their EEG.
It is designed for code-modulated visual evoked potential (c-VEP) experiments,
where each target follows a known visual stimulus sequence.

The decoder learns from labeled recordings for an individual participant.
During a new trial, it compares the incoming EEG with the available targets
and can select a target before the full recording has finished. The same
fitting interface accepts different stimulus codebooks; callers do not need
to supply a dataset name or choose a code-family setting.

This guide covers installation, input preparation, and use of the public
Python interface. Run the commands from the repository root in an authorized
checkout. The examples do not require a recording device.

## How GRASP works

A c-VEP experiment presents several targets at once. Each target changes over
time according to its own code. The participant attends to one target, and
the visual response is recorded through EEG.

GRASP uses calibration trials with known targets to learn how that
participant's EEG relates to the supplied stimulus codes. It adapts its
scoring to the calibration data, then evaluates new trials against the
candidate targets. As more EEG becomes available, it updates the target
scores and its estimate of decision reliability.

A trial may finish with an early selection or continue to a caller-defined
end point. At that end point, `final=True` requests a choice from the available
evidence. If an optional idle gate classifies the trial as idle, it suppresses
the selection even on the final update.

Calibration is participant-specific. Using one interface across codebooks
does not establish that a fitted decoder will transfer to a new participant,
recording session, or stimulus family without validation.

## Installation

Use Python 3.11, 3.12, or 3.13. With `uv` installed:

```bash
uv sync --extra benchmark --extra dev
uv run python -c "from cvep_decoder import GRASPDecoder; print('GRASP import OK')"
```

Alternatively, use a virtual environment and pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark,dev]'
```

On Windows, activate the environment with `.venv\Scripts\activate`.
For the commands below, pip users can omit `uv run` after activation.
The benchmark extras supply the recording and notebook dependencies; the dev
extras supply the test and lint tools.

## First run without EEG files

This example generates artificial EEG, fits a decoder, and selects a target
from a separate synthetic trial. It checks the basic workflow, not performance
on human recordings. Save it as `quickstart.py` in the repository root and run
`uv run python quickstart.py`.

```python
import numpy as np

from cvep_decoder import GRASPDecoder

rng = np.random.default_rng(8)
sample_rate = 100
n_targets, n_samples, n_channels = 5, 150, 8
codebook = rng.integers(0, 2, size=(n_targets, 100)).astype(float)
labels = np.tile(np.arange(n_targets), 4)
kernel = np.exp(-np.arange(18) / 4)
mixing = np.array([1.0, 0.8, 0.5, 0.2, -0.2, -0.4, 0.1, -0.1])


def make_trial(target):
    code = np.resize(codebook[target], n_samples)
    response = np.convolve(np.diff(code, prepend=code[0]), kernel)[:n_samples]
    noise = rng.normal(0, 0.12, size=(n_samples, n_channels))
    return response[:, None] * mixing + noise


calibration_eeg = np.stack([make_trial(target) for target in labels])
decoder = GRASPDecoder().fit(calibration_eeg, sample_rate, codebook, labels)

decoder.reset()
state = decoder.update(make_trial(2), final=True)
assert state.committed and state.target is not None
assert 0 <= state.target < n_targets
print(f"Selected target: {state.target}; reason: {state.reason}")
```

## Prepare real inputs

`GRASPDecoder.fit(eeg, sample_rate, codebook, labels)` takes four inputs:

| Input | Shape or value | Meaning |
|---|---|---|
| `eeg` | `(trials, samples, channels)` | Finite calibration EEG, in chronological trial order. |
| `sample_rate` | Positive number, in Hz | Sampling rate of the supplied EEG. |
| `codebook` | `(targets, stimulus_samples)` | Finite stimulus traces with one row per target, represented on the EEG sampling time base. |
| `labels` | `(trials,)` | Integer target indices starting at zero; each index refers to a codebook row. |

Provide at least two targets and six calibration trials. These are interface
minimums, not a recommendation for sufficient calibration. Include labeled
examples of the targets you intend to decode and assess calibration needs on
separate recordings.

Keep EEG channel count, channel order, units, and preprocessing consistent
between calibration and decoding. Align trial starts with stimulus onset and
preserve the actual stimulus timing when constructing the codebook. Do not
pass a display-frame sequence as though it were already sampled at the EEG
rate. For repeated codes, supply the repeating stimulus cycle.

Use separate calibration and evaluation trials. Keep evaluation labels out of
fitting, stopping-rule selection, and feedback until the evaluation is complete.

## Decode a trial as EEG arrives

Each update takes the **complete EEG prefix from the start of the current
trial**, shaped `(samples_so_far, channels)`. It does not take only the newest
chunk. Reset before each trial, stop after a committed selection, and deliver
that selection once.

```python
decoder.reset()

# eeg_prefixes contains progressively longer prefixes of one trial.
for index, prefix in enumerate(eeg_prefixes):
    state = decoder.update(prefix, final=index == len(eeg_prefixes) - 1)
    if state.committed and state.target is not None:
        selected_target = state.target
        break
```

Elapsed time defaults to the prefix length divided by the fitted sample rate.
If you supply `elapsed_s` explicitly, it must describe that same observation.

| Result field | Interpretation |
|---|---|
| `target` | Selected codebook row, or `None` while collecting or idle. |
| `committed` | Whether the caller should accept this selection. |
| `reason` | `collecting`, `calibrated_stop`, `final`, or `idle`. |
| `scores` | Per-target scores in codebook row order; these are not probabilities. |
| `estimated_accuracy` | Calibration-derived reliability estimate, not a guarantee. |
| `control_probability` | Idle gate's control estimate; defaults to `1.0` without a fitted gate. |
| `elapsed_s` | EEG observation duration, not computation time. |

For batch scoring, `decoder.decision_function(eeg)` returns a matrix with one
row per trial and one column per target. Batch scores alone do not apply the
sequential stopping or idle decision.

## Optional operations

Use these after fitting when the experiment needs them:

- `decoder.fit_idle(idle_eeg, control_eeg)` fits the optional idle gate using
  separately labeled idle and active recordings, each in trial/sample/channel
  order. Without this step, the decoder does not suppress idle activity.
- `decoder.set_codebook(new_codebook)` changes the candidate targets. Call it
  between trials, then reset. Labels and returned target indices refer to the
  new row order. Check performance after changing the stimulus set.
- `decoder.feedback(trial_eeg, predicted_target=chosen_target,
  corrected_target=confirmed_target)` updates the fitted decoder using a
  complete trial and an externally confirmed target. Pass the chosen target
  as the confirmed target only when that choice has actually been confirmed.

## Recordings and evaluation

Dataset files are separate from installation. Obtain recordings through their
original providers and follow their access and redistribution terms. Do not
commit participant recordings or locally generated caches to a public copy.

The data helpers can prepare supported recordings:

```bash
uv run python helpers/prepare_public_data.py --help
uv run python helpers/prepare_nbrs.py --help
```

The public-data helper accepts an extracted Dehais Ricker directory or can
download the Thielen 2021 cohort with `--thielen`. The NBRS helper requires a
supported archive, its `--group`, and local codebook metadata; see its help
before conversion. Downloads, conversion, and full-cohort evaluation can take
substantial time and disk space.

For authorized reviewers, `../private/grasp_validation.ipynb` contains the recording-based
evaluation. Run it from the repository root after preparing the required local
data. It includes an expensive Thielen 2025 analysis requiring separately
available condition files; the two helpers above do not prepare those files.
To execute the notebook and save a separate result:

```bash
uv run jupyter nbconvert --to notebook --execute grasp_validation.ipynb \
  --output grasp_validation.executed.ipynb --ExecutePreprocessor.timeout=-1
```

Report accuracy together with target count, calibration amount, observation
time, channel selection, and the evaluation split. Compare methods on the
same held-out trials and observation durations. Offline results alone do not
establish performance during live use or natural idle periods.

## Check the checkout

```bash
uv run pytest
uv run ruff check src tests helpers demo
```

Readers with source access can start with the exported `GRASPDecoder` interface
in `src/cvep_decoder/grasp.py` and the usage checks in `tests/test_generic.py`.
The package exports are listed in `src/cvep_decoder/__init__.py`.

If import fails, confirm that installation completed and that the command uses
the same environment. If data loading fails, check the local recording paths;
installing the package does not install datasets. If fitting rejects inputs,
check array order, finite values, target indices, and the number of calibration
trials. If results look implausible, first check stimulus alignment, channel
order, and separation between calibration and test recordings.

## License and sharing

The checkout includes a BSD 3-Clause [license](../LICENSE). Dataset permissions
are separate. This README omits internal design specifications and unpublished
numerical results. Sharing it does not remove sensitive material from source,
notebooks, other documents, or repository history; distribute only files
approved for the intended audience.
