import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="GRASP: a stimulus-agnostic c-VEP decoder",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GRASP

    ### Generic Representation-Adaptive Sequential Prediction for c-VEP decoding

    A c-VEP speller flickers every target with its own pseudo-random code. The decoder's
    job is to work out which code the user is looking at, from a few seconds of EEG over
    the back of the head.

    Every published c-VEP decoder is built around one **event alphabet**: a rule for
    turning a code into the sequence of neural events it should evoke. Different papers
    pick different rules, and each picks one. GRASP builds **eight** alphabets from the
    codebook alone and lets a participant's own calibration data decide how much each is
    worth.

    It is given four things and nothing else: the EEG, the sampling rate, the codebook,
    and the trial labels. No code family, no paradigm, no hand-chosen event type.

    Everything below runs on held-out trials from public recordings, using decoders
    fitted only on that participant's calibration data.
    """)
    return


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    def _locate():
        try:
            here = Path(__file__).resolve().parent
        except NameError:
            here = Path.cwd()
        for candidate in (here, here / "demo", here.parent / "demo"):
            if (candidate / "grasp_demo_data.py").exists():
                return candidate
        return here

    HERE = _locate()
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import grasp_demo_data as gd

    REPO = HERE.parent
    BLUE, GREY, ORANGE, DARK = "#2f6df6", "#aeb8c4", "#d95f02", "#45515e"
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
        }
    )
    return BLUE, DARK, GREY, ORANGE, REPO, gd, np, pd, plt


@app.cell(hide_code=True)
def _(gd, mo):
    pick = mo.ui.dropdown(
        options=list(gd.DATASETS),
        value="Narrow-band Random",
        label="**Dataset**",
    )
    mo.md(
        f"""
    ## Pick a paradigm

    Four public datasets, four different families of stimulus code, from 5 to 40 targets.
    The same decoder, fitted the same way, runs on all of them. Everything on this page
    follows this control.

    {pick}
    """
    )
    return (pick,)


@app.cell(hide_code=True)
def _(gd, mo, pick):
    _missing = not gd.ready(pick.value)
    _building = not gd.cache_path(pick.value).exists() or gd.cache_path(pick.value).stat().st_size == 0

    mo.stop(
        _missing,
        mo.callout(
            mo.md(
                f"""
    **`{pick.value}` is not available.**

    Neither `demo/cache/{gd.cache_path(pick.value).name}` nor the source recording at
    `{gd.SOURCE[pick.value]}` was found. Pick another dataset, or fetch the recording and
    run `python demo/grasp_demo_data.py "{pick.value}"`.
    """
            ),
            kind="warn",
        ),
    )

    fx = gd.load(pick.value)

    _facts = {
        "Targets": f"{fx.n_targets}",
        "Channels": f"{fx.n_channels}",
        "Sampling rate": f"{fx.sample_rate:g} Hz",
        "Trial length": f"{fx.trial_seconds:.2f} s",
        "Calibration": f"{fx.n_calibration_trials} trials, {fx.calibration_seconds:.0f} s",
        "Held-out trials": f"{fx.n_trials}",
        "Chance": f"{fx.chance:.1f} %",
    }
    _rows = "\n".join(f"| {k} | {v} |" for k, v in _facts.items())
    mo.md(
        f"""
    **{pick.value}** &nbsp;·&nbsp; {gd.CITATION[pick.value]} &nbsp;·&nbsp; {fx.subject}

    | | |
    |---|---|
    {_rows}
    """
    )
    return (fx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. The stimulus

    Each target carries its own code. Over a trial the screen replays that code, and the
    visual cortex produces a response roughly time-locked to it. Codes are designed to be
    as unlike one another as possible, so that a few seconds of response is enough to tell
    them apart.

    The block below is the whole codebook. One row per target, time running left to right.
    Walk the selector through the targets to see the code each one carries, and how alike
    it is to every other code in the set. Codes that correlate strongly are the ones the
    decoder will find hardest to tell apart.
    """)
    return


@app.cell(hide_code=True)
def _(fx, mo):
    target_ui = mo.ui.slider(
        start=0, stop=fx.n_targets - 1, step=1, value=int(fx.truth[0]),
        label="**Target**", show_value=True, full_width=True,
    )
    target_ui
    return (target_ui,)


@app.cell(hide_code=True)
def _(BLUE, GREY, ORANGE, fx, np, plt, target_ui):
    _pick = int(target_ui.value)
    _show_s = min(2.0, fx.codebook.shape[1] / fx.sample_rate)
    _n = int(_show_s * fx.sample_rate)

    _fig = plt.figure(figsize=(7.6, 4.6), constrained_layout=True)
    _grid = _fig.add_gridspec(2, 2, height_ratios=[2.4, 1], width_ratios=[2.6, 1])
    _book = _fig.add_subplot(_grid[0, :])
    _code = _fig.add_subplot(_grid[1, 0])
    _near = _fig.add_subplot(_grid[1, 1])

    _book.imshow(
        fx.codebook[:, :_n], aspect="auto", cmap="Greys", interpolation="nearest",
        extent=[0, _show_s, fx.n_targets - 0.5, -0.5],
    )
    _book.axhline(_pick - 0.5, color=BLUE, lw=1.4)
    _book.axhline(_pick + 0.5, color=BLUE, lw=1.4)
    _book.set_ylabel("target")
    _book.set_title(
        f"codebook: {fx.n_targets} codes, first {_show_s:.1f} s", fontsize=9, loc="left"
    )
    _book.grid(False)
    _book.set_xticklabels([])

    _t = np.arange(_n) / fx.sample_rate
    _code.step(_t, fx.codebook[_pick, :_n], where="post", color=BLUE, lw=1.2)
    _code.set_xlim(0, _show_s)
    _code.set_xlabel("time (s)")
    _code.set_ylabel("level")
    _code.set_title(f"code {_pick}", fontsize=9, loc="left")

    # How much this code looks like each of the others, at zero lag.
    _centred = fx.codebook - fx.codebook.mean(axis=1, keepdims=True)
    _norm = np.maximum(np.linalg.norm(_centred, axis=1), 1e-12)
    _similarity = (_centred @ _centred[_pick]) / (_norm * _norm[_pick])
    _others = np.delete(np.arange(fx.n_targets), _pick)
    _worst = _others[np.argmax(np.abs(_similarity[_others]))]
    _colors = [ORANGE if abs(_similarity[i]) >= 0.3 else GREY for i in _others]
    _near.bar(np.arange(len(_others)), _similarity[_others], color=_colors, width=1.0)
    _near.axhline(0, color="#bbb", lw=0.6)
    _near.set_ylim(-1, 1)
    _near.set_xlabel("every other code")
    _near.set_ylabel("correlation")
    _near.set_title(
        f"closest: code {_worst} (r = {_similarity[_worst]:+.2f})", fontsize=8, loc="left"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. One code, eight descriptions

    Here is the idea the whole decoder rests on. A code can be described as a sequence of
    events in more than one way, and the choice matters: a response locked to *rising
    edges* is not the response locked to *sustained level*.

    Published decoders commit to one description. Below are the ones GRASP derives from
    the code you selected above. Each becomes its own predicted response, fitted
    separately.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, DARK, ORANGE, fx, gd, np, plt, target_ui):
    _attended = int(target_ui.value)
    _show_s = min(1.2, fx.codebook.shape[1] / fx.sample_rate)
    _n = int(_show_s * fx.sample_rate)
    _t = np.arange(_n) / fx.sample_rate

    _named = [a for a in gd.RESPONSE_ALPHABETS if a in fx.events]
    _fig, _axes = plt.subplots(
        len(_named) + 1, 1, figsize=(7.4, 0.62 * (len(_named) + 1) + 0.8), sharex=True,
        constrained_layout=True,
    )
    _axes[0].step(_t, fx.codebook[_attended, :_n], where="post", color=DARK, lw=1.1)
    _axes[0].set_ylabel("code", rotation=0, ha="right", va="center", fontsize=8)
    _axes[0].set_yticks([])
    _axes[0].set_title("the code, then the event trains each alphabet predicts from it",
                       fontsize=9, loc="left")

    for _axis, _name in zip(_axes[1:], _named):
        _matrix, _events = gd.event_train(fx.codebook[_attended : _attended + 1], _name, fx.events)
        for _e in range(_matrix.shape[1]):
            _axis.step(
                _t,
                _matrix[0, _e, :_n] + 1.25 * _e,
                where="post",
                lw=1.1,
                color=[BLUE, ORANGE, DARK][_e % 3],
            )
        _axis.set_ylabel(gd.ALPHABET_LABEL[_name], rotation=0, ha="right", va="center", fontsize=8)
        _axis.set_yticks([])
    _axes[-1].set_xlabel("time (s)")
    _axes[-1].set_xlim(0, _show_s)
    _fig
    return


@app.cell(hide_code=True)
def _(gd, mo):
    mo.md(f"""
    Two of the eight are different in kind and have no event train to draw:

    - **{gd.ALPHABET_LABEL['spectral_components']}** &mdash; {gd.ALPHABET_NOTE['spectral_components']}
      It is the one component of the decoder that never sees EEG, so it works on a code
      it has never been calibrated against.
    - **{gd.ALPHABET_LABEL['transition_likelihood']}** &mdash; {gd.ALPHABET_NOTE['transition_likelihood']}
      Because it learns one event and one no-event response rather than a template per
      target, calibration does not have to contain every target.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. What calibration learns

    For each alphabet the decoder fits two things from the participant's calibration
    trials: a **spatial filter** that combines the channels, and a **temporal response**,
    the shape the brain makes for one event of that kind. Predicting a code's response is
    then a matter of convolving its event train with that shape.

    These are fitted per participant. Nothing here is shared between people or datasets.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, DARK, fx, gd, np, plt):
    _named = [a for a in gd.RESPONSE_ALPHABETS if a in fx.responses]
    _fig, _axes = plt.subplots(2, len(_named), figsize=(7.6, 3.0), constrained_layout=True)
    for _col, _name in enumerate(_named):
        _r = fx.responses[_name]
        _w = fx.filters[_name]
        _t = np.arange(len(_r)) / fx.sample_rate * 1000
        _axes[0, _col].plot(_t, _r, color=BLUE, lw=1.2)
        _axes[0, _col].axhline(0, color="#ccc", lw=0.6)
        _axes[0, _col].set_title(gd.ALPHABET_LABEL[_name], fontsize=8)
        _axes[0, _col].set_xlabel("ms", fontsize=7)
        _axes[0, _col].tick_params(labelsize=7)
        _axes[1, _col].bar(np.arange(len(_w)), _w, color=DARK, width=0.75)
        _axes[1, _col].axhline(0, color="#ccc", lw=0.6)
        _axes[1, _col].set_xlabel("channel", fontsize=7)
        _axes[1, _col].tick_params(labelsize=7)
        if _col:
            _axes[0, _col].set_yticklabels([])
            _axes[1, _col].set_yticklabels([])
    _axes[0, 0].set_ylabel("response", fontsize=8)
    _axes[1, 0].set_ylabel("filter weight", fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. One trial, end to end

    Everything so far has been a piece of the machine. Here is the whole path, for a
    single trial, in the order the decoder walks it: EEG arrives, the spatial filter
    collapses the channels to one signal, each candidate code is turned into a predicted
    response, the two are compared, the eight comparisons are weighted, and the largest
    total wins.

    Panels 3 and 4 are the heart of it. The same measured signal is drawn twice, against
    what the attended code predicts and against what the runner-up predicts. The decoder
    picks whichever prediction fits better.
    """)
    return


@app.cell(hide_code=True)
def _(fx, gd, mo):
    flow_trial_ui = mo.ui.slider(
        start=0, stop=fx.n_trials - 1, step=1, value=0,
        label="**Trial**", show_value=True,
    )
    flow_window_ui = mo.ui.slider(
        steps=[float(t) for t in fx.times], value=float(fx.times[-1]),
        label="**Window (s)**", show_value=True,
    )
    flow_alpha_ui = mo.ui.dropdown(
        options={gd.ALPHABET_LABEL[a]: a for a in fx.responses},
        value=gd.ALPHABET_LABEL[fx.lead_alphabet],
        label="**Alphabet to draw**",
    )
    mo.hstack([flow_trial_ui, flow_window_ui, flow_alpha_ui], justify="start", gap=2)
    return flow_alpha_ui, flow_trial_ui, flow_window_ui


@app.cell(hide_code=True)
def _(
    BLUE,
    DARK,
    GREY,
    ORANGE,
    flow_alpha_ui,
    flow_trial_ui,
    flow_window_ui,
    fx,
    gd,
    np,
    plt,
):
    flow = gd.walkthrough(
        fx, int(flow_trial_ui.value), float(flow_window_ui.value), flow_alpha_ui.value
    )
    _draw_s = min(1.2, flow["n_samples"] / fx.sample_rate)
    _m = int(_draw_s * fx.sample_rate)
    _t = np.arange(_m) / fx.sample_rate

    def _unit(values):
        values = np.asarray(values, float)[:_m]
        centred = values - values.mean()
        return centred / max(np.std(centred), 1e-12)

    _fig, _axes = plt.subplots(2, 3, figsize=(9.6, 4.8), constrained_layout=True)

    # 1. the recording
    _raw = flow["raw"][:_m]
    _raw = (_raw - _raw.mean(axis=0)) / np.maximum(_raw.std(axis=0), 1e-12)
    for _c in range(fx.n_channels):
        _axes[0, 0].plot(_t, _raw[:, _c] - _c * 4.0, lw=0.5, color=DARK, alpha=0.85)
    _axes[0, 0].set_yticks([])
    _axes[0, 0].set_xlabel("time (s)")
    _axes[0, 0].set_title(f"1. EEG in, {fx.n_channels} channels", fontsize=9, loc="left")

    # 2. the spatial filter
    _axes[0, 1].bar(np.arange(fx.n_channels), flow["spatial_filter"], color=DARK, width=0.75)
    _axes[0, 1].axhline(0, color="#ccc", lw=0.6)
    _axes[0, 1].set_xlabel("channel")
    _axes[0, 1].set_ylabel("weight")
    _axes[0, 1].set_title("2. spatial filter  w", fontsize=9, loc="left")

    # 3. measured against the attended code's prediction
    _measured = _unit(flow["measured"])
    _hit = _unit(flow["predicted_attended"])
    _miss = _unit(flow["predicted_runner_up"])
    _r_hit = float(np.corrcoef(_measured, _hit)[0, 1])
    _r_miss = float(np.corrcoef(_measured, _miss)[0, 1])

    _axes[0, 2].plot(_t, _measured, color=BLUE, lw=1.0, label="measured")
    _axes[0, 2].plot(_t, _hit, color=ORANGE, lw=1.0, ls="--", label="predicted")
    _axes[0, 2].set_xlabel("time (s)")
    _axes[0, 2].set_title(
        f"3. cued code {flow['attended']}:  r = {_r_hit:+.2f}", fontsize=9, loc="left"
    )
    _axes[0, 2].legend(frameon=False, fontsize=7, loc="upper right")
    _axes[0, 2].set_yticks([])

    # 4. the same signal against the closest competitor
    _axes[1, 0].plot(_t, _measured, color=BLUE, lw=1.0, label="measured")
    _axes[1, 0].plot(_t, _miss, color=GREY, lw=1.0, ls="--", label="predicted")
    _axes[1, 0].set_xlabel("time (s)")
    _axes[1, 0].set_title(
        f"4. runner-up {flow['runner_up']}:  r = {_r_miss:+.2f}", fontsize=9, loc="left"
    )
    _axes[1, 0].legend(frameon=False, fontsize=7, loc="upper right")
    _axes[1, 0].set_yticks([])

    # 5. every alphabet's verdict
    _x = np.arange(len(gd.ALPHABETS))
    _axes[1, 1].bar(_x - 0.2, flow["per_alphabet_attended"], width=0.38,
                    color=BLUE, label=f"code {flow['attended']} (cued)")
    _axes[1, 1].bar(_x + 0.2, flow["per_alphabet_runner_up"], width=0.38,
                    color=GREY, label=f"code {flow['runner_up']}")
    _axes[1, 1].axhline(0, color="#ccc", lw=0.6)
    _axes[1, 1].set_xticks(_x)
    _axes[1, 1].set_xticklabels(
        [gd.ALPHABET_LABEL[a][:9] for a in gd.ALPHABETS], rotation=60, ha="right", fontsize=6
    )
    _axes[1, 1].set_title("5. evidence per alphabet", fontsize=9, loc="left")
    _axes[1, 1].legend(frameon=False, fontsize=7)

    # 6. weighted total over every candidate
    _fused = flow["fused"]
    _bar_colors = [GREY] * fx.n_targets
    _bar_colors[flow["leader"]] = ORANGE
    _bar_colors[flow["attended"]] = BLUE
    _axes[1, 2].bar(np.arange(fx.n_targets), _fused, color=_bar_colors, width=0.85)
    _axes[1, 2].set_xlabel("candidate code")
    _axes[1, 2].set_title(
        "6. fused, take the largest  " + ("✓" if flow["correct"] else "✗"),
        fontsize=9, loc="left",
    )
    _fig
    return (flow,)


@app.cell(hide_code=True)
def _(flow, gd, mo):
    mo.md(f"""
    At **{flow['window']:.2f} s** the decoder settles on code **{flow['leader']}**; the cued
    code was **{flow['attended']}**, so this trial is
    **{'correct' if flow['correct'] else 'wrong'}**.

    The correlations printed in panels 3 and 4 are plain Pearson correlations between the
    drawn traces, shown to make the comparison visible. The score the decoder actually
    uses, equation (1), differs in three ways: the response is filtered as well as the
    EEG, the correlation is Fisher-transformed so alphabets working at different
    correlation levels can be added, and the result is standardised across candidates.
    Panel 5 shows those finished scores, and panel 6 the weighted sum of them.

    Only the **{gd.ALPHABET_LABEL[flow['alphabet']]}** alphabet is drawn in panels 3 and 4,
    because a single picture cannot hold eight predicted responses at once. Panel 5 is
    where the other seven appear, and the eighth, the filter bank, never forms a predicted
    response in the first place.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Watching one trial being decoded

    Now the decoder runs. Evidence accumulates as EEG arrives; the scores for all
    candidate codes are updated at each decision time. GRASP does not wait for a fixed
    trial length. It watches the gap between the leading candidate and the runner-up, and
    a calibrated model turns that gap plus the elapsed fraction of the trial into an
    estimate of its own probability of being right. When that estimate clears a threshold
    and the leader has held steady, it commits.

    Move the window slider and watch the evidence separate. Turn alphabets off to see
    which ones this participant was actually relying on.
    """)
    return


@app.cell(hide_code=True)
def _(fx, gd, mo, np):
    trial_ui = mo.ui.slider(
        start=0, stop=fx.n_trials - 1, step=1, value=0, label="**Held-out trial**",
        show_value=True, full_width=True,
    )
    window_ui = mo.ui.slider(
        steps=[float(t) for t in fx.times],
        value=float(fx.times[min(len(fx.times) - 1, len(fx.times) // 2)]),
        label="**Observation window (s)**",
        show_value=True,
        full_width=True,
    )
    alpha_ui = mo.ui.multiselect(
        options={gd.ALPHABET_LABEL[a]: a for a in gd.ALPHABETS},
        value=[gd.ALPHABET_LABEL[a] for a in np.asarray(gd.ALPHABETS)[fx.weights > 1e-6]],
        label="**Alphabets in use**",
    )
    mo.vstack([trial_ui, window_ui, alpha_ui])
    return alpha_ui, trial_ui, window_ui


@app.cell(hide_code=True)
def _(alpha_ui, fx, gd, mo, np, trial_ui):
    _selected = set(alpha_ui.value)
    _mask = np.array([a in _selected for a in gd.ALPHABETS], float)
    mo.stop(
        _mask.sum() == 0,
        mo.callout(mo.md("Select at least one alphabet."), kind="warn"),
    )

    # Keep the fitted proportions among whichever alphabets are still switched on.
    live_weights = fx.weights * _mask
    if live_weights.sum() <= 0:
        live_weights = _mask / _mask.sum()
    else:
        live_weights = live_weights / live_weights.sum()

    live_trace = fx.trial_trace(int(trial_ui.value), live_weights)
    live_truth = int(fx.truth[int(trial_ui.value)])
    return live_trace, live_truth, live_weights


@app.cell(hide_code=True)
def _(
    BLUE,
    DARK,
    GREY,
    ORANGE,
    fx,
    live_trace,
    live_truth,
    np,
    plt,
    window_ui,
):
    _k = int(np.argmin(np.abs(fx.times - float(window_ui.value))))
    _fused = live_trace["fused"]
    _emit = live_trace["emitted_at"]

    _fig, _axes = plt.subplots(
        1, 3, figsize=(9.4, 2.9), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1, 1]},
    )

    # -- candidate scores at the chosen window ---------------------------------
    _scores = _fused[_k]
    _colors = [GREY] * fx.n_targets
    _colors[int(np.argmax(_scores))] = ORANGE
    _colors[live_truth] = BLUE
    _axes[0].bar(np.arange(fx.n_targets), _scores, color=_colors, width=0.85)
    _axes[0].set_xlabel("candidate code")
    _axes[0].set_ylabel("fused evidence  S(c, t)")
    _axes[0].set_title(
        f"t = {fx.times[_k]:.2f} s   (blue = cued, orange = leader)", fontsize=9, loc="left"
    )

    # -- the top-two race ------------------------------------------------------
    _sorted = np.sort(_fused, axis=1)
    _axes[1].plot(fx.times, _sorted[:, -1], color=ORANGE, lw=1.6, label="leader")
    _axes[1].plot(fx.times, _sorted[:, -2], color=GREY, lw=1.4, label="runner-up")
    _axes[1].plot(fx.times, _fused[:, live_truth], color=BLUE, lw=1.2, ls=":", label="cued code")
    _axes[1].fill_between(fx.times, _sorted[:, -2], _sorted[:, -1], color=ORANGE, alpha=0.12)
    _axes[1].axvline(fx.times[_k], color=DARK, lw=0.9, ls="--")
    _axes[1].set_xlabel("observation window (s)")
    _axes[1].set_ylabel("evidence")
    _axes[1].set_title("margin over the runner-up", fontsize=9, loc="left")
    _axes[1].legend(frameon=False, fontsize=7, loc="lower right")

    # -- the stopping rule -----------------------------------------------------
    _axes[2].plot(fx.times, live_trace["q"], color=BLUE, lw=1.6)
    _axes[2].axhline(fx.stop_threshold, color=DARK, lw=1.0, ls="--")
    _axes[2].text(
        fx.times[0], fx.stop_threshold + 0.03, "threshold", fontsize=7, color=DARK
    )
    _axes[2].scatter(
        [fx.times[_emit]], [live_trace["q"][_emit]], s=46, zorder=5,
        color="white", edgecolor=ORANGE, linewidth=1.8,
    )
    _axes[2].axvline(fx.times[_k], color=DARK, lw=0.9, ls="--")
    _axes[2].set_ylim(0, 1.05)
    _axes[2].set_xlabel("observation window (s)")
    _axes[2].set_ylabel("q(t), estimated correctness")
    _axes[2].set_title("when it commits", fontsize=9, loc="left")
    _fig
    return


@app.cell(hide_code=True)
def _(fx, gd, live_trace, live_truth, mo, np, pick, window_ui):
    _k = int(np.argmin(np.abs(fx.times - float(window_ui.value))))
    _now = int(live_trace["winner"][_k])
    _hit = "correct" if live_trace["selection"] == live_truth else "wrong"
    _icon = "✓" if live_trace["selection"] == live_truth else "✗"
    _how = "forced at the end of the trial" if live_trace["forced"] else "committed early"
    _saving = fx.trial_seconds - live_trace["selection_time"]

    _curve = fx.accuracy_curve()
    _matched_k = int(np.argmin(np.abs(fx.times - gd.MATCHED_WINDOW[pick.value])))

    mo.md(
        f"""
    | | |
    |---|---|
    | Leader at {fx.times[_k]:.2f} s | code **{_now}** {'(cued)' if _now == live_truth else f'(cued was {live_truth})'} |
    | Selection | code **{live_trace['selection']}** &nbsp; {_icon} {_hit} |
    | Emitted at | **{live_trace['selection_time']:.2f} s**, {_how} |
    | Saved against the full trial | {_saving:.2f} s of {fx.trial_seconds:.2f} s |

    Across all {fx.n_trials} held-out trials of this participant, the fitted decoder reaches
    **{_curve[_matched_k]:.1f} %** at the {gd.MATCHED_WINDOW[pick.value]:.2f} s window used
    for the published comparison, against a chance level of {fx.chance:.1f} %.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Which alphabet mattered?

    The fusion weights are fitted per participant, on scores the response models produced
    for calibration blocks held out from their own fitting, so no trial ever contributes
    to the model that scores it. No single weight is allowed above 0.55, which stops one
    description from taking over.

    Switch datasets and watch the weights move. That movement is the point: the
    informative description of a c-VEP stimulus is not a constant of the paradigm, and
    fixing it in advance throws away whatever the other descriptions knew.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, GREY, fx, gd, live_weights, np, plt):
    _x = np.arange(len(gd.ALPHABETS))
    _labels = [gd.ALPHABET_LABEL[a] for a in gd.ALPHABETS]
    _fig, _axis = plt.subplots(figsize=(7.4, 2.5), constrained_layout=True)
    _axis.bar(_x - 0.2, fx.weights, width=0.38, color=BLUE, label="as fitted")
    _axis.bar(_x + 0.2, live_weights, width=0.38, color=GREY, label="as selected above")
    _axis.axhline(0.55, color="#d95f02", lw=1.0, ls="--")
    _axis.text(len(_x) - 0.6, 0.57, "cap 0.55", fontsize=7, color="#d95f02", ha="right")
    _axis.set_xticks(_x)
    _axis.set_xticklabels(_labels, rotation=28, ha="right", fontsize=8)
    _axis.set_ylabel("fusion weight")
    _axis.set_ylim(0, 0.7)
    _axis.legend(frameon=False, fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _(gd, mo):
    _installed = sorted(gd.embedders())
    _optional = [m for m in _installed if m not in
                 ("PCA", "t-SNE", "Isomap", "Spectral embedding", "MDS")]
    _extra = (
        f"Also available here: **{', '.join(_optional)}**."
        if _optional
        else "Only the scikit-learn methods are installed. "
             "`pip install umap-learn pacmap trimap openTSNE` adds UMAP, PaCMAP, "
             "LocalMAP, TriMap and openTSNE, and they appear in the menu automatically."
    )
    mo.md(
        f"""
    ---
    ## 7. Where the separation comes from

    The decoder works because the data has structure the codes can be read out of. Three
    different slices of it are worth looking at in two dimensions, and they answer
    different questions.

    {_extra}

    A note on what is being asked. Embedding whole trials and colouring by target only
    shows clusters when a dataset gives several trials per target, which most of these do
    not. The structure the decoder actually exploits lives at the level of **events**, not
    trials, which is why the first option below is the informative one.
    """
    )
    return


@app.cell(hide_code=True)
def _(gd, mo):
    embed_feature_ui = mo.ui.dropdown(
        options=list(gd.FEATURE_SETS),
        value="Candidate evidence",
        label="**What to embed**",
    )
    embed_method_ui = mo.ui.dropdown(
        options=sorted(gd.embedders()), value="PCA", label="**Method**"
    )
    embed_seed_ui = mo.ui.number(start=0, stop=999, step=1, value=0, label="**Seed**")
    mo.vstack(
        [
            mo.hstack([embed_feature_ui, embed_method_ui, embed_seed_ui],
                      justify="start", gap=2),
            mo.md(
                "*PCA is instant. t-SNE, UMAP and the rest take a few seconds on a few "
                "thousand points.*"
            ),
        ]
    )
    return embed_feature_ui, embed_method_ui, embed_seed_ui


@app.cell(hide_code=True)
def _(
    BLUE,
    GREY,
    ORANGE,
    embed_feature_ui,
    embed_method_ui,
    embed_seed_ui,
    fx,
    gd,
    mo,
    plt,
):
    _features, _labels, _legend, _kind = gd.embedding_inputs(fx, embed_feature_ui.value)

    mo.stop(
        len(_features) < 8,
        mo.callout(
            mo.md(
                f"""
    **{embed_feature_ui.value}** yields only {len(_features)} points for this dataset,
    too few to embed. Try another slice, or a dataset with more held-out trials.
    """
            ),
            kind="info",
        ),
    )

    _points = gd.embed(_features, embed_method_ui.value, int(embed_seed_ui.value))

    _fig, _axis = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    if _kind == "target":
        _scatter = _axis.scatter(
            _points[:, 0], _points[:, 1], c=_labels, cmap="turbo", s=34,
            edgecolor="white", linewidth=0.4,
        )
        _fig.colorbar(_scatter, ax=_axis, label="attended target")
    else:
        _palette = [BLUE, ORANGE, GREY] if _kind == "event" else [GREY, BLUE]
        _order = [2, 0, 1] if _kind == "event" else [0, 1]
        for _index in _order:
            _mask = _labels == _index
            if not _mask.any():
                continue
            _axis.scatter(
                _points[_mask, 0], _points[_mask, 1],
                s=12 if _kind == "event" else 9,
                color=_palette[_index % len(_palette)],
                alpha=0.45 if (_kind == "event" and _index == 2) else 0.7,
                linewidth=0, label=_legend[_index],
            )
        _axis.legend(frameon=False, fontsize=8, markerscale=2, loc="best")

    _axis.set_xticks([])
    _axis.set_yticks([])
    _axis.set_xlabel(f"{embed_method_ui.value} dimension 1")
    _axis.set_ylabel(f"{embed_method_ui.value} dimension 2")
    _axis.set_title(
        f"{embed_feature_ui.value}: {len(_features)} points, {_features.shape[1]} features",
        fontsize=9, loc="left",
    )
    embedding_shape = (len(_features), int(_features.shape[1]))
    embedding_score = gd.separability(_features, _labels)
    _fig
    return embedding_score, embedding_shape


@app.cell(hide_code=True)
def _(embed_feature_ui, embedding_score, embedding_shape, gd, mo):
    _probe = embedding_score["probe"]
    _verdict = (
        "too few points per class to measure"
        if _probe is None
        else f"**{100 * _probe:.0f} %** against **{100 * embedding_score['majority']:.0f} %** "
             f"for always guessing the largest class "
             f"({embedding_score['folds']}-fold cross-validated)"
    )
    mo.md(
        f"""
    **{embed_feature_ui.value}** &nbsp;·&nbsp; {embedding_shape[0]} points in
    {embedding_shape[1]} dimensions &nbsp;·&nbsp; a linear rule on the *full* feature
    space reaches {_verdict}.

    {gd.FEATURE_SETS[embed_feature_ui.value]}

    Two cautions. Every method here is unsupervised but the colouring is not, so apparent
    separation should be read as *the classes are separable in the input space*, not as
    *the method discovered them*. And neighbour-embedding methods place points by local
    neighbourhoods, so distances between far-apart clusters, and the size of the gaps
    between them, carry no meaning. Changing the seed will move things around; if a
    structure survives that, it is probably real.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. Swapping the codebook underneath a fitted decoder

    If a decoder really is stimulus-agnostic, then a model fitted on one code family
    should still work when the codes are replaced with ones it has never seen, without
    refitting a single response.

    That is testable. On the binary sequence set each participant saw ten code conditions
    from five families. Calibrate on one, replace the codebook with another family's, test
    on that family's trials.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, GREY, fx, gd, mo, np, pick, plt):
    if fx.swap_views is None:
        _out = mo.callout(
            mo.md(
                f"""
    This test needs the binary sequence set, which is the only dataset here that records
    the same participant under several code families. Currently showing **{pick.value}**.

    Switch the dataset control above to **Binary Sequence Set** to run it.
    """
            ),
            kind="info",
        )
    else:
        _own = fx.accuracy_curve()
        _swapped = fx.accuracy_curve(views=fx.swap_views, truth=fx.swap_truth)
        _fig, _axis = plt.subplots(figsize=(7.0, 2.6), constrained_layout=True)
        _axis.plot(fx.times, _own, color=BLUE, lw=1.8, marker="o", ms=3,
                   label="own codebook (calibrated family)")
        _axis.plot(fx.times, _swapped, color=GREY, lw=1.8, marker="s", ms=3,
                   label=f"codebook replaced with '{fx.swap_family}', no refitting")
        _axis.axhline(fx.chance, color="#777", ls=(0, (2, 2)), lw=0.9)
        _axis.text(fx.times[0], fx.chance + 3, f"chance {fx.chance:.1f}%",
                   fontsize=7, color="#777", ha="left")
        _axis.set_xlabel("observation window (s)")
        _axis.set_ylabel("accuracy (%)")
        _axis.set_ylim(0, 105)
        _axis.legend(frameon=False, fontsize=8, loc="lower right")

        _k = int(np.argmin(np.abs(fx.times - gd.MATCHED_WINDOW[pick.value])))
        _out = mo.vstack(
            [
                _fig,
                mo.md(
                    f"""
    At the full {fx.times[_k]:.2f} s trial this participant reaches **{_own[_k]:.1f} %** on the
    family it was calibrated on and **{_swapped[_k]:.1f} %** after the codebook was swapped
    for one from another family, with no response model refitted. That is one fold of
    {fx.n_trials} trials, so it moves around; the cohort figure is in section 7.

    The pair shown is not chosen to flatter. Of the 80 ordered cross-family pairs, this
    one's 26-participant mean is the closest of any to the cross-family average, so it is
    a representative case rather than a best or worst one. One participant is still an
    anecdote: section 9 has the whole matrix.
    """
                ),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 9. How it does, across every participant

    Everything above was one person. These are the full cohorts, each compared against a
    re-implementation of the decoder published with that dataset, scored on the same
    trials, the same channels and the same observation window.
    """)
    return


@app.cell(hide_code=True)
def _(REPO, gd, mo, pd):
    _path = REPO / "data" / "validation_results.csv"
    if not _path.exists():
        cohort = None
        cohort_note = mo.callout(
            mo.md(
                f"""
    `{_path.relative_to(REPO)}` was not found, so the cohort results cannot be shown.
    Run `grasp_validation_new.ipynb` to produce it.
    """
            ),
            kind="warn",
        )
    else:
        _raw = pd.read_csv(_path)
        _grasp, _base = "GRASP (matched duration)", {
            "Dehais 2024": "xDAWN-TS-LDA reconstruction",
            "NBRS-15": "paper-specified FBCCA reconstruction",
            "Thielen 2021": "authors' PyntBCI n-train rCCA",
            "Thielen 2025": "Thielen rCCA reference",
        }
        _rows = []
        for _display in gd.DATASETS:
            _name = gd.INTERNAL_NAME[_display]
            _frame = _raw[_raw.dataset == _name]
            _a = _frame[_frame.method == _grasp].groupby("subject").accuracy.mean()
            _b = _frame[_frame.method == _base[_name]].groupby("subject").accuracy.mean()
            _shared = _a.index.intersection(_b.index)
            _a, _b = _a.loc[_shared], _b.loc[_shared]
            _rows.append(
                {
                    "Dataset": _display,
                    "Participants": len(_shared),
                    "Targets": int(_frame.targets.iloc[0]),
                    "Window (s)": gd.MATCHED_WINDOW[_display],
                    "Published decoder (%)": 100 * _b.mean(),
                    "GRASP (%)": 100 * _a.mean(),
                    "Difference (points)": 100 * (_a - _b).mean(),
                }
            )
        cohort = pd.DataFrame(_rows)
        cohort_note = mo.md(
            f"""
    GRASP is more accurate on all four, by {cohort['Difference (points)'].min():.1f} to
    {cohort['Difference (points)'].max():.1f} points. The binary sequence set row is the
    hard case: there both decoders are tested on a codebook from a *different family* than
    the one they were calibrated on.
    """
        )

    mo.vstack(
        [
            mo.ui.table(
                cohort.round(2), selection=None, page_size=8
            ) if cohort is not None else mo.md(""),
            cohort_note,
        ]
    )
    return


@app.cell(hide_code=True)
def _(BLUE, GREY, REPO, gd, mo, pd, pick, plt):
    _path = REPO / "data" / "cache" / "curves.csv"
    if not _path.exists():
        _path = REPO / "data" / "cache" / "time_curves.csv"
    if not _path.exists():
        _curve_out = mo.md("")
    else:
        _curves = pd.read_csv(_path)
        _name = gd.INTERNAL_NAME[pick.value]
        _frame = _curves[_curves.dataset == _name]
        _baseline_method = [
            m for m in _frame.method.unique() if not m.startswith("GRASP")
        ]
        _g = _frame[_frame.method == "GRASP (matched duration)"].groupby("time_s").accuracy.mean()
        _b = _frame[_frame.method == _baseline_method[0]].groupby("time_s").accuracy.mean()
        _fig, _axis = plt.subplots(figsize=(7.0, 2.6), constrained_layout=True)
        _axis.plot(_g.index, 100 * _g.values, color=BLUE, lw=1.8, marker="o", ms=3.5,
                   mfc="white", label="GRASP")
        _axis.plot(_b.index, 100 * _b.values, color=GREY, lw=1.6, marker="s", ms=3.5,
                   mfc="white", ls="--", label=gd.BASELINE_NAME[pick.value])
        _axis.axhline(100 / _frame.targets.iloc[0], color="#777", ls=(0, (2, 2)), lw=0.9)
        _axis.set_xlabel("observation window (s)")
        _axis.set_ylabel("accuracy (%)")
        _axis.set_ylim(0, 105)
        _axis.set_title(
            f"{pick.value}: {_frame.subject.nunique()} participants", fontsize=9, loc="left"
        )
        _axis.legend(frameon=False, fontsize=8, loc="lower right")
        _curve_out = _fig
    _curve_out
    return


@app.cell(hide_code=True)
def _(REPO, mo, np, plt):
    _folder = REPO / "data" / "cache" / "thielen2025"
    _files = sorted(_folder.glob("sub-*.npz")) if _folder.exists() else []
    if not _files:
        _matrix_out = mo.md("")
    else:
        _conditions = ["mseq", "m-mseq", "debruijn", "m-debruijn", "golay",
                       "m-golay", "gold", "m-gold", "gold-set", "m-gold-set"]
        _stack = np.stack([np.load(f, allow_pickle=False)["grasp"] for f in _files], axis=2)
        _mean = 100 * np.nanmean(_stack, axis=2)
        _fig, _axis = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
        _image = _axis.imshow(_mean, cmap="RdYlGn", vmin=25, vmax=100)
        for _i in range(10):
            for _j in range(10):
                _axis.text(_j, _i, f"{_mean[_i, _j]:.0f}", ha="center", va="center", fontsize=6)
        for _edge in range(2, 10, 2):
            _axis.axhline(_edge - 0.5, color="black", lw=0.6)
            _axis.axvline(_edge - 0.5, color="black", lw=0.6)
        _axis.set_xticks(range(10), _conditions, rotation=55, ha="right", fontsize=7)
        _axis.set_yticks(range(10), _conditions, fontsize=7)
        _axis.set_xlabel("codebook it was tested on")
        _axis.set_ylabel("codebook it was calibrated on")
        _axis.set_title(f"held-out codebook, {len(_files)} participants", fontsize=9, loc="left")
        _axis.grid(False)
        _fig.colorbar(_image, ax=_axis, label="accuracy (%)", shrink=0.85)
        # Group the 100 ordered pairs the way the manuscript does.
        _family = np.array([c[2:] if c.startswith("m-") else c for c in _conditions])
        _same = _family[:, None] == _family[None, :]
        _diagonal = np.eye(10, dtype=bool)
        _groups = {
            "same condition (control)": _diagonal,
            "other member of the same family": _same & ~_diagonal,
            "a different family": ~_same,
        }
        _summary = {
            _label: 100 * np.nanmean(np.nanmean(_stack, axis=2)[_mask])
            for _label, _mask in _groups.items()
        }
        _rows = "\n".join(
            f"| {_label} | {int(_groups[_label].sum())} | {_value:.1f} % |"
            for _label, _value in _summary.items()
        )
        _matrix_out = mo.vstack(
            [
                _fig,
                mo.md(
                    f"""
    The diagonal is the control: calibrated and tested on the same condition. Every cell
    off it is a codebook the fitted decoder had never seen, and it is still well above the
    3.1 % chance level for 32 targets.

    | tested on | pairs | accuracy |
    |---|---|---|
    {_rows}

    The middle row is the surprise. Swapping a code for the *modulated version of its own
    family* costs more than crossing to a different family altogether, which points at
    modulation depth rather than family membership as what a fitted model fails to carry
    over.
    """
                ),
            ]
        )
    _matrix_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## What this does and does not show

    Everything here is **offline**, replayed from recordings. No one in these plots was
    controlling anything in real time, and the comparators are our re-implementations of
    published decoders rather than the original online systems.

    Each dataset holds one session per participant, so the codebook-replacement result is
    a within-session result. Transfer across sessions, or from one person's fitted weights
    to another's stimulus, is untested.

    The decoder also carries an idle gate that withholds output when the user is not
    trying to select. No public c-VEP dataset records genuine non-control periods, so it
    was evaluated against cued pre-stimulus rest, which is a proxy and almost certainly
    optimistic.

    ---

    *Rebuild the cached evidence with `python demo/grasp_demo_data.py`. Full analysis in
    `grasp_validation_new.ipynb`.*
    """)
    return


if __name__ == "__main__":
    app.run()
