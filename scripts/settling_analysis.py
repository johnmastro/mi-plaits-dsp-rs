"""Deeper analysis of the trigger-settling listening artifacts.

Reads the four WAVs per case from out/diagnostics/trigger-settling, recovers the
raw relative gain between immediate_same and immediate_prepared from the
boosted residual, and reports level, timing, pitch, and spectral-shape
differences over time. Writes PNGs and a markdown summary to
out/diagnostics/trigger-settling/analysis.
"""

import os
import sys
import glob
import numpy as np
from scipy.io import wavfile
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 48000
BLOCK = 24
SRC = "out/diagnostics/trigger-settling"
DST = os.path.join(SRC, "analysis")
os.makedirs(DST, exist_ok=True)
EPS = 1e-12


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), EPS))


def read(path):
    sr, d = wavfile.read(path)
    assert sr == SR
    return d.astype(np.float64)


def block_rms_db(x):
    n = len(x) // BLOCK
    b = x[: n * BLOCK].reshape(n, BLOCK)
    return np.maximum(db(np.sqrt((b ** 2).mean(axis=1))), -100.0)


def recover_gain_ratio(S, P, R):
    """R = a*S - b*P (least squares over both channels). Returns a, b, fit error."""
    A = np.column_stack([S.ravel(), -P.ravel()])
    coef, *_ = np.linalg.lstsq(A, R.ravel(), rcond=None)
    a, b = coef
    fit = A @ coef
    err = np.sqrt(((R.ravel() - fit) ** 2).mean()) / (np.sqrt((R.ravel() ** 2).mean()) + EPS)
    return a, b, err


def best_lag(x, y, max_lag=120):
    """Lag (samples) that best aligns y to x, by normalized cross-correlation."""
    n = min(len(x), len(y))
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    c = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    m = np.abs(lags) <= max_lag
    c = c[m]
    lags = lags[m]
    norm = np.sqrt((x ** 2).sum() * (y ** 2).sum()) + EPS
    i = np.argmax(c)
    return int(lags[i]), float(c[i] / norm)


def shifted(y, lag):
    out = np.zeros_like(y)
    if lag > 0:
        out[lag:] = y[:-lag]
    elif lag < 0:
        out[:lag] = y[-lag:]
    else:
        out[:] = y
    return out


def pitch_track(x, win=1024, hop=120, fmin=50.0, fmax=4000.0, tmax_s=0.15):
    """Autocorrelation pitch track. Returns (times_ms, f0_hz, confidence)."""
    lag_min = int(SR / fmax)
    lag_max = int(SR / fmin)
    times, f0s, confs = [], [], []
    n = min(len(x), int(tmax_s * SR))
    for start in range(0, max(1, n - win), hop):
        seg = x[start : start + win]
        if len(seg) < win:
            break
        seg = seg - seg.mean()
        e = (seg ** 2).sum()
        if e < 1e-9:
            times.append(start / SR * 1000)
            f0s.append(np.nan)
            confs.append(0.0)
            continue
        ac = signal.correlate(seg, seg, mode="full")[win - 1 :]
        ac = ac / (ac[0] + EPS)
        seg_ac = ac[lag_min:lag_max]
        # first significant peak after the zero-lag lobe
        k = np.argmax(seg_ac)
        lag = k + lag_min
        # parabolic interpolation
        if 1 <= lag < len(ac) - 1:
            y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > EPS else 0.0
            lag_f = lag + delta
        else:
            lag_f = lag
        times.append((start + win / 2) / SR * 1000)
        f0s.append(SR / lag_f)
        confs.append(float(seg_ac[k]))
    return np.array(times), np.array(f0s), np.array(confs)


def spectro_db(x, nperseg=512, hop=96):
    f, t, Z = signal.stft(x, fs=SR, nperseg=nperseg, noverlap=nperseg - hop, boundary=None, padded=False)
    return f, t * 1000, np.maximum(db(Z), -100.0)


def spectral_diff(x, y, mask_db=40.0):
    """Phase-insensitive spectral-shape difference over time.

    Per frame: energy-weighted mean |dB difference| and max |dB difference| over
    bins within mask_db of the frame's loudest bin in either signal.
    """
    f, t, X = spectro_db(x)
    _, _, Y = spectro_db(y)
    D = X - Y
    ref = np.maximum(X, Y)
    frame_max = ref.max(axis=0, keepdims=True)
    mask = ref > (frame_max - mask_db)
    w = 10 ** (ref / 20) * mask
    mean_abs = (np.abs(D) * w).sum(axis=0) / (w.sum(axis=0) + EPS)
    max_abs = np.where(mask, np.abs(D), 0).max(axis=0)
    return f, t, D, mask, mean_abs, max_abs


def summarize_windows(t_ms, series, windows):
    out = {}
    for name, (a, b) in windows.items():
        m = (t_ms >= a) & (t_ms < b)
        out[name] = float(np.nanmax(series[m])) if m.any() else float("nan")
    return out


def cents(f_a, f_b):
    return 1200 * np.log2(f_a / f_b)


def analyze_case(case):
    S = read(f"{SRC}/{case}_immediate_same.wav")
    P = read(f"{SRC}/{case}_immediate_prepared.wav")
    L = read(f"{SRC}/{case}_legacy_gate.wav")
    R = read(f"{SRC}/{case}_boosted_residual.wav")
    res = {"case": case}

    if np.abs(R).max() == 0.0 and np.array_equal(S, P):
        res["identical"] = True
        return res
    res["identical"] = False

    a, b, err = recover_gain_ratio(S, P, R)
    res["gain_fit_error"] = err
    res["same_minus_prepared_level_db"] = float(20 * np.log10(a / b))
    P_rel = P * (b / a)  # prepared, expressed in immediate_same's scale

    chan = {"out": 0, "aux": 1}
    per_chan = {}
    for cname, ci in chan.items():
        s, p, l = S[:, ci], P_rel[:, ci], L[:, ci]
        # Skip silent channels.
        if np.sqrt((s ** 2).mean()) < 1e-6 and np.sqrt((p ** 2).mean()) < 1e-6:
            continue
        c = {}
        # Envelope (block RMS, dB)
        es, ep, el = block_rms_db(s), block_rms_db(p), block_rms_db(l)
        env_diff = es - ep
        c["env_diff_max_abs_first20_blocks_db"] = float(np.max(np.abs(env_diff[:20])))
        c["env_diff_block0_db"] = float(env_diff[0])
        c["env_diff_block1_db"] = float(env_diff[1])
        c["env_diff_max_abs_after20_db"] = float(np.max(np.abs(env_diff[20:])))
        # First block after which |diff| stays under 1 dB.
        over = np.where(np.abs(env_diff) > 1.0)[0]
        c["env_last_block_over_1db"] = int(over[-1]) if len(over) else -1
        c["env_blocks_over_1db"] = int(len(over))

        # Timing: best lag between same and prepared, and residual explained by it.
        lag, corr = best_lag(s[: 100 * BLOCK], p[: 100 * BLOCK])
        raw_res = np.sqrt(((s - p) ** 2).mean())
        lag_res = np.sqrt(((s - shifted(p, lag)) ** 2).mean())
        c["best_lag_samples"] = lag
        c["best_lag_corr"] = corr
        c["residual_rel_db"] = float(20 * np.log10(raw_res / (np.sqrt((s ** 2).mean()) + EPS)))
        c["residual_after_lag_rel_db"] = float(20 * np.log10(lag_res / (np.sqrt((s ** 2).mean()) + EPS)))

        # Pitch trajectories (first 150 ms)
        ts, fs_, cs = pitch_track(s)
        _, fp, cp = pitch_track(p)
        _, fl, cl = pitch_track(l)
        conf_ok = (cs > 0.6) & (cp > 0.6)
        if conf_ok.any():
            dc = np.where(conf_ok, cents(fs_, fp), np.nan)
            c["pitch_conf_frames"] = int(conf_ok.sum())
            c["pitch_diff_cents_first_20ms_max"] = float(np.nanmax(np.abs(dc[ts < 20]))) if (ts < 20).any() else float("nan")
            c["pitch_diff_cents_after_20ms_max"] = float(np.nanmax(np.abs(dc[ts >= 20]))) if np.isfinite(dc[ts >= 20]).any() else float("nan")
            c["pitch_same_first_frame_hz"] = float(fs_[conf_ok][0])
            c["pitch_prepared_first_frame_hz"] = float(fp[conf_ok][0])
            steady = conf_ok & (ts > 60)
            c["pitch_same_steady_hz"] = float(np.nanmedian(fs_[steady])) if steady.any() else float("nan")
        else:
            c["pitch_conf_frames"] = 0

        # Spectral-shape difference (phase-insensitive), same vs prepared and same vs legacy
        windows = {"0_20ms": (0, 20), "20_100ms": (20, 100), "100_128ms": (100, 130)}
        f, t, D, mask, mean_abs, max_abs = spectral_diff(s, p)
        c["spec_meanabs_db"] = summarize_windows(t, mean_abs, windows)
        c["spec_maxabs_db"] = summarize_windows(t, max_abs, windows)
        _, _, DL, maskL, mean_abs_L, max_abs_L = spectral_diff(s, l)
        c["spec_vs_legacy_meanabs_db"] = summarize_windows(t, mean_abs_L, windows)
        # prepared versus legacy (both RMS-matched files): do the two references agree?
        _, _, _, _, mean_abs_PL, _ = spectral_diff(P[:, ci], l)
        c["spec_prepared_vs_legacy_meanabs_db"] = summarize_windows(t, mean_abs_PL, windows)

        c["_plot"] = dict(es=es, ep=ep, el=el, ts=ts, fs=fs_, fp=fp, fl=fl, cs=cs, cp=cp, cl=cl,
                          f=f, t=t, D=np.where(mask, D, np.nan), mean_abs=mean_abs, mean_abs_L=mean_abs_L)
        per_chan[cname] = c
    res["channels"] = per_chan
    return res


def plot_case(res):
    if res.get("identical"):
        return
    chans = [c for c in ("out", "aux") if c in res["channels"]]
    fig, axes = plt.subplots(len(chans), 3, figsize=(17, 4.2 * len(chans)), squeeze=False)
    for row, cname in enumerate(chans):
        c = res["channels"][cname]
        p = c["_plot"]
        ax = axes[row, 0]
        nblk = 120
        tb = np.arange(nblk) * BLOCK / SR * 1000
        ax.plot(tb, p["es"][:nblk], label="immediate_same", lw=1.2)
        ax.plot(tb, p["ep"][:nblk], label="immediate_prepared (raw-relative)", lw=1.2)
        ax.plot(tb, p["el"][:nblk], label="legacy_gate (RMS-matched)", lw=0.9, alpha=0.7)
        ax.set_title(f"{res['case']} [{cname}] block RMS envelope")
        ax.set_xlabel("ms after rising edge"); ax.set_ylabel("dBFS"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(p["ts"], p["fs"], ".-", label="same", ms=3)
        ax.plot(p["ts"], p["fp"], ".-", label="prepared", ms=3)
        ax.plot(p["ts"], p["fl"], ".-", label="legacy", ms=3, alpha=0.6)
        ax.set_yscale("log")
        ax.set_title("autocorrelation pitch track (low confidence frames included)")
        ax.set_xlabel("ms after rising edge"); ax.set_ylabel("Hz"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

        ax = axes[row, 2]
        f, t, D = p["f"], p["t"], p["D"]
        fm = f <= 12000
        im = ax.pcolormesh(t, f[fm], D[fm], cmap="RdBu_r", vmin=-12, vmax=12, shading="auto")
        ax.set_title("spectral dB difference: same minus prepared (masked to loud bins)")
        ax.set_xlabel("ms after rising edge"); ax.set_ylabel("Hz")
        fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(os.path.join(DST, f"{res['case']}.png"), dpi=110)
    plt.close(fig)


def fmt(v, nd=1):
    if isinstance(v, float):
        if np.isnan(v):
            return "n/a"
        return f"{v:.{nd}f}"
    return str(v)


def main():
    cases = sorted({os.path.basename(p).replace("_immediate_same.wav", "")
                    for p in glob.glob(f"{SRC}/*_immediate_same.wav")})
    results = [analyze_case(c) for c in cases]
    for r in results:
        plot_case(r)

    lines = []
    lines.append("# Trigger-settling listening artifacts: deeper analysis\n")
    lines.append("Generated by settling_analysis.py from the WAVs in out/diagnostics/trigger-settling.\n")
    lines.append("All comparisons are `immediate_same` versus `immediate_prepared` unless a column says legacy. "
                 "The policy WAVs are RMS-normalised independently, so the true relative level between same and prepared "
                 "was recovered from the boosted residual by least squares (fit error is reported; near zero means the recovery is exact). "
                 "Legacy comparisons are RMS-matched shape comparisons only.\n")

    lines.append("## Level and timing\n")
    lines.append("| case | chan | raw level same−prepared (dB) | gain fit err | env diff block 0 / block 1 (dB) | max env diff first 20 blocks (dB) | max env diff after block 20 (dB) | blocks over 1 dB / last | best lag (samples) | residual rel. (dB) | after lag align (dB) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["identical"]:
            lines.append(f"| {r['case']} | both | 0.0 | exact | 0.0 / 0.0 | 0.0 | 0 / — | 0 | −inf | −inf |")
            continue
        for cname, c in r["channels"].items():
            lines.append(f"| {r['case']} | {cname} | {r['same_minus_prepared_level_db']:+.2f} | {r['gain_fit_error']:.1e} | "
                         f"{c['env_diff_block0_db']:+.1f} / {c['env_diff_block1_db']:+.1f} | {c['env_diff_max_abs_first20_blocks_db']:.1f} | {c['env_diff_max_abs_after20_db']:.1f} | "
                         f"{c['env_blocks_over_1db']} / {c['env_last_block_over_1db']} | {c['best_lag_samples']} | "
                         f"{c['residual_rel_db']:.1f} | {c['residual_after_lag_rel_db']:.1f} |")

    lines.append("\n## Pitch\n")
    lines.append("| case | chan | confident frames | first-frame f0 same / prepared (Hz) | steady f0 same (Hz) | max |Δcents| < 20 ms | max |Δcents| ≥ 20 ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if r["identical"]:
            continue
        for cname, c in r["channels"].items():
            if c.get("pitch_conf_frames", 0) == 0:
                lines.append(f"| {r['case']} | {cname} | 0 | n/a | n/a | n/a | n/a |")
                continue
            lines.append(f"| {r['case']} | {cname} | {c['pitch_conf_frames']} | {fmt(c['pitch_same_first_frame_hz'])} / {fmt(c['pitch_prepared_first_frame_hz'])} | "
                         f"{fmt(c['pitch_same_steady_hz'])} | {fmt(c['pitch_diff_cents_first_20ms_max'])} | {fmt(c['pitch_diff_cents_after_20ms_max'])} |")

    lines.append("\n## Spectral shape (phase-insensitive)\n")
    lines.append("Energy-weighted mean |dB difference| across loud bins per 10.7 ms frame, maximum within each time window. "
                 "Roughly: under 1 dB is at the level-JND floor, 1–3 dB is a plausible timbre difference, above 3 dB is a clear one. "
                 "Max-bin values are the single worst bin and are more sensitive to phase-cancellation notches.\n")
    lines.append("| case | chan | mean dB 0–20 ms | mean dB 20–100 ms | mean dB 100–128 ms | max-bin dB 0–20 ms | max-bin dB 20–100 ms | same vs legacy mean dB 0–20 ms | same vs legacy 20–100 ms | prepared vs legacy 0–20 ms | prepared vs legacy 20–100 ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["identical"]:
            lines.append(f"| {r['case']} | both | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |")
            continue
        for cname, c in r["channels"].items():
            m, x, l, pl = c["spec_meanabs_db"], c["spec_maxabs_db"], c["spec_vs_legacy_meanabs_db"], c["spec_prepared_vs_legacy_meanabs_db"]
            lines.append(f"| {r['case']} | {cname} | {m['0_20ms']:.2f} | {m['20_100ms']:.2f} | {m['100_128ms']:.2f} | "
                         f"{x['0_20ms']:.1f} | {x['20_100ms']:.1f} | {l['0_20ms']:.2f} | {l['20_100ms']:.2f} | {pl['0_20ms']:.2f} | {pl['20_100ms']:.2f} |")

    lines.append("\nPer-case figures: `<case>.png` (envelopes, pitch tracks, spectral-difference heatmap).\n")
    with open(os.path.join(DST, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
