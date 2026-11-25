"""

Batch processing:
Hypopnea / Obstructive Apnea / Desaturation -> RSA

Searches the root folder and all subfolders for EDF files.

For each EDF, saves:
    <edfname>_new_hypopnea_rsa_60s.csv
    <edfname>_new_obstructive_apnea_rsa_60s.csv
    <edfname>_new_desaturation_rsa_60s.csv

Respiration signal:
    CPAP patient -> CFlow
    PSG patient  -> PFlow

RSA windows:
    Pre  = 60 seconds before event start
    Para = 60 seconds from event start
    Post = next 60 seconds

Install:
    pip install numpy pandas scipy neurokit2
"""

import argparse
import os
import io
import contextlib
import traceback
import warnings

import numpy as np
import pandas as pd
import scipy.signal as sg

from scipy.ndimage import (
    maximum_filter1d,
    minimum_filter1d,
    uniform_filter1d
)


# ============================================================
# SETTINGS
# ============================================================

ROOT_FOLDER = r"E:\sleep_apnea_data"

W = 60

EVENT_TYPES = [
    "hypopnea",
    "obstructive apnea",
    "desaturation"
]

EVENT_LABEL = {
    "hypopnea": "Hypopnea",
    "obstructive apnea": "Obstructive Apnea",
    "desaturation": "Desaturation"
}


# ============================================================
# RSA FUNCTION
# NeuroKit2 if available, otherwise local implementation
# ============================================================

try:
    import neurokit2 as nk

    if not hasattr(nk, "ecg_rsa"):
        raise ImportError("nk.ecg_rsa is not available")

    ecg_rsa = nk.ecg_rsa

    print("Using NeuroKit2:", nk.__version__)

except (ImportError, AttributeError):

    print("Using local ecg_rsa implementation")

    def signal_resample(
        signal,
        sampling_rate=1000,
        desired_sampling_rate=2
    ):
        signal = np.asarray(signal, dtype=float)

        n = int(round(
            len(signal) * desired_sampling_rate / sampling_rate
        ))

        if n < 2:
            return np.array([])

        return np.interp(
            np.linspace(0, len(signal) - 1, n),
            np.arange(len(signal)),
            signal
        )

    def signal_filter(
        signal,
        sampling_rate=1000,
        lowcut=None,
        highcut=None,
        method="butterworth",
        order=5,
        window_size=None
    ):
        signal = np.asarray(signal, dtype=float)

        if method == "savgol":
            window_size = min(window_size, len(signal))

            if window_size % 2 == 0:
                window_size -= 1

            if window_size <= order:
                return signal.copy()

            return sg.savgol_filter(
                signal,
                window_size,
                order
            )

        if lowcut is not None and highcut is not None:
            freqs = [lowcut, highcut]
            btype = "bandpass"

        elif lowcut is not None:
            freqs = lowcut
            btype = "highpass"

        else:
            freqs = highcut
            btype = "lowpass"

        sos = sg.butter(
            order,
            freqs,
            btype=btype,
            output="sos",
            fs=sampling_rate
        )

        return sg.sosfiltfilt(sos, signal)

    def signal_interpolate(x_values, y_values, x_new):
        return np.interp(x_new, x_values, y_values)

    def signal_rate(
        peaks,
        sampling_rate=1000,
        desired_length=None
    ):
        rate = 60 / (
            np.diff(peaks) / sampling_rate
        )

        if len(rate) == 0:
            return np.full(desired_length, np.nan)

        return np.interp(
            np.arange(desired_length),
            peaks[1:],
            rate
        )

    def _signal_formatpeaks_sanitize(peaks, key="Peaks"):

        if isinstance(peaks, dict):
            matching = [
                k for k in peaks
                if key in k
            ]

            if not matching:
                raise ValueError("R-peak information not found")

            peaks = peaks[matching[0]]

        return np.asarray(peaks).astype(int)

    def ecg_rsa(
        ecg_signals,
        rsp_signals=None,
        rpeaks=None,
        sampling_rate=1000,
        continuous=False
    ):

        signals, ecg_period, rpeaks, _ = _ecg_rsa_formatinput(
            ecg_signals,
            rsp_signals,
            rpeaks,
            sampling_rate
        )

        rsp_cycles = _ecg_rsa_cycles(signals)

        rsp_onsets = rsp_cycles["RSP_Inspiration_Onsets"]

        rsp_peaks = np.argwhere(
            signals["RSP_Peaks"].values == 1
        )[:, 0]

        if len(rsp_onsets) == 0:
            raise ValueError("No respiratory cycles found")

        rsp_peaks = rsp_peaks[
            rsp_peaks > rsp_onsets[0]
        ]

        if len(rsp_peaks) - len(rsp_onsets) == 0:
            rsp_peaks = rsp_peaks[:-1]

        rsa_p2t = _ecg_rsa_p2t(
            rsp_onsets,
            rpeaks,
            sampling_rate,
            continuous=continuous,
            ecg_period=ecg_period,
            rsp_peaks=rsp_peaks
        )

        rsa_pb = _ecg_rsa_pb(
            ecg_period,
            sampling_rate,
            continuous=continuous
        )

        if continuous is False:
            rsa = {}
            rsa.update(rsa_p2t)
            rsa.update(rsa_pb)
            return rsa

        return pd.DataFrame({"RSA_P2T": rsa_p2t})

    def _ecg_rsa_p2t(
        rsp_onsets,
        rpeaks,
        sampling_rate,
        continuous=False,
        ecg_period=None,
        rsp_peaks=None
    ):

        cycles_rri = []

        for idx in range(len(rsp_onsets) - 1):

            cycle = rpeaks[
                np.logical_and(
                    rpeaks >= rsp_onsets[idx],
                    rpeaks < rsp_onsets[idx + 1]
                )
            ]

            cycles_rri.append(cycle)

        rsa_values = np.full(
            len(cycles_rri),
            np.nan
        )

        for i, cycle in enumerate(cycles_rri):

            RRis = np.diff(cycle) / sampling_rate

            if len(RRis) > 1:
                rsa_values[i] = (
                    np.max(RRis) - np.min(RRis)
                )

        if continuous is False:

            mean_value = (
                np.nanmean(rsa_values)
                if np.any(np.isfinite(rsa_values))
                else np.nan
            )

            rsa = {
                "RSA_P2T_Mean": mean_value,
                "RSA_P2T_SD": (
                    np.nanstd(rsa_values, ddof=1)
                    if np.sum(np.isfinite(rsa_values)) > 1
                    else np.nan
                ),
                "RSA_P2T_NoRSA": int(
                    np.sum(~np.isfinite(rsa_values))
                )
            }

            rsa["RSA_P2T_Mean_log"] = (
                np.log(mean_value)
                if np.isfinite(mean_value) and mean_value > 0
                else np.nan
            )

            return rsa

        valid = np.isfinite(rsa_values)

        if not np.any(valid):
            return np.full(len(ecg_period), np.nan)

        return signal_interpolate(
            rsp_peaks[valid],
            rsa_values[valid],
            np.arange(len(ecg_period))
        )

    def _ecg_rsa_pb(
        ecg_period,
        sampling_rate,
        continuous=False
    ):

        if continuous is True:
            return None

        resampled = signal_resample(
            ecg_period,
            sampling_rate=sampling_rate,
            desired_sampling_rate=2
        )

        if len(resampled) < 30:
            return {"RSA_PorgesBohrer": np.nan}

        trend = signal_filter(
            resampled,
            sampling_rate=2,
            lowcut=0.095,
            method="savgol",
            order=3,
            window_size=21
        )

        zero_mean = resampled - trend

        zero_mean_filtered = signal_filter(
            zero_mean,
            sampling_rate=2,
            lowcut=0.12,
            highcut=0.40
        )

        time = np.arange(
            len(zero_mean_filtered)
        ) / 2

        df = pd.DataFrame({
            "Epoch Index": time // 30,
            "Signal": zero_mean_filtered
        })

        variance_values = []

        for _, epoch in df.groupby("Epoch Index"):

            variance = epoch["Signal"].var()

            if np.isfinite(variance) and variance > 0:
                variance_values.append(
                    np.log(variance / 1000)
                )

        pb = (
            float(np.mean(variance_values))
            if variance_values
            else np.nan
        )

        return {"RSA_PorgesBohrer": pb}

    def _ecg_rsa_cycles(signals):

        insp = np.intersect1d(
            np.where(signals["RSP_Phase"] == 1)[0],
            np.where(
                signals["RSP_Phase_Completion"] == 0
            )[0]
        )

        exp = np.intersect1d(
            np.where(signals["RSP_Phase"] == 0)[0],
            np.where(
                signals["RSP_Phase_Completion"] == 0
            )[0]
        )

        return {
            "RSP_Inspiration_Onsets": insp,
            "RSP_Expiration_Onsets": exp,
            "RSP_Cycles_Length": np.diff(insp)
        }

    def _ecg_rsa_formatinput(
        ecg_signals,
        rsp_signals,
        rpeaks=None,
        sampling_rate=1000
    ):

        if isinstance(ecg_signals, tuple):
            ecg_signals = ecg_signals[0]
            rpeaks = None

        if not isinstance(ecg_signals, pd.DataFrame):
            raise ValueError("ECG signals must be a DataFrame")

        ecg_cols = [
            c for c in ecg_signals.columns
            if "ECG_Rate" in c
        ]

        if ecg_cols:
            ecg_period = ecg_signals[ecg_cols[0]].values

        else:
            ecg_cols = [
                c for c in ecg_signals.columns
                if "ECG_R_Peaks" in c
            ]

            if not ecg_cols:
                raise ValueError("ECG rate or R-peaks not found")

            ecg_period = signal_rate(
                rpeaks,
                sampling_rate=sampling_rate,
                desired_length=len(ecg_signals)
            )

        if rsp_signals is None:
            raise ValueError("RSP signal not found")

        if isinstance(rsp_signals, tuple):
            rsp_signals = rsp_signals[0]

        if not isinstance(rsp_signals, pd.DataFrame):
            raise ValueError("RSP signals must be a DataFrame")

        required = [
            "RSP_Phase",
            "RSP_Phase_Completion",
            "RSP_Peaks"
        ]

        for col in required:
            if col not in rsp_signals.columns:
                raise ValueError(
                    f"Missing respiratory column: {col}"
                )

        rpeaks = _signal_formatpeaks_sanitize(
            ecg_signals if rpeaks is None else rpeaks,
            key="ECG_R_Peaks"
        )

        signals = pd.concat(
            [ecg_signals, rsp_signals],
            axis=1
        )

        return signals, ecg_period, rpeaks, None


# ============================================================
# GENERAL HELPERS
# ============================================================

def csv_path_for(fn, typ):

    base = os.path.splitext(
        os.path.basename(fn)
    )[0]

    filename = (
        f"{base}_new_{typ.replace(' ', '_')}_rsa_60s.csv"
    )

    return os.path.join(
        os.path.dirname(fn),
        filename
    )


def runs(mask, minlen, gap=2):

    idx = np.where(mask)[0]

    if len(idx) == 0:
        return []

    out = []

    s = p = idx[0]

    for i in idx[1:]:

        if i - p > gap + 1:

            if p - s + 1 >= minlen:
                out.append((s, p + 1))

            s = i

        p = i

    if p - s + 1 >= minlen:
        out.append((s, p + 1))

    return out


def detect_rpeaks(x, fs):

    x = np.asarray(x, dtype=float)

    if fs <= 30:
        raise ValueError(
            f"ECG sampling rate is too low: {fs} Hz"
        )

    b, a = sg.butter(
        2,
        [5 / (fs / 2), 15 / (fs / 2)],
        "band"
    )

    y = sg.filtfilt(b, a, x)

    lim = np.percentile(
        np.abs(y),
        99.5
    ) * 3

    y = np.clip(y, -lim, lim)

    dd = np.diff(
        y,
        prepend=y[0]
    ) ** 2

    ww = max(1, int(0.12 * fs))

    m = np.convolve(
        dd,
        np.ones(ww) / ww,
        "same"
    )

    cand, _ = sg.find_peaks(
        m,
        distance=int(0.3 * fs)
    )

    thr = np.zeros(len(m))

    seg = max(1, int(10 * fs))

    for k in range(0, len(m), seg):

        chunk = m[k:k + seg]

        if len(chunk):
            thr[k:k + seg] = (
                0.35 * np.percentile(chunk, 98)
            )

    cand = cand[
        m[cand] > thr[cand]
    ]

    xs = sg.filtfilt(
        *sg.butter(
            2,
            [0.5 / (fs / 2), 30 / (fs / 2)],
            "band"
        ),
        x
    )

    kk = max(1, int(0.06 * fs))

    peaks = []

    for c in cand:

        left = max(0, c - kk)
        right = min(len(xs), c + kk + 1)

        if right > left:
            peak = left + np.argmax(
                np.abs(xs[left:right])
            )
            peaks.append(peak)

    return np.unique(peaks)


def safe_bandpass(x, fs, low, high, order=2):

    x = np.asarray(x, dtype=float)

    nyquist = fs / 2

    high = min(high, 0.95 * nyquist)

    if low <= 0 or high <= low:
        raise ValueError(
            f"Invalid bandpass limits: {low}, {high}, fs={fs}"
        )

    sos = sg.butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sg.sosfiltfilt(sos, x)


# ============================================================
# PROCESS ONE EDF FILE
# ============================================================

def process_edf(fn):

    print("\nReading EDF header...")

    # --------------------------------------------------------
    # EDF HEADER
    # --------------------------------------------------------

    with open(fn, "rb") as f:

        h = f.read(256)

        hdr_bytes = int(h[184:192])
        rec_dur = float(h[244:252])
        ns = int(h[252:256])

        if ns <= 0 or rec_dur <= 0:
            raise ValueError("Invalid EDF header")

        r = f.read(ns * 256)

    ws = [
        16, 80, 8, 8, 8,
        8, 8, 80, 8, 32
    ]

    offs = []
    o = 0

    for width in ws:
        offs.append(o)
        o += width * ns

    def fld(k):

        return [
            r[
                offs[k] + i * ws[k]:
                offs[k] + (i + 1) * ws[k]
            ].decode("latin1").strip()
            for i in range(ns)
        ]

    labels = fld(0)
    trans = fld(1)

    pmin = np.array(fld(3), dtype=float)
    pmax = np.array(fld(4), dtype=float)
    dmin = np.array(fld(5), dtype=float)
    dmax = np.array(fld(6), dtype=float)

    spr = np.array(fld(8), dtype=int)

    if np.any(spr <= 0):
        raise ValueError("Invalid samples-per-record value")

    record_bytes = 2 * int(spr.sum())

    data_bytes = os.path.getsize(fn) - hdr_bytes

    nrec = data_bytes // record_bytes

    if nrec <= 0:
        raise ValueError("No EDF data records found")

    fs_ch = spr / rec_dur

    st = np.r_[
        0,
        np.cumsum(spr)
    ]

    d = np.memmap(
        fn,
        dtype="<i2",
        mode="r",
        offset=hdr_bytes,
        shape=(nrec, int(spr.sum()))
    )

    def sig(i):

        x = np.asarray(
            d[:, st[i]:st[i + 1]]
        ).reshape(-1).astype(float)

        denominator = dmax[i] - dmin[i]

        if denominator == 0:
            raise ValueError(
                f"Invalid digital calibration for channel {labels[i]}"
            )

        return (
            pmin[i]
            + (x - dmin[i])
            * (pmax[i] - pmin[i])
            / denominator
        )

    # --------------------------------------------------------
    # CHANNEL SELECTION
    # --------------------------------------------------------

    L = [x.lower() for x in labels]
    T = [x.lower() for x in trans]

    def pick(conditions):

        for condition in conditions:

            idx = [
                i for i in range(ns)
                if condition(i)
            ]

            if idx:
                return idx[0]

        return None

    def has(i, *words):

        return any(
            word in L[i] or word in T[i]
            for word in words
        )

    # CFlow: CPAP respiratory flow
    cflow_i = pick([
        lambda i: has(i, "cflow"),
        lambda i: (
            "flow" in L[i]
            and has(i, "therapy", "device", "xpap", "cpap")
        ),
        lambda i: (
            "flow" in L[i]
            and any(
                w in T[i]
                for w in ("lpm", "l/min")
            )
        )
    ])

    # PFlow: PSG respiratory flow
    pflow_i = pick([
        lambda i: has(i, "pflow"),
        lambda i: (
            "flow" in L[i]
            and "press" in T[i]
        ),
        lambda i: (
            "flow" in L[i]
            and "therm" in T[i]
        ),
        lambda i: "nasal" in L[i]
    ])

    # Patient type
    ptype = (
        "cpap"
        if cflow_i is not None
        else "psg"
    )

    # Select the correct flow for BOTH event detection and RSA
    flow_i = (
        cflow_i
        if ptype == "cpap"
        else pflow_i
    )

    if flow_i is None:
        raise RuntimeError(
            f"{ptype.upper()} respiratory channel not found. "
            f"CFlow={cflow_i}, PFlow={pflow_i}. "
            f"Available channels: {labels}"
        )

    spo2_i = pick([
        lambda i: (
            "spo2" in L[i]
            or "sao2" in L[i]
            or "sp02" in L[i]
        )
    ])

    tho_i = pick([
        lambda i: (
            "tho" in L[i]
            and "effort" in L[i]
        ),
        lambda i: (
            "thor" in L[i]
            or "chest" in L[i]
        )
    ])

    abd_i = pick([
        lambda i: "abd" in L[i]
    ])

    ecg_i = pick([
        lambda i: L[i].strip() == "ecg ii",
        lambda i: L[i].strip() in ("ecg 2", "ecg2"),
        lambda i: (
            "ecg" in L[i]
            and "ii" in L[i]
        ),
        lambda i: L[i].startswith("ecg"),
        lambda i: "ekg" in L[i]
    ])

    need = {
        "flow": flow_i,
        "SpO2": spo2_i,
        "thorax effort": tho_i,
        "ECG": ecg_i
    }

    missing = [
        name
        for name, value in need.items()
        if value is None
    ]

    if missing:
        raise RuntimeError(
            "Channels not found: "
            + ", ".join(missing)
            + f". Available channels: {labels}"
        )

    if abd_i is None:
        abd_i = tho_i

    print(
        f"  Patient type: {ptype.upper()}\n"
        f"  Flow channel: {labels[flow_i]} "
        f"(CFlow for CPAP / PFlow for PSG)\n"
        f"  SpO2 channel: {labels[spo2_i]}\n"
        f"  Thorax channel: {labels[tho_i]}\n"
        f"  Abdomen channel: {labels[abd_i]}\n"
        f"  ECG channel: {labels[ecg_i]}"
    )

    # --------------------------------------------------------
    # LOAD SIGNALS
    # --------------------------------------------------------

    fl = sig(flow_i)
    sp = sig(spo2_i)
    th = sig(tho_i)
    ab = sig(abd_i)
    ecg = sig(ecg_i)

    fs_fl = float(fs_ch[flow_i])
    fs_sp = float(fs_ch[spo2_i])
    fs_ef = float(fs_ch[tho_i])
    FS = float(fs_ch[ecg_i])
    FR = fs_fl

    if min(fs_fl, fs_sp, fs_ef, FS) <= 0:
        raise ValueError("Invalid signal sampling rate")

    print(
        f"  Sampling rates: Flow={fs_fl:g} Hz, "
        f"SpO2={fs_sp:g} Hz, "
        f"Effort={fs_ef:g} Hz, "
        f"ECG={FS:g} Hz"
    )

    # --------------------------------------------------------
    # EVENT DETECTION
    # --------------------------------------------------------

    print("\nDetecting events...")

    # Flow envelope
    f = safe_bandpass(
        np.nan_to_num(fl),
        fs_fl,
        0.1,
        min(2.0, 0.45 * fs_fl)
    )

    flow_window = max(
        1,
        int(round(4 * fs_fl))
    )

    amp = (
        maximum_filter1d(f, flow_window)
        - minimum_filter1d(f, flow_window)
    )

    amp = uniform_filter1d(
        amp,
        max(1, int(round(fs_fl)))
    )

    s_fl = max(1, int(round(fs_fl)))
    s_sp = max(1, int(round(fs_sp)))
    s_ef = max(1, int(round(fs_ef)))

    amp1 = amp[::s_fl]

    # Effort envelopes
    def env(x):

        y = safe_bandpass(
            np.nan_to_num(x),
            fs_ef,
            0.1,
            min(1.0, 0.45 * fs_ef)
        )

        window = max(
            1,
            int(round(8 * fs_ef))
        )

        p2p = (
            maximum_filter1d(y, window)
            - minimum_filter1d(y, window)
        )

        return p2p[::s_ef]

    eth = env(th)
    eab = env(ab)

    # SpO2 one-second samples
    sp1 = sp[::s_sp].astype(float)

    # Use the common available duration.
    N = min(
        len(amp1),
        len(eth),
        len(eab),
        len(sp1)
    )

    if N < 180:
        raise ValueError(
            "Not enough aligned data for event detection"
        )

    amp1 = amp1[:N]
    eth = eth[:N]
    eab = eab[:N]
    sp1 = sp1[:N]

    # Local normal baseline
    def q(x):

        return (
            pd.Series(x)
            .rolling(
                180,
                min_periods=60
            )
            .quantile(0.9)
            .shift(1)
            .bfill()
            .values
        )

    def sdiv(num, den):

        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)

        ok = den > 1e-9

        return np.where(
            ok,
            num / np.where(ok, den, 1.0),
            1.0
        )

    ratio = sdiv(
        amp1,
        q(amp1)
    )

    eff_ratio = np.maximum(
        sdiv(eth, q(eth)),
        sdiv(eab, q(eab))
    )

    # SpO2 desaturation
    sp1 = np.where(
        sp1 < 50,
        np.nan,
        sp1
    )

    sp_s = pd.Series(sp1).interpolate(
        limit=10
    )

    ref = (
        sp_s
        .rolling(100, min_periods=20)
        .max()
        .shift(1)
    )

    drop = (
        ref - sp_s
    ).values

    # Apnea detection
    rs = (
        pd.Series(ratio)
        .rolling(
            3,
            center=True,
            min_periods=1
        )
        .mean()
        .values
    )

    apn = runs(
        rs < 0.2,
        6,
        gap=1
    )

    # Hypopnea candidates
    hyp_all = runs(
        ratio < 0.7,
        7,
        gap=2
    )

    def has_desat(s, e):

        lo = s
        hi = min(N, e + 30)

        if hi <= lo:
            return False

        values = drop[lo:hi]

        return (
            np.any(np.isfinite(values))
            and np.nanmax(values) >= 3
        )

    rows = []

    # Obstructive apnea candidates
    for s, e in apn:

        if np.mean(eff_ratio[s:e]) >= 0.3:

            rows.append((
                "obstructive apnea",
                s,
                e
            ))

        # Central apnea is not included in the requested CSVs.

    # Hypopnea candidates
    for s, e in hyp_all:

        overlaps_apnea = any(
            not (e <= a_ or s >= b_)
            for a_, b_ in apn
        )

        if overlaps_apnea:
            continue

        if has_desat(s, e):

            rows.append((
                "hypopnea",
                s,
                e
            ))

    ev = pd.DataFrame(
        rows,
        columns=[
            "type",
            "start_s",
            "end_s"
        ]
    )

    if len(ev):

        ev["start_s"] = (
            ev["start_s"] - 2
        ).clip(lower=0)

        ev["end_s"] = ev["end_s"] + 2

    # Independent desaturation events
    des = [
        ("desaturation", s, e)
        for s, e in runs(
            np.nan_to_num(drop) >= 3,
            3,
            gap=5
        )
    ]

    des_df = pd.DataFrame(
        des,
        columns=[
            "type",
            "start_s",
            "end_s"
        ]
    )

    ev = pd.concat(
        [ev, des_df],
        ignore_index=True
    )

    ev = (
        ev[
            ev["type"].isin(EVENT_TYPES)
        ]
        .sort_values("start_s")
        .reset_index(drop=True)
    )

    print(
        "  Detected events:",
        ev["type"].value_counts().to_dict()
        if len(ev)
        else {}
    )

    # --------------------------------------------------------
    # ECG R-PEAKS
    # --------------------------------------------------------

    print("\nDetecting ECG R-peaks...")

    FS_INT = int(round(FS))
    FR_INT = int(round(FR))

    if abs(FS - FS_INT) > 0.01:
        raise ValueError(
            f"Non-integer ECG sampling rate ({FS} Hz) "
            "is not supported by this R-peak implementation."
        )

    if abs(FR - FR_INT) > 0.01:
        raise ValueError(
            f"Non-integer flow sampling rate ({FR} Hz) "
            "is not supported by this implementation."
        )

    FS = FS_INT
    FR = FR_INT

    rpk = detect_rpeaks(
        ecg,
        FS
    )

    # --------------------------------------------------------
    # RESPIRATION FROM SELECTED FLOW CHANNEL
    # CPAP = CFlow
    # PSG  = PFlow
    # --------------------------------------------------------

    print("Detecting respiratory cycles from flow...")

    resp = safe_bandpass(
        np.nan_to_num(fl),
        FR,
        0.1,
        0.6
    )

    if ptype == "psg":

        # Nasal pressure is often negative during inspiration.
        resp = -resp

    gstd = np.std(resp)

    lstd = np.sqrt(
        np.clip(
            uniform_filter1d(
                resp ** 2,
                max(1, int(30 * FR))
            ),
            0,
            None
        )
    )

    resp_n = resp / np.maximum(
        lstd,
        0.3 * gstd
    )

    # Respiratory troughs
    troughs, _ = sg.find_peaks(
        -resp_n,
        distance=int(1.5 * FR),
        prominence=0.4
    )

    hours = (
        nrec * rec_dur / 3600
    )

    print(
        f"  R peaks found: {len(rpk)} "
        f"({len(rpk) / hours:.0f}/h)"
    )

    print(
        f"  Respiratory cycles found: {len(troughs)} "
        f"({len(troughs) / hours:.0f}/h)"
    )

    # --------------------------------------------------------
    # RSA DIAGNOSTICS
    # --------------------------------------------------------

    diag = {
        "ok": 0,
        "too few R peaks (<10)": 0,
        "too few breaths (<3)": 0,
        "no usable cycle (<2 beats per breath)": 0,
        "ecg_rsa error": 0
    }

    first_err = []

    # --------------------------------------------------------
    # RSA WINDOW FUNCTION
    # --------------------------------------------------------

    def rsa_window(t0, t1):

        # Clip the window to the recording duration.
        duration = min(
            len(ecg) / FS,
            len(resp) / FR
        )

        t0 = max(0.0, float(t0))
        t1 = min(float(t1), duration)

        if t1 <= t0:
            return np.nan, np.nan

        # ECG window indices
        a = int(round(t0 * FS))
        b = int(round(t1 * FS))

        a = max(0, a)
        b = min(len(ecg), b)

        n = b - a

        if n <= 0:
            return np.nan, np.nan

        # R-peaks relative to the current ECG window
        pk = rpk[
            (rpk >= a)
            & (rpk < b)
        ] - a

        # Flow respiratory troughs in the current time window
        tr = troughs[
            (troughs >= int(round(t0 * FR)))
            & (troughs < int(round(t1 * FR)))
        ]

        if len(pk) < 10:

            diag["too few R peaks (<10)"] += 1

            return np.nan, np.nan

        if len(tr) < 3:

            diag["too few breaths (<3)"] += 1

            return np.nan, np.nan

        # ECG rate
        rate = (
            60 / (np.diff(pk) / FS)
        )

        if len(rate) == 0:
            return np.nan, np.nan

        ecg_signals = pd.DataFrame({
            "ECG_Rate": np.interp(
                np.arange(n),
                pk[1:],
                rate
            ),
            "ECG_R_Peaks": np.isin(
                np.arange(n),
                pk
            ).astype(int)
        })

        # Respiratory phase
        phase = np.zeros(n, dtype=float)

        # Start outside a detected cycle as an incomplete phase.
        comp = np.ones(n, dtype=float)

        rflag = np.zeros(n, dtype=float)

        usable_cycles = 0

        for i in range(len(tr) - 1):

            left_flow = int(tr[i])
            right_flow = int(tr[i + 1])

            if right_flow <= left_flow:
                continue

            segment = resp[
                left_flow:right_flow
            ]

            if len(segment) == 0:
                continue

            # Respiratory peak between two troughs
            peak_flow = (
                left_flow
                + int(np.argmax(segment))
            )

            # IMPORTANT:
            # Convert FLOW sample indices to time first,
            # then convert time to ECG-window sample indices.
            # Do not multiply an already-global sample index
            # by FS/FR without subtracting the window start.

            a2 = int(round(
                (left_flow / FR - t0) * FS
            ))

            p2 = int(round(
                (peak_flow / FR - t0) * FS
            ))

            b2 = int(round(
                (right_flow / FR - t0) * FS
            ))

            a2 = max(0, min(n, a2))
            p2 = max(0, min(n, p2))
            b2 = max(0, min(n, b2))

            if not (a2 < p2 < b2):
                continue

            # Inspiration phase
            phase[a2:p2] = 1

            comp[a2:p2] = np.linspace(
                0,
                1,
                p2 - a2,
                endpoint=False
            )

            # Expiration phase
            phase[p2:b2] = 0

            comp[p2:b2] = np.linspace(
                0,
                1,
                b2 - p2,
                endpoint=False
            )

            # Respiratory peak
            rflag[p2] = 1

            usable_cycles += 1

        if usable_cycles == 0:

            diag[
                "no usable cycle (<2 beats per breath)"
            ] += 1

            return np.nan, np.nan

        rsp_signals = pd.DataFrame({
            "RSP_Phase": phase,
            "RSP_Phase_Completion": comp,
            "RSP_Peaks": rflag
        })

        try:

            with (
                contextlib.redirect_stdout(io.StringIO()),
                warnings.catch_warnings()
            ):

                warnings.simplefilter("ignore")

                result = ecg_rsa(
                    ecg_signals,
                    rsp_signals,
                    {"ECG_R_Peaks": pk},
                    sampling_rate=FS,
                    continuous=False
                )

            p2t = result.get(
                "RSA_P2T_Mean",
                np.nan
            )

            pb = result.get(
                "RSA_PorgesBohrer",
                np.nan
            )

            p2t = float(p2t)
            pb = float(pb)

            # Local fallback returns P2T in seconds.
            # NeuroKit2 output units may differ, so do not
            # assume its P2T value has the same unit.
            if isinstance(ecg_rsa, type(None)):
                pass

            if np.isfinite(p2t):
                p2t_ms = p2t * 1000
            else:
                p2t_ms = np.nan

            if np.isfinite(p2t_ms):
                diag["ok"] += 1
            else:
                diag[
                    "no usable cycle (<2 beats per breath)"
                ] += 1

            return p2t_ms, pb

        except Exception as ex:

            diag["ecg_rsa error"] += 1

            if not first_err:
                first_err.append(repr(ex))

            return np.nan, np.nan

    # --------------------------------------------------------
    # RSA: PRE / PARA / POST
    # --------------------------------------------------------

    print("\nCalculating RSA...")

    out_rows = []

    for k, event in ev.iterrows():

        s0 = int(event.start_s)
        e0 = int(event.end_s)

        row = {
            "Event No": k + 1,
            "Event Type": event.type,
            "Start (s)": s0,
            "End (s)": e0
        }

        windows = {
            "Pre": (
                s0 - W,
                s0
            ),
            "Para": (
                s0,
                s0 + W
            ),
            "Post": (
                s0 + W,
                s0 + 2 * W
            )
        }

        for name, (t0, t1) in windows.items():

            p2t, pb = rsa_window(
                t0,
                t1
            )

            row[
                f"{name} RSA P2T Mean (ms)"
            ] = (
                round(p2t, 2)
                if np.isfinite(p2t)
                else np.nan
            )

            row[
                f"{name} RSA PorgesBohrer"
            ] = (
                round(pb, 3)
                if np.isfinite(pb)
                else np.nan
            )

        out_rows.append(row)

        print(
            f"  Event {k + 1}/{len(ev)}: "
            f"{event.type}, "
            f"{s0}-{e0} seconds"
        )

    print("\nRSA window diagnostics:")
    print(diag)

    if first_err:
        print(
            "First ecg_rsa error:",
            first_err[0]
        )

    # --------------------------------------------------------
    # CREATE OUTPUT DATAFRAME
    # --------------------------------------------------------

    cols = [
        "Event No",
        "Event Type",
        "Start (s)",
        "End (s)"
    ]

    for name in ("Pre", "Para", "Post"):

        cols.append(
            f"{name} RSA P2T Mean (ms)"
        )

        cols.append(
            f"{name} RSA PorgesBohrer"
        )

    out = pd.DataFrame(
        out_rows,
        columns=cols
    )

    # --------------------------------------------------------
    # SAVE THREE CSV FILES
    # --------------------------------------------------------

    print("\nSaving CSV files...")

    for typ in EVENT_TYPES:

        sub = (
            out[out["Event Type"] == typ]
            .copy()
            .reset_index(drop=True)
        )

        sub["Event No"] = np.arange(
            1,
            len(sub) + 1
        )

        lab = EVENT_LABEL[typ]

        new = pd.DataFrame({
            "Event No": sub["Event No"],

            f"Detected {lab} Start (s)": sub["Start (s)"],

            f"Detected {lab} End (s)": sub["End (s)"]
        })

        # P2T columns
        for name in ("Pre", "Para", "Post"):

            new[
                f"{name}-{lab} RSA P2T Mean (ms)"
            ] = sub[
                f"{name} RSA P2T Mean (ms)"
            ]

        # Porges-Bohrer columns
        for name in ("Pre", "Para", "Post"):

            new[
                f"{name}-{lab} RSA PorgesBohrer (ln ms^2)"
            ] = sub[
                f"{name} RSA PorgesBohrer"
            ]

        out_name = csv_path_for(
            fn,
            typ
        )

        new.to_csv(
            out_name,
            index=False
        )

        # Verify the CSV
        check = pd.read_csv(out_name)

        if len(check.columns) > 3:

            filled = int(
                check.iloc[:, 3:]
                .notna()
                .sum()
                .sum()
            )

            total = check.iloc[:, 3:].size

        else:
            filled = 0
            total = 0

        print(
            f"  Saved: {os.path.abspath(out_name)}"
        )

        print(
            f"  Events: {len(check)} | "
            f"Filled RSA cells: {filled}/{total}"
        )

    return fn


# ============================================================
# BATCH PROCESS ALL EDF FILES
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root",
        nargs="?",
        default=ROOT_FOLDER,
        help="Root folder containing EDF files"
    )

    # parser.add_argument(
    #     "--skip-done",
    #     action="store_true",
    #     help="Skip EDF files whose three CSV files already exist"
    # )
    parser.add_argument(
    "--skip-done",
    action="store_true",
    default=True,
    help="Skip EDF files whose three CSV files already exist"
)

    args = parser.parse_args()

    root = args.root

    if not os.path.isabs(root):

        root = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            root
        )

    if not os.path.isdir(root):

        raise SystemExit(
            f"Folder not found: {root}\n"
            "Check ROOT_FOLDER or provide the correct folder path."
        )

    # Find EDF files in every subfolder
    edfs = sorted(
        os.path.join(dp, filename)
        for dp, _, files in os.walk(root)
        for filename in files
        if filename.lower().endswith(".edf")
    )

    print("\n======================================")
    print("       BATCH RSA PROCESSING")
    print("======================================")

    print("Root folder:", root)
    print("EDF files found:", len(edfs))

    ok = []
    failed = []
    skipped = []

    for number, fn in enumerate(edfs, 1):

        print("\n--------------------------------------")
        print(f"[{number}/{len(edfs)}] {fn}")
        print("--------------------------------------")

        # Skip already processed files if requested
        if args.skip_done:

            all_exist = all(
                os.path.exists(
                    csv_path_for(fn, typ)
                )
                for typ in EVENT_TYPES
            )

            if all_exist:

                print("Already processed. Skipping.")

                skipped.append(fn)

                continue

        try:

            process_edf(fn)

            ok.append(fn)

            print("EDF processing completed successfully.")

        except Exception as ex:

            print("FAILED:", ex)

            traceback.print_exc()

            failed.append(
                (fn, str(ex))
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n======================================")
    print("           FINAL SUMMARY")
    print("======================================")

    print("Total EDF files:", len(edfs))
    print("Successfully processed:", len(ok))
    print("Skipped:", len(skipped))
    print("Failed:", len(failed))

    n_csv = sum(
        filename.endswith("_rsa_60s.csv")
        for _, _, files in os.walk(root)
        for filename in files
    )

    print("Total RSA CSV files under root:", n_csv)

    if failed:

        print("\nEDF files that failed:")

        for fn, message in failed:

            print("\nFile:", fn)
            print("Reason:", message)

    print("\nBatch processing finished.")
