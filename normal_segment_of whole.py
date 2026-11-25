"""
Find NORMAL segments (whole normal stretches, any length) in MANY PSG EDF files (e.g. 10 files), using only
the raw signals (no scoring annotations needed). A normal segment has:
  - no wake                     (EEG alpha / EMG activity)
  - no N1                       (theta-dominant, low-delta, low-alpha EEG)
  - no island data              (short sleep bouts, plus signal dropouts)
  - no hypopnea / obstructive apnea  (airflow reduction)
  - no desaturation             (SpO2 drop)

Usage:
  pip install numpy scipy
  python normal_segments_batch.py
  (edit EDF_FOLDER / OUTPUT_FOLDER / MAX_FILES below first)
  or: python normal_segments_batch.py "D:\\edf folder" "D:\\output folder"

Every EDF gets its own sub-folder inside OUTPUT_FOLDER (same folder names as the
EDF, e.g. Batch_001\\Patient_003) holding its two CSV files. In OUTPUT_FOLDER you
also get all_normal_segments.csv (all files together) and batch_report.csv.

NOTE: stages here are estimated from signal features, not true AASM scoring.
Check the segments it returns against the real hypnogram before relying on them.
"""
import os
import sys
import csv
import glob
import traceback
import numpy as np
from scipy import signal

# ============== SETTINGS ==============
EDF_FOLDER = r"E:\sleep_apnea_data"        # searched with all sub-folders
OUTPUT_FOLDER = r"D:\Purnita folder\normal_output"        # only used if SAVE_IN_PATIENT_FOLDER = False
SAVE_IN_PATIENT_FOLDER = True   # True = each patient's CSVs are saved inside that patient's own folder (next to the EDF)
MAX_FILES = None           # None = all EDF files (e.g. all 100 patients); a number like 10 = only the first 10

# Or list the files yourself (then EDF_FOLDER is only used to build the folder names):
EDF_PATHS = []
# EDF_PATHS = [
#     r"D:\Purnita folder\sleep_apnea_data\Batch_001\Patient_001\A0013366.edf",
#     r"D:\Purnita folder\sleep_apnea_data\Batch_001\Patient_002\A0013367.edf",
# ]

SKIP_IF_DONE = True        # skip an EDF if its segments CSV already exists in the output folder

EPOCH_S = 30
SPLIT_INTO_SEGMENTS = False  # False = report each whole normal stretch (any length), NOT cut into 2-3 min pieces
MIN_STRETCH_S = 120         # ignore normal stretches shorter than this many seconds (0 = keep all)
SEG_MIN_S = 120            # only used if SPLIT_INTO_SEGMENTS = True
SEG_MAX_S = 180            # only used if SPLIT_INTO_SEGMENTS = True
MIN_SLEEP_BOUT_S = 5 * 60  # sleep bouts shorter than this = "island" sleep
EVENT_BUFFER_MIN = 1       # NO hypopnea / apnea / desaturation within this many minutes before OR after a normal segment
PRE_BUFFER_S = EVENT_BUFFER_MIN * 60    # clear time BEFORE a normal segment
POST_BUFFER_S = EVENT_BUFFER_MIN * 60   # clear time AFTER a normal segment
BUFFER_AROUND_WAKE_N1 = False  # True = also keep the same clear time away from wake / N1 epochs

DESAT_DROP = 3             # % drop below running max
DESAT_MIN_S = 5
FLOW_DROP = 0.30           # >=30 % amplitude reduction
FLOW_MIN_S = 10

N1_THETA_MIN = 0.30        # theta ratio above this ...
N1_DELTA_MAX = 0.35        # ... with delta below this and alpha below wake level = N1
BAD_FRACTION = 0.20        # epoch is "bad data" if >20 % of SpO2 samples missing / flow flat
# ======================================


def read_edf(path):
    """Read a basic EDF file (EDF+ annotation channel is ignored later)."""
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256])
        nrec = int(h[236:244])
        dur = float(h[244:252])

        def read_fields(n):
            return [f.read(n).decode("latin1").strip() for _ in range(ns)]

        labels = read_fields(16)
        read_fields(80)
        units = read_fields(8)
        pmin = [float(x) for x in read_fields(8)]
        pmax = [float(x) for x in read_fields(8)]
        dmin = [float(x) for x in read_fields(8)]
        dmax = [float(x) for x in read_fields(8)]
        read_fields(80)
        samples = [int(x) for x in read_fields(8)]
        f.read(32 * ns)

        rec_samples = sum(samples)
        offsets = np.cumsum([0] + samples)
        raw = np.fromfile(f, dtype="<i2", count=rec_samples * nrec).reshape(nrec, rec_samples)

    channels = []
    for i in range(ns):
        x = raw[:, offsets[i]:offsets[i + 1]].reshape(-1).astype(float)
        denominator = dmax[i] - dmin[i]
        if denominator == 0:
            gain, off = 1.0, 0.0
            x = x * gain
        else:
            gain = (pmax[i] - pmin[i]) / denominator
            x = (x - dmin[i]) * gain + pmin[i]
        channels.append((labels[i], units[i], x, samples[i] / dur))

    return channels, nrec * dur


def pick(channels, keywords, unit_pref=None, exclude=()):
    for keyword in keywords:
        hits = [
            ch for ch in channels
            if keyword in ch[0].lower()
            and "annotation" not in ch[0].lower()
            and not any(e in ch[0].lower() for e in exclude)
        ]
        if hits:
            if unit_pref:
                preferred = [ch for ch in hits if ch[1].lower() == unit_pref.lower()]
                hits = preferred or hits
            label, unit, x, fs = hits[0]
            return x, fs, label
    return None


def runs(mask, min_len=1):
    """Continuous groups of True values -> list of (start, end)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def to_1hz(x, fs):
    """Average a signal down to 1 value per second (NaN-safe)."""
    step = max(1, int(round(fs)))
    n = (len(x) // step) * step
    return np.nanmean(x[:n].reshape(-1, step), axis=1) if n else np.array([])


def rolling_max(x, window):
    """Running max over the previous `window` samples, NaN-safe."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        v = x[max(0, i - window):i + 1]
        if np.any(np.isfinite(v)):
            out[i] = np.nanmax(v)
    return out


def detect_desats(spo2, fs):
    x = np.asarray(spo2, dtype=float).copy()
    x[x < 50] = np.nan
    s1 = to_1hz(x, fs)
    baseline = rolling_max(s1, 120)
    low = np.nan_to_num(baseline - s1, nan=0.0) >= DESAT_DROP
    return runs(low, DESAT_MIN_S)


def detect_flow_events(flow, fs):
    """Airflow reduction >= FLOW_DROP lasting >= FLOW_MIN_S (apnea / hypopnea candidates)."""
    if fs <= 0.25:
        raise ValueError("Flow sampling frequency is too low.")
    upper = min(1.0, 0.45 * fs)
    b, a = signal.butter(2, [0.1, upper], btype="bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, np.nan_to_num(flow))
    envelope = np.abs(signal.hilbert(filtered))

    env1 = to_1hz(envelope, fs)
    env1 = np.convolve(env1, np.ones(5) / 5, mode="same")

    half = 300
    baseline = np.array([
        np.percentile(env1[max(0, i - half):i + half], 85) for i in range(len(env1))
    ])
    low = env1 < (1 - FLOW_DROP) * baseline
    return runs(low, FLOW_MIN_S)


def band_ratios(segment, fs):
    f, p = signal.welch(segment, fs=fs, nperseg=min(len(segment), max(4, int(4 * fs))))
    total = p[(f >= 0.5) & (f <= 30)].sum()
    if total <= 0:
        return np.nan, np.nan, np.nan
    delta = p[(f >= 0.5) & (f < 4)].sum() / total
    theta = p[(f >= 4) & (f < 8)].sum() / total
    alpha = p[(f >= 8) & (f <= 12)].sum() / total
    return delta, theta, alpha


def epoch_features(x, fs, n_epochs):
    """Per epoch: delta, theta, alpha power ratios and RMS."""
    delta, theta, alpha, rms = [], [], [], []
    for e in range(n_epochs):
        seg = x[int(e * EPOCH_S * fs):int((e + 1) * EPOCH_S * fs)]
        if len(seg) < 8:
            delta.append(np.nan); theta.append(np.nan); alpha.append(np.nan); rms.append(np.nan)
            continue
        d, t, a = band_ratios(seg - np.mean(seg), fs)
        delta.append(d); theta.append(t); alpha.append(a)
        rms.append(np.sqrt(np.mean(seg ** 2)))
    return np.array(delta), np.array(theta), np.array(alpha), np.array(rms)


def bad_signal_epochs(spo2, flow, n_epochs):
    """Epochs with signal dropout / flat lines ("island" data)."""
    bad = np.zeros(n_epochs, dtype=bool)
    if spo2 is not None:
        x, fs, _ = spo2
        for e in range(n_epochs):
            seg = x[int(e * EPOCH_S * fs):int((e + 1) * EPOCH_S * fs)]
            if len(seg) == 0 or np.mean(~np.isfinite(seg) | (seg < 50)) > BAD_FRACTION:
                bad[e] = True
    if flow is not None:
        x, fs, _ = flow
        for e in range(n_epochs):
            seg = x[int(e * EPOCH_S * fs):int((e + 1) * EPOCH_S * fs)]
            if len(seg) == 0 or np.nanstd(seg) < 1e-9:
                bad[e] = True
    return bad


def hhmmss(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def split_stretch(start, end):
    """Cut a stretch into 3-minute pieces; keep a final piece if it is >= 2 min."""
    out, t = [], start
    while end - t >= SEG_MIN_S:
        length = min(SEG_MAX_S, end - t)
        out.append((t, t + length))
        t += length
    return out


def process_file(path, out_dir):
    """Run the full analysis on ONE EDF and save its two CSVs into out_dir.
    Returns the list of segments as (stretch_no, start_s, end_s)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EDF file not found: {path}")

    channels, total_seconds = read_edf(path)
    n_epochs = int(total_seconds // EPOCH_S)
    print(f"\nEDF file: {path}\nDuration: {total_seconds / 3600:.2f} hours")

    spo2 = pick(channels, ["spo2", "sao2", "spo_2", "saturation", "oxygen", "sat"])
    flow = pick(channels, ["flow patient", "airflow", "flow", "nasal", "cannula", "therm"],
                unit_pref="LPM", exclude=("leak", "xpap"))
    # exclude SpO2/SaO2 etc., because the "o2" keyword would otherwise match "SpO2"
    eeg = pick(channels, ["o1", "o2", "c3", "c4", "f3", "f4", "eeg"],
               exclude=("spo2", "sao2", "pao2", "flow", "emg", "eog", "ecg"))
    emg = pick(channels, ["chin", "emg"], exclude=("leg",))

    for name, ch in [("SpO2", spo2), ("Flow", flow), ("EEG", eeg), ("EMG", emg)]:
        print(f"{name}: {ch[2] + ' at ' + str(ch[1]) + ' Hz' if ch else 'NOT FOUND'}")

    if (spo2 is None and flow is None) or (eeg is None and emg is None):
        print("\nAvailable channels:")
        for label, unit, x, fs in channels:
            print(f"  {label}: {unit}, {fs} Hz")
        raise ValueError("Need SpO2 or Flow, and EEG or EMG.")

    # 1. Candidate events: desaturation + airflow reduction (obstructive apnea / hypopnea)
    events = []
    if spo2 is not None:
        oxygen, fs_sp, _ = spo2
        if np.any(np.isfinite(oxygen)) and np.nanmax(oxygen) <= 1.5:
            oxygen = oxygen * 100
        spo2 = (oxygen, fs_sp, spo2[2])
        events += detect_desats(oxygen, fs_sp)
    if flow is not None:
        events += detect_flow_events(flow[0], flow[1])
    print(f"\nCandidate events (desat + flow reduction): {len(events)}")

    # 2. Wake and N1 estimation
    wake = np.zeros(n_epochs, dtype=bool)
    n1 = np.zeros(n_epochs, dtype=bool)

    if eeg is not None:
        delta, theta, alpha, _ = epoch_features(eeg[0], eeg[1], n_epochs)
        valid = np.isfinite(alpha)
        if np.any(valid):
            thr = np.nanpercentile(alpha[valid], 70)
            wake_eeg = valid & (alpha > thr) & (alpha > 0.25)
            wake |= wake_eeg
            n1 |= (np.isfinite(theta) & ~wake_eeg
                   & (theta > N1_THETA_MIN) & (delta < N1_DELTA_MAX))

    if emg is not None:
        _, _, _, emg_rms = epoch_features(emg[0], emg[1], n_epochs)
        valid = np.isfinite(emg_rms)
        if np.any(valid):
            wake |= valid & (emg_rms > 2 * np.nanpercentile(emg_rms[valid], 50))

    n1 &= ~wake
    print(f"Estimated wake epochs: {wake.sum()}  |  estimated N1 epochs: {n1.sum()}")

    # 3. Sleep = not wake and not N1; remove short bouts (island sleep)
    sleep = ~wake & ~n1
    for s, e in runs(sleep):
        if (e - s) * EPOCH_S < MIN_SLEEP_BOUT_S:
            sleep[s:e] = False

    # 4. Remove bad-data epochs (dropouts / flat lines)
    sleep &= ~bad_signal_epochs(spo2, flow, n_epochs)

    # 5. Exclude buffer around candidate events
    ok = sleep.copy()
    all_events = list(events)
    if BUFFER_AROUND_WAKE_N1:       # treat wake / N1 epochs as events too
        all_events += [(s * EPOCH_S, e * EPOCH_S) for s, e in runs(wake | n1)]
    for onset, offset in all_events:
        a = max(0, int((onset - PRE_BUFFER_S) // EPOCH_S))
        b = min(n_epochs, int(np.ceil((offset + POST_BUFFER_S) / EPOCH_S)))
        ok[a:b] = False

    # 6. Normal stretches (whole stretches, or cut into 2-3 min pieces if SPLIT_INTO_SEGMENTS)
    stretches = [(s * EPOCH_S, e * EPOCH_S) for s, e in runs(ok)
                 if (e - s) * EPOCH_S >= MIN_STRETCH_S]
    segments = []
    for no, (s, e) in enumerate(stretches, 1):
        if SPLIT_INTO_SEGMENTS:
            for a, b in split_stretch(s, e):
                segments.append((no, a, b))
        else:
            segments.append((no, s, e))

    print(f"\nNormal stretches: {len(stretches)}")
    print(f"Normal segments saved: {len(segments)}")
    for no, (s, e) in enumerate(stretches, 1):
        print(f"Stretch {no}: {hhmmss(s)} - {hhmmss(e)} | {(e - s) / 60:.1f} min")

    # 7. Save CSVs (into this EDF's own output folder)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]

    seg_csv = os.path.join(out_dir, stem + "_normal_segments.csv")
    with open(seg_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Segment No", "Normal Stretch No", "Start (s)", "End (s)",
                    "Start (hh:mm:ss)", "End (hh:mm:ss)", "Duration (s)"])
        for n, (no, a, b) in enumerate(segments, 1):
            w.writerow([n, no, int(a), int(b), hhmmss(a), hhmmss(b), int(b - a)])
    print(f"\nSegments CSV saved: {seg_csv}")

    sum_csv = os.path.join(out_dir, stem + "_normal_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Normal Stretch No", "Start", "End", "Length (min)", "Number of segments"])
        for no, (s, e) in enumerate(stretches, 1):
            w.writerow([no, hhmmss(s), hhmmss(e), f"{(e - s) / 60:.1f}",
                        sum(1 for g in segments if g[0] == no)])
        w.writerow(["Total", "", "", f"{sum(e - s for s, e in stretches) / 60:.1f}", len(segments)])
    print(f"Summary CSV saved: {sum_csv}")

    # 8. Save everything the console printed for this EDF as one CSV row
    def describe(ch):
        return f"{ch[2]} at {ch[1]} Hz" if ch else "NOT FOUND"

    stats = {
        "EDF file": os.path.basename(path),
        "Duration (hours)": f"{total_seconds / 3600:.2f}",
        "SpO2 channel": describe(spo2),
        "Flow channel": describe(flow),
        "EEG channel": describe(eeg),
        "EMG channel": describe(emg),
        "Candidate events (desat + flow reduction)": len(events),
        "Estimated wake epochs": int(wake.sum()),
        "Estimated N1 epochs": int(n1.sum()),
        "Normal stretches": len(stretches),
        "Normal segments": len(segments),
        "Total normal time (min)": f"{sum(e - s for s, e in stretches) / 60:.1f}",
        "Stretches (start - end | minutes)": "; ".join(
            f"{hhmmss(s)}-{hhmmss(e)} | {(e - s) / 60:.1f}" for s, e in stretches),
    }
    out_csv = os.path.join(out_dir, stem + "_output_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.keys()))
        w.writeheader()
        w.writerow(stats)
    print(f"Output summary CSV saved: {out_csv}")

    return stats


def find_edfs(root):
    files = glob.glob(os.path.join(root, "**", "*.edf"), recursive=True)
    files += glob.glob(os.path.join(root, "**", "*.EDF"), recursive=True)
    unique = {}
    for f in files:
        full = os.path.abspath(f)
        unique[os.path.normcase(full)] = full
    return sorted(unique.values())


def read_segment_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main(root, out_root):
    root = os.path.abspath(root)
    out_root = root if SAVE_IN_PATIENT_FOLDER else os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)   # combined files (all_*.csv, batch_report.csv) go here

    edfs = [os.path.abspath(p) for p in EDF_PATHS] if EDF_PATHS else find_edfs(root)
    if MAX_FILES:
        edfs = edfs[:MAX_FILES]
    print(f"Files to process: {len(edfs)}")
    if SAVE_IN_PATIENT_FOLDER:
        print("CSV files will be saved inside each patient's own folder (next to its EDF).")
        print(f"Combined files will be saved in: {out_root}")
    else:
        print(f"CSV files will be saved under: {out_root}")
    if not edfs:
        print("No EDF files found. Check EDF_FOLDER.")
        return

    report = []
    all_rows = []
    output_rows = []   # one row per EDF with everything the console printed

    for i, edf in enumerate(edfs, 1):
        name = os.path.basename(edf)
        stem = os.path.splitext(name)[0]

        # this EDF's own output folder, e.g. normal_output\Batch_001\Patient_003
        rel_dir = os.path.relpath(os.path.dirname(edf), root)
        if rel_dir == "." or rel_dir.startswith(".."):
            rel_dir = stem                      # EDF sits directly in the root: use its own name
        if SAVE_IN_PATIENT_FOLDER:
            out_dir = os.path.dirname(edf)      # same folder as the EDF (the patient folder)
        else:
            out_dir = os.path.join(out_root, rel_dir)
        seg_csv = os.path.join(out_dir, stem + "_normal_segments.csv")

        print("\n" + "=" * 70)
        print(f"[{i}/{len(edfs)}] {edf}")

        out_csv = os.path.join(out_dir, stem + "_output_summary.csv")
        stats = None
        try:
            if SKIP_IF_DONE and os.path.isfile(seg_csv):
                status, note = "SKIPPED", "CSV already exists"
                print("  -> already done, skipping")
                if os.path.isfile(out_csv):
                    old = read_segment_rows(out_csv)
                    stats = old[0] if old else None
            else:
                stats = process_file(edf, out_dir)
                status, note = "OK", ""
        except Exception as exc:               # one bad file must not stop the loop
            status, note = "FAILED", f"{type(exc).__name__}: {exc}"
            print(f"  !! FAILED: {note}")
            traceback.print_exc()

        row = {"Patient folder": rel_dir, "Status": status}
        if stats:
            row.update(stats)
        else:
            row["EDF file"] = name
        row["Note"] = note
        output_rows.append(row)

        n_seg = 0
        if os.path.isfile(seg_csv):
            rows = read_segment_rows(seg_csv)
            n_seg = len(rows)
            for r in rows:
                all_rows.append({"EDF file": name, "Patient folder": rel_dir, **r})

        report.append({"EDF file": name, "Patient folder": rel_dir, "Status": status,
                       "Segments": n_seg, "Output folder": out_dir, "Note": note})

    # combined files
    if all_rows:
        combined = os.path.join(out_root, "all_normal_segments_updated.csv")
        with open(combined, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nCombined segments CSV: {combined}")

    # all console outputs of all EDFs in one CSV (one row per EDF)
    if output_rows:
        fields = []
        for r in output_rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        out_all = os.path.join(out_root, "all_output_summary.csv")
        with open(out_all, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(output_rows)
        print(f"All outputs CSV: {out_all}")

    rep = os.path.join(out_root, "batch_report.csv")
    with open(rep, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["EDF file", "Patient folder", "Status",
                                          "Segments", "Output folder", "Note"])
        w.writeheader()
        w.writerows(report)
    print(f"Batch report CSV: {rep}")

    ok = sum(r["Status"] == "OK" for r in report)
    skipped = sum(r["Status"] == "SKIPPED" for r in report)
    failed = sum(r["Status"] == "FAILED" for r in report)
    print(f"\nDone. OK: {ok}  |  Skipped: {skipped}  |  Failed: {failed}  |  "
          f"Total segments: {len(all_rows)}")


if __name__ == "__main__":
    root_arg = sys.argv[1] if len(sys.argv) > 1 else EDF_FOLDER
    out_arg = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FOLDER
    main(root_arg, out_arg)