

import os

import re

import warnings

import traceback



import numpy as np

import pandas as pd

import mne

import neurokit2 as nk



warnings.filterwarnings("ignore")



# ============================================================

# SETTINGS

# ============================================================



ROOT = r"E:\sleep_apnea_data"



WIN_S = 60



# Minimum number of R-peaks required in a window

MIN_RPEAKS = 10



# ============================================================

# IMPORT RSA FUNCTION

# ============================================================



try:

    from neurokit2.ecg.ecg_rsa import ecg_rsa

except ImportError:

    ecg_rsa = getattr(nk, "ecg_rsa", None)



    if ecg_rsa is None:

        ecg_rsa = nk.hrv_rsa





# ============================================================

# CHANNEL SELECTION

# ============================================================



def normalize_name(name):

    return re.sub(r"[^a-z0-9]", "", str(name).lower())





def find_channel(channel_names, candidates):

    """

    Find a channel using normalized channel names.

    """



    normalized = {

        normalize_name(name): name

        for name in channel_names

    }



    # Exact matching first

    for candidate in candidates:

        key = normalize_name(candidate)



        if key in normalized:

            return normalized[key]



    # Partial matching next

    for candidate in candidates:

        key = normalize_name(candidate)



        for normalized_name, original_name in normalized.items():

            if key and key in normalized_name:

                return original_name



    return None





def find_ecg_channel(channel_names):

    return find_channel(

        channel_names,

        ["ECG II", "ECG2", "ECGII"]

    )





def find_flow_channel(channel_names):

    """

    Prefer CFlow for CPAP recordings.

    Use PFlow for PSG recordings.

    """



    cflow = find_channel(

        channel_names,

        ["CFlow", "Flow Patient CFlow"]

    )



    pflow = find_channel(

        channel_names,

        ["PFlow", "Flow Patient PFlow"]

    )



    if cflow is not None:

        return cflow



    if pflow is not None:

        return pflow



    # Additional fallback labels

    return find_channel(

        channel_names,

        [

            "Flow Patient-0",

            "Flow Patient-1",

            "Flow Patient-2"

        ]

    )





# ============================================================

# FIND MATCHING NORMAL SEGMENT CSV

# ============================================================



def find_normal_csv(edf_path):



    folder = os.path.dirname(edf_path)



    stem = os.path.splitext(

        os.path.basename(edf_path)

    )[0]



    csv_name = stem + "_normal_segments.csv"



    # First search in the same folder as the EDF

    local_csv = os.path.join(folder, csv_name)



    if os.path.isfile(local_csv):

        return local_csv



    # If not found, search recursively in ROOT

    matches = []



    for current_root, dirs, files in os.walk(ROOT):



        if csv_name in files:

            matches.append(

                os.path.join(current_root, csv_name)

            )



    if len(matches) == 1:

        return matches[0]



    if len(matches) > 1:

        print("  Multiple matching normal CSV files found:")

        for path in matches:

            print("   ", path)



        print("  Skipping this EDF to avoid using the wrong CSV.")

        return None



    return None





# ============================================================

# RSA CALCULATION FOR ONE 60-SECOND WINDOW

# ============================================================



def rsa_window(ecg_signal, flow_signal, rpeaks, fs):



    result = {

        "RSA P2T Mean (ms)": np.nan,

        "RSA Porges-Bohrer": np.nan

    }



    try:



        ecg_signal = np.asarray(ecg_signal, dtype=float)

        flow_signal = np.asarray(flow_signal, dtype=float)



        if len(ecg_signal) < int(30 * fs):

            return result



        if len(flow_signal) < int(30 * fs):

            return result



        if not np.all(np.isfinite(ecg_signal)):

            return result



        if not np.all(np.isfinite(flow_signal)):

            return result



        # Local R-peak indices for this window

        local_peaks = np.asarray(rpeaks, dtype=int)



        local_peaks = local_peaks[

            (local_peaks >= 0) &

            (local_peaks < len(ecg_signal))

        ]



        if len(local_peaks) < MIN_RPEAKS:

            return result



        # Prepare R-peak dictionary

        peak_dict = {

            "ECG_R_Peaks": local_peaks

        }



        # Process respiration

        rsp_signals, rsp_info = nk.rsp_process(

            flow_signal,

            sampling_rate=fs

        )



        # Calculate RSA using the same NeuroKit2 method

            # Create ECG DataFrame

        ecg_df = pd.DataFrame({

            "ECG_R_Peaks": np.zeros(len(ecg_signal), dtype=int)

        })



        # Mark R-peak positions

        ecg_df.loc[local_peaks, "ECG_R_Peaks"] = 1



        # Calculate RSA

        rsa_result = ecg_rsa(

            ecg_df,

            rsp_signals,

            peak_dict,

            sampling_rate=fs,

            continuous=False

        )



        if rsa_result is None:

            return result



        if isinstance(rsa_result, pd.DataFrame):

            rsa_values = rsa_result.iloc[0].to_dict()



        elif isinstance(rsa_result, dict):

            rsa_values = rsa_result



        else:

            return result



        # Extract P2T

        p2t = rsa_values.get(

            "RSA_P2T_Mean",

            np.nan

        )



        # Extract Porges-Bohrer

        pb = rsa_values.get(

            "RSA_PorgesBohrer",

            np.nan

        )



        # Convert P2T to milliseconds if returned in seconds

        if pd.notna(p2t):



            p2t = float(p2t)



            if abs(p2t) < 5:

                p2t = p2t * 1000



        if pd.notna(pb):

            pb = float(pb)



        result["RSA P2T Mean (ms)"] = p2t

        result["RSA Porges-Bohrer"] = pb



        return result



    except Exception as e:



        print("    RSA window error:", str(e))



        return result





# ============================================================

# PROCESS ONE EDF FILE

# ============================================================



def process_edf(edf_path):



    folder = os.path.dirname(edf_path)



    stem = os.path.splitext(

        os.path.basename(edf_path)

    )[0]



    print("\n" + "=" * 65)

    print("Processing:", os.path.basename(edf_path))

    print("=" * 65)



    # --------------------------------------------------------

    # Find corresponding normal segment CSV

    # --------------------------------------------------------



    normal_csv = find_normal_csv(edf_path)



    if normal_csv is None:



        print("  Normal segment CSV not found. Skipping EDF.")

        return



    print("  Normal CSV:", normal_csv)



    # --------------------------------------------------------

    # Read normal segment CSV

    # --------------------------------------------------------



    segments_df = pd.read_csv(normal_csv)



    required_columns = [

        "Start (s)",

        "End (s)"

    ]



    for column in required_columns:



        if column not in segments_df.columns:



            print("  Missing column:", column)

            print("  Skipping EDF.")

            return



    if "Segment No" not in segments_df.columns:

        segments_df["Segment No"] = np.arange(

            1, len(segments_df) + 1

        )



    # --------------------------------------------------------

    # Load EDF

    # --------------------------------------------------------



    print("  Loading EDF...")



    raw = mne.io.read_raw_edf(

        edf_path,

        preload=True,

        verbose="ERROR"

    )



    fs = float(raw.info["sfreq"])



    channel_names = raw.ch_names



    # --------------------------------------------------------

    # Find ECG and respiration channels

    # --------------------------------------------------------



    ecg_channel = find_ecg_channel(channel_names)

    flow_channel = find_flow_channel(channel_names)



    if ecg_channel is None:



        print("  ECG II channel not found.")

        return



    if flow_channel is None:



        print("  CFlow/PFlow channel not found.")

        return



    print("  ECG channel:", ecg_channel)

    print("  Flow channel:", flow_channel)

    print("  Sampling rate:", fs)



    # --------------------------------------------------------

    # Extract signals

    # --------------------------------------------------------



    ecg = raw.get_data(

        picks=[ecg_channel]

    )[0]



    flow = raw.get_data(

        picks=[flow_channel]

    )[0]



    # --------------------------------------------------------

    # Standardize respiration signal

    # Same whole-recording standardization as event code

    # --------------------------------------------------------



    flow_mean = np.nanmean(flow)

    flow_std = np.nanstd(flow)



    if not np.isfinite(flow_std) or flow_std == 0:



        print("  Invalid respiration signal.")

        return



    flow_z = (flow - flow_mean) / flow_std



    # --------------------------------------------------------

    # Clean ECG

    # --------------------------------------------------------



    print("  Cleaning ECG...")



    ecg_clean = nk.ecg_clean(

        ecg,

        sampling_rate=fs

    )



    # --------------------------------------------------------

    # Detect R-peaks once for the complete recording

    # --------------------------------------------------------



    print("  Detecting R-peaks...")



    _, peak_info = nk.ecg_peaks(

        ecg_clean,

        sampling_rate=fs

    )



    all_rpeaks = np.asarray(

        peak_info["ECG_R_Peaks"],

        dtype=int

    )



    print("  Total R-peaks:", len(all_rpeaks))



    # --------------------------------------------------------

    # Calculate RSA for each normal segment

    # --------------------------------------------------------



    rows = []



    for index, segment in segments_df.iterrows():



        segment_no = segment["Segment No"]



        start_s = pd.to_numeric(

            pd.Series([segment["Start (s)"]]),

            errors="coerce"

        ).iloc[0]



        end_s = pd.to_numeric(

            pd.Series([segment["End (s)"]]),

            errors="coerce"

        ).iloc[0]



        if pd.isna(start_s) or pd.isna(end_s):

            continue



        start_s = float(start_s)

        end_s = float(end_s)



        if start_s < 0 or end_s <= start_s:

            continue



        # Do not go beyond recording duration

        recording_duration = min(

            len(ecg_clean),

            len(flow_z)

        ) / fs



        end_s = min(end_s, recording_duration)



        if end_s <= start_s:

            continue



        segment_duration = end_s - start_s



        print(

            f"  Segment {segment_no}: "

            f"{start_s:.1f} to {end_s:.1f} seconds"

        )



        # Consecutive non-overlapping 60-second windows

        window_number = 1



        window_start = start_s



        while window_start + WIN_S <= end_s:



            window_end = window_start + WIN_S



            start_sample = int(round(window_start * fs))

            end_sample = int(round(window_end * fs))



            ecg_window = ecg_clean[

                start_sample:end_sample

            ]



            flow_window = flow_z[

                start_sample:end_sample

            ]



            # Select R-peaks belonging to this window

            window_peaks = all_rpeaks[

                (all_rpeaks >= start_sample) &

                (all_rpeaks < end_sample)

            ]



            # Convert absolute R-peak positions to local positions

            local_peaks = window_peaks - start_sample



            # Calculate RSA

            rsa_values = rsa_window(

                ecg_window,

                flow_window,

                local_peaks,

                fs

            )



            rows.append({



                "Segment No": segment_no,



                "Window No": window_number,



                "Window Start (s)": window_start,



                "Window End (s)": window_end,



                "Window Duration (s)": WIN_S,



                "RSA P2T Mean (ms)": rsa_values[

                    "RSA P2T Mean (ms)"

                ],



                "RSA Porges-Bohrer": rsa_values[

                    "RSA Porges-Bohrer"

                ]



            })



            window_number += 1

            window_start += WIN_S



    # ========================================================

    # SAVE PER-WINDOW RSA CSV

    # ========================================================



    result_columns = [

        "Segment No",

        "Window No",

        "Window Start (s)",

        "Window End (s)",

        "Window Duration (s)",

        "RSA P2T Mean (ms)",

        "RSA Porges-Bohrer"

    ]



    result_df = pd.DataFrame(

        rows,

        columns=result_columns

    )



    output_csv = os.path.join(

        folder,

        stem + "_normal_rsa_60s.csv"

    )



    result_df.to_csv(

        output_csv,

        index=False

    )



    print("  RSA CSV saved:", output_csv)



    # ========================================================

    # CREATE SEGMENT-WISE SUMMARY

    # ========================================================



    summary_rows = []



    if not result_df.empty:



        for segment_no, group in result_df.groupby(

            "Segment No",

            sort=False

        ):



            p2t = pd.to_numeric(

                group["RSA P2T Mean (ms)"],

                errors="coerce"

            ).dropna()



            pb = pd.to_numeric(

                group["RSA Porges-Bohrer"],

                errors="coerce"

            ).dropna()



            summary_rows.append({



                "Segment No": segment_no,



                "Number of windows": len(group),



                "Valid P2T windows": len(p2t),



                "Valid PB windows": len(pb),



                "Analysed window duration (s)": group[

                    "Window Duration (s)"

                ].sum(),



                "Mean P2T RSA (ms)": p2t.mean(),



                "Median P2T RSA (ms)": p2t.median(),



                "SD P2T RSA (ms)": p2t.std(),



                "Mean PB RSA": pb.mean(),



                "Median PB RSA": pb.median(),



                "SD PB RSA": pb.std()



            })



    summary_columns = [

        "Segment No",

        "Number of windows",

        "Valid P2T windows",

        "Valid PB windows",

        "Analysed window duration (s)",

        "Mean P2T RSA (ms)",

        "Median P2T RSA (ms)",

        "SD P2T RSA (ms)",

        "Mean PB RSA",

        "Median PB RSA",

        "SD PB RSA"

    ]



    summary_df = pd.DataFrame(

        summary_rows,

        columns=summary_columns

    )



    summary_csv = os.path.join(

        folder,

        stem + "_normal_rsa_summary.csv"

    )



    summary_df.to_csv(

        summary_csv,

        index=False

    )



    print("  Summary CSV saved:", summary_csv)



    print("  Total normal segments:", len(segments_df))

    print("  Total RSA windows:", len(result_df))





# ============================================================

# MAIN BATCH LOOP

# ============================================================



def main():



    print("\nStarting batch normal RSA processing...")



    edf_files = []



    # Find all EDF files recursively

    for current_root, dirs, files in os.walk(ROOT):



        for filename in files:



            if filename.lower().endswith(".edf"):



                edf_files.append(

                    os.path.join(current_root, filename)

                )



    edf_files.sort()



    print("\nTotal EDF files found:", len(edf_files))



    if not edf_files:



        print("No EDF files found. Check ROOT path.")

        return



    successful = 0

    skipped_or_failed = 0
    already_done = 0



    # Process every EDF file

    for number, edf_path in enumerate(edf_files, start=1):



        print("\n----------------------------------------------")

        print(f"[{number}/{len(edf_files)}]")

        print("----------------------------------------------")



        # Skip this EDF only when both RSA output files already exist

        folder = os.path.dirname(edf_path)
        stem = os.path.splitext(os.path.basename(edf_path))[0]

        output_csv = os.path.join(
            folder,
            stem + "_normal_rsa_60s.csv"
        )

        summary_csv = os.path.join(
            folder,
            stem + "_normal_rsa_summary.csv"
        )

        if os.path.isfile(output_csv) and os.path.isfile(summary_csv):
            print("  -> Already processed. Skipping this EDF.")
            already_done += 1
            continue



        try:



            process_edf(edf_path)

            successful += 1



        except Exception:



            skipped_or_failed += 1



            print("\nError while processing:")

            print(edf_path)



            traceback.print_exc()



    print("\n==============================================")

    print("BATCH PROCESSING COMPLETED")

    print("==============================================")



    print("Total EDF files:", len(edf_files))

    print("Processed without an uncaught error:", successful)

    print("Already completed and skipped:", already_done)

    print("Failed with an uncaught error:", skipped_or_failed)





# ============================================================

# RUN

# ============================================================



if __name__ == "__main__":

    main()



