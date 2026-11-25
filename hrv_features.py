
# import os
# import glob
# import numpy as np
# import pandas as pd
# import mne
# import neurokit2 as nk

# # ==========================================
# # PATHS
# # ==========================================

# ROOT_FOLDER = r"E:\sleep_apnea_data"

# OUTPUT_FILE = os.path.join(
#     ROOT_FOLDER,
#     "all_patients_HRV_features.csv"
# )

# # Set to None for automatic ECG channel detection.
# # If automatic detection fails, enter your actual channel name.
# ECG_CHANNEL = None


# # ==========================================
# # HRV CALCULATION
# # ==========================================

# def calculate_hrv(raw, start_sec, end_sec):

#     try:
#         sf = raw.info["sfreq"]

#         start_sec = float(start_sec)
#         end_sec = float(end_sec)

#         if start_sec < 0 or end_sec <= start_sec:
#             return None

#         if end_sec > raw.times[-1]:
#             return None

#         start_index = int(start_sec * sf)
#         end_index = int(end_sec * sf)

#         ecg = raw.get_data(
#             picks=[ECG_CHANNEL]
#         )[0]

#         segment = ecg[start_index:end_index]

#         if len(segment) < int(sf * 5):
#             return None

#         # Clean ECG
#         cleaned = nk.ecg_clean(
#             segment,
#             sampling_rate=sf
#         )

#         # Detect R-peaks
#         _, info = nk.ecg_peaks(
#             cleaned,
#             sampling_rate=sf
#         )

#         peaks = info["ECG_R_Peaks"]

#         if len(peaks) < 3:
#             return None

#         # Calculate RR intervals in milliseconds
#         rr = np.diff(peaks) / sf * 1000

#         if len(rr) < 2:
#             return None

#         # Basic RR plausibility check
#         if np.any((rr < 300) | (rr > 2000)):
#             return None

#         # Time-domain HRV features
#         mean_rr = np.mean(rr)
#         sdnn = np.std(rr, ddof=1)

#         diff_rr = np.diff(rr)

#         rmssd = np.sqrt(np.mean(diff_rr ** 2))

#         pnn50 = (
#             np.sum(np.abs(diff_rr) > 50)
#             / len(diff_rr)
#         ) * 100

#         # Poincare features
#         sd1 = rmssd / np.sqrt(2)

#         sd2_value = 2 * (sdnn ** 2) - (sd1 ** 2)

#         sd2 = (
#             np.sqrt(sd2_value)
#             if sd2_value >= 0
#             else np.nan
#         )

#         return {
#             "Mean_RR": mean_rr,
#             "SDNN": sdnn,
#             "RMSSD": rmssd,
#             "pNN50": pnn50,
#             "SD1": sd1,
#             "SD2": sd2
#         }

#     except Exception as e:
#         print("HRV calculation error:", e)
#         return None


# # ==========================================
# # IDENTIFY CONDITION AND TIME COLUMNS
# # ==========================================

# def get_condition_columns(filename):

#     filename = filename.lower()

#     if "hypopnea" in filename:
#         return (
#             "Hypopnea",
#             "Detected Hypopnea Start (s)",
#             "Detected Hypopnea End (s)",
#             "Event No"
#         )

#     elif "obstructive_apnea" in filename:
#         return (
#             "Obstructive_Apnea",
#             "Detected Obstructive Apnea Start (s)",
#             "Detected Obstructive Apnea End (s)",
#             "Event No"
#         )

#     elif "desaturation" in filename:
#         return (
#             "Desaturation",
#             "Detected Desaturation Start (s)",
#             "Detected Desaturation End (s)",
#             "Event No"
#         )

#     elif "normal" in filename:
#         return (
#             "Normal",
#             "Window Start (s)",
#             "Window End (s)",
#             "Window No"
#         )

#     return None


# # ==========================================
# # FIND ECG CHANNEL
# # ==========================================

# def find_ecg_channel(raw):

#     if ECG_CHANNEL is not None:

#         if ECG_CHANNEL in raw.ch_names:
#             return ECG_CHANNEL

#         raise ValueError(
#             f"Configured ECG channel '{ECG_CHANNEL}' not found. "
#             f"Available channels: {raw.ch_names}"
#         )

#     # Prefer channels identified by MNE as ECG
#     ecg_channels = [
#         name for name, channel_type
#         in zip(raw.ch_names, raw.get_channel_types())
#         if channel_type == "ecg"
#     ]

#     if ecg_channels:
#         return ecg_channels[0]

#     # Otherwise check channel names
#     for name in raw.ch_names:
#         if "ecg" in name.lower() or "ekg" in name.lower():
#             return name

#     raise ValueError(
#         "ECG channel not detected. Available channels: "
#         + str(raw.ch_names)
#     )


# # ==========================================
# # PROCESS ALL BATCHES AND PATIENTS
# # ==========================================

# all_results = []

# for batch_folder in sorted(
#     glob.glob(os.path.join(ROOT_FOLDER, "Batch_*"))
# ):

#     for patient_folder in sorted(
#         glob.glob(os.path.join(batch_folder, "Patient_*"))
#     ):

#         print("\n================================")
#         print("Processing:", patient_folder)
#         print("================================")

#         csv_files = sorted(
#             glob.glob(
#                 os.path.join(patient_folder, "*rsa*60s.csv")
#             )
#         )

#         if not csv_files:
#             print("No RSA CSV files found.")
#             continue

#         edf_files = glob.glob(
#             os.path.join(patient_folder, "*.edf")
#         )

#         if not edf_files:
#             print("No EDF file found.")
#             continue

#         # Match EDF to the CSV recording prefix
#         edf_file = edf_files[0]

#         try:
#             raw = mne.io.read_raw_edf(
#                 edf_file,
#                 preload=True,
#                 verbose=False
#             )

#             ECG_CHANNEL = find_ecg_channel(raw)

#             print("EDF:", os.path.basename(edf_file))
#             print("ECG channel:", ECG_CHANNEL)

#         except Exception as e:
#             print("Could not load ECG:", e)
#             continue

#         # ==================================
#         # PROCESS EACH RSA CSV
#         # ==================================

#         for csv_file in csv_files:

#             filename = os.path.basename(csv_file)

#             info = get_condition_columns(filename)

#             if info is None:
#                 continue

#             condition, start_col, end_col, id_col = info

#             print("\nReading:", filename)

#             try:
#                 df = pd.read_csv(csv_file)

#             except Exception as e:
#                 print("CSV reading error:", e)
#                 continue

#             required = [start_col, end_col]

#             if not all(col in df.columns for col in required):
#                 print("Required timing columns missing.")
#                 print("Available columns:", list(df.columns))
#                 continue

#             for index, row in df.iterrows():

#                 start_sec = row[start_col]
#                 end_sec = row[end_col]

#                 if pd.isna(start_sec) or pd.isna(end_sec):
#                     continue

#                 event_id = (
#                     row[id_col]
#                     if id_col in df.columns
#                     else index + 1
#                 )

#                 features = calculate_hrv(
#                     raw,
#                     start_sec,
#                     end_sec
#                 )

#                 if features is None:
#                     print(
#                         "Skipped:",
#                         condition,
#                         event_id
#                     )
#                     continue

#                 result = {
#                     "Batch": os.path.basename(batch_folder),
#                     "Patient": os.path.basename(patient_folder),
#                     "EDF_File": os.path.basename(edf_file),
#                     "Condition": condition,
#                     "Event_ID": event_id,
#                     "Start_sec": start_sec,
#                     "End_sec": end_sec
#                 }

#                 result.update(features)

#                 all_results.append(result)

#                 print(
#                     "Completed:",
#                     condition,
#                     event_id
#                 )

#         raw.close()


# # ==========================================
# # SAVE FINAL OUTPUT
# # ==========================================

# if all_results:

#     output_df = pd.DataFrame(all_results)

#     output_df.to_csv(
#         OUTPUT_FILE,
#         index=False
#     )

#     print("\n================================")
#     print("HRV PROCESSING COMPLETED")
#     print("================================")

#     print("Total calculated segments:", len(output_df))
#     print("Output file:", OUTPUT_FILE)

#     print("\nCondition-wise results:")
#     print(output_df["Condition"].value_counts())

# else:

#     print("\nNo HRV results generated.")
#     print("Check ECG channel and event durations.")
