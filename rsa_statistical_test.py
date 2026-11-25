
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# --------------------------------------------------
# 1. MAIN FOLDER
# --------------------------------------------------

DATA_FOLDER = r"E:\sleep_apnea_data"

OUTPUT_FOLDER = os.path.join(
    DATA_FOLDER, "rsa_statistical_results"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# 2. EVENT TYPES AND ACTUAL CSV COLUMN NAMES
# --------------------------------------------------

EVENT_COLUMNS = {
    "hypopnea": {
        "P2T": [
            "Pre-Hypopnea RSA P2T Mean (ms)",
            "Para-Hypopnea RSA P2T Mean (ms)",
            "Post-Hypopnea RSA P2T Mean (ms)"
        ],
        "PB": [
            "Pre-Hypopnea RSA PorgesBohrer (ln ms^2)",
            "Para-Hypopnea RSA PorgesBohrer (ln ms^2)",
            "Post-Hypopnea RSA PorgesBohrer (ln ms^2)"
        ]
    },

    "obstructive_apnea": {
        "P2T": [
            "Pre-Obstructive Apnea RSA P2T Mean (ms)",
            "Para-Obstructive Apnea RSA P2T Mean (ms)",
            "Post-Obstructive Apnea RSA P2T Mean (ms)"
        ],
        "PB": [
            "Pre-Obstructive Apnea RSA PorgesBohrer (ln ms^2)",
            "Para-Obstructive Apnea RSA PorgesBohrer (ln ms^2)",
            "Post-Obstructive Apnea RSA PorgesBohrer (ln ms^2)"
        ]
    },

    "desaturation": {
        "P2T": [
            "Pre-Desaturation RSA P2T Mean (ms)",
            "Para-Desaturation RSA P2T Mean (ms)",
            "Post-Desaturation RSA P2T Mean (ms)"
        ],
        "PB": [
            "Pre-Desaturation RSA PorgesBohrer (ln ms^2)",
            "Para-Desaturation RSA PorgesBohrer (ln ms^2)",
            "Post-Desaturation RSA PorgesBohrer (ln ms^2)"
        ]
    }
}


# --------------------------------------------------
# 3. FIND CSV FILES RECURSIVELY
# --------------------------------------------------

PATTERNS = {
    "hypopnea": "*_new_hypopnea_rsa_60s.csv",
    "obstructive_apnea": "*_new_obstructive_apnea_rsa_60s.csv",
    "desaturation": "*_new_desaturation_rsa_60s.csv",
    "normal": "*_normal_rsa_60s.csv"
}

all_files = {}

for event_type, pattern in PATTERNS.items():

    files = glob.glob(
        os.path.join(DATA_FOLDER, "**", pattern),
        recursive=True
    )

    all_files[event_type] = files

    print("\nEvent:", event_type)
    print("Files found:", len(files))


# --------------------------------------------------
# 4. CHECK UNIQUE PATIENT FOLDERS AND PATIENT IDS
# --------------------------------------------------

patient_folders = glob.glob(
    os.path.join(DATA_FOLDER, "Batch_*", "Patient_*")
)

print("\nTotal patient folders:", len(patient_folders))


# --------------------------------------------------
# 5. CREATE PATIENT-LEVEL EVENT DATA
# --------------------------------------------------

patient_records = []

for event_type in EVENT_COLUMNS:

    for file in all_files[event_type]:

        filename = os.path.basename(file)

        patient_id = filename.split("_new_")[0]

        try:
            df = pd.read_csv(file)

            record = {
                "Patient_ID": patient_id,
                "Event_Type": event_type
            }

            for feature, columns in EVENT_COLUMNS[event_type].items():

                for phase, column in zip(
                    ["Pre", "Para", "Post"], columns
                ):

                    if column not in df.columns:
                        raise ValueError(
                            f"Missing column: {column} in {filename}"
                        )

                    values = pd.to_numeric(
                        df[column], errors="coerce"
                    )

                    record[f"{feature}_{phase}"] = values.mean()

            patient_records.append(record)

        except Exception as e:
            print("\nError reading:", file)
            print(e)


patient_df = pd.DataFrame(patient_records)


if patient_df.empty:
    raise SystemExit(
        "No patient event data created. Check the CSV files."
    )


patient_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER, "patient_level_event_phase_data.csv"
    ),
    index=False
)

print("\nPatient-level event data created.")
print(patient_df.head())


# --------------------------------------------------
# 6. HOLM CORRECTION FUNCTION
# --------------------------------------------------

def holm_correction(p_values):

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    order = np.argsort(p_values)

    adjusted = np.empty(n, dtype=float)

    running_max = 0

    for rank, index in enumerate(order):

        corrected = (n - rank) * p_values[index]

        running_max = max(running_max, corrected)

        adjusted[index] = min(running_max, 1.0)

    return adjusted


# --------------------------------------------------
# 7. FRIEDMAN AND WILCOXON TESTS
# --------------------------------------------------

friedman_results = []
wilcoxon_results = []

comparisons = [
    ("Pre", "Para"),
    ("Para", "Post"),
    ("Pre", "Post")
]

for event_type in EVENT_COLUMNS:

    event_data = patient_df[
        patient_df["Event_Type"] == event_type
    ].copy()

    for feature in ["P2T", "PB"]:

        columns = [
            f"{feature}_Pre",
            f"{feature}_Para",
            f"{feature}_Post"
        ]

        # Keep patients having all three phase values
        complete_data = event_data[
            ["Patient_ID"] + columns
        ].dropna(subset=columns)

        pre = complete_data[columns[0]].to_numpy()
        para = complete_data[columns[1]].to_numpy()
        post = complete_data[columns[2]].to_numpy()

        n = len(complete_data)

        print(
            f"\n{event_type} - {feature}: "
            f"{n} complete patients"
        )

        if n < 2:
            print("Not enough complete patients.")
            continue

        # Friedman test
        try:
            statistic, p_value = friedmanchisquare(
                pre, para, post
            )

            friedman_results.append({
                "Event_Type": event_type,
                "Feature": feature,
                "N_Patients": n,
                "Friedman_Statistic": statistic,
                "P_Value": p_value,
                "Significant_0.05": p_value < 0.05
            })

            print("Friedman statistic:", statistic)
            print("Friedman p-value:", p_value)

        except Exception as e:
            print("Friedman error:", e)
            continue

        # Wilcoxon pairwise tests
        raw_p_values = []
        comparison_results = []

        for phase1, phase2 in comparisons:

            col1 = f"{feature}_{phase1}"
            col2 = f"{feature}_{phase2}"

            try:
                result = wilcoxon(
                    complete_data[col1],
                    complete_data[col2],
                    alternative="two-sided",
                    method="auto"
                )

                raw_p_values.append(result.pvalue)

                comparison_results.append({
                    "Event_Type": event_type,
                    "Feature": feature,
                    "Comparison": f"{phase1} vs {phase2}",
                    "N_Patients": n,
                    "Wilcoxon_Statistic": result.statistic,
                    "P_Value": result.pvalue
                })

            except ValueError as e:
                print(
                    f"Wilcoxon error for "
                    f"{event_type}, {feature}, "
                    f"{phase1} vs {phase2}: {e}"
                )

        # Holm correction within each event-feature family
        if raw_p_values:

            adjusted_p = holm_correction(raw_p_values)

            for result, adjusted in zip(
                comparison_results, adjusted_p
            ):

                result["Holm_Adjusted_P"] = adjusted

                result["Significant_0.05"] = (
                    adjusted < 0.05
                )

                wilcoxon_results.append(result)


# --------------------------------------------------
# 8. SAVE STATISTICAL RESULTS
# --------------------------------------------------

friedman_df = pd.DataFrame(friedman_results)

wilcoxon_df = pd.DataFrame(wilcoxon_results)

friedman_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "friedman_results.csv"),
    index=False
)

wilcoxon_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "wilcoxon_results.csv"),
    index=False
)


# --------------------------------------------------
# 9. CREATE PATIENT-LEVEL NORMAL DATA
# --------------------------------------------------

normal_records = []

for file in all_files["normal"]:

    filename = os.path.basename(file)

    patient_id = filename.split("_normal_rsa_60s")[0]

    try:
        df = pd.read_csv(file)

        p2t_col = "RSA P2T Mean (ms)"
        pb_col = "RSA Porges-Bohrer"

        if p2t_col not in df.columns or pb_col not in df.columns:
            print("Normal CSV columns missing:", filename)
            continue

        normal_records.append({
            "Patient_ID": patient_id,
            "Normal_P2T_Mean": pd.to_numeric(
                df[p2t_col], errors="coerce"
            ).mean(),
            "Normal_PB_Mean": pd.to_numeric(
                df[pb_col], errors="coerce"
            ).mean()
        })

    except Exception as e:
        print("Normal CSV error:", filename, e)


normal_df = pd.DataFrame(normal_records)

normal_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER, "normal_patient_level_RSA_data.csv"
    ),
    index=False
)


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n----------------------------------")
print("STATISTICAL ANALYSIS COMPLETED")
print("----------------------------------")

print("\nFriedman results:")
print(friedman_df)

print("\nWilcoxon results:")
print(wilcoxon_df)

print("\nNormal patient-level data:")
print(normal_df.head())

print("\nUnique patient IDs per category:")

for event_type, files in all_files.items():

    ids = set()

    for file in files:

        name = os.path.basename(file)

        if event_type == "normal":
            patient_id = name.split("_normal_rsa_60s")[0]
        else:
            patient_id = name.split("_new_")[0]

        ids.add(patient_id)

    print(event_type, ":", len(ids))

print("\nOutput saved in:")
print(OUTPUT_FOLDER)
