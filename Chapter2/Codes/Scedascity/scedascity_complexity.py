# A script to subsample cumulative complexity and average the Cc over multiple fragment lengths
# Needs: folder with csv's with the complexity calculated for each bin over a reference genome
# Generates: 
#   a csv file for random locations
#   a csv file with the locations, and average Cc from the subsequent bins; rows as locations

import os
import pandas as pd
import numpy as np

# ======================
# USER INPUT
# ======================
INPUT_FOLDER = "/PATH/TO/FOLDER/WITH/CSV/WITH/CUMULATIVEENTROPY"
OUTPUT_FOLDER = "/PATH/TO/OUTPUT/FOLDER"

TOTAL_SAMPLES = 10000
BIN_SIZE = 10

WINDOW_SIZES_BP = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

# ======================
# SETUP
# ======================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

WINDOW_SIZES_BINS = [w // BIN_SIZE for w in WINDOW_SIZES_BP]
MAX_WINDOW_BINS = max(WINDOW_SIZES_BINS)

OUTPUT_LOCATIONS = os.path.join(OUTPUT_FOLDER, "random_locations.csv")
OUTPUT_MATRIX = os.path.join(OUTPUT_FOLDER, "complexity_windows.csv")

# ======================
# STEP 1: LOAD DATA
# ======================
print("Loading chromosome data...")

chr_data = {}
chr_lengths = {}

for file in os.listdir(INPUT_FOLDER):
    if not file.endswith(".csv"):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)

    df = pd.read_csv(filepath)

    chr_name = str(df["Chromosome"].iloc[0])
    entropy_array = df["Cumulative_Entropy"].values

    n_bins = len(entropy_array)

    valid_bins = n_bins - MAX_WINDOW_BINS

    if valid_bins <= 0:
        print(f"Skipping {chr_name} (too short for largest window)")
        continue

    chr_data[chr_name] = entropy_array
    chr_lengths[chr_name] = valid_bins

print(f"Loaded {len(chr_data)} chromosomes.")

# ======================
# STEP 2: SAMPLE LOCATIONS
# ======================
print("Sampling genome-wide locations...")

chromosomes = list(chr_lengths.keys())
lengths = np.array([chr_lengths[c] for c in chromosomes], dtype=float)

# probability proportional to chromosome length
probabilities = lengths / lengths.sum()

sampled_chr = np.random.choice(
    chromosomes,
    size=TOTAL_SAMPLES,
    p=probabilities
)

locations = []

for chr_name in sampled_chr:
    max_index = chr_lengths[chr_name]

    idx = np.random.randint(0, max_index)
    start_bp = idx * BIN_SIZE

    locations.append({
        "chromosome": chr_name,
        "bin_index": idx,
        "start": start_bp
    })

locations_df = pd.DataFrame(locations)
locations_df.to_csv(OUTPUT_LOCATIONS, index=False)

print(f"Saved locations → {OUTPUT_LOCATIONS}")

# ======================
# STEP 3: COMPUTE WINDOWS
# ======================
print("Computing complexity windows...")

results = []

grouped = locations_df.groupby("chromosome")

for chr_name, group in grouped:
    entropy = chr_data[chr_name]

    for _, row in group.iterrows():
        idx = int(row["bin_index"])

        result_row = {
            "chromosome": chr_name,
            "start": int(row["start"])
        }

        for w_bp, w_bins in zip(WINDOW_SIZES_BP, WINDOW_SIZES_BINS):
            window = entropy[idx: idx + w_bins]
            result_row[f"{w_bp}bp"] = np.mean(window)

        results.append(result_row)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_MATRIX, index=False)

print(f"Saved results → {OUTPUT_MATRIX}")

print("Done.")
