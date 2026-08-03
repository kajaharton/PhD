# A script to remove all rows touching a track from genome_configtraits csv 
# Creates a new, track-free csv 
  
import pandas as pd
import pyranges as pr

# ------------------ FILE PATHS ------------------ #
bins_csv = "/PATH/TO/BINNED/CONFIGTRAIT_SPECIES.csv"

polyA_csvs = [
    "/PATH/TO/CSV/LOCI/TO/REMOVE/ATRACKS.csv",
    ...,
    ...
]

output_path = "/PATH/TO/OUTPUT/files.csv"

# ------------------ IMPORTANT COLUMNS ------------------ #
cols_to_keep = [
    "chromosome",
    "bin_start",
    "bin_end",
    "dH",
    "dS",
    "dG",
    "GCcont",
    "sequence",
    "cumulative_complexity"
]

# Memory-efficient dtypes
dtype_map = {
    "chromosome": "category",
    "bin_start": "int32",
    "bin_end": "int32",
    "dH": "float32",
    "dS": "float32",
    "dG": "float32",
    "GCcont": "float32",
    "sequence": "category",  # reduces memory if repeated sequences
    "cumulative_complexity": "float32"
}

# ------------------ LOAD TRACKS (SMALL) ------------------ #
tracks_list = []

for f in polyA_csvs:
    df = pd.read_csv(f, usecols=["chromosome", "start", "end"])
    df = df.rename(columns={
        "chromosome": "Chromosome",
        "start": "Start",
        "end": "End"
    })
    tracks_list.append(pr.PyRanges(df))

tracks_gr = pr.concat(tracks_list)

# ------------------ PROCESS BINS IN CHUNKS ------------------ #
chunk_size = 500_000  # safer starting point
first_chunk = True

for chunk in pd.read_csv(
    bins_csv,
    chunksize=chunk_size,
    usecols=cols_to_keep,
    dtype=dtype_map
):
    # Rename for PyRanges
    chunk = chunk.rename(columns={
        "chromosome": "Chromosome",
        "bin_start": "Start",
        "bin_end": "End"
    })

    # Create PyRanges object
    bins_gr = pr.PyRanges(chunk)

    # Remove overlaps
    filtered = bins_gr.overlap(tracks_gr, invert=True)

    # Convert back to DataFrame
    out_df = filtered.df.rename(columns={
        "Chromosome": "chromosome",
        "Start": "bin_start",
        "End": "bin_end"
    })

    # Write incrementally
    out_df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=first_chunk
    )

    first_chunk = False

print(f"Finished. Output saved to: {output_path}")
