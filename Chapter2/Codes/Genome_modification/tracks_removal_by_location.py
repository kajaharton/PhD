# A script to remove pre-defined fragments (csv file, chromosome, start, end columns) from a .fna file
# Used to remove mononucleotide tracks from reference genomes

import pandas as pd
import pysam
from pathlib import Path
from collections import defaultdict

# -----------------------------
# User inputs
# -----------------------------

fasta_path = "/PATH/TO/REFGENOME/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna"

csv_files = [
    "/PATH/TO/FILE/Atracks_homo_sapiens.csv",
    "/PATH/TO/FILE/Ttracks_homo_sapiens.csv",
    "/PATH/TO/FILE/Gtracks_homo_sapiens.csv",
    "/PATH/TO/FILE/Ctracks_homo_sapiens.csv"
]

output_fasta = "/PATH/TO/OUPUT/homo_sapiens_T2T-CHM13v2.0/GCA_009914755.4_T2T-CHM13v2.0_clean_nomononucleotidetracks.fna"

# -----------------------------
# Load and combine intervals
# -----------------------------

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df[["chromosome", "start", "end"]])

intervals = pd.concat(dfs, ignore_index=True)

# ensure integer
intervals["start"] = intervals["start"].astype(int)
intervals["end"] = intervals["end"].astype(int)

# -----------------------------
# Merge overlapping intervals
# -----------------------------

def merge_intervals(df):
    df = df.sort_values("start")
    merged = []
    for s, e in zip(df.start, df.end):
        if not merged:
            merged.append([s, e])
        else:
            last_s, last_e = merged[-1]
            if s <= last_e + 1:
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])
    return merged

merged_intervals = {}

for chrom, sub in intervals.groupby("chromosome"):
    merged_intervals[chrom] = merge_intervals(sub)

# -----------------------------
# Process FASTA
# -----------------------------

fasta = pysam.FastaFile(fasta_path)

with open(output_fasta, "w") as out:

    for chrom in fasta.references:

        seq = fasta.fetch(chrom)
        length = len(seq)

        if chrom not in merged_intervals:
            # nothing to remove
            out.write(f">{chrom}\n")
            for i in range(0, length, 60):
                out.write(seq[i:i+60] + "\n")
            continue

        intervals_chr = merged_intervals[chrom]

        pieces = []
        prev_end = 1

        for s, e in intervals_chr:
            # convert to 0-based
            keep_start = prev_end - 1
            keep_end = s - 1
            if keep_end > keep_start:
                pieces.append(seq[keep_start:keep_end])
            prev_end = e + 1

        # tail
        if prev_end <= length:
            pieces.append(seq[prev_end-1:])

        new_seq = "".join(pieces)

        out.write(f">{chrom}\n")
        for i in range(0, len(new_seq), 60):
            out.write(new_seq[i:i+60] + "\n")

fasta.close()

print("Finished. Output:", output_fasta)
