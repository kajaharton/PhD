# A script to generate a permutational ensemble and randomly mutate it to hit the target GC content
# Need input: target GC content, path for the csv 
# Creates a csv file with a single column, of mutated permutational ensemble 

import itertools
import random
import csv

# ── Hard-coded parameters ─────────────────────────────────────────────────────
TARGET_GC   = 0.447                   # <-- target GC content (0.0 – 1.0)
OUTPUT_PATH = "/PATH/TO/OUTPUT/permens_10mer_0.447GCcontent.csv"    # <-- output file path
# ──────────────────────────────────────────────────────────────────────────────

NUCLEOTIDES = ["A", "T", "G", "C"]
GC = {"G", "C"}
AT = {"A", "T"}
K = 10  # k-mer length


def gc_content(sequences):
    total_bases = len(sequences) * K
    gc_count = sum(base in GC for seq in sequences for base in seq)
    return gc_count / total_bases


def adjust_gc(sequences, target_gc):
    # Work with mutable lists of characters
    seqs = [list(seq) for seq in sequences]
    total_bases = len(seqs) * K

    current_gc = gc_content(sequences)
    target_gc_count = round(target_gc * total_bases)

    # Build flat index of all (seq_idx, pos) tuples, split by AT / GC
    at_positions = [
        (i, j) for i, seq in enumerate(seqs) for j, base in enumerate(seq) if base in AT
    ]
    gc_positions = [
        (i, j) for i, seq in enumerate(seqs) for j, base in enumerate(seq) if base in GC
    ]

    current_gc_count = len(gc_positions)
    delta = target_gc_count - current_gc_count

    if delta > 0:
        # Need more GC: randomly pick AT positions and flip to G or C
        random.shuffle(at_positions)
        for i, j in at_positions[:delta]:
            seqs[i][j] = random.choice(["G", "C"])

    elif delta < 0:
        # Need less GC: randomly pick GC positions and flip to A or T
        random.shuffle(gc_positions)
        for i, j in gc_positions[: abs(delta)]:
            seqs[i][j] = random.choice(["A", "T"])

    return ["".join(seq) for seq in seqs]


def main():
    print("Generating all 10-mers...")
    sequences = ["".join(p) for p in itertools.product(NUCLEOTIDES, repeat=K)]
    print(f"  Total sequences : {len(sequences):,}")
    print(f"  Initial GC      : {gc_content(sequences):.4f}")

    print(f"  Target GC       : {TARGET_GC:.4f}")
    print("Adjusting GC content...")
    sequences = adjust_gc(sequences, TARGET_GC)
    print(f"  Final GC        : {gc_content(sequences):.4f}")

    print(f"Writing to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence"])
        for seq in sequences:
            writer.writerow([seq])

    print(f"Done. {len(sequences):,} sequences written to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
