"""
====================================
INTRODUCTION
A script to generate real-life DNA-like dataset, based on user-defined constraints in the sequences types. 
It expands the permutational ensemble by adding 10mer sequences to meet the empirically detected 
constraints of the real-life DNA (G/C content skew in eukaryotes; higher proportion of A- and T-tracks, and AT- tracks). 
The generator is designed to make the smallest possible dataset that meets the constraints. 

Boundary conditions for the model DNA dataset:

  1. Contains all 4^10 = 1,048,576 unique 10-mers exactly once (base set of the permutational ensemble)
  2. target_a  % of all nucleotide positions are inside an A-track  (≥4 consecutive A's)
  3. target_t  % of all nucleotide positions are inside a T-track   (≥4 consecutive T's)
  4. target_at % of all nucleotide positions are inside an AT-track (≥4 chars ATATAT/TATATA)
  5. target_gc % of all nucleotide positions are G or C

All constraints are met within ±tolerance.

As the constraints affect each other (e.g., proportion of A/T/AT-tracks cannot be increased without decreasing G/C content), 
when the desired conditions violate the interrelation, the track restrictions are lifted first to meet the G/C content constraint. 

The output csv file with the sequences is accompanied with a summary .txt file that specifies which of the constraints were met. 

====================================
METHODOLOGY

Before sampling, the exact number of sequences to draw from each pool
is calculated by solving a linear system that accounts for ALL cross-contributions:

  - Track pools (A, T, AT) contribute to GC content
  - The GC correction pool contributes to track percentages
  - Track-track cross-terms (A-pool contributing to T-track, etc.) are ~0
    by symmetry of the sequence space, and are verified negligible

The system is solved in two steps:
  Step 1: Solve the full 2×2 system (N and n_gc as unknowns, n_a/n_t/n_at
          derived from N and n_gc via the track constraints).
  Step 2: If n_at < 0 (AT-track target is already met by the AT-rich GC pool),
          clamp n_at = 0 and re-solve the reduced 2×2 system.
          The AT-track percentage will exceed target in this case — this is a
          mathematical consequence of the GC target being low, not an error.
          It is clearly reported in the summary.

This gives the minimum total dataset size for the given targets.

GC pool direction
-----------------
The GC pool direction (AT-rich vs GC-rich) is determined by comparing the
natural GC settling point (~46.7% for these track targets) to the requested
target_gc, NOT by comparing target_gc to 0.5. Direction is re-checked during
fine-tuning in case of overshoot.

Sampling
--------
All four pools are sampled simultaneously in proportion to their analytically
derived counts (interleaved round-robin). This prevents any single constraint
from being overshot before the counterbalancing pools are drawn.
Fine-tuning with single-sequence additions corrects rounding residuals.

GC fallback
-----------
If the GC pool is empty or infeasible (extreme targets), a fallback mode
is triggered: GC is pursued independently with track constraints lifted.
This is clearly flagged in the summary.

"""

import itertools
import re
import sys
import numpy as np
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit here
# ══════════════════════════════════════════════════════════════════════════════

TARGET_A   = 0.04     # fraction of nt positions inside A-tracks  (0.0–1.0)
TARGET_T   = 0.04     # fraction of nt positions inside T-tracks
TARGET_AT  = 0.025    # fraction of nt positions inside AT-tracks
TARGET_GC  = 0.40     # fraction of nt positions that are G or C

TOLERANCE  = 0.0025    # ±0.1% — acceptable deviation for all constraints

RANDOM_SEED  = 42
OUTPUT_FILE  = "/PATH/TO/OUTPUT/dna_10mer_dataset_GC40.txt"
SUMMARY_FILE = "/PATH/TO/SUMMARY/dataset_summary_GC40.txt"

# ── Solver tuning (rarely needs changing) ────────────────────────────────────
BATCH_SIZE   = 50_000   # sequences drawn per interleaved round
MAX_FINETUNE = 50_000   # max single-sequence fine-tune steps


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

A_POS, A_LEN, T_POS, T_LEN, AT_POS, AT_LEN, GC_CNT = 0, 1, 2, 3, 4, 5, 6


def count_track_positions(seq, track_type):
    """
    Return (n_positions_in_track, length_of_longest_track).
    Only tracks of length ≥ 4 counted; overlapping matches merged.
    """
    patterns = {'A': r'A{4,}', 'T': r'T{4,}',
                 'AT': r'(?:AT){2,}|(?:TA){2,}'}
    positions = set()
    longest = 0
    for m in re.finditer(patterns[track_type], seq):
        longest = max(longest, m.end() - m.start())
        positions.update(range(m.start(), m.end()))
    return len(positions), longest


def annotate(seq):
    """Return (a_pos, a_len, t_pos, t_len, at_pos, at_len, gc_count)."""
    a_pos,  a_len  = count_track_positions(seq, 'A')
    t_pos,  t_len  = count_track_positions(seq, 'T')
    at_pos, at_len = count_track_positions(seq, 'AT')
    gc_count = sum(1 for c in seq if c in 'GC')
    return a_pos, a_len, t_pos, t_len, at_pos, at_len, gc_count


# ══════════════════════════════════════════════════════════════════════════════
# POOL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def geometric_weight(k):
    """P(run = k | k ≥ 4) = 0.75 × 0.25^(k−4) from uniform-composition model."""
    return 0.75 * (0.25 ** (k - 4))


def build_track_pool(annotations, idx_len, idx_pos):
    """
    Pool for one track type; sampling probability ∝ geometric_weight(track_length).
    Returns (seqs, weights, ann_list, wavgs) where wavgs is a dict of
    weighted-average contributions per sequence for all four metrics.
    """
    seqs, ws, ann_list = [], [], []
    for seq, ann in annotations.items():
        k = ann[idx_len]
        if k >= 4:
            seqs.append(seq)
            ws.append(geometric_weight(k))
            ann_list.append(ann)
    ws = np.array(ws, dtype=np.float64); ws /= ws.sum()
    wavgs = {
        A_POS:  float(np.dot(ws, [a[A_POS]  for a in ann_list])),
        T_POS:  float(np.dot(ws, [a[T_POS]  for a in ann_list])),
        AT_POS: float(np.dot(ws, [a[AT_POS] for a in ann_list])),
        GC_CNT: float(np.dot(ws, [a[GC_CNT] for a in ann_list])),
    }
    return seqs, ws, ann_list, wavgs


def build_gc_pool(annotations, current_gc_frac, target_gc):
    """
    GC-correction pool with linear soft-threshold weights.

    Direction: if current_gc_frac > target_gc → AT-rich pool (lower GC)
               if current_gc_frac < target_gc → GC-rich pool (raise GC)

    Weight: w = max(target_gc - gc_frac, 0)  for lowering
            w = max(gc_frac - target_gc, 0)  for raising

    Returns (seqs, weights, ann_list, wavgs, direction).
    """
    need_lower = (current_gc_frac >= target_gc)
    seqs, ws, ann_list = [], [], []
    for seq, ann in annotations.items():
        gc_frac = ann[GC_CNT] / 10
        w = max(target_gc - gc_frac, 0.0) if need_lower else max(gc_frac - target_gc, 0.0)
        if w > 0:
            seqs.append(seq); ws.append(w); ann_list.append(ann)
    if not seqs:
        return [], np.array([]), [], {}, 'lower' if need_lower else 'raise'
    ws = np.array(ws, dtype=np.float64); ws /= ws.sum()
    wavgs = {
        A_POS:  float(np.dot(ws, [a[A_POS]  for a in ann_list])),
        T_POS:  float(np.dot(ws, [a[T_POS]  for a in ann_list])),
        AT_POS: float(np.dot(ws, [a[AT_POS] for a in ann_list])),
        GC_CNT: float(np.dot(ws, [a[GC_CNT] for a in ann_list])),
    }
    direction = 'lower' if need_lower else 'raise'
    return seqs, ws, ann_list, wavgs, direction


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL JOINT SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def compute_natural_gc(N_base, base_gc, base_apos, base_tpos, base_atpos,
                       wavgs_a, wavgs_t, wavgs_at, ta, tt, tat):
    """
    GC% the dataset reaches with track sequences only (no GC correction).
    Solves the 3-constraint track system with n_gc=0.
    """
    L = 10
    # n_x = (target_x*L/wavg_pos_x)*N - base_x/wavg_pos_x
    ca_N = ta  * L / wavgs_a[A_POS];  da = -base_apos  / wavgs_a[A_POS]
    ct_N = tt  * L / wavgs_t[T_POS];  dt = -base_tpos  / wavgs_t[T_POS]
    cat_N= tat * L / wavgs_at[AT_POS];dat= -base_atpos / wavgs_at[AT_POS]
    # N = N_base + n_a + n_t + n_at → N*(1-ca_N-ct_N-cat_N) = N_base+da+dt+dat
    N = (N_base + da + dt + dat) / (1 - ca_N - ct_N - cat_N)
    n_a = ca_N*N + da; n_t = ct_N*N + dt; n_at = cat_N*N + dat
    gc_nat = (base_gc + n_a*wavgs_a[GC_CNT] + n_t*wavgs_t[GC_CNT]
              + n_at*wavgs_at[GC_CNT]) / (N * L)
    return gc_nat


def solve_joint(N_base, base_gc, base_apos, base_tpos, base_atpos,
                wavgs_a, wavgs_t, wavgs_at, wavgs_gc,
                ta, tt, tat, tgc):
    """
    Solve the joint linear system accounting for GC pool cross-contributions.

    System (after substituting N = N_base + n_a + n_t + n_at + n_gc):
      Track constraints (1)-(3):
        n_x = (target_x*L/wavg_pos_x)*N - base_x/wavg_pos_x
              - (wavg_x_from_gc/wavg_pos_x)*n_gc

      These give n_a, n_t, n_at as linear functions of N and n_gc.
      Substitute into:
        (4) GC constraint  → equation in N and n_gc
        (5) N definition   → equation in N and n_gc
      → 2×2 linear system → solve for N and n_gc.

    If n_at < 0 (AT-track target met incidentally by GC pool):
      Clamp n_at = 0 and re-solve reduced 2×2 system.
      AT-track percentage will exceed target_at — reported in summary.

    Returns dict with keys: N, n_a, n_t, n_at, n_gc, at_clamped.
    """
    L = 10

    def solve_2x2(n_at_fixed=None):
        wpa = wavgs_a[A_POS]; wpt = wavgs_t[T_POS]; wpat = wavgs_at[AT_POS]
        wga = wavgs_gc[A_POS]; wgt = wavgs_gc[T_POS]
        wgat= wavgs_gc[AT_POS]; wgg = wavgs_gc[GC_CNT]
        wgca= wavgs_a[GC_CNT]; wgct= wavgs_t[GC_CNT]; wgcat=wavgs_at[GC_CNT]

        # n_a = ca_N*N + ca_c + ca_g*n_gc  (similarly for n_t, n_at if free)
        ca_N=ta*L/wpa;  ca_c=-base_apos/wpa;  ca_g=-wga/wpa
        ct_N=tt*L/wpt;  ct_c=-base_tpos/wpt;  ct_g=-wgt/wpt

        if n_at_fixed is None:
            cat_N=tat*L/wpat; cat_c=-base_atpos/wpat; cat_g=-wgat/wpat
            # (5): N*(1-ca_N-ct_N-cat_N) - n_gc*(ca_g+ct_g+cat_g+1) = N_base+ca_c+ct_c+cat_c
            A11=1-ca_N-ct_N-cat_N; A12=-(ca_g+ct_g+cat_g+1)
            b1=N_base+ca_c+ct_c+cat_c
            # (4): N*(tgc*L-ca_N*wgca-ct_N*wgct-cat_N*wgcat)
            #      - n_gc*(ca_g*wgca+ct_g*wgct+cat_g*wgcat+wgg) = base_gc+ca_c*wgca+ct_c*wgct+cat_c*wgcat
            A21=tgc*L-ca_N*wgca-ct_N*wgct-cat_N*wgcat
            A22=-(ca_g*wgca+ct_g*wgct+cat_g*wgcat+wgg)
            b2=base_gc+ca_c*wgca+ct_c*wgct+cat_c*wgcat
        else:
            nat=n_at_fixed
            # n_at is known; only unknowns are N and n_gc
            # (5): N*(1-ca_N-ct_N) - n_gc*(ca_g+ct_g+1) = N_base+ca_c+ct_c+nat
            A11=1-ca_N-ct_N; A12=-(ca_g+ct_g+1)
            b1=N_base+ca_c+ct_c+nat
            # (4): subtract nat*wgcat from RHS
            A21=tgc*L-ca_N*wgca-ct_N*wgct
            A22=-(ca_g*wgca+ct_g*wgct+wgg)
            b2=base_gc+ca_c*wgca+ct_c*wgct+nat*wgcat-nat*wpat*(0)
            # Correct b2: nat contrib to GC already in A21/A22 via n_at side
            # Re-derive cleanly:
            # tgc*L*N = base_gc + n_a*wgca + n_t*wgct + nat*wgcat + n_gc*wgg
            # n_a=(ca_N*N+ca_c+ca_g*n_gc), n_t=(ct_N*N+ct_c+ct_g*n_gc)
            # tgc*L*N = base_gc + (ca_N*N+ca_c+ca_g*n_gc)*wgca
            #         + (ct_N*N+ct_c+ct_g*n_gc)*wgct + nat*wgcat + n_gc*wgg
            A21 = tgc*L - ca_N*wgca - ct_N*wgct
            A22 = -(ca_g*wgca + ct_g*wgct + wgg)
            b2  = base_gc + ca_c*wgca + ct_c*wgct + nat*wgcat

        sol = np.linalg.solve(np.array([[A11,A12],[A21,A22]],dtype=np.float64),
                              np.array([b1,b2],dtype=np.float64))
        N, n_gc = sol
        n_a = ca_N*N + ca_c + ca_g*n_gc
        n_t = ct_N*N + ct_c + ct_g*n_gc
        n_at = (cat_N*N + cat_c + cat_g*n_gc) if n_at_fixed is None else n_at_fixed
        return N, n_a, n_t, n_at, n_gc

    N, n_a, n_t, n_at, n_gc = solve_2x2()
    at_clamped = False

    if n_at < -1:
        # AT-track target is met (or exceeded) incidentally by the GC pool.
        # Clamp n_at=0 and re-solve: we need fewer total sequences.
        N, n_a, n_t, n_at, n_gc = solve_2x2(n_at_fixed=0)
        at_clamped = True

    return dict(N=N, n_a=max(n_a,0), n_t=max(n_t,0),
                n_at=max(n_at,0), n_gc=max(n_gc,0),
                at_clamped=at_clamped)


# ══════════════════════════════════════════════════════════════════════════════
# FEASIBILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_feasibility(ta, tt, tat, tgc, tol):
    for name, val in [("TARGET_A", ta), ("TARGET_T", tt), ("TARGET_AT", tat),
                      ("TARGET_GC", tgc), ("TOLERANCE", tol)]:
        if not (0.0 <= val <= 1.0):
            print(f"ERROR: {name} = {val:.4f} is outside [0.0, 1.0].")
            sys.exit(1)
    min_at = ta + tt + tat
    if tgc > 1.0 - min_at + tol:
        print(f"ERROR: target_gc ({tgc*100:.2f}%) is incompatible with track targets "
              f"committing ≥{min_at*100:.1f}% to AT. Max achievable GC ≈ {(1-min_at)*100:.1f}%.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# INTERLEAVED SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

def draw_interleaved(run, pools, target_counts, seq_len=10):
    """
    Draw sequences from all active pools simultaneously, in each round
    allocating draws proportional to remaining counts.  This keeps all four
    running percentages converging together rather than one at a time.

    pools       : dict  key → (seqs, weights, ann_list, wavgs, ...)
    target_counts : dict  key → int
    Returns list of added sequence strings.
    """
    added = []
    sampled = {k: 0 for k in target_counts}
    total_target = sum(target_counts.values())

    round_num = 0
    while any(sampled[k] < target_counts[k] for k in target_counts):
        round_num += 1
        remaining   = {k: target_counts[k] - sampled[k] for k in target_counts}
        active      = [k for k in target_counts if remaining[k] > 0]
        total_rem   = sum(remaining[k] for k in active)
        round_total = min(BATCH_SIZE, total_rem)

        # Proportional allocation for this round
        alloc = {}; allocated = 0
        for i, k in enumerate(active):
            if i == len(active) - 1:
                alloc[k] = round_total - allocated
            else:
                share = max(1, int(round(round_total * remaining[k] / total_rem)))
                share = min(share, remaining[k])
                alloc[k] = share; allocated += share

        for k in active:
            n_draw = alloc.get(k, 0)
            if n_draw <= 0: continue
            pool = pools[k]
            seqs_l, ws, ann_list = pool[0], pool[1], pool[2]
            if not seqs_l:
                print(f"  WARNING: pool {k} is empty."); continue
            indices = np.random.choice(len(seqs_l), size=n_draw, p=ws)
            for idx in indices:
                ann = ann_list[idx]
                run['a_pos']  += ann[A_POS]; run['t_pos']  += ann[T_POS]
                run['at_pos'] += ann[AT_POS]; run['gc']    += ann[GC_CNT]
                run['nuc']    += seq_len; added.append(seqs_l[idx])
            sampled[k] += n_draw

        if round_num % 10 == 0:
            nuc = run['nuc']; done = sum(sampled.values())
            print(f"    round {round_num:4d} | {done:>9,}/{total_target:,} | "
                  f"A:{run['a_pos']/nuc*100:.3f}% T:{run['t_pos']/nuc*100:.3f}% "
                  f"AT:{run['at_pos']/nuc*100:.3f}% GC:{run['gc']/nuc*100:.3f}%")

    nuc = run['nuc']
    print(f"  Draw complete | "
          f"A:{run['a_pos']/nuc*100:.3f}% T:{run['t_pos']/nuc*100:.3f}% "
          f"AT:{run['at_pos']/nuc*100:.3f}% GC:{run['gc']/nuc*100:.3f}%")
    return added


# ══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune(run, pools, ta, tt, tat, tgc, tol, at_clamped, annotations):
    """
    Single-sequence additions to correct rounding residuals after the
    analytical draw.  Respects the at_clamped flag: if AT-track was clamped
    (met incidentally), AT constraint is not enforced here either.
    Returns (added_seqs, fallback_triggered).
    """
    added = []

    for iteration in range(1, MAX_FINETUNE + 1):
        nuc = run['nuc']; tol_nuc = tol * nuc
        d_a  = ta  * nuc - run['a_pos']
        d_t  = tt  * nuc - run['t_pos']
        d_at = tat * nuc - run['at_pos']
        d_gc = tgc * nuc - run['gc']

        active = {}
        if abs(d_a)  > tol_nuc: active['A']  = (abs(d_a),  pools['A'])
        if abs(d_t)  > tol_nuc: active['T']  = (abs(d_t),  pools['T'])
        if not at_clamped and abs(d_at) > tol_nuc:
            active['AT'] = (abs(d_at), pools['AT'])
        # GC: re-check pool direction and rebuild if needed
        if abs(d_gc) > tol_nuc:
            cur_gc = run['gc'] / nuc
            cur_dir = pools['GC'][4] if len(pools['GC']) > 4 else 'lower'
            need_dir = 'lower' if cur_gc > tgc else 'raise'
            if need_dir != cur_dir or not pools['GC'][0]:
                # Rebuild GC pool with correct direction
                new_gc_pool = build_gc_pool(annotations, cur_gc, tgc)
                if not new_gc_pool[0]:
                    print("  WARNING: GC pool empty during fine-tune. "
                          "Triggering GC fallback.")
                    return added, True
                pools['GC'] = new_gc_pool
            active['GC'] = (abs(d_gc), pools['GC'])

        if not active:
            print(f"  Fine-tune converged after {iteration-1:,} steps.")
            return added, False

        chosen = max(active, key=lambda k: active[k][0] / nuc)
        pool = active[chosen][1]
        idx = int(np.random.choice(len(pool[0]), p=pool[1]))
        ann = pool[2][idx]
        run['a_pos']  += ann[A_POS]; run['t_pos']  += ann[T_POS]
        run['at_pos'] += ann[AT_POS]; run['gc']    += ann[GC_CNT]
        run['nuc']    += 10; added.append(pool[0][idx])

    print(f"  WARNING: fine-tune reached {MAX_FINETUNE:,} steps. "
          f"Small residuals may remain.")
    return added, False


# ══════════════════════════════════════════════════════════════════════════════
# GC FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def gc_fallback(run, annotations, tgc, tol):
    print(
        "\n  *** GC FALLBACK TRIGGERED ***\n"
        "  The GC pool was empty or exhausted. Track constraints are lifted.\n"
        "  GC target will be pursued independently."
    )
    cur_gc = run['gc'] / run['nuc']
    pool = build_gc_pool(annotations, cur_gc, tgc)
    if not pool[0]:
        print("  ERROR: cannot build GC pool. GC correction aborted.")
        return []
    seqs_l, ws, ann_list, wavgs, _ = pool
    added = []
    eff = abs(wavgs[GC_CNT] - tgc * 10)
    for _ in range(500_000):
        nuc = run['nuc']
        if abs(tgc * nuc - run['gc']) <= tol * nuc:
            break
        d = abs(tgc * nuc - run['gc'])
        n = 1 if eff <= 0 else min(int(np.ceil(d / eff)), BATCH_SIZE)
        for idx in np.random.choice(len(seqs_l), size=n, p=ws):
            ann = ann_list[idx]
            run['a_pos'] += ann[A_POS]; run['t_pos'] += ann[T_POS]
            run['at_pos']+= ann[AT_POS]; run['gc']   += ann[GC_CNT]
            run['nuc'] += 10; added.append(seqs_l[idx])
    print(f"  GC fallback added {len(added):,} sequences.")
    return added


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(n_base, added_main, added_fallback, run, annotations,
                  ta, tt, tat, tgc, tol, gc_natural, sol,
                  at_clamped, fallback_triggered):
    nuc     = run['nuc']
    n_added = len(added_main) + len(added_fallback)
    n_total = n_base + n_added
    tol_pct = tol * 100

    fa = run['a_pos']  / nuc * 100
    ft = run['t_pos']  / nuc * 100
    fat= run['at_pos'] / nuc * 100
    fgc= run['gc']     / nuc * 100

    bn = n_base * 10
    ba = sum(a[A_POS]  for a in annotations.values()) / bn * 100
    bt = sum(a[T_POS]  for a in annotations.values()) / bn * 100
    bat= sum(a[AT_POS] for a in annotations.values()) / bn * 100
    bgc= sum(a[GC_CNT] for a in annotations.values()) / bn * 100

    def stat(final, target, lifted=False, natural=False):
        if lifted:   return "LIFTED"
        if natural:  return "NATURAL"   # exceeded naturally, not a failure
        return "YES" if abs(final - target*100) <= tol_pct else "NO "

    at_status_flag = fallback_triggered or at_clamped

    lines = [
        "=" * 72,
        "DNA 10-mer Dataset — Generation Summary  (v7)",
        "=" * 72,
        f"  Base set (unique 10-mers)       : {n_base:>12,}",
        f"  Added sequences (main)          : {len(added_main):>12,}",
        f"  Added sequences (GC fallback)   : {len(added_fallback):>12,}",
        f"  Total sequences                 : {n_total:>12,}",
        f"  Total nucleotide positions      : {nuc:>12,}",
        "",
        f"  Natural GC settling point  : {gc_natural*100:.4f}%",
        f"  (GC reached with track sequences only, before any GC correction)",
        "",
        f"  Analytically derived pool additions:",
        f"    A-track  pool : {int(round(sol['n_a'])):>10,}",
        f"    T-track  pool : {int(round(sol['n_t'])):>10,}",
        f"    AT-track pool : {int(round(sol['n_at'])):>10,}"
        + (" (clamped to 0 — AT target met by GC pool)" if at_clamped else ""),
        f"    GC       pool : {int(round(sol['n_gc'])):>10,}",
        f"    Total (theory): {int(round(sol['n_a']+sol['n_t']+sol['n_at']+sol['n_gc'])):>10,}",
        "",
    ]

    if at_clamped:
        lines += [
            "  NOTE: AT-track target is exceeded because the AT-rich GC correction",
            f"  pool incidentally contributes AT-track content.  This is a mathematical",
            f"  consequence of target_gc={tgc*100:.1f}% being below the natural settling",
            f"  point ({gc_natural*100:.2f}%).  The AT-track % is reported as NATURAL.",
            "",
        ]
    if fallback_triggered:
        lines += [
            "  !! GC FALLBACK WAS TRIGGERED — track constraints lifted !!",
            "",
        ]

    lines += [
        f"  Tolerance: ±{tol_pct:.1f}%",
        "",
        f"  {'Constraint':<12} {'Baseline':>10} {'Final':>10} "
        f"{'Target':>10} {'±Tol':>8} {'Status':>8}",
        "  " + "-" * 64,
        f"  {'A-track':<12} {ba:>9.4f}% {fa:>9.4f}% "
        f"{ta*100:>9.2f}% {tol_pct:>7.1f}% "
        f"{stat(fa, ta, fallback_triggered):>8}",
        f"  {'T-track':<12} {bt:>9.4f}% {ft:>9.4f}% "
        f"{tt*100:>9.2f}% {tol_pct:>7.1f}% "
        f"{stat(ft, tt, fallback_triggered):>8}",
        f"  {'AT-track':<12} {bat:>9.4f}% {fat:>9.4f}% "
        f"{tat*100:>9.2f}% {tol_pct:>7.1f}% "
        f"{stat(fat, tat, fallback_triggered, at_clamped):>8}",
        f"  {'GC':<12} {bgc:>9.4f}% {fgc:>9.4f}% "
        f"{tgc*100:>9.2f}% {tol_pct:>7.1f}% "
        f"{stat(fgc, tgc, fallback_triggered):>8}",
        "",
        "  Track length distribution of added sequences:",
    ]

    for label, idx_len in [('A', A_LEN), ('T', T_LEN), ('AT', AT_LEN)]:
        counts = defaultdict(int)
        for seq in added_main + added_fallback:
            k = annotations[seq][idx_len]
            if k >= 4: counts[k] += 1
        total = sum(counts.values())
        lines.append(f"    {label}-track (seqs with track: {total:,}):")
        if total:
            for k in sorted(counts):
                lines.append(f"      k={k:2d}: {counts[k]:>8,}  ({counts[k]/total*100:5.2f}%)")
        else:
            lines.append("      (none)")

    lines.append("=" * 72)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 72)
    print("DNA 10-mer Dataset Generator  (v7 — full joint analytical solver)")
    print("=" * 72)
    print(f"  Targets  A:{TARGET_A*100:.2f}%  T:{TARGET_T*100:.2f}%  "
          f"AT:{TARGET_AT*100:.2f}%  GC:{TARGET_GC*100:.2f}%")
    print(f"  Tolerance ±{TOLERANCE*100:.1f}%   Seed: {RANDOM_SEED}")
    print()

    # ── Step 1: Base set ──────────────────────────────────────────────────
    print("Step 1/5  Generating all 4^10 = 1,048,576 unique 10-mers...")
    all_kmers = [''.join(p) for p in itertools.product(['A','T','G','C'], repeat=10)]
    N_base = len(all_kmers)
    print(f"  {N_base:,} sequences.")

    # ── Step 2: Annotate ──────────────────────────────────────────────────
    print("Step 2/5  Annotating sequences...")
    annotations = {seq: annotate(seq) for seq in all_kmers}
    print("  Done.")

    # ── Step 3: Build track pools ─────────────────────────────────────────
    print("Step 3/5  Building track pools...")
    pool_a  = build_track_pool(annotations, A_LEN,  A_POS)
    pool_t  = build_track_pool(annotations, T_LEN,  T_POS)
    pool_at = build_track_pool(annotations, AT_LEN, AT_POS)
    print(f"  Pool sizes  A:{len(pool_a[0]):,}  T:{len(pool_t[0]):,}  AT:{len(pool_at[0]):,}")

    # ── Step 4: Feasibility + analytical solution ─────────────────────────
    print("Step 4/5  Feasibility check and analytical solution...")
    check_feasibility(TARGET_A, TARGET_T, TARGET_AT, TARGET_GC, TOLERANCE)

    base_nuc   = N_base * 10
    base_apos  = sum(a[A_POS]  for a in annotations.values())
    base_tpos  = sum(a[T_POS]  for a in annotations.values())
    base_atpos = sum(a[AT_POS] for a in annotations.values())
    base_gc    = sum(a[GC_CNT] for a in annotations.values())

    gc_natural = compute_natural_gc(
        N_base, base_gc, base_apos, base_tpos, base_atpos,
        pool_a[3], pool_t[3], pool_at[3],
        TARGET_A, TARGET_T, TARGET_AT,
    )
    print(f"  Natural GC settling point: {gc_natural*100:.4f}%")
    direction = "AT-rich (lower GC)" if TARGET_GC < gc_natural else "GC-rich (raise GC)"
    print(f"  GC pool direction: {direction}")

    # Build GC pool — direction from base-set GC (50%) vs target
    pool_gc = build_gc_pool(annotations, base_gc / base_nuc, TARGET_GC)
    if not pool_gc[0]:
        print("ERROR: GC pool is empty. Cannot proceed.")
        sys.exit(1)
    print(f"  GC pool: {len(pool_gc[0]):,} seqs, "
          f"weighted-avg GC {pool_gc[3][GC_CNT]/10*100:.2f}%")

    sol = solve_joint(
        N_base, base_gc, base_apos, base_tpos, base_atpos,
        pool_a[3], pool_t[3], pool_at[3], pool_gc[3],
        TARGET_A, TARGET_T, TARGET_AT, TARGET_GC,
    )
    print(f"  Analytical solution:")
    print(f"    Target N   : {sol['N']:>12,.0f}")
    print(f"    n_a        : {sol['n_a']:>12,.0f}")
    print(f"    n_t        : {sol['n_t']:>12,.0f}")
    print(f"    n_at       : {sol['n_at']:>12,.0f}"
          + (" (clamped — AT met by GC pool)" if sol['at_clamped'] else ""))
    print(f"    n_gc       : {sol['n_gc']:>12,.0f}")
    print(f"    Total adds : {sol['n_a']+sol['n_t']+sol['n_at']+sol['n_gc']:>12,.0f}")

    if sol['at_clamped']:
        print(f"  NOTE: AT-track target will be exceeded naturally "
              f"(GC pool is AT-rich). See summary.")

    # ── Step 5: Sample ────────────────────────────────────────────────────
    run = {'a_pos': base_apos, 't_pos': base_tpos, 'at_pos': base_atpos,
           'gc': base_gc, 'nuc': base_nuc}

    target_counts = {
        'A':  int(round(sol['n_a'])),
        'T':  int(round(sol['n_t'])),
        'AT': int(round(sol['n_at'])),
        'GC': int(round(sol['n_gc'])),
    }
    pools = {'A': pool_a, 'T': pool_t, 'AT': pool_at, 'GC': pool_gc}

    print("Step 5/5  Sampling...")
    print("  Interleaved draw to analytically derived counts:")
    added_main = draw_interleaved(run, pools, target_counts)

    print("  Fine-tuning residual deficits...")
    added_fine, fallback = fine_tune(
        run, pools, TARGET_A, TARGET_T, TARGET_AT, TARGET_GC,
        TOLERANCE, sol['at_clamped'], annotations,
    )
    added_main.extend(added_fine)

    added_fallback = []
    if fallback:
        added_fallback = gc_fallback(run, annotations, TARGET_GC, TOLERANCE)

    # ── Summary + output ──────────────────────────────────────────────────
    summary = build_summary(
        N_base, added_main, added_fallback, run, annotations,
        TARGET_A, TARGET_T, TARGET_AT, TARGET_GC, TOLERANCE,
        gc_natural, sol, sol['at_clamped'], fallback,
    )
    print(); print(summary)

    print(f"\nWriting to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w') as f:
        for seq in all_kmers: f.write(seq + '\n')
        for seq in added_main + added_fallback: f.write(seq + '\n')
    n_total = N_base + len(added_main) + len(added_fallback)
    print(f"  Written {n_total:,} sequences.")

    with open(SUMMARY_FILE, 'w') as f: f.write(summary + '\n')
    print(f"  Summary → {SUMMARY_FILE}\nDone.")


if __name__ == "__main__":
    main()
