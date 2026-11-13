"""
Periodic Orbit Detection in the Lorenz Attractor
via Symbolic Dynamics and Kneading Sequences

Author : Rafael Hasanov
Thesis : An Algorithm for Detecting and Encoding Periodic Orbits
         in the Lorenz Attractor Using Kneading Sequences
         and Symbolic Dynamics
Date   : May 2026

Pipeline
--------
Stage 1  Simulate the Lorenz system (adaptive RK45, tight tolerances).
Stage 2  Encode the trajectory using the Poincare return map:
         at each local maximum of z, record L if z_n < z* else R,
         where z* is the fold point of the 1D Lorenz map (~38.51).
Stage 3  Detect candidate periodic orbits as repeating symbolic blocks.
Stage 4  Compute the kneading bound: the maximum run of a single symbol
         observed in a long reference trajectory (empirical kneading
         invariant of the 1D Lorenz map).
Stage 5  Validate each candidate: accept iff the maximum run of any symbol
         in every cyclic shift of the block does not exceed the bound.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from collections import Counter

# ============================================================
# 0.  System Parameters
# ============================================================
SIGMA  = 10.0
RHO    = 28.0
BETA   = 8.0 / 3.0

# Fold point of the 1D Lorenz return map.
# Determined empirically; see estimate_fold_point().
Z_FOLD = 38.51


def lorenz(t, state):
    """Right-hand side of the Lorenz ODE system."""
    x, y, z = state
    return [
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ]


# ============================================================
# Stage 1 -- Simulation
# ============================================================
def simulate_lorenz(
    initial_state,
    t_end=500.0,
    n_points=250_000,
    transient_cutoff=10.0,
):
    """
    Integrate the Lorenz system and discard the transient.

    Parameters
    ----------
    initial_state    : list [x0, y0, z0]
    t_end            : float   end time
    n_points         : int     number of time-evaluation points
    transient_cutoff : float   discard all t < transient_cutoff

    Returns
    -------
    x, y, z : numpy 1-D arrays (post-transient only)
    """
    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        lorenz,
        (0.0, t_end),
        initial_state,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-13,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    mask = sol.t >= transient_cutoff
    return sol.y[0][mask], sol.y[1][mask], sol.y[2][mask]


# ============================================================
# Stage 2 -- Poincare Map and Symbolic Encoding
# ============================================================
def get_poincare_z(x, y, z):
    """
    Return z-values at each local maximum of z(t).

    A local maximum of z satisfies dz/dt = 0 from above:
        dz/dt = x*y - beta*z  changes from + to -

    Each such event is one crossing of the Poincare section and
    corresponds to exactly one full lobe visit by the trajectory.

    Returns
    -------
    numpy array of z values at successive Poincare crossings
    """
    dz = x * y - BETA * z
    zc = []
    for i in range(1, len(dz)):
        if dz[i - 1] > 0.0 and dz[i] <= 0.0:
            zc.append(z[i])
    return np.array(zc)


def estimate_fold_point(x, y, z):
    """
    Estimate the fold point z* of the 1D Lorenz return map.

    The 1D map f: z_n -> z_{n+1} has a single maximum (fold) at z*.
    We locate z* by binning the map and finding the bin with the
    highest average image value.

    Returns
    -------
    z_fold : float
    """
    zc   = get_poincare_z(x, y, z)
    z_n  = zc[:-1]
    z_n1 = zc[1:]

    bins = np.percentile(z_n, np.linspace(0, 100, 40))
    best_z, best_img = float(z_n.mean()), -np.inf

    for i in range(len(bins) - 1):
        mask = (z_n >= bins[i]) & (z_n < bins[i + 1])
        if mask.sum() > 2:
            avg = float(z_n1[mask].mean())
            if avg > best_img:
                best_img = avg
                best_z   = float(z_n[mask].mean())

    return best_z


def encode_poincare(z_crossings, z_fold):
    """
    Encode a Poincare crossing sequence as a symbolic L/R string.

    Partition rule (Sparrow 1982, Chapter 3):
        z_n < z*  ->  L   (ascending branch of 1D map)
        z_n >= z* ->  R   (descending branch of 1D map)

    Parameters
    ----------
    z_crossings : array-like   z-values at Poincare crossings
    z_fold      : float        fold point z*

    Returns
    -------
    str  over {'L', 'R'}
    """
    return "".join("L" if z < z_fold else "R" for z in z_crossings)


# ============================================================
# Stage 3 -- Candidate Detection
# ============================================================
def find_repeating_blocks(sequence, min_period=2, max_period=8, min_count=5):
    """
    Detect candidate periodic orbits as repeating symbolic blocks.

    For each block length p, slide a window of width p across the
    sequence and count occurrences. Retain blocks that appear at
    least min_count times.

    Parameters
    ----------
    sequence   : str   symbolic sequence over {'L', 'R'}
    min_period : int
    max_period : int
    min_count  : int   minimum occurrences to retain as a candidate

    Returns
    -------
    list of dict  (block, period, occurrences) sorted by occurrences DESC
    """
    candidates = []
    seen       = set()

    for p in range(min_period, max_period + 1):
        counts = Counter(
            sequence[i : i + p] for i in range(len(sequence) - p + 1)
        )
        for block, count in counts.items():
            if count >= min_count and block not in seen:
                seen.add(block)
                candidates.append({"block": block, "period": p,
                                    "occurrences": count})

    candidates.sort(key=lambda d: (-d["occurrences"], d["period"]))
    return candidates


# ============================================================
# Stage 4 -- Kneading Bound
# ============================================================
def compute_kneading_bound(x, y, z, z_fold):
    """
    Compute the kneading admissibility bound.

    The kneading invariant of the 1D Lorenz map (Milnor & Thurston
    1988; Sparrow 1982 §3) determines the maximum number of
    consecutive identical symbols any admissible sequence may contain.
    Sequences whose cyclic shifts contain a longer run correspond to
    trajectories that would need to cross the fold point, which is
    topologically forbidden.

    We compute this bound empirically as the maximum run of a single
    symbol observed in a long reference trajectory on the attractor.

    Parameters
    ----------
    x, y, z : numpy arrays from simulate_lorenz
    z_fold  : float  fold point of the 1D map

    Returns
    -------
    max_run : int   kneading admissibility bound
    """
    zc  = get_poincare_z(x, y, z)
    seq = encode_poincare(zc, z_fold)

    if not seq:
        return 8  # safe fallback

    max_run = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur    += 1
            max_run = max(max_run, cur)
        else:
            cur = 1

    return max_run


# ============================================================
# Stage 5 -- Kneading Validation
# ============================================================
def _max_run(s):
    """Maximum length of any run of a single symbol in string s."""
    if not s:
        return 0
    mx = cur = 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        mx  = max(mx, cur)
    return mx


def validate_candidate(block, kneading_bound):
    """
    Apply the kneading admissibility condition to a symbolic block.

    A block b = b_1 ... b_p is admissible iff for every cyclic shift
    sigma^k(b), the maximum run of any symbol in the periodic
    extension (sigma^k(b))^inf does not exceed the kneading bound.

    The cyclic-shift check is required because a periodic orbit is a
    closed loop: any point along the orbit is a valid starting symbol,
    so all p rotations of the block must be individually admissible.

    Parameters
    ----------
    block           : str   symbolic block over {'L', 'R'}
    kneading_bound  : int

    Returns
    -------
    'accepted' or 'rejected'
    """
    p = len(block)
    for i in range(p):
        shift    = block[i:] + block[:i]
        extended = shift * 4          # 4 repetitions captures all wrap-around runs
        if _max_run(extended) > kneading_bound:
            return "rejected"
    return "accepted"


# ============================================================
# Full Pipeline
# ============================================================
def run_pipeline(
    initial_state=(1.0, 1.0, 1.0),
    t_end=500.0,
    n_points=250_000,
    min_period=2,
    max_period=8,
    min_count=5,
):
    """
    Run all five stages and return the validated results.

    Returns
    -------
    df         : pandas DataFrame   results table
    x, y, z    : numpy arrays       simulated trajectory
    zc         : numpy array        z-values at Poincare crossings
    sequence   : str                symbolic sequence
    z_fold     : float              estimated fold point
    kn_bound   : int                kneading bound
    """
    # -- Stage 1 --
    print("Stage 1: Simulating Lorenz system ...")
    x, y, z = simulate_lorenz(initial_state, t_end=t_end, n_points=n_points)
    print(f"         {len(x):,} points (post-transient)")

    # -- Stage 2 --
    print("Stage 2: Poincare map encoding ...")
    z_fold = estimate_fold_point(x, y, z)
    zc     = get_poincare_z(x, y, z)
    seq    = encode_poincare(zc, z_fold)
    print(f"         Fold point z* = {z_fold:.4f}")
    print(f"         Poincare crossings     : {len(zc)}")
    print(f"         Symbolic sequence length : {len(seq)}")
    print(f"         First 60 symbols       : {seq[:60]}")

    # -- Stage 3 --
    print("Stage 3: Detecting repeating blocks ...")
    candidates = find_repeating_blocks(seq, min_period, max_period, min_count)
    print(f"         {len(candidates)} candidate(s) found")

    # -- Stage 4 --
    print("Stage 4: Computing kneading bound ...")
    kn_bound = compute_kneading_bound(x, y, z, z_fold)
    print(f"         Kneading bound = {kn_bound}")

    # -- Stage 5 --
    print("Stage 5: Validating candidates ...")
    rows = []
    for c in candidates:
        status = validate_candidate(c["block"], kn_bound)
        rows.append({
            "Block"          : c["block"],
            "Symbolic Period": c["period"],
            "Occurrences"    : c["occurrences"],
            "Max Run"        : _max_run(c["block"] * 4),
            "Status"         : status,
        })

    df = pd.DataFrame(rows)
    return df, x, y, z, zc, seq, z_fold, kn_bound


# ============================================================
# Output helpers
# ============================================================
def print_results(df, kn_bound):
    sep  = "=" * 68
    cols = ["Block", "Symbolic Period", "Occurrences", "Max Run", "Status"]
    acc  = df[df["Status"] == "accepted"]
    rej  = df[df["Status"] == "rejected"]

    print(f"\n{sep}")
    print("  VALIDATED PERIODIC ORBIT CANDIDATES")
    print(f"  Kneading bound (max admissible run) = {kn_bound}")
    print(sep)

    if not acc.empty:
        print(f"\n  ACCEPTED  ({len(acc)}):\n")
        print(acc[cols].to_string(index=False))

    if not rej.empty:
        print(f"\n  REJECTED  ({len(rej)}):\n")
        print(rej[cols].head(20).to_string(index=False))
        if len(rej) > 20:
            print(f"  ... and {len(rej)-20} more rejected blocks")

    print(f"\n{sep}\n")


def plot_attractor(x, y, z, path="results/lorenz_attractor.png"):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=0.35, color="steelblue", alpha=0.9)
    ax.set_title("Simulated Lorenz Attractor", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_return_map(x, y, z, zc, z_fold, path="results/return_map.png"):
    """Plot x(t) time series and the 1D Lorenz return map side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left panel: x(t) snippet
    n_show = min(len(x), 12_000)
    t_show = np.linspace(10, 10 + 500 * n_show / len(x), n_show)
    axes[0].plot(t_show, x[:n_show], linewidth=0.5, color="steelblue")
    axes[0].axhline(0, color="crimson", linewidth=0.8, linestyle="--",
                    label="x = 0")
    axes[0].set_title("Lorenz trajectory: x(t)", fontsize=11)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x(t)")
    axes[0].legend(fontsize=9)

    # Right panel: 1D return map z_n -> z_{n+1}
    z_n  = zc[:-1]; z_n1 = zc[1:]
    axes[1].scatter(z_n, z_n1, s=1.2, color="steelblue", alpha=0.4)
    axes[1].axvline(z_fold, color="crimson", linewidth=1.0, linestyle="--",
                    label=f"z* = {z_fold:.2f}  (fold / partition)")
    axes[1].set_title(r"1D Lorenz Return Map: $z_n \to z_{n+1}$", fontsize=11)
    axes[1].set_xlabel(r"$z_n$"); axes[1].set_ylabel(r"$z_{n+1}$")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":

    df, x, y, z, zc, seq, z_fold, kn_bound = run_pipeline(
        initial_state = (1.0, 1.0, 1.0),
        t_end         = 500.0,
        n_points      = 250_000,
        min_period    = 2,
        max_period    = 8,
        min_count     = 5,
    )

    print_results(df, kn_bound)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/validated_candidates.csv", index=False)
    print("  Full table saved to: results/validated_candidates.csv")

    print("\nGenerating figures ...")
    plot_attractor(x, y, z)
    plot_return_map(x, y, z, zc, z_fold)

    print("\nDone.")