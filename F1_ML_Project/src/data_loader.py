"""
data_loader.py
==============
Utilities for loading, merging, and cleaning the Kaggle Formula 1 dataset
(14 CSV files).  Designed to be the single entry-point for all downstream
notebooks and model scripts.

Typical usage
-------------
    from src.data_loader import load_all_data, merge_master_df, clean_master_df

    data   = load_all_data("../data/raw/")
    master = merge_master_df(data)
    master = clean_master_df(master)
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 14 files that make up the Kaggle F1 dataset
EXPECTED_FILES = [
    "circuits.csv",
    "constructor_results.csv",
    "constructor_standings.csv",
    "constructors.csv",
    "driver_standings.csv",
    "drivers.csv",
    "lap_times.csv",
    "pit_stops.csv",
    "qualifying.csv",
    "races.csv",
    "results.csv",
    "seasons.csv",
    "sprint_results.csv",
    "status.csv",
]

# Kaggle uses literal "\N" as a null sentinel in every CSV
KAGGLE_NULL = "\\N"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_files(data_path: str) -> tuple[list[str], list[str]]:
    """Check which expected CSV files are present/missing in *data_path*.

    Parameters
    ----------
    data_path : str
        Directory to scan for CSV files.

    Returns
    -------
    found : list[str]
        File names that exist on disk.
    missing : list[str]
        File names that are absent.
    """
    found, missing = [], []
    for fname in EXPECTED_FILES:
        full = os.path.join(data_path, fname)
        if os.path.exists(full):
            found.append(fname)
        else:
            missing.append(fname)

    if missing:
        print(f"[WARNING] The following {len(missing)} file(s) were NOT found in '{data_path}':")
        for f in missing:
            print(f"          ✗  {f}")
    else:
        print(f"[OK] All {len(EXPECTED_FILES)} expected files found in '{data_path}'.")

    return found, missing


def _convert_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every Kaggle ``\\N`` sentinel with ``np.nan`` in *df*.

    Operates column-by-column so that dtype inference is preserved for
    columns that have no nulls.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may contain ``"\\N"`` strings).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``"\\N"`` replaced by ``np.nan``.
    """
    return df.replace(KAGGLE_NULL, np.nan)


def _aggregate_pit_stops(pit_stops_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pit-stop data to one row per (raceId, driverId).

    Computed columns
    ----------------
    pit_stop_count      : int   – total number of stops in the race
    pit_stop_avg_ms     : float – mean stop duration in milliseconds
    pit_stop_min_ms     : float – fastest individual stop in milliseconds
    pit_stop_total_ms   : float – cumulative pit time in milliseconds

    Parameters
    ----------
    pit_stops_df : pd.DataFrame
        Raw ``pit_stops.csv`` DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame indexed by (raceId, driverId).
    """
    df = pit_stops_df.copy()

    # milliseconds column is the most reliable duration field
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")

    agg = (
        df.groupby(["raceId", "driverId"])
        .agg(
            pit_stop_count=("stop", "count"),
            pit_stop_avg_ms=("milliseconds", "mean"),
            pit_stop_min_ms=("milliseconds", "min"),
            pit_stop_total_ms=("milliseconds", "sum"),
        )
        .reset_index()
    )
    return agg


def _aggregate_lap_times(lap_times_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lap-time data to one row per (raceId, driverId).

    Computed columns
    ----------------
    lap_time_mean_ms : float – mean lap time in milliseconds
    lap_time_std_ms  : float – std dev of lap times (consistency proxy)
    lap_time_best_ms : float – fastest lap in milliseconds
    laps_completed   : int   – number of laps recorded

    Parameters
    ----------
    lap_times_df : pd.DataFrame
        Raw ``lap_times.csv`` DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame indexed by (raceId, driverId).
    """
    df = lap_times_df.copy()
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")

    agg = (
        df.groupby(["raceId", "driverId"])
        .agg(
            lap_time_mean_ms=("milliseconds", "mean"),
            lap_time_std_ms=("milliseconds", "std"),
            lap_time_best_ms=("milliseconds", "min"),
            laps_completed=("lap", "count"),
        )
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all_data(data_path: str = "../data/raw/") -> dict:
    """Load all 14 Kaggle F1 CSV files from *data_path*.

    Missing files are reported but do **not** raise an exception – the
    returned dict simply omits them.  ``\\N`` null sentinels are replaced
    with ``np.nan`` immediately after loading each file.

    Parameters
    ----------
    data_path : str, optional
        Directory that contains the raw CSV files.
        Defaults to ``"../data/raw/"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are the table name (CSV stem, e.g. ``"results"``), values are
        the corresponding DataFrames.

    Examples
    --------
    >>> data = load_all_data("../data/raw/")
    >>> data["results"].shape
    (26080, 18)
    """
    if not os.path.exists(data_path):
        print(f"[ERROR] Data directory '{data_path}' does not exist.")
        return {}

    found, missing = _check_files(data_path)

    if missing:
        print(
            f"\n[INFO] Proceeding with {len(found)} available file(s). "
            "Merges that require missing tables will be skipped.\n"
        )

    data: dict[str, pd.DataFrame] = {}
    for fname in found:
        full_path = os.path.join(data_path, fname)
        table_name = fname.replace(".csv", "")
        try:
            df = pd.read_csv(full_path, low_memory=False)
            df = _convert_nulls(df)
            data[table_name] = df
            print(f"  ✓  {fname:<35}  {df.shape[0]:>7,} rows  ×  {df.shape[1]:>3} cols")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗  {fname}: failed to load — {exc}")

    print(f"\n[OK] Loaded {len(data)} table(s) into data dict.")
    return data


def merge_master_df(data_dict: dict) -> pd.DataFrame:
    """Merge all available DataFrames into a single master DataFrame.

    The merge follows a star-schema pattern with ``results`` as the fact
    table.  All joins are LEFT joins unless the dimension table is
    guaranteed to be complete (drivers, constructors, races, circuits,
    status).

    Merge order
    -----------
    1. results          (master / fact table)
    2. ← races          on raceId
    3. ← circuits       on circuitId  (via races)
    4. ← drivers        on driverId
    5. ← constructors   on constructorId
    6. ← qualifying     on raceId + driverId   LEFT JOIN
    7. ← pit_stops agg  on raceId + driverId   LEFT JOIN
    8. ← lap_times agg  on raceId + driverId   LEFT JOIN
    9. ← driver_standings   on raceId + driverId   LEFT JOIN
    10. ← constructor_standings on raceId + constructorId LEFT JOIN
    11. ← status        on statusId

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Dictionary returned by :func:`load_all_data`.

    Returns
    -------
    pd.DataFrame
        Master (wide) DataFrame.  May have many columns; use
        :func:`clean_master_df` next.

    Raises
    ------
    KeyError
        If the mandatory ``"results"`` table is absent from *data_dict*.
    """
    if "results" not in data_dict:
        raise KeyError(
            "'results' table is required but was not found in data_dict. "
            "Check that results.csv exists in your data directory."
        )

    master = data_dict["results"].copy()
    print(f"[merge] Start — results:               {master.shape}")

    # ------------------------------------------------------------------
    # 1. races  ← circuitId needed for circuits join later
    # ------------------------------------------------------------------
    if "races" in data_dict:
        races = data_dict["races"].copy()
        master = master.merge(races, on="raceId", how="left", suffixes=("", "_race"))
        print(f"[merge] After races:                   {master.shape}")
    else:
        print("[merge] SKIP races — table not available")

    # ------------------------------------------------------------------
    # 2. circuits  (via circuitId that came in from races)
    # ------------------------------------------------------------------
    if "circuits" in data_dict and "circuitId" in master.columns:
        circuits = data_dict["circuits"].copy()
        master = master.merge(
            circuits, on="circuitId", how="left", suffixes=("", "_circuit")
        )
        print(f"[merge] After circuits:                {master.shape}")
    else:
        print("[merge] SKIP circuits — table or circuitId column not available")

    # ------------------------------------------------------------------
    # 3. drivers
    # ------------------------------------------------------------------
    if "drivers" in data_dict:
        drivers = data_dict["drivers"].copy()
        master = master.merge(drivers, on="driverId", how="left", suffixes=("", "_driver"))
        print(f"[merge] After drivers:                 {master.shape}")
    else:
        print("[merge] SKIP drivers — table not available")

    # ------------------------------------------------------------------
    # 4. constructors
    # ------------------------------------------------------------------
    if "constructors" in data_dict:
        constructors = data_dict["constructors"].copy()
        master = master.merge(
            constructors, on="constructorId", how="left", suffixes=("", "_constructor")
        )
        print(f"[merge] After constructors:            {master.shape}")
    else:
        print("[merge] SKIP constructors — table not available")

    # ------------------------------------------------------------------
    # 5. qualifying  — LEFT JOIN, suffix _qual to avoid clashes with
    #    results columns (e.g. both have 'number', 'position')
    # ------------------------------------------------------------------
    if "qualifying" in data_dict:
        qual = data_dict["qualifying"].copy()
        # Keep only the columns we need to avoid a column explosion
        qual_cols = ["raceId", "driverId", "constructorId", "number",
                     "position", "q1", "q2", "q3"]
        qual = qual[[c for c in qual_cols if c in qual.columns]]
        qual = qual.rename(
            columns={
                "number": "number_qual",
                "position": "grid_qual_position",
                "q1": "q1_time",
                "q2": "q2_time",
                "q3": "q3_time",
            }
        )
        master = master.merge(
            qual, on=["raceId", "driverId"], how="left", suffixes=("", "_qual")
        )
        print(f"[merge] After qualifying:              {master.shape}")
    else:
        print("[merge] SKIP qualifying — table not available")

    # ------------------------------------------------------------------
    # 6. pit_stops  (aggregated — one row per raceId+driverId)
    # ------------------------------------------------------------------
    if "pit_stops" in data_dict:
        pit_agg = _aggregate_pit_stops(data_dict["pit_stops"])
        master = master.merge(pit_agg, on=["raceId", "driverId"], how="left")
        print(f"[merge] After pit_stops (agg):         {master.shape}")
    else:
        print("[merge] SKIP pit_stops — table not available")

    # ------------------------------------------------------------------
    # 7. lap_times  (aggregated — one row per raceId+driverId)
    # ------------------------------------------------------------------
    if "lap_times" in data_dict:
        lap_agg = _aggregate_lap_times(data_dict["lap_times"])
        master = master.merge(lap_agg, on=["raceId", "driverId"], how="left")
        print(f"[merge] After lap_times (agg):         {master.shape}")
    else:
        print("[merge] SKIP lap_times — table not available")

    # ------------------------------------------------------------------
    # 8. driver_standings  — standings BEFORE this race (use cautiously
    #    to avoid leakage; flag clearly)
    # ------------------------------------------------------------------
    if "driver_standings" in data_dict:
        ds = data_dict["driver_standings"].copy()
        ds = ds.rename(
            columns={
                "points": "ds_points",
                "position": "ds_position",
                "wins": "ds_wins",
                "positionText": "ds_positionText",
            }
        )
        master = master.merge(ds, on=["raceId", "driverId"], how="left", suffixes=("", "_ds"))
        print(f"[merge] After driver_standings:        {master.shape}")
    else:
        print("[merge] SKIP driver_standings — table not available")

    # ------------------------------------------------------------------
    # 9. constructor_standings
    # ------------------------------------------------------------------
    if "constructor_standings" in data_dict:
        cs = data_dict["constructor_standings"].copy()
        cs = cs.rename(
            columns={
                "points": "cs_points",
                "position": "cs_position",
                "wins": "cs_wins",
                "positionText": "cs_positionText",
            }
        )
        master = master.merge(
            cs, on=["raceId", "constructorId"], how="left", suffixes=("", "_cs")
        )
        print(f"[merge] After constructor_standings:   {master.shape}")
    else:
        print("[merge] SKIP constructor_standings — table not available")

    # ------------------------------------------------------------------
    # 10. status  (maps statusId → status description string)
    # ------------------------------------------------------------------
    if "status" in data_dict:
        status = data_dict["status"].copy()
        master = master.merge(status, on="statusId", how="left")
        print(f"[merge] After status:                  {master.shape}")
    else:
        print("[merge] SKIP status — table not available")

    print(f"\n[OK] Master DataFrame shape: {master.shape}")
    return master


def clean_master_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the master DataFrame produced by :func:`merge_master_df`.

    Operations performed (in order)
    --------------------------------
    1. Replace any remaining ``\\N`` strings with ``np.nan``.
    2. Cast obviously numeric columns to appropriate dtypes.
    3. Use ``positionOrder`` as the canonical finish position
       (``position`` in results.csv contains ``\\N`` for DNFs).
    4. Flag ``pit_lane_start`` where ``grid == 0``
       (grid=0 means the driver started from the pit lane, *not* a data
       error — do **not** delete these rows).
    5. Create an ``is_dnf`` boolean flag derived from the ``status``
       column: ``True`` for any status that is neither ``'Finished'``
       nor a ``'+N Laps'`` variant.
    6. Create a ``driver_name`` convenience column (``forename surname``).

    Parameters
    ----------
    df : pd.DataFrame
        Raw master DataFrame from :func:`merge_master_df`.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the master DataFrame.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Mop up any remaining \N sentinels (e.g. from post-merge columns)
    # ------------------------------------------------------------------
    df = _convert_nulls(df)

    # ------------------------------------------------------------------
    # 2. Numeric coercions
    # ------------------------------------------------------------------
    numeric_cols = [
        # results
        "grid", "positionOrder", "points", "laps",
        "milliseconds", "fastestLap", "fastestLapSpeed", "rank",
        # races
        "year", "round",
        # standings
        "ds_points", "ds_position", "ds_wins",
        "cs_points", "cs_position", "cs_wins",
        # aggregates added by this pipeline
        "pit_stop_count", "pit_stop_avg_ms", "pit_stop_min_ms", "pit_stop_total_ms",
        "lap_time_mean_ms", "lap_time_std_ms", "lap_time_best_ms", "laps_completed",
        # qualifying
        "grid_qual_position",
        # circuits
        "lat", "lng", "alt",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 3. Use positionOrder as canonical finish position
    #    (results.position has \N for DNFs which is now NaN after step 1)
    # ------------------------------------------------------------------
    if "positionOrder" in df.columns:
        df["finish_position"] = df["positionOrder"]
    elif "position" in df.columns:
        df["finish_position"] = pd.to_numeric(df["position"], errors="coerce")

    # ------------------------------------------------------------------
    # 4. Flag pit-lane starts  (grid == 0 is valid, not a data error)
    # ------------------------------------------------------------------
    if "grid" in df.columns:
        df["pit_lane_start"] = df["grid"] == 0

    # ------------------------------------------------------------------
    # 5. DNF flag
    #    'Finished' → not DNF
    #    '+1 Lap', '+2 Laps', … → still classified, not DNF
    #    Everything else (engine, accident, …) → DNF
    # ------------------------------------------------------------------
    if "status" in df.columns:
        finished_pattern = r"^(\+\d+\s+Laps?|Finished)$"
        df["is_dnf"] = ~df["status"].str.match(finished_pattern, na=False)
    else:
        df["is_dnf"] = np.nan  # can't determine without status table

    # ------------------------------------------------------------------
    # 6. Convenience driver_name column
    # ------------------------------------------------------------------
    if "forename" in df.columns and "surname" in df.columns:
        df["driver_name"] = (
            df["forename"].fillna("").str.strip()
            + " "
            + df["surname"].fillna("").str.strip()
        ).str.strip()

    print(f"[clean] Cleaning complete — final shape: {df.shape}")
    return df


def get_data_summary(df: pd.DataFrame) -> None:
    """Print a concise diagnostic summary of *df*.

    Outputs
    -------
    * Shape (rows × columns)
    * Per-column dtype and non-null count
    * Top-20 columns by missing-value percentage (only if any are missing)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to summarise (typically the cleaned master DataFrame).
    """
    print("=" * 60)
    print(f"  Shape : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print("=" * 60)

    # dtype overview
    print("\n── Column dtypes ──────────────────────────────────────────")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):<15}  {count} column(s)")

    # missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = (
        pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
    )

    if missing_df.empty:
        print("\n── Missing values ─────────────────────────────────────────")
        print("  No missing values detected.")
    else:
        print(f"\n── Top missing columns ({len(missing_df)} total with gaps) ────────────")
        print(missing_df.head(20).to_string())

    # memory usage
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"\n── Memory usage: {mem_mb:.1f} MB")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------


def build_master(data_path: str = "../data/raw/") -> pd.DataFrame:
    """One-shot helper: load → merge → clean → return master DataFrame.

    Parameters
    ----------
    data_path : str, optional
        Path to the directory containing the raw CSV files.

    Returns
    -------
    pd.DataFrame
        Cleaned master DataFrame ready for EDA or modelling.
    """
    data = load_all_data(data_path)
    if not data:
        raise RuntimeError(f"No data loaded from '{data_path}'. Check the path and retry.")
    master = merge_master_df(data)
    master = clean_master_df(master)
    return master


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Allow overriding the data path from the command line:
    #   python data_loader.py /path/to/data/raw/
    DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/"

    print("\n" + "=" * 60)
    print("  F1 ML Project — data_loader.py demo")
    print("=" * 60 + "\n")

    # Step 1: load raw tables
    print(">>> Step 1: Loading raw CSV files …\n")
    data = load_all_data(DATA_PATH)

    if not data:
        print("\n[DEMO] No data files found. "
              "Download the Kaggle F1 dataset and place CSVs in '../data/raw/'.")
        sys.exit(0)

    # Step 2: merge into master DataFrame
    print("\n>>> Step 2: Merging tables …\n")
    master = merge_master_df(data)

    # Step 3: clean
    print("\n>>> Step 3: Cleaning …\n")
    master = clean_master_df(master)

    # Step 4: summary
    print("\n>>> Step 4: Data summary\n")
    get_data_summary(master)

    # Step 5: quick sanity checks
    print("\n>>> Step 5: Sanity checks\n")

    if "pit_lane_start" in master.columns:
        n_pit_lane = master["pit_lane_start"].sum()
        print(f"  Pit-lane starts flagged : {n_pit_lane:,}")

    if "is_dnf" in master.columns:
        n_dnf = master["is_dnf"].sum()
        pct = n_dnf / len(master) * 100
        print(f"  DNF rows                : {n_dnf:,}  ({pct:.1f}%)")

    if "driver_name" in master.columns:
        print(f"  Sample driver names     : {master['driver_name'].dropna().unique()[:5].tolist()}")

    if "finish_position" in master.columns:
        print(f"  finish_position range   : "
              f"{master['finish_position'].min():.0f} – {master['finish_position'].max():.0f}")

    print("\n[DONE] data_loader demo complete.\n")
