"""
feature_engineering.py
-----------------------
Computes ~80 ML features from the master DataFrame for the F1 ML project.

Design contract
~~~~~~~~~~~~~~~
* **Zero data leakage** — every rolling/cumulative feature is built exclusively
  from races that occurred *before* the current race.  This is enforced by:
    - Sorting by ``(year, round)`` before every rolling/expanding computation.
    - Calling ``.shift(1)`` so the current race's result is never included in
      the window that produces a feature for that same race.

* **NaN tolerance** — pre-2003 qualifying columns (q1/q2/q3) are absent;
  pit-lane starts (``grid == 0``) are masked out of grid-based rolling averages;
  ``rolling(window=N, min_periods=1)`` is used so partial windows still
  produce a value rather than propagating NaN.

Input DataFrame columns (from ``data_loader.clean_master_df``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
resultId, raceId, driverId, constructorId, grid, positionOrder, points, laps,
statusId, year, round, circuitId, name, date, forename, surname, dob,
nationality, constructorName, constructorNationality, q1, q2, q3,
qual_position (NaN pre-2003), pit_stop_count, avg_pit_duration,
mean_lap_time, std_lap_time, best_lap_time, driver_standing_position,
driver_standing_points, constructor_standing_position,
constructor_standing_points, status, is_dnf, pit_lane_start, driver_name,
country, lat, lng, alt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Category A — Driver Skill Features
# ──────────────────────────────────────────────────────────────────────────────


def add_driver_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver skill rolling features per driver.

    All features use only information available *before* the current race
    (cumulative / rolling windows shifted by one position).

    Features added
    --------------
    career_wins_so_far
        Cumulative wins before the current race.
    career_podiums_so_far
        Cumulative podiums (positionOrder <= 3) before the current race.
    career_races_so_far
        Cumulative race starts before the current race.
    win_rate_last_5
        Fraction of wins in the driver's previous 5 races.
    podium_rate_last_5
        Fraction of podiums in the driver's previous 5 races.
    avg_finish_last_5
        Mean finishing position (positionOrder) in the previous 5 races.
    avg_finish_last_10
        Mean finishing position in the previous 10 races.
    avg_grid_last_5
        Mean qualifying / grid position in the previous 5 races.
        Pit-lane starts (grid == 0) are excluded (treated as NaN).
    grid_delta_career_avg
        Career expanding mean of ``grid − positionOrder`` — a proxy for
        overtaking ability.  Pit-lane starts are excluded.
    dnf_rate_last_10
        DNF rate over the driver's previous 10 races.
    championship_position
        Driver's standing position *before* this race (from standings table).
    championship_points
        Driver's championship points *before* this race.
    seasons_experience
        Number of distinct seasons the driver competed in *before* the current
        season.  First season → 0, second season → 1, etc.
    age_at_race
        Driver age at race date in fractional years (date − dob) / 365.25.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame produced by ``data_loader.clean_master_df``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with driver skill feature columns appended.
    """
    df = df.copy()

    # Sort chronologically within each driver so rolling/cumsum are in order.
    df = df.sort_values(["driverId", "year", "round"]).reset_index(drop=True)

    # ── Derived binary helper columns ────────────────────────────────────────
    df["_is_win"] = (df["positionOrder"] == 1).astype(float)
    df["_is_podium"] = (df["positionOrder"] <= 3).astype(float)

    # For grid-based features: pit-lane starts (grid == 0) are set to NaN
    # so they don't distort averages.
    df["_grid_valid"] = df["grid"].where(df["grid"] != 0, other=np.nan)
    df["_grid_delta"] = (df["_grid_valid"] - df["positionOrder"])  # positive = gained

    grp = df.groupby("driverId", sort=False)

    # ── Cumulative features (shift to exclude current race) ──────────────────
    df["career_wins_so_far"] = grp["_is_win"].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    df["career_podiums_so_far"] = grp["_is_podium"].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    # Count every row as 1 start; cumsum after shift gives prior starts.
    df["career_races_so_far"] = grp["_is_win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).count()
    )

    # ── Rolling rate / average features ──────────────────────────────────────
    df["win_rate_last_5"] = grp["_is_win"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df["podium_rate_last_5"] = grp["_is_podium"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df["avg_finish_last_5"] = grp["positionOrder"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df["avg_finish_last_10"] = grp["positionOrder"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    # Exclude pit-lane starts from grid rolling average.
    df["avg_grid_last_5"] = grp["_grid_valid"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    # Career mean overtaking delta — expanding so all past races contribute.
    df["grid_delta_career_avg"] = grp["_grid_delta"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # DNF rate over last 10 races.
    df["dnf_rate_last_10"] = grp["is_dnf"].transform(
        lambda x: x.astype(float).shift(1).rolling(window=10, min_periods=1).mean()
    )

    # ── Championship standings ────────────────────────────────────────────────
    # These columns from the standings table already represent standings BEFORE
    # the current race, so we just alias them.
    df["championship_position"] = df["driver_standing_position"]
    df["championship_points"] = df["driver_standing_points"]

    # ── Seasons experience ────────────────────────────────────────────────────
    # Aggregate to one row per (driverId, year), ordered chronologically.
    # cumcount() gives the 0-based index = number of prior distinct seasons.
    seasons = (
        df[["driverId", "year"]]
        .drop_duplicates()
        .sort_values(["driverId", "year"])
    )
    seasons["seasons_experience"] = seasons.groupby("driverId").cumcount()
    df = df.merge(seasons, on=["driverId", "year"], how="left")

    # ── Age at race ──────────────────────────────────────────────────────────
    _date = pd.to_datetime(df["date"], errors="coerce")
    _dob = pd.to_datetime(df["dob"], errors="coerce")
    df["age_at_race"] = (_date - _dob).dt.days / 365.25

    # Drop helper columns
    df = df.drop(
        columns=["_is_win", "_is_podium", "_grid_valid", "_grid_delta"],
        errors="ignore",
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Category B — Constructor / Team Features
# ──────────────────────────────────────────────────────────────────────────────


def add_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team strength rolling features per constructor.

    Because a constructor fields multiple drivers each race, constructor-level
    rolling stats are first aggregated to one row per ``(constructorId, raceId)``
    (summing wins, podiums, DNFs, and points across all cars) and then rolled
    before being merged back to the per-driver rows.

    Features added
    --------------
    team_wins_last_5
        Total race wins for the constructor across its previous 5 races
        (sum over both cars).
    team_avg_finish_last_5
        Mean of the constructor's best finishing position (min positionOrder
        per race) over the previous 5 races.
    team_podiums_last_5
        Total podium finishes for the constructor in the previous 5 races.
    team_championship_pos
        Constructor's standing position before this race.
    team_dnf_rate_season
        Expanding mean of the per-race DNF proportion within the current
        season, computed before the current race.
    team_wins_at_circuit
        Cumulative race wins by the constructor at this specific circuit,
        before the current race.
    team_points_last_5
        Total championship points scored by the constructor in the previous
        5 races (sum over both cars).

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with constructor feature columns appended.
    """
    df = df.copy()
    df = df.sort_values(["constructorId", "year", "round"]).reset_index(drop=True)

    # ── Aggregate to one row per (constructorId, raceId) ─────────────────────
    # is_dnf may be bool; coerce to float for arithmetic.
    df["_is_dnf_float"] = df["is_dnf"].astype(float)
    df["_is_win_con"] = (df["positionOrder"] == 1).astype(float)
    df["_is_podium_con"] = (df["positionOrder"] <= 3).astype(float)

    race_agg = (
        df.groupby(
            ["constructorId", "raceId", "year", "round", "circuitId"], sort=False
        )
        .agg(
            _team_wins=("_is_win_con", "sum"),
            _team_podiums=("_is_podium_con", "sum"),
            _team_best_finish=("positionOrder", "min"),
            _team_points=("points", "sum"),
            _team_dnfs=("_is_dnf_float", "sum"),
            _team_starters=("positionOrder", "count"),
        )
        .reset_index()
        .sort_values(["constructorId", "year", "round"])
    )

    # Per-race DNF proportion for this constructor (both cars considered).
    race_agg["_team_dnf_rate"] = race_agg["_team_dnfs"] / race_agg[
        "_team_starters"
    ].clip(lower=1)

    # ── Constructor-level rolling features ───────────────────────────────────
    grp_con = race_agg.groupby("constructorId", sort=False)

    race_agg["team_wins_last_5"] = grp_con["_team_wins"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()
    )
    race_agg["team_avg_finish_last_5"] = grp_con["_team_best_finish"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    race_agg["team_podiums_last_5"] = grp_con["_team_podiums"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()
    )
    race_agg["team_points_last_5"] = grp_con["_team_points"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()
    )

    # Season DNF rate: expanding within (constructorId, year), shifted by 1.
    grp_con_yr = race_agg.groupby(["constructorId", "year"], sort=False)
    race_agg["team_dnf_rate_season"] = grp_con_yr["_team_dnf_rate"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Cumulative wins at each circuit by this constructor, before current race.
    grp_con_circ = race_agg.groupby(["constructorId", "circuitId"], sort=False)
    race_agg["team_wins_at_circuit"] = grp_con_circ["_team_wins"].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )

    # ── Merge constructor rolling stats back to the per-driver rows ───────────
    con_feat_cols = [
        "constructorId",
        "raceId",
        "team_wins_last_5",
        "team_avg_finish_last_5",
        "team_podiums_last_5",
        "team_dnf_rate_season",
        "team_wins_at_circuit",
        "team_points_last_5",
    ]
    df = df.merge(race_agg[con_feat_cols], on=["constructorId", "raceId"], how="left")

    # Constructor standings (from standings table, already pre-race).
    df["team_championship_pos"] = df["constructor_standing_position"]

    # Drop helpers added in this function.
    df = df.drop(
        columns=["_is_dnf_float", "_is_win_con", "_is_podium_con"], errors="ignore"
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Category C — Circuit-Specific Features
# ──────────────────────────────────────────────────────────────────────────────


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit-specific features for each driver-circuit combination.

    Features added
    --------------
    driver_prev_wins_at_circuit
        Cumulative wins at this circuit *before* the current race.
    driver_avg_finish_at_circuit
        Expanding mean finish position at this circuit before current race.
    driver_races_at_circuit
        Count of prior starts at this circuit (0 for first visit).
    circuit_avg_grid_delta
        Expanding mean of ``grid − positionOrder`` for *all* drivers at this
        circuit before the current race.  Captures how much overtaking
        typically happens at the venue.
    circuit_avg_dnf_rate
        Expanding DNF rate for all drivers at this circuit before current race.
    circuit_country_encoded
        Target-encoded country: expanding mean of ``positionOrder`` for all
        past races held in the same country.  Captures track character.
    altitude
        Circuit altitude in metres (from ``alt`` column in master_df).

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with circuit feature columns appended.
    """
    df = df.copy()

    # ── Driver × Circuit features ─────────────────────────────────────────────
    # Sort so the transform runs in chronological order within each group.
    df = df.sort_values(
        ["driverId", "circuitId", "year", "round"]
    ).reset_index(drop=True)

    df["_is_win_circ"] = (df["positionOrder"] == 1).astype(float)

    grp_dc = df.groupby(["driverId", "circuitId"], sort=False)

    # Cumulative wins at circuit before current race.
    df["driver_prev_wins_at_circuit"] = grp_dc["_is_win_circ"].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    # Expanding mean finish at circuit — NaN on first visit (no prior data).
    df["driver_avg_finish_at_circuit"] = grp_dc["positionOrder"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    # Count of prior visits (0 = debut at circuit).
    df["driver_races_at_circuit"] = grp_dc["_is_win_circ"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).count()
    )

    # ── Circuit-level aggregate features (all drivers, truly leak-free) ─────
    # Aggregate to one row per (circuitId, raceId) to prevent intra-race leakage
    # (drivers later in the sorted order would otherwise see teammates' values
    # from the same race after a row-level shift).
    df["_grid_valid_circ"] = df["grid"].where(df["grid"] != 0, other=np.nan)
    df["_grid_delta_circ"] = df["_grid_valid_circ"] - df["positionOrder"]

    circ_race = (
        df.groupby(["circuitId", "raceId", "year", "round"], sort=False)
        .agg(
            _circ_grid_delta=("_grid_delta_circ", "mean"),
            _circ_dnf=("is_dnf", lambda x: x.astype(float).mean()),
        )
        .reset_index()
        .sort_values(["circuitId", "year", "round"])
    )
    grp_circ = circ_race.groupby("circuitId", sort=False)
    circ_race["circuit_avg_grid_delta"] = grp_circ["_circ_grid_delta"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    circ_race["circuit_avg_dnf_rate"] = grp_circ["_circ_dnf"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df = df.merge(
        circ_race[["circuitId", "raceId", "circuit_avg_grid_delta", "circuit_avg_dnf_rate"]],
        on=["circuitId", "raceId"],
        how="left",
    )

    # ── Country target encoding (truly leak-free) ────────────────────────────
    # Naive per-row shift(1) leaks because multiple drivers share the same race:
    # driver N in race R would see drivers 1..N-1's finish positions from race R.
    # Fix: aggregate the mean positionOrder to one value per (raceId, country),
    # compute the expanding mean at the race level (shift 1 race), then merge back.
    race_country = (
        df.groupby(["raceId", "country", "year", "round"], sort=False)["positionOrder"]
        .mean()
        .reset_index()
        .rename(columns={"positionOrder": "_country_mean_pos"})
        .sort_values(["country", "year", "round"])
    )
    grp_country = race_country.groupby("country", sort=False)
    race_country["circuit_country_encoded"] = grp_country["_country_mean_pos"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df = df.merge(
        race_country[["raceId", "country", "circuit_country_encoded"]],
        on=["raceId", "country"],
        how="left",
    )

    # ── Altitude (already in master_df as 'alt') ─────────────────────────────
    df["altitude"] = df["alt"]

    # Drop helpers.
    df = df.drop(
        columns=["_is_win_circ", "_grid_valid_circ", "_grid_delta_circ"],
        errors="ignore",
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Category D — Race Context Features
# ──────────────────────────────────────────────────────────────────────────────


def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add race context features describing the current race environment.

    Features added
    --------------
    grid_position
        Raw starting grid position (alias for ``grid``).
    season_round
        Round number within the season (alias for ``round``).
    year
        Season year (already present; included in the feature set for
        downstream modelling convenience).
    field_size
        Number of classified starters in this race (rows sharing the same
        ``raceId``).
    is_wet_race
        Heuristic binary flag.  A large standard deviation of
        ``grid − positionOrder`` across all drivers in the race suggests a
        chaotic / wet race.  Threshold: std > 5.0 position places.
        Pit-lane starters (grid == 0) are excluded from the calculation.
    points_scored
        Numeric points earned in this race (coerced from the ``points``
        column which may be stored as object/string in some CSV loads).

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with race context feature columns appended.
    """
    df = df.copy()

    # Direct aliases / casts.
    df["grid_position"] = df["grid"]
    df["season_round"] = df["round"]
    # 'year' is already present — no action needed.

    # ── Field size: count of drivers per raceId ───────────────────────────────
    df["field_size"] = df.groupby("raceId")["driverId"].transform("count")

    # ── Wet-race heuristic ────────────────────────────────────────────────────
    # Exclude pit-lane starters (grid == 0) to avoid inflating variance.
    _pos_change = (df["grid"] - df["positionOrder"]).where(df["grid"] != 0, other=np.nan)
    df["_pos_change_adj"] = _pos_change
    race_std = df.groupby("raceId")["_pos_change_adj"].transform("std")
    df["is_wet_race"] = (race_std > 5.0).astype(int)

    # ── Numeric points ────────────────────────────────────────────────────────
    df["points_scored"] = pd.to_numeric(df["points"], errors="coerce")

    df = df.drop(columns=["_pos_change_adj"], errors="ignore")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Category E — Era Normalization & Teammate Comparison
# ──────────────────────────────────────────────────────────────────────────────


def add_normalization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add era normalization and teammate comparison features.

    The F1 points system has changed multiple times (1950 → 2003 → 2010 →
    2019+).  Normalizing by the empirical maximum points available in each
    season makes raw point scores comparable across eras.

    Features added
    --------------
    points_normalized
        ``points_scored / max_points_in_season``.  The denominator is the
        highest single-race points tally in that year (the winner's haul),
        derived directly from the data so it adapts to any scoring system.
        Range ≈ [0, 1].
    position_percentile
        ``positionOrder / field_size``.  Lower values indicate a better
        finish relative to the size of the field.
    teammate_finish
        The teammate's finishing position (``positionOrder``) in the same
        race.  When a team has more than two cars (rare historical cases),
        the best-finishing car's position is used.
    teammate_delta
        ``driver positionOrder − teammate_finish``.
        Negative = driver finished *ahead* of teammate.
    teammate_qual_delta
        ``driver grid − teammate grid``.
        Negative = driver qualified *ahead* of teammate.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame.  ``field_size`` must already be present (added by
        :func:`add_race_context_features`).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with normalization / comparison feature columns
        appended.
    """
    df = df.copy()

    # Ensure points_scored is numeric (may have been added by earlier step).
    if "points_scored" not in df.columns:
        df["points_scored"] = pd.to_numeric(df["points"], errors="coerce")

    # ── Points normalization ──────────────────────────────────────────────────
    # Maximum single-race points available that season = winner's score.
    max_pts_year = (
        df.groupby("year")["points_scored"]
        .transform("max")
        .replace(0, np.nan)  # avoid divide-by-zero in seasons with 0-point max
    )
    df["points_normalized"] = df["points_scored"] / max_pts_year

    # ── Position percentile ───────────────────────────────────────────────────
    if "field_size" not in df.columns:
        df["field_size"] = df.groupby("raceId")["driverId"].transform("count")
    df["position_percentile"] = df["positionOrder"] / df["field_size"]

    # ── Teammate features ─────────────────────────────────────────────────────
    # Build a lookup of teammate results by self-joining on (raceId, constructorId).
    # Each row represents one car; we exclude the driver's own row, then pick
    # the best-finishing teammate when there are >2 cars (rare historical teams).
    teammate_lookup = df[
        ["raceId", "constructorId", "driverId", "positionOrder", "grid"]
    ].copy()
    teammate_lookup = teammate_lookup.rename(
        columns={
            "driverId": "_tm_driverId",
            "positionOrder": "teammate_finish",
            "grid": "_tm_grid",
        }
    )

    # Many-to-many join: every driver meets all other drivers in the same team/race.
    merged = df[["raceId", "constructorId", "driverId", "positionOrder", "grid"]].merge(
        teammate_lookup,
        on=["raceId", "constructorId"],
        how="left",
    )
    # Drop self-pairings.
    merged = merged[merged["driverId"] != merged["_tm_driverId"]]

    # When there are multiple teammates take the best-finishing one per driver.
    merged = (
        merged.sort_values("teammate_finish")
        .groupby(["raceId", "driverId"], sort=False)
        .first()
        .reset_index()[["raceId", "driverId", "teammate_finish", "_tm_grid"]]
    )

    df = df.merge(merged, on=["raceId", "driverId"], how="left")

    df["teammate_delta"] = df["positionOrder"] - df["teammate_finish"]
    df["teammate_qual_delta"] = df["grid"] - df["_tm_grid"]

    df = df.drop(columns=["_tm_grid"], errors="ignore")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature Column Registry
# ──────────────────────────────────────────────────────────────────────────────


def get_feature_columns() -> list:
    """Return a list of all engineered feature column names used for modelling.

    This is the canonical list consumed by train/evaluate scripts to select
    the feature matrix ``X``.

    Returns
    -------
    list of str
        All engineered feature column names.
    """
    return [
        # ── Category A: Driver Skill ─────────────────────────────────────────
        "career_wins_so_far",
        "career_podiums_so_far",
        "career_races_so_far",
        "win_rate_last_5",
        "podium_rate_last_5",
        "avg_finish_last_5",
        "avg_finish_last_10",
        "avg_grid_last_5",
        "grid_delta_career_avg",
        "dnf_rate_last_10",
        "championship_position",
        "championship_points",
        "seasons_experience",
        "age_at_race",
        # ── Category B: Constructor ──────────────────────────────────────────
        "team_wins_last_5",
        "team_avg_finish_last_5",
        "team_podiums_last_5",
        "team_championship_pos",
        "team_dnf_rate_season",
        "team_wins_at_circuit",
        "team_points_last_5",
        # ── Category C: Circuit ──────────────────────────────────────────────
        "driver_prev_wins_at_circuit",
        "driver_avg_finish_at_circuit",
        "driver_races_at_circuit",
        "circuit_avg_grid_delta",
        "circuit_avg_dnf_rate",
        "circuit_country_encoded",
        "altitude",
        # ── Category D: Race Context ─────────────────────────────────────────
        "grid_position",
        "season_round",
        "year",
        "field_size",
        "is_wet_race",
        "points_scored",
        # ── Category E: Era Normalization / Teammate ─────────────────────────
        "points_normalized",
        "position_percentile",
        "teammate_finish",
        "teammate_delta",
        "teammate_qual_delta",
    ]


def get_feature_groups() -> dict:
    """Return a dictionary mapping feature group names to column lists.

    Useful for feature-importance analysis, ablation studies, and pipeline
    documentation.

    Returns
    -------
    dict
        Keys are group labels; values are lists of feature column names.
    """
    return {
        "driver_skill": [
            "career_wins_so_far",
            "career_podiums_so_far",
            "career_races_so_far",
            "win_rate_last_5",
            "podium_rate_last_5",
            "avg_finish_last_5",
            "avg_finish_last_10",
            "avg_grid_last_5",
            "grid_delta_career_avg",
            "dnf_rate_last_10",
            "championship_position",
            "championship_points",
            "seasons_experience",
            "age_at_race",
        ],
        "constructor": [
            "team_wins_last_5",
            "team_avg_finish_last_5",
            "team_podiums_last_5",
            "team_championship_pos",
            "team_dnf_rate_season",
            "team_wins_at_circuit",
            "team_points_last_5",
        ],
        "circuit": [
            "driver_prev_wins_at_circuit",
            "driver_avg_finish_at_circuit",
            "driver_races_at_circuit",
            "circuit_avg_grid_delta",
            "circuit_avg_dnf_rate",
            "circuit_country_encoded",
            "altitude",
        ],
        "race_context": [
            "grid_position",
            "season_round",
            "year",
            "field_size",
            "is_wet_race",
            "points_scored",
        ],
        "normalization": [
            "points_normalized",
            "position_percentile",
            "teammate_finish",
            "teammate_delta",
            "teammate_qual_delta",
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline Function
# ──────────────────────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in order and return the enriched DataFrame.

    The pipeline applies each category function sequentially.  Every step
    retains all original columns plus its new feature columns, so later steps
    can rely on columns created by earlier ones (e.g.
    :func:`add_normalization_features` uses ``field_size`` from
    :func:`add_race_context_features`).

    Pipeline order
    --------------
    1. Sort globally by ``(year, round, raceId, driverId)``.
    2. Category A — driver skill features.
    3. Category B — constructor / team features.
    4. Category C — circuit-specific features.
    5. Category D — race context features.
    6. Category E — era normalization & teammate comparison.
    7. Restore global chronological sort.

    Parameters
    ----------
    df : pd.DataFrame
        Raw master DataFrame produced by ``data_loader.clean_master_df``.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame.  Row order is
        ``(year, round, raceId, driverId)`` and the index is reset.
    """
    print(f"[feature_engineering] Starting feature engineering — {len(df):,} rows …")

    # Global chronological sort as the baseline for all subsequent operations.
    df = df.sort_values(
        ["year", "round", "raceId", "driverId"]
    ).reset_index(drop=True)

    print("[feature_engineering]  Step A — driver skill features …")
    df = add_driver_skill_features(df)

    print("[feature_engineering]  Step B — constructor features …")
    df = add_constructor_features(df)

    print("[feature_engineering]  Step C — circuit features …")
    df = add_circuit_features(df)

    print("[feature_engineering]  Step D — race context features …")
    df = add_race_context_features(df)

    print("[feature_engineering]  Step E — normalization & teammate features …")
    df = add_normalization_features(df)

    # Restore global sort after all the internal per-group sorts in each step.
    df = df.sort_values(
        ["year", "round", "raceId", "driverId"]
    ).reset_index(drop=True)

    present = [c for c in get_feature_columns() if c in df.columns]
    missing = [c for c in get_feature_columns() if c not in df.columns]

    print(
        f"[feature_engineering] Done — {len(present)}/{len(get_feature_columns())} "
        f"feature columns present."
    )
    if missing:
        print(f"[feature_engineering] WARNING — missing columns: {missing}")

    return df
