"""
hypothetical_engine.py
----------------------
Hypothetical scenario engine for the F1 ML Project.

Allows users to ask "what-if" questions like:
    - "Can Vettel win in an Alfa Romeo?"
    - "How many points would Hamilton score driving for Williams in 2023?"
    - "What would Schumacher's 2012 season look like in a Red Bull?"

Core Approach
~~~~~~~~~~~~~
The engine decouples **driver skill** from **car performance** by extracting
two independent feature profiles from the historical feature DataFrame:

    1. **Driver skill profile** — captured via teammate_delta, win/podium rates,
       qualifying pace, career wins, age, experience, and grid-to-finish delta.
    2. **Car (team) performance profile** — captured via team average finish,
       wins, podiums, constructor championship position, DNF rate, and points.

These two profiles are merged into a single synthetic feature row that the
trained race-winner model can score, yielding a win probability and predicted
finishing position for any driver/team/year combination — even if that pairing
never occurred historically.

Public API
~~~~~~~~~~
    get_driver_skill_profile(featured_df, driver, year)
    get_team_car_profile(featured_df, team, year)
    hypothetical_scenario(featured_df, driver, team, year, **kwargs)

Internal Helpers
~~~~~~~~~~~~~~~~
    _apply_parameter_overrides(feature_row, params)
    _simulate_single_race(feature_row, model, reliability_factor, race_incidents)
    _simulate_full_season(driver_profile, team_profile, year, featured_df, params, model)
    _get_f1_points(position, year)
    _get_year_calendar(featured_df, year)
"""

from __future__ import annotations

import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# F1 Points Systems by Era
# ---------------------------------------------------------------------------

# Modern system (2010+): top 10 score points
_POINTS_2010_PLUS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                     6: 8,  7: 6,  8: 4,  9: 2,  10: 1}

# 1991–2009 system: top 8 score points
_POINTS_1991_2009 = {1: 10, 2: 6, 3: 4, 4: 3, 5: 2, 6: 1}

# Pre-1991 system: top 6 score points
_POINTS_PRE_1991 = {1: 9, 2: 6, 3: 4, 4: 3, 5: 2, 6: 1}

# ---------------------------------------------------------------------------
# Regulation Era Adjustments
# ---------------------------------------------------------------------------
# Each era key maps to a dict of feature deltas applied on top of the
# extracted profiles.  Positive values worsen the feature; negative improve.
_ERA_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "v10": {
        # Pre-2006: high-revving V10s, mechanical grip dominant
        "team_dnf_rate_season": 0.05,   # more mechanical failures
        "grid_delta_career_avg": 0.5,   # overtaking harder on narrow circuits
    },
    "v8": {
        # 2006–2013: V8 era, aero-dominated, Pirelli/Bridgestone
        "team_dnf_rate_season": 0.02,
        "grid_delta_career_avg": 0.2,
    },
    "v6_hybrid": {
        # 2014–2021: turbo-hybrid era, massive PU advantage
        "team_avg_finish_last_5": -0.5,  # top teams dominated more
        "team_dnf_rate_season": 0.03,    # reliability concerns early on
    },
    "ground_effect": {
        # 2022+: ground effect aerodynamics, closer racing
        "team_avg_finish_last_5": 0.3,   # field closer together
        "team_dnf_rate_season": 0.04,    # new regs → teething issues
        "grid_delta_career_avg": -0.3,   # overtaking marginally easier
    },
}

# ---------------------------------------------------------------------------
# Teammate Skill Level → teammate_delta offset
# ---------------------------------------------------------------------------
_TEAMMATE_SKILL_DELTA: dict[str, float] = {
    "elite":    -2.0,  # beating a world-champion-calibre teammate is hard
    "midfield":  0.0,  # baseline — average midfield driver
    "rookie":    3.0,  # significant edge expected over a rookie
}

# Column candidates for driver name (tried in order)
_DRIVER_NAME_COLS = ["driver_name", "driverRef", "driverId"]
# Column candidates for constructor/team name
_TEAM_NAME_COLS = ["constructorName", "constructorRef", "constructorId", "team"]


# ===========================================================================
# Internal utilities
# ===========================================================================

def _resolve_driver_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first available driver-name column present in *df*."""
    for col in _DRIVER_NAME_COLS:
        if col in df.columns:
            return col
    return None


def _resolve_team_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first available team/constructor-name column present in *df*."""
    for col in _TEAM_NAME_COLS:
        if col in df.columns:
            return col
    return None


def _icontains(series: pd.Series, query: str) -> pd.Series:
    """Case-insensitive substring mask for a string Series."""
    return series.astype(str).str.contains(query, case=False, na=False, regex=False)


def _find_driver_rows(
    df: pd.DataFrame,
    driver: str,
    year: int,
    window: int = 3,
) -> pd.DataFrame:
    """
    Return rows for *driver* at *year* (or within ±*window* years).

    Parameters
    ----------
    df:
        Full featured DataFrame.
    driver:
        Driver name / partial name (case-insensitive substring match).
    year:
        Target season year.
    window:
        Year range to fall back to when the exact year yields no rows.

    Returns
    -------
    pd.DataFrame
        Filtered rows, or empty DataFrame if driver not found at all.
    """
    driver_col = _resolve_driver_col(df)
    if driver_col is None:
        return pd.DataFrame()

    mask_driver = _icontains(df[driver_col], driver)
    driver_df = df[mask_driver]

    if driver_df.empty:
        return driver_df

    # Exact year first
    exact = driver_df[driver_df["year"] == year]
    if not exact.empty:
        return exact

    # Fall back to nearest available year within window
    years_available = driver_df["year"].unique()
    nearby = years_available[np.abs(years_available - year) <= window]
    if len(nearby) == 0:
        # Use closest year overall
        closest = years_available[np.argmin(np.abs(years_available - year))]
        nearby = np.array([closest])
    best_year = nearby[np.argmin(np.abs(nearby - year))]
    return driver_df[driver_df["year"] == best_year]


def _find_team_rows(
    df: pd.DataFrame,
    team: str,
    year: int,
    window: int = 3,
) -> pd.DataFrame:
    """
    Return rows for *team* at *year* (or within ±*window* years).

    Same fallback logic as :func:`_find_driver_rows`.
    """
    team_col = _resolve_team_col(df)
    if team_col is None:
        return pd.DataFrame()

    mask_team = _icontains(df[team_col], team)
    team_df = df[mask_team]

    if team_df.empty:
        return team_df

    exact = team_df[team_df["year"] == year]
    if not exact.empty:
        return exact

    years_available = team_df["year"].unique()
    nearby = years_available[np.abs(years_available - year) <= window]
    if len(nearby) == 0:
        closest = years_available[np.argmin(np.abs(years_available - year))]
        nearby = np.array([closest])
    best_year = nearby[np.argmin(np.abs(nearby - year))]
    return team_df[team_df["year"] == best_year]


# ===========================================================================
# Public Profile Extractors
# ===========================================================================

def get_driver_skill_profile(
    featured_df: pd.DataFrame,
    driver: str,
    year: int,
) -> dict:
    """
    Extract a driver's skill feature profile for a given year (or nearest
    available year within ±3 seasons).

    The returned dictionary captures **driver-intrinsic** skill attributes that
    remain meaningful when the driver is transplanted to a different team:

    Features extracted
    ------------------
    avg_finish_last_10
        Rolling mean finishing position over the previous 10 races.
    win_rate_last_5
        Fraction of wins in the driver's previous 5 races.
    podium_rate_last_5
        Fraction of podiums in the driver's previous 5 races.
    dnf_rate_last_10
        DNF (did-not-finish) rate over the previous 10 races.
    career_wins_so_far
        Cumulative wins up to (but not including) the current race.
    teammate_delta
        Career-average finishing position relative to teammate. Negative
        means the driver beats teammates consistently.
    age_at_race
        Driver age at race date in fractional years.
    seasons_experience
        Number of distinct seasons competed in before the current season.
    grid_delta_career_avg
        Career expanding mean of ``grid − positionOrder``. Positive means
        the driver overtakes more than they lose positions on average.

    Parameters
    ----------
    featured_df : pd.DataFrame
        Full feature-engineered DataFrame (output of engineer_features).
    driver : str
        Driver name or partial name (case-insensitive substring).
    year : int
        Target season year.

    Returns
    -------
    dict
        Mapping of skill feature names to scalar float values (medians over
        all matching rows to smooth out race-to-race noise).

    Raises
    ------
    ValueError
        If the driver cannot be found anywhere in featured_df.
    """
    rows = _find_driver_rows(featured_df, driver, year)

    if rows.empty:
        driver_col = _resolve_driver_col(featured_df)
        available = (
            sorted(featured_df[driver_col].dropna().unique().tolist())
            if driver_col
            else []
        )
        print(
            f"[hypothetical_engine] Driver '{driver}' not found in data.\n"
            f"  Available drivers (sample): {available[:20]}"
        )
        raise ValueError(f"Driver '{driver}' not found in featured_df.")

    used_year = int(rows["year"].iloc[0])
    if used_year != year:
        print(
            f"[hypothetical_engine] '{driver}' has no data for {year}; "
            f"using nearest year {used_year} instead."
        )

    # ── Extract skill features (median over all matching rows) ──────────────
    def _med(col: str, fallback: float = 0.0) -> float:
        if col in rows.columns:
            val = rows[col].dropna()
            return float(val.median()) if not val.empty else fallback
        return fallback

    # Teammate delta: use career mean across all available rows for the driver
    # to get a stable cross-team measure of relative skill.
    driver_col = _resolve_driver_col(featured_df)
    all_driver_rows = featured_df[_icontains(featured_df[driver_col], driver)]
    tm_delta_all = (
        all_driver_rows["teammate_delta"].dropna()
        if "teammate_delta" in all_driver_rows.columns
        else pd.Series(dtype=float)
    )
    career_teammate_delta = float(tm_delta_all.mean()) if not tm_delta_all.empty else 0.0

    profile = {
        "avg_finish_last_10":     _med("avg_finish_last_10",     fallback=10.0),
        "win_rate_last_5":        _med("win_rate_last_5",         fallback=0.0),
        "podium_rate_last_5":     _med("podium_rate_last_5",      fallback=0.0),
        "dnf_rate_last_10":       _med("dnf_rate_last_10",        fallback=0.1),
        "career_wins_so_far":     _med("career_wins_so_far",      fallback=0.0),
        "career_podiums_so_far":  _med("career_podiums_so_far",   fallback=0.0),
        "career_races_so_far":    _med("career_races_so_far",     fallback=0.0),
        "teammate_delta":         career_teammate_delta,
        "age_at_race":            _med("age_at_race",             fallback=27.0),
        "seasons_experience":     _med("seasons_experience",      fallback=3.0),
        "grid_delta_career_avg":  _med("grid_delta_career_avg",   fallback=0.0),
        # Additional rolling window features used as model inputs
        "avg_finish_last_5":      _med("avg_finish_last_5",       fallback=10.0),
        "win_rate_last_5":        _med("win_rate_last_5",         fallback=0.0),
        "podium_rate_last_5":     _med("podium_rate_last_5",      fallback=0.0),
        "avg_grid_last_5":        _med("avg_grid_last_5",         fallback=10.0),
        # Qualifying delta (normalised teammate comparison)
        "teammate_qual_delta":    _med("teammate_qual_delta",     fallback=0.0),
        # Driver-level circuit history (averaged across circuits)
        "driver_prev_wins_at_circuit":  _med("driver_prev_wins_at_circuit", fallback=0.0),
        "driver_avg_finish_at_circuit": _med("driver_avg_finish_at_circuit", fallback=10.0),
        "driver_races_at_circuit":      _med("driver_races_at_circuit",     fallback=1.0),
    }

    return profile


def get_team_car_profile(
    featured_df: pd.DataFrame,
    team: str,
    year: int,
) -> dict:
    """
    Extract a team's car performance feature profile for a given year (or
    nearest available year within ±3 seasons).

    The returned dictionary captures **car/team-intrinsic** attributes that
    remain meaningful when transplanting any driver into the car:

    Features extracted
    ------------------
    team_avg_finish_last_5
        Team's mean best-finisher position over the previous 5 races.
    team_wins_last_5
        Number of team wins over the previous 5 races.
    team_podiums_last_5
        Number of team podiums over the previous 5 races.
    team_championship_pos
        Constructor championship standing position before the current race.
    team_dnf_rate_season
        Team's season-to-date DNF rate.
    team_points_last_5
        Total team points scored over the previous 5 races.
    team_wins_at_circuit
        Cumulative team wins at this circuit (circuit-average used here).

    Parameters
    ----------
    featured_df : pd.DataFrame
        Full feature-engineered DataFrame (output of engineer_features).
    team : str
        Team/constructor name or partial name (case-insensitive substring).
    year : int
        Target season year.

    Returns
    -------
    dict
        Mapping of car feature names to scalar float values (medians over
        all matching rows).

    Raises
    ------
    ValueError
        If the team cannot be found anywhere in featured_df.
    """
    rows = _find_team_rows(featured_df, team, year)

    if rows.empty:
        team_col = _resolve_team_col(featured_df)
        available = (
            sorted(featured_df[team_col].dropna().unique().tolist())
            if team_col
            else []
        )
        print(
            f"[hypothetical_engine] Team '{team}' not found in data.\n"
            f"  Available teams (sample): {available[:20]}"
        )
        raise ValueError(f"Team '{team}' not found in featured_df.")

    used_year = int(rows["year"].iloc[0])
    if used_year != year:
        print(
            f"[hypothetical_engine] '{team}' has no data for {year}; "
            f"using nearest year {used_year} instead."
        )

    def _med(col: str, fallback: float = 0.0) -> float:
        if col in rows.columns:
            val = rows[col].dropna()
            return float(val.median()) if not val.empty else fallback
        return fallback

    profile = {
        "team_avg_finish_last_5":  _med("team_avg_finish_last_5",  fallback=12.0),
        "team_wins_last_5":        _med("team_wins_last_5",         fallback=0.0),
        "team_podiums_last_5":     _med("team_podiums_last_5",      fallback=0.0),
        "team_championship_pos":   _med("team_championship_pos",    fallback=8.0),
        "team_dnf_rate_season":    _med("team_dnf_rate_season",     fallback=0.1),
        "team_points_last_5":      _med("team_points_last_5",       fallback=0.0),
        "team_wins_at_circuit":    _med("team_wins_at_circuit",     fallback=0.0),
        # Circuit-level aggregate features (averaged across team's circuits)
        "circuit_avg_grid_delta":  _med("circuit_avg_grid_delta",   fallback=0.0),
        "circuit_avg_dnf_rate":    _med("circuit_avg_dnf_rate",     fallback=0.1),
        "circuit_country_encoded": _med("circuit_country_encoded",  fallback=0.0),
        "altitude":                _med("altitude",                  fallback=50.0),
    }

    return profile


# ===========================================================================
# Points System Helper
# ===========================================================================

def _get_f1_points(position: int, year: int = 2023) -> int:
    """
    Return F1 championship points for a given finishing position.

    Three distinct points systems are supported based on era:

    * **Pre-1991**    : 9-6-4-3-2-1  (top 6 positions score)
    * **1991–2009**   : 10-6-4-3-2-1 (top 8 positions score)
    * **2010+**       : 25-18-15-12-10-8-6-4-2-1 (top 10 positions score)

    Parameters
    ----------
    position : int
        Finishing position (1 = winner).
    year : int
        Season year, used to select the correct points system.

    Returns
    -------
    int
        Championship points awarded (0 if position is outside the points).
    """
    if year >= 2010:
        return _POINTS_2010_PLUS.get(position, 0)
    elif year >= 1991:
        return _POINTS_1991_2009.get(position, 0)
    else:
        return _POINTS_PRE_1991.get(position, 0)


# ===========================================================================
# Calendar Helper
# ===========================================================================

def _get_year_calendar(
    featured_df: pd.DataFrame,
    year: int,
) -> list[tuple[int, str, int]]:
    """
    Return the list of races for a given season as (raceId, circuitName, round).

    If the year does not exist in the data, falls back to the nearest year
    within ±3 seasons and prints a warning.

    Parameters
    ----------
    featured_df : pd.DataFrame
        Full feature-engineered DataFrame.
    year : int
        Target season year.

    Returns
    -------
    list of (raceId, circuitName, round)
        One entry per unique race in the season, sorted by round number.
    """
    year_df = featured_df[featured_df["year"] == year]

    if year_df.empty:
        years_available = featured_df["year"].unique()
        closest = int(years_available[np.argmin(np.abs(years_available - year))])
        print(
            f"[hypothetical_engine] No races found for {year}; "
            f"using {closest} calendar as proxy."
        )
        year_df = featured_df[featured_df["year"] == closest]

    # Determine circuit name column
    name_col = "name" if "name" in year_df.columns else None

    races = (
        year_df[["raceId", "round"] + ([name_col] if name_col else [])]
        .drop_duplicates("raceId")
        .sort_values("round")
    )

    calendar = []
    for _, row in races.iterrows():
        race_id = int(row["raceId"])
        circuit_name = str(row[name_col]) if name_col else f"Race {int(row['round'])}"
        rnd = int(row["round"])
        calendar.append((race_id, circuit_name, rnd))

    return calendar


# ===========================================================================
# Parameter Override Helper
# ===========================================================================

def _apply_parameter_overrides(
    feature_row: dict,
    params: dict,
) -> dict:
    """
    Apply the 20 configurable scenario parameters to a feature row dict.

    This function mutates a *copy* of feature_row, adjusting numeric feature
    values according to the scenario parameters before the row is fed to the
    trained model.

    Parameter effects
    -----------------
    grid_position_override
        Replaces ``grid_position``. Also adjusts ``avg_grid_last_5``.
    weather (wet)
        Increases ``team_dnf_rate_season`` by 0.10, ``dnf_rate_last_10`` by
        0.05, and reduces grid advantage (``grid_delta_career_avg`` by 0.5)
        to reflect the levelling effect of wet conditions.
    home_race
        Adds a 0.5-position boost to ``avg_finish_last_5`` and
        ``avg_finish_last_10`` (lower is better, so we subtract).
    driver_confidence_factor
        Scales ``win_rate_last_5`` and ``podium_rate_last_5`` by a factor
        derived from the 0–1 input. 0.0 → ×0.5 scaling; 1.0 → ×1.5 scaling.
    teammate_skill_level
        Sets ``teammate_delta`` to –2 (elite), 0 (midfield), or +3 (rookie).
    regulation_era
        Applies era-specific feature deltas from ``_ERA_ADJUSTMENTS``.
    car_development_rate
        Adds ``car_development_rate`` to ``team_avg_finish_last_5`` each round
        (the round number is stored in params as ``_season_round``). Positive
        = development, negative = car gets worse.
    driver_age_override
        Replaces ``age_at_race``.
    num_pit_stops
        If provided, adjusts ``avg_finish_last_10`` by ±0.3 per extra stop
        over the "standard" 1-stop strategy.
    reliability_factor
        Stored; used in _simulate_single_race.
    starting_championship_points
        Sets ``championship_points``.
    season_round
        Updates ``season_round`` feature.
    tire_strategy (soft)
        Aggressive soft-tyre strategy: improves grid_delta_career_avg by 0.5
        but increases dnf_rate_last_10 by 0.02.

    Parameters
    ----------
    feature_row : dict
        Base feature values from combined driver + team profiles.
    params : dict
        All scenario parameters passed to hypothetical_scenario.

    Returns
    -------
    dict
        Updated feature row with overrides applied.
    """
    row = feature_row.copy()

    # ── Grid position override ───────────────────────────────────────────────
    if params.get("grid_position_override") is not None:
        grid = float(params["grid_position_override"])
        row["grid_position"] = grid
        row["avg_grid_last_5"] = grid  # approximate impact on rolling average

    # ── Weather adjustment ───────────────────────────────────────────────────
    if params.get("weather", "dry") == "wet":
        row["is_wet_race"] = 1.0
        row["team_dnf_rate_season"] = row.get("team_dnf_rate_season", 0.1) + 0.10
        row["dnf_rate_last_10"] = row.get("dnf_rate_last_10", 0.1) + 0.05
        # Wet weather reduces the value of raw pace advantage — cars bunch up
        row["grid_delta_career_avg"] = row.get("grid_delta_career_avg", 0.0) - 0.5
    else:
        row["is_wet_race"] = 0.0

    # ── Home race boost ──────────────────────────────────────────────────────
    if params.get("home_race", False):
        # Lower finishing position = better; subtract to simulate boost
        row["avg_finish_last_5"] = max(1.0, row.get("avg_finish_last_5", 10.0) - 0.5)
        row["avg_finish_last_10"] = max(1.0, row.get("avg_finish_last_10", 10.0) - 0.5)

    # ── Driver confidence factor (0.0 → dampens; 1.0 → amplifies) ───────────
    confidence = float(params.get("driver_confidence_factor", 0.5))
    # Scale around 0.5: 0.0 → ×0.5, 0.5 → ×1.0, 1.0 → ×1.5
    confidence_scale = 0.5 + confidence
    row["win_rate_last_5"] = min(1.0, row.get("win_rate_last_5", 0.0) * confidence_scale)
    row["podium_rate_last_5"] = min(1.0, row.get("podium_rate_last_5", 0.0) * confidence_scale)

    # ── Teammate skill level → teammate_delta ────────────────────────────────
    skill_level = params.get("teammate_skill_level", "midfield")
    if skill_level in _TEAMMATE_SKILL_DELTA:
        row["teammate_delta"] = _TEAMMATE_SKILL_DELTA[skill_level]
    # (If an unrecognised level is passed, keep the profiled value)

    # ── Regulation era adjustments ───────────────────────────────────────────
    era = params.get("regulation_era")
    if era and era in _ERA_ADJUSTMENTS:
        for feat, delta in _ERA_ADJUSTMENTS[era].items():
            row[feat] = row.get(feat, 0.0) + delta

    # ── Car development rate (per round) ─────────────────────────────────────
    dev_rate = float(params.get("car_development_rate", 0.0))
    if dev_rate != 0.0:
        rnd = float(params.get("_season_round", 1))
        # Positive dev_rate means car is improving → avg_finish gets lower (better)
        row["team_avg_finish_last_5"] = max(
            1.0, row.get("team_avg_finish_last_5", 10.0) - dev_rate * rnd
        )

    # ── Driver age override ──────────────────────────────────────────────────
    if params.get("driver_age_override") is not None:
        row["age_at_race"] = float(params["driver_age_override"])

    # ── Number of pit stops (relative to 1-stop baseline) ───────────────────
    if params.get("num_pit_stops") is not None:
        extra_stops = int(params["num_pit_stops"]) - 1  # 1-stop = standard
        # Each extra stop costs roughly 0.3 of a position on average
        row["avg_finish_last_10"] = row.get("avg_finish_last_10", 10.0) + 0.3 * extra_stops

    # ── Starting championship points ─────────────────────────────────────────
    row["championship_points"] = float(
        params.get("starting_championship_points", row.get("championship_points", 0.0))
    )

    # ── Season round ─────────────────────────────────────────────────────────
    if params.get("season_round") is not None:
        row["season_round"] = float(params["season_round"])
    elif params.get("_season_round") is not None:
        row["season_round"] = float(params["_season_round"])

    # ── Tire strategy ────────────────────────────────────────────────────────
    tire = params.get("tire_strategy")
    if tire == "soft":
        # Aggressive strategy: faster pace but higher mechanical risk
        row["grid_delta_career_avg"] = row.get("grid_delta_career_avg", 0.0) + 0.5
        row["dnf_rate_last_10"] = min(1.0, row.get("dnf_rate_last_10", 0.1) + 0.02)
    elif tire == "hard":
        # Conservative strategy: slower but more reliable
        row["grid_delta_career_avg"] = row.get("grid_delta_career_avg", 0.0) - 0.3
        row["dnf_rate_last_10"] = max(0.0, row.get("dnf_rate_last_10", 0.1) - 0.01)

    # ── Year override for era-based scoring (pass-through) ───────────────────
    if "year" not in row:
        row["year"] = float(params.get("year", 2023))

    return row


# ===========================================================================
# Single Race Simulation
# ===========================================================================

def _simulate_single_race(
    feature_row: dict,
    model,
    reliability_factor: float = 0.9,
    race_incidents: bool = True,
) -> dict:
    """
    Simulate a single race given a feature row and a trained model.

    The simulation proceeds as:
    1. Apply a DNF roll based on reliability_factor (and track-incident rate).
    2. If not DNF, build a 1-row DataFrame, query the model's predict_proba.
    3. Convert win probability to a predicted finishing position via Gaussian
       noise around an expected rank (lower win_prob → worse position).

    Parameters
    ----------
    feature_row : dict
        Combined, override-applied feature dict ready for model input.
    model : sklearn-compatible classifier
        Must implement predict_proba returning shape (1, 2) probabilities.
    reliability_factor : float
        Probability of *finishing* the race (0–1). E.g. 0.9 = 10% DNF chance.
    race_incidents : bool
        When True, adds a small random incident probability on top of the
        mechanical reliability factor.

    Returns
    -------
    dict
        Keys: win_prob (float), predicted_position (int), dnf_occurred (bool).
    """
    # ── DNF roll ─────────────────────────────────────────────────────────────
    # Base mechanical DNF
    dnf_roll = random.random()
    dnf_threshold = 1.0 - reliability_factor  # e.g. 0.1 for 90% reliability

    # Optional on-track incidents (collisions, spin-offs, etc.)
    if race_incidents:
        incident_extra = 0.03  # ~3% additional incident chance per race
        dnf_threshold = min(1.0, dnf_threshold + incident_extra)

    if dnf_roll < dnf_threshold:
        # Driver did not finish — no points, worst position
        return {"win_prob": 0.0, "predicted_position": 20, "dnf_occurred": True}

    # ── Build feature matrix ─────────────────────────────────────────────────
    # Import here to avoid circular imports at module level
    try:
        from src.models import FEATURE_COLS
    except ImportError:
        from models import FEATURE_COLS  # fallback for direct execution

    row_data = {col: feature_row.get(col, np.nan) for col in FEATURE_COLS}
    X_df = pd.DataFrame([row_data])

    # Impute any remaining NaNs with column medians (median = 0 for unknowns)
    X_arr = X_df.fillna(0.0).to_numpy(dtype=float)

    # ── Model prediction ─────────────────────────────────────────────────────
    if model is not None:
        try:
            proba = model.predict_proba(X_arr)
            # Class index 1 = is_win = True
            win_prob = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        except Exception as exc:
            warnings.warn(
                f"[hypothetical_engine] Model prediction failed: {exc}. "
                "Falling back to heuristic win probability."
            )
            win_prob = _heuristic_win_prob(feature_row)
    else:
        # No model provided — use heuristic based on profile features
        win_prob = _heuristic_win_prob(feature_row)

    # ── Win probability → predicted finishing position ───────────────────────
    # Win probability of ~0.25 → position ≈ 1; ~0.01 → position ≈ 12–15
    # Use an inverse-logistic mapping with Gaussian noise for realism.
    if win_prob > 0.0:
        expected_pos = max(1.0, 1.0 / (win_prob + 1e-6) * 0.25)
        noise = random.gauss(0, 1.5)
        predicted_pos = max(1, min(20, int(round(expected_pos + noise))))
    else:
        predicted_pos = random.randint(12, 20)

    return {
        "win_prob": win_prob,
        "predicted_position": predicted_pos,
        "dnf_occurred": False,
    }


def _heuristic_win_prob(feature_row: dict) -> float:
    """
    Fallback heuristic win probability when no model is available.

    Derived from a simple weighted combination of the most predictive features.
    This is intentionally conservative and is only used as a safety net.

    Parameters
    ----------
    feature_row : dict
        Combined feature dict.

    Returns
    -------
    float
        Estimated win probability in [0, 1].
    """
    # Start from win rate as the primary signal
    win_rate = float(feature_row.get("win_rate_last_5", 0.05))

    # Team quality modifier: championship position 1 → full boost; 10 → no boost
    team_pos = float(feature_row.get("team_championship_pos", 8.0))
    team_quality = max(0.0, (10.0 - team_pos) / 10.0)  # 0 to 1

    # Grid position: starting P1 is a strong predictor
    grid = float(feature_row.get("grid_position", 10.0))
    grid_boost = max(0.0, (20.0 - grid) / 20.0) * 0.15

    # Combine signals
    raw = win_rate * 0.4 + team_quality * 0.4 + grid_boost
    # Clip to sensible range
    return float(np.clip(raw, 0.001, 0.95))


# ===========================================================================
# Full Season Simulation
# ===========================================================================

def _simulate_full_season(
    driver_profile: dict,
    team_profile: dict,
    year: int,
    featured_df: pd.DataFrame,
    params: dict,
    model,
) -> dict:
    """
    Simulate all races in a season and accumulate championship points.

    For each race in the season calendar the function:
    1. Combines the driver skill profile and team car profile into a base row.
    2. Applies per-race parameter overrides (round number, car development, etc.).
    3. Calls :func:`_simulate_single_race`.
    4. Accumulates F1 points, wins, and podiums.

    Parameters
    ----------
    driver_profile : dict
        Output of :func:`get_driver_skill_profile`.
    team_profile : dict
        Output of :func:`get_team_car_profile`.
    year : int
        Season year (used for calendar lookup and points system).
    featured_df : pd.DataFrame
        Full feature-engineered DataFrame.
    params : dict
        Full parameter dict passed down from :func:`hypothetical_scenario`.
    model : sklearn-compatible classifier or None
        Trained race-winner model.

    Returns
    -------
    dict
        Keys: season_points (int), wins (int), podiums (int),
        race_results (list of per-race result dicts).
    """
    calendar = _get_year_calendar(featured_df, year)

    # Optionally limit to a subset of races
    num_races = params.get("num_races_to_simulate")
    if num_races is not None and isinstance(num_races, int) and num_races > 0:
        calendar = calendar[:num_races]

    season_points = params.get("starting_championship_points", 0)
    wins = 0
    podiums = 0
    race_results = []

    for race_id, circuit_name, rnd in calendar:
        # Build combined base feature row for this race
        base_row = {**driver_profile, **team_profile}

        # Inject race-context features
        base_row["season_round"] = float(rnd)
        base_row["field_size"] = 20.0  # standard modern field

        # Per-race circuit data: pull from featured_df for this specific race
        circuit_rows = featured_df[featured_df["raceId"] == race_id]
        if not circuit_rows.empty:
            for circ_feat in [
                "circuit_avg_grid_delta", "circuit_avg_dnf_rate",
                "circuit_country_encoded", "altitude",
                "driver_prev_wins_at_circuit", "driver_avg_finish_at_circuit",
                "driver_races_at_circuit",
            ]:
                if circ_feat in circuit_rows.columns:
                    val = circuit_rows[circ_feat].dropna()
                    if not val.empty:
                        base_row[circ_feat] = float(val.median())

        # Apply all parameter overrides for this specific round
        round_params = dict(params)
        round_params["_season_round"] = rnd  # feed round number to development rate

        feature_row = _apply_parameter_overrides(base_row, round_params)

        # Simulate the race
        race_result = _simulate_single_race(
            feature_row,
            model,
            reliability_factor=params.get("reliability_factor", 0.9),
            race_incidents=params.get("race_incidents", True),
        )

        # Accumulate championship points
        pos = race_result["predicted_position"]
        dnf = race_result["dnf_occurred"]
        pts = _get_f1_points(pos, year) if not dnf else 0
        season_points += pts

        if not dnf:
            if pos == 1:
                wins += 1
            if pos <= 3:
                podiums += 1

        race_results.append(
            {
                "race_id": race_id,
                "circuit": circuit_name,
                "round": rnd,
                "win_prob": race_result["win_prob"],
                "predicted_position": pos,
                "dnf": dnf,
                "points_scored": pts,
            }
        )

    return {
        "season_points": season_points,
        "wins": wins,
        "podiums": podiums,
        "race_results": race_results,
    }


# ===========================================================================
# Main Public Function
# ===========================================================================

def hypothetical_scenario(
    featured_df: pd.DataFrame,
    driver: str,
    team: str,
    year: int,
    track: str = None,
    grid_position_override: int = None,
    teammate: str = None,
    weather: str = "dry",
    num_pit_stops: int = None,
    reliability_factor: float = 0.9,
    season_round: int = None,
    driver_age_override: int = None,
    car_development_rate: float = 0.0,
    tire_strategy: str = None,
    starting_championship_points: int = 0,
    driver_confidence_factor: float = 0.5,
    regulation_era: str = None,
    num_races_to_simulate: int = None,
    teammate_skill_level: str = "midfield",
    race_incidents: bool = True,
    home_race: bool = False,
    model=None,
    n_monte_carlo: int = 1000,
) -> dict:
    """
    Run a hypothetical F1 scenario by combining any driver's skill profile
    with any team's car performance profile.

    This is the primary public interface of the hypothetical engine. It:

    1. Extracts the driver's **skill profile** from featured_df.
    2. Extracts the team's **car profile** from featured_df.
    3. Merges the two profiles into a synthetic feature row.
    4. Applies all 20 configurable parameters via
       :func:`_apply_parameter_overrides`.
    5. For a single race: predicts win probability and finishing position.
    6. For a full season (track is None or num_races_to_simulate > 1):
       loops through the season calendar and accumulates championship points.
    7. For Monte Carlo simulation (n_monte_carlo > 0): repeats the season
       simulation n_monte_carlo times with randomised DNF events to produce
       a distribution of outcomes.

    Configurable Parameters (20 total)
    ------------------------------------
    track : str or None
        Circuit name (partial, case-insensitive). If None → full season.
    grid_position_override : int or None
        Override the starting grid position for all races.
    teammate : str or None
        Named teammate (for display purposes; skill level set separately).
    weather : {'dry', 'wet'}
        Race weather condition.
    num_pit_stops : int or None
        Number of pit stops (relative to 1-stop baseline).
    reliability_factor : float
        Probability of finishing each race (0–1).  0.9 = 10% DNF chance.
    season_round : int or None
        Specific round number override (single race only).
    driver_age_override : int or None
        Override driver age at race (years).
    car_development_rate : float
        Team car performance improvement per round. Positive = improvement.
    tire_strategy : {'soft', 'medium', 'hard'} or None
        Tire compound strategy.
    starting_championship_points : int
        Points the driver enters the season with.
    driver_confidence_factor : float
        0.0 (crisis of confidence) → 1.0 (peak form). Default 0.5.
    regulation_era : {'v10', 'v8', 'v6_hybrid', 'ground_effect'} or None
        Force a regulation-era context for feature adjustments.
    num_races_to_simulate : int or None
        Limit the season simulation to this many races.
    teammate_skill_level : {'elite', 'midfield', 'rookie'}
        Calibre of teammate; affects teammate_delta feature.
    race_incidents : bool
        Include random on-track incident DNF probability.
    home_race : bool
        Driver is racing at their home Grand Prix.
    model : sklearn-compatible classifier or None
        Pre-trained race-winner model. Loaded from disk if None.
    n_monte_carlo : int
        Number of Monte Carlo iterations for season simulation. 0 to disable.

    Parameters
    ----------
    featured_df : pd.DataFrame
        Full feature-engineered DataFrame (output of engineer_features).
    driver : str
        Driver name or partial name (case-insensitive).
    team : str
        Team/constructor name or partial name (case-insensitive).
    year : int
        Target season year.
    **All 20 configurable parameters listed above.**

    Returns
    -------
    dict
        Keys
        ----
        driver : str
            Matched driver name string.
        team : str
            Matched team name string.
        year : int
            Target year.
        track : str
            Circuit name or 'Full Season'.
        predicted_win_probability : float or None
            Win probability for single-race scenarios; None for full season.
        predicted_position : int or None
            Predicted finishing position (single race); None for full season.
        season_points_estimate : int or None
            Estimated championship points for full-season scenarios.
        season_wins_estimate : int or None
            Estimated race wins for full-season scenarios.
        season_podiums_estimate : int or None
            Estimated podiums for full-season scenarios.
        championship_position_estimate : int or None
            Estimated constructor championship position (crude heuristic).
        monte_carlo_results : list of int
            n_monte_carlo season-points totals from Monte Carlo iterations.
        confidence_interval : tuple of (float, float)
            (5th percentile, 95th percentile) of Monte Carlo points totals.
        parameter_summary : dict
            All 20 parameter values used in this scenario.
    """
    # ── 0. Bundle all parameters for downstream functions ──────────────────
    params = {
        "track":                       track,
        "grid_position_override":      grid_position_override,
        "teammate":                    teammate,
        "weather":                     weather,
        "num_pit_stops":               num_pit_stops,
        "reliability_factor":          reliability_factor,
        "season_round":                season_round,
        "driver_age_override":         driver_age_override,
        "car_development_rate":        car_development_rate,
        "tire_strategy":               tire_strategy,
        "starting_championship_points": starting_championship_points,
        "driver_confidence_factor":    driver_confidence_factor,
        "regulation_era":              regulation_era,
        "num_races_to_simulate":       num_races_to_simulate,
        "teammate_skill_level":        teammate_skill_level,
        "race_incidents":              race_incidents,
        "home_race":                   home_race,
        "n_monte_carlo":               n_monte_carlo,
        "year":                        year,
    }

    # ── 1. Optionally load model from disk ────────────────────────────────
    if model is None:
        try:
            try:
                from src.models import load_model, _DEFAULT_MODEL_DIR
            except ImportError:
                from models import load_model, _DEFAULT_MODEL_DIR  # type: ignore

            model_path = str(_DEFAULT_MODEL_DIR / "race_winner_model.pkl")
            model = load_model(model_path)
            if model is None:
                print(
                    "[hypothetical_engine] No saved model found at "
                    f"'{model_path}'; using heuristic predictions."
                )
        except Exception as exc:
            print(
                f"[hypothetical_engine] Could not load model ({exc}); "
                "using heuristic predictions."
            )
            model = None

    # ── 2. Extract profiles ───────────────────────────────────────────────
    print(f"[hypothetical_engine] Extracting driver profile: '{driver}' ({year})")
    driver_profile = get_driver_skill_profile(featured_df, driver, year)

    print(f"[hypothetical_engine] Extracting team profile: '{team}' ({year})")
    team_profile = get_team_car_profile(featured_df, team, year)

    # ── 3. Determine resolved names for reporting ─────────────────────────
    driver_col = _resolve_driver_col(featured_df)
    team_col = _resolve_team_col(featured_df)

    driver_rows = _find_driver_rows(featured_df, driver, year)
    team_rows = _find_team_rows(featured_df, team, year)

    resolved_driver = (
        str(driver_rows[driver_col].iloc[0])
        if (driver_col and not driver_rows.empty)
        else driver
    )
    resolved_team = (
        str(team_rows[team_col].iloc[0])
        if (team_col and not team_rows.empty)
        else team
    )

    print(
        f"[hypothetical_engine] Scenario: {resolved_driver} → {resolved_team} "
        f"({year}){' @ ' + track if track else ', full season'}"
    )

    # ── 4. Single-race vs full-season branching ───────────────────────────
    is_full_season = track is None or (
        num_races_to_simulate is not None and num_races_to_simulate > 1
    )

    # Initialise result containers
    predicted_win_probability = None
    predicted_position = None
    season_points_estimate = None
    season_wins_estimate = None
    season_podiums_estimate = None
    championship_position_estimate = None
    mc_results: list[int] = []
    confidence_interval = (None, None)

    if not is_full_season:
        # ── Single-race scenario ─────────────────────────────────────────
        base_row = {**driver_profile, **team_profile}
        base_row["season_round"] = float(season_round) if season_round else 1.0
        base_row["field_size"] = 20.0

        # Pull circuit-specific features if track data is available
        if track:
            track_mask = (
                _icontains(featured_df.get("name", pd.Series(dtype=str)), track)
            ) & (featured_df["year"] == year)
            # Widen year window if exact year has no data
            track_rows = featured_df[track_mask]
            if track_rows.empty:
                track_mask = _icontains(
                    featured_df.get("name", pd.Series(dtype=str)), track
                )
                track_rows = featured_df[track_mask]

            if not track_rows.empty:
                for circ_feat in [
                    "circuit_avg_grid_delta", "circuit_avg_dnf_rate",
                    "circuit_country_encoded", "altitude",
                    "driver_prev_wins_at_circuit", "driver_avg_finish_at_circuit",
                    "driver_races_at_circuit", "team_wins_at_circuit",
                ]:
                    if circ_feat in track_rows.columns:
                        val = track_rows[circ_feat].dropna()
                        if not val.empty:
                            base_row[circ_feat] = float(val.median())

        feature_row = _apply_parameter_overrides(base_row, params)

        # Run a single deterministic race prediction (no DNF roll for the
        # single-race point estimate; use the win probability directly)
        if n_monte_carlo > 0:
            # Monte Carlo over a single race
            mc_positions = []
            mc_wins = 0
            for _ in range(n_monte_carlo):
                result = _simulate_single_race(
                    feature_row, model,
                    reliability_factor=reliability_factor,
                    race_incidents=race_incidents,
                )
                mc_positions.append(result["predicted_position"])
                if result["predicted_position"] == 1 and not result["dnf_occurred"]:
                    mc_wins += 1

            predicted_win_probability = float(mc_wins / n_monte_carlo)
            predicted_position = int(np.median(mc_positions))
            mc_results = mc_positions
            confidence_interval = (
                float(np.percentile(mc_positions, 5)),
                float(np.percentile(mc_positions, 95)),
            )
        else:
            # Single deterministic prediction
            try:
                from src.models import FEATURE_COLS
            except ImportError:
                from models import FEATURE_COLS  # type: ignore

            row_data = {col: feature_row.get(col, np.nan) for col in FEATURE_COLS}
            X_arr = pd.DataFrame([row_data]).fillna(0.0).to_numpy(dtype=float)

            if model is not None:
                try:
                    proba = model.predict_proba(X_arr)
                    predicted_win_probability = float(
                        proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[hypothetical_engine] predict_proba failed: {exc}"
                    )
                    predicted_win_probability = _heuristic_win_prob(feature_row)
            else:
                predicted_win_probability = _heuristic_win_prob(feature_row)

            # Map win probability to a finishing position estimate
            if predicted_win_probability > 0.0:
                predicted_position = max(
                    1, min(20, int(round(1.0 / (predicted_win_probability + 1e-6) * 0.25)))
                )
            else:
                predicted_position = 15

    else:
        # ── Full-season scenario ─────────────────────────────────────────
        if n_monte_carlo > 0:
            print(
                f"[hypothetical_engine] Running {n_monte_carlo} Monte Carlo "
                f"season simulations …"
            )
            for i in range(n_monte_carlo):
                # Re-seed per iteration for independent random draws
                season_result = _simulate_full_season(
                    driver_profile, team_profile, year,
                    featured_df, params, model,
                )
                mc_results.append(season_result["season_points"])

            season_points_estimate = int(np.median(mc_results))
            season_wins_estimate = None   # aggregated over MC iterations below
            season_podiums_estimate = None
            confidence_interval = (
                float(np.percentile(mc_results, 5)),
                float(np.percentile(mc_results, 95)),
            )

            # Run one final deterministic season to get wins/podiums
            final_season = _simulate_full_season(
                driver_profile, team_profile, year,
                featured_df, params, model,
            )
            season_wins_estimate = final_season["wins"]
            season_podiums_estimate = final_season["podiums"]

        else:
            # Single deterministic season simulation
            season_result = _simulate_full_season(
                driver_profile, team_profile, year,
                featured_df, params, model,
            )
            season_points_estimate = season_result["season_points"]
            season_wins_estimate = season_result["wins"]
            season_podiums_estimate = season_result["podiums"]

        # Crude championship position estimate based on expected points
        # (heuristic: assume front-running teams score ~400–600 points)
        pts = season_points_estimate or 0
        if pts >= 400:
            championship_position_estimate = 1
        elif pts >= 300:
            championship_position_estimate = 2
        elif pts >= 200:
            championship_position_estimate = 3
        elif pts >= 130:
            championship_position_estimate = 4
        elif pts >= 80:
            championship_position_estimate = 5
        elif pts >= 40:
            championship_position_estimate = 7
        elif pts >= 15:
            championship_position_estimate = 9
        else:
            championship_position_estimate = 12

    # ── 5. Compose result dict ────────────────────────────────────────────
    return {
        "driver":                        resolved_driver,
        "team":                          resolved_team,
        "year":                          year,
        "track":                         track if track else "Full Season",
        "predicted_win_probability":     predicted_win_probability,
        "predicted_position":            predicted_position,
        "season_points_estimate":        season_points_estimate,
        "season_wins_estimate":          season_wins_estimate,
        "season_podiums_estimate":       season_podiums_estimate,
        "championship_position_estimate": championship_position_estimate,
        "monte_carlo_results":           mc_results,
        "confidence_interval":           confidence_interval,
        "parameter_summary":             params,
    }
