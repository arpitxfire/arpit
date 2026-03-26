"""
F1 GOAT Analysis Module
=======================
Implements the 5-lens Greatest of All Time (GOAT) analysis framework for
Formula 1 drivers.

Lenses:
    1. Raw Statistics (Era-Normalized)
    2. Teammate Comparison
    3. ML "Same Car" Test
    4. Adaptability Score
    5. Dominance Index

Usage
-----
>>> from src.data_loader import build_master
>>> from src.feature_engineering import engineer_features
>>> master   = build_master("data/raw/")
>>> featured = engineer_features(master)
>>> goat_table = run_goat_analysis(master, featured)
"""

import warnings
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GOAT_CANDIDATES: list = [
    "Lewis Hamilton", "Michael Schumacher", "Max Verstappen",
    "Sebastian Vettel", "Alain Prost", "Ayrton Senna",
    "Juan Manuel Fangio", "Jim Clark", "Jackie Stewart",
    "Niki Lauda", "Fernando Alonso",
]

# 2010+ F1 points system used to recalculate all historical race points
MODERN_POINTS: dict = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8,  7: 6,  8: 4,  9: 2,  10: 1,
}

# Regulation eras: (start_year, end_year) inclusive
REGULATION_ERAS: dict = {
    "pre_turbo":    (1950, 1988),
    "post_turbo":   (1989, 1994),
    "v10":          (1995, 2005),
    "v8":           (2006, 2013),
    "v6_hybrid":    (2014, 2021),
    "ground_effect":(2022, 2100),
}

# Reference season length used for per-season normalization
REFERENCE_SEASON_LENGTH: int = 20

# Weights for the 5 lenses (must sum to 1.0)
GOAT_WEIGHTS: dict = {
    "normalized_stats_score": 0.20,
    "teammate_score":         0.30,
    "same_car_score":         0.25,
    "adaptability_score":     0.15,
    "dominance_score":        0.10,
}

# Fraction to compress finish positions toward midfield in heuristic simulation
MEDIAN_CAR_COMPRESSION: float = 0.45


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_0_100(series: pd.Series) -> pd.Series:
    """Min-max normalize *series* to [0, 100]. Ties return 50. NaN preserved."""
    min_val = series.min(skipna=True)
    max_val = series.max(skipna=True)
    if pd.isna(min_val) or max_val == min_val:
        return series.where(series.isna(), 50.0)
    return (series - min_val) / (max_val - min_val) * 100.0


def _get_modern_points(position) -> float:
    """Points for a finish position under the 2010+ scoring system."""
    try:
        return float(MODERN_POINTS.get(int(position), 0.0))
    except (ValueError, TypeError):
        return 0.0


def _get_regulation_era(year: int) -> str:
    """Return regulation-era label for a given season year."""
    for era, (start, end) in REGULATION_ERAS.items():
        if start <= year <= end:
            return era
    return "unknown"


def _validate_candidates(df: pd.DataFrame, candidates: list) -> list:
    """Return only candidates present in df["driver_name"], warning about others."""
    if "driver_name" not in df.columns:
        raise KeyError("DataFrame must contain a 'driver_name' column.")
    available = set(df["driver_name"].unique())
    valid = []
    for name in candidates:
        if name in available:
            valid.append(name)
        else:
            warnings.warn(
                f"GOAT candidate '{name}' not found in dataset — skipping.",
                UserWarning, stacklevel=2,
            )
    return valid


def _season_champions(master_df: pd.DataFrame) -> pd.DataFrame:
    """Derive season champions by totalling race points per driver per year.

    Returns DataFrame with columns: year, champion_driver_name, champion_points.
    """
    season_pts = (
        master_df.groupby(["year", "driver_name"])["points"]
        .sum().reset_index()
    )
    idx = season_pts.groupby("year")["points"].idxmax()
    return season_pts.loc[idx].rename(
        columns={"driver_name": "champion_driver_name", "points": "champion_points"}
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LENS 1 — RAW STATISTICS (ERA-NORMALIZED)
# ─────────────────────────────────────────────────────────────────────────────

def compute_raw_stats(
    master_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    candidates: list = None,
) -> pd.DataFrame:
    """
    Compute era-normalized raw statistics for GOAT candidates.

    Normalizes by field size and season length to compare across eras.

    Parameters
    ----------
    master_df : pd.DataFrame
        Cleaned master DataFrame. Required columns: driver_name, year, round,
        positionOrder, points, grid, grid_qual_position.
    featured_df : pd.DataFrame
        Feature-engineered DataFrame (supplemental).
    candidates : list, optional
        Driver names to analyse. Defaults to DEFAULT_GOAT_CANDIDATES.

    Returns
    -------
    pd.DataFrame
        One row per candidate with columns:
        - driver_name, total_wins, total_podiums, total_poles
        - total_championships, total_races
        - win_rate_pct: wins/races * 100
        - podium_rate_pct: podiums/races * 100
        - normalized_wins: wins per 20-race season equivalent
        - normalized_podiums: podiums per 20-race season equivalent
        - normalized_championships: championships / seasons_raced
        - points_in_modern_system: recalculate all historical points using 2010+ system
        - career_span_years
        - peak_win_rate: highest win rate in any 3-season window
        - normalized_stats_score: composite 0-100 score
    """
    if candidates is None:
        candidates = DEFAULT_GOAT_CANDIDATES
    candidates = _validate_candidates(master_df, candidates)
    if not candidates:
        return pd.DataFrame()

    df = master_df[master_df["driver_name"].isin(candidates)].copy()

    # Identify season champions from the full dataset for title counting
    champions  = _season_champions(master_df)
    champion_set = set(zip(champions["year"], champions["champion_driver_name"]))

    # Pole flag: prefer grid_qual_position (post-2003), fall back to grid == 1
    if "grid_qual_position" in df.columns:
        df["_is_pole"] = (
            (df["grid_qual_position"] == 1)
            | (df["grid_qual_position"].isna() & (df["grid"] == 1))
        ).fillna(False)
    else:
        df["_is_pole"] = (df["grid"] == 1).fillna(False)

    df["_is_win"]     = df["positionOrder"] == 1
    df["_is_podium"]  = df["positionOrder"] <= 3
    df["_modern_pts"] = df["positionOrder"].apply(_get_modern_points)

    records = []
    for driver in candidates:
        ddf = df[df["driver_name"] == driver]
        if ddf.empty:
            continue

        years_raced   = sorted(ddf["year"].unique())
        seasons_raced = len(years_raced)
        total_races   = len(ddf)
        total_wins    = int(ddf["_is_win"].sum())
        total_podiums = int(ddf["_is_podium"].sum())
        total_poles   = int(ddf["_is_pole"].sum())
        career_span   = (max(years_raced) - min(years_raced) + 1) if seasons_raced > 1 else 1

        # Championship titles: seasons where driver accumulated the most points
        total_championships = sum(1 for yr in years_raced if (yr, driver) in champion_set)

        # Rate metrics
        win_rate_pct    = (total_wins    / total_races * 100) if total_races else 0.0
        podium_rate_pct = (total_podiums / total_races * 100) if total_races else 0.0

        # Normalize wins/podiums to a reference 20-race season
        avg_races_per_season = total_races / seasons_raced if seasons_raced else total_races
        season_scale = REFERENCE_SEASON_LENGTH / avg_races_per_season if avg_races_per_season else 1.0
        normalized_wins          = (total_wins    / seasons_raced) * season_scale if seasons_raced else 0.0
        normalized_podiums       = (total_podiums / seasons_raced) * season_scale if seasons_raced else 0.0
        normalized_championships = total_championships / seasons_raced if seasons_raced else 0.0

        points_in_modern_system = float(ddf["_modern_pts"].sum())

        # Peak win rate: best rolling 3-season average
        season_wr = (
            ddf.groupby("year")["_is_win"].sum()
            / ddf.groupby("year")["_is_win"].count()
        ).fillna(0.0)
        if len(season_wr) >= 3:
            peak_win_rate = float(season_wr.rolling(3).mean().max()) * 100.0
        elif len(season_wr) > 0:
            peak_win_rate = float(season_wr.mean()) * 100.0
        else:
            peak_win_rate = 0.0

        records.append({
            "driver_name":              driver,
            "total_wins":               total_wins,
            "total_podiums":            total_podiums,
            "total_poles":              total_poles,
            "total_championships":      total_championships,
            "total_races":              total_races,
            "win_rate_pct":             round(win_rate_pct, 2),
            "podium_rate_pct":          round(podium_rate_pct, 2),
            "normalized_wins":          round(normalized_wins, 3),
            "normalized_podiums":       round(normalized_podiums, 3),
            "normalized_championships": round(normalized_championships, 4),
            "points_in_modern_system":  round(points_in_modern_system, 1),
            "career_span_years":        career_span,
            "peak_win_rate":            round(peak_win_rate, 2),
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # Composite score from six equally-weighted sub-metrics
    score_df = pd.DataFrame({
        "win_rate":    _normalize_0_100(result["win_rate_pct"]),
        "podium_rate": _normalize_0_100(result["podium_rate_pct"]),
        "norm_champs": _normalize_0_100(result["normalized_championships"]),
        "mod_points":  _normalize_0_100(result["points_in_modern_system"]),
        "peak_wins":   _normalize_0_100(result["peak_win_rate"]),
        "norm_wins":   _normalize_0_100(result["normalized_wins"]),
    }, index=result.index)
    result["normalized_stats_score"] = score_df.mean(axis=1).round(2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  LENS 2 — TEAMMATE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compute_teammate_comparison(
    master_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    candidates: list = None,
) -> pd.DataFrame:
    """
    Compare each GOAT candidate against their teammates in the same car.
    This removes car advantage entirely.

    For every season, compare driver to teammate:
    - qual_h2h: qualifying H2H wins (driver beat teammate in qualifying)
    - race_h2h: race finish H2H wins
    - points_h2h: season points vs teammate

    Qualifying H2H is only reliable from 2003 onward; pre-2003 careers
    will have NaN for qualifying metrics.

    Parameters
    ----------
    master_df : pd.DataFrame
        Cleaned master DataFrame. Required columns: driver_name, raceId,
        year, positionOrder, grid, constructorId, is_dnf.
    featured_df : pd.DataFrame
        Feature-engineered DataFrame. Uses pre-computed teammate_delta and
        teammate_qual_delta when present; otherwise recomputes from master_df.
    candidates : list, optional
        Driver names to analyse. Defaults to DEFAULT_GOAT_CANDIDATES.

    Returns
    -------
    pd.DataFrame
        One row per candidate with columns:
        - driver_name
        - career_qual_h2h_pct: % of qualifying sessions where beat teammate
        - career_race_h2h_pct: % of races where finished ahead of teammate
        - career_avg_teammate_qual_delta: average qualifying position difference
        - career_avg_teammate_race_delta: average race finish position difference
        - teammate_score: normalized 0-100 score
        - num_teammates: number of distinct teammates compared
    """
    if candidates is None:
        candidates = DEFAULT_GOAT_CANDIDATES
    candidates = _validate_candidates(master_df, candidates)
    if not candidates:
        return pd.DataFrame()

    # Use pre-computed deltas from featured_df when available; recompute otherwise
    has_tm_delta = "teammate_delta"      in featured_df.columns
    has_tm_qual  = "teammate_qual_delta" in featured_df.columns

    src_df = (
        featured_df[featured_df["driver_name"].isin(candidates)].copy()
        if has_tm_delta
        else _compute_teammate_deltas(master_df, candidates)
    )

    records = []
    for driver in candidates:
        ddf = src_df[src_df["driver_name"] == driver].copy()
        if ddf.empty:
            records.append(_empty_teammate_record(driver))
            continue

        # Race H2H: teammate_delta < 0 means driver finished ahead of teammate.
        # Exclude DNF rows so attrition victories are not credited.
        race_rows = ddf[ddf["teammate_delta"].notna()]
        if "is_dnf" in race_rows.columns:
            race_rows = race_rows[~race_rows["is_dnf"].fillna(False)]
        n_race            = len(race_rows)
        race_h2h_wins     = int((race_rows["teammate_delta"] < 0).sum())
        career_race_h2h   = (race_h2h_wins / n_race * 100) if n_race else np.nan
        avg_race_delta    = race_rows["teammate_delta"].mean() if n_race else np.nan

        # Qualifying H2H (2003+ only — single-lap qualifying era)
        if has_tm_qual and "teammate_qual_delta" in ddf.columns:
            qual_rows = ddf[ddf["teammate_qual_delta"].notna() & (ddf["year"] >= 2003)]
        else:
            qual_rows = pd.DataFrame()
        n_qual         = len(qual_rows)
        qual_h2h_wins  = int((qual_rows["teammate_qual_delta"] < 0).sum()) if n_qual else 0
        career_qual_h2h = (qual_h2h_wins / n_qual * 100) if n_qual else np.nan
        avg_qual_delta  = qual_rows["teammate_qual_delta"].mean() if n_qual else np.nan

        # Count distinct teammates faced over career
        num_tm = _count_distinct_teammates(master_df, driver)

        records.append({
            "driver_name":                   driver,
            "career_qual_h2h_pct":           round(career_qual_h2h, 2) if pd.notna(career_qual_h2h) else np.nan,
            "career_race_h2h_pct":           round(career_race_h2h, 2) if pd.notna(career_race_h2h) else np.nan,
            "career_avg_teammate_qual_delta": round(avg_qual_delta,  3) if pd.notna(avg_qual_delta)  else np.nan,
            "career_avg_teammate_race_delta": round(avg_race_delta,  3) if pd.notna(avg_race_delta)  else np.nan,
            "num_teammates":                  num_tm,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # teammate_score: higher H2H pct + more negative deltas = better
    # Pre-2003 drivers receive neutral 50.0 for qual sub-scores (no penalty)
    race_pct   = _normalize_0_100(result["career_race_h2h_pct"].fillna(50.0))
    qual_pct   = _normalize_0_100(result["career_qual_h2h_pct"].fillna(50.0))
    race_delta = _normalize_0_100(-result["career_avg_teammate_race_delta"].fillna(0.0))
    qual_delta = _normalize_0_100(-result["career_avg_teammate_qual_delta"].fillna(0.0))

    result["teammate_score"] = (
        0.40 * race_pct + 0.30 * race_delta + 0.20 * qual_pct + 0.10 * qual_delta
    ).round(2)
    return result


def _empty_teammate_record(driver: str) -> dict:
    """All-NaN teammate record for a driver with no usable data."""
    return {
        "driver_name":                   driver,
        "career_qual_h2h_pct":           np.nan,
        "career_race_h2h_pct":           np.nan,
        "career_avg_teammate_qual_delta": np.nan,
        "career_avg_teammate_race_delta": np.nan,
        "num_teammates":                  0,
    }


def _compute_teammate_deltas(master_df: pd.DataFrame, candidates: list) -> pd.DataFrame:
    """Self-join master_df on (raceId, constructorId) to compute teammate deltas.

    Fallback used when featured_df lacks pre-computed teammate columns.
    When 3+ cars are fielded, keeps only the best-finishing teammate.
    """
    df  = master_df.copy()
    tm  = df[["raceId","constructorId","driver_name","positionOrder","grid","year"]].rename(
        columns={"driver_name":"tm_driver","positionOrder":"tm_positionOrder","grid":"tm_grid"}
    )
    merged = df.merge(tm, on=["raceId","constructorId"], how="left")
    merged = merged[merged["driver_name"] != merged["tm_driver"]]
    # When multiple teammates, keep the best-finishing one
    merged = (merged.sort_values("tm_positionOrder")
              .groupby(["raceId","driver_name"], as_index=False).first())
    merged["teammate_delta"]      = merged["positionOrder"] - merged["tm_positionOrder"]
    merged["teammate_qual_delta"] = merged["grid"]          - merged["tm_grid"]
    return merged[merged["driver_name"].isin(candidates)].copy()


def _count_distinct_teammates(master_df: pd.DataFrame, driver: str) -> int:
    """Count distinct teammates a driver faced (same race, same constructor)."""
    driver_races = master_df.loc[
        master_df["driver_name"] == driver, ["raceId","constructorId"]
    ]
    teammates = master_df.merge(driver_races, on=["raceId","constructorId"])
    return int(teammates.loc[teammates["driver_name"] != driver, "driver_name"].nunique())


# ─────────────────────────────────────────────────────────────────────────────
#  LENS 3 — ML "SAME CAR" TEST
# ─────────────────────────────────────────────────────────────────────────────

def compute_same_car_ml_score(
    featured_df: pd.DataFrame,
    candidates: list = None,
    model=None,
    n_races: int = 22,
) -> pd.DataFrame:
    """
    Use hypothetical engine to place every GOAT candidate in a median-performance car.
    Simulate a full 22-race season for each.

    With a model: builds feature vectors combining the driver's skill profile
    with median team features and queries the model per race.
    Without a model: uses a heuristic that draws from the driver's historical
    finish-position distribution, compressed toward midfield (MEDIAN_CAR_COMPRESSION).

    Parameters
    ----------
    featured_df : pd.DataFrame
        Feature-engineered DataFrame. Must contain driver_name, year, positionOrder.
    candidates : list, optional
        Driver names to analyse. Defaults to DEFAULT_GOAT_CANDIDATES.
    model : trained ML model, optional
        Estimator with predict_proba or predict. If None, heuristic is used.
    n_races : int
        Races to simulate per season (default 22).

    Returns
    -------
    pd.DataFrame
        One row per candidate with columns:
        - driver_name
        - simulated_season_points
        - simulated_wins
        - simulated_podiums
        - same_car_score: normalized 0-100 score
    """
    if candidates is None:
        candidates = DEFAULT_GOAT_CANDIDATES
    candidates = _validate_candidates(featured_df, candidates)
    if not candidates:
        return pd.DataFrame()

    # Median car profile: median of all team_* numeric columns across the grid
    team_cols = [c for c in featured_df.columns
                 if c.startswith("team_") and pd.api.types.is_numeric_dtype(featured_df[c])]
    median_car = featured_df[team_cols].median() if team_cols else pd.Series(dtype=float)

    driver_skill_cols = [c for c in [
        "career_wins_so_far","career_podiums_so_far","career_races_so_far",
        "win_rate_last_5","podium_rate_last_5","avg_finish_last_5",
        "avg_finish_last_10","avg_grid_last_5","grid_delta_career_avg",
        "dnf_rate_last_10","seasons_experience","age_at_race",
    ] if c in featured_df.columns]

    records = []
    for driver in candidates:
        ddf = featured_df[featured_df["driver_name"] == driver].copy()
        if ddf.empty:
            records.append({"driver_name": driver, "simulated_season_points": 0.0,
                            "simulated_wins": 0, "simulated_podiums": 0})
            continue

        # Use most recent 3 seasons to capture peak skill
        recent_years = sorted(ddf["year"].unique())[-3:]
        recent_ddf   = ddf[ddf["year"].isin(recent_years)]
        skill_profile = recent_ddf[driver_skill_cols].mean() if driver_skill_cols else pd.Series(dtype=float)

        if model is not None:
            pts, wins, pods = _simulate_season_with_model(
                model, skill_profile, median_car, featured_df, n_races
            )
        else:
            pts, wins, pods = _simulate_season_heuristic(recent_ddf, n_races)

        records.append({
            "driver_name":             driver,
            "simulated_season_points": round(float(pts), 1),
            "simulated_wins":          int(wins),
            "simulated_podiums":       int(pods),
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["same_car_score"] = _normalize_0_100(result["simulated_season_points"]).round(2)
    return result


def _simulate_season_with_model(model, skill_profile, median_car, featured_df, n_races):
    """Build feature matrix, query model, return (points, wins, podiums)."""
    try:
        from src.feature_engineering import get_feature_columns  # type: ignore
        feature_cols = get_feature_columns()
    except Exception:
        feature_cols = list(featured_df.select_dtypes(include=[np.number]).columns)

    sample = featured_df.sample(n=n_races, replace=True, random_state=42).copy()
    for col in skill_profile.index:
        if col in sample.columns:
            sample[col] = skill_profile[col]
    for col in median_car.index:
        if col in sample.columns:
            sample[col] = median_car[col]

    avail = [c for c in feature_cols if c in sample.columns]
    X = sample[avail].fillna(sample[avail].median())

    pts, wins, pods = 0.0, 0, 0
    try:
        rng = np.random.default_rng(seed=42)
        if hasattr(model, "predict_proba"):
            for p in model.predict_proba(X)[:, 1]:
                finish = _sample_pos_from_win_prob(float(p), rng)
                pts += _get_modern_points(finish)
                wins  += int(finish == 1)
                pods  += int(finish <= 3)
        else:
            for pos in model.predict(X):
                pos = int(np.clip(round(float(pos)), 1, 20))
                pts += _get_modern_points(pos)
                wins  += int(pos == 1)
                pods  += int(pos <= 3)
    except Exception as exc:
        warnings.warn(f"Model prediction failed ({exc}); using heuristic.", UserWarning, stacklevel=2)
        return _simulate_season_heuristic(featured_df.tail(n_races * 3), n_races)

    return pts, wins, pods


def _sample_pos_from_win_prob(win_prob: float, rng: np.random.Generator, field: int = 20) -> int:
    """Convert win probability to a sampled finish position via exponential scaling."""
    expected_rank = field - win_prob * (field - 1)
    raw = rng.exponential(scale=max(expected_rank, 1.0))
    return int(np.clip(round(raw), 1, field))


def _simulate_season_heuristic(recent_ddf: pd.DataFrame, n_races: int) -> tuple:
    """Heuristic same-car simulation.

    Draws finish positions from the driver's recent history and applies
    MEDIAN_CAR_COMPRESSION to pull them toward midfield (P11),
    simulating the effect of being placed in a median-performance car.
    """
    if recent_ddf.empty or "positionOrder" not in recent_ddf.columns:
        return _get_modern_points(8) * n_races, 0, 0

    positions = recent_ddf["positionOrder"].dropna().values.astype(float)
    if len(positions) == 0:
        return 0.0, 0, 0

    rng     = np.random.default_rng(seed=42)
    sampled = rng.choice(positions, size=n_races, replace=(len(positions) < n_races))

    # Pull toward field median (P11) to remove car-performance advantage
    adjusted = sampled * (1 - MEDIAN_CAR_COMPRESSION) + 11.0 * MEDIAN_CAR_COMPRESSION
    adjusted = np.clip(np.round(adjusted), 1, 20).astype(int)

    pts  = float(sum(_get_modern_points(p) for p in adjusted))
    wins = int((adjusted == 1).sum())
    pods = int((adjusted <= 3).sum())
    return pts, wins, pods


# ─────────────────────────────────────────────────────────────────────────────
#  LENS 4 — ADAPTABILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptability_score(
    master_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    candidates: list = None,
) -> pd.DataFrame:
    """
    Measure driver adaptability across different contexts.

    Evaluates performance across multiple constructors, circuits, regulation
    eras, and team transitions to measure how well drivers adapted to
    different machinery and rule sets.

    Parameters
    ----------
    master_df : pd.DataFrame
        Cleaned master DataFrame. Required columns: driver_name, year, round,
        positionOrder, constructorId/constructorName, circuitId/name.
    featured_df : pd.DataFrame
        Feature-engineered DataFrame (not directly queried in this lens).
    candidates : list, optional
        Driver names to analyse. Defaults to DEFAULT_GOAT_CANDIDATES.

    Returns
    -------
    pd.DataFrame
        One row per candidate with columns:
        - driver_name
        - num_teams_won_with: number of different teams driver won with
        - num_circuits_won_at: number of distinct circuits won at
        - races_to_first_win: races before first career win
        - regulation_era_wins: wins spanning different regulation eras
        - team_switch_adaptation: avg performance in first season at new team vs previous
        - adaptability_score: normalized 0-100 score
    """
    if candidates is None:
        candidates = DEFAULT_GOAT_CANDIDATES
    candidates = _validate_candidates(master_df, candidates)
    if not candidates:
        return pd.DataFrame()

    df = master_df[master_df["driver_name"].isin(candidates)].copy()
    df["_is_win"]  = df["positionOrder"] == 1
    df["_reg_era"] = df["year"].apply(_get_regulation_era)
    df = df.sort_values(["driver_name","year","round"]).reset_index(drop=True)

    team_col    = "constructorName" if "constructorName" in df.columns else "constructorId"
    circuit_col = "circuitId"       if "circuitId"       in df.columns else "name"

    records = []
    for driver in candidates:
        ddf = df[df["driver_name"] == driver]
        if ddf.empty:
            records.append(_empty_adaptability_record(driver))
            continue

        wins_df = ddf[ddf["_is_win"]]

        num_teams_won_with  = int(wins_df[team_col].nunique())
        num_circuits_won_at = int(wins_df[circuit_col].nunique())

        first_win_idx      = int(ddf["_is_win"].values.argmax()) if ddf["_is_win"].any() else -1
        races_to_first_win = first_win_idx if first_win_idx >= 0 else len(ddf)

        regulation_era_wins   = int(wins_df["_reg_era"].nunique())
        team_switch_delta     = _compute_team_switch_adaptation(ddf, team_col)

        records.append({
            "driver_name":            driver,
            "num_teams_won_with":     num_teams_won_with,
            "num_circuits_won_at":    num_circuits_won_at,
            "races_to_first_win":     races_to_first_win,
            "regulation_era_wins":    regulation_era_wins,
            "team_switch_adaptation": round(team_switch_delta, 3),
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    scores = pd.DataFrame({
        "teams_won":    _normalize_0_100(result["num_teams_won_with"].astype(float)),
        "circuits_won": _normalize_0_100(result["num_circuits_won_at"].astype(float)),
        # Fewer races to first win is better → invert
        "fast_starter": _normalize_0_100(-result["races_to_first_win"].astype(float)),
        "era_wins":     _normalize_0_100(result["regulation_era_wins"].astype(float)),
        # Negative delta (improved at new team) → higher score after inversion
        "switch_adapt": _normalize_0_100(-result["team_switch_adaptation"]),
    }, index=result.index)

    result["adaptability_score"] = (
        0.25 * scores["teams_won"]    + 0.25 * scores["circuits_won"]
        + 0.15 * scores["fast_starter"] + 0.20 * scores["era_wins"]
        + 0.15 * scores["switch_adapt"]
    ).round(2)
    return result


def _empty_adaptability_record(driver: str) -> dict:
    """Zeroed adaptability record for a driver with no data."""
    return {
        "driver_name": driver, "num_teams_won_with": 0,
        "num_circuits_won_at": 0, "races_to_first_win": 999,
        "regulation_era_wins": 0, "team_switch_adaptation": 0.0,
    }


def _compute_team_switch_adaptation(ddf: pd.DataFrame, team_col: str) -> float:
    """Mean performance change (finish position delta) across all team switches.

    For each switch from Team A → Team B:
      delta = first_season_avg_pos_at_B - last_season_avg_pos_at_A
    Negative = driver improved. Returns 0.0 if no switches occurred.
    """
    if team_col not in ddf.columns:
        return 0.0
    timeline = (
        ddf.groupby(["year", team_col])["positionOrder"].mean()
        .reset_index().sort_values("year").reset_index(drop=True)
    )
    if len(timeline) < 2:
        return 0.0

    deltas, prev_team, prev_avg = [], None, None
    for _, row in timeline.iterrows():
        if prev_team is not None and row[team_col] != prev_team:
            deltas.append(float(row["positionOrder"] - prev_avg))
        prev_team, prev_avg = row[team_col], row["positionOrder"]

    return float(np.mean(deltas)) if deltas else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  LENS 5 — DOMINANCE INDEX
# ─────────────────────────────────────────────────────────────────────────────

def compute_dominance_index(
    master_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    candidates: list = None,
) -> pd.DataFrame:
    """
    Measure peak dominance.

    Captures not just whether a driver won titles but how decisively — via
    crushing points gaps, historic win streaks, and sustained winning seasons.

    Parameters
    ----------
    master_df : pd.DataFrame
        Cleaned master DataFrame. Required columns: driver_name, year, round,
        positionOrder, points.
    featured_df : pd.DataFrame
        Feature-engineered DataFrame (not directly queried in this lens).
    candidates : list, optional
        Driver names to analyse. Defaults to DEFAULT_GOAT_CANDIDATES.

    Returns
    -------
    pd.DataFrame
        One row per candidate with columns:
        - driver_name
        - peak_3yr_win_rate: win rate in best consecutive 3 seasons
        - best_championship_gap: largest gap to 2nd in championship (points)
        - consecutive_wins_record: most consecutive race wins
        - dominant_seasons_count: seasons where win rate > 50%
        - dominance_score: normalized 0-100 score
    """
    if candidates is None:
        candidates = DEFAULT_GOAT_CANDIDATES
    candidates = _validate_candidates(master_df, candidates)
    if not candidates:
        return pd.DataFrame()

    df = master_df[master_df["driver_name"].isin(candidates)].copy()
    df = df.sort_values(["driver_name","year","round"]).reset_index(drop=True)
    df["_is_win"] = df["positionOrder"] == 1

    champ_gaps = _compute_championship_gaps(master_df)

    records = []
    for driver in candidates:
        ddf = df[df["driver_name"] == driver]
        if ddf.empty:
            records.append(_empty_dominance_record(driver))
            continue

        season_stats = (
            ddf.groupby("year")
            .agg(wins=("_is_win","sum"), races=("_is_win","count"))
            .assign(win_rate=lambda x: x["wins"] / x["races"])
        )

        if len(season_stats) >= 3:
            peak_3yr_win_rate = float(season_stats["win_rate"].rolling(3).mean().max()) * 100.0
        elif len(season_stats) > 0:
            peak_3yr_win_rate = float(season_stats["win_rate"].mean()) * 100.0
        else:
            peak_3yr_win_rate = 0.0

        driver_gaps = champ_gaps[champ_gaps["driver_name"] == driver]
        best_championship_gap = float(driver_gaps["pts_gap_to_2nd"].max()) if not driver_gaps.empty else 0.0

        consecutive_wins_record = _longest_consecutive_wins(ddf)
        dominant_seasons_count  = int((season_stats["win_rate"] > 0.50).sum())

        records.append({
            "driver_name":             driver,
            "peak_3yr_win_rate":       round(peak_3yr_win_rate, 2),
            "best_championship_gap":   round(best_championship_gap, 1),
            "consecutive_wins_record": consecutive_wins_record,
            "dominant_seasons_count":  dominant_seasons_count,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    scores = pd.DataFrame({
        "peak_wins":   _normalize_0_100(result["peak_3yr_win_rate"]),
        "champ_gap":   _normalize_0_100(result["best_championship_gap"]),
        "consec_wins": _normalize_0_100(result["consecutive_wins_record"].astype(float)),
        "dom_seasons": _normalize_0_100(result["dominant_seasons_count"].astype(float)),
    }, index=result.index)

    result["dominance_score"] = (
        0.30 * scores["peak_wins"]   + 0.30 * scores["champ_gap"]
        + 0.20 * scores["consec_wins"] + 0.20 * scores["dom_seasons"]
    ).round(2)
    return result


def _empty_dominance_record(driver: str) -> dict:
    """Zeroed dominance record for a driver with no data."""
    return {
        "driver_name": driver, "peak_3yr_win_rate": 0.0,
        "best_championship_gap": 0.0, "consecutive_wins_record": 0,
        "dominant_seasons_count": 0,
    }


def _compute_championship_gaps(master_df: pd.DataFrame) -> pd.DataFrame:
    """Points gap between champion and runner-up for each season.

    Returns DataFrame with columns: year, driver_name, pts_gap_to_2nd.
    Only the season winner appears; gap is always >= 0.
    """
    season_pts = master_df.groupby(["year","driver_name"])["points"].sum().reset_index()
    gaps = []
    for year, grp in season_pts.groupby("year"):
        grp = grp.sort_values("points", ascending=False).reset_index(drop=True)
        if len(grp) < 2:
            continue
        gaps.append({
            "year":           year,
            "driver_name":    grp.loc[0, "driver_name"],
            "pts_gap_to_2nd": float(grp.loc[0, "points"] - grp.loc[1, "points"]),
        })
    return pd.DataFrame(gaps) if gaps else pd.DataFrame(
        columns=["year","driver_name","pts_gap_to_2nd"])


def _longest_consecutive_wins(ddf: pd.DataFrame) -> int:
    """Longest consecutive race-win streak. Any non-win resets the counter."""
    if "positionOrder" not in ddf.columns:
        return 0
    is_win = ddf.sort_values(["year","round"])["positionOrder"].eq(1).astype(int).values
    max_streak = cur = 0
    for w in is_win:
        cur = cur + 1 if w else 0
        max_streak = max(max_streak, cur)
    return int(max_streak)


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL GOAT SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_goat_scores(
    raw_stats: pd.DataFrame,
    teammate_comparison: pd.DataFrame,
    same_car_ml: pd.DataFrame,
    adaptability: pd.DataFrame,
    dominance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all 5 lenses into final GOAT score.

    Each lens sub-score is re-normalized to [0,100] across the candidate set
    before weighting, ensuring relative spread within each lens is preserved.

    Formula:
    GOAT_Score = (0.20 × Normalized_Stats_Score) +
                 (0.30 × Teammate_Delta_Score) +
                 (0.25 × Same_Car_ML_Score) +
                 (0.15 × Adaptability_Score) +
                 (0.10 × Dominance_Index)

    Each lens score is normalized to 0-100 before weighting.

    Parameters
    ----------
    raw_stats : pd.DataFrame
        Output of compute_raw_stats(). Must contain normalized_stats_score.
    teammate_comparison : pd.DataFrame
        Output of compute_teammate_comparison(). Must contain teammate_score.
    same_car_ml : pd.DataFrame
        Output of compute_same_car_ml_score(). Must contain same_car_score.
    adaptability : pd.DataFrame
        Output of compute_adaptability_score(). Must contain adaptability_score.
    dominance : pd.DataFrame
        Output of compute_dominance_index(). Must contain dominance_score.

    Returns
    -------
    pd.DataFrame sorted by GOAT_Score descending with all component scores.
    """
    merged = (
        raw_stats
        .merge(teammate_comparison[["driver_name","teammate_score",
                                    "career_race_h2h_pct","career_qual_h2h_pct",
                                    "career_avg_teammate_race_delta",
                                    "career_avg_teammate_qual_delta","num_teammates"]],
               on="driver_name", how="outer")
        .merge(same_car_ml[["driver_name","same_car_score",
                             "simulated_season_points","simulated_wins","simulated_podiums"]],
               on="driver_name", how="outer")
        .merge(adaptability[["driver_name","adaptability_score",
                              "num_teams_won_with","num_circuits_won_at",
                              "races_to_first_win","regulation_era_wins",
                              "team_switch_adaptation"]],
               on="driver_name", how="outer")
        .merge(dominance[["driver_name","dominance_score",
                           "peak_3yr_win_rate","best_championship_gap",
                           "consecutive_wins_record","dominant_seasons_count"]],
               on="driver_name", how="outer")
    )
    if merged.empty:
        return merged

    score_cols = ["normalized_stats_score","teammate_score","same_car_score",
                  "adaptability_score","dominance_score"]
    for col in score_cols:
        if col not in merged.columns:
            merged[col] = 0.0
        # Fill NaN with column minimum to avoid rewarding missing data
        col_min = merged[col].min(skipna=True)
        merged[col] = merged[col].fillna(col_min if pd.notna(col_min) else 0.0)

    # Re-normalize each sub-score across the candidate set for fair comparison
    for col in score_cols:
        merged[col] = _normalize_0_100(merged[col]).round(2)

    merged["GOAT_Score"] = (
        GOAT_WEIGHTS["normalized_stats_score"] * merged["normalized_stats_score"]
        + GOAT_WEIGHTS["teammate_score"]        * merged["teammate_score"]
        + GOAT_WEIGHTS["same_car_score"]        * merged["same_car_score"]
        + GOAT_WEIGHTS["adaptability_score"]    * merged["adaptability_score"]
        + GOAT_WEIGHTS["dominance_score"]       * merged["dominance_score"]
    ).round(2)

    merged = merged.sort_values("GOAT_Score", ascending=False).reset_index(drop=True)
    merged.insert(0, "GOAT_Rank", range(1, len(merged) + 1))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_goat_analysis(
    master_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    model=None,
    candidates: list = None,
) -> pd.DataFrame:
    """Run complete GOAT analysis. Returns final ranked GOAT table.

    Orchestrates all five lens computations and combines them into a single
    ranked GOAT table using compute_goat_scores().

    Parameters
    ----------
    master_df : pd.DataFrame
        Cleaned master DataFrame produced by data_loader.build_master().
    featured_df : pd.DataFrame
        Feature-engineered DataFrame produced by
        feature_engineering.engineer_features().
    model : trained ML model, optional
        A trained estimator from models.py for the Same Car Test (Lens 3).
        When None, a skill-based heuristic is used instead.
    candidates : list, optional
        Driver names to include. Defaults to DEFAULT_GOAT_CANDIDATES (11 drivers).

    Returns
    -------
    pd.DataFrame
        Final GOAT table sorted by GOAT_Score descending. Columns:
        GOAT_Rank, driver_name, GOAT_Score, normalized_stats_score,
        teammate_score, same_car_score, adaptability_score, dominance_score,
        plus all constituent statistics from each lens.

    Examples
    --------
    >>> from src.data_loader import build_master
    >>> from src.feature_engineering import engineer_features
    >>> master   = build_master("data/raw/")
    >>> featured = engineer_features(master)
    >>> goat = run_goat_analysis(master, featured)
    >>> print(goat[["GOAT_Rank", "driver_name", "GOAT_Score"]].to_string())
    """
    if candidates is None:
        candidates = DEFAULT_GOAT_CANDIDATES

    print(f"[GOAT Analysis] Starting analysis for {len(candidates)} candidates ...")
    print(f"[GOAT Analysis] Dataset: {len(master_df):,} race results "
          f"({master_df['year'].min()}\u2013{master_df['year'].max()})")

    # Lens 1: Era-normalised raw statistics
    print("[GOAT Analysis] [1/5] Computing era-normalised raw statistics ...")
    raw_stats = compute_raw_stats(master_df, featured_df, candidates)

    # Lens 2: Intra-team head-to-head performance
    print("[GOAT Analysis] [2/5] Computing intra-team head-to-head metrics ...")
    teammate_comparison = compute_teammate_comparison(master_df, featured_df, candidates)

    # Lens 3: Same-car ML simulation (or heuristic fallback)
    model_label = "ML model" if model is not None else "heuristic fallback"
    print(f"[GOAT Analysis] [3/5] Running same-car simulation ({model_label}) ...")
    same_car_ml = compute_same_car_ml_score(featured_df, candidates, model)

    # Lens 4: Adaptability across teams, circuits, and regulation eras
    print("[GOAT Analysis] [4/5] Computing adaptability scores ...")
    adaptability = compute_adaptability_score(master_df, featured_df, candidates)

    # Lens 5: Peak dominance metrics
    print("[GOAT Analysis] [5/5] Computing dominance indices ...")
    dominance = compute_dominance_index(master_df, featured_df, candidates)

    # Combine all five lenses into final weighted GOAT score
    print("[GOAT Analysis] Combining lenses into final GOAT score ...")
    final_table = compute_goat_scores(
        raw_stats, teammate_comparison, same_car_ml, adaptability, dominance
    )

    top3 = final_table["driver_name"].head(3).tolist()
    print(f"[GOAT Analysis] \u2713 Analysis complete. Top 3: {', '.join(top3)}")
    return final_table
