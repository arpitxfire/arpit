"""
models.py
---------
Model training, hyperparameter tuning, evaluation, and prediction for the
Formula 1 Machine Learning project.

Pipeline overview
-----------------
1. Time-based train/val/test split (NEVER random for time-series data).
2. Feature matrix preparation with NaN imputation.
3. Race-winner classification model (XGBoost / LightGBM / RandomForest).
4. Optuna hyperparameter optimisation using TimeSeriesSplit CV.
5. Model evaluation with F1-relevant metrics.
6. User-facing prediction helpers (by track + year).
7. Constructor championship model (team-season aggregation).
8. Persistence helpers (save / load via joblib, with SHAP explainer).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

# Suppress verbose Optuna output globally
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical list of engineered feature columns produced by feature_engineering.py
FEATURE_COLS: list[str] = [
    # ── Category A: Driver Skill ─────────────────────────────────────────────
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
    # ── Category B: Constructor ──────────────────────────────────────────────
    "team_wins_last_5",
    "team_avg_finish_last_5",
    "team_podiums_last_5",
    "team_championship_pos",
    "team_dnf_rate_season",
    "team_wins_at_circuit",
    "team_points_last_5",
    # ── Category C: Circuit ──────────────────────────────────────────────────
    "driver_prev_wins_at_circuit",
    "driver_avg_finish_at_circuit",
    "driver_races_at_circuit",
    "circuit_avg_grid_delta",
    "circuit_avg_dnf_rate",
    "circuit_country_encoded",
    "altitude",
    # ── Category D: Race Context ─────────────────────────────────────────────
    "grid_position",
    "season_round",
    "year",
    "field_size",
    "is_wet_race",
    "points_scored",
    # ── Category E: Era Normalization / Teammate ─────────────────────────────
    "points_normalized",
    "position_percentile",
    "teammate_finish",
    "teammate_delta",
    "teammate_qual_delta",
]

#: Binary target — 1 when the driver won the race (positionOrder == 1)
TARGET_WIN: str = "is_win"

#: Regression target — final classified finishing position
TARGET_POS: str = "positionOrder"

# Default model directory relative to this source file
_DEFAULT_MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# 1. Time-Based Train / Val / Test Split
# ---------------------------------------------------------------------------


def get_time_split(
    df: pd.DataFrame,
    train_end_year: int = 2021,
    val_year: int = 2022,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return time-based train, validation, and test subsets of *df*.

    The split respects temporal order to prevent data leakage:

    * **Train**  — all rows where ``year <= train_end_year``
    * **Val**    — all rows where ``year == val_year``
    * **Test**   — all rows where ``year > val_year``

    Parameters
    ----------
    df:
        Featured DataFrame produced by ``feature_engineering.engineer_features``.
        Must contain a ``year`` column.
    train_end_year:
        Last season (inclusive) included in the training set.  Defaults to 2021.
    val_year:
        Season used exclusively for validation / early-stopping.  Defaults to 2022.

    Returns
    -------
    tuple of (train_df, val_df, test_df)
        Three non-overlapping DataFrames ordered chronologically.

    Notes
    -----
    NEVER use random splits for time-series data — doing so would allow future
    information to leak into the training set and produce unrealistically
    optimistic evaluation metrics.
    """
    if "year" not in df.columns:
        raise ValueError("DataFrame must contain a 'year' column for time-based splitting.")

    train_df = df[df["year"] <= train_end_year].copy()
    val_df = df[df["year"] == val_year].copy()
    test_df = df[df["year"] > val_year].copy()

    print(
        f"[models] Time split — "
        f"train: {len(train_df):,} rows (≤{train_end_year}), "
        f"val: {len(val_df):,} rows ({val_year}), "
        f"test: {len(test_df):,} rows (>{val_year})"
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. Feature List Management & Preparation
# ---------------------------------------------------------------------------


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix and target arrays from a featured DataFrame.

    NaN values are imputed column-wise with the column median so that models
    receive a fully dense matrix.  A warning is printed for any feature column
    that is absent from *df* so callers can diagnose pipeline issues without
    raising hard errors.

    Parameters
    ----------
    df:
        Featured DataFrame; must contain ``positionOrder`` (used to derive
        ``is_win``) and ideally all columns listed in ``FEATURE_COLS``.
    feature_cols:
        Explicit list of feature column names.  Defaults to ``FEATURE_COLS``.

    Returns
    -------
    tuple of (X, y_win, y_pos)
        * **X**      — ``np.ndarray`` of shape ``(n, p)`` with imputed features.
        * **y_win**  — ``np.ndarray`` of shape ``(n,)`` with binary win labels.
        * **y_pos**  — ``np.ndarray`` of shape ``(n,)`` with integer positions.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # ── Target arrays ────────────────────────────────────────────────────────
    if TARGET_POS not in df.columns:
        raise ValueError(f"DataFrame must contain '{TARGET_POS}' column.")

    y_pos = df[TARGET_POS].to_numpy(dtype=float)

    if TARGET_WIN in df.columns:
        y_win = df[TARGET_WIN].to_numpy(dtype=int)
    else:
        y_win = (df[TARGET_POS] == 1).astype(int).to_numpy()

    # ── Feature matrix ───────────────────────────────────────────────────────
    present_cols = []
    missing_cols = []
    for col in feature_cols:
        if col in df.columns:
            present_cols.append(col)
        else:
            missing_cols.append(col)

    if missing_cols:
        warnings.warn(
            f"[models] {len(missing_cols)} feature column(s) missing from DataFrame "
            f"and will be filled with zeros: {missing_cols}",
            UserWarning,
            stacklevel=2,
        )

    X_df = df[present_cols].copy()

    # Fill missing columns with zeros
    for col in missing_cols:
        X_df[col] = 0.0

    # Reorder to canonical order
    X_df = X_df[feature_cols]

    # Median imputation for NaN values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df)

    return X, y_win, y_pos


# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------


def train_race_winner_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "xgboost",
    use_optuna: bool = True,
    n_trials: int = 100,
) -> object:
    """Train a race-winner prediction model.

    Class imbalance (wins are ~5 % of rows) is addressed via ``scale_pos_weight``
    (XGBoost / LightGBM) or ``class_weight='balanced'`` (RandomForest).

    Parameters
    ----------
    X_train:
        Feature matrix for training.
    y_train:
        Binary win labels (1 = winner, 0 = non-winner).
    model_type:
        One of ``'xgboost'``, ``'lightgbm'``, or ``'random_forest'``.
    use_optuna:
        When ``True`` (default), run Optuna hyperparameter search before
        training the final model.
    n_trials:
        Number of Optuna trials.  Only used when ``use_optuna=True``.

    Returns
    -------
    Trained classifier with a ``predict_proba`` method.
    """
    if model_type not in ("xgboost", "lightgbm", "random_forest"):
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: xgboost, lightgbm, random_forest.")

    # Class imbalance ratio
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    if use_optuna:
        print(f"[models] Running Optuna ({n_trials} trials) for {model_type} …")
        best_params = tune_with_optuna(X_train, y_train, model_type=model_type, n_trials=n_trials)
        print(f"[models] Best params: {best_params}")
    else:
        best_params = {}

    model = _build_model(model_type, best_params, scale_pos_weight)
    model.fit(X_train, y_train)
    print(f"[models] {model_type} training complete.")
    return model


def _build_model(model_type: str, params: dict, scale_pos_weight: float) -> object:
    """Instantiate a classifier from *model_type*, *params*, and imbalance ratio.

    Parameters
    ----------
    model_type:
        One of ``'xgboost'``, ``'lightgbm'``, or ``'random_forest'``.
    params:
        Hyperparameters dict (e.g. from Optuna).
    scale_pos_weight:
        Ratio of negative to positive class samples for imbalance handling.

    Returns
    -------
    Untrained classifier instance.
    """
    if model_type == "xgboost":
        defaults = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            verbosity=0,
            random_state=42,
        )
        defaults.update(params)
        # Ensure imbalance weight and silence are always set
        defaults["scale_pos_weight"] = scale_pos_weight
        defaults["verbosity"] = 0
        return xgb.XGBClassifier(**defaults)

    if model_type == "lightgbm":
        defaults = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
            random_state=42,
        )
        defaults.update(params)
        defaults["scale_pos_weight"] = scale_pos_weight
        defaults["verbosity"] = -1
        return lgb.LGBMClassifier(**defaults)

    # random_forest
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    defaults.update(params)
    defaults["class_weight"] = "balanced"
    return RandomForestClassifier(**defaults)


# ---------------------------------------------------------------------------
# 4. Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------


def tune_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "xgboost",
    n_trials: int = 100,
) -> dict:
    """Run Optuna hyperparameter optimisation, minimising validation log-loss.

    Cross-validation uses :class:`sklearn.model_selection.TimeSeriesSplit` with
    ``n_splits=5`` to respect temporal ordering and avoid look-ahead bias.

    Parameters
    ----------
    X_train:
        Feature matrix.
    y_train:
        Binary win labels.
    model_type:
        One of ``'xgboost'``, ``'lightgbm'``, or ``'random_forest'``.
    n_trials:
        Number of Optuna trials to run.

    Returns
    -------
    dict
        Best hyperparameter dictionary found by Optuna.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    tscv = TimeSeriesSplit(n_splits=5)

    def _objective(trial: optuna.Trial) -> float:
        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            clf = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )

        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            clf = lgb.LGBMClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                verbosity=-1,
                random_state=42,
            )

        else:  # random_forest
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            clf = RandomForestClassifier(
                **params,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )

        cv_losses: list[float] = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

            # Skip fold if validation set has only one class
            if len(np.unique(y_cv_val)) < 2:
                continue

            clf.fit(X_cv_train, y_cv_train)
            proba = clf.predict_proba(X_cv_val)[:, 1]
            cv_losses.append(log_loss(y_cv_val, proba))

        return float(np.mean(cv_losses)) if cv_losses else float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# 5. Model Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test_win: np.ndarray,
    y_test_pos: np.ndarray,
    race_ids: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate a trained race-winner model against hold-out test data.

    Parameters
    ----------
    model:
        Trained classifier with a ``predict_proba`` method.
    X_test:
        Feature matrix for test rows.
    y_test_win:
        Binary win labels for test rows.
    y_test_pos:
        Integer finishing positions for test rows.
    race_ids:
        Optional array of race identifiers aligned with *X_test* rows, used to
        compute per-race metrics (``precision_at_1``, ``top3_accuracy``).
        When ``None`` each row is treated as its own race.

    Returns
    -------
    dict with keys:
        * ``roc_auc``        — ROC-AUC for win-probability predictions.
        * ``precision_at_1`` — fraction of races where the top-predicted driver
          actually won.
        * ``top3_accuracy``  — fraction of races where the actual winner was in
          the model's top-3 predicted drivers.
        * ``log_loss``       — binary cross-entropy of win probabilities.
        * ``position_mae``   — MAE between predicted rank and actual position.
        * ``position_r2``    — R² between predicted rank and actual position.
    """
    win_proba = model.predict_proba(X_test)[:, 1]

    # ── Global classification metrics ────────────────────────────────────────
    roc_auc = roc_auc_score(y_test_win, win_proba) if len(np.unique(y_test_win)) > 1 else float("nan")
    ll = log_loss(y_test_win, win_proba)

    # ── Per-race metrics ─────────────────────────────────────────────────────
    if race_ids is None:
        race_ids = np.arange(len(y_test_win))

    precision_at_1_scores: list[float] = []
    top3_accuracy_scores: list[float] = []
    predicted_ranks: list[float] = []

    for race_id in np.unique(race_ids):
        mask = race_ids == race_id
        proba_race = win_proba[mask]
        y_win_race = y_test_win[mask]

        if len(proba_race) == 0:
            continue

        sorted_idx = np.argsort(proba_race)[::-1]
        actual_winner_idx = np.where(y_win_race == 1)[0]

        # precision@1: did the highest-probability driver actually win?
        precision_at_1_scores.append(float(y_win_race[sorted_idx[0]] == 1))

        # top-3 accuracy: was the actual winner in the top-3 predictions?
        top3 = set(sorted_idx[:3].tolist())
        if len(actual_winner_idx) > 0:
            top3_accuracy_scores.append(float(actual_winner_idx[0] in top3))

        # Predicted rank: rank of each driver by predicted probability
        ranks = np.empty(len(proba_race))
        ranks[sorted_idx] = np.arange(1, len(proba_race) + 1)
        predicted_ranks.extend(ranks.tolist())

    precision_at_1 = float(np.mean(precision_at_1_scores)) if precision_at_1_scores else float("nan")
    top3_accuracy = float(np.mean(top3_accuracy_scores)) if top3_accuracy_scores else float("nan")

    # ── Position regression metrics ──────────────────────────────────────────
    pred_ranks_arr = np.array(predicted_ranks) if predicted_ranks else np.zeros(len(y_test_pos))
    # Align lengths (edge case when race_ids has fewer unique values than rows)
    min_len = min(len(pred_ranks_arr), len(y_test_pos))
    position_mae = mean_absolute_error(y_test_pos[:min_len], pred_ranks_arr[:min_len])
    position_r2 = r2_score(y_test_pos[:min_len], pred_ranks_arr[:min_len])

    metrics = {
        "roc_auc": roc_auc,
        "precision_at_1": precision_at_1,
        "top3_accuracy": top3_accuracy,
        "log_loss": ll,
        "position_mae": position_mae,
        "position_r2": position_r2,
    }

    print(
        f"[models] Evaluation — "
        f"ROC-AUC: {roc_auc:.4f} | "
        f"P@1: {precision_at_1:.4f} | "
        f"Top-3: {top3_accuracy:.4f} | "
        f"LogLoss: {ll:.4f} | "
        f"MAE: {position_mae:.4f} | "
        f"R²: {position_r2:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# 6. User-Facing Prediction Functions
# ---------------------------------------------------------------------------


def predict_race_winner(
    featured_df: pd.DataFrame,
    track: str,
    year: int,
    model=None,
) -> pd.DataFrame:
    """Predict win probability for each driver at a given track and year.

    Looks up the race in *featured_df* by matching ``circuitId`` / ``name``
    (case-insensitive substring) and ``year``.  If ``model`` is ``None`` the
    saved model is loaded from ``../models/race_winner_model.pkl`` relative to
    this file.

    Parameters
    ----------
    featured_df:
        Full featured DataFrame (output of ``engineer_features``).
    track:
        Circuit name or partial name (case-insensitive substring match against
        the ``name`` column, or exact ``circuitId`` string match).
    year:
        Season year for which to make predictions.
    model:
        Pre-loaded classifier.  When ``None``, loads from the default path.

    Returns
    -------
    pd.DataFrame
        Columns: ``driver_name``, ``team``, ``grid``, ``win_probability``,
        ``predicted_position``; sorted by ``win_probability`` descending.
    """
    if model is None:
        model_path = str(_DEFAULT_MODEL_DIR / "race_winner_model.pkl")
        model = load_model(model_path)
        if model is None:
            raise FileNotFoundError(
                f"No model found at '{model_path}'. Train a model first or pass model= explicitly."
            )

    race_df = _filter_race(featured_df, track, year)

    X, _, _ = prepare_features(race_df)
    win_proba = model.predict_proba(X)[:, 1]

    # Predicted position = rank by descending probability
    ranks = pd.Series(win_proba).rank(ascending=False).astype(int).to_numpy()

    result = pd.DataFrame(
        {
            "driver_name": _safe_col(race_df, ["driverRef", "driver_name", "driverId"], "unknown"),
            "team": _safe_col(race_df, ["constructorRef", "team", "constructorId"], "unknown"),
            "grid": _safe_col(race_df, ["grid_position", "grid"], np.nan),
            "win_probability": win_proba,
            "predicted_position": ranks,
        }
    )
    return result.sort_values("win_probability", ascending=False).reset_index(drop=True)


def predict_historical_race(
    featured_df: pd.DataFrame,
    track: str,
    year: int,
) -> pd.DataFrame:
    """Predict race outcome for a historical what-if scenario.

    Identical to :func:`predict_race_winner` but semantically scoped to
    historical races where the actual results are already known, allowing
    model predictions to be compared against reality.

    Parameters
    ----------
    featured_df:
        Full featured DataFrame.
    track:
        Circuit name or partial name (case-insensitive).
    year:
        Season year.

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`predict_race_winner`, with an additional
        ``actual_position`` column when ``positionOrder`` is available.
    """
    result = predict_race_winner(featured_df, track, year)

    # Attach actual positions when available for side-by-side comparison
    race_df = _filter_race(featured_df, track, year).reset_index(drop=True)
    if TARGET_POS in race_df.columns:
        # Align by row order (predict_race_winner does not reorder rows, only sorts output)
        # Recompute to get unsorted alignment
        model_path = str(_DEFAULT_MODEL_DIR / "race_winner_model.pkl")
        model = load_model(model_path)
        if model is not None:
            X, _, _ = prepare_features(race_df)
            win_proba = model.predict_proba(X)[:, 1]
            driver_col = _safe_col(race_df, ["driverRef", "driver_name", "driverId"], "unknown")
            actual_map = dict(zip(driver_col, race_df[TARGET_POS].to_numpy()))
            result["actual_position"] = result["driver_name"].map(actual_map)

    return result


# ---------------------------------------------------------------------------
# 7. Constructor Championship Model
# ---------------------------------------------------------------------------


def build_constructor_season_df(featured_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate race-level data to one row per constructor per season.

    Computes the following per-constructor, per-season aggregates used as
    features and targets for the constructor championship model:

    * ``total_points``             — sum of ``points_scored`` for the season.
    * ``avg_finish``               — mean finishing position.
    * ``total_wins``               — number of wins.
    * ``total_podiums``            — number of podiums (top 3).
    * ``total_dnfs``               — number of DNFs (position > field size or
      status-derived; approximated from finishing position outliers).
    * ``development_trajectory``   — linear slope of cumulative points across
      race rounds (proxy for car development rate).
    * ``is_champion``              — 1 if the constructor won the championship.
    * ``championship_final_pos``   — ordinal finishing position in the
      constructors' championship.

    Parameters
    ----------
    featured_df:
        Full featured DataFrame with at least ``year``, ``constructorId``,
        ``positionOrder``, ``points_scored``, ``season_round``.

    Returns
    -------
    pd.DataFrame
        One row per ``(constructorId, year)`` pair.
    """
    required = {"year", "constructorId", "positionOrder", "points_scored", "season_round"}
    missing = required - set(featured_df.columns)
    if missing:
        raise ValueError(f"featured_df is missing required columns: {missing}")

    df = featured_df.copy()

    # Approximate DNF as position > 20 (positional codes for retirements)
    df["_is_dnf"] = (df["positionOrder"] > 20).astype(int)
    df["_is_win"] = (df["positionOrder"] == 1).astype(int)
    df["_is_podium"] = (df["positionOrder"] <= 3).astype(int)

    agg = (
        df.groupby(["constructorId", "year"])
        .agg(
            total_points=("points_scored", "sum"),
            avg_finish=("positionOrder", "mean"),
            total_wins=("_is_win", "sum"),
            total_podiums=("_is_podium", "sum"),
            total_dnfs=("_is_dnf", "sum"),
        )
        .reset_index()
    )

    # ── Development trajectory: slope of cumulative points vs. round ─────────
    def _slope(group: pd.DataFrame) -> float:
        rounds = group["season_round"].to_numpy(dtype=float)
        cum_pts = group["points_scored"].cumsum().to_numpy(dtype=float)
        if len(rounds) < 2:
            return 0.0
        coeffs = np.polyfit(rounds, cum_pts, 1)
        return float(coeffs[0])

    traj = (
        df.groupby(["constructorId", "year"])
        .apply(_slope)
        .reset_index(name="development_trajectory")
    )
    agg = agg.merge(traj, on=["constructorId", "year"], how="left")

    # ── Championship position per season ─────────────────────────────────────
    season_pts = (
        agg.groupby("year")
        .apply(lambda g: g.assign(championship_final_pos=g["total_points"].rank(ascending=False).astype(int)))
        .reset_index(drop=True)
    )

    season_pts["is_champion"] = (season_pts["championship_final_pos"] == 1).astype(int)

    return season_pts.drop(columns=["_is_dnf", "_is_win", "_is_podium"], errors="ignore")


def train_constructor_model(constructor_season_df: pd.DataFrame) -> object:
    """Train a constructor championship position predictor.

    Uses a time-based split (seasons up to 2021 for training, 2022+ for test)
    and a :class:`sklearn.linear_model.Ridge` regressor to predict
    ``championship_final_pos``.

    Parameters
    ----------
    constructor_season_df:
        DataFrame produced by :func:`build_constructor_season_df`.

    Returns
    -------
    Trained estimator.
    """
    feature_cols = [
        "total_points",
        "avg_finish",
        "total_wins",
        "total_podiums",
        "total_dnfs",
        "development_trajectory",
    ]

    train_df = constructor_season_df[constructor_season_df["year"] <= 2021]
    test_df = constructor_season_df[constructor_season_df["year"] > 2021]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[feature_cols])
    y_train = train_df["championship_final_pos"].to_numpy()

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    if len(test_df) > 0:
        X_test = imputer.transform(test_df[feature_cols])
        y_test = test_df["championship_final_pos"].to_numpy()
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"[models] Constructor model — test MAE: {mae:.3f} championship positions")

    # Attach imputer as attribute for use in prediction
    model._imputer = imputer  # type: ignore[attr-defined]
    model._feature_cols = feature_cols  # type: ignore[attr-defined]
    return model


def predict_constructor_championship(
    constructor_season_df: pd.DataFrame,
    team: str,
    year: int,
    model=None,
) -> dict:
    """Predict a constructor's championship outcome for a given season.

    Parameters
    ----------
    constructor_season_df:
        DataFrame produced by :func:`build_constructor_season_df`.
    team:
        Constructor identifier (case-insensitive substring match against
        ``constructorId``).
    year:
        Target season.
    model:
        Pre-trained constructor model.  When ``None``, loads from
        ``../models/constructor_model.pkl``.

    Returns
    -------
    dict
        Keys: ``team``, ``predicted_position``, ``win_probability``,
        ``predicted_points``.
    """
    if model is None:
        model_path = str(_DEFAULT_MODEL_DIR / "constructor_model.pkl")
        model = load_model(model_path)
        if model is None:
            raise FileNotFoundError(
                f"No constructor model at '{model_path}'. Train first or pass model= explicitly."
            )

    # Find matching team row
    mask_year = constructor_season_df["year"] == year
    mask_team = constructor_season_df["constructorId"].str.lower().str.contains(
        team.lower(), na=False
    )
    row = constructor_season_df[mask_year & mask_team]

    if row.empty:
        raise ValueError(f"No data found for team='{team}', year={year}.")

    feature_cols = getattr(model, "_feature_cols", [
        "total_points", "avg_finish", "total_wins",
        "total_podiums", "total_dnfs", "development_trajectory",
    ])
    imputer = getattr(model, "_imputer", SimpleImputer(strategy="median"))

    X = imputer.transform(row[feature_cols].to_numpy())
    pred_pos = float(model.predict(X)[0])
    pred_pos_int = max(1, round(pred_pos))

    # Win probability: soft heuristic from predicted position
    n_teams = int(constructor_season_df[mask_year]["constructorId"].nunique())
    win_prob = max(0.0, 1.0 - (pred_pos - 1) / max(n_teams - 1, 1))

    return {
        "team": team,
        "predicted_position": pred_pos_int,
        "win_probability": round(win_prob, 4),
        "predicted_points": float(row["total_points"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# 8. Model Persistence
# ---------------------------------------------------------------------------


def save_model(model, path: str) -> None:
    """Persist *model* to *path* using joblib.

    Also attempts to build and save a SHAP :class:`~shap.TreeExplainer`
    alongside the model at ``<path>.shap_explainer.pkl``.  If SHAP explainer
    creation fails (e.g., for unsupported model types) a warning is emitted and
    the model is still saved successfully.

    Parameters
    ----------
    model:
        Any scikit-learn–compatible estimator.
    path:
        Destination file path (e.g. ``'../models/race_winner_model.pkl'``).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    print(f"[models] Model saved → {dest}")

    # ── SHAP explainer ────────────────────────────────────────────────────────
    shap_path = dest.with_suffix(".shap_explainer.pkl")
    try:
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, shap_path)
        print(f"[models] SHAP explainer saved → {shap_path}")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"[models] Could not save SHAP explainer ({type(exc).__name__}: {exc}). "
            "Model was saved successfully.",
            UserWarning,
            stacklevel=2,
        )


def load_model(path: str) -> Optional[object]:
    """Load a model from *path* using joblib.

    Parameters
    ----------
    path:
        File path to a joblib-serialised model.

    Returns
    -------
    Loaded model, or ``None`` if the file does not exist (a warning is emitted).
    """
    src = Path(path)
    if not src.exists():
        warnings.warn(
            f"[models] Model file not found: '{src}'. Returning None.",
            UserWarning,
            stacklevel=2,
        )
        return None
    model = joblib.load(src)
    print(f"[models] Model loaded ← {src}")
    return model


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filter_race(df: pd.DataFrame, track: str, year: int) -> pd.DataFrame:
    """Return rows from *df* that match *track* (substring) and *year*.

    Tries to match ``name`` first, then ``circuitId``.  Raises ``ValueError``
    when no rows are found.

    Parameters
    ----------
    df:
        Featured DataFrame.
    track:
        Case-insensitive substring of the circuit name or exact ``circuitId``.
    year:
        Season year.

    Returns
    -------
    pd.DataFrame
        Subset of *df* for the specified race.
    """
    year_mask = df["year"] == year

    # Try 'name' column first (human-readable circuit name)
    if "name" in df.columns:
        circuit_mask = df["name"].str.lower().str.contains(track.lower(), na=False)
        race_df = df[year_mask & circuit_mask]
        if not race_df.empty:
            return race_df

    # Fall back to circuitId
    if "circuitId" in df.columns:
        circuit_mask = df["circuitId"].str.lower().str.contains(track.lower(), na=False)
        race_df = df[year_mask & circuit_mask]
        if not race_df.empty:
            return race_df

    raise ValueError(
        f"No race data found for track='{track}', year={year}. "
        "Check that the track name matches 'name' or 'circuitId' in the DataFrame."
    )


def _safe_col(
    df: pd.DataFrame,
    candidates: list[str],
    default,
) -> np.ndarray:
    """Return the first candidate column found in *df*, or an array of *default*.

    Parameters
    ----------
    df:
        Source DataFrame.
    candidates:
        Ordered list of column names to try.
    default:
        Scalar fill value when no candidate column exists.

    Returns
    -------
    np.ndarray
        Values from the first found column, or array filled with *default*.
    """
    for col in candidates:
        if col in df.columns:
            return df[col].to_numpy()
    return np.full(len(df), default)
