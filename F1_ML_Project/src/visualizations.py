"""
visualizations.py — Publication-quality F1 visualizations
==========================================================
All chart functions for the F1 ML Analytics project.
Uses matplotlib, seaborn, and plotly. F1 team colour palette included.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# F1 team colour palette
# ---------------------------------------------------------------------------
F1_COLORS = {
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "Red Bull": "#0600EF",
    "McLaren": "#FF8700",
    "Alpine": "#0090FF",
    "Aston Martin": "#006F62",
    "Williams": "#005AFF",
    "AlphaTauri": "#2B4562",
    "Alfa Romeo": "#900000",
    "Haas F1 Team": "#FFFFFF",
    "Renault": "#FFF500",
    "Force India": "#F596C8",
    "Racing Point": "#F596C8",
    "Sauber": "#9B0000",
    "Toro Rosso": "#469BFF",
    "Lotus F1": "#FFB800",
    "default": "#888888",
}

DRIVER_COLORS = {
    "Lewis Hamilton": "#00D2BE",
    "Michael Schumacher": "#DC0000",
    "Max Verstappen": "#0600EF",
    "Sebastian Vettel": "#0600EF",
    "Alain Prost": "#DC0000",
    "Ayrton Senna": "#005AFF",
    "Juan Manuel Fangio": "#AAAAAA",
    "Jim Clark": "#006F62",
    "Jackie Stewart": "#006F62",
    "Niki Lauda": "#DC0000",
    "Fernando Alonso": "#FF8700",
}


def get_team_color(team_name: str) -> str:
    """Return F1 hex colour for a constructor name."""
    for key, color in F1_COLORS.items():
        if key.lower() in str(team_name).lower():
            return color
    return F1_COLORS["default"]


# ---------------------------------------------------------------------------
# 1. SHAP Summary Plot
# ---------------------------------------------------------------------------
def plot_shap_summary(shap_values, X: pd.DataFrame, max_display: int = 20,
                      save_path: str = None):
    """SHAP beeswarm summary plot — which features drive win predictions most."""
    try:
        import shap
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title("SHAP Feature Importance — Race Winner Prediction", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print("✅ SHAP summary plot generated.")
    except ImportError:
        print("⚠️  shap not installed. pip install shap")


# ---------------------------------------------------------------------------
# 2. SHAP Force Plot
# ---------------------------------------------------------------------------
def plot_shap_force(shap_explainer, X_row: pd.DataFrame, save_path: str = None):
    """SHAP force plot for a single prediction (e.g., why Verstappen wins)."""
    try:
        import shap
        shap_vals = shap_explainer(X_row)
        fig = shap.plots.waterfall(shap_vals[0], max_display=15, show=False)
        plt.title("SHAP Force Plot — Single Race Prediction", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print("✅ SHAP force plot generated.")
    except ImportError:
        print("⚠️  shap not installed. pip install shap")


# ---------------------------------------------------------------------------
# 3. Predicted vs Actual Scatter
# ---------------------------------------------------------------------------
def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Predicted vs Actual — Race Position",
                              save_path: str = None):
    """Scatter plot of predicted vs actual race positions with R² annotation."""
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, color="#0600EF", edgecolors="none", s=20)
    lims = [1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Position", fontsize=12)
    ax.set_ylabel("Predicted Position", fontsize=12)
    ax.set_title(f"{title}\n$R^2$ = {r2:.3f}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Predicted vs Actual plot — R² = {r2:.3f}")


# ---------------------------------------------------------------------------
# 4. GOAT Radar Chart (Plotly)
# ---------------------------------------------------------------------------
def plot_goat_radar(goat_df: pd.DataFrame, save_path: str = None):
    """Interactive Plotly radar chart comparing GOAT candidates on 5 lenses."""
    try:
        import plotly.graph_objects as go
        categories = ["Raw Stats", "Teammate Delta", "Same Car ML", "Adaptability", "Dominance"]
        score_cols = ["normalized_stats_score", "teammate_score", "same_car_score",
                      "adaptability_score", "dominance_score"]
        # Fallback if score columns missing
        available = [c for c in score_cols if c in goat_df.columns]
        if len(available) < 2:
            print("⚠️  GOAT score columns not found — run goat_analysis first.")
            return
        fig = go.Figure()
        for _, row in goat_df.iterrows():
            driver = row.get("driver_name", "Unknown")
            vals = [row.get(c, 50) for c in score_cols]
            vals += [vals[0]]  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                name=driver,
                fill="toself",
                opacity=0.6,
                line=dict(color=DRIVER_COLORS.get(driver, "#888888"), width=2)
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="🏆 F1 GOAT Analysis — 5-Lens Radar Chart",
            showlegend=True,
            template="plotly_dark",
            height=600
        )
        if save_path:
            fig.write_html(save_path)
        fig.show()
        print("✅ GOAT radar chart generated.")
    except ImportError:
        print("⚠️  plotly not installed. pip install plotly")


# ---------------------------------------------------------------------------
# 5. Hypothetical Season Simulation — Animated Cumulative Points (Plotly)
# ---------------------------------------------------------------------------
def plot_hypothetical_season(simulation_results: dict, save_path: str = None):
    """
    Animated cumulative points chart from hypothetical scenario simulation.

    simulation_results should have 'race_by_race' key with list of
    {'round': int, 'circuit': str, 'points': int} dicts.
    """
    try:
        import plotly.graph_objects as go
        races = simulation_results.get("race_by_race", [])
        if not races:
            print("⚠️  No race-by-race data in simulation_results.")
            return
        rounds = [r["round"] for r in races]
        circuits = [r.get("circuit", f"R{r['round']}") for r in races]
        cumulative = np.cumsum([r.get("points", 0) for r in races])
        driver = simulation_results.get("driver", "Driver")
        team = simulation_results.get("team", "Team")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=circuits, y=cumulative,
            mode="lines+markers",
            name=f"{driver} ({team})",
            line=dict(color="#FF8700", width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f"📊 Hypothetical Season — {driver} in {team} ({simulation_results.get('year', '')})",
            xaxis_title="Race",
            yaxis_title="Cumulative Points",
            template="plotly_dark",
            height=450,
            xaxis=dict(tickangle=-45)
        )
        if save_path:
            fig.write_html(save_path)
        fig.show()
        print("✅ Hypothetical season chart generated.")
    except ImportError:
        print("⚠️  plotly not installed. pip install plotly")


# ---------------------------------------------------------------------------
# 6. Feature Importance Bar Chart (seaborn)
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                             save_path: str = None):
    """Top-N feature importance bar chart (works for XGBoost, LightGBM, RF)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_booster"):
        importances = model.feature_importances_
    else:
        print("⚠️  Model does not have feature_importances_.")
        return
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=fi_df, y="feature", x="importance", palette="rocket_r", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances — Race Winner Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Feature importance chart — top {top_n} features.")


# ---------------------------------------------------------------------------
# 7. Confusion Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels=None, save_path: str = None):
    """Confusion matrix heatmap for win/no-win classification."""
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = ["No Win", "Win"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix — Win Prediction", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(classification_report(y_true, y_pred, target_names=labels))
    print("✅ Confusion matrix plotted.")


# ---------------------------------------------------------------------------
# 8. Win Probability by Grid Position
# ---------------------------------------------------------------------------
def plot_win_prob_by_grid(featured_df: pd.DataFrame, model=None,
                          save_path: str = None):
    """
    Seaborn line/bar chart of historical win probability by grid position.
    If model is provided, also plots model-predicted win probability.
    """
    df = featured_df.copy()
    df = df[df["grid"].between(1, 20)]
    df["is_win"] = (df["positionOrder"] == 1).astype(int)
    hist = df.groupby("grid")["is_win"].mean().reset_index()
    hist.columns = ["grid_position", "win_probability"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=hist, x="grid_position", y="win_probability",
                color="#0600EF", alpha=0.7, ax=ax)
    ax.set_title("Historical Win Probability by Grid Position (1950–2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Grid Position")
    ax.set_ylabel("Win Probability")
    ax.set_ylim(0, 0.6)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate r value
    from scipy.stats import pearsonr
    r, _ = pearsonr(hist["grid_position"], hist["win_probability"])
    ax.text(0.7, 0.85, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            fontsize=11, color="red")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Win probability by grid position — Pearson r = {r:.3f}")


# ---------------------------------------------------------------------------
# 9. Era-Adjusted Career Trajectories (Plotly)
# ---------------------------------------------------------------------------
def plot_career_trajectories(master_df: pd.DataFrame, drivers: list = None,
                              save_path: str = None):
    """
    Interactive Plotly line chart of era-adjusted cumulative wins per season.
    Normalizes wins by season length (races) to compare across eras.
    """
    try:
        import plotly.express as px
        if drivers is None:
            drivers = ["Lewis Hamilton", "Michael Schumacher", "Max Verstappen",
                       "Sebastian Vettel", "Alain Prost", "Ayrton Senna"]
        df = master_df.copy()
        if "driver_name" not in df.columns:
            df["driver_name"] = df.get("forename", "") + " " + df.get("surname", "")
            df["driver_name"] = df["driver_name"].str.strip()
        df = df[df["driver_name"].isin(drivers)]
        df["is_win"] = (df["positionOrder"] == 1).astype(int)
        season_wins = df.groupby(["driver_name", "year"]).agg(
            wins=("is_win", "sum"),
            races=("raceId", "nunique")
        ).reset_index()
        season_wins["win_rate"] = season_wins["wins"] / season_wins["races"]
        season_wins["cum_wins"] = season_wins.groupby("driver_name")["wins"].cumsum()
        fig = px.line(
            season_wins, x="year", y="win_rate", color="driver_name",
            title="Era-Adjusted Career Win Rate by Season",
            labels={"win_rate": "Win Rate (wins/races)", "year": "Season"},
            color_discrete_map=DRIVER_COLORS,
            template="plotly_dark"
        )
        fig.update_traces(mode="lines+markers", marker=dict(size=5))
        fig.update_layout(height=500, legend_title="Driver")
        if save_path:
            fig.write_html(save_path)
        fig.show()
        print("✅ Career trajectories chart generated.")
    except ImportError:
        print("⚠️  plotly not installed. pip install plotly")


# ---------------------------------------------------------------------------
# 10. Monte Carlo Distribution
# ---------------------------------------------------------------------------
def plot_monte_carlo_distribution(mc_results: list, driver: str = "Driver",
                                  team: str = "Team", year: int = None,
                                  save_path: str = None):
    """
    Histogram of 1000 Monte Carlo simulated season points totals.
    Shows mean, median, and 90% confidence interval.
    """
    arr = np.array(mc_results)
    mean_pts = np.mean(arr)
    median_pts = np.median(arr)
    p5, p95 = np.percentile(arr, [5, 95])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(arr, bins=50, color="#0600EF", alpha=0.7, edgecolor="white")
    ax.axvline(mean_pts, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_pts:.0f} pts")
    ax.axvline(median_pts, color="orange", linestyle="-", linewidth=2, label=f"Median: {median_pts:.0f} pts")
    ax.axvspan(p5, p95, alpha=0.15, color="green", label=f"90% CI: [{p5:.0f}, {p95:.0f}]")
    title = f"Monte Carlo Season Points — {driver} in {team}"
    if year:
        title += f" ({year})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Season Points Total")
    ax.set_ylabel("Frequency (out of 1000 simulations)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Monte Carlo distribution — Mean: {mean_pts:.0f}, 90% CI: [{p5:.0f}, {p95:.0f}]")


# ---------------------------------------------------------------------------
# 11. Teammate Comparison Chart
# ---------------------------------------------------------------------------
def plot_teammate_comparison(teammate_df: pd.DataFrame, save_path: str = None):
    """
    Grouped bar chart showing qual H2H % and race H2H % for GOAT candidates.
    """
    df = teammate_df.copy()
    if "driver_name" not in df.columns:
        print("⚠️  teammate_df must have 'driver_name' column.")
        return
    df = df.sort_values("career_race_h2h_pct", ascending=False)
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width / 2, df.get("career_qual_h2h_pct", 50),
                width, label="Qualifying H2H %", color="#00D2BE", alpha=0.85)
    b2 = ax.bar(x + width / 2, df.get("career_race_h2h_pct", 50),
                width, label="Race H2H %", color="#DC0000", alpha=0.85)
    ax.axhline(50, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["driver_name"], rotation=30, ha="right")
    ax.set_ylabel("Head-to-Head Win % vs Teammates")
    ax.set_title("GOAT Candidates — Teammate Comparison (H2H %)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Teammate comparison chart generated.")


# ---------------------------------------------------------------------------
# 12. Constructor Dominance Timeline
# ---------------------------------------------------------------------------
def plot_constructor_dominance(master_df: pd.DataFrame, save_path: str = None):
    """
    Stacked area / line chart showing constructor championship wins per era.
    Each era shows which team dominated.
    """
    df = master_df.copy()
    if "constructorName" not in df.columns and "name_constructor" in df.columns:
        df["constructorName"] = df["name_constructor"]
    df["is_win"] = (df["positionOrder"] == 1).astype(int)
    top_teams = df.groupby("constructorName")["is_win"].sum().nlargest(10).index.tolist()
    df_top = df[df["constructorName"].isin(top_teams)]
    season_wins = df_top.groupby(["year", "constructorName"])["is_win"].sum().reset_index()
    pivot = season_wins.pivot(index="year", columns="constructorName", values="is_win").fillna(0)
    colors = [get_team_color(t) for t in pivot.columns]
    fig, ax = plt.subplots(figsize=(16, 7))
    pivot.plot(kind="area", stacked=True, ax=ax, color=colors, alpha=0.75)
    ax.set_title("Constructor Dominance Timeline (1950–2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Race Wins")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # Era markers
    for era_year, era_label in [(1966, "3L"), (1983, "Turbo"), (1994, "V10"),
                                 (2006, "V8"), (2014, "Hybrid"), (2022, "GE")]:
        ax.axvline(era_year, color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(era_year + 0.2, ax.get_ylim()[1] * 0.9, era_label, fontsize=7, color="white")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Constructor dominance timeline generated.")


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------
def generate_all_plots(master_df: pd.DataFrame, featured_df: pd.DataFrame,
                        goat_df: pd.DataFrame = None, model=None,
                        output_dir: str = "../outputs/plots/"):
    """Generate all 12 standard plots and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"📊 Generating all F1 ML visualizations → {output_dir}")
    # 8 — Win probability by grid (no model needed)
    plot_win_prob_by_grid(featured_df,
                         save_path=os.path.join(output_dir, "08_win_prob_by_grid.png"))
    # 9 — Career trajectories
    plot_career_trajectories(master_df,
                             save_path=os.path.join(output_dir, "09_career_trajectories.html"))
    # 12 — Constructor dominance
    plot_constructor_dominance(master_df,
                               save_path=os.path.join(output_dir, "12_constructor_dominance.png"))
    # 4 — GOAT radar (needs goat_df)
    if goat_df is not None:
        plot_goat_radar(goat_df, save_path=os.path.join(output_dir, "04_goat_radar.html"))
        plot_teammate_comparison(goat_df,
                                 save_path=os.path.join(output_dir, "11_teammate_comparison.png"))
    print(f"✅ Plots saved to {output_dir}")
