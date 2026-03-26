# 🏎️ Formula 1 Machine Learning Analytics
### Race Prediction, Hypothetical Scenarios & GOAT Analysis

A world-class Formula 1 machine learning project that answers 4 core predictive questions
and performs a definitive Greatest of All Time analysis across 75 years of racing data
(1950–2024).

---

## 🎯 What This Project Does

| Model | Question Answered |
|-------|------------------|
| **Race Winner** | *Who is likely to win the Australian Grand Prix?* |
| **Historical What-If** | *If the 2014 season happened today, who would win?* |
| **Constructor Championship** | *Will Mercedes win the constructor title?* |
| **Hypothetical Scenario** | *Can Vettel win in Alfa Romeo?* (20 configurable parameters) |
| **GOAT Analysis** | *Who is the greatest F1 driver of all time?* (5 scientific lenses) |

---

## 📊 Statistical Foundation

Prior statistical analysis on 26,000+ race records established the key predictors:

| Test | Result | Interpretation |
|------|--------|----------------|
| Pearson Correlation | r = 0.828, R² = 0.685, p < 0.001 | Grid position explains 68.5% of race finish variation |
| One-Way ANOVA | F = 47.3, p < 0.001, ω² = 0.31 | Constructor identity explains 31% of finish position |
| Multiple Regression | R² = 0.531, Grid β = −0.82 | 4 variables explain 53% of points scored per race |

These form the backbone of the ML feature set.

---

## 🗂️ Project Structure

```
F1_ML_Project/
├── data/
│   ├── README.md              ← How to download the Kaggle dataset
│   ├── raw/                   ← Place 14 Kaggle CSV files here
│   └── processed/             ← Generated: master_df.csv, featured_df.csv, etc.
├── notebooks/
│   ├── 01_data_prep.ipynb          ← Load & merge all 14 CSVs → master_df
│   ├── 02_feature_engineering.ipynb ← Compute ~80 ML features
│   ├── 03_model_race_winner.ipynb   ← XGBoost race winner + SHAP
│   ├── 04_model_constructor_champ.ipynb ← Constructor championship model
│   ├── 05_model_hypothetical.ipynb  ← 20-param hypothetical engine
│   ├── 06_goat_analysis.ipynb       ← 5-lens GOAT analysis
│   └── 07_visualizations.ipynb      ← All 12 publication-quality charts
├── src/
│   ├── __init__.py
│   ├── data_loader.py         ← Load & merge all 14 CSVs
│   ├── feature_engineering.py ← ~80 leakage-free ML features
│   ├── models.py              ← XGBoost/LightGBM/RF + Optuna tuning
│   ├── hypothetical_engine.py ← 20-parameter scenario engine
│   ├── goat_analysis.py       ← 5-lens GOAT scoring
│   └── visualizations.py      ← 12 publication-quality charts
├── models/                    ← Saved trained models (joblib)
├── outputs/plots/             ← Generated visualizations
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.12+
- ~4 GB RAM (for full dataset)

```bash
cd F1_ML_Project
pip install -r requirements.txt
```

### 2. Download the Dataset

See [`data/README.md`](data/README.md) for full instructions. Quick version:

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API credentials (see data/README.md)

# Download dataset
kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 \
    -p data/raw/ --unzip
```

**Dataset**: [Kaggle — Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)  
**Author**: rohanrao (based on [Ergast Motor Racing API](http://ergast.com/mrd/))

### 3. Run the Notebooks

Run in order (each notebook saves outputs for the next):

```bash
cd notebooks
jupyter notebook
```

| Step | Notebook | Runtime | Output |
|------|----------|---------|--------|
| 1 | `01_data_prep.ipynb` | ~2 min | `master_df.csv` |
| 2 | `02_feature_engineering.ipynb` | ~5 min | `featured_df.csv` |
| 3 | `03_model_race_winner.ipynb` | ~15 min | `race_winner_model.pkl` |
| 4 | `04_model_constructor_champ.ipynb` | ~3 min | `constructor_model.pkl` |
| 5 | `05_model_hypothetical.ipynb` | ~10 min | Scenario results |
| 6 | `06_goat_analysis.ipynb` | ~10 min | `goat_results.csv` |
| 7 | `07_visualizations.ipynb` | ~5 min | 12 charts in `outputs/plots/` |

---

## 🤖 Models

### Model 1 — Race Winner Prediction

- **Algorithm**: XGBoost (primary), LightGBM, Random Forest (comparison)
- **Target**: `is_win` (binary) + `positionOrder` (regression)
- **Training split**: Time-based — Train 2000–2021, Val 2022, Test 2023–2024 (**never random**)
- **Tuning**: Optuna with 50–100 trials, TimeSeriesSplit(5) cross-validation
- **Performance**: ROC-AUC ~0.92, Top-3 Accuracy ~0.75, Precision@1 ~0.35

```python
from src.models import predict_race_winner
import pandas as pd

result = predict_race_winner(featured_df, track="Australian Grand Prix", year=2023)
print(result)
# driver_name     team        grid  win_probability  predicted_position
# Max Verstappen  Red Bull       1            0.412                 1.2
# ...
```

### Model 2 — Constructor Championship

```python
from src.models import predict_constructor_championship
result = predict_constructor_championship(constructor_df, team="Red Bull", year=2023)
# {'team': 'Red Bull', 'predicted_position': 1, 'win_probability': 0.74, 'predicted_points': 860}
```

### Model 3 — Hypothetical Scenario Engine (20 parameters)

```python
from src.hypothetical_engine import hypothetical_scenario

result = hypothetical_scenario(
    featured_df         = featured_df,
    driver              = "Sebastian Vettel",
    team                = "Alfa Romeo",
    year                = 2022,
    reliability_factor  = 0.88,
    regulation_era      = "ground_effect",
    race_incidents      = True,
    teammate_skill_level= "midfield",
    n_monte_carlo       = 1000
)
# {'season_points_estimate': 73.4, 'season_wins_estimate': 0.2,
#  'championship_position_estimate': 8, 'confidence_interval': (28, 121)}
```

**All 20 parameters**: `driver`, `team`, `year`, `track`, `grid_position_override`,
`teammate`, `weather`, `num_pit_stops`, `reliability_factor`, `season_round`,
`driver_age_override`, `car_development_rate`, `tire_strategy`,
`starting_championship_points`, `driver_confidence_factor`, `regulation_era`,
`num_races_to_simulate`, `teammate_skill_level`, `race_incidents`, `home_race`

---

## 🏆 GOAT Analysis

Five scientific lenses determine the Greatest F1 Driver of All Time:

```
GOAT_Score = (0.20 × Raw Statistics)   +
             (0.30 × Teammate H2H)     +   ← Removes car advantage entirely
             (0.25 × Same-Car ML Test) +
             (0.15 × Adaptability)     +
             (0.10 × Dominance Index)
```

| Lens | Method | Key Insight |
|------|--------|-------------|
| **Raw Stats** | Era-normalize wins/poles by field size & season length | Compares across 9-pt and 25-pt eras |
| **Teammate H2H** | Qualifying and race finish vs same-car teammate | Purest skill metric — car advantage zeroed out |
| **Same-Car ML** | Hypothetical engine: each driver in median car | ML simulation across 22 races |
| **Adaptability** | Teams won with, regulation changes survived, circuit variety | Tests versatility |
| **Dominance** | Peak 3-season win rate, championship gaps, consecutive wins | Peak performance ceiling |

**Candidates**: Hamilton, Schumacher, Verstappen, Vettel, Prost, Senna, Fangio, Clark, Stewart, Lauda, Alonso

---

## 📊 Visualizations

12 publication-quality charts generated in `outputs/plots/`:

1. SHAP beeswarm summary (feature importance)
2. SHAP waterfall force plot (single prediction explanation)
3. Predicted vs Actual scatter (R² annotation)
4. **GOAT radar chart** (interactive Plotly, 5-axis)
5. **Hypothetical season animation** (interactive Plotly, cumulative points)
6. Feature importance bar chart (top 20, seaborn)
7. Confusion matrix heatmap
8. Win probability by grid position (validates r = 0.828 finding)
9. Era-adjusted career trajectories (interactive Plotly)
10. Monte Carlo distribution (1,000-iteration histogram)
11. Teammate comparison chart (GOAT H2H %)
12. Constructor dominance timeline (1950–2024)

---

## 🔬 ML Engineering Choices

| Decision | Choice | Reason |
|----------|--------|--------|
| Train/test split | **Time-based only** | Race data is temporal; random splits cause leakage |
| Rolling features | **shift(1) before aggregation** | Prevents current race result contaminating feature |
| Class imbalance | **scale_pos_weight** | ~5% win rate requires balancing |
| Hyperparameter tuning | **Optuna + TimeSeriesSplit** | Temporal CV respects time ordering |
| Missing data | **Indicator variables + median imputation** | Qualifying NaN pre-2003, lap times NaN pre-1996 |
| Target column | **positionOrder** (not position) | `position` has `\N` for DNFs; `positionOrder` is always 1–26 |
| Grid = 0 | **Flagged as `pit_lane_start`** | Not deleted; contains valid racing info |

---

## 📦 Dependencies

```
pandas>=2.0        numpy>=1.24        matplotlib>=3.7    seaborn>=0.12
scipy>=1.10        scikit-learn>=1.3  xgboost>=2.0       lightgbm>=4.0
optuna>=3.0        shap>=0.42         plotly>=5.15        joblib>=1.3
jupyter>=1.0       category_encoders>=2.6
```

---

## 🙏 Credits

- **Dataset**: [Kaggle — Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) by **rohanrao**
- **Data Source**: [Ergast Motor Racing Developer API](http://ergast.com/mrd/)
- **Statistical Analysis**: Pearson/ANOVA/Regression benchmarks from prior Tableau & JASP analysis
- **ML Framework**: XGBoost, LightGBM, Optuna, SHAP
