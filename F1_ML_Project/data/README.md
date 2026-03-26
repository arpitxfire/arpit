# F1 Dataset Download Instructions

## Primary Dataset: Kaggle Formula 1 World Championship (1950–2024)

### Step 1: Install Kaggle CLI

```bash
pip install kaggle
```

### Step 2: Set Up Kaggle API Credentials

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section and click **"Create New API Token"**
3. This downloads a `kaggle.json` file. Place it at:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Step 3: Download the Dataset

```bash
# Download to the raw data directory
kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 -p F1_ML_Project/data/raw/ --unzip
```

**Dataset URL**: [https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

**Author**: rohanrao (based on Ergast Motor Racing Developer API)

### Step 4: Verify Files

After downloading, the `F1_ML_Project/data/raw/` directory should contain these 14 CSV files:

```
circuits.csv
constructor_results.csv
constructor_standings.csv
constructors.csv
driver_standings.csv
drivers.csv
lap_times.csv
pit_stops.csv
qualifying.csv
races.csv
results.csv
seasons.csv
sprint_results.csv
status.csv
```

---

## Alternative: Manual Download

1. Visit: [https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
2. Click **"Download"** (requires free Kaggle account)
3. Unzip the downloaded file into `F1_ML_Project/data/raw/`

---

## Supplementary GitHub Datasets

Some additional F1 data is available from:
- **Ergast API Mirror**: [https://github.com/nicholasgasior/gsfour-f1-data](https://github.com/nicholasgasior/gsfour-f1-data)
- **FastF1 Python Library** (for recent race telemetry): `pip install fastf1`

---

## Data Notes

| File | Coverage | Notes |
|------|----------|-------|
| results.csv | 1950–2024 | MASTER TABLE. `position` column has `\N` for DNFs — always use `positionOrder` |
| qualifying.csv | 2003–2024 | Only exists post-2003 |
| lap_times.csv | 1996–2024 | Only exists post-1996 |
| pit_stops.csv | 2012–2024 | Official pit stop data starts 2012 |
| sprint_results.csv | 2021–2024 | Sprint format introduced in 2021 |

---

## Data Path Configuration

The default data path used in all notebooks is `../data/raw/`. You can change this by setting the `DATA_PATH` variable at the top of each notebook:

```python
DATA_PATH = "../data/raw/"  # Change this to your actual path
```
