# ⚡ GridGuard — Full Project Analysis & Model Upgrade Report

> **Project purpose:** Predict delay (in months) and risk level (Low / Medium / High) for Indian power-grid transmission projects.
> **Dataset moving from:** `GridGuard_Dataset_2000.xlsx` → `GridGuard_Dataset_50000.csv`

---

## 1. 📂 File Map — What Every File Does

| File | Role | Status |
|------|------|--------|
| `train_final.py` | **Main training script** — loads data, engineers features, trains classifier + regressor, saves `.pkl` | ✅ Active — needs update for CSV |
| `custom_model.py` | Defines `CustomGridGuardClassifier` (wraps `GradientBoostingClassifier`) | ✅ Active — referenced by train + predict |
| `feature_engineering.py` | Shared feature engineering module — creates 10 derived features from raw columns | ✅ Active — used by both train and predict |
| `predict.py` | Loads saved `.pkl`, runs inference on single or batch projects; also demo runner | ✅ Active — used by app |
| `app.py` | Streamlit web UI — form inputs → calls `predict_project()` → shows results | ✅ Active |
| `preprocess_data.py` | Old script for `GridGuard_Hybrid_Dataset.csv` with TF-IDF on vendor remarks | ❌ **USELESS** — completely different pipeline, not connected to anything |
| `generate_synthetic_logs.py` | Generates `GridGuard_Hybrid_Dataset.csv` from `construction_project_dataset.csv` | ❌ **USELESS** — old experiment, not used |
| `gridguard_best_model.pkl` | Saved trained model bundle (regressor + classifier + scaler + metadata) | ⚠️ Stale — trained on 2000-row XLSX, must retrain |

### Dataset Files

| File | Size | Used? | Notes |
|------|------|-------|-------|
| `GridGuard_Dataset_2000.xlsx` | 2 000 rows | ⚠️ Currently used | Old, smaller |
| `GridGuard_Dataset_50000.csv` | **50 000 rows** | 🎯 **Target** | Has `Time_Elapsed_Months` + `Site_Engineer_Remark` |
| `GridGuard_Hybrid_Dataset.csv` | ~16 MB | ❌ Unused | Old experiment |
| `construction_dataset.csv` | Small | ❌ Unused | Old experiment |
| `construction_project_dataset.csv` | ~12 MB | ❌ Unused | Source for hybrid dataset |
| `construction_project_performance_dataset.csv` | ~2 MB | ❌ Unused | Old experiment |

---

## 2. 🤖 Current Model Being Used

### Classifier (Risk Level: Low / Medium / High)
- **Algorithm:** `GradientBoostingClassifier` (sklearn)
- **Wrapped in:** `CustomGridGuardClassifier` class in `custom_model.py`
- **Hyperparams:** `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `min_samples_leaf=8`, `subsample=0.8`
- **Class balancing:** `compute_sample_weight("balanced")` — handles imbalanced classes

### Regressor (Delay in Months)
- **Algorithm:** `GradientBoostingRegressor` (sklearn)
- **Tuned via:** `GridSearchCV` with `n_estimators=[200,400]`, `max_depth=[3,5]`, `learning_rate=[0.05,0.1]`
- **Objective:** Minimize MAE (Mean Absolute Error)

### Feature Engineering (10 features added on top of raw 9 columns)
| Feature | What it represents |
|---|---|
| `budget_per_km` | Budget intensity ₹Cr/km |
| `progress_rate` | % progress per planned month |
| `remaining_work_ratio` | Remaining work pressure |
| `land_risk_score` | Ordinal (Clear=0, Pending=1, Disputed=2) |
| `forest_risk_score` | Ordinal (Approved=0, Stage-II=1, Stage-I=2) |
| `vendor_risk_score` | Ordinal (On Track=0, Delayed=1, Insolvent=2) |
| `composite_risk_score` | Sum of 3 risk scores above |
| `schedule_adherence` | Time elapsed / planned duration |
| `progress_deficit` | Expected % - Actual % |
| `months_behind_schedule` | Months already lost |

---

## 3. 📊 Dataset Comparison: XLSX vs CSV

| Field | 2000.xlsx | 50000.csv | Notes |
|-------|-----------|-----------|-------|
| `Project_ID` | ❓ | ✅ `GRID-XXXXX` | New — skip for training |
| `Project_Type` | ✅ (4 types incl. `400kV Transmission Line`) | ✅ (3 types — **no `400kV Transmission Line`**) | ⚠️ One type removed |
| `Region` | ✅ 5 regions | ✅ same 5 | OK |
| `Budget_Cr` | ✅ | ✅ | OK |
| `Line_Length_CKM` | ✅ | ✅ | OK |
| `Planned_Duration_Months` | ✅ | ✅ | OK |
| `Physical_Progress_Pct` | ✅ | ✅ | OK |
| `Land_RoW_Status` | ✅ | ✅ | OK |
| `Forest_Clearance_Status` | ✅ | ✅ | OK |
| `Vendor_Status` | ✅ | ✅ | OK |
| `Site_Engineer_Remark` | ❌ absent | ✅ **NEW** | Rich text — can use for NLP |
| `Actual_Delay_Months` | ✅ | ✅ | Regression target |
| `Risk_Level` | ✅ | ✅ | Classification target |
| `months_elapsed` | ❌ — was **synthesized** in train script | = `Time_Elapsed_Months` → **actually in dataset!** | ✅ Real data now |

### Key Differences (What Must Change)
1. `skiprows=2` must be removed (CSV has no header rows to skip)
2. `pd.read_excel()` → `pd.read_csv()`
3. `months_elapsed` synthesis block removed — use `Time_Elapsed_Months` directly
4. `400kV Transmission Line` no longer in data — remove from `app.py` dropdown
5. `Site_Engineer_Remark` is a new TEXT column — can optionally use for NLP boost

---

## 4. 🧪 Models to Test & Expected Accuracy

Recommendation for testing multiple models on the **50K dataset** (classification task: Low/Medium/High risk):

| # | Model | Why Test It | Expected Accuracy |
|---|-------|-------------|-------------------|
| 1 | **GradientBoostingClassifier** *(current)* | Baseline — already implemented | ~85-90% |
| 2 | **Random Forest** | Fast, robust, less prone to overfit, great on tabular | ~83-88% |
| 3 | **XGBoost** | Industry gold standard for tabular classification | **~88-93%** |
| 4 | **LightGBM** | Fastest on 50K rows, handles imbalance well | **~88-93%** |
| 5 | **CatBoost** | Best for data with many categoricals | ~87-92% |
| 6 | **Logistic Regression** | Sanity check / interpretable baseline | ~65-75% |
| 7 | **SVM (RBF kernel)** | Good on scaled features but slow on 50K | ~75-82% |

### My Recommendation
> **Use LightGBM as primary model.** Reasons:
> - 50K rows → GBM variants shine at this scale
> - `class_weight` / `is_unbalance` handles the severe Low/Med/High imbalance (772:211:17 ratio in sample)
> - 10-50x faster training than sklearn GBM
> - `num_leaves`, `min_data_in_leaf` easy to tune
> - Can optionally include TF-IDF of `Site_Engineer_Remark` as extra features

---

## 5. 🚀 Action Plan — Step by Step

### Step 1: Install dependencies
```bash
pip install lightgbm xgboost catboost scikit-learn pandas openpyxl streamlit
```

### Step 2: Run `train_compare.py` (NEW file to create)
This will train **6 models** simultaneously and print an accuracy comparison table.

### Step 3: Retrain `train_final.py` with best model + new dataset

### Step 4: Run `app.py` to test the UI with the new model
```bash
streamlit run app.py
```

### Step 5: (Optional) Add NLP features from `Site_Engineer_Remark`

---

## 6. ⚠️ Files That Are Useless

| File | Why Useless |
|------|-------------|
| `preprocess_data.py` | Uses a completely different dataset (`GridGuard_Hybrid_Dataset.csv`) and different feature set. Not connected to the main pipeline at all. **Can be deleted.** |
| `generate_synthetic_logs.py` | Generates the hybrid dataset from `construction_project_dataset.csv`. It's a one-off data generation script that is no longer needed. **Can be deleted.** |
| `dataset/GridGuard_Hybrid_Dataset.csv` | Output of the above script. Not used. |
| `dataset/construction_dataset.csv` | Old construction dataset, not used |
| `dataset/construction_project_dataset.csv` | Old construction dataset, not used |
| `dataset/construction_project_performance_dataset.csv` | Old construction dataset, not used |
