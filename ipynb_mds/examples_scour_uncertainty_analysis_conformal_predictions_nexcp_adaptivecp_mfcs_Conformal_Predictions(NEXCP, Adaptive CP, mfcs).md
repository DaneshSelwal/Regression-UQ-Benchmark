# **Start Section:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
os.environ["PIP_CONSTRAINT"] = "/tmp/numpy_constraint.txt"
!echo "numpy==1.26.4" > /tmp/numpy_constraint.txt

!pip uninstall -y torch torchaudio torchvision \
    torchao torchcodec torchdata torchtune torchsummary -q 2>/dev/null

!pip uninstall -y tensorflow tensorflow-text tensorflow-hub tf-keras \
    tensorflow_decision_forests tensorflow-probability \
    tensorflow-Datasets tensorflow-metadata -q 2>/dev/null

!pip uninstall -y numpy scikit-learn shap xgboost lightgbm dask \
    seaborn plotly openpyxl Cython catboost interpret lime -q 2>/dev/null

!pip install numpy==1.26.4 -q
!pip install scikit-learn==1.6.1 -q
!pip install torch==2.9.0 -q

!pip install lightgbm==4.6.0 -q
!pip install xgboost==3.1.2 -q
!pip install catboost==1.2.8 -q
!pip install gpboost==1.6.1 -q
!pip install ngboost==0.5.8 -q
!pip install pgbm==2.2.0 -q
!pip install pytorch-tabnet2==4.5.3 -q

!pip install bayesian-optimization==3.2.0 -q
!pip install optuna==4.6.0 -q
!pip install optunahub==0.4.0 -q
!pip install cmaes==0.12.0 -q

!pip install shap==0.44.0 -q
!pip install lime==0.2.0.1 -q
!pip install interpret==0.7.4 -q

!pip install mapie==0.6.0 -q
!pip install puncc==0.8.0 -q
!pip install skorch==1.3.1 -q
!pip install properscoring==0.1 -q

!pip install dask[dataframe]==2025.12.0 -q
!pip install cython==3.0.12 -q
!pip install seaborn==0.13.2 -q
!pip install plotly==5.24.1 -q
!pip install kaleido==1.2.0 -q
!pip install openpyxl==3.1.5 -q
!pip install XlsxWriter==3.2.9 -q
!pip install cp==2020.12.3 -q

!pip install numpy==1.26.4 --force-reinstall --no-deps -q

os._exit(0)
```

# **Imports**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
import ngboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from gpboost import GPBoostRegressor
from ngboost import NGBRegressor
import optuna
import optunahub
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from interpret import show
from interpret.blackbox import LimeTabular, ShapKernel
from optuna.samplers import RandomSampler
import random
import time
from ngboost.distns import Normal
from ngboost.scores import LogScore
from scipy.stats import norm
from optuna.samplers import BaseSampler
from optuna.samplers import GridSampler
from optuna.samplers import TPESampler
from optuna.samplers import PartialFixedSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import QMCSampler
from optuna.samplers import NSGAIIISampler
from optuna.samplers import NSGAIISampler
from optuna.samplers import BruteForceSampler
from optuna.samplers import GPSampler
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
import plotly.express as px
from io import BytesIO
from openpyxl import Workbook, load_workbook
import os
import properscoring as ps
import io
from openpyxl.drawing.image import Image as openpyxlImage
import warnings
import xlsxwriter
from openpyxl.drawing.image import Image
from pgbm.sklearn import HistGradientBoostingRegressor
import torch
from pgbm.torch import PGBM
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
import pickle
import json
from mapie.subsample import Subsample
from mapie.regression import MapieRegressor
from deel.puncc.metrics import regression_sharpness, regression_mean_coverage
from deel.puncc.api.prediction import BasePredictor, DualPredictor
from deel.puncc.regression import SplitCP, CVPlus, CQR
from deel.puncc.plotting import plot_prediction_intervals
from sklearn.model_selection import train_test_split
from typing_extensions import TypedDict
from typing import Union
from mapie.metrics import regression_coverage_score
from sklearn.model_selection import KFold
from PIL import Image as PImage
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.stats import genpareto
from pytorch_tabnet import TabNetRegressor
import torch.nn as nn
import warnings
import io
import time
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
```

```python
# Go to find & replace button and replace (scour_uncertainty_analysis) with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (str) with actual data label name.
```

```python
feature_names = ['Ps', 'Pw', ' Skew  ', ' Velocity  ', ' Depth  ', ' D50  ', ' Gradation  ']
```

```python
train_data_path = "./drive/MyDrive/Regression/scour_uncertainty_analysis/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/scour_uncertainty_analysis/data/test.csv"
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
print("Training data loaded successfully.")
print("Test data loaded successfully.")
```

```python
print("\nShape of training data:", train_data.shape)
print("First 5 rows of training data:\n", train_data.head(5))
print("\nShape of test data:", test_data.shape)
print("First 5 rows of test data:\n", test_data.head(5))
```

```python
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
x_test= X_test
print("\nShape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
```

```python
# Apply z-score normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the first five rows of the normalized data
print("\nFirst five rows of normalized X_train:")
print(X_train[:5])

print("\nFirst five rows of normalized X_test:")
print(X_test[:5])
```

# **Functions:**

```python
# Define the model classes
model_classes = {
    'Random Forest': RandomForestRegressor,
    'Gradient Boosting': GradientBoostingRegressor,
    'XGBoost': XGBRegressor,
    'LightGBM': LGBMRegressor,
    'GPBoost': GPBoostRegressor,
    'CatBoost': CatBoostRegressor,
    'HistGradientBoosting': HistGradientBoostingRegressor,
    'TabNet': TabNetRegressor,
    'NGBoost': NGBRegressor
}

def _ensure_excel_file(path):
    import os
    import pandas as pd
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pd.DataFrame().to_excel(path)
    return path
```

```python
def get_best_model_params(results, model_name):
    # Map model names to dictionary keys, assuming keys are strings like 'XGBoost' and not objects
    model_keys = {
        'LightGBM': 'LightGBM',
        'XGBoost': 'XGBoost',
        'GPBoost': 'GPBoost',
        'NGBoost': 'NGBoost',
        'GradientBoosting': 'Gradient Boosting',
        'HGBR' : 'HistGradientBoosting',
        'CatBoost' : 'CatBoost',
        'TabNet' : 'TabNet',
        'PGBM' : 'PGBM'
    }

    # Ensure the requested model name is valid
    if model_name not in model_keys:
        raise ValueError(f"Model name '{model_name}' is not recognized. Available models are: {list(model_keys.keys())}")

    # Filter out entries for the specified model
    model_entries = {key: value for key, value in results.items() if key[0] == model_keys[model_name]}

    # Find the entry with the best (lowest) 'best_score'
    best_entry_key, best_entry_value = min(model_entries.items(), key=lambda item: item[1]['best_score'])

    # Return the best hyperparameters
    return best_entry_value['best_params']
```

```python
class validate:
    @staticmethod
    def coverage(int_pred, y_test):
        return {strat_name: np.mean((y_test.to_numpy() >= int_pred[strat_name][:, 0]) & (y_test.to_numpy() <= int_pred[strat_name][:, 1])) for strat_name in int_pred}
    @staticmethod
    def width(int_pred): # Corrected: Removed y_test argument as it's not defined and not used
        return {strat_name: np.mean(int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]) for strat_name in int_pred}
    @staticmethod
    def rmse(y_pred, y_test):
        return {strat_name: np.sqrt(mean_squared_error(y_test, y_pred[strat_name])) for strat_name in y_pred}
    @staticmethod
    def cwc(int_pred, y_test, miscoverage):
        widths = validate.width(int_pred)
        coverages = validate.coverage(int_pred, y_test)
        return {strat_name: widths[strat_name] * (1 + (coverages[strat_name] < (1 - miscoverage)) * (1 - coverages[strat_name])) for strat_name in int_pred}

    @staticmethod
    def cond_coverage(int_pred, y_test, num_bins=10):
        cond_coverages = {}
        for strat_name in int_pred:
            widths = int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]
            bin_edges = np.histogram_bin_edges(widths, bins=num_bins)
            digitized_widths = np.digitize(widths, bin_edges)
            strategy_cond_coverage = []
            for bin_num in range(1, num_bins + 1):
                indices_in_bin = np.where(digitized_widths == bin_num)[0]
                if len(indices_in_bin) > 0:
                    bin_coverage = np.mean((y_test.to_numpy()[indices_in_bin] >= int_pred[strat_name][indices_in_bin, 0]) & (y_test.to_numpy()[indices_in_bin] <= int_pred[strat_name][indices_in_bin, 1]))
                    strategy_cond_coverage.append(bin_coverage)
                else:
                    strategy_cond_coverage.append(np.nan) # or handle empty bins as needed
            cond_coverages[strat_name] = strategy_cond_coverage
        return cond_coverages


class visualize:
    @staticmethod
    def coverage(int_pred, y_test):
        return {strat_name: np.mean((y_test.to_numpy() >= int_pred[strat_name][:, 0]) & (y_test.to_numpy() <= int_pred[strat_name][:, 1])) for strat_name in int_pred}
    @staticmethod
    def width(int_pred):
        return {strat_name: np.mean(int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]) for strat_name in int_pred}
    @staticmethod
    def rmse(y_pred, y_test):
        return {strat_name: np.sqrt(mean_squared_error(y_test, y_pred[strat_name])) for strat_name in y_pred}
    @staticmethod
    def cwc(int_pred, y_test, miscoverage):
        widths = visualize.width(int_pred)
        coverages = visualize.coverage(int_pred, y_test)
        return {strat_name: widths[strat_name] * (1 + (coverages[strat_name] < (1 - miscoverage)) * (1 - coverages[strat_name])) for strat_name in int_pred}
    @staticmethod
    def goodness(y_true, y_pred, y_pred_low, y_pred_up, coverage, width, rmse, cwc, ax=None, title="Goodness Plot"):
        if ax is None:
            fig, ax = plt.subplots()
        n_samples = len(y_true)
        y_true_np = y_true.to_numpy().ravel() # Ensure y_true is numpy array and flattened

        # Determine points inside and outside the interval
        inside_interval = (y_true_np >= y_pred_low) & (y_true_np <= y_pred_up)
        outside_interval = ~inside_interval

        # Plot points inside the interval in green
        ax.scatter(y_true_np[inside_interval], y_pred[inside_interval], color='green', s=10, label='Inside Interval')
        # Plot points outside the interval in orange
        ax.scatter(y_true_np[outside_interval], y_pred[outside_interval], color='orange', s=10, label='Outside Interval')

        # Plot all ground truth values as blue crosses
        ax.scatter(y_true_np, y_true_np, color='blue', marker='x', s=30, linewidths=0.7, label='Ground Truth Values')


        ax.vlines(y_true_np, y_pred_low, y_pred_up, color='gray', alpha=0.5, label='Prediction Intervals')
        ax.plot([min(y_true_np), max(y_true_np)], [min(y_true_np), max(y_true_np)], linestyle='--', color='blue', label='Ideal Prediction Line') # Changed label for clarity
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.set_title(f'{title}\nCoverage: {coverage*100:.2f}%, Width: {width:.2f}, RMSE: {rmse:.2f}, CWC: {cwc:.2f}')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax

    @staticmethod
    def width_size_occurrence(int_pred, train_intervals, num_bins=10, ax=None, x_lim=None, title="Width Size Occurrence"):
        if ax is None:
            fig, ax = plt.subplots()

        # Calculate interval widths for test and train sets
        widths_test = np.abs(int_pred[:, 1] - int_pred[:, 0])
        widths_train = np.abs(train_intervals[:, 1] - train_intervals[:, 0])

        # Plot histograms for both train and test widths
        ax.hist(widths_test, bins=num_bins, alpha=0.6, label='Test Interval Widths', color='red')
        ax.hist(widths_train, bins=num_bins, alpha=0.4, label='Train Interval Widths', color='blue')


        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel('Interval Width')
        ax.set_ylabel('Occurrence (Frequency)')
        ax.set_title(f'{title} - Interval Width Occurrence')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax

    @staticmethod
    def coverage_by_width(y_test, int_pred, miscoverage, cond_coverages, num_bins=10, ax=None, title="Coverage vs Width"):
        if ax is None:
            fig, ax = plt.subplots()

        widths = int_pred[:, 1] - int_pred[:, 0]
        bin_edges = np.histogram_bin_edges(widths, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        ax.bar(bin_centers, cond_coverages, width=bin_width, color='blue', alpha=0.7, edgecolor='black', linewidth=0.8, label='Conditional Coverage')
        ax.axhline(1 - miscoverage, color='red', linestyle='--', label=f'Nominal Coverage ({(1 - miscoverage)*100:.0f}%)')

        ax.set_xlabel('Interval Width')
        ax.set_ylabel('Conditional Coverage')
        ax.set_title(f'{title} - Coverage vs Interval Width')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax


def weighted_quantile(values, weights, q):
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cum_weights = np.cumsum(weights)
    cum_weights = cum_weights / cum_weights[-1]
    return np.interp(q, cum_weights, values)
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_NEXCP(
    model_class,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path=None,
    alpha=0.05,
    decay=0.99,
    n_folds=5,
    n_bootstrap=50,
):

    warnings.filterwarnings("ignore")

    SEED = 123
    np.random.seed(SEED)

    strategies_names = {
        "NexCP-Split": "NexCP Split",
        "NexCP-Full": "NexCP Full",
        "NexCP-J+aB": "NexCP Jackknife+ aB",
        "NexCP-CV+": "NexCP CV+",
    }

    y_pred, int_pred = {}, {}

    # =========================================================
    # NexCP – Split
    # =========================================================
    model = model_class(**best_params)
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    res = np.abs(y_train.to_numpy() - yhat_train)
    weights = decay ** np.arange(len(res) - 1, -1, -1)
    q = weighted_quantile(res, weights, 1 - alpha)

    y_pred["NexCP-Split"] = yhat_test
    int_pred["NexCP-Split"] = np.c_[yhat_test - q, yhat_test + q]

    # =========================================================
    # NexCP – Full
    # =========================================================
    model = model_class(**best_params)
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    q = np.quantile(np.abs(y_train.to_numpy() - yhat_train), 1 - alpha)

    y_pred["NexCP-Full"] = yhat_test
    int_pred["NexCP-Full"] = np.c_[yhat_test - q, yhat_test + q]

    # =========================================================
    # NexCP – Jackknife+ after Bootstrap
    # =========================================================
    rng = np.random.default_rng(SEED)
    lowers_t, uppers_t, preds_t = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_train), len(X_train), replace=True)

        model = model_class(**best_params)
        model.fit(X_train[idx], y_train.iloc[idx])

        yte = model.predict(X_test)

        q = np.quantile(
            np.abs(y_train.iloc[idx].to_numpy() - model.predict(X_train[idx])),
            1 - alpha,
        )

        lowers_t.append(yte - q)
        uppers_t.append(yte + q)
        preds_t.append(yte)

    y_pred["NexCP-J+aB"] = np.mean(preds_t, axis=0)
    int_pred["NexCP-J+aB"] = np.c_[np.min(lowers_t, axis=0), np.max(uppers_t, axis=0)]

    # =========================================================
    # NexCP – CV+
    # =========================================================
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    lowers_t, uppers_t, preds_t = [], [], []

    for tr, cal in kf.split(X_train):
        model = model_class(**best_params)
        model.fit(X_train[tr], y_train.iloc[tr])

        y_cal = model.predict(X_train[cal])
        q = np.quantile(np.abs(y_train.iloc[cal].to_numpy() - y_cal), 1 - alpha)

        yte = model.predict(X_test)

        lowers_t.append(yte - q)
        uppers_t.append(yte + q)
        preds_t.append(yte)

    y_pred["NexCP-CV+"] = np.mean(preds_t, axis=0)
    int_pred["NexCP-CV+"] = np.c_[np.mean(lowers_t, axis=0), np.mean(uppers_t, axis=0)]

    # =========================================================
    # Metrics
    # =========================================================
    def coverage(y, l, u):
        return np.mean((y >= l) & (y <= u))

    def width(l, u):
        return np.mean(u - l)

    def cwc(l, u, y, alpha):
        cov = coverage(y, l, u)
        w = width(l, u)
        penalty = np.exp(-50 * (cov - (1 - alpha)))
        return w * (1 + penalty if cov < (1 - alpha) else 1)

    y_true = y_test.to_numpy()

    summary_rows = []

    print("\n===== NEXCP RESULTS =====")

    for s in strategies_names:
        cov = coverage(y_true, int_pred[s][:, 0], int_pred[s][:, 1])
        wid = width(int_pred[s][:, 0], int_pred[s][:, 1])
        rmse = np.sqrt(mean_squared_error(y_true, y_pred[s]))
        cwc_val = cwc(int_pred[s][:, 0], int_pred[s][:, 1], y_true, alpha)

        summary_rows.append([
            strategies_names[s],
            cov,
            wid,
            rmse,
            cwc_val,
        ])

        print(f"{strategies_names[s]} - Coverage: {cov*100:.3f}% | Width: {wid:.3f}")

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["Strategy", "Coverage", "Width", "RMSE", "CWC"],
    )

    # =========================================================
    # Plotting (SAVE ONLY — NO SHOW)
    # =========================================================
    idx = np.arange(len(y_test))
    plot_buffers = {}

    for s in strategies_names:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.scatter(idx, y_test, color="blue", s=10, alpha=0.7, label="True")
        ax.fill_between(
            idx,
            int_pred[s][:, 0],
            int_pred[s][:, 1],
            color="gray",
            alpha=0.5,
            label="Prediction interval",
        )
        ax.plot(idx, y_pred[s], color="red", linewidth=1, label="Predicted mean")

        ax.set_title(strategies_names[s])
        ax.set_xlabel("Sample Number")
        ax.set_ylabel("str")
        ax.legend(loc="upper left")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plot_buffers[s] = buf

        plt.close(fig)

    # =========================================================
    # Excel Export
    # =========================================================
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        # ---- Summary sheet ----
        sheet_name = "NEXCP_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name, 0)
        for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c_idx, val in enumerate(row, 1):
                ws.cell(row=r_idx, column=c_idx, value=val)

        # ---- Plot sheets ----
        for s, buf in plot_buffers.items():
            img = Image(buf)

            sheet_name = f"NEXCP_Plot_{s}"
            base = sheet_name
            i = 1
            while sheet_name in wb.sheetnames:
                sheet_name = f"{base}_{i}"
                i += 1

            ws = wb.create_sheet(sheet_name)
            ws.add_image(img, "A1")

        # ---- Values sheet ----
        results_dict = {}
        for s in strategies_names:
            results_dict[f"{s}_true"] = y_true
            results_dict[f"{s}_pred"] = y_pred[s]
            results_dict[f"{s}_lower"] = int_pred[s][:, 0]
            results_dict[f"{s}_upper"] = int_pred[s][:, 1]

        df_vals = pd.DataFrame(results_dict)

        sheet_name = "NEXCP_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r_idx, row in enumerate(dataframe_to_rows(df_vals, index=False, header=True), 1):
            for c_idx, val in enumerate(row, 1):
                ws.cell(row=r_idx, column=c_idx, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))

    return summary_df

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_online_super(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path=None,
    alpha=0.05,
    strategies="all",
    window_min=50,
    window_max=500,
    window_grid=(25, 50, 100, 200, 400),
    n_bootstrap=30,
):

    warnings.filterwarnings("ignore")

    SEED = 123
    np.random.seed(SEED)

    STRATEGY_NAMES = {
        "Online-Split": "Online Split",
        "Online-CV+": "Online CV+",
        "Online-J+aB": "Online Jackknife+ aB",
        "SAOCP": "SAOCP",
        "SF-OGD": "SF-OGD",
        "FACI": "FACI",
        "MFCS-Split": "MFCS Split",
        "MFCS-Full": "MFCS Full",
    }

    ALL_STRATEGIES = list(STRATEGY_NAMES.keys())
    if strategies == "all":
        strategies = ALL_STRATEGIES

    # =========================================================
    # Base model
    # =========================================================
    base_model = model_class(**model_params)
    base_model.fit(X_train, y_train)

    yhat_train = base_model.predict(X_train)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train.to_numpy(), y_test.to_numpy()])

    n_train = len(X_train)
    T = len(y_all)

    y_hat_all = np.zeros(T)
    y_hat_all[:n_train] = yhat_train
    for t in range(n_train, T):
        y_hat_all[t] = base_model.predict(X_all[t:t + 1])[0]

    residuals = np.abs(y_all - y_hat_all)

    y_pred, int_pred = {}, {}

    # =========================================================
    # Online methods (UNCHANGED LOGIC)
    # =========================================================
    if "Online-Split" in strategies:
        q = np.quantile(residuals[:n_train], 1 - alpha)
        y_pred["Online-Split"] = y_hat_all[n_train:]
        int_pred["Online-Split"] = np.c_[y_hat_all[n_train:] - q,
                                         y_hat_all[n_train:] + q]

    if "Online-CV+" in strategies:
        lower, upper = [], []
        for t in range(n_train, T):
            q = np.quantile(residuals[:t], 1 - alpha)
            lower.append(y_hat_all[t] - q)
            upper.append(y_hat_all[t] + q)
        y_pred["Online-CV+"] = y_hat_all[n_train:]
        int_pred["Online-CV+"] = np.c_[lower, upper]

    if "Online-J+aB" in strategies:
        lowers, uppers, preds = [], [], []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_train, n_train, replace=True)
            m = model_class(**model_params)
            m.fit(X_train[idx], y_train.iloc[idx])
            res = np.abs(y_train.iloc[idx] - m.predict(X_train[idx]))
            q = np.quantile(res, 1 - alpha)
            p = m.predict(X_test)
            preds.append(p)
            lowers.append(p - q)
            uppers.append(p + q)
        y_pred["Online-J+aB"] = np.mean(preds, axis=0)
        int_pred["Online-J+aB"] = np.c_[np.min(lowers, axis=0),
                                        np.max(uppers, axis=0)]

    if "SAOCP" in strategies:
        lower, upper = [], []
        for t in range(n_train, T):
            w = min(max(window_min, int(t / 2)), window_max)
            start = max(0, t - w)
            q = np.quantile(residuals[start:t], 1 - alpha)
            lower.append(y_hat_all[t] - q)
            upper.append(y_hat_all[t] + q)
        y_pred["SAOCP"] = y_hat_all[n_train:]
        int_pred["SAOCP"] = np.c_[lower, upper]

    if "SF-OGD" in strategies:
        q_t = np.quantile(residuals[:n_train], 1 - alpha)
        lower, upper = [], []
        for t in range(n_train, T):
            r_t = residuals[t]
            grad = alpha - (r_t <= q_t)
            eta = 1.0 / np.sqrt(t - n_train + 1)
            q_t = max(0.0, q_t - eta * grad)
            lower.append(y_hat_all[t] - q_t)
            upper.append(y_hat_all[t] + q_t)
        y_pred["SF-OGD"] = y_hat_all[n_train:]
        int_pred["SF-OGD"] = np.c_[lower, upper]

    if "FACI" in strategies:
        windows = list(window_grid)
        losses = np.zeros(len(windows))
        lower, upper = [], []
        for t in range(n_train, T):
            r_t = residuals[t]
            qs = []
            for i, w in enumerate(windows):
                start = max(0, t - w)
                q = np.quantile(residuals[start:t], 1 - alpha)
                qs.append(q)
                losses[i] += alpha * q + (r_t - q) * (r_t > q)
            q_star = qs[np.argmin(losses)]
            lower.append(y_hat_all[t] - q_star)
            upper.append(y_hat_all[t] + q_star)
        y_pred["FACI"] = y_hat_all[n_train:]
        int_pred["FACI"] = np.c_[lower, upper]

    if "MFCS-Split" in strategies:
        q = np.quantile(residuals[:n_train], 1 - alpha)
        y_pred["MFCS-Split"] = y_hat_all[n_train:]
        int_pred["MFCS-Split"] = np.c_[y_hat_all[n_train:] - q,
                                       y_hat_all[n_train:] + q]

    if "MFCS-Full" in strategies:
        lower, upper, preds = [], [], []
        for t in range(n_train, T):
            m = model_class(**model_params)
            m.fit(X_all[:t], y_all[:t])
            yhat = m.predict(X_all[t:t + 1])[0]
            preds.append(yhat)
            res = np.abs(y_all[:t] - m.predict(X_all[:t]))
            q = np.quantile(res, 1 - alpha)
            lower.append(yhat - q)
            upper.append(yhat + q)
        y_pred["MFCS-Full"] = np.array(preds)
        int_pred["MFCS-Full"] = np.c_[lower, upper]

    # =========================================================
    # METRICS
    # =========================================================
    y_true = y_test.to_numpy().ravel()

    rows = []
    for s in y_pred:
        ymin, ymax = int_pred[s].T
        cov = np.mean((y_true >= ymin) & (y_true <= ymax))
        wid = np.mean(ymax - ymin)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred[s]))
        cwc = wid * (1 + (cov < (1 - alpha)) * (1 - cov))

        rows.append([STRATEGY_NAMES[s], cov, wid, rmse, cwc])

    summary_df = pd.DataFrame(
        rows, columns=["Strategy", "Coverage", "Width", "RMSE", "CWC"]
    )

    print("\n===== ONLINE CP RESULTS =====")
    print(summary_df)

    # =========================================================
    # PLOTTING (SAVE ONLY)
    # =========================================================
    plot_buffers = {}
    idx = np.arange(len(y_test))

    for s in y_pred:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.scatter(idx, y_true, color="blue", s=10, alpha=0.7, label="True")
        ax.fill_between(idx,
                        int_pred[s][:, 0],
                        int_pred[s][:, 1],
                        color="gray",
                        alpha=0.5,
                        label="Prediction interval")
        ax.plot(idx, y_pred[s], color="red", linewidth=1, label="Predicted mean")

        ax.set_title(f"{STRATEGY_NAMES[s]} ({model_name})")
        ax.set_xlabel("Sample Number")
        ax.set_ylabel("str")
        ax.legend(loc="upper left")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plot_buffers[s] = buf

        plt.close(fig)

    # =========================================================
    # EXCEL EXPORT (APPEND SAFE)
    # =========================================================
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        # ---- Summary ----
        sheet_name = "ONLINECP_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- Plots ----
        for s, buf in plot_buffers.items():
            img = Image(buf)
            sheet_name = f"ONLINECP_Plot_{s}"
            base = sheet_name
            i = 1
            while sheet_name in wb.sheetnames:
                sheet_name = f"{base}_{i}"
                i += 1
            ws = wb.create_sheet(sheet_name)
            ws.add_image(img, "A1")

        # ---- Values ----
        values = {}
        for s in y_pred:
            values[f"{s}_true"] = y_true
            values[f"{s}_pred"] = y_pred[s]
            values[f"{s}_lower"] = int_pred[s][:, 0]
            values[f"{s}_upper"] = int_pred[s][:, 1]

        df_vals = pd.DataFrame(values)

        sheet_name = "ONLINECP_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(df_vals, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))

    return summary_df

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_COP(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path=None,
    alpha=0.05,
    lr=0.05,          # Learning rate for COP
    T_burnin=50,      # Window size for distribution estimation
    scale=0.5,        # Scale factor for the integrator
    random_state=123,
):
    """
    Conformal Optimistic Prediction (COP)
    ✔ Implements the Distribution-Informed Gradient logic from the repository.
    ✔ Matches the "Excel-safe" and plotting structure of your previous functions.
    """

    warnings.filterwarnings("ignore")
    np.random.seed(random_state)

    # =========================================================
    # 1. Train Base Model & Generate Test Predictions
    # =========================================================
    # COP is an online calibration method, but we usually start with a pre-trained model
    # or an online model. Here we fit on Train and calibrate on Test stream.
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred_test = model.predict(X_test).ravel()
    y_true_test = y_test.to_numpy().ravel()

    # Calculate non-conformity scores (Absolute Residuals)
    # This assumes symmetric intervals.
    scores = np.abs(y_true_test - y_pred_test)
    T_test = len(scores)

    # =========================================================
    # 2. COP Initialization
    # =========================================================
    # Initialize quantiles based on training data residuals if possible
    y_pred_train = model.predict(X_train).ravel()
    scores_train = np.abs(y_train.to_numpy().ravel() - y_pred_train)
    init_q = np.quantile(scores_train, 1 - alpha)

    # Arrays to store history
    qs = np.zeros(T_test)          # Final adaptive quantile
    qts = np.zeros(T_test)         # Base quantile term
    integrators = np.zeros(T_test) # Optimistic integrator term
    covereds = np.zeros(T_test)    # Coverage boolean

    # Set initial values
    qs[0] = init_q
    qts[0] = init_q

    # =========================================================
    # 3. Online COP Loop
    # =========================================================
    # Logic adapted from `core/methods.py` function `COP`

    for t in range(T_test - 1):
        # 1. Observe coverage at current step t
        covereds[t] = 1 if qs[t] >= scores[t] else 0

        # 2. Calculate Standard Gradient
        # If covered, we decrease q (grad is positive). If not, increase q.
        grad = alpha if covereds[t] else -(1 - alpha)

        # 3. Calculate Distribution-Informed Gradient (The "Optimistic" part)
        if t < T_burnin:
            grad_i = 0
        else:
            # Look at the window of recent scores
            window_s = scores[t - T_burnin : t]

            # Calculate ECDF at the predicted next quantile step
            # logic: np.mean(window_s <= qts[t] - lr * grad) - (1 - alpha)
            current_target = qts[t] - lr * grad
            grad_i = np.mean(window_s <= current_target) - (1 - alpha)

        # 4. Update Terms
        integrator = -scale * grad_i

        # Update base quantile (Gradient Descent)
        qts[t + 1] = qts[t] - lr * grad

        # Update integrator state
        integrators[t + 1] = lr * integrator

        # Combine for final quantile
        qs[t + 1] = qts[t + 1] + integrators[t + 1]

        # Safety: Ensure quantile implies non-negative width
        qs[t + 1] = max(0, qs[t + 1])

    # Handle the final step coverage for metrics
    covereds[-1] = 1 if qs[-1] >= scores[-1] else 0

    # =========================================================
    # 4. Construct Prediction Intervals
    # =========================================================
    ymin = y_pred_test - qs
    ymax = y_pred_test + qs

    # =========================================================
    # 5. Metrics
    # =========================================================
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    coverage = np.mean(covereds)
    avg_width = np.mean(ymax - ymin)

    print(f"\n===== CONFORMAL OPTIMISTIC PREDICTION ({model_name}) =====")
    print(f"Coverage : {coverage*100:.3f}%")
    print(f"Avg Width: {avg_width:.3f}")
    print(f"RMSE     : {rmse:.3f}")
    print(f"Final q  : {qs[-1]:.3f}")

    # =========================================================
    # 6. Summary Tables
    # =========================================================
    summary_df = pd.DataFrame([{
        "Strategy": "COP",
        "Coverage": coverage,
        "Width": avg_width,
        "RMSE": rmse,
        "Alpha": alpha,
        "LR": lr,
        "Scale": scale
    }])

    values_df = pd.DataFrame({
        "sample_index": np.arange(len(y_true_test)),
        "y_true": y_true_test,
        "y_pred": y_pred_test,
        "ymin": ymin,
        "ymax": ymax,
        "width": ymax - ymin,
        "is_covered": covereds.astype(int),
        "score_residual": scores,
        "adaptive_q": qs
    })

    # =========================================================
    # 7. Plotting
    # =========================================================
    idx = np.arange(len(y_true_test))
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(idx, y_true_test, color="blue", s=10, alpha=0.7, label="True")
    ax.fill_between(idx, ymin, ymax, color="gray", alpha=0.5, label="COP Interval")
    ax.plot(idx, y_pred_test, color="red", linewidth=1, label="Predicted mean")

    ax.set_title(f"Conformal Optimistic Prediction (COP) - {model_name}")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("str")
    ax.legend(loc="upper left")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    # =========================================================
    # 8. Excel Export
    # =========================================================
    if excel_file_path:
        try:
            wb = load_workbook(_ensure_excel_file(excel_file_path))
        except FileNotFoundError:
            # Create a new workbook if it doesn't exist
            from openpyxl import Workbook
            wb = Workbook()
            wb.remove(wb.active) # Remove default sheet

        # ---- Summary ----
        sheet_name = "COP_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- Plot ----
        img = Image(buf)
        sheet_name = "COP_Plot"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        ws.add_image(img, "A1")

        # ---- Values ----
        sheet_name = "COP_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))

    return summary_df
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_EXTREME(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path=None,
    alpha=0.05,
    calib_size=0.2,
    gpd_threshold_quantile=0.95,
    n_bootstrap=30,
    random_state=123,
):
    """
    Extreme Conformal Prediction (upper-tail, single-sided)
    ✔ Same structure as NexCP & OnlineCP
    ✔ No return values
    ✔ Excel-safe
    """

    warnings.filterwarnings("ignore")
    np.random.seed(random_state)

    # =========================================================
    # Train / calibration split
    # =========================================================
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=calib_size,
        random_state=random_state,
        shuffle=True,
    )

    # =========================================================
    # Train model
    # =========================================================
    model = model_class(**model_params)
    model.fit(X_fit, y_fit)

    y_pred_test = model.predict(X_test).ravel()
    y_pred_calib = model.predict(X_calib).ravel()

    # =========================================================
    # Nonconformity scores (upper tail only)
    # =========================================================
    scores = y_calib.to_numpy().ravel() - y_pred_calib
    scores = scores[scores > 0]

    if len(scores) < 30:
        raise RuntimeError("Too few calibration exceedances for GPD fitting.")

    # =========================================================
    # Peaks-over-threshold
    # =========================================================
    u = np.quantile(scores, gpd_threshold_quantile)
    excesses = scores[scores > u] - u

    # =========================================================
    # Bootstrap GPD extreme quantile
    # =========================================================
    boot_q = []
    n_exc = len(excesses)
    n_scores = len(scores)

    for _ in range(n_bootstrap):
        resampled = np.random.choice(excesses, size=n_exc, replace=True)
        xi_b, _, sigma_b = genpareto.fit(resampled, floc=0)

        q_b = (
            u
            + (sigma_b / xi_b)
            * ((n_scores / (n_scores * alpha)) ** xi_b - 1)
        )
        boot_q.append(q_b)

    q_extreme = np.quantile(boot_q, 0.95)

    # =========================================================
    # Finite prediction intervals (for plotting & width)
    # =========================================================
    ymin = y_pred_test - q_extreme
    ymax = y_pred_test + q_extreme

    # =========================================================
    # Metrics
    # =========================================================
    y_true = y_test.to_numpy().ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_test))
    coverage = np.mean(y_true <= ymax)
    width = q_extreme  # meaningful one-sided width

    print(f"\n===== EXTREME CONFORMAL PREDICTION ({model_name}) =====")
    print(f"Coverage : {coverage*100:.3f}%")
    print(f"Width    : {width:.3f}")
    print(f"RMSE     : {rmse:.3f}")
    print(f"q_extreme: {q_extreme:.3f}")
    print(f"u        : {u:.3f}")
    print(f"#Exceed. : {len(scores)}")

    # =========================================================
    # Summary table
    # =========================================================
    summary_df = pd.DataFrame([{
        "Strategy": "EXTREME CP",
        "Coverage": coverage,
        "Width": width,
        "RMSE": rmse,
        "q_extreme": q_extreme,
        "threshold_u": u,
        "n_exceedances": len(scores),
    }])

    # =========================================================
    # Values table
    # =========================================================
    values_df = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred_test,
        "ymin": ymin,
        "ymax": ymax,
        "width": ymax - ymin,
        "is_covered": (y_true <= ymax).astype(int),
        "residual": y_true - y_pred_test,
    })

    # =========================================================
    # Plot (SAVE ONLY)
    # =========================================================
    idx = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(idx, y_true, s=10, alpha=0.7, label="True")
    ax.fill_between(idx, ymin, ymax, alpha=0.4, label="Extreme PI")
    ax.plot(idx, y_pred_test, color="red", lw=1, label="Prediction")

    ax.set_title(f"Extreme Conformal Prediction ({model_name})")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("str")
    ax.legend(loc="upper left")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    # =========================================================
    # Excel export (append-safe, SAME FILE)
    # =========================================================
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        # ---- Summary ----
        sheet_name = "EXTREMECP_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- Plot ----
        img = Image(buf)
        sheet_name = "EXTREMECP_Plot"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        ws.add_image(img, "A1")

        # ---- Values ----
        sheet_name = "EXTREMECP_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path



import os
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[Info] Install tqdm for progress bars: pip install tqdm")


# ═══════════════════════════════════════════════════════════
# AlphaNet Architecture  (FIXED)
# ═══════════════════════════════════════════════════════════
class AlphaNet(nn.Module):
    """
    Maps a feature vector (sum of LOO scores) → learned α̃.
    Output is clamped to (alpha_min, 1) so that the conformal
    denominator α̃·(n+1)−1 is always positive.
    """

    def __init__(self, input_dim, alpha_min=0.05):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.alpha_min = alpha_min  # lower bound for valid α̃
        self._initialize_near_one()

    def _initialize_near_one(self):
        """Bias output toward α̃ ≈ 1 at init (sigmoid(5) ≈ 0.993)."""
        with torch.no_grad():
            self.fc2.weight.data.normal_(0, 0.01)
            self.fc2.bias.data.fill_(5.0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        raw = self.sigmoid(self.fc2(x)).squeeze(-1)
        # Remap from (0,1) → (alpha_min, 1) so denominator stays positive
        return self.alpha_min + (1.0 - self.alpha_min) * raw


def _interval_size(scores, alpha, eps=1e-8):
    """
    Conformal interval width = 2 * Σscores / (α̃·(n+1) − 1).
    Denominator is clamped to be strictly positive.
    """
    n = len(scores)
    denom = alpha * (n + 1) - 1
    denom = torch.clamp(denom, min=eps)   # ← KEY FIX: prevent negative/zero
    r = scores.sum() / denom
    return 2 * r


def _fmt_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"


# ═══════════════════════════════════════════════════════════
# Main function  (CORRECTED + PROGRESS TRACKING)
# ═══════════════════════════════════════════════════════════
def conformal_predictions_AlphaNet(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path=None,
    alpha=0.05,
    lambda_reg=20.0,
    num_epochs=50,
    batch_size=32,
    lr_alpha=1e-3,
    calib_fraction=0.5,
    num_runs=4,
    random_state=123,
):
    """
    Learned-α Conformal Prediction via AlphaNet.
    - FIXED: α̃ clamped to valid range (no negative intervals)
    - FIXED: LOO scores precomputed (massive speedup)
    - ADDED: tqdm progress bars + ETA estimates
    """

    warnings.filterwarnings("ignore")
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_start_total = time.time()

    # --- numpy-ify if pandas ---
    y_train_np = y_train.to_numpy().ravel() if hasattr(y_train, "to_numpy") else np.asarray(y_train).ravel()
    y_test_np  = y_test.to_numpy().ravel()  if hasattr(y_test,  "to_numpy") else np.asarray(y_test).ravel()
    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    X_test_np  = X_test.to_numpy()  if hasattr(X_test,  "to_numpy") else np.asarray(X_test)

    # =================================================================
    # 1.  Split training data → fit + calibration
    # =================================================================
    n_total = len(X_train_np)
    n_fit   = int(n_total * (1 - calib_fraction))
    perm    = np.random.permutation(n_total)

    fit_idx   = perm[:n_fit]
    calib_idx = perm[n_fit:]

    X_fit,   y_fit   = X_train_np[fit_idx],   y_train_np[fit_idx]
    X_calib, y_calib = X_train_np[calib_idx],  y_train_np[calib_idx]
    n_calib = len(X_calib)

    # Compute alpha_min: must satisfy α̃·(n+1) − 1 > 0  →  α̃ > 1/(n+1)
    alpha_min = 1.0 / (n_calib + 1) + 0.01   # small margin above threshold
    print(f"  [Info] n_calib={n_calib} | alpha_min={alpha_min:.4f}")

    # =================================================================
    # 2.  Train base regression model
    # =================================================================
    print(f"  [Step 2/9] Training base {model_name} model...")
    t0 = time.time()
    base_model = model_class(**model_params)
    base_model.fit(X_fit, y_fit)
    print(f"  [Step 2/9] Done in {_fmt_time(time.time()-t0)}")

    # =================================================================
    # 3.  PRECOMPUTE all LOO scores  (KEY SPEEDUP)
    # =================================================================
    print(f"  [Step 3/9] Precomputing {n_calib} LOO score vectors...")
    t0 = time.time()

    precomputed_scores = []
    precomputed_sums   = []

    iterator = tqdm(range(n_calib), desc="    LOO precompute", leave=False) if HAS_TQDM else range(n_calib)

    for i in iterator:
        loo_X = np.delete(X_calib, i, axis=0)
        loo_y = np.delete(y_calib, i, axis=0)
        scores_loo = np.abs(base_model.predict(loo_X).ravel() - loo_y)

        scores_tensor = torch.tensor(scores_loo, dtype=torch.float32).to(device)
        precomputed_scores.append(scores_tensor)
        precomputed_sums.append(scores_loo.sum())

    elapsed_precompute = time.time() - t0
    print(f"  [Step 3/9] Done in {_fmt_time(elapsed_precompute)}")

    # Build feature matrix (sum of LOO scores per sample)
    X_alpha = torch.tensor(
        np.array(precomputed_sums).reshape(-1, 1),
        dtype=torch.float32,
    ).to(device)

    # =================================================================
    # 4.  Train AlphaNet (multiple runs, keep best)
    # =================================================================
    print(f"  [Step 4/9] Training AlphaNet: {num_runs} runs × {num_epochs} epochs")

    best_net   = None
    best_loss  = np.inf
    history    = {"losses": [], "sizes": [], "alphas": []}

    t0_train = time.time()

    for run in range(num_runs):
        np.random.seed(run)
        torch.manual_seed(run)

        alpha_net = AlphaNet(input_dim=1, alpha_min=alpha_min).to(device)
        optimizer = optim.Adam(alpha_net.parameters(), lr=lr_alpha)

        Dataset = TensorDataset(X_alpha, torch.arange(n_calib))
        loader  = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

        run_losses, run_sizes, run_alphas = [], [], []

        epoch_iter = (
            tqdm(range(num_epochs), desc=f"    Run {run+1}/{num_runs}", leave=False)
            if HAS_TQDM else range(num_epochs)
        )

        for epoch in epoch_iter:
            t_epoch_start = time.time()
            epoch_loss = epoch_size = epoch_alpha = 0.0

            for x_batch, idx_batch in loader:
                x_batch = x_batch.to(device)
                alpha_pred = alpha_net(x_batch)

                batch_sizes = []
                for j, idx in enumerate(idx_batch):
                    # USE PRECOMPUTED scores instead of calling predict()
                    scores_t = precomputed_scores[idx.item()]
                    size_j = _interval_size(scores_t, alpha_pred[j])
                    batch_sizes.append(size_j)

                batch_sizes = torch.stack(batch_sizes)
                loss = (batch_sizes + lambda_reg * alpha_pred).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss  += loss.item()
                epoch_size  += batch_sizes.mean().item()
                epoch_alpha += alpha_pred.mean().item()

            n_batches = len(loader)
            run_losses.append(epoch_loss  / n_batches)
            run_sizes.append(epoch_size   / n_batches)
            run_alphas.append(epoch_alpha / n_batches)

            epoch_time = time.time() - t_epoch_start

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                epochs_done = (run * num_epochs) + epoch + 1
                epochs_total = num_runs * num_epochs
                elapsed_train = time.time() - t0_train
                eta = (elapsed_train / epochs_done) * (epochs_total - epochs_done)

                print(
                    f"  λ={lambda_reg} | Run {run+1}/{num_runs} | "
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Loss: {run_losses[-1]:.4f} | "
                    f"Size: {run_sizes[-1]:.4f} | "
                    f"α̃: {run_alphas[-1]:.4f} | "
                    f"⏱ {epoch_time:.2f}s/epoch | "
                    f"ETA: {_fmt_time(eta)}"
                )

        if run_losses[-1] < best_loss:
            best_loss = run_losses[-1]
            best_net  = alpha_net

        history["losses"].append(run_losses)
        history["sizes"].append(run_sizes)
        history["alphas"].append(run_alphas)

        print(f"  ✓ Run {run+1} complete | Final loss: {run_losses[-1]:.4f}")

    train_time = time.time() - t0_train
    print(f"  [Step 4/9] Training done in {_fmt_time(train_time)}")

    alpha_net = best_net
    alpha_net.eval()

    # =================================================================
    # 5.  Compute conformal intervals on the TEST set
    # =================================================================
    print(f"  [Step 5/9] Computing conformal intervals on test set...")

    scores_calib = np.abs(base_model.predict(X_calib).ravel() - y_calib)
    sum_scores   = scores_calib.sum()

    with torch.no_grad():
        feat_test = torch.tensor([sum_scores], dtype=torch.float32).to(device)
        alpha_val = alpha_net(feat_test).item()

    denom = alpha_val * (n_calib + 1) - 1
    if denom <= 0:
        q = np.quantile(scores_calib, 1 - alpha)
        half_width = q
        print(f"  [Warning] α̃ too small (denom={denom:.4f}); fallback q={q:.4f}")
    else:
        half_width = sum_scores / denom

    y_pred_test = base_model.predict(X_test_np).ravel()
    ymin = y_pred_test - half_width
    ymax = y_pred_test + half_width

    # =================================================================
    # 6.  Metrics
    # =================================================================
    y_true_test = y_test_np
    is_covered  = ((y_true_test >= ymin) & (y_true_test <= ymax)).astype(int)
    coverage    = is_covered.mean()
    avg_width   = np.mean(ymax - ymin)
    rmse        = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

    total_time = time.time() - t_start_total

    print(f"\n{'='*55}")
    print(f"  ALPHANET CONFORMAL ({model_name})")
    print(f"{'='*55}")
    print(f"  Learned α̃     : {alpha_val:.4f}")
    print(f"  Half-width (r) : {half_width:.4f}")
    print(f"  Coverage       : {coverage*100:.3f}%")
    print(f"  Avg Width      : {avg_width:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  Total time     : {_fmt_time(total_time)}")
    print(f"    ├─ Precompute: {_fmt_time(elapsed_precompute)}")
    print(f"    └─ Training  : {_fmt_time(train_time)}")
    print(f"{'='*55}")

    # =================================================================
    # 7.  Summary & values DataFrames
    # =================================================================
    summary_df = pd.DataFrame([{
        "Strategy":       "AlphaNet Conformal",
        "Coverage":       coverage,
        "Width":          avg_width,
        "RMSE":           rmse,
        "Alpha_nominal":  alpha,
        "Alpha_learned":  alpha_val,
        "Lambda_reg":     lambda_reg,
        "Total_time_sec": round(total_time, 2),
    }])

    values_df = pd.DataFrame({
        "sample_index": np.arange(len(y_true_test)),
        "y_true":       y_true_test,
        "y_pred":       y_pred_test,
        "ymin":         ymin,
        "ymax":         ymax,
        "width":        ymax - ymin,
        "is_covered":   is_covered,
    })

    # =================================================================
    # 8.  Plotting (SAVE ONLY — NO SHOW)
    # =================================================================
    print(f"  [Step 8/9] Generating plots...")
    idx = np.arange(len(y_true_test))

    # --- PI plot ---
    fig_pi, ax_pi = plt.subplots(figsize=(10, 4))
    ax_pi.scatter(idx, y_true_test, color="blue", s=10, alpha=0.7, label="True")
    ax_pi.fill_between(idx, ymin, ymax, color="gray", alpha=0.5,
                       label=f"AlphaNet Interval (α̃={alpha_val:.3f})")
    ax_pi.plot(idx, y_pred_test, color="red", linewidth=1, label="Predicted mean")
    ax_pi.set_title(f"AlphaNet Conformal – {model_name}")
    ax_pi.set_xlabel("Sample Number")
    ax_pi.set_ylabel("str")
    ax_pi.legend(loc="upper left")
    plt.tight_layout()

    buf_pi = io.BytesIO()
    fig_pi.savefig(buf_pi, format="png")
    buf_pi.seek(0)
    plt.close(fig_pi)

    # --- Training curves ---
    arr_losses = np.array(history["losses"])
    arr_sizes  = np.array(history["sizes"])
    arr_alphas = np.array(history["alphas"])
    epochs_arr = np.arange(1, num_epochs + 1)

    fig_tc, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, ylabel, title in zip(
        axs,
        [arr_losses, arr_sizes, arr_alphas],
        ["Loss", "Mean Interval Size", r"Mean $\tilde{\alpha}$"],
        ["Training Loss", "Interval Size", r"Learned $\tilde{\alpha}$"],
    ):
        m = data.mean(axis=0)
        s = data.std(axis=0)
        ax.plot(epochs_arr, m, color="#0077bb")
        ax.fill_between(epochs_arr, m - s, m + s, color="#0077bb", alpha=0.25)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.5)
    plt.tight_layout()

    buf_tc = io.BytesIO()
    fig_tc.savefig(buf_tc, format="png")
    buf_tc.seek(0)
    plt.close(fig_tc)

    # =================================================================
    # 9.  Excel Export
    # =================================================================
    if excel_file_path:
        print(f"  [Step 9/9] Saving to Excel...")
        try:
            wb = load_workbook(_ensure_excel_file(excel_file_path))
        except FileNotFoundError:
            from openpyxl import Workbook
            wb = Workbook()
            wb.remove(wb.active)

        def _unique_sheet(wb, name):
            base, i = name, 1
            while name in wb.sheetnames:
                name = f"{base}_{i}"; i += 1
            return name

        sn = _unique_sheet(wb, "AlphaNet_Summary")
        ws = wb.create_sheet(sn, 0)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        sn = _unique_sheet(wb, "AlphaNet_Plot_PI")
        ws = wb.create_sheet(sn)
        ws.add_image(Image(buf_pi), "A1")

        sn = _unique_sheet(wb, "AlphaNet_Plot_Train")
        ws = wb.create_sheet(sn)
        ws.add_image(Image(buf_tc), "A1")

        sn = _unique_sheet(wb, "AlphaNet_Values")
        ws = wb.create_sheet(sn)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))
        print(f"  [Step 9/9] Saved!")

    print(f"\n  ✅ AlphaNet Conformal complete in {_fmt_time(total_time)}")
    return summary_df
```

# **Tuned Parameters**

```python
best_scores_autosampler = {('Random Forest', 'MedianPruner'): {'best_score': 0.14791743306272762,
  'best_params': {'n_estimators': 700,
   'criterion': 'friedman_mse',
   'max_depth': 20,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 'log2',
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 0.14791743306272762,
  'test_rmse': 0.3846003549955819,
  'test_corr_coef': 0.9402858662399178,
  'pruner': 'MedianPruner'},
 ('Random Forest', 'NopPruner'): {'best_score': 0.12433881301696886,
  'best_params': {'n_estimators': 200,
   'criterion': 'friedman_mse',
   'max_depth': 20,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 0.5,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 0.12433881301696886,
  'test_rmse': 0.35261709121505846,
  'test_corr_coef': 0.944977558514824,
  'pruner': 'NopPruner'},
 ('Random Forest', 'PatientPruner'): {'best_score': 0.14286646298335165,
  'best_params': {'n_estimators': 700,
   'criterion': 'friedman_mse',
   'max_depth': 10,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 0.3,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01},
  'test_mse': 0.14286646298335165,
  'test_rmse': 0.37797680217620716,
  'test_corr_coef': 0.9390325866763238,
  'pruner': 'PatientPruner'},
 ('Random Forest', 'PercentilePruner'): {'best_score': 0.1716436314060288,
  'best_params': {'n_estimators': 700,
   'criterion': 'squared_error',
   'max_depth': 40,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 'sqrt',
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 0.1716436314060288,
  'test_rmse': 0.414298963800332,
  'test_corr_coef': 0.9290762183966498,
  'pruner': 'PercentilePruner'},
 ('Random Forest',
  'SuccessiveHalvingPruner'): {'best_score': 0.1360667977550869, 'best_params': {'n_estimators': 100,
   'criterion': 'friedman_mse',
   'max_depth': 30,
   'min_samples_split': 5,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 0.5,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001}, 'test_mse': 0.1360667977550869, 'test_rmse': 0.36887233259636987, 'test_corr_coef': 0.9427132938138372, 'pruner': 'SuccessiveHalvingPruner'},
 ('Random Forest', 'HyperbandPruner'): {'best_score': 0.1219501754807693,
  'best_params': {'n_estimators': 200,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 0.1219501754807693,
  'test_rmse': 0.3492136530560758,
  'test_corr_coef': 0.9494702677569818,
  'pruner': 'HyperbandPruner'},
 ('Random Forest', 'ThresholdPruner'): {'best_score': 0.12756209935897436,
  'best_params': {'n_estimators': 100,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 5,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 0.3,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 0.12756209935897436,
  'test_rmse': 0.3571583673372001,
  'test_corr_coef': 0.9503026753242991,
  'pruner': 'ThresholdPruner'},
 ('Random Forest', 'WilcoxonPruner'): {'best_score': 0.11958701205128164,
  'best_params': {'n_estimators': 500,
   'criterion': 'absolute_error',
   'max_depth': 40,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 0.11958701205128164,
  'test_rmse': 0.3458135509942918,
  'test_corr_coef': 0.9496010672407575,
  'pruner': 'WilcoxonPruner'},
 ('Gradient Boosting', 'MedianPruner'): {'best_score': 0.1182432763281944,
  'best_params': {'loss': 'quantile',
   'learning_rate': 0.1,
   'n_estimators': 700,
   'subsample': 1.0,
   'criterion': 'friedman_mse',
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 10,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 0.1182432763281944,
  'test_rmse': 0.3438652008101349,
  'test_corr_coef': 0.9512317270112821,
  'pruner': 'MedianPruner'},
 ('Gradient Boosting', 'NopPruner'): {'best_score': 0.11068989080859347,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.1,
   'n_estimators': 300,
   'subsample': 1.0,
   'criterion': 'squared_error',
   'min_samples_split': 10,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 3,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 0.11068989080859347,
  'test_rmse': 0.33270090292722904,
  'test_corr_coef': 0.9493762500601717,
  'pruner': 'NopPruner'},
 ('Gradient Boosting', 'PatientPruner'): {'best_score': 0.10530363807402264,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.2,
   'n_estimators': 500,
   'subsample': 0.5,
   'criterion': 'friedman_mse',
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 7,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 0.10530363807402264,
  'test_rmse': 0.3245052204110477,
  'test_corr_coef': 0.9556719416391912,
  'pruner': 'PatientPruner'},
 ('Gradient Boosting', 'PercentilePruner'): {'best_score': 0.1066795811263761,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.2,
   'n_estimators': 200,
   'subsample': 1.0,
   'criterion': 'squared_error',
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 5,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 'log2',
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 0.1066795811263761,
  'test_rmse': 0.3266184029205582,
  'test_corr_coef': 0.9506642390421116,
  'pruner': 'PercentilePruner'},
 ('Gradient Boosting',
  'SuccessiveHalvingPruner'): {'best_score': 0.11512599100928875, 'best_params': {'loss': 'squared_error',
   'learning_rate': 0.2,
   'n_estimators': 200,
   'subsample': 0.5,
   'criterion': 'friedman_mse',
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 7,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 'log2',
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.01}, 'test_mse': 0.11512599100928875, 'test_rmse': 0.3393022119133454, 'test_corr_coef': 0.9467955625384996, 'pruner': 'SuccessiveHalvingPruner'},
 ('Gradient Boosting', 'HyperbandPruner'): {'best_score': 0.10465810020884284,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.1,
   'n_estimators': 500,
   'subsample': 0.9,
   'criterion': 'friedman_mse',
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 7,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 0.10465810020884284,
  'test_rmse': 0.3235090419274905,
  'test_corr_coef': 0.9530774863356877,
  'pruner': 'HyperbandPruner'},
 ('Gradient Boosting', 'ThresholdPruner'): {'best_score': 0.1520043695038204,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.2,
   'n_estimators': 500,
   'subsample': 0.5,
   'criterion': 'squared_error',
   'min_samples_split': 10,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 10,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.01},
  'test_mse': 0.1520043695038204,
  'test_rmse': 0.38987737752249796,
  'test_corr_coef': 0.9301229502534659,
  'pruner': 'ThresholdPruner'},
 ('Gradient Boosting', 'WilcoxonPruner'): {'best_score': 0.1048585215709674,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.05,
   'n_estimators': 100,
   'subsample': 1.0,
   'criterion': 'friedman_mse',
   'min_samples_split': 5,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 7,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 0.1048585215709674,
  'test_rmse': 0.3238186553782339,
  'test_corr_coef': 0.9550241321708607,
  'pruner': 'WilcoxonPruner'},
 ('XGBoost', 'MedianPruner'): {'best_score': 0.10869347274211374,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.15,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.10869347274211374,
  'test_rmse': 0.32968693140935046,
  'test_corr_coef': 0.9497533023472757,
  'pruner': 'MedianPruner'},
 ('XGBoost', 'NopPruner'): {'best_score': 0.09949036331525066,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.09949036331525066,
  'test_rmse': 0.3154209303696422,
  'test_corr_coef': 0.9541253529294844,
  'pruner': 'NopPruner'},
 ('XGBoost', 'PatientPruner'): {'best_score': 0.09125516551878989,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.15,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.09125516551878989,
  'test_rmse': 0.30208469924640324,
  'test_corr_coef': 0.9583479568255933,
  'pruner': 'PatientPruner'},
 ('XGBoost', 'PercentilePruner'): {'best_score': 0.07695790156981148,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.07695790156981148,
  'test_rmse': 0.27741287203338544,
  'test_corr_coef': 0.9651233444070674,
  'pruner': 'PercentilePruner'},
 ('XGBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.09164219499731802,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.09164219499731802,
  'test_rmse': 0.30272461908030873,
  'test_corr_coef': 0.9581183522627823,
  'pruner': 'SuccessiveHalvingPruner'},
 ('XGBoost', 'HyperbandPruner'): {'best_score': 0.11796674493109749,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.15,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'colsample_bylevel': 0.5,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.11796674493109749,
  'test_rmse': 0.3434628727112983,
  'test_corr_coef': 0.946356519178199,
  'pruner': 'HyperbandPruner'},
 ('XGBoost', 'ThresholdPruner'): {'best_score': 0.09541265332196727,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.09541265332196727,
  'test_rmse': 0.3088893868716879,
  'test_corr_coef': 0.9562760733629969,
  'pruner': 'ThresholdPruner'},
 ('XGBoost', 'WilcoxonPruner'): {'best_score': 0.103271054671266,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.103271054671266,
  'test_rmse': 0.3213581408199674,
  'test_corr_coef': 0.9538543379680603,
  'pruner': 'WilcoxonPruner'},
 ('LightGBM', 'MedianPruner'): {'best_score': 0.09579461308882922,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 1,
   'subsample': 1.0,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 0.01,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.09579461308882922,
  'test_rmse': 0.30950704852850963,
  'test_corr_coef': 0.9560150939665216,
  'pruner': 'MedianPruner'},
 ('LightGBM', 'NopPruner'): {'best_score': 0.08845220567510384,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 63,
   'max_depth': -1,
   'min_child_samples': 1,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'reg_alpha': 0,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.08845220567510384,
  'test_rmse': 0.29740915533168083,
  'test_corr_coef': 0.9594235811119701,
  'pruner': 'NopPruner'},
 ('LightGBM', 'PatientPruner'): {'best_score': 0.09266510020619696,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': 5,
   'min_child_samples': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 0,
   'min_child_weight': 1e-05,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.09266510020619696,
  'test_rmse': 0.304409428576378,
  'test_corr_coef': 0.9579286288190465,
  'pruner': 'PatientPruner'},
 ('LightGBM', 'PercentilePruner'): {'best_score': 0.08757106043665228,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'num_leaves': 63,
   'max_depth': -1,
   'min_child_samples': 1,
   'subsample': 0.8,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.08757106043665228,
  'test_rmse': 0.2959240788389013,
  'test_corr_coef': 0.9605348055306665,
  'pruner': 'PercentilePruner'},
 ('LightGBM', 'SuccessiveHalvingPruner'): {'best_score': 0.08648767059707319,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': 5,
   'min_child_samples': 1,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.08648767059707319,
  'test_rmse': 0.29408786203628534,
  'test_corr_coef': 0.9604547374297212,
  'pruner': 'SuccessiveHalvingPruner'},
 ('LightGBM', 'HyperbandPruner'): {'best_score': 0.08515664903701535,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'num_leaves': 63,
   'max_depth': 7,
   'min_child_samples': 1,
   'subsample': 0.8,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.08515664903701535,
  'test_rmse': 0.29181612196212764,
  'test_corr_coef': 0.9613052396923227,
  'pruner': 'HyperbandPruner'},
 ('LightGBM', 'ThresholdPruner'): {'best_score': 0.08478315550638713,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.1,
   'num_leaves': 63,
   'max_depth': 7,
   'min_child_samples': 1,
   'subsample': 0.8,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 0,
   'min_child_weight': 0.001,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.08478315550638713,
  'test_rmse': 0.2911754720205449,
  'test_corr_coef': 0.9616059431088415,
  'pruner': 'ThresholdPruner'},
 ('LightGBM', 'WilcoxonPruner'): {'best_score': 0.10080726312813501,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.15,
   'num_leaves': 63,
   'max_depth': -1,
   'min_child_samples': 1,
   'subsample': 0.9,
   'colsample_bytree': 1.0,
   'reg_alpha': 0,
   'reg_lambda': 0,
   'min_child_weight': 0.1,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.10080726312813501,
  'test_rmse': 0.31750159547336926,
  'test_corr_coef': 0.9572179824158072,
  'pruner': 'WilcoxonPruner'},
 ('GPBoost', 'MedianPruner'): {'best_score': 0.10077992875084572,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 1,
   'subsample': 0.8,
   'colsample_bytree': 0.5,
   'reg_alpha': 0,
   'reg_lambda': 1.0,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.10077992875084572,
  'test_rmse': 0.31745854650780114,
  'test_corr_coef': 0.9541909436911268,
  'pruner': 'MedianPruner'},
 ('GPBoost', 'NopPruner'): {'best_score': 0.0971604175926899,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.15,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 1,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 0,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.0971604175926899,
  'test_rmse': 0.3117056585830451,
  'test_corr_coef': 0.9562940288053847,
  'pruner': 'NopPruner'},
 ('GPBoost', 'PatientPruner'): {'best_score': 0.09965556516984378,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 1,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 1.0,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.09965556516984378,
  'test_rmse': 0.31568269697568757,
  'test_corr_coef': 0.9543914128525639,
  'pruner': 'PatientPruner'},
 ('GPBoost', 'PercentilePruner'): {'best_score': 0.0990669606322464,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 5,
   'num_leaves': 15,
   'min_child_samples': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.0990669606322464,
  'test_rmse': 0.31474904389409414,
  'test_corr_coef': 0.9543610186924695,
  'pruner': 'PercentilePruner'},
 ('GPBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.10197570010396986,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.15,
   'max_depth': 3,
   'num_leaves': 15,
   'min_child_samples': 1,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.5,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.10197570010396986,
  'test_rmse': 0.3193363432244596,
  'test_corr_coef': 0.9532787263368191,
  'pruner': 'SuccessiveHalvingPruner'},
 ('GPBoost', 'HyperbandPruner'): {'best_score': 0.10238570187103908,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_depth': 5,
   'num_leaves': 63,
   'min_child_samples': 1,
   'subsample': 1.0,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.10238570187103908,
  'test_rmse': 0.3199776583935808,
  'test_corr_coef': 0.9530000723853655,
  'pruner': 'HyperbandPruner'},
 ('GPBoost', 'ThresholdPruner'): {'best_score': 0.09772508132347915,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 7,
   'num_leaves': 15,
   'min_child_samples': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.5,
   'reg_alpha': 0,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.09772508132347915,
  'test_rmse': 0.3126101107185741,
  'test_corr_coef': 0.9570970943904238,
  'pruner': 'ThresholdPruner'},
 ('GPBoost', 'WilcoxonPruner'): {'best_score': 0.10061184888027573,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_depth': 5,
   'num_leaves': 63,
   'min_child_samples': 1,
   'subsample': 1.0,
   'colsample_bytree': 0.9,
   'reg_alpha': 0,
   'reg_lambda': 0.5,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.10061184888027573,
  'test_rmse': 0.3171937087652839,
  'test_corr_coef': 0.9539440570404897,
  'pruner': 'WilcoxonPruner'},
 ('CatBoost', 'MedianPruner'): {'best_score': 0.10208631062341576,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 20,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.10208631062341576,
  'test_rmse': 0.3195094844029137,
  'test_corr_coef': 0.9536489292255467,
  'pruner': 'MedianPruner'},
 ('CatBoost', 'NopPruner'): {'best_score': 0.10208631062341576,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 1,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.10208631062341576,
  'test_rmse': 0.3195094844029137,
  'test_corr_coef': 0.9536489292255467,
  'pruner': 'NopPruner'},
 ('CatBoost', 'PatientPruner'): {'best_score': 0.10239428808328108,
  'best_params': {'iterations': 500,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 1,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.10239428808328108,
  'test_rmse': 0.3199910750056649,
  'test_corr_coef': 0.9529055377273937,
  'pruner': 'PatientPruner'},
 ('CatBoost', 'PercentilePruner'): {'best_score': 0.10239428808328108,
  'best_params': {'iterations': 500,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.10239428808328108,
  'test_rmse': 0.3199910750056649,
  'test_corr_coef': 0.9529055377273937,
  'pruner': 'PercentilePruner'},
 ('CatBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.09767295423337319,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 8,
   'l2_leaf_reg': 3,
   'border_count': 32,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.09767295423337319,
  'test_rmse': 0.31252672563058215,
  'test_corr_coef': 0.9555811756306276,
  'pruner': 'SuccessiveHalvingPruner'},
 ('CatBoost', 'HyperbandPruner'): {'best_score': 0.10172583553263423,
  'best_params': {'iterations': 500,
   'learning_rate': 0.05,
   'depth': 6,
   'l2_leaf_reg': 7,
   'border_count': 32,
   'min_data_in_leaf': 5,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.10172583553263423,
  'test_rmse': 0.3189448785176433,
  'test_corr_coef': 0.9534628951122747,
  'pruner': 'HyperbandPruner'},
 ('CatBoost', 'ThresholdPruner'): {'best_score': 0.10106934356001397,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 20,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.10106934356001397,
  'test_rmse': 0.3179140505860255,
  'test_corr_coef': 0.9546386164786963,
  'pruner': 'ThresholdPruner'},
 ('CatBoost', 'WilcoxonPruner'): {'best_score': 0.10305473754773768,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.05,
   'depth': 6,
   'l2_leaf_reg': 9,
   'border_count': 32,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.10305473754773768,
  'test_rmse': 0.3210213973362799,
  'test_corr_coef': 0.9525240172898933,
  'pruner': 'WilcoxonPruner'},
 ('NGBoost', 'MedianPruner'): {'best_score': 0.09600403873086634,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.1,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.09600403873086634,
  'test_rmse': 0.3098451851019576,
  'test_corr_coef': 0.956182831996212,
  'pruner': 'MedianPruner'},
 ('NGBoost', 'NopPruner'): {'best_score': 0.10298358370095521,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.9,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.10298358370095521,
  'test_rmse': 0.32091055405043195,
  'test_corr_coef': 0.9527094596683632,
  'pruner': 'NopPruner'},
 ('NGBoost', 'PatientPruner'): {'best_score': 0.09839264124301354,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.09839264124301354,
  'test_rmse': 0.31367601317763133,
  'test_corr_coef': 0.9558170219923404,
  'pruner': 'PatientPruner'},
 ('NGBoost', 'PercentilePruner'): {'best_score': 0.09460671518634067,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'natural_gradient': True,
   'minibatch_frac': 1.0,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.09460671518634067,
  'test_rmse': 0.30758204626788715,
  'test_corr_coef': 0.9571004798444099,
  'pruner': 'PercentilePruner'},
 ('NGBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.10276710383346067,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.1,
   'natural_gradient': True,
   'minibatch_frac': 1.0,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.10276710383346067,
  'test_rmse': 0.32057308657069244,
  'test_corr_coef': 0.9536974280638412,
  'pruner': 'SuccessiveHalvingPruner'},
 ('NGBoost', 'HyperbandPruner'): {'best_score': 0.10401084305085206,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.9,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.10401084305085206,
  'test_rmse': 0.32250712093045647,
  'test_corr_coef': 0.9522320235967583,
  'pruner': 'HyperbandPruner'},
 ('NGBoost', 'ThresholdPruner'): {'best_score': 0.09436348922436896,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.1,
   'natural_gradient': True,
   'minibatch_frac': 1.0,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.09436348922436896,
  'test_rmse': 0.30718640794209784,
  'test_corr_coef': 0.9574072281018148,
  'pruner': 'ThresholdPruner'},
 ('NGBoost', 'WilcoxonPruner'): {'best_score': 0.1021906955551333,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.9,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.1021906955551333,
  'test_rmse': 0.31967279451829067,
  'test_corr_coef': 0.9530038393086567,
  'pruner': 'WilcoxonPruner'},
 ('TabNet', 'MedianPruner'): {'best_score': 0.13239617355888386,
  'best_params': {'n_d': 32,
   'n_a': 64,
   'n_steps': 3,
   'gamma': 2.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.13239617355888386,
  'test_rmse': 0.3638628499295907,
  'test_corr_coef': 0.9433413977700718,
  'pruner': 'MedianPruner'},
 ('TabNet', 'NopPruner'): {'best_score': 0.09878494770127107,
  'best_params': {'n_d': 8,
   'n_a': 32,
   'n_steps': 3,
   'gamma': 1.5,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.09878494770127107,
  'test_rmse': 0.3143007281271729,
  'test_corr_coef': 0.9543780290231134,
  'pruner': 'NopPruner'},
 ('TabNet', 'PatientPruner'): {'best_score': 0.12364709858619594,
  'best_params': {'n_d': 32,
   'n_a': 16,
   'n_steps': 3,
   'gamma': 1.5,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 3,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.12364709858619594,
  'test_rmse': 0.3516348938689048,
  'test_corr_coef': 0.9478452944545069,
  'pruner': 'PatientPruner'},
 ('TabNet', 'PercentilePruner'): {'best_score': 0.12222229520470243,
  'best_params': {'n_d': 8,
   'n_a': 32,
   'n_steps': 3,
   'gamma': 1.5,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 3,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.12222229520470243,
  'test_rmse': 0.34960305376913176,
  'test_corr_coef': 0.9465193577141441,
  'pruner': 'PercentilePruner'},
 ('TabNet', 'SuccessiveHalvingPruner'): {'best_score': 0.15466336601968256,
  'best_params': {'n_d': 8,
   'n_a': 32,
   'n_steps': 5,
   'gamma': 1.5,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.15466336601968256,
  'test_rmse': 0.3932726357371977,
  'test_corr_coef': 0.942886500568057,
  'pruner': 'SuccessiveHalvingPruner'},
 ('TabNet', 'HyperbandPruner'): {'best_score': 0.19984999784655258,
  'best_params': {'n_d': 8,
   'n_a': 64,
   'n_steps': 5,
   'gamma': 2.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.19984999784655258,
  'test_rmse': 0.44704585653661144,
  'test_corr_coef': 0.9231143978702843,
  'pruner': 'HyperbandPruner'},
 ('TabNet', 'ThresholdPruner'): {'best_score': 0.144243897326256,
  'best_params': {'n_d': 8,
   'n_a': 16,
   'n_steps': 3,
   'gamma': 2.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.144243897326256,
  'test_rmse': 0.37979454620393904,
  'test_corr_coef': 0.9346994486373399,
  'pruner': 'ThresholdPruner'},
 ('TabNet', 'WilcoxonPruner'): {'best_score': 0.14351252811533713,
  'best_params': {'n_d': 16,
   'n_a': 64,
   'n_steps': 7,
   'gamma': 2.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.14351252811533713,
  'test_rmse': 0.3788304741112271,
  'test_corr_coef': 0.9484248001017362,
  'pruner': 'WilcoxonPruner'},
 ('HistGradientBoosting', 'MedianPruner'): {'best_score': 0.1301291266557467,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 500,
   'max_depth': 5,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 63,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.1301291266557467,
  'test_rmse': 0.3607341495558006,
  'test_corr_coef': 0.9402650900812275,
  'pruner': 'MedianPruner'},
 ('HistGradientBoosting', 'NopPruner'): {'best_score': 0.11637483386311197,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 500,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.11637483386311197,
  'test_rmse': 0.34113755856415456,
  'test_corr_coef': 0.9465434876799057,
  'pruner': 'NopPruner'},
 ('HistGradientBoosting', 'PatientPruner'): {'best_score': 0.12481160582992522,
  'best_params': {'learning_rate': 0.15,
   'max_iter': 100,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 31,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.12481160582992522,
  'test_rmse': 0.3532868605395978,
  'test_corr_coef': 0.9420773956520916,
  'pruner': 'PatientPruner'},
 ('HistGradientBoosting',
  'PercentilePruner'): {'best_score': 0.13894046607482627, 'best_params': {'learning_rate': 0.01,
   'max_iter': 500,
   'max_depth': 5,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 63,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.13894046607482627, 'test_rmse': 0.37274718788319017, 'test_corr_coef': 0.9364689242288793, 'pruner': 'PercentilePruner'},
 ('HistGradientBoosting',
  'SuccessiveHalvingPruner'): {'best_score': 0.12026621236290441, 'best_params': {'learning_rate': 0.1,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.12026621236290441, 'test_rmse': 0.34679419309282616, 'test_corr_coef': 0.944283682974236, 'pruner': 'SuccessiveHalvingPruner'},
 ('HistGradientBoosting',
  'HyperbandPruner'): {'best_score': 0.12176951099751362, 'best_params': {'learning_rate': 0.15,
   'max_iter': 500,
   'max_depth': 5,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 0.1,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.12176951099751362, 'test_rmse': 0.34895488389978674, 'test_corr_coef': 0.9435402434102056, 'pruner': 'HyperbandPruner'},
 ('HistGradientBoosting',
  'ThresholdPruner'): {'best_score': 0.12481160582992522, 'best_params': {'learning_rate': 0.15,
   'max_iter': 100,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 15,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.12481160582992522, 'test_rmse': 0.3532868605395978, 'test_corr_coef': 0.9420773956520916, 'pruner': 'ThresholdPruner'},
 ('HistGradientBoosting',
  'WilcoxonPruner'): {'best_score': 0.12481160582992522, 'best_params': {'learning_rate': 0.15,
   'max_iter': 100,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 15,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.12481160582992522, 'test_rmse': 0.3532868605395978, 'test_corr_coef': 0.9420773956520916, 'pruner': 'WilcoxonPruner'},
 ('PGBM', 'MedianPruner'): {'best_score': 0.10807448000966534,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 16,
   'min_split_gain': 0.0,
   'reg_lambda': 0.1,
   'feature_fraction': 1.0,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 0.10807448000966534,
  'test_rmse': 0.3287468326990624,
  'test_corr_coef': 0.9503918343780713,
  'pruner': 'MedianPruner'},
 ('PGBM', 'NopPruner'): {'best_score': 0.08667379001153155,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_leaves': 28,
   'min_split_gain': 0.1,
   'reg_lambda': 0.1,
   'feature_fraction': 1.0,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 0.08667379001153155,
  'test_rmse': 0.29440412702870106,
  'test_corr_coef': 0.9600984793909815,
  'pruner': 'NopPruner'},
 ('PGBM', 'PatientPruner'): {'best_score': 0.11656466955545558,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'max_leaves': 51,
   'min_split_gain': 0.0,
   'reg_lambda': 1.0,
   'feature_fraction': 0.5,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 0.11656466955545558,
  'test_rmse': 0.3414156844016625,
  'test_corr_coef': 0.9472150287044171,
  'pruner': 'PatientPruner'},
 ('PGBM', 'PercentilePruner'): {'best_score': 0.10504306741874887,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_leaves': 35,
   'min_split_gain': 0.0,
   'reg_lambda': 0.1,
   'feature_fraction': 0.7,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 0.10504306741874887,
  'test_rmse': 0.32410348257732263,
  'test_corr_coef': 0.9522201719592369,
  'pruner': 'PercentilePruner'},
 ('PGBM', 'SuccessiveHalvingPruner'): {'best_score': 0.08020696452628674,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_leaves': 62,
   'min_split_gain': 0.1,
   'reg_lambda': 0.1,
   'feature_fraction': 0.9,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'laplace'},
  'test_mse': 0.08020696452628674,
  'test_rmse': 0.2832083412018204,
  'test_corr_coef': 0.9633873653399692,
  'pruner': 'SuccessiveHalvingPruner'},
 ('PGBM', 'HyperbandPruner'): {'best_score': 0.10453188660026444,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_leaves': 52,
   'min_split_gain': 0.5,
   'reg_lambda': 0.1,
   'feature_fraction': 1.0,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 0.10453188660026444,
  'test_rmse': 0.3233139134034668,
  'test_corr_coef': 0.9536901078039018,
  'pruner': 'HyperbandPruner'},
 ('PGBM', 'ThresholdPruner'): {'best_score': 0.09087868478061968,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 63,
   'min_split_gain': 0.1,
   'reg_lambda': 1.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 0.09087868478061968,
  'test_rmse': 0.3014609175011243,
  'test_corr_coef': 0.9587530984381141,
  'pruner': 'ThresholdPruner'},
 ('PGBM', 'WilcoxonPruner'): {'best_score': 0.11795102908774344,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.15,
   'max_leaves': 31,
   'min_split_gain': 0.0,
   'reg_lambda': 1.0,
   'feature_fraction': 0.5,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 0.11795102908774344,
  'test_rmse': 0.34343999343079346,
  'test_corr_coef': 0.9468668707705644,
  'pruner': 'WilcoxonPruner'}}
```

# **Conformal Predictions with Lightgbm**

```python
best_params = get_best_model_params(best_scores_autosampler, 'LightGBM')
```

```python
conformal_predictions_NEXCP(
    model_class=LGBMRegressor,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="LightGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/LightGBM.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_online_super(
    model_class=LGBMRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="LightGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/LightGBM.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_COP(
    model_class=LGBMRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="LightGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/LightGBM.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_EXTREME(
    model_class=LGBMRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="LightGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/LightGBM.xlsx",
    alpha=0.05,
    calib_size=0.3,
)

```

```python
conformal_predictions_AlphaNet(
    model_class=LGBMRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="LightGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/LightGBM.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with XGBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'XGBoost')
```

```python
conformal_predictions_NEXCP(
    model_class=XGBRegressor,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="XGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/XGBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_online_super(
    model_class=XGBRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="XGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/XGBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_COP(
    model_class=XGBRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="XGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/XGBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_EXTREME(
    model_class=XGBRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="XGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/XGBoost.xlsx",
    alpha=0.05,
    calib_size=0.3,
)

```

```python
conformal_predictions_AlphaNet(
    model_class=XGBRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="XGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/XGBoost.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with GPBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'GPBoost')
```

```python
conformal_predictions_NEXCP(
    model_class=GPBoostRegressor,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GPBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/GPBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_online_super(
    model_class=GPBoostRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GPBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/GPBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_COP(
    model_class=GPBoostRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GPBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/GPBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_EXTREME(
    model_class=GPBoostRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GPBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/GPBoost.xlsx",
    alpha=0.05,
    calib_size=0.3,
)

```

```python
conformal_predictions_AlphaNet(
    model_class=GPBoostRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GPBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/GPBoost.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with NGBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'NGBoost')
```

```python
conformal_predictions_NEXCP(
    model_class=NGBRegressor,
    best_params={**best_params, "Dist": Normal},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="NGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/NGBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_online_super(
    model_class=NGBRegressor,
    model_params={**best_params, "Dist": Normal},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="NGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/NGBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_COP(
    model_class=NGBRegressor,
    model_params={**best_params, "Dist": Normal},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="NGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/NGBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_EXTREME(
    model_class=NGBRegressor,
    model_params={**best_params, "Dist": Normal},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="NGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/NGBoost.xlsx",
    alpha=0.05,
    calib_size=0.3,
)

```

```python
conformal_predictions_AlphaNet(
    model_class=NGBRegressor,
    model_params={**best_params, "Dist": Normal},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="NGBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/NGBoost.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with Gradient Boosting**

```python
best_params = get_best_model_params(best_scores_autosampler, 'GradientBoosting')
```

```python
conformal_predictions_NEXCP(
    model_class=GradientBoostingRegressor,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GradientBoosting",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/Gradient Boosting.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_online_super(
    model_class=GradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GradientBoosting",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/Gradient Boosting.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_COP(
    model_class=GradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GradientBoosting",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/Gradient Boosting.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_EXTREME(
    model_class=GradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GradientBoosting",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/Gradient Boosting.xlsx",
    alpha=0.05,
    calib_size=0.3,
)

```

```python
conformal_predictions_AlphaNet(
    model_class=GradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="GradientBoosting",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/Gradient Boosting.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with CatBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'CatBoost')
```

```python
conformal_predictions_NEXCP(
    model_class=CatBoostRegressor,
    best_params={**best_params, "verbose": False},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="CatBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/CatBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_online_super(
    model_class=CatBoostRegressor,
    model_params={**best_params, "verbose": False},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="CatBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/CatBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_COP(
    model_class=CatBoostRegressor,
    model_params={**best_params, "verbose": False},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="CatBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/CatBoost.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_EXTREME(
    model_class=CatBoostRegressor,
    model_params={**best_params, "verbose": False},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="CatBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/CatBoost.xlsx",
    alpha=0.05,
    calib_size=0.3,
)

```

```python
conformal_predictions_AlphaNet(
    model_class=CatBoostRegressor,
    model_params={**best_params, "verbose": False},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="CatBoost",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/CatBoost.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with HistGradientBoosting**

```python
best_params = get_best_model_params(best_scores_autosampler, 'HGBR')
```

```python
conformal_predictions_NEXCP(
    model_class=HistGradientBoostingRegressor,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="HGBR",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/HGBM.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_online_super(
    model_class=HistGradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="HGBR",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/HGBM.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_COP(
    model_class=HistGradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="HGBR",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/HGBM.xlsx",
    alpha=0.05,
)

```

```python
conformal_predictions_EXTREME(
    model_class=HistGradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="HGBR",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/HGBM.xlsx",
    alpha=0.05,
    calib_size=0.3,
)

```

```python
conformal_predictions_AlphaNet(
    model_class=HistGradientBoostingRegressor,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="HGBR",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/HGBM.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with PGBM**

```python
class PGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X, y):
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        y_ = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
        self.model = PGBM()
        self.model.train(
            train_set=(X_, y_),
            objective=mseloss_objective,
            metric=rmseloss_metric,
            params=self.params
        )
        return self

    def predict(self, X):
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        return self.model.predict(X_).numpy()

```

```python
# Assuming best_params is obtained correctly
best_params = get_best_model_params(best_scores_autosampler, 'PGBM')

# Remove any conflicting parameters from best_params
incompatible_keys = ['Dist', 'Score']
for key in incompatible_keys:
    best_params.pop(key, None)

# Initialize PGBM model
pgbm_model = PGBM()
def mseloss_objective(yhat, y, sample_weight=None):
    # Ensure that yhat and y are PyTorch tensors
    if not torch.is_tensor(yhat):
        yhat = torch.from_numpy(np.array(yhat)).float()
    if not torch.is_tensor(y):
        y = torch.from_numpy(np.array(y)).float()
    gradient = yhat - y
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    # Ensure that yhat and y are PyTorch tensors
    if not torch.is_tensor(yhat):
        yhat = torch.from_numpy(np.array(yhat)).float()
    if not torch.is_tensor(y):
        y = torch.from_numpy(np.array(y)).float()
    loss = torch.sqrt(torch.mean((yhat - y) ** 2))
    return loss

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Fit the model
pgbm_model.train((X_train, y_train), objective=mseloss_objective, metric=rmseloss_metric, params=best_params)

# Predict the distribution
pred_dist = pgbm_model.predict_dist(X_test)

# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# DataFrame to store predictions
predictions_PGBM_df = pd.DataFrame()

# Calculate and store quantiles
for q in quantiles:
    predictions_PGBM_df[q] = np.quantile(pred_dist, q, axis=0)

# Add actual str to the DataFrame
predictions_PGBM_df['Actual'] = y_test.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_PGBM_df.head())
```

```python
class validate:
    @staticmethod
    def coverage(int_pred, y_test):
        return {strat_name: np.mean((y_test >= int_pred[strat_name][:, 0]) & (y_test <= int_pred[strat_name][:, 1])) for strat_name in int_pred}
    @staticmethod
    def width(int_pred): # Corrected: Removed y_test argument as it's not defined and not used
        return {strat_name: np.mean(int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]) for strat_name in int_pred}
    @staticmethod
    def rmse(y_pred, y_test):
        return {strat_name: np.sqrt(mean_squared_error(y_test, y_pred[strat_name])) for strat_name in y_pred}
    @staticmethod
    def cwc(int_pred, y_test, miscoverage):
        widths = validate.width(int_pred)
        coverages = validate.coverage(int_pred, y_test)
        return {strat_name: widths[strat_name] * (1 + (coverages[strat_name] < (1 - miscoverage)) * (1 - coverages[strat_name])) for strat_name in int_pred}

    @staticmethod
    def cond_coverage(int_pred, y_test, num_bins=10):
        cond_coverages = {}
        for strat_name in int_pred:
            widths = int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]
            bin_edges = np.histogram_bin_edges(widths, bins=num_bins)
            digitized_widths = np.digitize(widths, bin_edges)
            strategy_cond_coverage = []
            for bin_num in range(1, num_bins + 1):
                indices_in_bin = np.where(digitized_widths == bin_num)[0]
                if len(indices_in_bin) > 0:
                    bin_coverage = np.mean((y_test[indices_in_bin] >= int_pred[strat_name][indices_in_bin, 0]) & (y_test[indices_in_bin] <= int_pred[strat_name][indices_in_bin, 1]))
                    strategy_cond_coverage.append(bin_coverage)
                else:
                    strategy_cond_coverage.append(np.nan) # or handle empty bins as needed
            cond_coverages[strat_name] = strategy_cond_coverage
        return cond_coverages


class visualize:
    @staticmethod
    def coverage(int_pred, y_test):
        return {strat_name: np.mean((y_test >= int_pred[strat_name][:, 0]) & (y_test.to_numpy() <= int_pred[strat_name][:, 1])) for strat_name in int_pred}
    @staticmethod
    def width(int_pred):
        return {strat_name: np.mean(int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]) for strat_name in int_pred}
    @staticmethod
    def rmse(y_pred, y_test):
        return {strat_name: np.sqrt(mean_squared_error(y_test, y_pred[strat_name])) for strat_name in y_pred}
    @staticmethod
    def cwc(int_pred, y_test, miscoverage):
        widths = visualize.width(int_pred)
        coverages = visualize.coverage(int_pred, y_test)
        return {strat_name: widths[strat_name] * (1 + (coverages[strat_name] < (1 - miscoverage)) * (1 - coverages[strat_name])) for strat_name in int_pred}
    @staticmethod
    def goodness(y_true, y_pred, y_pred_low, y_pred_up, coverage, width, rmse, cwc, ax=None, title="Goodness Plot"):
        if ax is None:
            fig, ax = plt.subplots()
        n_samples = len(y_true)
        y_true_np = y_true.ravel() # Ensure y_true is numpy array and flattened

        # Determine points inside and outside the interval
        inside_interval = (y_true_np >= y_pred_low) & (y_true_np <= y_pred_up)
        outside_interval = ~inside_interval

        # Plot points inside the interval in green
        ax.scatter(y_true_np[inside_interval], y_pred[inside_interval], color='green', s=10, label='Inside Interval')
        # Plot points outside the interval in orange
        ax.scatter(y_true_np[outside_interval], y_pred[outside_interval], color='orange', s=10, label='Outside Interval')

        # Plot all ground truth values as blue crosses
        ax.scatter(y_true_np, y_true_np, color='blue', marker='x', s=30, linewidths=0.7, label='Ground Truth Values')


        ax.vlines(y_true_np, y_pred_low, y_pred_up, color='gray', alpha=0.5, label='Prediction Intervals')
        ax.plot([min(y_true_np), max(y_true_np)], [min(y_true_np), max(y_true_np)], linestyle='--', color='blue', label='Ideal Prediction Line') # Changed label for clarity
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.set_title(f'{title}\nCoverage: {coverage*100:.2f}%, Width: {width:.2f}, RMSE: {rmse:.2f}, CWC: {cwc:.2f}')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax

    @staticmethod
    def width_size_occurrence(int_pred, train_intervals, num_bins=10, ax=None, x_lim=None, title="Width Size Occurrence"):
        if ax is None:
            fig, ax = plt.subplots()

        # Calculate interval widths for test and train sets
        widths_test = np.abs(int_pred[:, 1] - int_pred[:, 0])
        widths_train = np.abs(train_intervals[:, 1] - train_intervals[:, 0])

        # Plot histograms for both train and test widths
        ax.hist(widths_test, bins=num_bins, alpha=0.6, label='Test Interval Widths', color='red')
        ax.hist(widths_train, bins=num_bins, alpha=0.4, label='Train Interval Widths', color='blue')


        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel('Interval Width')
        ax.set_ylabel('Occurrence (Frequency)')
        ax.set_title(f'{title} - Interval Width Occurrence')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax

    @staticmethod
    def coverage_by_width(y_test, int_pred, miscoverage, cond_coverages, num_bins=10, ax=None, title="Coverage vs Width"):
        if ax is None:
            fig, ax = plt.subplots()

        widths = int_pred[:, 1] - int_pred[:, 0]
        bin_edges = np.histogram_bin_edges(widths, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        ax.bar(bin_centers, cond_coverages, width=bin_width, color='blue', alpha=0.7, edgecolor='black', linewidth=0.8, label='Conditional Coverage')
        ax.axhline(1 - miscoverage, color='red', linestyle='--', label=f'Nominal Coverage ({(1 - miscoverage)*100:.0f}%)')

        ax.set_xlabel('Interval Width')
        ax.set_ylabel('Conditional Coverage')
        ax.set_title(f'{title} - Coverage vs Interval Width')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax




```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


import os
def to_numpy_1d(y):
    return y.to_numpy().ravel() if hasattr(y, "to_numpy") else np.asarray(y).ravel()


def weighted_quantile(values, weights, q):
    sorter = np.argsort(values)
    values, weights = values[sorter], weights[sorter]
    cdf = np.cumsum(weights) / np.sum(weights)
    return np.interp(q, cdf, values)



# ============================================================
# NexCP for PGBM (Excel-safe, NexCP-style, NO RETURNS)
# ============================================================

def conformal_predictions_NEXCP_PGBM(
    model_class,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="PGBM",
    excel_file_path=None,
    alpha=0.05,
    decay=0.99,
    n_folds=5,
    n_bootstrap=50,
    seed=123,
):

    warnings.filterwarnings("ignore")
    np.random.seed(seed)

    # ========================================================
    # data conversion (PGBM-specific)
    # ========================================================
    Xtr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    Xte = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)
    ytr = to_numpy_1d(y_train)
    yte = to_numpy_1d(y_test)

    STRATEGIES = {
        "NexCP-Split": "NexCP Split",
        "NexCP-Full": "NexCP Full",
        "NexCP-J+aB": "NexCP Jackknife+ aB",
        "NexCP-CV+": "NexCP CV+",
    }

    y_pred, int_pred = {}, {}

    # ========================================================
    # Fit base model
    # ========================================================
    base_model = model_class(**best_params)
    base_model.fit(Xtr, ytr)

    yhat_tr = base_model.predict(Xtr)
    yhat_te = base_model.predict(Xte)
    residuals = np.abs(ytr - yhat_tr)

    # ========================================================
    # NexCP – Split (Exponentially Weighted)
    # ========================================================
    weights = decay ** np.arange(len(residuals) - 1, -1, -1)
    q = weighted_quantile(residuals, weights, 1 - alpha)

    y_pred["NexCP-Split"] = yhat_te
    int_pred["NexCP-Split"] = np.c_[yhat_te - q, yhat_te + q]

    # ========================================================
    # NexCP – Full
    # ========================================================
    q = np.quantile(residuals, 1 - alpha)
    y_pred["NexCP-Full"] = yhat_te
    int_pred["NexCP-Full"] = np.c_[yhat_te - q, yhat_te + q]

    # ========================================================
    # NexCP – Jackknife+ aB
    # ========================================================
    rng = np.random.default_rng(seed)
    lowers, uppers, preds = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(Xtr), len(Xtr), replace=True)
        m = model_class(**best_params)
        m.fit(Xtr[idx], ytr[idx])

        q = np.quantile(np.abs(ytr[idx] - m.predict(Xtr[idx])), 1 - alpha)
        p = m.predict(Xte)

        preds.append(p)
        lowers.append(p - q)
        uppers.append(p + q)

    y_pred["NexCP-J+aB"] = np.mean(preds, axis=0)
    int_pred["NexCP-J+aB"] = np.c_[np.min(lowers, axis=0), np.max(uppers, axis=0)]

    # ========================================================
    # NexCP – CV+
    # ========================================================
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    lowers, uppers, preds = [], [], []

    for tr_idx, cal_idx in kf.split(Xtr):
        m = model_class(**best_params)
        m.fit(Xtr[tr_idx], ytr[tr_idx])

        q = np.quantile(np.abs(ytr[cal_idx] - m.predict(Xtr[cal_idx])), 1 - alpha)
        p = m.predict(Xte)

        preds.append(p)
        lowers.append(p - q)
        uppers.append(p + q)

    y_pred["NexCP-CV+"] = np.mean(preds, axis=0)
    int_pred["NexCP-CV+"] = np.c_[np.mean(lowers, axis=0), np.mean(uppers, axis=0)]

    # ========================================================
    # Metrics + Summary
    # ========================================================
    summary_rows = []
    values_dict = {}

    print("\n===== NEXCP (PGBM) RESULTS =====")

    for s in STRATEGIES:
        ymin, ymax = int_pred[s].T
        yhat = y_pred[s]

        coverage = np.mean((yte >= ymin) & (yte <= ymax))
        width = np.mean(ymax - ymin)
        rmse = np.sqrt(mean_squared_error(yte, yhat))

        print(f"{STRATEGIES[s]} - Coverage: {coverage*100:.3f}% | Width: {width:.3f}")

        summary_rows.append({
            "Strategy": STRATEGIES[s],
            "Coverage": coverage,
            "Width": width,
            "RMSE": rmse,
        })

        values_dict[f"{s}_true"] = yte
        values_dict[f"{s}_pred"] = yhat
        values_dict[f"{s}_lower"] = ymin
        values_dict[f"{s}_upper"] = ymax

    summary_df = pd.DataFrame(summary_rows)
    values_df = pd.DataFrame(values_dict)

    # ========================================================
    # Plotting (SAVE ONLY — NO SHOW)
    # ========================================================
    idx = np.arange(len(yte))
    plot_buffers = {}

    for s in STRATEGIES:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.scatter(idx, yte, s=10, color="blue", alpha=0.7, label="True")
        ax.fill_between(idx, int_pred[s][:, 0], int_pred[s][:, 1],
                        color="gray", alpha=0.5, label="Prediction interval")
        ax.plot(idx, y_pred[s], color="red", lw=1, label="Predicted mean")

        ax.set_title(f"{STRATEGIES[s]} ({model_name})")
        ax.set_xlabel("Sample Number")
        ax.set_ylabel("str")
        ax.legend(loc="upper left")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plot_buffers[s] = buf

        plt.close(fig)

    # ========================================================
    # Excel Export (append-safe)
    # ========================================================
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        # ---- Summary sheet ----
        sheet_name = "NEXCP_PGBM_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- Plot sheets ----
        for s, buf in plot_buffers.items():
            img = Image(buf)
            sheet_name = f"NEXCP_PGBM_Plot_{s}"
            base = sheet_name
            i = 1
            while sheet_name in wb.sheetnames:
                sheet_name = f"{base}_{i}"
                i += 1
            ws = wb.create_sheet(sheet_name)
            ws.add_image(img, "A1")

        # ---- Values sheet ----
        sheet_name = "NEXCP_PGBM_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))


conformal_predictions_NEXCP_PGBM(
    model_class=PGBMWrapper,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="PGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/PGBM.xlsx",
    alpha=0.05,
)

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_online_super_PGBM(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="PGBM",
    excel_file_path=None,
    alpha=0.05,
    strategies="all",
    window_min=50,
    window_max=500,
    window_grid=(25, 50, 100, 200, 400),
    n_bootstrap=30,
    seed=123,
):

    warnings.filterwarnings("ignore")
    np.random.seed(seed)

    STRATEGIES = {
        "Online-Split": "Online Split",
        "Online-CV+": "Online CV+",
        "Online-J+aB": "Online Jackknife+ aB",
        "SAOCP": "SAOCP",
        "SF-OGD": "SF-OGD",
        "FACI": "FACI",
        "MFCS-Split": "MFCS Split",
        "MFCS-Full": "MFCS Full",
    }

    if strategies == "all":
        strategies = list(STRATEGIES.keys())

    # --------------------------------------------------------
    # Safe NumPy conversion (PGBM-specific)
    # --------------------------------------------------------
    Xtr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    Xte = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)
    ytr = to_numpy_1d(y_train)
    yte = to_numpy_1d(y_test)

    # --------------------------------------------------------
    # Base model (trained once)
    # --------------------------------------------------------
    base_model = model_class(**model_params)
    base_model.fit(Xtr, ytr)

    yhat_train = base_model.predict(Xtr)

    # --------------------------------------------------------
    # Streaming predictions
    # --------------------------------------------------------
    X_all = np.vstack([Xtr, Xte])
    y_all = np.concatenate([ytr, yte])

    n_train = len(Xtr)
    T = len(y_all)

    y_hat_all = np.zeros(T)
    y_hat_all[:n_train] = yhat_train

    for t in range(n_train, T):
        y_hat_all[t] = base_model.predict(X_all[t:t + 1])[0]

    residuals = np.abs(y_all - y_hat_all)

    y_pred, int_pred = {}, {}

    # ========================================================
    # Online Split
    # ========================================================
    if "Online-Split" in strategies:
        q = np.quantile(residuals[:n_train], 1 - alpha)
        y_pred["Online-Split"] = y_hat_all[n_train:]
        int_pred["Online-Split"] = np.c_[y_pred["Online-Split"] - q,
                                          y_pred["Online-Split"] + q]

    # ========================================================
    # Online CV+
    # ========================================================
    if "Online-CV+" in strategies:
        lower, upper = [], []
        for t in range(n_train, T):
            q = np.quantile(residuals[:t], 1 - alpha)
            lower.append(y_hat_all[t] - q)
            upper.append(y_hat_all[t] + q)
        y_pred["Online-CV+"] = y_hat_all[n_train:]
        int_pred["Online-CV+"] = np.c_[lower, upper]

    # ========================================================
    # Online Jackknife+ aB
    # ========================================================
    if "Online-J+aB" in strategies:
        lowers, uppers, preds = [], [], []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_train, n_train, replace=True)
            m = model_class(**model_params)
            m.fit(Xtr[idx], ytr[idx])

            q = np.quantile(np.abs(ytr[idx] - m.predict(Xtr[idx])), 1 - alpha)
            p = m.predict(Xte)

            preds.append(p)
            lowers.append(p - q)
            uppers.append(p + q)

        y_pred["Online-J+aB"] = np.mean(preds, axis=0)
        int_pred["Online-J+aB"] = np.c_[np.min(lowers, axis=0),
                                        np.max(uppers, axis=0)]

    # ========================================================
    # SAOCP
    # ========================================================
    if "SAOCP" in strategies:
        lower, upper = [], []
        for t in range(n_train, T):
            w = min(max(window_min, int(t / 2)), window_max)
            start = max(0, t - w)
            q = np.quantile(residuals[start:t], 1 - alpha)
            lower.append(y_hat_all[t] - q)
            upper.append(y_hat_all[t] + q)
        y_pred["SAOCP"] = y_hat_all[n_train:]
        int_pred["SAOCP"] = np.c_[lower, upper]

    # ========================================================
    # SF-OGD
    # ========================================================
    if "SF-OGD" in strategies:
        q_t = np.quantile(residuals[:n_train], 1 - alpha)
        lower, upper = [], []
        for t in range(n_train, T):
            grad = alpha - (residuals[t] <= q_t)
            eta = 1.0 / np.sqrt(t - n_train + 1)
            q_t = max(0.0, q_t - eta * grad)
            lower.append(y_hat_all[t] - q_t)
            upper.append(y_hat_all[t] + q_t)
        y_pred["SF-OGD"] = y_hat_all[n_train:]
        int_pred["SF-OGD"] = np.c_[lower, upper]

    # ========================================================
    # FACI
    # ========================================================
    if "FACI" in strategies:
        windows = list(window_grid)
        losses = np.zeros(len(windows))
        lower, upper = [], []
        for t in range(n_train, T):
            r_t = residuals[t]
            qs = []
            for i, w in enumerate(windows):
                start = max(0, t - w)
                q = np.quantile(residuals[start:t], 1 - alpha)
                qs.append(q)
                losses[i] += alpha * q + (r_t - q) * (r_t > q)
            q_star = qs[np.argmin(losses)]
            lower.append(y_hat_all[t] - q_star)
            upper.append(y_hat_all[t] + q_star)
        y_pred["FACI"] = y_hat_all[n_train:]
        int_pred["FACI"] = np.c_[lower, upper]

    # ========================================================
    # MFCS-Split ✅ (ADDED)
    # ========================================================
    if "MFCS-Split" in strategies:
        q = np.quantile(residuals[:n_train], 1 - alpha)
        y_pred["MFCS-Split"] = y_hat_all[n_train:]
        int_pred["MFCS-Split"] = np.c_[y_pred["MFCS-Split"] - q,
                                       y_pred["MFCS-Split"] + q]

    # ========================================================
    # MFCS-Full
    # ========================================================
    if "MFCS-Full" in strategies:
        lower, upper, preds = [], [], []
        for t in range(n_train, T):
            m = model_class(**model_params)
            m.fit(X_all[:t], y_all[:t])
            yhat = m.predict(X_all[t:t + 1])[0]
            q = np.quantile(np.abs(y_all[:t] - m.predict(X_all[:t])), 1 - alpha)
            preds.append(yhat)
            lower.append(yhat - q)
            upper.append(yhat + q)
        y_pred["MFCS-Full"] = np.array(preds)
        int_pred["MFCS-Full"] = np.c_[lower, upper]

    # ========================================================
    # Metrics + Summary
    # ========================================================
    summary_rows = []
    values_dict = {}

    print("\n===== ONLINE CP (PGBM) RESULTS =====")

    for s in y_pred:
        ymin, ymax = int_pred[s].T
        yhat = y_pred[s]

        coverage = np.mean((yte >= ymin) & (yte <= ymax))
        width = np.mean(ymax - ymin)
        rmse = np.sqrt(mean_squared_error(yte, yhat))

        print(f"{STRATEGIES[s]} - Coverage: {coverage*100:.3f}% | Width: {width:.3f}")

        summary_rows.append({
            "Strategy": STRATEGIES[s],
            "Coverage": coverage,
            "Width": width,
            "RMSE": rmse,
        })

        values_dict[f"{s}_true"] = yte
        values_dict[f"{s}_pred"] = yhat
        values_dict[f"{s}_lower"] = ymin
        values_dict[f"{s}_upper"] = ymax

    summary_df = pd.DataFrame(summary_rows)
    values_df = pd.DataFrame(values_dict)

    # ========================================================
    # Plotting (SAVE ONLY — NO SHOW)
    # ========================================================
    idx = np.arange(len(yte))
    plot_buffers = {}

    for s in y_pred:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.scatter(idx, yte, s=10, color="blue", alpha=0.7, label="True")
        ax.fill_between(idx, int_pred[s][:, 0], int_pred[s][:, 1],
                        color="gray", alpha=0.5, label="Prediction interval")
        ax.plot(idx, y_pred[s], color="red", lw=1, label="Predicted mean")

        ax.set_title(f"{STRATEGIES[s]} ({model_name})")
        ax.set_xlabel("Sample Number")
        ax.set_ylabel("str")
        ax.legend(loc="upper left")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plot_buffers[s] = buf

        plt.close(fig)

    # ========================================================
    # Excel Export (append-safe)
    # ========================================================
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        # ---- Summary ----
        sheet_name = "ONLINECP_PGBM_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- Plots ----
        for s, buf in plot_buffers.items():
            img = Image(buf)
            sheet_name = f"ONLINECP_PGBM_Plot_{s}"
            base = sheet_name
            i = 1
            while sheet_name in wb.sheetnames:
                sheet_name = f"{base}_{i}"
                i += 1
            ws = wb.create_sheet(sheet_name)
            ws.add_image(img, "A1")

        # ---- Values ----
        sheet_name = "ONLINECP_PGBM_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))


conformal_predictions_online_super_PGBM(
    model_class=PGBMWrapper,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="PGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/PGBM.xlsx",
    alpha=0.05,
    strategies="all",
)

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_COP_PGBM(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="PGBM",
    excel_file_path=None,
    alpha=0.05,
    lr=0.05,
    T_burnin=50,
    scale=0.5,
    random_state=123,
):

    warnings.filterwarnings("ignore")
    np.random.seed(random_state)

    # =========================================================
    # Safe NumPy conversion (PGBM-specific)
    # =========================================================
    Xtr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    Xte = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)
    ytr = to_numpy_1d(y_train)
    yte = to_numpy_1d(y_test)

    # =========================================================
    # Train base PGBM model
    # =========================================================
    model = model_class(**model_params)
    model.fit(Xtr, ytr)

    y_pred_test = model.predict(Xte).ravel()
    y_pred_train = model.predict(Xtr).ravel()

    # Absolute residuals (symmetric COP)
    scores_test = np.abs(yte - y_pred_test)
    scores_train = np.abs(ytr - y_pred_train)

    T_test = len(scores_test)

    # =========================================================
    # COP initialization
    # =========================================================
    init_q = np.quantile(scores_train, 1 - alpha)

    qs = np.zeros(T_test)
    qts = np.zeros(T_test)
    integrators = np.zeros(T_test)
    covereds = np.zeros(T_test)

    qs[0] = init_q
    qts[0] = init_q

    # =========================================================
    # Online COP loop
    # =========================================================
    for t in range(T_test - 1):

        covereds[t] = 1 if qs[t] >= scores_test[t] else 0

        # Standard COP gradient
        grad = alpha if covereds[t] else -(1 - alpha)

        # Distribution-informed (optimistic) gradient
        if t < T_burnin:
            grad_i = 0.0
        else:
            window_s = scores_test[t - T_burnin : t]
            target = qts[t] - lr * grad
            grad_i = np.mean(window_s <= target) - (1 - alpha)

        # Updates
        integrator = -scale * grad_i
        qts[t + 1] = qts[t] - lr * grad
        integrators[t + 1] = lr * integrator
        qs[t + 1] = max(0.0, qts[t + 1] + integrators[t + 1])

    covereds[-1] = 1 if qs[-1] >= scores_test[-1] else 0

    # =========================================================
    # Prediction intervals
    # =========================================================
    ymin = y_pred_test - qs
    ymax = y_pred_test + qs

    # =========================================================
    # Metrics
    # =========================================================
    rmse = np.sqrt(mean_squared_error(yte, y_pred_test))
    coverage = np.mean(covereds)
    avg_width = np.mean(ymax - ymin)

    print(f"\n===== COP (PGBM) =====")
    print(f"Coverage : {coverage*100:.3f}%")
    print(f"Width    : {avg_width:.3f}")
    print(f"RMSE     : {rmse:.3f}")
    print(f"Final q  : {qs[-1]:.3f}")

    # =========================================================
    # Tables
    # =========================================================
    summary_df = pd.DataFrame([{
        "Strategy": "COP",
        "Coverage": coverage,
        "Width": avg_width,
        "RMSE": rmse,
        "Alpha": alpha,
        "LR": lr,
        "Scale": scale,
    }])

    values_df = pd.DataFrame({
        "sample_index": np.arange(len(yte)),
        "y_true": yte,
        "y_pred": y_pred_test,
        "ymin": ymin,
        "ymax": ymax,
        "width": ymax - ymin,
        "is_covered": covereds.astype(int),
        "score": scores_test,
        "adaptive_q": qs,
    })

    # =========================================================
    # Plot (SAVE ONLY)
    # =========================================================
    idx = np.arange(len(yte))
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(idx, yte, s=10, color="blue", alpha=0.7, label="True")
    ax.fill_between(idx, ymin, ymax, color="gray", alpha=0.5, label="COP Interval")
    ax.plot(idx, y_pred_test, color="red", lw=1, label="Predicted mean")


    ax.set_title(f"COP ({model_name})")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("str")
    ax.legend(loc="upper left")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    # =========================================================
    # Excel export (append-safe)
    # =========================================================
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        # ---- Summary ----
        sheet_name = "COP_PGBM_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- Plot ----
        img = Image(buf)
        sheet_name = "COP_PGBM_Plot"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1
        ws = wb.create_sheet(sheet_name)
        ws.add_image(img, "A1")

        # ---- Values ----
        sheet_name = "COP_PGBM_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))

    return summary_df

conformal_predictions_COP_PGBM(
    model_class=PGBMWrapper,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="PGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/PGBM.xlsx",
    alpha=0.05,
)

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_EXTREME_PGBM(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="PGBM",
    excel_file_path=None,
    alpha=0.05,
    calib_size=0.2,
    gpd_threshold_quantile=0.95,
    n_bootstrap=30,
    random_state=123,
):

    warnings.filterwarnings("ignore")
    np.random.seed(random_state)

    # --------------------------------------------------------
    # Safe NumPy conversion (PGBM-specific)
    # --------------------------------------------------------
    Xtr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    Xte = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)
    ytr = to_numpy_1d(y_train)
    yte = to_numpy_1d(y_test)

    # --------------------------------------------------------
    # Train / calibration split
    # --------------------------------------------------------
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        Xtr, ytr, test_size=calib_size, random_state=random_state, shuffle=True
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------
    model = model_class(**model_params)
    model.fit(X_fit, y_fit)

    y_pred_test = model.predict(Xte)
    y_pred_cal = model.predict(X_cal)

    # --------------------------------------------------------
    # Upper-tail nonconformity scores
    # --------------------------------------------------------
    scores = y_cal - y_pred_cal
    scores = scores[scores > 0]

    if len(scores) < 30:
        raise RuntimeError("Too few exceedances for EVT fitting.")

    # --------------------------------------------------------
    # Peaks-over-threshold
    # --------------------------------------------------------
    u = np.quantile(scores, gpd_threshold_quantile)
    excesses = scores[scores > u] - u

    # --------------------------------------------------------
    # Bootstrap EVT quantile
    # --------------------------------------------------------
    boot_q = []
    n_exc = len(excesses)
    n_scores = len(scores)

    for _ in range(n_bootstrap):
        res = np.random.choice(excesses, size=n_exc, replace=True)
        xi, _, sigma = genpareto.fit(res, floc=0)
        q = u + (sigma / xi) * ((n_scores / (n_scores * alpha)) ** xi - 1)
        boot_q.append(q)

    q_extreme = np.quantile(boot_q, 0.95)

    # --------------------------------------------------------
    # FINITE prediction intervals (for plotting & width)
    # --------------------------------------------------------
    ymin = y_pred_test - q_extreme
    ymax = y_pred_test + q_extreme

    # --------------------------------------------------------
    # Metrics (one-sided coverage)
    # --------------------------------------------------------
    rmse = np.sqrt(mean_squared_error(yte, y_pred_test))
    coverage = np.mean(yte <= ymax)
    width = q_extreme

    print(f"\n===== EXTREME CP (PGBM) =====")
    print(f"Coverage     : {coverage*100:.3f}%")
    print(f"Width        : {width:.3f}")
    print(f"RMSE         : {rmse:.3f}")
    print(f"q_extreme    : {q_extreme:.3f}")
    print(f"threshold_u  : {u:.3f}")
    print(f"#Exceedances : {len(scores)}")

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------
    summary_df = pd.DataFrame([{
        "Strategy": "EXTREME CP",
        "Coverage": coverage,
        "Width": width,
        "RMSE": rmse,
        "q_extreme": q_extreme,
        "threshold_u": u,
        "n_exceedances": len(scores),
    }])

    # --------------------------------------------------------
    # Values table
    # --------------------------------------------------------
    values_df = pd.DataFrame({
        "sample_index": np.arange(len(yte)),
        "y_true": yte,
        "y_pred": y_pred_test,
        "ymin": ymin,
        "ymax": ymax,
        "width": ymax - ymin,
        "is_covered": (yte <= ymax).astype(int),
        "residual": yte - y_pred_test,
    })

    # --------------------------------------------------------
    # Plot (SAVE ONLY — NO SHOW)
    # --------------------------------------------------------
    idx = np.arange(len(yte))
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(idx, yte, s=10, color="blue", alpha=0.7, label="True")
    ax.fill_between(idx, ymin, ymax, alpha=0.4, label="Extreme PI")
    ax.plot(idx, y_pred_test, color="red", lw=1, label="Prediction")

    ax.set_title(f"Extreme CP ({model_name})")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("str")
    ax.legend(loc="upper left")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    # --------------------------------------------------------
    # Excel export (append-safe)
    # --------------------------------------------------------
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        # ---- Summary ----
        sheet_name = "EXTREMECP_PGBM_Summary"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- Plot ----
        img = Image(buf)
        sheet_name = "EXTREMECP_PGBM_Plot"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        ws.add_image(img, "A1")

        # ---- Values ----
        sheet_name = "EXTREMECP_PGBM_Values"
        base = sheet_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet_name)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))


conformal_predictions_EXTREME_PGBM(
    model_class=PGBMWrapper,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="PGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/PGBM.xlsx",
    alpha=0.05,
    calib_size=0.3,
)
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_AlphaNet_PGBM(
    model_class,
    model_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="PGBM",
    excel_file_path=None,
    alpha=0.05,
    lambda_reg=20.0,
    num_epochs=50,
    batch_size=32,
    lr_alpha=1e-3,
    calib_fraction=0.5,
    num_runs=4,
    random_state=123,
):
    warnings.filterwarnings("ignore")
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_start_total = time.time()

    # =========================================================
    # data conversion (PGBM-specific)
    # =========================================================
    Xtr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    Xte = X_test.to_numpy()  if hasattr(X_test,  "to_numpy") else np.asarray(X_test)
    ytr = to_numpy_1d(y_train)
    yte = to_numpy_1d(y_test)

    # =========================================================
    # 1. Split training data → fit + calibration
    # =========================================================
    n_total = len(Xtr)
    n_fit   = int(n_total * (1 - calib_fraction))
    perm    = np.random.permutation(n_total)

    fit_idx   = perm[:n_fit]
    calib_idx = perm[n_fit:]

    X_fit,   y_fit   = Xtr[fit_idx],   ytr[fit_idx]
    X_calib, y_calib = Xtr[calib_idx],  ytr[calib_idx]
    n_calib = len(X_calib)

    alpha_min = 1.0 / (n_calib + 1) + 0.01
    print(f"  [Info] n_calib={n_calib} | alpha_min={alpha_min:.4f}")

    # =========================================================
    # 2. Train base PGBM model
    # =========================================================
    print(f"  [Step 2/9] Training base {model_name} model...")
    t0 = time.time()
    base_model = model_class(**model_params)
    base_model.fit(X_fit, y_fit)
    print(f"  [Step 2/9] Done in {_fmt_time(time.time()-t0)}")

    def _safe_predict(m, X):
        p = m.predict(X)
        return p.ravel() if hasattr(p, "ravel") else np.asarray(p).ravel()

    # =========================================================
    # 3. PRECOMPUTE all LOO scores
    # =========================================================
    print(f"  [Step 3/9] Precomputing {n_calib} LOO score vectors...")
    t0 = time.time()

    precomputed_scores = []
    precomputed_sums   = []

    iterator = tqdm(range(n_calib), desc="    LOO precompute", leave=False) if HAS_TQDM else range(n_calib)

    for i in iterator:
        loo_X = np.delete(X_calib, i, axis=0)
        loo_y = np.delete(y_calib, i, axis=0)
        scores_loo = np.abs(_safe_predict(base_model, loo_X) - loo_y)

        scores_tensor = torch.tensor(scores_loo, dtype=torch.float32).to(device)
        precomputed_scores.append(scores_tensor)
        precomputed_sums.append(scores_loo.sum())

    elapsed_precompute = time.time() - t0
    print(f"  [Step 3/9] Done in {_fmt_time(elapsed_precompute)}")

    X_alpha = torch.tensor(
        np.array(precomputed_sums).reshape(-1, 1),
        dtype=torch.float32,
    ).to(device)

    # =========================================================
    # 4. Train AlphaNet
    # =========================================================
    print(f"  [Step 4/9] Training AlphaNet: {num_runs} runs × {num_epochs} epochs")

    best_net  = None
    best_loss = np.inf
    history   = {"losses": [], "sizes": [], "alphas": []}

    t0_train = time.time()

    for run in range(num_runs):
        np.random.seed(run)
        torch.manual_seed(run)

        alpha_net = AlphaNet(input_dim=1, alpha_min=alpha_min).to(device)
        optimizer = optim.Adam(alpha_net.parameters(), lr=lr_alpha)

        Dataset = TensorDataset(X_alpha, torch.arange(n_calib))
        loader  = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

        run_losses, run_sizes, run_alphas = [], [], []

        epoch_iter = (
            tqdm(range(num_epochs), desc=f"    Run {run+1}/{num_runs}", leave=False)
            if HAS_TQDM else range(num_epochs)
        )

        for epoch in epoch_iter:
            t_epoch_start = time.time()
            epoch_loss = epoch_size = epoch_alpha = 0.0

            for x_batch, idx_batch in loader:
                x_batch = x_batch.to(device)
                alpha_pred = alpha_net(x_batch)

                batch_sizes = []
                for j, idx in enumerate(idx_batch):
                    scores_t = precomputed_scores[idx.item()]
                    size_j = _interval_size(scores_t, alpha_pred[j])
                    batch_sizes.append(size_j)

                batch_sizes = torch.stack(batch_sizes)
                loss = (batch_sizes + lambda_reg * alpha_pred).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss  += loss.item()
                epoch_size  += batch_sizes.mean().item()
                epoch_alpha += alpha_pred.mean().item()

            n_batches = len(loader)
            run_losses.append(epoch_loss  / n_batches)
            run_sizes.append(epoch_size   / n_batches)
            run_alphas.append(epoch_alpha / n_batches)

            epoch_time = time.time() - t_epoch_start

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                epochs_done  = (run * num_epochs) + epoch + 1
                epochs_total = num_runs * num_epochs
                elapsed_train = time.time() - t0_train
                eta = (elapsed_train / epochs_done) * (epochs_total - epochs_done)

                print(
                    f"  λ={lambda_reg} | Run {run+1}/{num_runs} | "
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Loss: {run_losses[-1]:.4f} | "
                    f"Size: {run_sizes[-1]:.4f} | "
                    f"α̃: {run_alphas[-1]:.4f} | "
                    f"⏱ {epoch_time:.2f}s/epoch | "
                    f"ETA: {_fmt_time(eta)}"
                )

        if run_losses[-1] < best_loss:
            best_loss = run_losses[-1]
            best_net  = alpha_net

        history["losses"].append(run_losses)
        history["sizes"].append(run_sizes)
        history["alphas"].append(run_alphas)

        print(f"  ✓ Run {run+1} complete | Final loss: {run_losses[-1]:.4f}")

    train_time = time.time() - t0_train
    print(f"  [Step 4/9] Training done in {_fmt_time(train_time)}")

    alpha_net = best_net
    alpha_net.eval()

    # =========================================================
    # 5. Compute conformal intervals on TEST set
    # =========================================================
    print(f"  [Step 5/9] Computing conformal intervals...")
    scores_calib = np.abs(_safe_predict(base_model, X_calib) - y_calib)
    sum_scores   = scores_calib.sum()

    with torch.no_grad():
        feat_test = torch.tensor([sum_scores], dtype=torch.float32).to(device)
        alpha_val = alpha_net(feat_test).item()

    denom = alpha_val * (n_calib + 1) - 1
    if denom <= 0:
        q = np.quantile(scores_calib, 1 - alpha)
        half_width = q
        print(f"  [Warning] α̃ too small; fallback q={q:.4f}")
    else:
        half_width = sum_scores / denom

    y_pred_test = _safe_predict(base_model, Xte)
    ymin = y_pred_test - half_width
    ymax = y_pred_test + half_width

    # =========================================================
    # 6. Metrics
    # =========================================================
    is_covered = ((yte >= ymin) & (yte <= ymax)).astype(int)
    coverage   = is_covered.mean()
    avg_width  = np.mean(ymax - ymin)
    rmse       = np.sqrt(mean_squared_error(yte, y_pred_test))

    total_time = time.time() - t_start_total

    print(f"\n{'='*55}")
    print(f"  ALPHANET CONFORMAL ({model_name})")
    print(f"{'='*55}")
    print(f"  Learned α̃     : {alpha_val:.4f}")
    print(f"  Half-width (r) : {half_width:.4f}")
    print(f"  Coverage       : {coverage*100:.3f}%")
    print(f"  Avg Width      : {avg_width:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  Total time     : {_fmt_time(total_time)}")
    print(f"    ├─ Precompute: {_fmt_time(elapsed_precompute)}")
    print(f"    └─ Training  : {_fmt_time(train_time)}")
    print(f"{'='*55}")

    # =========================================================
    # 7. Summary & Values
    # =========================================================
    summary_df = pd.DataFrame([{
        "Strategy":       "AlphaNet Conformal",
        "Coverage":       coverage,
        "Width":          avg_width,
        "RMSE":           rmse,
        "Alpha_nominal":  alpha,
        "Alpha_learned":  alpha_val,
        "Lambda_reg":     lambda_reg,
        "Total_time_sec": round(total_time, 2),
    }])

    values_df = pd.DataFrame({
        "sample_index": np.arange(len(yte)),
        "y_true":       yte,
        "y_pred":       y_pred_test,
        "ymin":         ymin,
        "ymax":         ymax,
        "width":        ymax - ymin,
        "is_covered":   is_covered,
    })

    # =========================================================
    # 8. Plotting (SAVE ONLY)
    # =========================================================
    print(f"  [Step 8/9] Generating plots...")
    idx_arr = np.arange(len(yte))

    fig_pi, ax_pi = plt.subplots(figsize=(10, 4))
    ax_pi.scatter(idx_arr, yte, color="blue", s=10, alpha=0.7, label="True")
    ax_pi.fill_between(idx_arr, ymin, ymax, color="gray", alpha=0.5,
                       label=f"AlphaNet Interval (α̃={alpha_val:.3f})")
    ax_pi.plot(idx_arr, y_pred_test, color="red", lw=1, label="Predicted mean")
    ax_pi.set_title(f"AlphaNet Conformal ({model_name})")
    ax_pi.set_xlabel("Sample Number")
    ax_pi.set_ylabel("str")
    ax_pi.legend(loc="upper left")
    plt.tight_layout()

    buf_pi = io.BytesIO()
    fig_pi.savefig(buf_pi, format="png")
    buf_pi.seek(0)
    plt.close(fig_pi)

    arr_losses = np.array(history["losses"])
    arr_sizes  = np.array(history["sizes"])
    arr_alphas = np.array(history["alphas"])
    epochs_arr = np.arange(1, num_epochs + 1)

    fig_tc, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, ylabel, title in zip(
        axs,
        [arr_losses, arr_sizes, arr_alphas],
        ["Loss", "Mean Interval Size", r"Mean $\tilde{\alpha}$"],
        ["Training Loss", "Interval Size", r"Learned $\tilde{\alpha}$"],
    ):
        m = data.mean(axis=0)
        s = data.std(axis=0)
        ax.plot(epochs_arr, m, color="#0077bb")
        ax.fill_between(epochs_arr, m - s, m + s, color="#0077bb", alpha=0.25)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.5)
    plt.tight_layout()

    buf_tc = io.BytesIO()
    fig_tc.savefig(buf_tc, format="png")
    buf_tc.seek(0)
    plt.close(fig_tc)

    # =========================================================
    # 9. Excel Export
    # =========================================================
    if excel_file_path:
        print(f"  [Step 9/9] Saving to Excel...")
        try:
            wb = load_workbook(_ensure_excel_file(excel_file_path))
        except FileNotFoundError:
            from openpyxl import Workbook
            wb = Workbook()
            wb.remove(wb.active)

        def _unique_sheet(wb, name):
            base, i = name, 1
            while name in wb.sheetnames:
                name = f"{base}_{i}"; i += 1
            return name

        sn = _unique_sheet(wb, "AlphaNet_PGBM_Summary")
        ws = wb.create_sheet(sn)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        img_pi = Image(buf_pi)
        sn = _unique_sheet(wb, "AlphaNet_PGBM_Plot_PI")
        ws = wb.create_sheet(sn)
        ws.add_image(img_pi, "A1")

        img_tc = Image(buf_tc)
        sn = _unique_sheet(wb, "AlphaNet_PGBM_Plot_Train")
        ws = wb.create_sheet(sn)
        ws.add_image(img_tc, "A1")

        sn = _unique_sheet(wb, "AlphaNet_PGBM_Values")
        ws = wb.create_sheet(sn)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))
        print(f"  [Step 9/9] Saved!")

    print(f"\n  ✅ AlphaNet PGBM complete in {_fmt_time(total_time)}")
    return summary_df
```

```python
conformal_predictions_AlphaNet_PGBM(
    model_class=PGBMWrapper,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="PGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/PGBM.xlsx",
    alpha=0.05,
)
```

# **Conformal Predictions with TabNet**

```python
best_params = get_best_model_params(best_scores_autosampler, 'TabNet')
print(best_params)
```

```python
class validate:
    @staticmethod
    def coverage(int_pred, y_test):
        y_arr = np.array(y_test)
        return {strat_name: np.mean((y_arr >= int_pred[strat_name][:, 0]) & (y_arr <= int_pred[strat_name][:, 1])) for strat_name in int_pred}
    @staticmethod
    def width(int_pred):
        return {strat_name: np.mean(int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]) for strat_name in int_pred}
    @staticmethod
    def rmse(y_pred, y_test):
        y_arr = np.array(y_test)
        return {strat_name: np.sqrt(mean_squared_error(y_arr, y_pred[strat_name])) for strat_name in y_pred}
    @staticmethod
    def cwc(int_pred, y_test, miscoverage):
        widths = validate.width(int_pred)
        coverages = validate.coverage(int_pred, y_test)
        return {strat_name: widths[strat_name] * (1 + (coverages[strat_name] < (1 - miscoverage)) * (1 - coverages[strat_name])) for strat_name in int_pred}
    @staticmethod
    def cond_coverage(int_pred, y_test, num_bins=10):
        y_arr = np.array(y_test)
        cond_coverages = {}
        for strat_name in int_pred:
            widths = int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0]
            bin_edges = np.histogram_bin_edges(widths, bins=num_bins)
            digitized_widths = np.digitize(widths, bin_edges)
            strategy_cond_coverage = []
            for bin_num in range(1, num_bins + 1):
                indices_in_bin = np.where(digitized_widths == bin_num)[0]
                if len(indices_in_bin) > 0:
                    bin_coverage = np.mean((y_arr[indices_in_bin] >= int_pred[strat_name][indices_in_bin, 0]) &
                                           (y_arr[indices_in_bin] <= int_pred[strat_name][indices_in_bin, 1]))
                    strategy_cond_coverage.append(bin_coverage)
                else:
                    strategy_cond_coverage.append(np.nan)
            cond_coverages[strat_name] = strategy_cond_coverage
        return cond_coverages
class visualize:
    @staticmethod
    def goodness(y_true, y_pred, y_pred_low, y_pred_up, coverage, width, rmse, cwc, ax=None, title="Goodness Plot"):
        if ax is None:
            fig, ax = plt.subplots()
        y_true_np = np.array(y_true).ravel()
        inside_interval = (y_true_np >= y_pred_low) & (y_true_np <= y_pred_up)
        outside_interval = ~inside_interval
        ax.scatter(y_true_np[inside_interval], y_pred[inside_interval], color='green', s=10, label='Inside Interval')
        ax.scatter(y_true_np[outside_interval], y_pred[outside_interval], color='orange', s=10, label='Outside Interval')
        ax.scatter(y_true_np, y_true_np, color='blue', marker='x', s=30, linewidths=0.7, label='Ground Truth Values')
        ax.vlines(y_true_np, y_pred_low, y_pred_up, color='gray', alpha=0.5, label='Prediction Intervals')
        ax.plot([min(y_true_np), max(y_true_np)], [min(y_true_np), max(y_true_np)],
                 linestyle='--', color='blue', label='Ideal Prediction Line')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.set_title(f'{title}\nCoverage: {coverage*100:.2f}%, Width: {width:.2f}, RMSE: {rmse:.2f}, CWC: {cwc:.2f}')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax

    @staticmethod
    def width_size_occurrence(int_pred, train_intervals, num_bins=10, ax=None, x_lim=None, title="Width Size Occurrence"):
        if ax is None:
            fig, ax = plt.subplots()
        widths_test = np.abs(int_pred[:, 1] - int_pred[:, 0])
        widths_train = np.abs(train_intervals[:, 1] - train_intervals[:, 0])
        ax.hist(widths_test, bins=num_bins, alpha=0.6, label='Test Interval Widths', color='red')
        ax.hist(widths_train, bins=num_bins, alpha=0.4, label='Train Interval Widths', color='blue')
        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel('Interval Width')
        ax.set_ylabel('Occurrence (Frequency)')
        ax.set_title(f'{title} - Interval Width Occurrence')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax

    @staticmethod
    def coverage_by_width(y_test, int_pred, miscoverage, cond_coverages, num_bins=10, ax=None, title="Coverage vs Width"):
        if ax is None:
            fig, ax = plt.subplots()
        widths = int_pred[:, 1] - int_pred[:, 0]
        bin_edges = np.histogram_bin_edges(widths, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, cond_coverages, width=bin_width, color='blue', alpha=0.7, edgecolor='black', linewidth=0.8, label='Conditional Coverage')
        ax.axhline(1 - miscoverage, color='red', linestyle='--', label=f'Nominal Coverage ({(1 - miscoverage)*100:.0f}%)')
        ax.set_xlabel('Interval Width')
        ax.set_ylabel('Conditional Coverage')
        ax.set_title(f'{title} - Coverage vs Interval Width')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return ax

class TabNetRegressorCP(TabNetRegressor):
    def fit(self, X, y, *args, **kwargs):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        return super().predict(X).flatten()

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_NEXCP_tabnet(
    X_train,
    y_train,
    X_test,
    y_test,
    best_params,
    model_name="TabNet",
    excel_file_path=None,
    alpha=0.1,
    decay=0.99,
    n_folds=5,
    n_bootstrap=50,
):
    warnings.filterwarnings("ignore")

    SEED = 123
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    if "verbose" in best_params:
        best_params = dict(best_params)
        best_params.pop("verbose", None)

    model_class = TabNetRegressorCP

    strategies = {
        "NEXCP-Split": "NexCP Split",
        "NEXCP-Full": "NexCP Full",
        "NEXCP-J+aB": "NexCP Jackknife+ aB",
        "NEXCP-CV+": "NexCP CV+",
    }

    y_pred, int_pred = {}, {}

    y_train_np = np.asarray(y_train).ravel()
    y_test_np = np.asarray(y_test).ravel()

    # =========================
    # NexCP – Split (Weighted)
    # =========================
    model = model_class(**best_params)
    model.fit(X_train, y_train_np)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    residuals = np.abs(y_train_np - yhat_train)
    weights = decay ** np.arange(len(residuals) - 1, -1, -1)
    q = weighted_quantile(residuals, weights, 1 - alpha)

    y_pred["NEXCP-Split"] = yhat_test
    int_pred["NEXCP-Split"] = np.c_[yhat_test - q, yhat_test + q]

    # =========================
    # NexCP – Full
    # =========================
    model = model_class(**best_params)
    model.fit(X_train, y_train_np)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    q = np.quantile(np.abs(y_train_np - yhat_train), 1 - alpha)

    y_pred["NEXCP-Full"] = yhat_test
    int_pred["NEXCP-Full"] = np.c_[yhat_test - q, yhat_test + q]

    # =========================
    # NexCP – Jackknife+ aB
    # =========================
    lowers, uppers, preds = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_train), len(X_train), replace=True)

        model = model_class(**best_params)
        model.fit(X_train[idx], y_train_np[idx])

        yhat_boot_train = model.predict(X_train[idx])
        yhat_test = model.predict(X_test)

        q = np.quantile(
            np.abs(y_train_np[idx] - yhat_boot_train),
            1 - alpha,
        )

        lowers.append(yhat_test - q)
        uppers.append(yhat_test + q)
        preds.append(yhat_test)

    y_pred["NEXCP-J+aB"] = np.mean(preds, axis=0)
    int_pred["NEXCP-J+aB"] = np.c_[np.min(lowers, axis=0), np.max(uppers, axis=0)]

    # =========================
    # NexCP – CV+
    # =========================
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    lowers, uppers, preds = [], [], []

    for tr, cal in kf.split(X_train):
        model = model_class(**best_params)
        model.fit(X_train[tr], y_train_np[tr])

        y_cal = model.predict(X_train[cal])
        q = np.quantile(np.abs(y_train_np[cal] - y_cal), 1 - alpha)

        yhat_test = model.predict(X_test)

        lowers.append(yhat_test - q)
        uppers.append(yhat_test + q)
        preds.append(yhat_test)

    y_pred["NEXCP-CV+"] = np.mean(preds, axis=0)
    int_pred["NEXCP-CV+"] = np.c_[np.mean(lowers, axis=0), np.mean(uppers, axis=0)]

    # =========================
    # Metrics
    # =========================
    def coverage(y, l, u):
        return np.mean((y >= l) & (y <= u))

    def width(l, u):
        return np.mean(u - l)

    def cwc(l, u, y, alpha):
        cov = coverage(y, l, u)
        w = width(l, u)
        penalty = np.exp(-50 * (cov - (1 - alpha)))
        return w * (1 + penalty if cov < (1 - alpha) else 1)

    summary = []

    print("\n===== NEXCP TABNET RESULTS =====")
    for s in strategies:
        cov = coverage(y_test_np, int_pred[s][:, 0], int_pred[s][:, 1])
        wid = width(int_pred[s][:, 0], int_pred[s][:, 1])
        rmse = np.sqrt(mean_squared_error(y_test_np, y_pred[s]))
        cwc_val = cwc(int_pred[s][:, 0], int_pred[s][:, 1], y_test_np, alpha)

        summary.append([strategies[s], cov, wid, rmse, cwc_val])
        print(f"{strategies[s]} | Coverage: {cov*100:.3f}% | Width: {wid:.3f}")

    summary_df = pd.DataFrame(
        summary,
        columns=["Strategy", "Coverage", "Width", "RMSE", "CWC"],
    )

    # =========================
    # Plot + Excel Export
    # =========================
    if excel_file_path:
        wb = load_workbook(_ensure_excel_file(excel_file_path))

        sheet = "NEXCP_TabNet_Summary"
        base = sheet
        i = 1
        while sheet in wb.sheetnames:
            sheet = f"{base}_{i}"
            i += 1

        ws = wb.create_sheet(sheet)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        for s in strategies:
            fig, ax = plt.subplots(figsize=(10, 4))
            idx = np.arange(len(y_test_np))

            ax.scatter(idx, y_test_np, s=10, alpha=0.7, label="True")
            ax.fill_between(idx, int_pred[s][:, 0], int_pred[s][:, 1], alpha=0.4)
            ax.plot(idx, y_pred[s], linewidth=1, label="Pred")

            ax.set_title(f"{strategies[s]} ({model_name})")
            ax.legend()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)

            img = Image(buf)
            ps = f"NEXCP_TabNet_{s}"
            base = ps
            i = 1
            while ps in wb.sheetnames:
                ps = f"{base}_{i}"
                i += 1

            ws = wb.create_sheet(ps)
            ws.add_image(img, "A1")

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))

    return summary_df


conformal_predictions_NEXCP_tabnet(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    best_params=best_params,
    model_name="TabNet",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/TabNet.xlsx",
    alpha=0.1,
    decay=0.99,
    n_folds=5,
    n_bootstrap=50
)
```

```python
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

conformal_predictions_online_super(
    model_class=TabNetRegressorCP,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="TabNet",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/TabNet.xlsx",
    alpha=0.1,
    strategies="all",
    window_min=50,
    window_max=500,
    window_grid=(25, 50, 100, 200, 400),
    n_bootstrap=30
)
```

```python
conformal_predictions_COP(
    model_class=TabNetRegressorCP,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="TabNet",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/TabNet.xlsx",
    alpha=0.1,
    lr=0.05,
    T_burnin=50,
    scale=0.5
)
```

```python
conformal_predictions_EXTREME(
    model_class=TabNetRegressorCP,
    model_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="TabNet",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/TabNet.xlsx",
    alpha=0.05,
    calib_size=0.4,
    gpd_threshold_quantile=0.95,
    n_bootstrap=30
)

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_AlphaNet_tabnet(
    X_train,
    y_train,
    X_test,
    y_test,
    best_params,
    model_name="TabNet",
    excel_file_path=None,
    alpha=0.05,
    lambda_reg=20.0,
    num_epochs=50,
    batch_size=32,
    lr_alpha=1e-3,
    calib_fraction=0.5,
    num_runs=4,
    random_state=123,
):
    """
    Learned-α Conformal Prediction via AlphaNet — TabNet-specific variant.

    Handles:
    - TabNetRegressorCP wrapper (y reshape, predict flatten)
    - Removes 'verbose' from best_params if present
    - Precomputed LOO scores for speed
    - Progress tracking with tqdm + ETA
    """

    warnings.filterwarnings("ignore")
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_start_total = time.time()

    # =========================================================
    # TabNet-specific: clean params & set model class
    # =========================================================
    tabnet_params = dict(best_params)
    tabnet_params.pop("verbose", None)           # TabNet doesn't use verbose here
    model_class = TabNetRegressorCP

    # =========================================================
    # data conversion
    # =========================================================
    Xtr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    Xte = X_test.to_numpy()  if hasattr(X_test,  "to_numpy") else np.asarray(X_test)
    ytr = to_numpy_1d(y_train)
    yte = to_numpy_1d(y_test)

    # =========================================================
    # 1. Split training data → fit + calibration
    # =========================================================
    n_total = len(Xtr)
    n_fit   = int(n_total * (1 - calib_fraction))
    perm    = np.random.permutation(n_total)

    fit_idx   = perm[:n_fit]
    calib_idx = perm[n_fit:]

    X_fit,   y_fit   = Xtr[fit_idx],   ytr[fit_idx]
    X_calib, y_calib = Xtr[calib_idx],  ytr[calib_idx]
    n_calib = len(X_calib)

    # alpha_min: must satisfy α̃·(n+1) − 1 > 0  →  α̃ > 1/(n+1)
    alpha_min = 1.0 / (n_calib + 1) + 0.01
    print(f"  [Info] n_calib={n_calib} | alpha_min={alpha_min:.4f}")

    # =========================================================
    # 2. Train base TabNet model
    # =========================================================
    print(f"  [Step 2/9] Training base {model_name} model...")
    t0 = time.time()
    base_model = model_class(**tabnet_params)
    base_model.fit(X_fit, y_fit)
    print(f"  [Step 2/9] Done in {_fmt_time(time.time()-t0)}")

    # =========================================================
    # 3. PRECOMPUTE all LOO scores (KEY SPEEDUP)
    # =========================================================
    print(f"  [Step 3/9] Precomputing {n_calib} LOO score vectors...")
    t0 = time.time()

    precomputed_scores = []
    precomputed_sums   = []

    iterator = (
        tqdm(range(n_calib), desc="    LOO precompute", leave=False)
        if HAS_TQDM else range(n_calib)
    )

    for i in iterator:
        loo_X = np.delete(X_calib, i, axis=0)
        loo_y = np.delete(y_calib, i, axis=0)
        preds_loo = base_model.predict(loo_X).ravel()
        scores_loo = np.abs(preds_loo - loo_y)

        scores_tensor = torch.tensor(scores_loo, dtype=torch.float32).to(device)
        precomputed_scores.append(scores_tensor)
        precomputed_sums.append(scores_loo.sum())

    elapsed_precompute = time.time() - t0
    print(f"  [Step 3/9] Done in {_fmt_time(elapsed_precompute)}")

    # Build feature matrix (sum of LOO scores per sample)
    X_alpha = torch.tensor(
        np.array(precomputed_sums).reshape(-1, 1),
        dtype=torch.float32,
    ).to(device)

    # =========================================================
    # 4. Train AlphaNet (multiple runs, keep best)
    # =========================================================
    print(f"  [Step 4/9] Training AlphaNet: {num_runs} runs × {num_epochs} epochs")

    best_net  = None
    best_loss = np.inf
    history   = {"losses": [], "sizes": [], "alphas": []}

    t0_train = time.time()

    for run in range(num_runs):
        np.random.seed(run)
        torch.manual_seed(run)

        alpha_net = AlphaNet(input_dim=1, alpha_min=alpha_min).to(device)
        optimizer = optim.Adam(alpha_net.parameters(), lr=lr_alpha)

        Dataset = TensorDataset(X_alpha, torch.arange(n_calib))
        loader  = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

        run_losses, run_sizes, run_alphas = [], [], []

        epoch_iter = (
            tqdm(range(num_epochs), desc=f"    Run {run+1}/{num_runs}", leave=False)
            if HAS_TQDM else range(num_epochs)
        )

        for epoch in epoch_iter:
            t_epoch_start = time.time()
            epoch_loss = epoch_size = epoch_alpha = 0.0

            for x_batch, idx_batch in loader:
                x_batch = x_batch.to(device)
                alpha_pred = alpha_net(x_batch)

                batch_sizes = []
                for j, idx in enumerate(idx_batch):
                    # USE PRECOMPUTED scores — no predict() call here
                    scores_t = precomputed_scores[idx.item()]
                    size_j = _interval_size(scores_t, alpha_pred[j])
                    batch_sizes.append(size_j)

                batch_sizes = torch.stack(batch_sizes)
                loss = (batch_sizes + lambda_reg * alpha_pred).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss  += loss.item()
                epoch_size  += batch_sizes.mean().item()
                epoch_alpha += alpha_pred.mean().item()

            n_batches = len(loader)
            run_losses.append(epoch_loss  / n_batches)
            run_sizes.append(epoch_size   / n_batches)
            run_alphas.append(epoch_alpha / n_batches)

            epoch_time = time.time() - t_epoch_start

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                epochs_done  = (run * num_epochs) + epoch + 1
                epochs_total = num_runs * num_epochs
                elapsed_train = time.time() - t0_train
                eta = (elapsed_train / epochs_done) * (epochs_total - epochs_done)

                print(
                    f"  λ={lambda_reg} | Run {run+1}/{num_runs} | "
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Loss: {run_losses[-1]:.4f} | "
                    f"Size: {run_sizes[-1]:.4f} | "
                    f"α̃: {run_alphas[-1]:.4f} | "
                    f"⏱ {epoch_time:.2f}s/epoch | "
                    f"ETA: {_fmt_time(eta)}"
                )

        if run_losses[-1] < best_loss:
            best_loss = run_losses[-1]
            best_net  = alpha_net

        history["losses"].append(run_losses)
        history["sizes"].append(run_sizes)
        history["alphas"].append(run_alphas)

        print(f"  ✓ Run {run+1} complete | Final loss: {run_losses[-1]:.4f}")

    train_time = time.time() - t0_train
    print(f"  [Step 4/9] Training done in {_fmt_time(train_time)}")

    alpha_net = best_net
    alpha_net.eval()

    # =========================================================
    # 5. Compute conformal intervals on TEST set
    # =========================================================
    print(f"  [Step 5/9] Computing conformal intervals...")

    calib_preds = base_model.predict(X_calib).ravel()
    scores_calib = np.abs(calib_preds - y_calib)
    sum_scores   = scores_calib.sum()

    with torch.no_grad():
        feat_test = torch.tensor([sum_scores], dtype=torch.float32).to(device)
        alpha_val = alpha_net(feat_test).item()

    denom = alpha_val * (n_calib + 1) - 1
    if denom <= 0:
        q = np.quantile(scores_calib, 1 - alpha)
        half_width = q
        print(f"  [Warning] α̃ too small (denom={denom:.4f}); fallback q={q:.4f}")
    else:
        half_width = sum_scores / denom

    y_pred_test = base_model.predict(Xte).ravel()
    ymin = y_pred_test - half_width
    ymax = y_pred_test + half_width

    # =========================================================
    # 6. Metrics
    # =========================================================
    is_covered = ((yte >= ymin) & (yte <= ymax)).astype(int)
    coverage   = is_covered.mean()
    avg_width  = np.mean(ymax - ymin)
    rmse       = np.sqrt(mean_squared_error(yte, y_pred_test))

    total_time = time.time() - t_start_total

    print(f"\n{'='*55}")
    print(f"  ALPHANET CONFORMAL ({model_name})")
    print(f"{'='*55}")
    print(f"  Learned α̃     : {alpha_val:.4f}")
    print(f"  Half-width (r) : {half_width:.4f}")
    print(f"  Coverage       : {coverage*100:.3f}%")
    print(f"  Avg Width      : {avg_width:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  Total time     : {_fmt_time(total_time)}")
    print(f"    ├─ Precompute: {_fmt_time(elapsed_precompute)}")
    print(f"    └─ Training  : {_fmt_time(train_time)}")
    print(f"{'='*55}")

    # =========================================================
    # 7. Summary & Values
    # =========================================================
    summary_df = pd.DataFrame([{
        "Strategy":       "AlphaNet Conformal",
        "Coverage":       coverage,
        "Width":          avg_width,
        "RMSE":           rmse,
        "Alpha_nominal":  alpha,
        "Alpha_learned":  alpha_val,
        "Lambda_reg":     lambda_reg,
        "Total_time_sec": round(total_time, 2),
    }])

    values_df = pd.DataFrame({
        "sample_index": np.arange(len(yte)),
        "y_true":       yte,
        "y_pred":       y_pred_test,
        "ymin":         ymin,
        "ymax":         ymax,
        "width":        ymax - ymin,
        "is_covered":   is_covered,
    })

    # =========================================================
    # 8. Plotting (SAVE ONLY — NO SHOW)
    # =========================================================
    print(f"  [Step 8/9] Generating plots...")
    idx_arr = np.arange(len(yte))

    # --- Prediction-interval plot ---
    fig_pi, ax_pi = plt.subplots(figsize=(10, 4))
    ax_pi.scatter(idx_arr, yte, color="blue", s=10, alpha=0.7, label="True")
    ax_pi.fill_between(idx_arr, ymin, ymax, color="gray", alpha=0.5,
                       label=f"AlphaNet Interval (α̃={alpha_val:.3f})")
    ax_pi.plot(idx_arr, y_pred_test, color="red", lw=1, label="Predicted mean")
    ax_pi.set_title(f"AlphaNet Conformal ({model_name})")
    ax_pi.set_xlabel("Sample Number")
    ax_pi.set_ylabel("str")
    ax_pi.legend(loc="upper left")
    plt.tight_layout()

    buf_pi = io.BytesIO()
    fig_pi.savefig(buf_pi, format="png")
    buf_pi.seek(0)
    plt.close(fig_pi)

    # --- Training-curves plot ---
    arr_losses = np.array(history["losses"])
    arr_sizes  = np.array(history["sizes"])
    arr_alphas = np.array(history["alphas"])
    epochs_arr = np.arange(1, num_epochs + 1)

    fig_tc, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, ylabel, title in zip(
        axs,
        [arr_losses, arr_sizes, arr_alphas],
        ["Loss", "Mean Interval Size", r"Mean $\tilde{\alpha}$"],
        ["Training Loss", "Interval Size", r"Learned $\tilde{\alpha}$"],
    ):
        m = data.mean(axis=0)
        s = data.std(axis=0)
        ax.plot(epochs_arr, m, color="#0077bb")
        ax.fill_between(epochs_arr, m - s, m + s, color="#0077bb", alpha=0.25)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.5)
    plt.tight_layout()

    buf_tc = io.BytesIO()
    fig_tc.savefig(buf_tc, format="png")
    buf_tc.seek(0)
    plt.close(fig_tc)

    # =========================================================
    # 9. Excel Export (append-safe)
    # =========================================================
    if excel_file_path:
        print(f"  [Step 9/9] Saving to Excel...")
        try:
            wb = load_workbook(_ensure_excel_file(excel_file_path))
        except FileNotFoundError:
            from openpyxl import Workbook
            wb = Workbook()
            wb.remove(wb.active)

        def _unique_sheet(wb, name):
            base, i = name, 1
            while name in wb.sheetnames:
                name = f"{base}_{i}"; i += 1
            return name

        # ---- Summary ----
        sn = _unique_sheet(wb, "AlphaNet_TabNet_Summary")
        ws = wb.create_sheet(sn)
        for r, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        # ---- PI Plot ----
        img_pi = Image(buf_pi)
        sn = _unique_sheet(wb, "AlphaNet_TabNet_Plot_PI")
        ws = wb.create_sheet(sn)
        ws.add_image(img_pi, "A1")

        # ---- Training Curves Plot ----
        img_tc = Image(buf_tc)
        sn = _unique_sheet(wb, "AlphaNet_TabNet_Plot_Train")
        ws = wb.create_sheet(sn)
        ws.add_image(img_tc, "A1")

        # ---- Values ----
        sn = _unique_sheet(wb, "AlphaNet_TabNet_Values")
        ws = wb.create_sheet(sn)
        for r, row in enumerate(dataframe_to_rows(values_df, index=False, header=True), 1):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)

        _ensure_parent_dir(excel_file_path)
        wb.save(_ensure_parent_dir(excel_file_path))
        print(f"  [Step 9/9] Saved!")

    print(f"\n  ✅ AlphaNet TabNet complete in {_fmt_time(total_time)}")
    return summary_df
```

```python
conformal_predictions_AlphaNet_tabnet(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    best_params=best_params,
    model_name="TabNet",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/TabNet.xlsx",
    alpha=0.05,
)
```

