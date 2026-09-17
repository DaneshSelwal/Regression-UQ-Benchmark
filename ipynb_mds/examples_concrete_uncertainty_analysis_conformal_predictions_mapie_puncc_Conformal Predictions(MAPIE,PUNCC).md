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
from pytorch_tabnet import TabNetRegressor
```

```python
# Go to find & replace button and replace (concrete_uncertainty_analysis) with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (str) with actual data label name.
```

```python
feature_names = ['C', 'mp', 'FA', 'CA', 'F', 'W_P', 'Adm']
```

```python
train_data_path = "./drive/MyDrive/Regression/concrete_uncertainty_analysis/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/concrete_uncertainty_analysis/data/test.csv"
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
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
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

def conformal_predictions_MAPIE(model_class, best_params, X_train, y_train, X_test, y_test, model_name, excel_file_path):
    """
    Train and plot conformal prediction intervals using different strategies and save the plots and
    table of ymin/ymax/y_pred to different worksheets in an Excel file for each strategy.

    Parameters:
    - model_class: The class of the model to be used (e.g., LGBMRegressor).
    - best_params: Dictionary of best parameters for the model.
    - X_train: Training features.
    - y_train: Training target.
    - X_test: Test features.
    - y_test: Test target.
    - model_name: Name of the model for labeling purposes.
    - excel_file_path: Path to the Excel file where the plots will be saved.
    """
    # Set random seed for reproducibility
    SEED: int = 123
    np.random.seed(SEED)

    # Define miscoverage
    MISCOVERAGE: float = 0.05  # MISCOVERAGE = alpha # CONFIDENCE = 1-alpha
    warnings.filterwarnings("ignore")  # to suppress them

    # Create output directory
    os.makedirs('output/regression', exist_ok=True)

    # Define strategies and parameters
    strategies_names = {
        'SCP': 'Split Conformal Prediction',
        'CV+': 'Cross Validation +',
        'J+aB': 'Jackknife+ after Bootstrapping',
    }
    strategies_params = {
        'SCP': {'method': 'base', 'cv': KFold(n_splits=5)},
        'CV+': {'method': "plus", 'cv': 10},
        'J+aB': {'method': "minmax", 'cv': Subsample(n_resamplings=50)},
    }

    # Define base estimators
    base_estimators = {
        strat_name: model_class(**best_params) for strat_name in strategies_names
    }

    # Train strategies using MAPIE
    y_pred, int_pred = {}, {}
    y_pred_train, int_pred_train = {}, {} # Store train predictions and intervals
    for strat_name, base_estimator in base_estimators.items():
        mapie = MapieRegressor(base_estimator, **strategies_params[strat_name])
        mapie.fit(X_train, y_train)
        y_pred[strat_name], int_pred[strat_name] = mapie.predict(X_test, alpha=MISCOVERAGE)
        y_pred_train[strat_name], int_pred_train[strat_name] = mapie.predict(X_train, alpha=MISCOVERAGE) # Predict on train set

    # Calculate metrics
    coverages: dict = validate.coverage(int_pred, y_test)
    widths: dict = validate.width(int_pred) # Corrected line: Removed y_test
    rmse_vals: dict = validate.rmse(y_pred, y_test)
    cwc_vals: dict = validate.cwc(int_pred, y_test, MISCOVERAGE)


    # Print metrics and prepare for plotting
    for strat_name in strategies_names.keys():
        if strat_name in y_pred:
            # Calculate RMSE (already calculated in validate.rmse, using rmse_vals now)
            rmse = rmse_vals[strat_name]

            # Calculate interval width (already calculated in validate.width, using widths now)
            interval_width = widths[strat_name]

            # Calculate coverage (already calculated in validate.coverage, using coverages now)
            coverage = coverages[strat_name]

            # Calculate CWC (already calculated in validate.cwc, using cwc_vals now)
            cwc = cwc_vals[strat_name]

            # Calculate SSC (Symmetric Scoring Criterion)
            ssc = np.mean((y_test.to_numpy() - y_pred[strat_name])**2 + (int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0])**2)

            print(f"Strategy: {strategies_names[strat_name]}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Interval Width: {interval_width:.4f}")
            print(f"  Coverage: {coverage * 100:.2f}%")
            print(f"  CWC: {cwc:.4f}")
            print(f"  SSC: {ssc:.4f}")
            print()

    # Plot the results for the test set
    _strategies = [_s for _s in strategies_names.keys() if _s != 'SCP']
    NUM_BINS: int = 10
    n_figs: int = len(strategies_names) * 3 + 1 + len(_strategies) # Strategy plots + Goodness plots + Width plots + Comparison plot + Coverage vs Width plots
    fig, axs = plt.subplots(nrows=n_figs, figsize=(10, 5 * n_figs))

    plot_index = 0 # Keep track of the current plot index

    # Individual strategy plots (original plots)
    for _i, strat_name in enumerate(strategies_names.keys()):
        if strat_name in y_pred:
            # Predicted intervals
            _y_pred_low = int_pred[strat_name][:, 0].ravel()
            _y_pred_up = int_pred[strat_name][:, 1].ravel()

            # Visualize
            sample_indices = np.arange(len(X_test))
            axs[plot_index].scatter(sample_indices, y_test, label='Test data', color='blue', s=10, alpha=0.7)
            axs[plot_index].fill_between(sample_indices, _y_pred_low, _y_pred_up, color='gray', alpha=0.5, label='Prediction interval')
            axs[plot_index].plot(sample_indices, y_pred[strat_name], color='red', label='Predicted mean', linewidth=1)
            axs[plot_index].set_title(f'{strategies_names[strat_name]} ({model_name})')
            axs[plot_index].set_xlabel('Sample Number')
            axs[plot_index].set_ylabel('str')
            axs[plot_index].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plot_index += 1

    # Combined plot for comparison
    comparison_ax = axs[plot_index]
    sample_indices = np.arange(len(X_test))
    comparison_ax.scatter(sample_indices, y_test, label='Test data', color='blue', s=10, alpha=0.7)
    for strat_name in y_pred:
        _y_pred_low = int_pred[strat_name][:, 0].ravel()
        _y_pred_up = int_pred[strat_name][:, 1].ravel()
        comparison_ax.fill_between(sample_indices, _y_pred_low, _y_pred_up, alpha=0.3, label=f'{strat_name} interval')
        comparison_ax.plot(sample_indices, y_pred[strat_name], label=f'{strat_name} mean', linewidth=1)

    comparison_ax.set_title(f'Comparison of Strategies ({model_name})')
    comparison_ax.set_xlabel('Sample Number')
    comparison_ax.set_ylabel('str')
    comparison_ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plot_index += 1

    # Goodness plots
    for _i, strat_name in enumerate(strategies_names.keys()):
        if strat_name in y_pred:
            _y_pred_low_goodness = int_pred[strat_name][:, 0].ravel()
            _y_pred_up_goodness = int_pred[strat_name][:, 1].ravel()

            axs[plot_index] = visualize.goodness(
                y_test, y_pred[strat_name],
                _y_pred_low_goodness,
                _y_pred_up_goodness,
                coverages[strat_name],
                widths[strat_name],
                rmse_vals[strat_name],
                cwc_vals[strat_name],
                ax=axs[plot_index],
                title=f'{strategies_names[strat_name]} Goodness ({model_name})', # Removed subsample
            )
            axs[plot_index].set_ylabel('Predicted str')
            plot_index += 1

    # Width Occurrence plots
    _x_max = (1+1e-3) * np.max([np.abs(int_pred[strat_name][:, 0] - int_pred[strat_name][:, 1]) for strat_name in strategies_names.keys() if strat_name in int_pred])
    _x_min = np.min([np.abs(int_pred[strat_name][:, 0] - int_pred[strat_name][:, 1]) for strat_name in strategies_names.keys() if strat_name in int_pred])

    for _i, strat_name in enumerate(strategies_names.keys()):
        if strat_name in y_pred:
            axs[plot_index] = visualize.width_size_occurrence(
                int_pred[strat_name],
                train_intervals=int_pred_train[strat_name],
                num_bins=10,
                ax=axs[plot_index],
                x_lim=[_x_min, _x_max],
                title=f'{strategies_names[strat_name]} Width Occurrence',
            )
            axs[plot_index].set_ylabel('Frequency')
            plot_index += 1

    # Coverage vs Width plots
    cond_coverages: dict = validate.cond_coverage(int_pred, y_test, num_bins=NUM_BINS)
    for _i, _strat in enumerate(_strategies):
        axs[plot_index] = visualize.coverage_by_width(
            y_test, int_pred[_strat], MISCOVERAGE,
            cond_coverages[_strat],
            num_bins=NUM_BINS,
            ax=axs[plot_index],
            title=f'{strategies_names[_strat]} Coverage vs Width',
        )
        plot_index += 1

    plt.tight_layout()

    # Save plots and tabular data to Excel
    with io.BytesIO() as buf:
        buf
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image(PImage.open(buf)) # Use PIL to open from BytesIO

        # Load the workbook and add a new sheet for the plots
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        plot_sheet_name = 'conformal_predictions_MAPIE'
        if plot_sheet_name in workbook.sheetnames:
            plot_sheet_name += '_new'
        plot_worksheet = workbook.create_sheet(title=plot_sheet_name)
        plot_worksheet.add_image(img, 'A1')

        all_rows = []
        y_true_np = y_test.to_numpy().ravel()  # To save multiple conversions
        for strat_name in y_pred:
            y_pred_np = y_pred[strat_name].ravel()
            ymin_np = int_pred[strat_name][:, 0].ravel()
            ymax_np = int_pred[strat_name][:, 1].ravel()
            width_np = ymax_np - ymin_np
            is_covered_np = (y_true_np >= ymin_np) & (y_true_np <= ymax_np)
            residual_np = y_true_np - y_pred_np

            strat_df = pd.DataFrame({
                'sample_index': np.arange(len(y_pred_np)),
                'strategy': strat_name,
                'y_true': y_true_np,
                'y_pred': y_pred_np,
                'ymin': ymin_np,
                'ymax': ymax_np,
                'width': width_np,
                'is_covered': is_covered_np.astype(int),
                'residual': residual_np
            })
            all_rows.append(strat_df)
        all_export_df = pd.concat(all_rows, axis=0, ignore_index=True)

        all_pred_sheet_name = 'all_pred_values'
        base_name = all_pred_sheet_name
        idx = 1
        while all_pred_sheet_name in workbook.sheetnames:
            all_pred_sheet_name = f'{base_name}_new{idx}'
            idx += 1

        worksheet = workbook.create_sheet(title=all_pred_sheet_name)

        for j, col_name in enumerate(all_export_df.columns, 1):
            worksheet.cell(row=1, column=j, value=col_name)
        for i, row in all_export_df.iterrows():
            for j, value in enumerate(row, 1):
                worksheet.cell(row=i+2, column=j, value=value)

        # Save the workbook
        _ensure_parent_dir(excel_file_path)
        workbook.save(_ensure_parent_dir(excel_file_path))

    plt.close()


```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def conformal_predictions_PUNCC(
    X_train, y_train, X_test, y_test, best_scores_autosampler, model_class,
    excel_file_path=None, model_params=None, alpha=0.1
):
    # Determine the default parameters for the model, or use provided ones
    if model_params is None:
        if model_class.__name__ == 'LGBMRegressor':
            model_params = get_best_model_params(best_scores_autosampler, 'LightGBM')
        else:
            model_params = {}

    # Initialize and train the model using the provided class and parameters
    try:
        model = model_class(**model_params)
    except TypeError as e:
        print(f"Error initializing model {model_class.__name__}: {e}")
        return

    model.fit(X_train, y_train)

    def evaluate_cp(X_test, y_test, model_cp, alpha):
        y_pred, y_pred_lower, y_pred_upper = model_cp.predict(X_test, alpha=alpha)
        sharpness = regression_sharpness(y_pred_lower, y_pred_upper)
        coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
        return sharpness, coverage

    # Wrap the model in a BasePredictor
    base_predictor = BasePredictor(model, is_trained=True)

    # Initialize and fit the SplitCP conformal predictor
    splitcp = SplitCP(base_predictor, train=True, random_state=0)
    splitcp.fit(X=X_train, y=y_train, fit_ratio=0.5)

    # Compute prediction intervals and metrics on the test set using SplitCP
    y_pred_splitcp, y_pred_lower_splitcp, y_pred_upper_splitcp = splitcp.predict(X_test, alpha=alpha)
    sharpness, coverage = evaluate_cp(X_test, y_test, splitcp, alpha)
    print(f"SplitCP - Average prediction intervals width (sharpness): {sharpness:.3f}")
    print(f"SplitCP - Average coverage: {coverage*100:.3f}%")

    # Plot the prediction intervals for SplitCP
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    axs[0].scatter(np.arange(len(y_test)), y_test, label='True', color='blue', s=10, alpha=0.7)
    axs[0].fill_between(np.arange(len(y_test)), y_pred_lower_splitcp, y_pred_upper_splitcp, color='gray', alpha=0.5, label='Prediction interval')
    axs[0].plot(np.arange(len(y_test)), y_pred_splitcp, color='red', label='Predicted mean', linewidth=1)
    axs[0].set_title('Split Conformal Prediction')
    axs[0].set_xlabel('Sample Number')
    axs[0].set_ylabel('str')
    axs[0].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Initialize and fit the CVPlus conformal predictor
    cvplus = CVPlus(base_predictor, K=5, random_state=0)
    cvplus.fit(X=X_train, y=y_train)

    # Compute prediction intervals and metrics on the test set using CVPlus
    y_pred_cvplus, y_pred_lower_cvplus, y_pred_upper_cvplus = cvplus.predict(X_test, alpha=alpha)
    sharpness, coverage = evaluate_cp(X_test, y_test, cvplus, alpha)
    print(f"CVPlus - Average prediction intervals width (sharpness): {sharpness:.3f}")
    print(f"CVPlus - Average coverage: {coverage*100:.3f}%")

    # Plot the prediction intervals for CVPlus
    axs[1].scatter(np.arange(len(y_test)), y_test, label='True', color='blue', s=10, alpha=0.7)
    axs[1].fill_between(np.arange(len(y_test)), y_pred_lower_cvplus, y_pred_upper_cvplus, color='gray', alpha=0.5, label='Prediction interval')
    axs[1].plot(np.arange(len(y_test)), y_pred_splitcp, color='red', label='Predicted mean', linewidth=1)
    axs[1].set_title('Cross Validation Plus')
    axs[1].set_xlabel('Sample Number ')
    axs[1].set_ylabel('str')
    axs[1].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Split the training data into proper training (fit) set and calibration set
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.5, random_state=0
    )

    # Fit the upper and lower quantile models
    upper_quantile_model = model_class(**model_params)
    lower_quantile_model = model_class(**model_params)

    _ = upper_quantile_model.fit(X_fit, y_fit)
    _ = lower_quantile_model.fit(X_fit, y_fit)

    # Wrap the upper and lower quantile models in a dual predictor
    dualpredictor = DualPredictor(
        [lower_quantile_model, upper_quantile_model], is_trained=[True, True]
    )

    # Initialize the CQR conformal predictor
    cqr = CQR(
        dualpredictor, train=False
    )  # train=False to use the pre-trained dual predictor

    # Compute nonconformity scores on the calibration set
    cqr.fit(X_calib=X_calib, y_calib=y_calib)

    # Compute prediction intervals and metrics on the test set
    y_pred_cqr, y_pred_lower_cqr, y_pred_upper_cqr = cqr.predict(X_test, alpha=alpha)
    y_pred_cqr = 0.5 * (y_pred_lower_cqr + y_pred_upper_cqr)
    sharpness, coverage = evaluate_cp(X_test, y_test, cqr, alpha)

    print(f"CQR - Average prediction intervals width (sharpness): {sharpness:.3f}")
    print(f"CQR - Average coverage: {coverage*100:.3f}%")

    # Plot the prediction intervals for CQR
    axs[2].scatter(np.arange(len(y_test)), y_test, label='True', color='blue', s=10, alpha=0.7)
    axs[2].fill_between(np.arange(len(y_test)), y_pred_lower_cqr, y_pred_upper_cqr, color='gray', alpha=0.5, label='Prediction interval')
    axs[2].plot(np.arange(len(y_test)), y_pred_cqr, color='red', label='Predicted mean', linewidth=1)
    axs[2].set_title('Conformalized Quantile Regression')
    axs[2].set_xlabel('Sample Number')
    axs[2].set_ylabel('str')
    axs[2].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()

    # --- Collect values for export, grouped by method ---
    results_dict = {
        'SplitCP_true': np.asarray(y_test).reshape(-1),
        'SplitCP_pred': np.asarray(y_pred_splitcp).reshape(-1),
        'SplitCP_lower': np.asarray(y_pred_lower_splitcp).reshape(-1),
        'SplitCP_upper': np.asarray(y_pred_upper_splitcp).reshape(-1),

        'CVPlus_true': np.asarray(y_test).reshape(-1),
        'CVPlus_pred': np.asarray(y_pred_splitcp).reshape(-1),
        'CVPlus_lower': np.asarray(y_pred_lower_cvplus).reshape(-1),
        'CVPlus_upper': np.asarray(y_pred_upper_cvplus).reshape(-1),

        'CQR_true': np.asarray(y_test).reshape(-1),
        'CQR_pred': np.asarray(y_pred_cqr).reshape(-1),
        'CQR_lower': np.asarray(y_pred_lower_cqr).reshape(-1),
        'CQR_upper': np.asarray(y_pred_upper_cqr).reshape(-1),
    }

    # Save all plots and values to Excel
    if excel_file_path:
        save_plot_to_excel(fig, excel_file_path, 'conformal_predictions_PUNCC')
        save_values_to_excel(results_dict, excel_file_path, 'conformal_predictions_PUNCC_values')

    plt.close(fig)

def save_plot_to_excel(fig, excel_file_path, sheet_name):
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image(buf)

        # Load the workbook and add a new sheet
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        if sheet_name in workbook.sheetnames:
            base_name = sheet_name
            i = 1
            while sheet_name in workbook.sheetnames:
                sheet_name = f"{base_name}_{i}"
                i += 1
        worksheet = workbook.create_sheet(title=sheet_name)

        # Add the image to the worksheet
        worksheet.add_image(img, 'A1')

        # Save the workbook
        _ensure_parent_dir(excel_file_path)
        workbook.save(_ensure_parent_dir(excel_file_path))

def save_values_to_excel(results_dict, excel_file_path, sheet_name):
    # Convert the dictionary to a DataFrame (columns will be in the order of insertion)
    df = pd.DataFrame(results_dict)
    # Load the workbook
    workbook = load_workbook(_ensure_excel_file(excel_file_path))
    # Ensure unique sheet name
    base_name = sheet_name
    i = 1
    while sheet_name in workbook.sheetnames:
        sheet_name = f"{base_name}_{i}"
        i += 1
    # Write DataFrame to a new sheet
    worksheet = workbook.create_sheet(title=sheet_name)
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            worksheet.cell(row=r_idx, column=c_idx, value=value)
    # Save the workbook
    _ensure_parent_dir(excel_file_path)
    workbook.save(_ensure_parent_dir(excel_file_path))
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def prediction_MAPIE_analysis(
    X_train,
    y_train,
    X_test,
    y_test,
    model_cls,
    model_params,
    excel_file_path=None,
    suptitle: str = "Prediction Intervals"
) -> None:
    # Initialize and fit the model with MAPIE using provided parameters
    mdl = model_cls(**model_params)
    mapie = MapieRegressor(mdl, method="plus", cv=KFold(n_splits=5, shuffle=True))
    mapie.fit(X_train, y_train)

    alpha = np.arange(0.05, 1, 0.05)
    y_train_pred, y_train_pis = mapie.predict(X_train, alpha=alpha)
    y_test_pred, y_test_pis = mapie.predict(X_test, alpha=alpha)

    # Visualization function
    def plot_predictionintervals(
        y_train,
        y_train_pred,
        y_train_pred_low,
        y_train_pred_high,
        y_test,
        y_test_pred,
        y_test_pred_low,
        y_test_pred_high,
        suptitle: str,
    ) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        ax1.errorbar(
            x=y_train,
            y=y_train_pred,
            yerr=(np.abs(y_train_pred - y_train_pred_low), np.abs(y_train_pred_high - y_train_pred)),
            alpha=0.8,
            label="train",
            fmt=".",
        )
        ax1.errorbar(
            x=y_test,
            y=y_test_pred,
            yerr=(np.abs(y_test_pred - y_test_pred_low), np.abs(y_test_pred_high - y_test_pred)),
            alpha=0.8,
            label="test",
            fmt=".",
        )
        ax1.plot(
            [y_train.min(), y_train.max()],
            [y_train.min(), y_train.max()],
            color="gray",
            alpha=0.5,
        )
        ax1.set_xlabel("Actual str", fontsize=12)
        ax1.set_ylabel("Predicted str", fontsize=12)
        ax1.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        ax2.scatter(
            x=y_train, y=y_train_pred_high - y_train_pred_low, alpha=0.8, label="train", marker="."
        )
        ax2.scatter(x=y_test, y=y_test_pred_high - y_test_pred_low, alpha=0.8, label="test", marker=".")
        ax2.set_xlabel("Actual str", fontsize=12)
        ax2.set_ylabel("Interval width", fontsize=12)
        ax2.set_xscale("linear")
        ax2.set_ylim([0, np.max(y_test_pred_high - y_test_pred_low)*1.1])
        ax2.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        std_all = np.concatenate([
            y_train_pred_high - y_train_pred_low, y_test_pred_high - y_test_pred_low
        ])
        type_all = np.array(["train"] * len(y_train) + ["test"] * len(y_test))
        x_all = np.arange(len(std_all))
        order_all = np.argsort(std_all)
        std_order = std_all[order_all]
        type_order = type_all[order_all]
        ax3.scatter(
            x=x_all[type_order == "train"],
            y=std_order[type_order == "train"],
            alpha=0.8,
            label="train",
            marker=".",
        )
        ax3.scatter(
            x=x_all[type_order == "test"],
            y=std_order[type_order == "test"],
            alpha=0.8,
            label="test",
            marker=".",
        )
        ax3.set_xlabel("Order", fontsize=12)
        ax3.set_ylabel("Interval width", fontsize=12)
        ax3.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax1.set_title("Actual vs Predicted str")
        ax2.set_title("Prediction interval width vs Actual str")
        ax3.set_title("Ordered prediction interval width")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle(suptitle, size=20)

        # Save plot to Excel
        if excel_file_path:
            save_plot_to_excel(fig, excel_file_path, 'prediction_MAPIE_analysis_1')

        plt.close(fig)

    alpha_plot = int(np.where(alpha == 0.1)[0])
    plot_predictionintervals(
        y_train,
        y_train_pred,
        y_train_pis[:, 0, alpha_plot],
        y_train_pis[:, 1, alpha_plot],
        y_test,
        y_test_pred,
        y_test_pis[:, 0, alpha_plot],
        y_test_pis[:, 1, alpha_plot],
        suptitle,
    )

    # Comparison of the uncertainty quantification methods
    Params = TypedDict("Params", {"method": str, "cv": Union[int, Subsample]})
    STRATEGIES = {
        "naive": Params(method="naive"),
        "cv": Params(method="base", cv=5),
        "cv_plus": Params(method="plus", cv=5),
        "cv_minmax": Params(method="minmax", cv=5),
        "jackknife_plus_ab": Params(method="plus", cv=Subsample(n_resamplings=20)),
    }
    y_pred, y_pis, scores = {}, {}, {}
    for strategy, params in STRATEGIES.items():
        mapie = MapieRegressor(mdl, **params)
        mapie.fit(X_train, y_train)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=alpha)
        scores[strategy] = [
            regression_coverage_score(y_test, y_pis[strategy][:, 0, i], y_pis[strategy][:, 1, i])
            for i, _ in enumerate(alpha)
        ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Target coverage (1 - alpha)")
    ax.set_ylabel("Effective coverage")
    for strategy, params in STRATEGIES.items():
        ax.plot(1 - alpha, scores[strategy], label=strategy)
    plt.subplots_adjust(right=0.75)
    ax.plot([0, 1], [0, 1], ls="--", color="k")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', borderaxespad=0.)

    # Save plot to Excel
    if excel_file_path:
        save_plot_to_excel(fig, excel_file_path, 'prediction_MAPIE_analysis_2')

    plt.close(fig)

def save_plot_to_excel(fig, excel_file_path, sheet_name):
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image(buf)

        # Load the workbook and add a new sheet
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        if sheet_name in workbook.sheetnames:
            sheet_name += '_new'
        worksheet = workbook.create_sheet(title=sheet_name)

        # Add the image to the worksheet
        worksheet.add_image(img, 'A1')

        # Save the workbook
        _ensure_parent_dir(excel_file_path)
        workbook.save(_ensure_parent_dir(excel_file_path))
```

# **Tuned Parameters**

```python
best_scores_autosampler = {('Random Forest', 'MedianPruner'): {'best_score': 13.182650929420474,
                                     'best_params': {'n_estimators': 700,
                                                     'criterion': 'friedman_mse',
                                                     'max_depth': None,
                                                     'min_samples_split': 5,
                                                     'min_samples_leaf': 1,
                                                     'min_weight_fraction_leaf': 0.0,
                                                     'max_features': 'sqrt',
                                                     'max_leaf_nodes': 100,
                                                     'min_impurity_decrease': 0.0,
                                                     'n_jobs': -1,
                                                     'random_state': 42,
                                                     'verbose': 0,
                                                     'warm_start': False,
                                                     'ccp_alpha': 0.001},
                                     'test_mse': 13.182650929420474,
                                     'test_rmse': 3.63079205262715,
                                     'test_corr_coef': 0.9362480867949565,
                                     'pruner': 'MedianPruner'},
 ('Random Forest', 'NopPruner'): {'best_score': 11.272303056476819,
                                  'best_params': {'n_estimators': 100,
                                                  'criterion': 'friedman_mse',
                                                  'max_depth': 30,
                                                  'min_samples_split': 2,
                                                  'min_samples_leaf': 1,
                                                  'min_weight_fraction_leaf': 0.0,
                                                  'max_features': 0.5,
                                                  'max_leaf_nodes': 50,
                                                  'min_impurity_decrease': 0.1,
                                                  'n_jobs': -1,
                                                  'random_state': 42,
                                                  'verbose': 0,
                                                  'warm_start': False,
                                                  'ccp_alpha': 0.05},
                                  'test_mse': 11.272303056476819,
                                  'test_rmse': 3.3574250634194085,
                                  'test_corr_coef': 0.9440478855010435,
                                  'pruner': 'NopPruner'},
 ('Random Forest', 'PatientPruner'): {'best_score': 11.437119799459747,
                                      'best_params': {'n_estimators': 500,
                                                      'criterion': 'squared_error',
                                                      'max_depth': 20,
                                                      'min_samples_split': 2,
                                                      'min_samples_leaf': 1,
                                                      'min_weight_fraction_leaf': 0.0,
                                                      'max_features': 'sqrt',
                                                      'max_leaf_nodes': None,
                                                      'min_impurity_decrease': 0.0,
                                                      'n_jobs': -1,
                                                      'random_state': 42,
                                                      'verbose': 0,
                                                      'warm_start': False,
                                                      'ccp_alpha': 0.01},
                                      'test_mse': 11.437119799459747,
                                      'test_rmse': 3.3818811036847154,
                                      'test_corr_coef': 0.9432772188634867,
                                      'pruner': 'PatientPruner'},
 ('Random Forest', 'PercentilePruner'): {'best_score': 10.941206762081073,
                                         'best_params': {'n_estimators': 500,
                                                         'criterion': 'poisson',
                                                         'max_depth': 20,
                                                         'min_samples_split': 0.01,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_features': 0.5,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0.0,
                                                         'n_jobs': -1,
                                                         'random_state': 42,
                                                         'verbose': 0,
                                                         'warm_start': False,
                                                         'ccp_alpha': 0.0},
                                         'test_mse': 10.941206762081073,
                                         'test_rmse': 3.3077495011081286,
                                         'test_corr_coef': 0.94500835984303,
                                         'pruner': 'PercentilePruner'},
 ('Random Forest', 'SuccessiveHalvingPruner'): {'best_score': 10.880552388940954,
                                                'best_params': {'n_estimators': 500,
                                                                'criterion': 'squared_error',
                                                                'max_depth': 30,
                                                                'min_samples_split': 2,
                                                                'min_samples_leaf': 1,
                                                                'min_weight_fraction_leaf': 0.0,
                                                                'max_features': 0.5,
                                                                'max_leaf_nodes': None,
                                                                'min_impurity_decrease': 0.0,
                                                                'n_jobs': -1,
                                                                'random_state': 42,
                                                                'verbose': 0,
                                                                'warm_start': False,
                                                                'ccp_alpha': 0.001},
                                                'test_mse': 10.880552388940954,
                                                'test_rmse': 3.298568233179504,
                                                'test_corr_coef': 0.9451752554913052,
                                                'pruner': 'SuccessiveHalvingPruner'},
 ('Random Forest', 'HyperbandPruner'): {'best_score': 12.352278459566854,
                                        'best_params': {'n_estimators': 500,
                                                        'criterion': 'friedman_mse',
                                                        'max_depth': 40,
                                                        'min_samples_split': 2,
                                                        'min_samples_leaf': 1,
                                                        'min_weight_fraction_leaf': 0.01,
                                                        'max_features': 0.5,
                                                        'max_leaf_nodes': 100,
                                                        'min_impurity_decrease': 0.1,
                                                        'n_jobs': -1,
                                                        'random_state': 42,
                                                        'verbose': 0,
                                                        'warm_start': False,
                                                        'ccp_alpha': 0.001},
                                        'test_mse': 12.352278459566854,
                                        'test_rmse': 3.5145808369657474,
                                        'test_corr_coef': 0.9402669188906791,
                                        'pruner': 'HyperbandPruner'},
 ('Random Forest', 'ThresholdPruner'): {'best_score': 12.342074924476556,
                                        'best_params': {'n_estimators': 700,
                                                        'criterion': 'poisson',
                                                        'max_depth': 20,
                                                        'min_samples_split': 5,
                                                        'min_samples_leaf': 1,
                                                        'min_weight_fraction_leaf': 0.0,
                                                        'max_features': 0.5,
                                                        'max_leaf_nodes': None,
                                                        'min_impurity_decrease': 0.0,
                                                        'n_jobs': -1,
                                                        'random_state': 42,
                                                        'verbose': 0,
                                                        'warm_start': False,
                                                        'ccp_alpha': 0.0},
                                        'test_mse': 12.342074924476556,
                                        'test_rmse': 3.5131289365004177,
                                        'test_corr_coef': 0.9395016962237023,
                                        'pruner': 'ThresholdPruner'},
 ('Random Forest', 'WilcoxonPruner'): {'best_score': 10.774834068806634,
                                       'best_params': {'n_estimators': 200,
                                                       'criterion': 'squared_error',
                                                       'max_depth': None,
                                                       'min_samples_split': 2,
                                                       'min_samples_leaf': 1,
                                                       'min_weight_fraction_leaf': 0.0,
                                                       'max_features': 0.5,
                                                       'max_leaf_nodes': None,
                                                       'min_impurity_decrease': 0.0,
                                                       'n_jobs': -1,
                                                       'random_state': 42,
                                                       'verbose': 0,
                                                       'warm_start': False,
                                                       'ccp_alpha': 0.05},
                                       'test_mse': 10.774834068806634,
                                       'test_rmse': 3.2825042374392503,
                                       'test_corr_coef': 0.9460418459494322,
                                       'pruner': 'WilcoxonPruner'},
 ('Gradient Boosting', 'MedianPruner'): {'best_score': 6.893508647469353,
                                         'best_params': {'loss': 'squared_error',
                                                         'learning_rate': 0.2,
                                                         'n_estimators': 200,
                                                         'subsample': 0.5,
                                                         'criterion': 'friedman_mse',
                                                         'min_samples_split': 10,
                                                         'min_samples_leaf': 3,
                                                         'min_weight_fraction_leaf': 0.01,
                                                         'max_depth': 5,
                                                         'min_impurity_decrease': 0.0,
                                                         'init': None,
                                                         'random_state': 42,
                                                         'max_features': 'sqrt',
                                                         'alpha': 0.1,
                                                         'verbose': 0,
                                                         'max_leaf_nodes': 10,
                                                         'warm_start': False,
                                                         'validation_fraction': 0.1,
                                                         'n_iter_no_change': 20,
                                                         'tol': 0.001,
                                                         'ccp_alpha': 0.01},
                                         'test_mse': 6.893508647469353,
                                         'test_rmse': 2.6255492087312615,
                                         'test_corr_coef': 0.9651848123284126,
                                         'pruner': 'MedianPruner'},
 ('Gradient Boosting', 'NopPruner'): {'best_score': 7.763276701627249,
                                      'best_params': {'loss': 'squared_error',
                                                      'learning_rate': 0.05,
                                                      'n_estimators': 700,
                                                      'subsample': 0.5,
                                                      'criterion': 'friedman_mse',
                                                      'min_samples_split': 5,
                                                      'min_samples_leaf': 5,
                                                      'min_weight_fraction_leaf': 0.05,
                                                      'max_depth': 5,
                                                      'min_impurity_decrease': 0.0,
                                                      'init': None,
                                                      'random_state': 42,
                                                      'max_features': 0.5,
                                                      'alpha': 0.9,
                                                      'verbose': 0,
                                                      'max_leaf_nodes': 30,
                                                      'warm_start': False,
                                                      'validation_fraction': 0.1,
                                                      'n_iter_no_change': None,
                                                      'tol': 0.0001,
                                                      'ccp_alpha': 0.001},
                                      'test_mse': 7.763276701627249,
                                      'test_rmse': 2.786265727030939,
                                      'test_corr_coef': 0.960299321352052,
                                      'pruner': 'NopPruner'},
 ('Gradient Boosting', 'PatientPruner'): {'best_score': 7.502699891774882,
                                          'best_params': {'loss': 'squared_error',
                                                          'learning_rate': 0.1,
                                                          'n_estimators': 300,
                                                          'subsample': 0.7,
                                                          'criterion': 'friedman_mse',
                                                          'min_samples_split': 10,
                                                          'min_samples_leaf': 3,
                                                          'min_weight_fraction_leaf': 0.01,
                                                          'max_depth': 5,
                                                          'min_impurity_decrease': 0.0,
                                                          'init': None,
                                                          'random_state': 42,
                                                          'max_features': None,
                                                          'alpha': 0.1,
                                                          'verbose': 0,
                                                          'max_leaf_nodes': None,
                                                          'warm_start': False,
                                                          'validation_fraction': 0.1,
                                                          'n_iter_no_change': None,
                                                          'tol': 0.001,
                                                          'ccp_alpha': 0.01},
                                          'test_mse': 7.502699891774882,
                                          'test_rmse': 2.739105673714485,
                                          'test_corr_coef': 0.9619355831635854,
                                          'pruner': 'PatientPruner'},
 ('Gradient Boosting', 'PercentilePruner'): {'best_score': 6.834971436808225,
                                             'best_params': {'loss': 'huber',
                                                             'learning_rate': 0.1,
                                                             'n_estimators': 700,
                                                             'subsample': 0.5,
                                                             'criterion': 'friedman_mse',
                                                             'min_samples_split': 0.01,
                                                             'min_samples_leaf': 1,
                                                             'min_weight_fraction_leaf': 0.01,
                                                             'max_depth': 3,
                                                             'min_impurity_decrease': 0.0,
                                                             'init': None,
                                                             'random_state': 42,
                                                             'max_features': 0.5,
                                                             'alpha': 0.9,
                                                             'verbose': 0,
                                                             'max_leaf_nodes': 10,
                                                             'warm_start': False,
                                                             'validation_fraction': 0.1,
                                                             'n_iter_no_change': 20,
                                                             'tol': 0.0001,
                                                             'ccp_alpha': 0.0},
                                             'test_mse': 6.834971436808225,
                                             'test_rmse': 2.614377829772932,
                                             'test_corr_coef': 0.9651111711288763,
                                             'pruner': 'PercentilePruner'},
 ('Gradient Boosting', 'SuccessiveHalvingPruner'): {'best_score': 7.689690948162852,
                                                    'best_params': {'loss': 'squared_error',
                                                                    'learning_rate': 0.1,
                                                                    'n_estimators': 500,
                                                                    'subsample': 0.9,
                                                                    'criterion': 'squared_error',
                                                                    'min_samples_split': 10,
                                                                    'min_samples_leaf': 3,
                                                                    'min_weight_fraction_leaf': 0.01,
                                                                    'max_depth': 10,
                                                                    'min_impurity_decrease': 0.01,
                                                                    'init': None,
                                                                    'random_state': 42,
                                                                    'max_features': None,
                                                                    'alpha': 0.9,
                                                                    'verbose': 0,
                                                                    'max_leaf_nodes': 10,
                                                                    'warm_start': False,
                                                                    'validation_fraction': 0.1,
                                                                    'n_iter_no_change': 20,
                                                                    'tol': 0.001,
                                                                    'ccp_alpha': 0.0},
                                                    'test_mse': 7.689690948162852,
                                                    'test_rmse': 2.7730292007411053,
                                                    'test_corr_coef': 0.9608862446607661,
                                                    'pruner': 'SuccessiveHalvingPruner'},
 ('Gradient Boosting', 'HyperbandPruner'): {'best_score': 7.33313968984125,
                                            'best_params': {'loss': 'squared_error',
                                                            'learning_rate': 0.1,
                                                            'n_estimators': 200,
                                                            'subsample': 0.9,
                                                            'criterion': 'friedman_mse',
                                                            'min_samples_split': 5,
                                                            'min_samples_leaf': 0.01,
                                                            'min_weight_fraction_leaf': 0.01,
                                                            'max_depth': 5,
                                                            'min_impurity_decrease': 0.0,
                                                            'init': None,
                                                            'random_state': 42,
                                                            'max_features': None,
                                                            'alpha': 0.1,
                                                            'verbose': 0,
                                                            'max_leaf_nodes': 50,
                                                            'warm_start': False,
                                                            'validation_fraction': 0.1,
                                                            'n_iter_no_change': 20,
                                                            'tol': 0.001,
                                                            'ccp_alpha': 0.001},
                                            'test_mse': 7.33313968984125,
                                            'test_rmse': 2.7079770475100506,
                                            'test_corr_coef': 0.9630585579097957,
                                            'pruner': 'HyperbandPruner'},
 ('Gradient Boosting', 'ThresholdPruner'): {'best_score': 7.477744369168925,
                                            'best_params': {'loss': 'huber',
                                                            'learning_rate': 0.1,
                                                            'n_estimators': 700,
                                                            'subsample': 0.7,
                                                            'criterion': 'friedman_mse',
                                                            'min_samples_split': 10,
                                                            'min_samples_leaf': 1,
                                                            'min_weight_fraction_leaf': 0.01,
                                                            'max_depth': 7,
                                                            'min_impurity_decrease': 0.1,
                                                            'init': None,
                                                            'random_state': 42,
                                                            'max_features': None,
                                                            'alpha': 0.9,
                                                            'verbose': 0,
                                                            'max_leaf_nodes': 50,
                                                            'warm_start': False,
                                                            'validation_fraction': 0.1,
                                                            'n_iter_no_change': 20,
                                                            'tol': 0.001,
                                                            'ccp_alpha': 0.0},
                                            'test_mse': 7.477744369168925,
                                            'test_rmse': 2.7345464649862734,
                                            'test_corr_coef': 0.9619684919611741,
                                            'pruner': 'ThresholdPruner'},
 ('Gradient Boosting', 'WilcoxonPruner'): {'best_score': 6.721076366834078,
                                           'best_params': {'loss': 'huber',
                                                           'learning_rate': 0.05,
                                                           'n_estimators': 700,
                                                           'subsample': 0.9,
                                                           'criterion': 'friedman_mse',
                                                           'min_samples_split': 5,
                                                           'min_samples_leaf': 5,
                                                           'min_weight_fraction_leaf': 0.0,
                                                           'max_depth': 7,
                                                           'min_impurity_decrease': 0.01,
                                                           'init': None,
                                                           'random_state': 42,
                                                           'max_features': 0.5,
                                                           'alpha': 0.9,
                                                           'verbose': 0,
                                                           'max_leaf_nodes': 10,
                                                           'warm_start': False,
                                                           'validation_fraction': 0.1,
                                                           'n_iter_no_change': None,
                                                           'tol': 0.0001,
                                                           'ccp_alpha': 0.001},
                                           'test_mse': 6.721076366834078,
                                           'test_rmse': 2.5925038798108053,
                                           'test_corr_coef': 0.9662115898059317,
                                           'pruner': 'WilcoxonPruner'},
 ('XGBoost', 'MedianPruner'): {'best_score': 7.2354047049265535,
                               'best_params': {'n_estimators': 400,
                                               'learning_rate': 0.05,
                                               'max_depth': 7,
                                               'min_child_weight': 3,
                                               'gamma': 1,
                                               'subsample': 0.5,
                                               'colsample_bytree': 0.9,
                                               'colsample_bylevel': 0.7,
                                               'reg_alpha': 0.1,
                                               'reg_lambda': 0.1,
                                               'objective': 'reg:squarederror',
                                               'random_state': 42,
                                               'n_jobs': -1},
                               'test_mse': 7.2354047049265535,
                               'test_rmse': 2.6898707598928526,
                               'test_corr_coef': 0.9632217709829815,
                               'pruner': 'MedianPruner'},
 ('XGBoost', 'NopPruner'): {'best_score': 7.849364083074588,
                            'best_params': {'n_estimators': 300,
                                            'learning_rate': 0.05,
                                            'max_depth': 7,
                                            'min_child_weight': 1,
                                            'gamma': 0,
                                            'subsample': 0.7,
                                            'colsample_bytree': 0.5,
                                            'colsample_bylevel': 0.9,
                                            'reg_alpha': 0,
                                            'reg_lambda': 1,
                                            'objective': 'reg:squarederror',
                                            'random_state': 42,
                                            'n_jobs': -1},
                            'test_mse': 7.849364083074588,
                            'test_rmse': 2.8016716586842554,
                            'test_corr_coef': 0.9615165743735467,
                            'pruner': 'NopPruner'},
 ('XGBoost', 'PatientPruner'): {'best_score': 7.802370007652632,
                                'best_params': {'n_estimators': 400,
                                                'learning_rate': 0.15,
                                                'max_depth': 7,
                                                'min_child_weight': 3,
                                                'gamma': 0,
                                                'subsample': 0.5,
                                                'colsample_bytree': 0.9,
                                                'colsample_bylevel': 0.5,
                                                'reg_alpha': 0.01,
                                                'reg_lambda': 10,
                                                'objective': 'reg:squarederror',
                                                'random_state': 42,
                                                'n_jobs': -1},
                                'test_mse': 7.802370007652632,
                                'test_rmse': 2.7932722759610513,
                                'test_corr_coef': 0.9601176613813349,
                                'pruner': 'PatientPruner'},
 ('XGBoost', 'PercentilePruner'): {'best_score': 7.5038561309289245,
                                   'best_params': {'n_estimators': 400,
                                                   'learning_rate': 0.15,
                                                   'max_depth': 3,
                                                   'min_child_weight': 1,
                                                   'gamma': 0,
                                                   'subsample': 0.8,
                                                   'colsample_bytree': 0.7,
                                                   'colsample_bylevel': 0.5,
                                                   'reg_alpha': 0,
                                                   'reg_lambda': 1,
                                                   'objective': 'reg:squarederror',
                                                   'random_state': 42,
                                                   'n_jobs': -1},
                                   'test_mse': 7.5038561309289245,
                                   'test_rmse': 2.739316727019518,
                                   'test_corr_coef': 0.9620933960059359,
                                   'pruner': 'PercentilePruner'},
 ('XGBoost', 'SuccessiveHalvingPruner'): {'best_score': 7.753185403631509,
                                          'best_params': {'n_estimators': 500,
                                                          'learning_rate': 0.15,
                                                          'max_depth': 3,
                                                          'min_child_weight': 1,
                                                          'gamma': 0,
                                                          'subsample': 0.8,
                                                          'colsample_bytree': 0.5,
                                                          'colsample_bylevel': 0.7,
                                                          'reg_alpha': 0,
                                                          'reg_lambda': 10,
                                                          'objective': 'reg:squarederror',
                                                          'random_state': 42,
                                                          'n_jobs': -1},
                                          'test_mse': 7.753185403631509,
                                          'test_rmse': 2.7844542380207127,
                                          'test_corr_coef': 0.9605410174421329,
                                          'pruner': 'SuccessiveHalvingPruner'},
 ('XGBoost', 'HyperbandPruner'): {'best_score': 7.342762082653903,
                                  'best_params': {'n_estimators': 400,
                                                  'learning_rate': 0.15,
                                                  'max_depth': 7,
                                                  'min_child_weight': 5,
                                                  'gamma': 0.1,
                                                  'subsample': 0.8,
                                                  'colsample_bytree': 0.9,
                                                  'colsample_bylevel': 0.5,
                                                  'reg_alpha': 1,
                                                  'reg_lambda': 10,
                                                  'objective': 'reg:squarederror',
                                                  'random_state': 42,
                                                  'n_jobs': -1},
                                  'test_mse': 7.342762082653903,
                                  'test_rmse': 2.7097531405376953,
                                  'test_corr_coef': 0.9624799383766623,
                                  'pruner': 'HyperbandPruner'},
 ('XGBoost', 'ThresholdPruner'): {'best_score': 7.946463097455834,
                                  'best_params': {'n_estimators': 500,
                                                  'learning_rate': 0.1,
                                                  'max_depth': 5,
                                                  'min_child_weight': 5,
                                                  'gamma': 0,
                                                  'subsample': 0.9,
                                                  'colsample_bytree': 0.5,
                                                  'colsample_bylevel': 0.7,
                                                  'reg_alpha': 0,
                                                  'reg_lambda': 10,
                                                  'objective': 'reg:squarederror',
                                                  'random_state': 42,
                                                  'n_jobs': -1},
                                  'test_mse': 7.946463097455834,
                                  'test_rmse': 2.8189471611677708,
                                  'test_corr_coef': 0.9595107073054235,
                                  'pruner': 'ThresholdPruner'},
 ('XGBoost', 'WilcoxonPruner'): {'best_score': 7.68912600041155,
                                 'best_params': {'n_estimators': 400,
                                                 'learning_rate': 0.1,
                                                 'max_depth': 7,
                                                 'min_child_weight': 5,
                                                 'gamma': 0,
                                                 'subsample': 0.9,
                                                 'colsample_bytree': 0.9,
                                                 'colsample_bylevel': 0.5,
                                                 'reg_alpha': 1,
                                                 'reg_lambda': 5,
                                                 'objective': 'reg:squarederror',
                                                 'random_state': 42,
                                                 'n_jobs': -1},
                                 'test_mse': 7.68912600041155,
                                 'test_rmse': 2.772927334138338,
                                 'test_corr_coef': 0.9608967677017526,
                                 'pruner': 'WilcoxonPruner'},
 ('LightGBM', 'MedianPruner'): {'best_score': 6.942836699007481,
                                'best_params': {'n_estimators': 300,
                                                'learning_rate': 0.15,
                                                'num_leaves': 15,
                                                'max_depth': 5,
                                                'min_child_samples': 5,
                                                'subsample': 0.5,
                                                'colsample_bytree': 0.9,
                                                'reg_alpha': 0.1,
                                                'reg_lambda': 1,
                                                'min_child_weight': 0.01,
                                                'bagging_freq': 5,
                                                'objective': 'regression',
                                                'random_state': 42,
                                                'n_jobs': -1,
                                                'verbose': -1},
                                'test_mse': 6.942836699007481,
                                'test_rmse': 2.634926317567055,
                                'test_corr_coef': 0.9649381803805447,
                                'pruner': 'MedianPruner'},
 ('LightGBM', 'NopPruner'): {'best_score': 6.339384073454502,
                             'best_params': {'n_estimators': 500,
                                             'learning_rate': 0.15,
                                             'num_leaves': 15,
                                             'max_depth': 3,
                                             'min_child_samples': 5,
                                             'subsample': 1.0,
                                             'colsample_bytree': 0.5,
                                             'reg_alpha': 0.1,
                                             'reg_lambda': 0.1,
                                             'min_child_weight': 0.001,
                                             'bagging_freq': 1,
                                             'objective': 'regression',
                                             'random_state': 42,
                                             'n_jobs': -1,
                                             'verbose': -1},
                             'test_mse': 6.339384073454502,
                             'test_rmse': 2.5178133515919128,
                             'test_corr_coef': 0.9677038688754944,
                             'pruner': 'NopPruner'},
 ('LightGBM', 'PatientPruner'): {'best_score': 6.758719441603299,
                                 'best_params': {'n_estimators': 300,
                                                 'learning_rate': 0.1,
                                                 'num_leaves': 63,
                                                 'max_depth': 5,
                                                 'min_child_samples': 1,
                                                 'subsample': 0.5,
                                                 'colsample_bytree': 0.5,
                                                 'reg_alpha': 0,
                                                 'reg_lambda': 0.1,
                                                 'min_child_weight': 0.01,
                                                 'bagging_freq': 5,
                                                 'objective': 'regression',
                                                 'random_state': 42,
                                                 'n_jobs': -1,
                                                 'verbose': -1},
                                 'test_mse': 6.758719441603299,
                                 'test_rmse': 2.5997537271063385,
                                 'test_corr_coef': 0.9660330847044307,
                                 'pruner': 'PatientPruner'},
 ('LightGBM', 'PercentilePruner'): {'best_score': 6.613581221431991,
                                    'best_params': {'n_estimators': 100,
                                                    'learning_rate': 0.15,
                                                    'num_leaves': 15,
                                                    'max_depth': 7,
                                                    'min_child_samples': 5,
                                                    'subsample': 0.9,
                                                    'colsample_bytree': 0.9,
                                                    'reg_alpha': 0,
                                                    'reg_lambda': 0.1,
                                                    'min_child_weight': 1e-05,
                                                    'bagging_freq': 0,
                                                    'objective': 'regression',
                                                    'random_state': 42,
                                                    'n_jobs': -1,
                                                    'verbose': -1},
                                    'test_mse': 6.613581221431991,
                                    'test_rmse': 2.571688398976826,
                                    'test_corr_coef': 0.9663576043350901,
                                    'pruner': 'PercentilePruner'},
 ('LightGBM', 'SuccessiveHalvingPruner'): {'best_score': 6.471978196181523,
                                           'best_params': {'n_estimators': 300,
                                                           'learning_rate': 0.15,
                                                           'num_leaves': 63,
                                                           'max_depth': 7,
                                                           'min_child_samples': 5,
                                                           'subsample': 0.6,
                                                           'colsample_bytree': 0.7,
                                                           'reg_alpha': 0,
                                                           'reg_lambda': 1,
                                                           'min_child_weight': 1e-05,
                                                           'bagging_freq': 5,
                                                           'objective': 'regression',
                                                           'random_state': 42,
                                                           'n_jobs': -1,
                                                           'verbose': -1},
                                           'test_mse': 6.471978196181523,
                                           'test_rmse': 2.5440082932611525,
                                           'test_corr_coef': 0.9670652131417232,
                                           'pruner': 'SuccessiveHalvingPruner'},
 ('LightGBM', 'HyperbandPruner'): {'best_score': 6.417986013529069,
                                   'best_params': {'n_estimators': 400,
                                                   'learning_rate': 0.1,
                                                   'num_leaves': 15,
                                                   'max_depth': 5,
                                                   'min_child_samples': 5,
                                                   'subsample': 0.6,
                                                   'colsample_bytree': 0.7,
                                                   'reg_alpha': 0,
                                                   'reg_lambda': 1,
                                                   'min_child_weight': 0.1,
                                                   'bagging_freq': 5,
                                                   'objective': 'regression',
                                                   'random_state': 42,
                                                   'n_jobs': -1,
                                                   'verbose': -1},
                                   'test_mse': 6.417986013529069,
                                   'test_rmse': 2.533374432161395,
                                   'test_corr_coef': 0.9674069314589737,
                                   'pruner': 'HyperbandPruner'},
 ('LightGBM', 'ThresholdPruner'): {'best_score': 6.4359570823775485,
                                   'best_params': {'n_estimators': 100,
                                                   'learning_rate': 0.15,
                                                   'num_leaves': 63,
                                                   'max_depth': 5,
                                                   'min_child_samples': 5,
                                                   'subsample': 1.0,
                                                   'colsample_bytree': 0.9,
                                                   'reg_alpha': 0.01,
                                                   'reg_lambda': 0.1,
                                                   'min_child_weight': 0.1,
                                                   'bagging_freq': 5,
                                                   'objective': 'regression',
                                                   'random_state': 42,
                                                   'n_jobs': -1,
                                                   'verbose': -1},
                                   'test_mse': 6.4359570823775485,
                                   'test_rmse': 2.536918816670638,
                                   'test_corr_coef': 0.967215553727604,
                                   'pruner': 'ThresholdPruner'},
 ('LightGBM', 'WilcoxonPruner'): {'best_score': 6.9216779919281155,
                                  'best_params': {'n_estimators': 400,
                                                  'learning_rate': 0.05,
                                                  'num_leaves': 63,
                                                  'max_depth': 5,
                                                  'min_child_samples': 5,
                                                  'subsample': 1.0,
                                                  'colsample_bytree': 1.0,
                                                  'reg_alpha': 0.01,
                                                  'reg_lambda': 1,
                                                  'min_child_weight': 0.01,
                                                  'bagging_freq': 1,
                                                  'objective': 'regression',
                                                  'random_state': 42,
                                                  'n_jobs': -1,
                                                  'verbose': -1},
                                  'test_mse': 6.9216779919281155,
                                  'test_rmse': 2.6309082066708664,
                                  'test_corr_coef': 0.9648241722428089,
                                  'pruner': 'WilcoxonPruner'},
 ('GPBoost', 'MedianPruner'): {'best_score': 6.791909837287244,
                               'best_params': {'n_estimators': 100,
                                               'learning_rate': 0.15,
                                               'max_depth': 5,
                                               'num_leaves': 15,
                                               'min_child_samples': 5,
                                               'subsample': 0.9,
                                               'colsample_bytree': 0.9,
                                               'reg_alpha': 1.0,
                                               'reg_lambda': 0.5,
                                               'min_child_weight': 0.001,
                                               'random_state': 42,
                                               'n_jobs': -1,
                                               'verbose': -1},
                               'test_mse': 6.791909837287244,
                               'test_rmse': 2.606129282535163,
                               'test_corr_coef': 0.9653900561558387,
                               'pruner': 'MedianPruner'},
 ('GPBoost', 'NopPruner'): {'best_score': 6.91595927653158,
                            'best_params': {'n_estimators': 300,
                                            'learning_rate': 0.1,
                                            'max_depth': 5,
                                            'num_leaves': 31,
                                            'min_child_samples': 5,
                                            'subsample': 0.7,
                                            'colsample_bytree': 0.9,
                                            'reg_alpha': 0,
                                            'reg_lambda': 0,
                                            'min_child_weight': 0.01,
                                            'random_state': 42,
                                            'n_jobs': -1,
                                            'verbose': -1},
                            'test_mse': 6.91595927653158,
                            'test_rmse': 2.629821149152843,
                            'test_corr_coef': 0.9648515786821914,
                            'pruner': 'NopPruner'},
 ('GPBoost', 'PatientPruner'): {'best_score': 6.637421657872717,
                                'best_params': {'n_estimators': 300,
                                                'learning_rate': 0.1,
                                                'max_depth': 3,
                                                'num_leaves': 63,
                                                'min_child_samples': 5,
                                                'subsample': 0.5,
                                                'colsample_bytree': 0.5,
                                                'reg_alpha': 0.1,
                                                'reg_lambda': 0.1,
                                                'min_child_weight': 0.1,
                                                'random_state': 42,
                                                'n_jobs': -1,
                                                'verbose': -1},
                                'test_mse': 6.637421657872717,
                                'test_rmse': 2.5763194013694646,
                                'test_corr_coef': 0.9661244350949006,
                                'pruner': 'PatientPruner'},
 ('GPBoost', 'PercentilePruner'): {'best_score': 6.981840317927532,
                                   'best_params': {'n_estimators': 100,
                                                   'learning_rate': 0.15,
                                                   'max_depth': 7,
                                                   'num_leaves': 15,
                                                   'min_child_samples': 5,
                                                   'subsample': 0.7,
                                                   'colsample_bytree': 1.0,
                                                   'reg_alpha': 1.0,
                                                   'reg_lambda': 0.5,
                                                   'min_child_weight': 0.01,
                                                   'random_state': 42,
                                                   'n_jobs': -1,
                                                   'verbose': -1},
                                   'test_mse': 6.981840317927532,
                                   'test_rmse': 2.6423172250749025,
                                   'test_corr_coef': 0.9644025830408257,
                                   'pruner': 'PercentilePruner'},
 ('GPBoost', 'SuccessiveHalvingPruner'): {'best_score': 6.800522852492671,
                                          'best_params': {'n_estimators': 400,
                                                          'learning_rate': 0.05,
                                                          'max_depth': 5,
                                                          'num_leaves': 15,
                                                          'min_child_samples': 5,
                                                          'subsample': 0.9,
                                                          'colsample_bytree': 0.7,
                                                          'reg_alpha': 0.1,
                                                          'reg_lambda': 0.1,
                                                          'min_child_weight': 0.1,
                                                          'random_state': 42,
                                                          'n_jobs': -1,
                                                          'verbose': -1},
                                          'test_mse': 6.800522852492671,
                                          'test_rmse': 2.607781212543083,
                                          'test_corr_coef': 0.9653714985141384,
                                          'pruner': 'SuccessiveHalvingPruner'},
 ('GPBoost', 'HyperbandPruner'): {'best_score': 6.783683738250894,
                                  'best_params': {'n_estimators': 200,
                                                  'learning_rate': 0.05,
                                                  'max_depth': -1,
                                                  'num_leaves': 15,
                                                  'min_child_samples': 5,
                                                  'subsample': 0.9,
                                                  'colsample_bytree': 0.9,
                                                  'reg_alpha': 0.5,
                                                  'reg_lambda': 0,
                                                  'min_child_weight': 1e-05,
                                                  'random_state': 42,
                                                  'n_jobs': -1,
                                                  'verbose': -1},
                                  'test_mse': 6.783683738250894,
                                  'test_rmse': 2.6045505827783213,
                                  'test_corr_coef': 0.9655492785886066,
                                  'pruner': 'HyperbandPruner'},
 ('GPBoost', 'ThresholdPruner'): {'best_score': 6.6942049830778245,
                                  'best_params': {'n_estimators': 100,
                                                  'learning_rate': 0.15,
                                                  'max_depth': -1,
                                                  'num_leaves': 15,
                                                  'min_child_samples': 5,
                                                  'subsample': 1.0,
                                                  'colsample_bytree': 0.9,
                                                  'reg_alpha': 1.0,
                                                  'reg_lambda': 0,
                                                  'min_child_weight': 0.01,
                                                  'random_state': 42,
                                                  'n_jobs': -1,
                                                  'verbose': -1},
                                  'test_mse': 6.6942049830778245,
                                  'test_rmse': 2.5873161737750228,
                                  'test_corr_coef': 0.965893811326036,
                                  'pruner': 'ThresholdPruner'},
 ('GPBoost', 'WilcoxonPruner'): {'best_score': 6.9867218571446,
                                 'best_params': {'n_estimators': 200,
                                                 'learning_rate': 0.05,
                                                 'max_depth': 7,
                                                 'num_leaves': 15,
                                                 'min_child_samples': 5,
                                                 'subsample': 0.6,
                                                 'colsample_bytree': 0.7,
                                                 'reg_alpha': 0,
                                                 'reg_lambda': 0.1,
                                                 'min_child_weight': 0.1,
                                                 'random_state': 42,
                                                 'n_jobs': -1,
                                                 'verbose': -1},
                                 'test_mse': 6.9867218571446,
                                 'test_rmse': 2.6432407868267696,
                                 'test_corr_coef': 0.964703797234493,
                                 'pruner': 'WilcoxonPruner'},
 ('CatBoost', 'MedianPruner'): {'best_score': 9.035186244423537,
                                'best_params': {'iterations': 500,
                                                'learning_rate': 0.1,
                                                'depth': 6,
                                                'l2_leaf_reg': 7,
                                                'border_count': 64,
                                                'min_data_in_leaf': 10,
                                                'rsm': 1.0,
                                                'bagging_temperature': 1,
                                                'random_seed': 42,
                                                'verbose': 0},
                                'test_mse': 9.035186244423537,
                                'test_rmse': 3.005858653433913,
                                'test_corr_coef': 0.9552278367535287,
                                'pruner': 'MedianPruner'},
 ('CatBoost', 'NopPruner'): {'best_score': 8.343074054901802,
                             'best_params': {'iterations': 500,
                                             'learning_rate': 0.1,
                                             'depth': 4,
                                             'l2_leaf_reg': 3,
                                             'border_count': 128,
                                             'min_data_in_leaf': 1,
                                             'rsm': 0.8,
                                             'bagging_temperature': 10,
                                             'random_seed': 42,
                                             'verbose': 0},
                             'test_mse': 8.343074054901802,
                             'test_rmse': 2.8884379956824073,
                             'test_corr_coef': 0.9577106066495047,
                             'pruner': 'NopPruner'},
 ('CatBoost', 'PatientPruner'): {'best_score': 9.296767024256646,
                                 'best_params': {'iterations': 1000,
                                                 'learning_rate': 0.05,
                                                 'depth': 6,
                                                 'l2_leaf_reg': 7,
                                                 'border_count': 64,
                                                 'min_data_in_leaf': 20,
                                                 'rsm': 1.0,
                                                 'bagging_temperature': 1,
                                                 'random_seed': 42,
                                                 'verbose': 0},
                                 'test_mse': 9.296767024256646,
                                 'test_rmse': 3.049060023065575,
                                 'test_corr_coef': 0.9538042100847459,
                                 'pruner': 'PatientPruner'},
 ('CatBoost', 'PercentilePruner'): {'best_score': 8.468465973062987,
                                    'best_params': {'iterations': 1000,
                                                    'learning_rate': 0.1,
                                                    'depth': 4,
                                                    'l2_leaf_reg': 5,
                                                    'border_count': 32,
                                                    'min_data_in_leaf': 20,
                                                    'rsm': 0.6,
                                                    'bagging_temperature': 1,
                                                    'random_seed': 42,
                                                    'verbose': 0},
                                    'test_mse': 8.468465973062987,
                                    'test_rmse': 2.910062881290194,
                                    'test_corr_coef': 0.9573193591785231,
                                    'pruner': 'PercentilePruner'},
 ('CatBoost', 'SuccessiveHalvingPruner'): {'best_score': 8.858038028261644,
                                           'best_params': {'iterations': 500,
                                                           'learning_rate': 0.1,
                                                           'depth': 4,
                                                           'l2_leaf_reg': 3,
                                                           'border_count': 32,
                                                           'min_data_in_leaf': 5,
                                                           'rsm': 0.8,
                                                           'bagging_temperature': 1,
                                                           'random_seed': 42,
                                                           'verbose': 0},
                                           'test_mse': 8.858038028261644,
                                           'test_rmse': 2.976245626332216,
                                           'test_corr_coef': 0.9555608145155611,
                                           'pruner': 'SuccessiveHalvingPruner'},
 ('CatBoost', 'HyperbandPruner'): {'best_score': 8.065492487905287,
                                   'best_params': {'iterations': 1000,
                                                   'learning_rate': 0.1,
                                                   'depth': 4,
                                                   'l2_leaf_reg': 3,
                                                   'border_count': 128,
                                                   'min_data_in_leaf': 5,
                                                   'rsm': 0.6,
                                                   'bagging_temperature': 10,
                                                   'random_seed': 42,
                                                   'verbose': 0},
                                   'test_mse': 8.065492487905287,
                                   'test_rmse': 2.8399810717512337,
                                   'test_corr_coef': 0.9591866054335858,
                                   'pruner': 'HyperbandPruner'},
 ('CatBoost', 'ThresholdPruner'): {'best_score': 8.468465973062987,
                                   'best_params': {'iterations': 1000,
                                                   'learning_rate': 0.1,
                                                   'depth': 4,
                                                   'l2_leaf_reg': 5,
                                                   'border_count': 32,
                                                   'min_data_in_leaf': 10,
                                                   'rsm': 0.6,
                                                   'bagging_temperature': 10,
                                                   'random_seed': 42,
                                                   'verbose': 0},
                                   'test_mse': 8.468465973062987,
                                   'test_rmse': 2.910062881290194,
                                   'test_corr_coef': 0.9573193591785231,
                                   'pruner': 'ThresholdPruner'},
 ('CatBoost', 'WilcoxonPruner'): {'best_score': 8.468465973062987,
                                  'best_params': {'iterations': 1000,
                                                  'learning_rate': 0.1,
                                                  'depth': 4,
                                                  'l2_leaf_reg': 5,
                                                  'border_count': 32,
                                                  'min_data_in_leaf': 1,
                                                  'rsm': 0.6,
                                                  'bagging_temperature': 1,
                                                  'random_seed': 42,
                                                  'verbose': 0},
                                  'test_mse': 8.468465973062987,
                                  'test_rmse': 2.910062881290194,
                                  'test_corr_coef': 0.9573193591785231,
                                  'pruner': 'WilcoxonPruner'},
 ('NGBoost', 'MedianPruner'): {'best_score': 6.6047378917594495,
                               'best_params': {'n_estimators': 1000,
                                               'learning_rate': 0.05,
                                               'natural_gradient': True,
                                               'minibatch_frac': 0.5,
                                               'col_sample': 0.5,
                                               'Dist': ngboost.distns.normal.Normal,
                                               'Score': ngboost.scores.LogScore,
                                               'random_state': 42,
                                               'verbose': 0},
                               'test_mse': 6.6047378917594495,
                               'test_rmse': 2.5699684612382794,
                               'test_corr_coef': 0.9663963757035304,
                               'pruner': 'MedianPruner'},
 ('NGBoost', 'NopPruner'): {'best_score': 6.293004523413356,
                            'best_params': {'n_estimators': 200,
                                            'learning_rate': 0.1,
                                            'natural_gradient': True,
                                            'minibatch_frac': 0.7,
                                            'col_sample': 0.5,
                                            'Dist': ngboost.distns.normal.Normal,
                                            'Score': ngboost.scores.LogScore,
                                            'random_state': 42,
                                            'verbose': 0},
                            'test_mse': 6.293004523413356,
                            'test_rmse': 2.508586160253093,
                            'test_corr_coef': 0.967928045275244,
                            'pruner': 'NopPruner'},
 ('NGBoost', 'PatientPruner'): {'best_score': 6.408298514668425,
                                'best_params': {'n_estimators': 500,
                                                'learning_rate': 0.1,
                                                'natural_gradient': True,
                                                'minibatch_frac': 0.7,
                                                'col_sample': 0.5,
                                                'Dist': ngboost.distns.normal.Normal,
                                                'Score': ngboost.scores.LogScore,
                                                'random_state': 42,
                                                'verbose': 0},
                                'test_mse': 6.408298514668425,
                                'test_rmse': 2.531461734782579,
                                'test_corr_coef': 0.967313795872684,
                                'pruner': 'PatientPruner'},
 ('NGBoost', 'PercentilePruner'): {'best_score': 7.3055893885657746,
                                   'best_params': {'n_estimators': 1000,
                                                   'learning_rate': 0.03,
                                                   'natural_gradient': True,
                                                   'minibatch_frac': 0.5,
                                                   'col_sample': 0.5,
                                                   'Dist': ngboost.distns.normal.Normal,
                                                   'Score': ngboost.scores.LogScore,
                                                   'random_state': 42,
                                                   'verbose': 0},
                                   'test_mse': 7.3055893885657746,
                                   'test_rmse': 2.702885382062246,
                                   'test_corr_coef': 0.9627184230098423,
                                   'pruner': 'PercentilePruner'},
 ('NGBoost', 'SuccessiveHalvingPruner'): {'best_score': 7.256071126182388,
                                          'best_params': {'n_estimators': 1000,
                                                          'learning_rate': 0.03,
                                                          'natural_gradient': True,
                                                          'minibatch_frac': 0.5,
                                                          'col_sample': 0.7,
                                                          'Dist': ngboost.distns.normal.Normal,
                                                          'Score': ngboost.scores.LogScore,
                                                          'random_state': 42,
                                                          'verbose': 0},
                                          'test_mse': 7.256071126182388,
                                          'test_rmse': 2.6937095474795325,
                                          'test_corr_coef': 0.9629438977217578,
                                          'pruner': 'SuccessiveHalvingPruner'},
 ('NGBoost', 'HyperbandPruner'): {'best_score': 6.05342993358606,
                                  'best_params': {'n_estimators': 500,
                                                  'learning_rate': 0.1,
                                                  'natural_gradient': True,
                                                  'minibatch_frac': 0.7,
                                                  'col_sample': 0.5,
                                                  'Dist': ngboost.distns.normal.Normal,
                                                  'Score': ngboost.scores.LogScore,
                                                  'random_state': 42,
                                                  'verbose': 0},
                                  'test_mse': 6.05342993358606,
                                  'test_rmse': 2.460371909607582,
                                  'test_corr_coef': 0.9691779486007454,
                                  'pruner': 'HyperbandPruner'},
 ('NGBoost', 'ThresholdPruner'): {'best_score': 6.446423616710878,
                                  'best_params': {'n_estimators': 1000,
                                                  'learning_rate': 0.1,
                                                  'natural_gradient': True,
                                                  'minibatch_frac': 0.5,
                                                  'col_sample': 0.5,
                                                  'Dist': ngboost.distns.normal.Normal,
                                                  'Score': ngboost.scores.LogScore,
                                                  'random_state': 42,
                                                  'verbose': 0},
                                  'test_mse': 6.446423616710878,
                                  'test_rmse': 2.538980822438578,
                                  'test_corr_coef': 0.9675295478887768,
                                  'pruner': 'ThresholdPruner'},
 ('NGBoost', 'WilcoxonPruner'): {'best_score': 6.758097580770275,
                                 'best_params': {'n_estimators': 500,
                                                 'learning_rate': 0.1,
                                                 'natural_gradient': True,
                                                 'minibatch_frac': 0.7,
                                                 'col_sample': 0.5,
                                                 'Dist': ngboost.distns.normal.Normal,
                                                 'Score': ngboost.scores.LogScore,
                                                 'random_state': 42,
                                                 'verbose': 0},
                                 'test_mse': 6.758097580770275,
                                 'test_rmse': 2.59963412440487,
                                 'test_corr_coef': 0.9655177093742602,
                                 'pruner': 'WilcoxonPruner'},
 ('TabNet', 'MedianPruner'): {'best_score': 29.748139001731996,
                              'best_params': {'n_d': 32,
                                              'n_a': 8,
                                              'n_steps': 3,
                                              'gamma': 1.0,
                                              'lambda_sparse': 0.0001,
                                              'optimizer_params': {'lr': 0.02},
                                              'mask_type': 'sparsemax',
                                              'n_shared': 1,
                                              'n_independent': 1,
                                              'scheduler_params': {'step_size': 10,
                                                                   'gamma': 0.9},
                                              'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                              'seed': 42,
                                              'verbose': 0},
                              'test_mse': 29.748139001731996,
                              'test_rmse': 5.454185457218337,
                              'test_corr_coef': 0.8416970792661294,
                              'pruner': 'MedianPruner'},
 ('TabNet', 'NopPruner'): {'best_score': 23.199777293401414,
                           'best_params': {'n_d': 64,
                                           'n_a': 8,
                                           'n_steps': 7,
                                           'gamma': 1.0,
                                           'lambda_sparse': 0.01,
                                           'optimizer_params': {'lr': 0.02},
                                           'mask_type': 'entmax',
                                           'n_shared': 3,
                                           'n_independent': 2,
                                           'scheduler_params': {'step_size': 10,
                                                                'gamma': 0.9},
                                           'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                           'seed': 42,
                                           'verbose': 0},
                           'test_mse': 23.199777293401414,
                           'test_rmse': 4.816614712991004,
                           'test_corr_coef': 0.8894447746865505,
                           'pruner': 'NopPruner'},
 ('TabNet', 'PatientPruner'): {'best_score': 29.134691912775462,
                               'best_params': {'n_d': 32,
                                               'n_a': 32,
                                               'n_steps': 10,
                                               'gamma': 1.0,
                                               'lambda_sparse': 0.01,
                                               'optimizer_params': {'lr': 0.02},
                                               'mask_type': 'entmax',
                                               'n_shared': 1,
                                               'n_independent': 3,
                                               'scheduler_params': {'step_size': 10,
                                                                    'gamma': 0.9},
                                               'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                               'seed': 42,
                                               'verbose': 0},
                               'test_mse': 29.134691912775462,
                               'test_rmse': 5.39765614992058,
                               'test_corr_coef': 0.8561179691375683,
                               'pruner': 'PatientPruner'},
 ('TabNet', 'PercentilePruner'): {'best_score': 28.023707069823285,
                                  'best_params': {'n_d': 64,
                                                  'n_a': 32,
                                                  'n_steps': 10,
                                                  'gamma': 1.0,
                                                  'lambda_sparse': 0.0001,
                                                  'optimizer_params': {'lr': 0.02},
                                                  'mask_type': 'sparsemax',
                                                  'n_shared': 1,
                                                  'n_independent': 3,
                                                  'scheduler_params': {'step_size': 10,
                                                                       'gamma': 0.9},
                                                  'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                                  'seed': 42,
                                                  'verbose': 0},
                                  'test_mse': 28.023707069823285,
                                  'test_rmse': 5.293742255703736,
                                  'test_corr_coef': 0.8619148631546711,
                                  'pruner': 'PercentilePruner'},
 ('TabNet', 'SuccessiveHalvingPruner'): {'best_score': 27.147522906259738,
                                         'best_params': {'n_d': 64,
                                                         'n_a': 64,
                                                         'n_steps': 10,
                                                         'gamma': 1.0,
                                                         'lambda_sparse': 0.001,
                                                         'optimizer_params': {'lr': 0.02},
                                                         'mask_type': 'entmax',
                                                         'n_shared': 1,
                                                         'n_independent': 3,
                                                         'scheduler_params': {'step_size': 10,
                                                                              'gamma': 0.9},
                                                         'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                                         'seed': 42,
                                                         'verbose': 0},
                                         'test_mse': 27.147522906259738,
                                         'test_rmse': 5.210328483527669,
                                         'test_corr_coef': 0.872476295813559,
                                         'pruner': 'SuccessiveHalvingPruner'},
 ('TabNet', 'HyperbandPruner'): {'best_score': 23.823451877728704,
                                 'best_params': {'n_d': 8,
                                                 'n_a': 8,
                                                 'n_steps': 7,
                                                 'gamma': 1.0,
                                                 'lambda_sparse': 0.001,
                                                 'optimizer_params': {'lr': 0.02},
                                                 'mask_type': 'entmax',
                                                 'n_shared': 1,
                                                 'n_independent': 2,
                                                 'scheduler_params': {'step_size': 10,
                                                                      'gamma': 0.9},
                                                 'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                                 'seed': 42,
                                                 'verbose': 0},
                                 'test_mse': 23.823451877728704,
                                 'test_rmse': 4.880927358374503,
                                 'test_corr_coef': 0.8866238340368057,
                                 'pruner': 'HyperbandPruner'},
 ('TabNet', 'ThresholdPruner'): {'best_score': 20.061086699947932,
                                 'best_params': {'n_d': 64,
                                                 'n_a': 16,
                                                 'n_steps': 10,
                                                 'gamma': 1.0,
                                                 'lambda_sparse': 0.001,
                                                 'optimizer_params': {'lr': 0.02},
                                                 'mask_type': 'entmax',
                                                 'n_shared': 3,
                                                 'n_independent': 3,
                                                 'scheduler_params': {'step_size': 10,
                                                                      'gamma': 0.9},
                                                 'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                                 'seed': 42,
                                                 'verbose': 0},
                                 'test_mse': 20.061086699947932,
                                 'test_rmse': 4.478960448580444,
                                 'test_corr_coef': 0.8968312069653815,
                                 'pruner': 'ThresholdPruner'},
 ('TabNet', 'WilcoxonPruner'): {'best_score': 25.77890292588716,
                                'best_params': {'n_d': 32,
                                                'n_a': 32,
                                                'n_steps': 10,
                                                'gamma': 1.0,
                                                'lambda_sparse': 0.01,
                                                'optimizer_params': {'lr': 0.02},
                                                'mask_type': 'entmax',
                                                'n_shared': 3,
                                                'n_independent': 3,
                                                'scheduler_params': {'step_size': 10,
                                                                     'gamma': 0.9},
                                                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                                                'seed': 42,
                                                'verbose': 0},
                                'test_mse': 25.77890292588716,
                                'test_rmse': 5.077292873755379,
                                'test_corr_coef': 0.8754712747987742,
                                'pruner': 'WilcoxonPruner'},
 ('HistGradientBoosting', 'MedianPruner'): {'best_score': 7.543467275831011,
                                            'best_params': {'learning_rate': 0.1,
                                                            'max_iter': 400,
                                                            'max_depth': 5,
                                                            'min_samples_leaf': 10,
                                                            'max_leaf_nodes': 31,
                                                            'l2_regularization': 1.0,
                                                            'max_bins': 128,
                                                            'early_stopping': False,
                                                            'validation_fraction': 0.2,
                                                            'n_iter_no_change': 10,
                                                            'loss': 'squared_error',
                                                            'random_state': 42,
                                                            'verbose': 0},
                                            'test_mse': 7.543467275831011,
                                            'test_rmse': 2.746537324674655,
                                            'test_corr_coef': 0.9616114377144743,
                                            'pruner': 'MedianPruner'},
 ('HistGradientBoosting', 'NopPruner'): {'best_score': 7.323067966858103,
                                         'best_params': {'learning_rate': 0.15,
                                                         'max_iter': 200,
                                                         'max_depth': 5,
                                                         'min_samples_leaf': 5,
                                                         'max_leaf_nodes': 63,
                                                         'l2_regularization': 0.1,
                                                         'max_bins': 255,
                                                         'early_stopping': False,
                                                         'validation_fraction': 0.2,
                                                         'n_iter_no_change': 15,
                                                         'loss': 'squared_error',
                                                         'random_state': 42,
                                                         'verbose': 0},
                                         'test_mse': 7.323067966858103,
                                         'test_rmse': 2.7061167688882355,
                                         'test_corr_coef': 0.9627605114567329,
                                         'pruner': 'NopPruner'},
 ('HistGradientBoosting', 'PatientPruner'): {'best_score': 7.185115606850218,
                                             'best_params': {'learning_rate': 0.15,
                                                             'max_iter': 400,
                                                             'max_depth': 7,
                                                             'min_samples_leaf': 5,
                                                             'max_leaf_nodes': 15,
                                                             'l2_regularization': 0.1,
                                                             'max_bins': 64,
                                                             'early_stopping': False,
                                                             'validation_fraction': 0.2,
                                                             'n_iter_no_change': 10,
                                                             'loss': 'squared_error',
                                                             'random_state': 42,
                                                             'verbose': 0},
                                             'test_mse': 7.185115606850218,
                                             'test_rmse': 2.6805065951887186,
                                             'test_corr_coef': 0.9635854689537728,
                                             'pruner': 'PatientPruner'},
 ('HistGradientBoosting', 'PercentilePruner'): {'best_score': 7.323067966858103,
                                                'best_params': {'learning_rate': 0.15,
                                                                'max_iter': 200,
                                                                'max_depth': 5,
                                                                'min_samples_leaf': 5,
                                                                'max_leaf_nodes': 31,
                                                                'l2_regularization': 0.1,
                                                                'max_bins': 255,
                                                                'early_stopping': False,
                                                                'validation_fraction': 0.2,
                                                                'n_iter_no_change': 5,
                                                                'loss': 'squared_error',
                                                                'random_state': 42,
                                                                'verbose': 0},
                                                'test_mse': 7.323067966858103,
                                                'test_rmse': 2.7061167688882355,
                                                'test_corr_coef': 0.9627605114567329,
                                                'pruner': 'PercentilePruner'},
 ('HistGradientBoosting', 'SuccessiveHalvingPruner'): {'best_score': 7.323067966858103,
                                                       'best_params': {'learning_rate': 0.15,
                                                                       'max_iter': 200,
                                                                       'max_depth': 5,
                                                                       'min_samples_leaf': 5,
                                                                       'max_leaf_nodes': 31,
                                                                       'l2_regularization': 0.1,
                                                                       'max_bins': 128,
                                                                       'early_stopping': False,
                                                                       'validation_fraction': 0.1,
                                                                       'n_iter_no_change': 15,
                                                                       'loss': 'squared_error',
                                                                       'random_state': 42,
                                                                       'verbose': 0},
                                                       'test_mse': 7.323067966858103,
                                                       'test_rmse': 2.7061167688882355,
                                                       'test_corr_coef': 0.9627605114567329,
                                                       'pruner': 'SuccessiveHalvingPruner'},
 ('HistGradientBoosting', 'HyperbandPruner'): {'best_score': 7.084038147266376,
                                               'best_params': {'learning_rate': 0.15,
                                                               'max_iter': 100,
                                                               'max_depth': 5,
                                                               'min_samples_leaf': 5,
                                                               'max_leaf_nodes': 15,
                                                               'l2_regularization': 0.0,
                                                               'max_bins': 128,
                                                               'early_stopping': False,
                                                               'validation_fraction': 0.2,
                                                               'n_iter_no_change': 15,
                                                               'loss': 'squared_error',
                                                               'random_state': 42,
                                                               'verbose': 0},
                                               'test_mse': 7.084038147266376,
                                               'test_rmse': 2.6615856452998794,
                                               'test_corr_coef': 0.9638241814772153,
                                               'pruner': 'HyperbandPruner'},
 ('HistGradientBoosting', 'ThresholdPruner'): {'best_score': 7.516099651812948,
                                               'best_params': {'learning_rate': 0.1,
                                                               'max_iter': 300,
                                                               'max_depth': 5,
                                                               'min_samples_leaf': 10,
                                                               'max_leaf_nodes': 15,
                                                               'l2_regularization': 1.0,
                                                               'max_bins': 255,
                                                               'early_stopping': False,
                                                               'validation_fraction': 0.1,
                                                               'n_iter_no_change': 10,
                                                               'loss': 'squared_error',
                                                               'random_state': 42,
                                                               'verbose': 0},
                                               'test_mse': 7.516099651812948,
                                               'test_rmse': 2.741550592605022,
                                               'test_corr_coef': 0.9616950051683296,
                                               'pruner': 'ThresholdPruner'},
 ('HistGradientBoosting', 'WilcoxonPruner'): {'best_score': 7.183181192498337,
                                              'best_params': {'learning_rate': 0.15,
                                                              'max_iter': 100,
                                                              'max_depth': 5,
                                                              'min_samples_leaf': 5,
                                                              'max_leaf_nodes': 63,
                                                              'l2_regularization': 0.0,
                                                              'max_bins': 128,
                                                              'early_stopping': False,
                                                              'validation_fraction': 0.1,
                                                              'n_iter_no_change': 10,
                                                              'loss': 'squared_error',
                                                              'random_state': 42,
                                                              'verbose': 0},
                                              'test_mse': 7.183181192498337,
                                              'test_rmse': 2.680145740906329,
                                              'test_corr_coef': 0.9634064407281163,
                                              'pruner': 'WilcoxonPruner'},
 ('PGBM', 'MedianPruner'): {'best_score': 7.411087162654288,
                            'best_params': {'n_estimators': 300,
                                            'learning_rate': 0.1,
                                            'max_leaves': 15,
                                            'min_split_gain': 0.0,
                                            'reg_lambda': 10.0,
                                            'feature_fraction': 0.9,
                                            'bagging_fraction': 0.7,
                                            'tree_correlation': 0.3,
                                            'min_data_in_leaf': 3,
                                            'max_bin': 128,
                                            'distribution': 'normal'},
                            'test_mse': 7.411087162654288,
                            'test_rmse': 2.7223312000295423,
                            'test_corr_coef': 0.9620991973469349,
                            'pruner': 'MedianPruner'},
 ('PGBM', 'NopPruner'): {'best_score': 6.962272728215163,
                         'best_params': {'n_estimators': 300,
                                         'learning_rate': 0.1,
                                         'max_leaves': 25,
                                         'min_split_gain': 0.0,
                                         'reg_lambda': 5.0,
                                         'feature_fraction': 0.5,
                                         'bagging_fraction': 0.7,
                                         'tree_correlation': 0.1,
                                         'min_data_in_leaf': 5,
                                         'max_bin': 128,
                                         'distribution': 'studentt'},
                         'test_mse': 6.962272728215163,
                         'test_rmse': 2.638611894200275,
                         'test_corr_coef': 0.9648235653869933,
                         'pruner': 'NopPruner'},
 ('PGBM', 'PatientPruner'): {'best_score': 7.07871143085572,
                             'best_params': {'n_estimators': 200,
                                             'learning_rate': 0.15,
                                             'max_leaves': 44,
                                             'min_split_gain': 0.1,
                                             'reg_lambda': 1.0,
                                             'feature_fraction': 0.7,
                                             'bagging_fraction': 0.7,
                                             'tree_correlation': 0.0,
                                             'min_data_in_leaf': 5,
                                             'max_bin': 128,
                                             'distribution': 'laplace'},
                             'test_mse': 7.07871143085572,
                             'test_rmse': 2.660584791141925,
                             'test_corr_coef': 0.963927642925922,
                             'pruner': 'PatientPruner'},
 ('PGBM', 'PercentilePruner'): {'best_score': 6.8496223951523225,
                                'best_params': {'n_estimators': 500,
                                                'learning_rate': 0.1,
                                                'max_leaves': 26,
                                                'min_split_gain': 1.0,
                                                'reg_lambda': 5.0,
                                                'feature_fraction': 0.7,
                                                'bagging_fraction': 0.7,
                                                'tree_correlation': 0.3,
                                                'min_data_in_leaf': 5,
                                                'max_bin': 256,
                                                'distribution': 'studentt'},
                                'test_mse': 6.8496223951523225,
                                'test_rmse': 2.617178326968249,
                                'test_corr_coef': 0.9650678810916228,
                                'pruner': 'PercentilePruner'},
 ('PGBM', 'SuccessiveHalvingPruner'): {'best_score': 7.386352397360811,
                                       'best_params': {'n_estimators': 100,
                                                       'learning_rate': 0.15,
                                                       'max_leaves': 17,
                                                       'min_split_gain': 0.5,
                                                       'reg_lambda': 1.0,
                                                       'feature_fraction': 0.9,
                                                       'bagging_fraction': 0.9,
                                                       'tree_correlation': 0.3,
                                                       'min_data_in_leaf': 5,
                                                       'max_bin': 64,
                                                       'distribution': 'laplace'},
                                       'test_mse': 7.386352397360811,
                                       'test_rmse': 2.7177844648464697,
                                       'test_corr_coef': 0.962229427551159,
                                       'pruner': 'SuccessiveHalvingPruner'},
 ('PGBM', 'HyperbandPruner'): {'best_score': 6.9223305669589,
                               'best_params': {'n_estimators': 500,
                                               'learning_rate': 0.05,
                                               'max_leaves': 47,
                                               'min_split_gain': 0.1,
                                               'reg_lambda': 1.0,
                                               'feature_fraction': 0.5,
                                               'bagging_fraction': 0.7,
                                               'tree_correlation': 0.3,
                                               'min_data_in_leaf': 5,
                                               'max_bin': 64,
                                               'distribution': 'laplace'},
                               'test_mse': 6.9223305669589,
                               'test_rmse': 2.631032224614305,
                               'test_corr_coef': 0.9651073798359422,
                               'pruner': 'HyperbandPruner'},
 ('PGBM', 'ThresholdPruner'): {'best_score': 6.676373983537216,
                               'best_params': {'n_estimators': 200,
                                               'learning_rate': 0.15,
                                               'max_leaves': 25,
                                               'min_split_gain': 0.1,
                                               'reg_lambda': 10.0,
                                               'feature_fraction': 0.7,
                                               'bagging_fraction': 0.7,
                                               'tree_correlation': 0.0,
                                               'min_data_in_leaf': 3,
                                               'max_bin': 128,
                                               'distribution': 'studentt'},
                               'test_mse': 6.676373983537216,
                               'test_rmse': 2.583868027500092,
                               'test_corr_coef': 0.9661443221631455,
                               'pruner': 'ThresholdPruner'},
 ('PGBM', 'WilcoxonPruner'): {'best_score': 7.351888809415018,
                              'best_params': {'n_estimators': 300,
                                              'learning_rate': 0.1,
                                              'max_leaves': 20,
                                              'min_split_gain': 0.1,
                                              'reg_lambda': 0.1,
                                              'feature_fraction': 0.7,
                                              'bagging_fraction': 0.9,
                                              'tree_correlation': 0.0,
                                              'min_data_in_leaf': 5,
                                              'max_bin': 128,
                                              'distribution': 'laplace'},
                              'test_mse': 7.351888809415018,
                              'test_rmse': 2.7114366688925298,
                              'test_corr_coef': 0.9625146907375025,
                              'pruner': 'WilcoxonPruner'}}

```

# **Conformal Predictions with Lightgbm**

```python
best_params = get_best_model_params(best_scores_autosampler, 'LightGBM')
```

```python
conformal_predictions_MAPIE(LGBMRegressor, best_params, X_train, y_train, X_test, y_test, "LightGBM","./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/LightGBM.xlsx")
```

```python
conformal_predictions_PUNCC(X_train, y_train, X_test, y_test, best_scores_autosampler, LGBMRegressor,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/LightGBM.xlsx",best_params)
```

```python
best_params = get_best_model_params(best_scores_autosampler, 'LightGBM')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, LGBMRegressor, best_params, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/LightGBM.xlsx", "LightGBM Prediction Intervals")
```

# **Conformal Predictions with XGBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'XGBoost')
```

```python
conformal_predictions_MAPIE(XGBRegressor, best_params, X_train, y_train, X_test, y_test, "XGBoost", "./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/XGBoost.xlsx")
```

```python
conformal_predictions_PUNCC(X_train, y_train, X_test, y_test, best_scores_autosampler, XGBRegressor,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/XGBoost.xlsx",best_params)
```

```python
best_params = get_best_model_params(best_scores_autosampler, 'XGBoost')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, XGBRegressor, best_params,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/XGBoost.xlsx", "XGBR Prediction Intervals")
```

# **Conformal Predictions with GPBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'GPBoost')
```

```python
conformal_predictions_MAPIE(GPBoostRegressor, best_params, X_train, y_train, X_test, y_test, "GPBoost","./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/GPBoost.xlsx")
```

```python
conformal_predictions_PUNCC(X_train, y_train, X_test, y_test, best_scores_autosampler, GPBoostRegressor,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/GPBoost.xlsx",best_params)
```

```python
best_params = get_best_model_params(best_scores_autosampler, 'GPBoost')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, GPBoostRegressor, best_params, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/GPBoost.xlsx", "GPBoost Prediction Intervals")
```

# **Conformal Predictions with NGBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'NGBoost')
```

```python
conformal_predictions_MAPIE(NGBRegressor, best_params, X_train, y_train, X_test, y_test, "NGBoost","./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/NGBoost.xlsx")
```

```python
conformal_predictions_PUNCC(X_train, y_train, X_test, y_test, best_scores_autosampler, NGBRegressor,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/NGBoost.xlsx",best_params)
```

```python
best_params = get_best_model_params(best_scores_autosampler, 'NGBoost')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, NGBRegressor, best_params,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/NGBoost.xlsx","NGBoost Prediction Intervals")
```

# **Conformal Predictions with Gradient Boosting**

```python
best_params = get_best_model_params(best_scores_autosampler, 'GradientBoosting')
```

```python
conformal_predictions_MAPIE(GradientBoostingRegressor, best_params, X_train, y_train, X_test, y_test, "GPBoost","./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/Gradient Boosting.xlsx")
```

```python
conformal_predictions_PUNCC(X_train, y_train, X_test, y_test, best_scores_autosampler, GradientBoostingRegressor,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/Gradient Boosting.xlsx",best_params)
```

```python
best_params = get_best_model_params(best_scores_autosampler, 'GradientBoosting')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, GradientBoostingRegressor, best_params, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/Gradient Boosting.xlsx", "Gradient Boosting Prediction Intervals")
```

# **Conformal Predictions with CatBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'CatBoost')
```

```python
conformal_predictions_MAPIE(CatBoostRegressor, best_params, X_train, y_train, X_test, y_test, "CatBoost","./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/CatBoost.xlsx")
```

```python
conformal_predictions_PUNCC(X_train, y_train, X_test, y_test, best_scores_autosampler, CatBoostRegressor,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/CatBoost.xlsx")
```

```python
best_params = get_best_model_params(best_scores_autosampler, 'CatBoost')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, CatBoostRegressor, best_params, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/CatBoost.xlsx", "CatBoost Prediction Intervals")
```

# **Conformal Predictions with HistGradientBoosting**

```python
best_params = get_best_model_params(best_scores_autosampler, 'HGBR')
```

```python
conformal_predictions_MAPIE(HistGradientBoostingRegressor, best_params, X_train, y_train, X_test, y_test, "HGBR","./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/HGBM.xlsx")
```

```python
conformal_predictions_PUNCC(X_train, y_train, X_test, y_test, best_scores_autosampler,HistGradientBoostingRegressor,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/HGBM.xlsx",best_params)
```

```python
best_params = get_best_model_params(best_scores_autosampler, 'HGBR')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, HistGradientBoostingRegressor, best_params,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/HGBM.xlsx", "HGBR Prediction Intervals")
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

# Add actual target values to the DataFrame
predictions_PGBM_df['Actual'] = y_test.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_PGBM_df.head())
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
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



def conformal_predictions_MAPIE_PGBM(
    model_class,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path
):
    SEED = 123
    np.random.seed(SEED)
    MISCOVERAGE = 0.05
    warnings.filterwarnings("ignore")
    os.makedirs('output/regression', exist_ok=True)

    # Convert y_test to a NumPy array if it has to_numpy
    y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test

    # Map method names to descriptive labels
    strategies_names = {
        'SCP': 'Split Conformal Prediction',
        'CV+': 'Cross Validation +',
        'J+aB': 'Jackknife+ after Bootstrapping',
    }

    # Corresponding parameters for each strategy
    strategies_params = {
        'SCP': {'method': 'base', 'cv': KFold(n_splits=5)},
        'CV+': {'method': "plus", 'cv': 10},
        'J+aB': {'method': "minmax", 'cv': Subsample(n_resamplings=50)},
    }

    # Create a base estimator object for each strategy
    base_estimators = {
        strat_name: model_class(**best_params) for strat_name in strategies_names
    }

    y_pred, int_pred = {}, {}
    y_pred_train, int_pred_train = {}, {} # Store train predictions and intervals
    # Fit & predict using MapieRegressor for each strategy
    for strat_name, base_estimator in base_estimators.items():
        mapie = MapieRegressor(base_estimator, **strategies_params[strat_name])
        mapie.fit(X_train, y_train)
        y_pred[strat_name], int_pred[strat_name] = mapie.predict(X_test, alpha=MISCOVERAGE)
        y_pred_train[strat_name], int_pred_train[strat_name] = mapie.predict(X_train, alpha=MISCOVERAGE) # Predict on train set


    # Calculate metrics
    coverages: dict = validate.coverage(int_pred, y_test)
    widths: dict = validate.width(int_pred)
    rmse_vals: dict = validate.rmse(y_pred, y_test)
    cwc_vals: dict = validate.cwc(int_pred, y_test, MISCOVERAGE)


    # Print metrics and prepare for plotting
    for strat_name in strategies_names.keys():
        if strat_name in y_pred:
            # Calculate RMSE (already calculated in validate.rmse, using rmse_vals now)
            rmse = rmse_vals[strat_name]

            # Calculate interval width (already calculated in validate.width, using widths now)
            interval_width = widths[strat_name]

            # Calculate coverage (already calculated in validate.coverage, using coverages now)
            coverage = coverages[strat_name]

            # Calculate CWC (already calculated in validate.cwc, using cwc_vals now)
            cwc = cwc_vals[strat_name]

            # Calculate SSC (Symmetric Scoring Criterion)
            ssc = np.mean((y_test - y_pred[strat_name])**2 + (int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0])**2)

            print(f"Strategy: {strategies_names[strat_name]}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Interval Width: {interval_width:.4f}")
            print(f"  Coverage: {coverage * 100:.2f}%")
            print(f"  CWC: {cwc:.4f}")
            print(f"  SSC: {ssc:.4f}")
            print()

    # Plot the intervals and predictions for each strategy
    _strategies = [_s for _s in strategies_names.keys() if _s != 'SCP']
    NUM_BINS: int = 10
    n_figs: int = len(strategies_names)*3 + 1 + len(_strategies)
    fig, axs = plt.subplots(nrows=n_figs, figsize=(12, 6 * n_figs))
    plot_index = 0

    # Individual strategy plots (original plots)
    for _i, strat_name in enumerate(strategies_names.keys()):
        if strat_name in y_pred:
            # Predicted intervals
            _y_pred_low = int_pred[strat_name][:, 0].ravel()
            _y_pred_up = int_pred[strat_name][:, 1].ravel()

            # Visualize
            sample_indices = np.arange(len(X_test))
            axs[plot_index].scatter(sample_indices, y_test_np, label='Test data', color='blue', s=10, alpha=0.7)
            axs[plot_index].fill_between(sample_indices, _y_pred_low, _y_pred_up,
                                            color='gray', alpha=0.5, label='Prediction interval')
            axs[plot_index].plot(sample_indices, y_pred[strat_name], color='red',
                                    label='Predicted mean', linewidth=1)
            axs[plot_index].set_title(f'{strategies_names[strat_name]} ({model_name})')
            axs[plot_index].set_xlabel('Sample Number')
            axs[plot_index].set_ylabel('str')
            axs[plot_index].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plot_index += 1

    # Comparison subplot (moved after individual strategy plots)
    comparison_ax = axs[plot_index]
    sample_indices = np.arange(len(X_test))
    comparison_ax.scatter(sample_indices, y_test_np, label='Test data', color='blue', s=10, alpha=0.7)
    for strat_name in y_pred:
        _y_pred_low = int_pred[strat_name][:, 0].ravel()
        _y_pred_up = int_pred[strat_name][:, 1].ravel()
        comparison_ax.fill_between(sample_indices, _y_pred_low, _y_pred_up,
                                        alpha=0.3, label=f'{strat_name} interval')
        comparison_ax.plot(sample_indices, y_pred[strat_name],
                                label=f'{strat_name} mean', linewidth=1)

    comparison_ax.set_title(f'Comparison of Strategies ({model_name})')
    comparison_ax.set_xlabel('Sample Number')
    comparison_ax.set_ylabel('str')
    comparison_ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plot_index += 1


    # Goodness plots
    for _i, strat_name in enumerate(strategies_names.keys()):
        if strat_name in y_pred:
            _y_pred_low_goodness = int_pred[strat_name][:, 0].ravel()
            _y_pred_up_goodness = int_pred[strat_name][:, 1].ravel()

            axs[plot_index] = visualize.goodness(
                y_test, y_pred[strat_name],
                _y_pred_low_goodness,
                _y_pred_up_goodness,
                coverages[strat_name],
                widths[strat_name],
                rmse_vals[strat_name],
                cwc_vals[strat_name],
                ax=axs[plot_index],
                title=f'{strategies_names[strat_name]} Goodness ({model_name})', # Removed subsample
            )
            axs[plot_index].set_ylabel('Predicted str')
            plot_index += 1

    # Width Occurrence plots
    _x_max = (1+1e-3) * np.max([np.abs(int_pred[strat_name][:, 0] - int_pred[strat_name][:, 1]) for strat_name in strategies_names.keys() if strat_name in int_pred])
    _x_min = np.min([np.abs(int_pred[strat_name][:, 0] - int_pred[strat_name][:, 1]) for strat_name in strategies_names.keys() if strat_name in int_pred])

    for _i, strat_name in enumerate(strategies_names.keys()):
        if strat_name in y_pred:
            axs[plot_index] = visualize.width_size_occurrence(
                int_pred[strat_name],
                train_intervals=int_pred_train[strat_name],
                num_bins=10,
                ax=axs[plot_index],
                x_lim=[_x_min, _x_max],
                title=f'{strategies_names[strat_name]} Width Occurrence',
            )
            axs[plot_index].set_ylabel('Frequency')
            plot_index += 1

    # Coverage vs Width plots
    cond_coverages: dict = validate.cond_coverage(int_pred, y_test, num_bins=NUM_BINS)
    for _i, _strat in enumerate(_strategies):
        axs[plot_index] = visualize.coverage_by_width(
            y_test, int_pred[_strat], MISCOVERAGE,
            cond_coverages[_strat],
            num_bins=NUM_BINS,
            ax=axs[plot_index],
            title=f'{strategies_names[_strat]} Coverage vs Width',
        )
        plot_index += 1


    fig.tight_layout()

    # Save plots to Excel
    # Save plots to Excel
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image(PImage.open(buf)) # Use PIL to open from BytesIO

        # Load the workbook and add a new sheet
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        sheet_name = 'conformal_predictions_MAPIE'
        if sheet_name in workbook.sheetnames:
            sheet_name += '_new'
        worksheet = workbook.create_sheet(title=sheet_name)

        # Add the image to the worksheet
        worksheet.add_image(img, 'A1')

        #NEW: Save all predictions/intervals to one worksheet
        all_rows = []
        # Always use y_test_np for consistent array-ness
        for strat_name in y_pred:
            y_pred_np = y_pred[strat_name].ravel()
            ymin_np = int_pred[strat_name][:, 0].ravel()
            ymax_np = int_pred[strat_name][:, 1].ravel()
            width_np = ymax_np - ymin_np
            is_covered_np = ((y_test_np >= ymin_np) & (y_test_np <= ymax_np)).astype(int)
            residual_np = y_test_np - y_pred_np

            strat_df = pd.DataFrame({
                'sample_index': np.arange(len(y_pred_np)),
                'strategy': strat_name,
                'y_true': y_test_np,
                'y_pred': y_pred_np,
                'ymin': ymin_np,
                'ymax': ymax_np,
                'width': width_np,
                'is_covered': is_covered_np,
                'residual': residual_np
            })
            all_rows.append(strat_df)
        all_export_df = pd.concat(all_rows, axis=0, ignore_index=True)

        all_pred_sheet_name = 'all_pred_values'
        base_name = all_pred_sheet_name
        idx = 1
        while all_pred_sheet_name in workbook.sheetnames:
            all_pred_sheet_name = f'{base_name}_new{idx}'
            idx += 1

        worksheet2 = workbook.create_sheet(title=all_pred_sheet_name)

        for j, col_name in enumerate(all_export_df.columns, 1):
            worksheet2.cell(row=1, column=j, value=col_name)
        for i, row in all_export_df.iterrows():
            for j, value in enumerate(row, 1):
                worksheet2.cell(row=i+2, column=j, value=float(value) if isinstance(value, np.generic) else value)

        # Save the workbook
        _ensure_parent_dir(excel_file_path)
        workbook.save(_ensure_parent_dir(excel_file_path))



# Example call (assuming best_params, X_train, y_train, X_test, y_test are defined elsewhere)
conformal_predictions_MAPIE_PGBM(
    model_class=PGBMWrapper,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="PGBM",
    excel_file_path="./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/PGBM.xlsx"
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
def conformal_predictions_PUNCC_PGBM(
    X_train,
    y_train,
    X_test,
    y_test,
    best_scores_autosampler,
    model_class,
    excel_file_path=None,
    model_params=None,
    alpha=0.1
):
    if model_params is None:
        if model_class.__name__ == 'PGBMWrapper':
            model_params = get_best_model_params(best_scores_autosampler, 'PGBM')
        else:
            model_params = {}

    # Initialize model
    try:
        model = model_class(**model_params)
    except TypeError as e:
        print(f"Error initializing model {model_class.__name__}: {e}")
        return

    # Convert to NumPy arrays if needed
    y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
    model.fit(X_train, y_train_np)

    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

    def evaluate_cp(X_test_eval, y_test_eval, model_cp, alpha_val):
        X_test_np = X_test_eval.to_numpy() if isinstance(X_test_eval, pd.DataFrame) else np.array(X_test_eval)
        y_pred, y_pred_lower, y_pred_upper = model_cp.predict(X_test_np, alpha=alpha_val)
        sharpness = regression_sharpness(y_pred_lower, y_pred_upper)
        coverage = regression_mean_coverage(y_test_eval, y_pred_lower, y_pred_upper)
        return y_pred, y_pred_lower, y_pred_upper, sharpness, coverage

    # Base predictor setup
    base_predictor = BasePredictor(model, is_trained=True)

    # SplitCP
    splitcp = SplitCP(base_predictor, train=True, random_state=0)
    splitcp.fit(X=X_train, y=y_train_np, fit_ratio=0.5)

    y_pred_sc, y_lower_sc, y_upper_sc, sharpness_sc, coverage_sc = evaluate_cp(X_test, y_test_np, splitcp, alpha)

    # CVPlus
    cvplus = CVPlus(base_predictor, K=5, random_state=0)
    cvplus.fit(X=X_train, y=y_train_np)
    y_pred, y_lower_cv, y_upper_cv, sharpness_cv, coverage_cv = evaluate_cp(X_test, y_test_np, cvplus, alpha)

    # CQR
    X_fit, X_calib, y_fit, y_calib = train_test_split(X_train, y_train_np, test_size=0.5, random_state=0)
    upper_quantile_model = model_class(**model_params)
    lower_quantile_model = model_class(**model_params)
    upper_quantile_model.fit(X_fit, y_fit)
    lower_quantile_model.fit(X_fit, y_fit)
    dualpredictor = DualPredictor([lower_quantile_model, upper_quantile_model], is_trained=[True, True])
    cqr = CQR(dualpredictor, train=False)
    cqr.fit(X_calib=X_calib, y_calib=y_calib)
    y_pred_cqr, y_lower_cqr, y_upper_cqr, sharpness_cqr, coverage_cqr = evaluate_cp(X_test, y_test_np, cqr, alpha)
    y_pred_cqr = 0.5 * (y_lower_cqr + y_upper_cqr)
    print(f"SplitCP - Average prediction intervals width (sharpness): {sharpness_sc:.3f}")
    print(f"SplitCP - Average coverage: {coverage_sc*100:.3f}%")
    print(f"CVPlus - Average prediction intervals width (sharpness): {sharpness_cv:.3f}")
    print(f"CVPlus - Average coverage: {coverage_cv*100:.3f}%")
    print(f"CQR - Average prediction intervals width (sharpness): {sharpness_cqr:.3f}")
    print(f"CQR - Average coverage: {coverage_cqr*100:.3f}%")


    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    sample_indices = np.arange(len(y_test_np))

    # Plot results for each method
    for idx, (y_pred, y_lower, y_upper, title) in enumerate([
        (y_pred_sc, y_lower_sc, y_upper_sc, 'SplitCP'),
        (y_pred_sc, y_lower_cv, y_upper_cv, 'CVPlus'),
        (y_pred_cqr, y_lower_cqr, y_upper_cqr, 'CQR')
    ]):
        axs[idx].scatter(sample_indices, y_test_np, label='True', color='blue', s=10, alpha=0.7)
        axs[idx].fill_between(sample_indices, y_lower, y_upper, color='gray', alpha=0.5, label='Prediction interval')
        axs[idx].plot(sample_indices, y_pred, color='red', label='Predicted mean', linewidth=1)
        axs[idx].set_title(f'{title}: Prediction Intervals with Conformal Prediction')
        axs[idx].set_xlabel('Sample Number')
        axs[idx].set_ylabel('str')
        axs[idx].legend(fontsize='small')

    plt.tight_layout()

    # --- Collect values for export, grouped by method ---
    results_dict = {
        'SplitCP_true': np.asarray(y_test).reshape(-1),
        'SplitCP_pred': np.asarray(y_pred_sc).reshape(-1),
        'SplitCP_lower': np.asarray(y_lower_sc).reshape(-1),
        'SplitCP_upper': np.asarray(y_upper_sc).reshape(-1),

        'CVPlus_true': np.asarray(y_test).reshape(-1),
        'CVPlus_pred': np.asarray(y_pred_sc).reshape(-1),
        'CVPlus_lower': np.asarray(y_lower_cv).reshape(-1),
        'CVPlus_upper': np.asarray(y_upper_cv).reshape(-1),

        'CQR_true': np.asarray(y_test).reshape(-1),
        'CQR_pred': np.asarray(y_pred_cqr).reshape(-1),
        'CQR_lower': np.asarray(y_lower_cqr).reshape(-1),
        'CQR_upper': np.asarray(y_upper_cqr).reshape(-1),
    }

    if excel_file_path:
        save_plot_to_excel(fig, excel_file_path, 'conformal_predictions_PUNCC')
        save_values_to_excel(results_dict, excel_file_path, 'conformal_predictions_PUNCC_values')

    plt.close(fig)

def save_plot_to_excel(fig, excel_file_path, sheet_name):
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image(buf)

        # Load the workbook and add a new sheet
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        if sheet_name in workbook.sheetnames:
            base_name = sheet_name
            i = 1
            while sheet_name in workbook.sheetnames:
                sheet_name = f"{base_name}_{i}"
                i += 1
        worksheet = workbook.create_sheet(title=sheet_name)

        # Add the image to the worksheet
        worksheet.add_image(img, 'A1')

        # Save the workbook
        _ensure_parent_dir(excel_file_path)
        workbook.save(_ensure_parent_dir(excel_file_path))

def save_values_to_excel(results_dict, excel_file_path, sheet_name):
    # Convert the dictionary to a DataFrame (columns will be in the order of insertion)
    df = pd.DataFrame(results_dict)
    # Load the workbook
    workbook = load_workbook(_ensure_excel_file(excel_file_path))
    # Ensure unique sheet name
    base_name = sheet_name
    i = 1
    while sheet_name in workbook.sheetnames:
        sheet_name = f"{base_name}_{i}"
        i += 1
    # Write DataFrame to a new sheet
    worksheet = workbook.create_sheet(title=sheet_name)
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            worksheet.cell(row=r_idx, column=c_idx, value=value)
    # Save the workbook
    _ensure_parent_dir(excel_file_path)
    workbook.save(_ensure_parent_dir(excel_file_path))

# Example usage (assuming X_train, y_train, X_test, y_test, best_scores_autosampler exist):
conformal_predictions_PUNCC_PGBM(
    X_train, y_train,
    X_test, y_test,
    best_scores_autosampler=best_scores_autosampler,
    model_class=PGBMWrapper,
    excel_file_path="./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/PGBM.xlsx",
    alpha=0.1
)

```

```python
best_params = get_best_model_params(best_scores_autosampler, 'PGBM')
prediction_MAPIE_analysis(X_train, y_train, X_test, y_test, PGBMWrapper, best_params,"./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/PGBM.xlsx", "PGBM Prediction Intervals")
```

# **Conformal Predictions with TabNet**

```python
best_params = get_best_model_params(best_scores_autosampler, 'TabNet')
print(best_params)
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
class TabNetRegressorCP(TabNetRegressor):
    def fit(self, X, y, *args, **kwargs):
        y_ = np.array(y)
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)
        return super().fit(X, y_, *args, **kwargs)
    def predict(self, X, *args, **kwargs):
        preds = super().predict(X, *args, **kwargs)
        return preds.flatten()
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

def conformal_predictions_MAPIE_tabnet(
    model_class, best_params, X_train, y_train, X_test, y_test, model_name, excel_file_path
):
    is_tabnet = ("TabNet" in model_name or "TabNet" in model_class.__name__)

    # Remove 'verbose' if present to avoid TypeError in TabNet
    if isinstance(best_params, dict) and "verbose" in best_params:
        best_params = dict(best_params) # make a copy!
        best_params.pop("verbose", None)

    # Set random seed for reproducibility
    SEED: int = 123
    np.random.seed(SEED)

    MISCOVERAGE = 0.1
    warnings.filterwarnings("ignore")
    os.makedirs('output/regression', exist_ok=True)
    strategies_names = {
        'SCP': 'Split Conformal Prediction',
        'CV+': 'Cross Validation +',
        'J+aB': 'Jackknife+ after Bootstrapping',
    }
    strategies_params = {
        'SCP': {'method': 'base', 'cv': KFold(n_splits=5)},
        'CV+': {'method': "plus", 'cv': 5},
        'J+aB': {'method': "minmax", 'cv': Subsample(n_resamplings=50)},
    }

    base_estimators = {
        strat_name: model_class(**best_params) for strat_name in strategies_names
    }
    y_pred, int_pred = {}, {}
    y_pred_train, int_pred_train = {}, {}

    for strat_name, base_estimator in base_estimators.items():
        mapie = MapieRegressor(base_estimator, **strategies_params[strat_name])
        mapie.fit(np.array(X_train), np.array(y_train))
        y_pred_raw, int_pred_raw = mapie.predict(np.array(X_test), alpha=MISCOVERAGE)
        y_pred[strat_name] = y_pred_raw.flatten()
        int_pred[strat_name] = np.array(int_pred_raw)
        y_pred_train_raw, int_pred_train_raw = mapie.predict(np.array(X_train), alpha=MISCOVERAGE)
        y_pred_train[strat_name] = y_pred_train_raw.flatten()
        int_pred_train[strat_name] = np.array(int_pred_train_raw)

    # All metric calls now use np.array(y_test)
    coverages = validate.coverage(int_pred, np.array(y_test))
    widths = validate.width(int_pred)
    rmse_vals = validate.rmse(y_pred, np.array(y_test))
    cwc_vals = validate.cwc(int_pred, np.array(y_test), MISCOVERAGE)

    for strat_name in strategies_names.keys():
        if strat_name in y_pred:
            rmse = rmse_vals[strat_name]
            interval_width = widths[strat_name]
            coverage = coverages[strat_name]
            cwc = cwc_vals[strat_name]
            ssc = np.mean((np.array(y_test) - y_pred[strat_name])**2 +
                          (int_pred[strat_name][:, 1] - int_pred[strat_name][:, 0])**2)
            print(f"Strategy: {strategies_names[strat_name]}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Interval Width: {interval_width:.4f}")
            print(f"  Coverage: {coverage * 100:.2f}%")
            print(f"  CWC: {cwc:.4f}")
            print(f"  SSC: {ssc:.4f}\n")

    _strategies = [_s for _s in strategies_names.keys() if _s != 'SCP']
    NUM_BINS = 10
    n_figs = len(strategies_names) * 3 + 1 + len(_strategies)
    fig, axs = plt.subplots(nrows=n_figs, figsize=(10, 5 * n_figs))
    plot_index = 0

    # Individual strategy plots
    for strat_name in strategies_names.keys():
        if strat_name in y_pred:
            _y_pred_low = int_pred[strat_name][:, 0].ravel()
            _y_pred_up = int_pred[strat_name][:, 1].ravel()
            sample_indices = np.arange(len(X_test))
            axs[plot_index].scatter(sample_indices, np.array(y_test), label='Test data', color='blue', s=10, alpha=0.7)
            axs[plot_index].fill_between(sample_indices, _y_pred_low, _y_pred_up, color='gray', alpha=0.5, label='Prediction interval')
            axs[plot_index].plot(sample_indices, y_pred[strat_name], color='red', label='Predicted mean', linewidth=1)
            axs[plot_index].set_title(f'{strategies_names[strat_name]} ({model_name})')
            axs[plot_index].set_xlabel('Sample Number')
            axs[plot_index].set_ylabel('str')
            axs[plot_index].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plot_index += 1

    # Combined plot
    comparison_ax = axs[plot_index]
    sample_indices = np.arange(len(X_test))
    comparison_ax.scatter(sample_indices, np.array(y_test), label='Test data', color='blue', s=10, alpha=0.7)
    for strat_name in y_pred:
        _y_pred_low = int_pred[strat_name][:, 0].ravel()
        _y_pred_up = int_pred[strat_name][:, 1].ravel()
        comparison_ax.fill_between(sample_indices, _y_pred_low, _y_pred_up, alpha=0.3, label=f'{strat_name} interval')
        comparison_ax.plot(sample_indices, y_pred[strat_name], label=f'{strat_name} mean', linewidth=1)
    comparison_ax.set_title(f'Comparison of Strategies ({model_name})')
    comparison_ax.set_xlabel('Sample Number')
    comparison_ax.set_ylabel('str')
    comparison_ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plot_index += 1

    # Goodness plots
    for strat_name in strategies_names.keys():
        if strat_name in y_pred:
            _y_pred_low_goodness = int_pred[strat_name][:, 0].ravel()
            _y_pred_up_goodness = int_pred[strat_name][:, 1].ravel()
            axs[plot_index] = visualize.goodness(
                np.array(y_test), y_pred[strat_name],
                _y_pred_low_goodness,
                _y_pred_up_goodness,
                coverages[strat_name],
                widths[strat_name],
                rmse_vals[strat_name],
                cwc_vals[strat_name],
                ax=axs[plot_index],
                title=f'{strategies_names[strat_name]} Goodness ({model_name})',
            )
            axs[plot_index].set_ylabel('Predicted str')
            plot_index += 1

    # Width Occurrence plots
    _x_max = (1+1e-3) * np.max([np.abs(int_pred[strat_name][:, 0] - int_pred[strat_name][:, 1]) for strat_name in strategies_names.keys() if strat_name in int_pred])
    _x_min = np.min([np.abs(int_pred[strat_name][:, 0] - int_pred[strat_name][:, 1]) for strat_name in strategies_names.keys() if strat_name in int_pred])
    for strat_name in strategies_names.keys():
        if strat_name in y_pred:
            axs[plot_index] = visualize.width_size_occurrence(
                int_pred[strat_name],
                train_intervals=int_pred_train[strat_name],
                num_bins=10,
                ax=axs[plot_index],
                x_lim=[_x_min, _x_max],
                title=f'{strategies_names[strat_name]} Width Occurrence',
            )
            axs[plot_index].set_ylabel('Frequency')
            plot_index += 1

    # Coverage vs Width plots
    cond_coverages = validate.cond_coverage(int_pred, np.array(y_test), num_bins=NUM_BINS)
    for _strat in _strategies:
        axs[plot_index] = visualize.coverage_by_width(
            np.array(y_test), int_pred[_strat], MISCOVERAGE,
            cond_coverages[_strat],
            num_bins=NUM_BINS,
            ax=axs[plot_index],
            title=f'{strategies_names[_strat]} Coverage vs Width',
        )
        plot_index += 1

    plt.tight_layout()

    # Save plots and tabular data to Excel
    with io.BytesIO() as buf:
        buf
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image(PImage.open(buf))
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        plot_sheet_name = f'conformal_predictions_MAPIE_{model_name}'
        if plot_sheet_name in workbook.sheetnames:
            plot_sheet_name += '_new'
        plot_worksheet = workbook.create_sheet(title=plot_sheet_name)
        plot_worksheet.add_image(img, 'A1')

        # Tabular export
        all_rows = []
        y_true_np = np.array(y_test).ravel()
        for strat_name in y_pred:
            y_pred_np = y_pred[strat_name].ravel()
            ymin_np = int_pred[strat_name][:, 0].ravel()
            ymax_np = int_pred[strat_name][:, 1].ravel()
            width_np = ymax_np - ymin_np
            is_covered_np = (y_true_np >= ymin_np) & (y_true_np <= ymax_np)
            residual_np = y_true_np - y_pred_np
            strat_df = pd.DataFrame({
                'sample_index': np.arange(len(y_pred_np)),
                'strategy': strat_name,
                'y_true': y_true_np,
                'y_pred': y_pred_np,
                'ymin': ymin_np,
                'ymax': ymax_np,
                'width': width_np,
                'is_covered': is_covered_np.astype(int),
                'residual': residual_np
            })
            all_rows.append(strat_df)
        all_export_df = pd.concat(all_rows, axis=0, ignore_index=True)
        all_pred_sheet_name = 'all_pred_values_' + model_name
        base_name = all_pred_sheet_name
        idx = 1
        while all_pred_sheet_name in workbook.sheetnames:
            all_pred_sheet_name = f'{base_name}_new{idx}'
            idx += 1
        worksheet = workbook.create_sheet(title=all_pred_sheet_name)
        for j, col_name in enumerate(all_export_df.columns, 1):
            worksheet.cell(row=1, column=j, value=col_name)
        for i, row in all_export_df.iterrows():
            for j, value in enumerate(row, 1):
                worksheet.cell(row=i+2, column=j, value=value)
        _ensure_parent_dir(excel_file_path)
        workbook.save(_ensure_parent_dir(excel_file_path))

    plt.close()


conformal_predictions_MAPIE_tabnet(
    TabNetRegressorCP,        # <------ use the wrapper here!
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "TabNet",
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/TabNet.xlsx"
)
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

def conformal_predictions_PUNCC_tabnet(
    X_train, y_train, X_test, y_test, best_scores_autosampler,
    excel_file_path=None, model_params=None, alpha=0.1
):
    # Get best parameters if not supplied
    if model_params is None:
        model_params = get_best_model_params(best_scores_autosampler, 'TabNet')
        if "verbose" in model_params:
            model_params = dict(model_params)
            model_params.pop("verbose", None)

    model_class = TabNetRegressorCP

    try:
        model = model_class(**model_params)
    except TypeError as e:
        print(f"Error initializing TabNet: {e}")
        return

    model.fit(X_train, y_train)

    def evaluate_cp(X_te, y_te, model_cp, alpha):
        y_pred, y_pred_lower, y_pred_upper = model_cp.predict(X_te, alpha=alpha)
        sharpness = regression_sharpness(y_pred_lower, y_pred_upper)
        coverage = regression_mean_coverage(np.array(y_te), y_pred_lower, y_pred_upper)
        return sharpness, coverage

    # Base predictor
    base_predictor = BasePredictor(model, is_trained=True)

    # SplitCP
    splitcp = SplitCP(base_predictor, train=True, random_state=0)
    splitcp.fit(X=X_train, y=y_train, fit_ratio=0.5)
    y_pred_splitcp, y_pred_lower_splitcp, y_pred_upper_splitcp = splitcp.predict(X_test, alpha=alpha)
    sharpness, coverage = evaluate_cp(X_test, y_test, splitcp, alpha)
    picp_splitcp = np.mean((np.array(y_test).ravel() >= y_pred_lower_splitcp) &
                           (np.array(y_test).ravel() <= y_pred_upper_splitcp))
    print(f"SplitCP - Average prediction intervals width (sharpness): {sharpness:.3f}")
    print(f"SplitCP - Average coverage: {coverage*100:.3f}%")
    print(f"SplitCP - PICP: {picp_splitcp*100:.3f}%")

    # CVPlus
    cvplus = CVPlus(base_predictor, K=5, random_state=0)
    cvplus.fit(X=X_train, y=y_train)
    y_pred_cvplus, y_pred_lower_cvplus, y_pred_upper_cvplus = cvplus.predict(X_test, alpha=alpha)
    sharpness_cvplus, coverage_cvplus = evaluate_cp(X_test, y_test, cvplus, alpha)
    picp_cvplus = np.mean((np.array(y_test).ravel() >= y_pred_lower_cvplus) &
                          (np.array(y_test).ravel() <= y_pred_upper_cvplus))
    print(f"CVPlus - Average prediction intervals width (sharpness): {sharpness_cvplus:.3f}")
    print(f"CVPlus - Average coverage: {coverage_cvplus*100:.3f}%")
    print(f"CVPlus - PICP: {picp_cvplus*100:.3f}%\n")

    # CQR
    from sklearn.model_selection import train_test_split
    X_fit, X_calib, y_fit, y_calib = train_test_split(X_train, y_train, test_size=0.5, random_state=0)
    upper_quantile_model = model_class(**model_params)
    lower_quantile_model = model_class(**model_params)
    upper_quantile_model.fit(X_fit, y_fit)
    lower_quantile_model.fit(X_fit, y_fit)
    dualpredictor = DualPredictor([lower_quantile_model, upper_quantile_model], is_trained=[True, True])
    cqr = CQR(dualpredictor, train=False)
    cqr.fit(X_calib=X_calib, y_calib=y_calib)
    y_pred_cqr, y_pred_lower_cqr, y_pred_upper_cqr = cqr.predict(X_test, alpha=alpha)
    y_pred_cqr = 0.5 * (y_pred_lower_cqr + y_pred_upper_cqr)
    sharpness_cqr, coverage_cqr = evaluate_cp(X_test, y_test, cqr, alpha)
    picp_cqr = np.mean((np.array(y_test).ravel() >= y_pred_lower_cqr) &
                       (np.array(y_test).ravel() <= y_pred_upper_cqr))
    print(f"CQR - Average prediction intervals width (sharpness): {sharpness_cqr:.3f}")
    print(f"CQR - Average coverage: {coverage_cqr*100:.3f}%")
    print(f"CQR - PICP: {picp_cqr*100:.3f}%\n")

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    arr_y_test = np.array(y_test).ravel()
    axs[0].scatter(np.arange(len(arr_y_test)), arr_y_test, label='True', color='blue', s=10, alpha=0.7)
    axs[0].fill_between(np.arange(len(arr_y_test)), y_pred_lower_splitcp, y_pred_upper_splitcp, color='gray', alpha=0.5, label='Prediction interval')
    axs[0].plot(np.arange(len(arr_y_test)), y_pred_splitcp, color='red', label='Predicted mean', linewidth=1)
    axs[0].set_title('Split Conformal Prediction')
    axs[0].set_xlabel('Sample Number')
    axs[0].set_ylabel('str')
    axs[0].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    axs[1].scatter(np.arange(len(arr_y_test)), arr_y_test, label='True', color='blue', s=10, alpha=0.7)
    axs[1].fill_between(np.arange(len(arr_y_test)), y_pred_lower_cvplus, y_pred_upper_cvplus, color='gray', alpha=0.5, label='Prediction interval')
    axs[1].plot(np.arange(len(arr_y_test)), y_pred_splitcp, color='red', label='Predicted mean', linewidth=1)
    axs[1].set_title('Cross Validation Plus')
    axs[1].set_xlabel('Sample Number ')
    axs[1].set_ylabel('str')
    axs[1].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    axs[2].scatter(np.arange(len(arr_y_test)), arr_y_test, label='True', color='blue', s=10, alpha=0.7)
    axs[2].fill_between(np.arange(len(arr_y_test)), y_pred_lower_cqr, y_pred_upper_cqr, color='gray', alpha=0.5, label='Prediction interval')
    axs[2].plot(np.arange(len(arr_y_test)), y_pred_cqr, color='red', label='Predicted mean', linewidth=1)
    axs[2].set_title('Conformalized Quantile Regression')
    axs[2].set_xlabel('Sample Number')
    axs[2].set_ylabel('str')
    axs[2].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()

    # --- Collect values for export ---
    results_dict = {
        'SplitCP_true': arr_y_test.reshape(-1),
        'SplitCP_pred': np.asarray(y_pred_splitcp).reshape(-1),
        'SplitCP_lower': np.asarray(y_pred_lower_splitcp).reshape(-1),
        'SplitCP_upper': np.asarray(y_pred_upper_splitcp).reshape(-1),

        'CVPlus_true': arr_y_test.reshape(-1),
        'CVPlus_pred': np.asarray(y_pred_splitcp).reshape(-1),
        'CVPlus_lower': np.asarray(y_pred_lower_cvplus).reshape(-1),
        'CVPlus_upper': np.asarray(y_pred_upper_cvplus).reshape(-1),

        'CQR_true': arr_y_test.reshape(-1),
        'CQR_pred': np.asarray(y_pred_cqr).reshape(-1),
        'CQR_lower': np.asarray(y_pred_lower_cqr).reshape(-1),
        'CQR_upper': np.asarray(y_pred_upper_cqr).reshape(-1),
    }

    if excel_file_path:
        save_plot_to_excel(fig, excel_file_path, 'conformal_predictions_PUNCC_TabNet')
        save_values_to_excel(results_dict, excel_file_path, 'conformal_predictions_PUNCC_TabNet_values')

    plt.close(fig)


conformal_predictions_PUNCC_tabnet(
    X_train, y_train, X_test, y_test, best_scores_autosampler,
    excel_file_path="./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/TabNet.xlsx",
    model_params=best_params  # your best params for TabNet
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
def prediction_MAPIE_analysis_tabnet(
    X_train,
    y_train,
    X_test,
    y_test,
    model_params,
    excel_file_path=None,
    suptitle: str = "TabNet Prediction Intervals"
) -> None:
    # Use wrapper for TabNet
    model_cls = TabNetRegressorCP

    # Train model with MAPIE
    mdl = model_cls(**model_params)
    mapie = MapieRegressor(mdl, method="plus", cv=KFold(n_splits=5, shuffle=True))
    mapie.fit(X_train, y_train)

    alpha = np.arange(0.05, 1, 0.05)
    y_train_pred, y_train_pis = mapie.predict(X_train, alpha=alpha)
    y_test_pred, y_test_pis = mapie.predict(X_test, alpha=alpha)

    def plot_predictionintervals(
        y_train,
        y_train_pred,
        y_train_pred_low,
        y_train_pred_high,
        y_test,
        y_test_pred,
        y_test_pred_low,
        y_test_pred_high,
        suptitle: str,
    ) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        ax1.errorbar(
            x=y_train,
            y=y_train_pred,
            yerr=(np.abs(y_train_pred - y_train_pred_low), np.abs(y_train_pred_high - y_train_pred)),
            alpha=0.8,
            label="train",
            fmt=".",
        )
        ax1.errorbar(
            x=y_test,
            y=y_test_pred,
            yerr=(np.abs(y_test_pred - y_test_pred_low), np.abs(y_test_pred_high - y_test_pred)),
            alpha=0.8,
            label="test",
            fmt=".",
        )
        ax1.plot(
            [np.array(y_train).min(), np.array(y_train).max()],
            [np.array(y_train).min(), np.array(y_train).max()],
            color="gray",
            alpha=0.5,
        )
        ax1.set_xlabel("Actual str", fontsize=12)
        ax1.set_ylabel("Predicted str", fontsize=12)
        ax1.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        ax2.scatter(
            x=y_train, y=y_train_pred_high - y_train_pred_low, alpha=0.8, label="train", marker="."
        )
        ax2.scatter(x=y_test, y=y_test_pred_high - y_test_pred_low, alpha=0.8, label="test", marker=".")
        ax2.set_xlabel("Actual str", fontsize=12)
        ax2.set_ylabel("Interval width", fontsize=12)
        ax2.set_xscale("linear")
        ax2.set_ylim([0, np.max(y_test_pred_high - y_test_pred_low)*1.1])
        ax2.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        std_all = np.concatenate([
            y_train_pred_high - y_train_pred_low, y_test_pred_high - y_test_pred_low
        ])
        type_all = np.array(["train"] * len(y_train) + ["test"] * len(y_test))
        x_all = np.arange(len(std_all))
        order_all = np.argsort(std_all)
        std_order = std_all[order_all]
        type_order = type_all[order_all]
        ax3.scatter(
            x=x_all[type_order == "train"],
            y=std_order[type_order == "train"],
            alpha=0.8,
            label="train",
            marker=".",
        )
        ax3.scatter(
            x=x_all[type_order == "test"],
            y=std_order[type_order == "test"],
            alpha=0.8,
            label="test",
            marker=".",
        )
        ax3.set_xlabel("Order", fontsize=12)
        ax3.set_ylabel("Interval width", fontsize=12)
        ax3.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax1.set_title("Actual vs Predicted str")
        ax2.set_title("Prediction interval width vs Actual str")
        ax3.set_title("Ordered prediction interval width")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle(suptitle, size=20)

        # Save plot to Excel
        if excel_file_path:
            save_plot_to_excel(fig, excel_file_path, 'prediction_MAPIE_analysis_tabnet_1')

        plt.close(fig)

    alpha_plot = int(np.where(alpha == 0.1)[0])
    plot_predictionintervals(
        np.array(y_train).ravel(),
        y_train_pred,
        y_train_pis[:, 0, alpha_plot],
        y_train_pis[:, 1, alpha_plot],
        np.array(y_test).ravel(),
        y_test_pred,
        y_test_pis[:, 0, alpha_plot],
        y_test_pis[:, 1, alpha_plot],
        suptitle,
    )

    # Comparison of the uncertainty quantification methods
    Params = TypedDict("Params", {"method": str, "cv": Union[int, Subsample]})
    STRATEGIES = {
        "naive": Params(method="naive"),
        "cv": Params(method="base", cv=5),
        "cv_plus": Params(method="plus", cv=5),
        "cv_minmax": Params(method="minmax", cv=5),
        "jackknife_plus_ab": Params(method="plus", cv=Subsample(n_resamplings=20)),
    }
    y_pred, y_pis, scores = {}, {}, {}
    for strategy, params in STRATEGIES.items():
        # Use a fresh model each time (sklearn style)
        mapie = MapieRegressor(model_cls(**model_params), **params)
        mapie.fit(X_train, y_train)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=alpha)
        scores[strategy] = [
            regression_coverage_score(y_test, y_pis[strategy][:, 0, i], y_pis[strategy][:, 1, i])
            for i, _ in enumerate(alpha)
        ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Target coverage (1 - alpha)")
    ax.set_ylabel("Effective coverage")
    for strategy, params in STRATEGIES.items():
        ax.plot(1 - alpha, scores[strategy], label=strategy)
    plt.subplots_adjust(right=0.75)
    ax.plot([0, 1], [0, 1], ls="--", color="k")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', borderaxespad=0.)

    # Save plot to Excel
    if excel_file_path:
        save_plot_to_excel(fig, excel_file_path, 'prediction_MAPIE_analysis_tabnet_2')

    plt.close(fig)

# Helper method for Excel plotting export
def save_plot_to_excel(fig, excel_file_path, sheet_name):
    import io
    from openpyxl import load_workbook
    from openpyxl.drawing.image import Image
    from PIL import Image as PILImage
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image(PILImage.open(buf))
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        if sheet_name in workbook.sheetnames:
            sheet_name += '_new'
        worksheet = workbook.create_sheet(title=sheet_name)
        worksheet.add_image(img, 'A1')
        _ensure_parent_dir(excel_file_path)
        workbook.save(_ensure_parent_dir(excel_file_path))


prediction_MAPIE_analysis_tabnet(
    X_train,
    y_train,
    X_test,
    y_test,
    model_params=best_params,
    excel_file_path="./drive/MyDrive/Regression/concrete_uncertainty_analysis/conformal_predictions_mapie_puncc/TabNet.xlsx",
    suptitle="TabNet Prediction Intervals"
)
```

