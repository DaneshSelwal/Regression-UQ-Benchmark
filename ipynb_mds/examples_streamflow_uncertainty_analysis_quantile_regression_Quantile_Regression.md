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
import ngboost
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
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
from pytorch_tabnet import TabNetRegressor
```

```python
# Go to find & replace button and replace (streamflow_uncertainty_analysis) with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (str) with actual data label name.
```

```python
feature_names = ['pH', 'Salinity', 'Turbidity ', 'Water Temperature']
```

```python
train_data_path = "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/data/test.csv"
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
def quantile_regression_print(predictions_df, file_path):
    # Extract actual values
    actual_values = predictions_df['Actual'].values

    # Debugging: Print the first few actual values
    print("First few actual values:", actual_values[:5])

    # Calculate coverage for the specific interval (0.05 to 0.95)
    lower_quantile = 0.05
    median_quantile = 0.5
    upper_quantile = 0.95

    lower_preds = predictions_df[lower_quantile].values
    median_preds = predictions_df[median_quantile].values
    upper_preds = predictions_df[upper_quantile].values

    in_interval = ((actual_values >= lower_preds) & (actual_values <= upper_preds))
    coverage = in_interval.mean()
    print(f"Coverage of 90% prediction interval: {coverage * 100:.2f}%")

    # Create a new Excel workbook and add a worksheet
    wb = Workbook()
    ws = wb.active
    ws.title = "quantile_regression_print"

    # Plot actual vs. all predicted quantiles
    num_points = len(predictions_df)
    indices = np.arange(num_points)

    plt.figure(figsize=(12, 6))
    plt.plot(indices, actual_values[:num_points], label='Actual str', marker='o', linestyle='-', color='black')

    # Loop over all columns assumed to be quantile predictions
    for column in predictions_df.columns:
        if column != 'Actual':  # Skip the actual values column
            plt.plot(indices, predictions_df[column][:num_points],
                     label=f'Predicted Quantile {column}', linestyle='--')

    plt.xlabel('Sample Number')
    plt.ylabel('str')
    plt.title('Actual vs. Predicted Quantile str')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Save the plot to a BytesIO object
    img_data = BytesIO()
    img_data
    plt.savefig(img_data, format='png', bbox_inches='tight')
    plt.close()
    img_data.seek(0)

    # Insert the image into the Excel sheet
    img = Image(img_data)
    ws.add_image(img, 'A1')

    # Plot actual values and specific prediction intervals (0.05, 0.5, 0.95)
    plt.figure(figsize=(12, 6))
    plt.plot(indices, actual_values, label='Actual str', marker='o', linestyle='-', color='black')  # Connect points with a line
    plt.plot(indices, median_preds, label=f'Median Prediction ({median_quantile})', marker='x', linestyle='-', color='blue')
    plt.fill_between(
        indices,
        lower_preds,
        upper_preds,
        color='blue',
        alpha=0.5,
        label=f'Prediction Interval ({lower_quantile}-{upper_quantile})'
    )
    plt.xlabel('Sample Number')
    plt.ylabel(' str')
    plt.title('Actual str with Prediction Intervals')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1, 1))

    # Save the second plot to another BytesIO object
    img_data = BytesIO()
    img_data
    plt.savefig(img_data, format='png', bbox_inches='tight')
    plt.close()
    img_data.seek(0)

    # Insert the second image into the Excel sheet
    img = Image(img_data)
    ws.add_image(img, 'A30')  # Adjust the cell position as needed

    # Save the workbook
    _ensure_parent_dir(file_path)
    wb.save(_ensure_parent_dir(file_path))
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def evaluation_features_print(predictions_df, X_test, file_path):
    # Ensure X_test is a DataFrame
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    X_test.columns = feature_names

    # Get quantile columns from predictions_df except 'Actual'
    quantile_columns = [col for col in predictions_df.columns if col != 'Actual']

    # Check if the Excel file already exists
    if os.path.exists(file_path):
        wb = load_workbook(_ensure_excel_file(file_path))
        ws = wb.create_sheet(title="New Evaluation Features")
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "evaluation_features_print"

    image_row = 1  # Where to start images

    for feature_to_plot in feature_names:
        feature_values = X_test[feature_to_plot]
        plot_df = predictions_df.copy()
        plot_df[feature_to_plot] = feature_values.values

        # Sort by current feature to plot lines well
        plot_df = plot_df.sort_values(by=feature_to_plot)

        # Begin Matplotlib plot
        plt.figure(figsize=(6, 4))
        # Plot each quantile as a line
        for quantile in quantile_columns:
            plt.plot(plot_df[feature_to_plot], plot_df[quantile], label=f"{quantile}")

        # Plot actual values as scatter
        plt.scatter(plot_df[feature_to_plot], plot_df['Actual'], color='black', s=12, label="Actual str")

        plt.title(f"Predicted Quantiles vs. {feature_to_plot}")
        plt.xlabel(feature_to_plot)
        plt.ylabel("str")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()

        # Save figure to a BytesIO stream
        img_bytes = BytesIO()
        img_bytes
        plt.savefig(img_bytes, format="png")
        plt.close()
        img_bytes.seek(0)

        img = Image(img_bytes)
        img.anchor = f"A{image_row}"
        ws.add_image(img)

        image_row += 20  # Adjust as you like

    _ensure_parent_dir(file_path)
    wb.save(_ensure_parent_dir(file_path))
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def evaluate_uncertainity_matrix(predictions_df, quantiles, excel_file_path, model_name='Model'):

   # Extract the actual target values
    y_true = predictions_df['Actual']

    # Open existing Excel file
    wb = load_workbook(_ensure_excel_file(excel_file_path))

    # Create unique sheet names
    def get_unique_sheet_name(wb, base_name):
        sheet_name = base_name
        i = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{base_name}_{i}"
            i += 1
        return sheet_name

    # Create separate sheets for plots and values
    plots_sheet_title = get_unique_sheet_name(wb, 'Plots')
    values_sheet_title = get_unique_sheet_name(wb, 'Values')

    ws_plots = wb.create_sheet(title=plots_sheet_title)
    ws_values = wb.create_sheet(title=values_sheet_title)

    # Positions to place images
    image_positions = ['A1', 'A25', 'A49', 'A73']  # Adjust as needed

    ### 1. Validity / Coverage ###

    # Compute empirical coverage for each quantile
    empirical_quantiles = []
    for q in quantiles:
        coverage = (predictions_df[q] >= y_true).mean()
        empirical_quantiles.append(coverage)

    # Plot theoretical quantiles vs empirical quantiles
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=quantiles, y=quantiles, color="magenta", linestyle='--', linewidth=2, label="Ideal")
    sns.lineplot(x=quantiles, y=empirical_quantiles, color="blue", linewidth=2, label="Empirical")
    sns.scatterplot(x=quantiles, y=empirical_quantiles, color="blue", s=50)
    plt.xlabel("Nominal Quantile Levels")
    plt.ylabel("Empirical Quantile Levels")
    plt.title(f"Validity / Coverage - {model_name}")
    plt.legend()

    # Save plot to buffer
    buffer = io.BytesIO()
    buffer
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Create an Image object
    img = openpyxlImage(buffer)
    img.width = 640
    img.height = 480

    # Add the image to the plots worksheet
    ws_plots.add_image(img, image_positions[0])

    ### 2. Sharpness / Interval Length ###

    # Define coverage levels to evaluate
    coverage_levels = [0.6, 0.8, 0.9]
    average_interval_lengths = []

    # Compute average interval lengths for each coverage level
    for c in coverage_levels:
        lower_q = (1 - c) / 2
        upper_q = 1 - lower_q
        # Find the nearest quantiles in our quantiles list
        lower_quantile = min(quantiles, key=lambda x: abs(x - lower_q))
        upper_quantile = min(quantiles, key=lambda x: abs(x - upper_q))

        # Compute interval length for each observation
        interval_length = predictions_df[upper_quantile] - predictions_df[lower_quantile]
        avg_interval_length = interval_length.mean()
        average_interval_lengths.append(avg_interval_length)
        # Write interval length information to the values worksheet
        ws_values.append([f'Coverage level: {c*100:.0f}%', f'Interval: [{lower_quantile}, {upper_quantile}]', f'Average Interval Length: {avg_interval_length:.4f}'])

    # Plot coverage levels vs average interval lengths
    plt.figure(figsize=(10, 6))
    plt.plot([c*100 for c in coverage_levels], average_interval_lengths, marker='o')
    plt.xlabel('Coverage Level (%)')
    plt.ylabel('Average Interval Length')
    plt.title(f'Sharpness / Interval Length - {model_name}')

    # Save plot to buffer
    buffer = io.BytesIO()
    buffer
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Create an Image object
    img = openpyxlImage(buffer)
    img.width = 640
    img.height = 480

    # Add image to the plots worksheet
    ws_plots.add_image(img, image_positions[1])

    ### 3. Negative Log-Likelihood (NLL) ###

    # Use the 5th and 95th percentiles to estimate standard deviation
    if 0.05 in quantiles and 0.95 in quantiles:
        z_lower = norm.ppf(0.05)  # z-score for 5% quantile (~ -1.6449)
        z_upper = norm.ppf(0.95)  # z-score for 95% quantile (~ +1.6449)

        # Extract quantile predictions
        q_lower = predictions_df[0.05]
        q_upper = predictions_df[0.95]

    elif 0.1 in quantiles and 0.9 in quantiles:
        z_lower = norm.ppf(0.1)  # z-score for 10% quantile (~ -1.2816)
        z_upper = norm.ppf(0.9)  # z-score for 90% quantile (~ +1.2816)

        # Extract quantile predictions
        q_lower = predictions_df[0.1]
        q_upper = predictions_df[0.9]
    else:
        raise ValueError("Required quantiles for NLL estimation not found in quantiles list.")

    # Use median as the mean estimate
    if 0.5 in quantiles:
        q_median = predictions_df[0.5]
    else:
        # If 0.5 quantile is not available, use the middle quantile
        q_median = predictions_df[quantiles[len(quantiles)/2]]

    # Estimate standard deviation for each observation
    std_estimates = (q_upper - q_lower) / (z_upper - z_lower)

    # Ensure standard deviations are positive and non-zero
    std_estimates = std_estimates.clip(lower=1e-6)

    # Extract mean estimates (median predictions)
    mean_estimates = q_median

    # Compute Negative Log-Likelihood for each observation
    nll = -norm.logpdf(y_true, loc=mean_estimates, scale=std_estimates)

    # Compute average NLL
    average_nll = nll.mean()
    # Write average NLL to the values worksheet
    ws_values.append(['Average Negative Log-Likelihood (NLL)', average_nll])

    # Write NLL values to the values worksheet
    ws_values.append(['NLL Values'])
    for value in nll:
        ws_values.append([value])

    # Plot ECDF of NLL values
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(nll, color='blue', linewidth=2)
    plt.xlabel('Negative Log-Likelihood (NLL)')
    plt.ylabel('ECDF')
    plt.title(f'NLL ECDF - {model_name}')

    # Save plot to buffer
    buffer = io.BytesIO()
    buffer
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Create an Image object
    img = openpyxlImage(buffer)
    img.width = 640
    img.height = 480

    # Add image to the plots worksheet
    ws_plots.add_image(img, image_positions[2])

    ### 4. Continuous Ranked Probability Score (CRPS) ###

    # Prepare ensemble of quantile predictions for each observation
    ensemble_predictions = predictions_df[quantiles].to_numpy()

    # Ensure quantile predictions are sorted for each observation
    ensemble_predictions.sort(axis=1)

    # Compute CRPS for each observation using ensemble predictions
    crps_values = ps.crps_ensemble(y_true, ensemble_predictions)

    # Compute average CRPS
    average_crps = crps_values.mean()
    # Write average CRPS to the values worksheet
    ws_values.append(['Average Continuous Ranked Probability Score (CRPS)', average_crps])

    # Write CRPS values to the values worksheet
    ws_values.append(['CRPS Values'])
    for value in crps_values:
        ws_values.append([value])

    # Plot ECDF of CRPS values
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(crps_values, color='green', linewidth=2)
    plt.xlabel('Continuous Ranked Probability Score (CRPS)')
    plt.ylabel('ECDF')
    plt.title(f'CRPS ECDF - {model_name}')

    # Save plot to buffer
    buffer = io.BytesIO()
    buffer
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Create an Image object
    img = openpyxlImage(buffer)
    img.width = 640
    img.height = 480

    # Add image to the plots worksheet
    ws_plots.add_image(img, image_positions[3])

    # Save the workbook
    _ensure_parent_dir(excel_file_path)
    wb.save(_ensure_parent_dir(excel_file_path))

    # Return a dictionary of results
    results = {
        'empirical_quantiles': empirical_quantiles,
        'average_interval_lengths': average_interval_lengths,
        'coverage_levels': coverage_levels,
        'average_nll': average_nll,
        'average_crps': average_crps,
        'nll_values': nll,
        'crps_values': crps_values
    }

    return results
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def calibrate_and_plot_intervals(predictions_df, excel_file_path, alpha=0.1, max_samples=1000, random_seed=42):
    # Extract necessary columns
    lower = predictions_df[0.05].fillna(0)
    upper = predictions_df[0.95].fillna(0)
    pred = predictions_df[0.5].fillna(0)
    labels = predictions_df['Actual'].fillna(0)

    # Problem setup
    total_samples = labels.shape[0]
    n = min(max_samples, total_samples)  # Ensure n does not exceed the number of samples

    # Ensure there are enough samples for both calibration and validation
    if total_samples <= n:
        n = total_samples // 2

    # Split the data into calibration and validation sets
    np.random.seed(random_seed)  # Set the random seed for reproducibility
    idx = np.array([1] * n + [0] * (total_samples - n)) > 0
    np.random.shuffle(idx)
    cal_labels, val_labels = labels[idx], labels[~idx]
    cal_upper, val_upper = upper[idx], upper[~idx]
    cal_lower, val_lower = lower[idx], lower[~idx]
    cal_pred, val_pred = pred[idx], pred[~idx]

    # Calculate uncertainty intervals
    cal_U = cal_upper - cal_lower
    val_U = val_upper - val_lower

    # Avoid division by zero by replacing zero intervals with a small number
    cal_U[cal_U == 0] = np.finfo(float).eps
    val_U[val_U == 0] = np.finfo(float).eps

    # Get scores
    cal_scores = np.abs(cal_pred - cal_labels) / cal_U

    # Debugging: Check for NaN in scores
    if np.isnan(cal_scores).any():
        print("NaN values found in cal_scores. Check data integrity.")

    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation='higher')

    # Deploy (output=lower and upper adjusted quantiles)
    prediction_sets = [val_pred - val_U * qhat, val_pred + val_U * qhat]

    # Calculate empirical coverage (before and after calibration)
    prediction_sets_uncalibrated = [val_lower, val_upper]
    empirical_coverage_uncalibrated = ((val_labels >= prediction_sets_uncalibrated[0]) & (val_labels <= prediction_sets_uncalibrated[1])).mean() * 100
    print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated:.2f}%")

    empirical_coverage = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean() * 100
    print(f"The empirical coverage after calibration is: {empirical_coverage:.2f}%")

    # Create a DataFrame with all the necessary data
    data = pd.DataFrame({
        'Index': np.arange(len(val_labels)),
        'Uncalibrated Lower': val_lower,
        'Uncalibrated Upper': val_upper,
        'Calibrated Lower': prediction_sets[0],
        'Calibrated Upper': prediction_sets[1],
        'True Label': val_labels,
        'Predicted Value': val_pred
    })

    plt.figure(figsize=(14, 8))

    # Plot the uncalibrated prediction intervals as a shaded area
    plt.fill_between(
        data['Index'],
        data['Uncalibrated Lower'],
        data['Uncalibrated Upper'],
        color='blue',
        alpha=0.5,
        label='Uncalibrated Prediction Interval'
    )

    # Plot the calibrated prediction intervals as a shaded area
    plt.fill_between(
        data['Index'],
        data['Calibrated Lower'],
        data['Calibrated Upper'],
        color='lightgreen',
        alpha=0.5,
        label='Calibrated Prediction Interval'
    )

    # Plot the true labels as points and connect them with a line
    plt.plot(
        data['Index'],
        data['True Label'],
        'o-',
        color='red',
        markersize=4,
        label='Actual Value'
    )

    # Plot the predicted values as points and connect them with a line
    plt.plot(
        data['Index'],
        data['Predicted Value'],
        'o-',
        color='black',
        markersize=4,
        label='Predicted Value'
    )

    plt.xlabel('Sample Number')
    plt.ylabel('str')
    plt.title('Calibrated and Uncalibrated Prediction Intervals with True and Predicted Values')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()

    # Save the plot as an image
    image_path = 'plot.png'
    _ensure_parent_dir(image_path)
    plt.savefig(_ensure_parent_dir(image_path))
    plt.close()

    # Load the existing Excel file
    workbook = load_workbook(_ensure_excel_file(excel_file_path))

    # Create a new sheet for the plot
    new_sheet_name = "calibrate_and_plot_intervals"
    workbook.create_sheet(title=new_sheet_name)

    # Insert the image into the new sheet
    sheet = workbook[new_sheet_name]
    img = Image(image_path)
    sheet.add_image(img, 'A1')

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
def plot_quantile_intervals_to_excel(predictions_df, file_path, alpha=0.1):
    """
    Plots conformal prediction intervals and their coverage, and saves the plots to an Excel file.

    Parameters:
    - predictions_df: DataFrame containing prediction quantiles and actual values.
    - file_path: Path to the Excel file where plots will be saved.
    - alpha: Desired coverage level (default is 0.1).
    """
    # Extract mean prediction and uncertainty
    mean_prediction = predictions_df[0.5]
    uncertainty = predictions_df[0.95] - predictions_df[0.05]
    actual_values = predictions_df['Actual']

    # Calculate conformal scores
    conformal_scores = np.abs(mean_prediction - actual_values) / uncertainty

    # Function to find weighted quantile
    def weighted_quantile(values, quantile, sample_weight=None):
        values = np.array(values)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sorter = np.argsort(values)
        values, sample_weight = values[sorter], sample_weight[sorter]
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantile, weighted_quantiles, values)

    # Calculate weighted quantile
    weights = np.ones_like(conformal_scores)  # Uniform weights for simplicity
    quantile = weighted_quantile(conformal_scores, 1 - alpha, sample_weight=weights)

    # Calculate prediction intervals
    lower_bound = mean_prediction - quantile * uncertainty
    upper_bound = mean_prediction + quantile * uncertainty

    # Naive conformal prediction
    naive_quantile = np.quantile(conformal_scores, 1 - alpha)
    naive_lower_bound = mean_prediction - naive_quantile * uncertainty
    naive_upper_bound = mean_prediction + naive_quantile * uncertainty

    # Coverage calculation (not plotted, just for potential diagnostics)
    weighted_coverage_points = (actual_values >= lower_bound) & (actual_values <= upper_bound)
    naive_coverage_points = (actual_values >= naive_lower_bound) & (actual_values <= naive_upper_bound)

    # Visualization using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(predictions_df))
    # Actual Values
    ax.plot(x, actual_values, label='Actual', color='black', linewidth=2)

    # Mean Prediction
    ax.plot(x, mean_prediction, label='Mean Prediction', color='darkorange', linewidth=2)

    # Weighted Prediction Interval (blue fill)
    ax.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.3, label='Weighted Interval')
    # Optionally, plot the bounds as lines if you want:
    # ax.plot(x, lower_bound, color='blue', linewidth=1, linestyle='--')
    # ax.plot(x, upper_bound, color='blue', linewidth=1, linestyle='--')

    # Naive Prediction Interval (red fill)
    ax.fill_between(x, naive_lower_bound, naive_upper_bound, color='red', alpha=0.3, label='Naive Interval')
    # Optionally, plot the bounds as lines:
    # ax.plot(x, naive_lower_bound, color='red', linewidth=1, linestyle='--')
    # ax.plot(x, naive_upper_bound, color='red', linewidth=1, linestyle='--')

    ax.set_title('Quantile Prediction Intervals')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('str')
    ax.grid(False)
    ax.set_facecolor('white')

    # Place the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0)

    plt.tight_layout(rect=[0,0,0.80,1.0])  # Make space for legend

    # Save the plot to a BytesIO object
    img_data = BytesIO()
    img_data
    plt.savefig(img_data, format='png', bbox_inches='tight')
    plt.close()
    img_data.seek(0)

    # Add image to Excel
    if os.path.exists(file_path):
        wb = load_workbook(_ensure_excel_file(file_path))
        ws = wb.create_sheet(title="plot_quantile_intervals_to_excel")
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "plot_quantile_intervals_to_excel"

    img = Image(img_data)
    img.anchor = 'A1'
    ws.add_image(img)
    ws.column_dimensions['A'].width = 30
    _ensure_parent_dir(file_path)
    wb.save(_ensure_parent_dir(file_path))
```

# **Tuned Parameters**

```python
best_scores_autosampler = {('Random Forest', 'MedianPruner'): {'best_score': 11396.714504124559,
  'best_params': {'n_estimators': 200,
   'criterion': 'poisson',
   'max_depth': 10,
   'min_samples_split': 2,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.2,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 11396.714504124559,
  'test_rmse': 106.75539566750038,
  'test_corr_coef': 0.8545175710230449,
  'pruner': 'MedianPruner'},
 ('Random Forest', 'NopPruner'): {'best_score': 11396.714504124559,
  'best_params': {'n_estimators': 200,
   'criterion': 'poisson',
   'max_depth': 10,
   'min_samples_split': 5,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.2,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.1},
  'test_mse': 11396.714504124559,
  'test_rmse': 106.75539566750038,
  'test_corr_coef': 0.8545175710230445,
  'pruner': 'NopPruner'},
 ('Random Forest', 'PatientPruner'): {'best_score': 12070.291743991513,
  'best_params': {'n_estimators': 200,
   'criterion': 'poisson',
   'max_depth': 20,
   'min_samples_split': 5,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.1},
  'test_mse': 12070.291743991513,
  'test_rmse': 109.86487948380734,
  'test_corr_coef': 0.8452553059844832,
  'pruner': 'PatientPruner'},
 ('Random Forest', 'PercentilePruner'): {'best_score': 11716.981138270823,
  'best_params': {'n_estimators': 100,
   'criterion': 'friedman_mse',
   'max_depth': 10,
   'min_samples_split': 2,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 11716.981138270823,
  'test_rmse': 108.24500514236591,
  'test_corr_coef': 0.8433099870855542,
  'pruner': 'PercentilePruner'},
 ('Random Forest',
  'SuccessiveHalvingPruner'): {'best_score': 11439.192756209346, 'best_params': {'n_estimators': 200,
   'criterion': 'poisson',
   'max_depth': 10,
   'min_samples_split': 0.01,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.05}, 'test_mse': 11439.192756209346, 'test_rmse': 106.9541619396335, 'test_corr_coef': 0.8533935563954336, 'pruner': 'SuccessiveHalvingPruner'},
 ('Random Forest', 'HyperbandPruner'): {'best_score': 11504.645347044134,
  'best_params': {'n_estimators': 500,
   'criterion': 'poisson',
   'max_depth': 10,
   'min_samples_split': 5,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.05},
  'test_mse': 11504.645347044134,
  'test_rmse': 107.2597098030949,
  'test_corr_coef': 0.8532357616223099,
  'pruner': 'HyperbandPruner'},
 ('Random Forest', 'ThresholdPruner'): {'best_score': 11439.192756209348,
  'best_params': {'n_estimators': 200,
   'criterion': 'poisson',
   'max_depth': 10,
   'min_samples_split': 2,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.1},
  'test_mse': 11439.192756209348,
  'test_rmse': 106.95416193963351,
  'test_corr_coef': 0.8533935563954336,
  'pruner': 'ThresholdPruner'},
 ('Random Forest', 'WilcoxonPruner'): {'best_score': 11734.35251996622,
  'best_params': {'n_estimators': 200,
   'criterion': 'poisson',
   'max_depth': 40,
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.05},
  'test_mse': 11734.35251996622,
  'test_rmse': 108.32521645473975,
  'test_corr_coef': 0.8529602600912333,
  'pruner': 'WilcoxonPruner'},
 ('Gradient Boosting', 'MedianPruner'): {'best_score': 7894.46813379009,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 500,
   'subsample': 1.0,
   'criterion': 'squared_error',
   'min_samples_split': 2,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 10,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.001,
   'ccp_alpha': 0.0},
  'test_mse': 7894.46813379009,
  'test_rmse': 88.85081954484207,
  'test_corr_coef': 0.8941590289814214,
  'pruner': 'MedianPruner'},
 ('Gradient Boosting', 'NopPruner'): {'best_score': 7979.404009165202,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 300,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 0.01,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 3,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 7979.404009165202,
  'test_rmse': 89.3275098117327,
  'test_corr_coef': 0.9028811706719992,
  'pruner': 'NopPruner'},
 ('Gradient Boosting', 'PatientPruner'): {'best_score': 8035.925747823539,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 300,
   'subsample': 1.0,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 7,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 8035.925747823539,
  'test_rmse': 89.64332517161297,
  'test_corr_coef': 0.8908089630271125,
  'pruner': 'PatientPruner'},
 ('Gradient Boosting', 'PercentilePruner'): {'best_score': 9421.813509142628,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.2,
   'n_estimators': 500,
   'subsample': 0.5,
   'criterion': 'squared_error',
   'min_samples_split': 0.01,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 7,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 9421.813509142628,
  'test_rmse': 97.06602654452601,
  'test_corr_coef': 0.8749406418115374,
  'pruner': 'PercentilePruner'},
 ('Gradient Boosting',
  'SuccessiveHalvingPruner'): {'best_score': 7832.37345464286, 'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 300,
   'subsample': 1.0,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.01}, 'test_mse': 7832.37345464286, 'test_rmse': 88.5006974811095, 'test_corr_coef': 0.8959547030435522, 'pruner': 'SuccessiveHalvingPruner'},
 ('Gradient Boosting', 'HyperbandPruner'): {'best_score': 8915.412611307433,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.01,
   'n_estimators': 500,
   'subsample': 0.5,
   'criterion': 'friedman_mse',
   'min_samples_split': 5,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.001,
   'ccp_alpha': 0.0},
  'test_mse': 8915.412611307433,
  'test_rmse': 94.4214626624023,
  'test_corr_coef': 0.8893245816512504,
  'pruner': 'HyperbandPruner'},
 ('Gradient Boosting', 'ThresholdPruner'): {'best_score': 7908.923722871204,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 200,
   'subsample': 1.0,
   'criterion': 'friedman_mse',
   'min_samples_split': 0.01,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 5,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 'log2',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.001,
   'ccp_alpha': 0.0},
  'test_mse': 7908.923722871204,
  'test_rmse': 88.93212986806964,
  'test_corr_coef': 0.8933287186955804,
  'pruner': 'ThresholdPruner'},
 ('Gradient Boosting', 'WilcoxonPruner'): {'best_score': 7826.507523618409,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 500,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 7,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 7826.507523618409,
  'test_rmse': 88.46755068169577,
  'test_corr_coef': 0.9025977077608344,
  'pruner': 'WilcoxonPruner'},
 ('XGBoost', 'MedianPruner'): {'best_score': 9592.75,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'max_depth': 7,
   'min_child_weight': 5,
   'gamma': 0.1,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 9592.75,
  'test_rmse': 97.94258522215962,
  'test_corr_coef': 0.8744618085762377,
  'pruner': 'MedianPruner'},
 ('XGBoost', 'NopPruner'): {'best_score': 10609.2060546875,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_depth': 3,
   'min_child_weight': 5,
   'gamma': 0,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 10609.2060546875,
  'test_rmse': 103.00100026061641,
  'test_corr_coef': 0.8661842817660108,
  'pruner': 'NopPruner'},
 ('XGBoost', 'PatientPruner'): {'best_score': 9704.4892578125,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_depth': 5,
   'min_child_weight': 5,
   'gamma': 0.5,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 9704.4892578125,
  'test_rmse': 98.51136613514453,
  'test_corr_coef': 0.8697166555588549,
  'pruner': 'PatientPruner'},
 ('XGBoost', 'PercentilePruner'): {'best_score': 10378.65234375,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_depth': 5,
   'min_child_weight': 1,
   'gamma': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 10378.65234375,
  'test_rmse': 101.8756710100601,
  'test_corr_coef': 0.8599670105990299,
  'pruner': 'PercentilePruner'},
 ('XGBoost', 'SuccessiveHalvingPruner'): {'best_score': 9919.0126953125,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'max_depth': 7,
   'min_child_weight': 5,
   'gamma': 0.5,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 9919.0126953125,
  'test_rmse': 99.59424027177727,
  'test_corr_coef': 0.8679753126698246,
  'pruner': 'SuccessiveHalvingPruner'},
 ('XGBoost', 'HyperbandPruner'): {'best_score': 9796.1591796875,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'max_depth': 3,
   'min_child_weight': 5,
   'gamma': 0,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 9796.1591796875,
  'test_rmse': 98.97554839296168,
  'test_corr_coef': 0.8695228483465185,
  'pruner': 'HyperbandPruner'},
 ('XGBoost', 'ThresholdPruner'): {'best_score': 9558.4912109375,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'max_depth': 3,
   'min_child_weight': 5,
   'gamma': 0.1,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 9558.4912109375,
  'test_rmse': 97.76753659030946,
  'test_corr_coef': 0.8742564436069653,
  'pruner': 'ThresholdPruner'},
 ('XGBoost', 'WilcoxonPruner'): {'best_score': 10723.3310546875,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'max_depth': 3,
   'min_child_weight': 5,
   'gamma': 0,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 10723.3310546875,
  'test_rmse': 103.55351782864501,
  'test_corr_coef': 0.8554622717330934,
  'pruner': 'WilcoxonPruner'},
 ('LightGBM', 'MedianPruner'): {'best_score': 8093.7407934217545,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 10,
   'subsample': 0.7,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 8093.7407934217545,
  'test_rmse': 89.96521990981712,
  'test_corr_coef': 0.8889794794195334,
  'pruner': 'MedianPruner'},
 ('LightGBM', 'NopPruner'): {'best_score': 9544.473558946216,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 63,
   'max_depth': 5,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'min_child_weight': 0.01,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 9544.473558946216,
  'test_rmse': 97.69582160433585,
  'test_corr_coef': 0.8763579708230304,
  'pruner': 'NopPruner'},
 ('LightGBM', 'PatientPruner'): {'best_score': 10096.33246815347,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'num_leaves': 31,
   'max_depth': 7,
   'min_child_samples': 5,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 1e-05,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10096.33246815347,
  'test_rmse': 100.4805079015501,
  'test_corr_coef': 0.8671432153117224,
  'pruner': 'PatientPruner'},
 ('LightGBM', 'PercentilePruner'): {'best_score': 7404.388873890984,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 63,
   'max_depth': 3,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.1,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7404.388873890984,
  'test_rmse': 86.04875870046577,
  'test_corr_coef': 0.9036610692100672,
  'pruner': 'PercentilePruner'},
 ('LightGBM', 'SuccessiveHalvingPruner'): {'best_score': 7826.370797911004,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 31,
   'max_depth': 5,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7826.370797911004,
  'test_rmse': 88.46677793336323,
  'test_corr_coef': 0.8968692934394386,
  'pruner': 'SuccessiveHalvingPruner'},
 ('LightGBM', 'HyperbandPruner'): {'best_score': 7716.182725214737,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': 7,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 1.0,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'min_child_weight': 0.001,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7716.182725214737,
  'test_rmse': 87.8418051113178,
  'test_corr_coef': 0.8996131442228372,
  'pruner': 'HyperbandPruner'},
 ('LightGBM', 'ThresholdPruner'): {'best_score': 7852.782654987631,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 10,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'min_child_weight': 0.01,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7852.782654987631,
  'test_rmse': 88.61592777253777,
  'test_corr_coef': 0.8942173174465775,
  'pruner': 'ThresholdPruner'},
 ('LightGBM', 'WilcoxonPruner'): {'best_score': 8390.306495329174,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 63,
   'max_depth': 7,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 1,
   'min_child_weight': 1e-05,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 8390.306495329174,
  'test_rmse': 91.59861623042771,
  'test_corr_coef': 0.8848324592834348,
  'pruner': 'WilcoxonPruner'},
 ('GPBoost', 'MedianPruner'): {'best_score': 7544.111602140408,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 5,
   'num_leaves': 15,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.5,
   'reg_lambda': 0,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7544.111602140408,
  'test_rmse': 86.85684545354158,
  'test_corr_coef': 0.899140689349327,
  'pruner': 'MedianPruner'},
 ('GPBoost', 'NopPruner'): {'best_score': 8298.997243869338,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': -1,
   'num_leaves': 15,
   'min_child_samples': 10,
   'subsample': 0.6,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.5,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 8298.997243869338,
  'test_rmse': 91.09883228598125,
  'test_corr_coef': 0.8859850624403404,
  'pruner': 'NopPruner'},
 ('GPBoost', 'PatientPruner'): {'best_score': 7404.2396920244355,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 10,
   'subsample': 0.7,
   'colsample_bytree': 0.5,
   'reg_alpha': 1.0,
   'reg_lambda': 0,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7404.2396920244355,
  'test_rmse': 86.0478918511339,
  'test_corr_coef': 0.90198786970229,
  'pruner': 'PatientPruner'},
 ('GPBoost', 'PercentilePruner'): {'best_score': 7421.776367870293,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 15,
   'min_child_samples': 10,
   'subsample': 0.7,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 1.0,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7421.776367870293,
  'test_rmse': 86.14973225652122,
  'test_corr_coef': 0.9027127769989374,
  'pruner': 'PercentilePruner'},
 ('GPBoost', 'SuccessiveHalvingPruner'): {'best_score': 7694.459345833749,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 10,
   'subsample': 0.6,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 1.0,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 7694.459345833749,
  'test_rmse': 87.7180673854238,
  'test_corr_coef': 0.8952650394354964,
  'pruner': 'SuccessiveHalvingPruner'},
 ('GPBoost', 'HyperbandPruner'): {'best_score': 8137.497471873475,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 15,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 1.0,
   'reg_alpha': 0,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 8137.497471873475,
  'test_rmse': 90.20807875059458,
  'test_corr_coef': 0.9025758228683337,
  'pruner': 'HyperbandPruner'},
 ('GPBoost', 'ThresholdPruner'): {'best_score': 8523.395691254658,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 10,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0.5,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 8523.395691254658,
  'test_rmse': 92.32223833537972,
  'test_corr_coef': 0.8844228113820186,
  'pruner': 'ThresholdPruner'},
 ('GPBoost', 'WilcoxonPruner'): {'best_score': 8901.75039358178,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': -1,
   'num_leaves': 15,
   'min_child_samples': 10,
   'subsample': 1.0,
   'colsample_bytree': 0.7,
   'reg_alpha': 1.0,
   'reg_lambda': 0.1,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 8901.75039358178,
  'test_rmse': 94.34908793190202,
  'test_corr_coef': 0.8769681275078564,
  'pruner': 'WilcoxonPruner'},
 ('CatBoost', 'MedianPruner'): {'best_score': 12059.717630039499,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 64,
   'min_data_in_leaf': 20,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12059.717630039499,
  'test_rmse': 109.81674567223115,
  'test_corr_coef': 0.834682576760058,
  'pruner': 'MedianPruner'},
 ('CatBoost', 'NopPruner'): {'best_score': 12200.843128318063,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 128,
   'min_data_in_leaf': 10,
   'rsm': 1.0,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12200.843128318063,
  'test_rmse': 110.45742676849784,
  'test_corr_coef': 0.828657462741331,
  'pruner': 'NopPruner'},
 ('CatBoost', 'PatientPruner'): {'best_score': 11828.352768700503,
  'best_params': {'iterations': 200,
   'learning_rate': 0.03,
   'depth': 4,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 1,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 11828.352768700503,
  'test_rmse': 108.75823080898522,
  'test_corr_coef': 0.8402340352771763,
  'pruner': 'PatientPruner'},
 ('CatBoost', 'PercentilePruner'): {'best_score': 12266.04121988882,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 4,
   'l2_leaf_reg': 3,
   'border_count': 64,
   'min_data_in_leaf': 20,
   'rsm': 1.0,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12266.04121988882,
  'test_rmse': 110.75216124251851,
  'test_corr_coef': 0.8284396615075661,
  'pruner': 'PercentilePruner'},
 ('CatBoost', 'SuccessiveHalvingPruner'): {'best_score': 12200.843128318063,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 128,
   'min_data_in_leaf': 1,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12200.843128318063,
  'test_rmse': 110.45742676849784,
  'test_corr_coef': 0.828657462741331,
  'pruner': 'SuccessiveHalvingPruner'},
 ('CatBoost', 'HyperbandPruner'): {'best_score': 12200.843128318063,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 128,
   'min_data_in_leaf': 10,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12200.843128318063,
  'test_rmse': 110.45742676849784,
  'test_corr_coef': 0.828657462741331,
  'pruner': 'HyperbandPruner'},
 ('CatBoost', 'ThresholdPruner'): {'best_score': 12295.893812610366,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.01,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 1,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12295.893812610366,
  'test_rmse': 110.88685139641385,
  'test_corr_coef': 0.8292402939039899,
  'pruner': 'ThresholdPruner'},
 ('CatBoost', 'WilcoxonPruner'): {'best_score': 12469.34371865179,
  'best_params': {'iterations': 500,
   'learning_rate': 0.03,
   'depth': 4,
   'l2_leaf_reg': 9,
   'border_count': 32,
   'min_data_in_leaf': 20,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12469.34371865179,
  'test_rmse': 111.66621565474398,
  'test_corr_coef': 0.8262490043890625,
  'pruner': 'WilcoxonPruner'},
 ('NGBoost', 'MedianPruner'): {'best_score': 12061.97957465404,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12061.97957465404,
  'test_rmse': 109.8270439129363,
  'test_corr_coef': 0.8313969676088081,
  'pruner': 'MedianPruner'},
 ('NGBoost', 'NopPruner'): {'best_score': 11887.17702290286,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 11887.17702290286,
  'test_rmse': 109.02833128550974,
  'test_corr_coef': 0.835197276506235,
  'pruner': 'NopPruner'},
 ('NGBoost', 'PatientPruner'): {'best_score': 12187.317404589345,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12187.317404589345,
  'test_rmse': 110.39618383164041,
  'test_corr_coef': 0.8296624201996675,
  'pruner': 'PatientPruner'},
 ('NGBoost', 'PercentilePruner'): {'best_score': 12090.956098491108,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12090.956098491108,
  'test_rmse': 109.9588836724487,
  'test_corr_coef': 0.8280868500336911,
  'pruner': 'PercentilePruner'},
 ('NGBoost', 'SuccessiveHalvingPruner'): {'best_score': 12186.9424450075,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12186.9424450075,
  'test_rmse': 110.39448557336323,
  'test_corr_coef': 0.8265686728872818,
  'pruner': 'SuccessiveHalvingPruner'},
 ('NGBoost', 'HyperbandPruner'): {'best_score': 12328.252116504349,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.5,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12328.252116504349,
  'test_rmse': 111.03266238591394,
  'test_corr_coef': 0.8267596939517394,
  'pruner': 'HyperbandPruner'},
 ('NGBoost', 'ThresholdPruner'): {'best_score': 12168.541471178041,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12168.541471178041,
  'test_rmse': 110.3111121835785,
  'test_corr_coef': 0.8303252321842222,
  'pruner': 'ThresholdPruner'},
 ('NGBoost', 'WilcoxonPruner'): {'best_score': 11952.222521790789,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 11952.222521790789,
  'test_rmse': 109.32622065081546,
  'test_corr_coef': 0.8343780376436447,
  'pruner': 'WilcoxonPruner'},
 ('TabNet', 'MedianPruner'): {'best_score': 10192.0947265625,
  'best_params': {'n_d': 16,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 1.3,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 10192.0947265625,
  'test_rmse': 100.95590486228382,
  'test_corr_coef': 0.8680742873891645,
  'pruner': 'MedianPruner'},
 ('TabNet', 'NopPruner'): {'best_score': 7993.10546875,
  'best_params': {'n_d': 16,
   'n_a': 8,
   'n_steps': 3,
   'gamma': 1.3,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 7993.10546875,
  'test_rmse': 89.40416919109533,
  'test_corr_coef': 0.9007085605894374,
  'pruner': 'NopPruner'},
 ('TabNet', 'PatientPruner'): {'best_score': 9981.9013671875,
  'best_params': {'n_d': 64,
   'n_a': 16,
   'n_steps': 5,
   'gamma': 1.3,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 9981.9013671875,
  'test_rmse': 99.90946585377934,
  'test_corr_coef': 0.8959780812686795,
  'pruner': 'PatientPruner'},
 ('TabNet', 'PercentilePruner'): {'best_score': 9387.9111328125,
  'best_params': {'n_d': 16,
   'n_a': 8,
   'n_steps': 5,
   'gamma': 1.5,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 9387.9111328125,
  'test_rmse': 96.89123351889221,
  'test_corr_coef': 0.8799101544458534,
  'pruner': 'PercentilePruner'},
 ('TabNet', 'SuccessiveHalvingPruner'): {'best_score': 10727.46875,
  'best_params': {'n_d': 32,
   'n_a': 32,
   'n_steps': 7,
   'gamma': 1.5,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 10727.46875,
  'test_rmse': 103.5734944375249,
  'test_corr_coef': 0.8761904698139649,
  'pruner': 'SuccessiveHalvingPruner'},
 ('TabNet', 'HyperbandPruner'): {'best_score': 7730.98291015625,
  'best_params': {'n_d': 16,
   'n_a': 64,
   'n_steps': 7,
   'gamma': 2.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 7730.98291015625,
  'test_rmse': 87.92600815547269,
  'test_corr_coef': 0.8954141726546125,
  'pruner': 'HyperbandPruner'},
 ('TabNet', 'ThresholdPruner'): {'best_score': 9506.19921875,
  'best_params': {'n_d': 64,
   'n_a': 16,
   'n_steps': 7,
   'gamma': 1.5,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 3,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 9506.19921875,
  'test_rmse': 97.49973958298555,
  'test_corr_coef': 0.8727852275919926,
  'pruner': 'ThresholdPruner'},
 ('TabNet', 'WilcoxonPruner'): {'best_score': 9029.6494140625,
  'best_params': {'n_d': 32,
   'n_a': 8,
   'n_steps': 7,
   'gamma': 1.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 2,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 9029.6494140625,
  'test_rmse': 95.02446744950744,
  'test_corr_coef': 0.8804388946225445,
  'pruner': 'WilcoxonPruner'},
 ('HistGradientBoosting', 'MedianPruner'): {'best_score': 8849.916118658877,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 500,
   'max_depth': 5,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 63,
   'l2_regularization': 1.0,
   'max_bins': 255,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 8849.916118658877,
  'test_rmse': 94.07399278577941,
  'test_corr_coef': 0.8792088280317188,
  'pruner': 'MedianPruner'},
 ('HistGradientBoosting', 'NopPruner'): {'best_score': 13912.001349439612,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 300,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 31,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 13912.001349439612,
  'test_rmse': 117.94914730272369,
  'test_corr_coef': 0.8016637520482625,
  'pruner': 'NopPruner'},
 ('HistGradientBoosting', 'PatientPruner'): {'best_score': 9029.10721281066,
  'best_params': {'learning_rate': 0.05,
   'max_iter': 100,
   'max_depth': 7,
   'min_samples_leaf': 10,
   'max_leaf_nodes': None,
   'l2_regularization': 0.5,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 9029.10721281066,
  'test_rmse': 95.02161445066412,
  'test_corr_coef': 0.8773438467284207,
  'pruner': 'PatientPruner'},
 ('HistGradientBoosting',
  'PercentilePruner'): {'best_score': 8317.046438364536, 'best_params': {'learning_rate': 0.01,
   'max_iter': 300,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 63,
   'l2_regularization': 1.0,
   'max_bins': 255,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 8317.046438364536, 'test_rmse': 91.1978422900703, 'test_corr_coef': 0.886267177095813, 'pruner': 'PercentilePruner'},
 ('HistGradientBoosting',
  'SuccessiveHalvingPruner'): {'best_score': 8062.298200739737, 'best_params': {'learning_rate': 0.01,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 63,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 8062.298200739737, 'test_rmse': 89.7903012621059, 'test_corr_coef': 0.9036784928536276, 'pruner': 'SuccessiveHalvingPruner'},
 ('HistGradientBoosting', 'HyperbandPruner'): {'best_score': 8062.298200739737,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 31,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 8062.298200739737,
  'test_rmse': 89.7903012621059,
  'test_corr_coef': 0.9036784928536276,
  'pruner': 'HyperbandPruner'},
 ('HistGradientBoosting', 'ThresholdPruner'): {'best_score': 8104.540855275137,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 31,
   'l2_regularization': 0.1,
   'max_bins': 128,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 8104.540855275137,
  'test_rmse': 90.02522343918474,
  'test_corr_coef': 0.9033273747525989,
  'pruner': 'ThresholdPruner'},
 ('HistGradientBoosting', 'WilcoxonPruner'): {'best_score': 8349.441232137877,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 300,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.5,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 8349.441232137877,
  'test_rmse': 91.37527691962349,
  'test_corr_coef': 0.8854457951837483,
  'pruner': 'WilcoxonPruner'},
 ('PGBM', 'MedianPruner'): {'best_score': 8845.042530599296,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'max_leaves': 32,
   'min_split_gain': 0.0,
   'reg_lambda': 10.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 5,
   'max_bin': 64,
   'distribution': 'laplace'},
  'test_mse': 8845.042530599296,
  'test_rmse': 94.04808626760725,
  'test_corr_coef': 0.8777931131479602,
  'pruner': 'MedianPruner'},
 ('PGBM', 'NopPruner'): {'best_score': 9175.70378659374,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_leaves': 44,
   'min_split_gain': 1.0,
   'reg_lambda': 5.0,
   'feature_fraction': 0.5,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 5,
   'max_bin': 128,
   'distribution': 'studentt'},
  'test_mse': 9175.70378659374,
  'test_rmse': 95.78989396900771,
  'test_corr_coef': 0.8755252327260262,
  'pruner': 'NopPruner'},
 ('PGBM', 'PatientPruner'): {'best_score': 8644.407437002892,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_leaves': 33,
   'min_split_gain': 1.0,
   'reg_lambda': 5.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 10,
   'max_bin': 128,
   'distribution': 'laplace'},
  'test_mse': 8644.407437002892,
  'test_rmse': 92.97530552250362,
  'test_corr_coef': 0.8825926843420658,
  'pruner': 'PatientPruner'},
 ('PGBM', 'PercentilePruner'): {'best_score': 8490.129698451294,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_leaves': 42,
   'min_split_gain': 0.0,
   'reg_lambda': 10.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 5,
   'max_bin': 128,
   'distribution': 'normal'},
  'test_mse': 8490.129698451294,
  'test_rmse': 92.14189979836152,
  'test_corr_coef': 0.8829833457176368,
  'pruner': 'PercentilePruner'},
 ('PGBM', 'SuccessiveHalvingPruner'): {'best_score': 7773.717983434995,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_leaves': 24,
   'min_split_gain': 0.5,
   'reg_lambda': 1.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 10,
   'max_bin': 128,
   'distribution': 'laplace'},
  'test_mse': 7773.717983434995,
  'test_rmse': 88.16869049404667,
  'test_corr_coef': 0.8954213036678549,
  'pruner': 'SuccessiveHalvingPruner'},
 ('PGBM', 'HyperbandPruner'): {'best_score': 7763.821071455185,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_leaves': 31,
   'min_split_gain': 0.1,
   'reg_lambda': 10.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 5,
   'max_bin': 256,
   'distribution': 'studentt'},
  'test_mse': 7763.821071455185,
  'test_rmse': 88.11254775260551,
  'test_corr_coef': 0.8964381399574919,
  'pruner': 'HyperbandPruner'},
 ('PGBM', 'ThresholdPruner'): {'best_score': 7663.838249707027,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 36,
   'min_split_gain': 1.0,
   'reg_lambda': 5.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 5,
   'max_bin': 256,
   'distribution': 'normal'},
  'test_mse': 7663.838249707027,
  'test_rmse': 87.54335068814208,
  'test_corr_coef': 0.9023151129811368,
  'pruner': 'ThresholdPruner'},
 ('PGBM', 'WilcoxonPruner'): {'best_score': 7710.988354850527,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_leaves': 28,
   'min_split_gain': 0.0,
   'reg_lambda': 10.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 5,
   'max_bin': 128,
   'distribution': 'laplace'},
  'test_mse': 7710.988354850527,
  'test_rmse': 87.81223351475879,
  'test_corr_coef': 0.9001164496423207,
  'pruner': 'WilcoxonPruner'}}
```

```python
best_scores_autosampler
```

# **Quantile Regression with Lightgbm**

```python
best_params = get_best_model_params(best_scores_autosampler, 'LightGBM')

# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# Dictionary to store models for each quantile
models_quantile = {}

# DataFrame to store predictions
predictions_LGBM_df = pd.DataFrame()


# Train a model for each quantile and make predictions using LightGBM
for q in quantiles:
    # Create a copy of best_params to avoid modifying the original
    params = best_params.copy()

    # Update params with quantile-specific settings
    params.update({
        'objective': 'quantile',
        'alpha': q,
        'random_state': 42
    })

    # Initialize the model with the updated parameters
    model = LGBMRegressor(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Store the model
    models_quantile[q] = model

    # Make predictions
    predictions = model.predict(X_test)

    # Add predictions to the DataFrame
    predictions_LGBM_df[q] = predictions

# Add actual target values to the DataFrame
predictions_LGBM_df['Actual'] = y_test.values.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_LGBM_df.head())
```

```python
quantile_regression_print(predictions_LGBM_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/LightGBM.xlsx")
```

```python
evaluation_features_print(predictions_LGBM_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/LightGBM.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_LGBM_df, quantiles, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/LightGBM.xlsx", model_name='LightGBM')
```

```python
calibrate_and_plot_intervals(predictions_LGBM_df, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/LightGBM.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_LGBM_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/LightGBM.xlsx")
```

# **Quantile Regression with XGBoost**

```python
# Get the best hyperparameters for XGBRegressor
best_paramst = get_best_model_params(best_scores_autosampler, 'XGBoost')

# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# Dictionary to store models for each quantile
models_quantile = {}

# DataFrame to store predictions
predictions_XGB_df = pd.DataFrame()

# Train a model for each quantile and make predictions
for q in quantiles:
    # Create a copy of best_params to avoid modifying the original
    params = best_params.copy()

    # Adjust according to your XGBoost version
    params.update({
        'objective': 'reg:quantileerror',  # Use the correct objective
        'quantile_alpha': q,                         # Use 'alpha' to set the quantile level
        'random_state': 42
    })

    # Initialize the model with the updated parameters
    model = XGBRegressor(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Store the model
    models_quantile[q] = model

    # Make predictions
    predictions = model.predict(X_test)

    # Add predictions to the DataFrame
    predictions_XGB_df[q] = predictions

# Add actual target values to the DataFrame
predictions_XGB_df['Actual'] = y_test.values.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_XGB_df.head())
```

```python
quantile_regression_print(predictions_XGB_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/XGBoost.xlsx")
```

```python
evaluation_features_print(predictions_XGB_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/XGBoost.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_XGB_df, quantiles, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/XGBoost.xlsx", model_name='XGBoost')
```

```python
calibrate_and_plot_intervals(predictions_XGB_df, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/XGBoost.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_XGB_df, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/XGBoost.xlsx")
```

# **Quantile Regression with GPBoost**

```python
# Get the best hyperparameters for GPBoostRegressor
best_params = get_best_model_params(best_scores_autosampler, 'GPBoost')

# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# Dictionary to store models for each quantile
models_quantile = {}

# DataFrame to store predictions
predictions_GPBoost_df = pd.DataFrame()

# Parameters that may conflict with quantile-specific settings
incompatible_keys = ['objective', 'alpha', 'quantile_alpha', 'boosting_type', 'metric', 'eval_metric']

# Train a model for each quantile and make predictions
for q in quantiles:
    # Create a copy of best_params to avoid modifying the original
    params = best_params.copy()

    # Remove incompatible parameters
    for key in incompatible_keys:
        params.pop(key, None)

    # Update params with quantile-specific settings
    params.update({
        'objective': 'quantile',
        'alpha': q,
        'random_state': 42
    })

    # Initialize the model with the updated parameters
    model = GPBoostRegressor(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Store the model
    models_quantile[q] = model

    # Make predictions
    predictions = model.predict(X_test)

    # Add predictions to the DataFrame
    predictions_GPBoost_df[q] = predictions

# Add actual target values to the DataFrame
predictions_GPBoost_df['Actual'] = y_test.values.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_GPBoost_df.head())
```

```python
quantile_regression_print(predictions_GPBoost_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/GPBoost.xlsx")
```

```python
evaluation_features_print(predictions_GPBoost_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/GPBoost.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_GPBoost_df, quantiles,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/GPBoost.xlsx", model_name='GPBoost')
```

```python
calibrate_and_plot_intervals(predictions_GPBoost_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/GPBoost.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_GPBoost_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/GPBoost.xlsx")
```

# **Quantile Regression with NGBoost**

```python
# Correct best_params assignment
best_params = get_best_model_params(best_scores_autosampler, 'NGBoost')

# DataFrame to store predictions
predictions_NGB_df = pd.DataFrame()

# Remove 'Score' and 'Dist' if present in best_params
for key in ['Score', 'Dist']:
    best_params.pop(key, None)

# Update params with model settings
best_params.update({
    'Dist': Normal,
    'Score': LogScore,
    'random_state': 42,
    'verbose': True
})

# Initialize and fit the NGBoost model once
model = NGBRegressor(**best_params)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
pred_dist = model.pred_dist(X_test)

# Extract the mean (mu) and standard deviation (sigma) of the predicted distributions
mu = pred_dist.loc    # Mean predictions
sigma = pred_dist.scale  # Standard deviation predictions

# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# For each quantile, compute predictions
for q in quantiles:
    # Calculate the z-score for the desired quantile
    z = norm.ppf(q)

    # Compute the quantile prediction per sample
    predictions = mu + z * sigma

    # Add predictions to the DataFrame
    predictions_NGB_df[q] = predictions

# Add actual target values to the DataFrame
predictions_NGB_df['Actual'] = y_test.values.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_NGB_df.head())
```

```python
quantile_regression_print(predictions_NGB_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/NGBoost.xlsx")
```

```python
evaluation_features_print(predictions_NGB_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/NGBoost.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_NGB_df, quantiles,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/NGBoost.xlsx", model_name='NGBoost')
```

```python
calibrate_and_plot_intervals(predictions_NGB_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/NGBoost.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_NGB_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/NGBoost.xlsx")
```

# **Quantile Regression with Gradient Boosting**

```python
best_params = get_best_model_params(best_scores_autosampler, 'GradientBoosting')


# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# Dictionary to store models for each quantile
models_quantile = {}

# DataFrame to store predictions
predictions_GBR_df = pd.DataFrame()

# Custom quantile loss function
def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))

# Train a model for each quantile and make predictions using GradientBoostingRegressor
for q in quantiles:

    params = best_params.copy()

    params.update({
        'loss': 'quantile',
        'alpha': q,
        'random_state': 42
    })
    # Initialize the model with quantile-specific settings
    model = GradientBoostingRegressor(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Store the model
    models_quantile[q] = model

    # Make predictions
    predictions = model.predict(X_test)

    # Add predictions to the DataFrame
    predictions_GBR_df[q] = predictions

# Add actual target values to the DataFrame
predictions_GBR_df['Actual'] = y_test.values.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_GBR_df.head())
```

```python
quantile_regression_print(predictions_GBR_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/Gradient Boosting.xlsx")
```

```python
evaluation_features_print(predictions_GBR_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/Gradient Boosting.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_GBR_df, quantiles,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/Gradient Boosting.xlsx", model_name='GBM')
```

```python
calibrate_and_plot_intervals(predictions_GBR_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/Gradient Boosting.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_GBR_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/Gradient Boosting.xlsx")
```

# **Quantile Regression with CatBoost**

```python
# Get the best hyperparameters for CatBoostRegressor
best_params = get_best_model_params(best_scores_autosampler, 'CatBoost')

# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# Dictionary to store models for each quantile
models_quantile = {}

# DataFrame to store predictions
predictions_CatBoost_df = pd.DataFrame()

# Parameters that may conflict with quantile-specific settings
incompatible_keys = ['objective', 'loss_function', 'eval_metric']

# Train a model for each quantile and make predictions
for q in quantiles:
    # Create a copy of best_params to avoid modifying the original
    params = best_params.copy()

    # Remove incompatible parameters
    for key in incompatible_keys:
        params.pop(key, None)

    # Update params with quantile-specific settings
    params.update({
        'loss_function': 'Quantile:alpha={}'.format(q),
        'random_seed': 42
    })

    # Initialize the model with the updated parameters
    model = CatBoostRegressor(**params)

    # Fit the model
    model.fit(X_train, y_train, verbose=0)

    # Store the model
    models_quantile[q] = model

    # Make predictions
    predictions = model.predict(X_test)

    # Add predictions to the DataFrame
    predictions_CatBoost_df[q] = predictions

# Add actual target values to the DataFrame
predictions_CatBoost_df['Actual'] = y_test.values.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_CatBoost_df.head())
```

```python
quantile_regression_print(predictions_CatBoost_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/CatBoost.xlsx")
```

```python
evaluation_features_print(predictions_CatBoost_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/CatBoost.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_CatBoost_df, quantiles,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/CatBoost.xlsx", model_name='CatBoost')
```

```python
calibrate_and_plot_intervals(predictions_CatBoost_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/CatBoost.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_CatBoost_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/CatBoost.xlsx")
```

# **Quantile Regression with HistGradientBoosting**

```python

# Assume best_params is a dictionary containing the best parameters for HistGradientBoostingRegressor
best_params = get_best_model_params(best_scores_autosampler, 'HGBR')

# Define the quantiles you want to predict
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# Dictionary to store models for each quantile
models_quantile = {}

# DataFrame to store predictions
predictions_HGBR_df = pd.DataFrame()

# Train a model for each quantile and make predictions using HistGradientBoostingRegressor
for q in quantiles:
    # Create a copy of best_params to avoid modifying the original
    params = best_params.copy()

    # Update params with quantile-specific settings
    params.update({
        'loss': 'quantile',
        'quantile': q,
        'random_state': 42
    })

    # Initialize the model with the updated parameters
    model = HistGradientBoostingRegressor(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Store the model
    models_quantile[q] = model

    # Make predictions
    predictions = model.predict(X_test)

    # Add predictions to the DataFrame
    predictions_HGBR_df[q] = predictions

# Add actual target values to the DataFrame
predictions_HGBR_df['Actual'] = y_test.values.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_HGBR_df.head())
```

```python
quantile_regression_print(predictions_HGBR_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/HGBM.xlsx")
```

```python
evaluation_features_print(predictions_HGBR_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/HGBM.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_HGBR_df, quantiles,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/HGBM.xlsx", model_name='HGBM')
```

```python
calibrate_and_plot_intervals(predictions_HGBR_df, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/HGBM.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_HGBR_df, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/HGBM.xlsx")
```

# **Quantile Regression with TabNet**

```python
best_params = get_best_model_params(best_scores_autosampler, 'TabNet')

# Quantiles
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8,
             0.85, 0.9, 0.95]

# Storage
models_quantile = {}
predictions_TabNet_df = pd.DataFrame()

# Convert data to numpy (TabNet requirement)
X_train_np = np.array(X_train)
y_train_np = np.array(y_train).reshape(-1, 1)
X_test_np  = np.array(X_test)

# Quantile loss (pinball loss)
def quantile_loss(q):
    def loss(y_pred, y_true):
        error = y_true - y_pred
        return torch.mean(torch.maximum(q * error, (q - 1) * error))
    return loss

# Train one TabNet per quantile
for q in quantiles:
    model = TabNetRegressor(
        **best_params,
    )

    model.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_train_np, y_train_np)],
        loss_fn=quantile_loss(q),
        max_epochs=200,
        patience=30,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    # Store model
    models_quantile[q] = model

    # Predict
    preds = model.predict(X_test_np).ravel()
    predictions_TabNet_df[q] = preds

# Add actual values
predictions_TabNet_df['Actual'] = y_test.values.ravel()

print(predictions_TabNet_df.head())
```

```python
quantile_regression_print(predictions_TabNet_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/TabNet.xlsx")
```

```python
evaluation_features_print(predictions_TabNet_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/TabNet.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_TabNet_df, quantiles, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/TabNet.xlsx", model_name='TabNet')
```

```python
calibrate_and_plot_intervals(predictions_TabNet_df, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/TabNet.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_TabNet_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/TabNet.xlsx")
```

# **Quantile Regression with PGBM**

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
quantile_regression_print(predictions_PGBM_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/PGBM.xlsx")
```

```python
evaluation_features_print(predictions_PGBM_df,x_test,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/PGBM.xlsx")
```

```python
evaluate_uncertainity_matrix(predictions_PGBM_df, quantiles, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/PGBM.xlsx", model_name='PGBM')
```

```python
calibrate_and_plot_intervals(predictions_PGBM_df, "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/PGBM.xlsx")
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

plot_quantile_intervals_to_excel(predictions_PGBM_df,"./drive/MyDrive/Regression/streamflow_uncertainty_analysis/quantile_regression/PGBM.xlsx")
```

