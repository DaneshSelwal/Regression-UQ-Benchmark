# **Start Section:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!pip install git+https://github.com/DaneshSelwal/treeffuser.git
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
from pathlib import Path
from scipy.stats import norm, kstest
from pytorch_tabnet import TabNetRegressor
from treeffuser import Treeffuser, Samples
from treeffuser._score_models import TreeBasedScoreModel
import torch
import torch.nn as nn
import torch.optim as optim
```

```python
# Go to find & replace button and replace (pile_uncertainty_analysis) with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (str) with actual data label name.
```

```python
feature_names = ['F1', 'F2', 'Zv1', 'Zv2', 'phi', 'Th', 'L']
```

```python
train_data_path = "./drive/MyDrive/Regression/pile_uncertainty_analysis/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/pile_uncertainty_analysis/data/test.csv"
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
def generate_and_save_plots(predictions_df, y_test, excel_path):
    # 1. Predicted vs. Actual Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Actual', y='Mean', data=predictions_df)
    plt.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
             [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
             color='red', linestyle='--')
    plt.title('Predicted str vs. Actual str')
    plt.xlabel('Actual str')
    plt.ylabel('Predicted Mean str')
    plt.tight_layout()
    predicted_vs_actual_path = 'predicted_vs_actual.png'
    _ensure_parent_dir(predicted_vs_actual_path)
    plt.savefig(_ensure_parent_dir(predicted_vs_actual_path))
    plt.close()

    # 2. Uncertainty Bands
    plt.figure(figsize=(12, 8))
    plt.errorbar(x=range(len(predictions_df)), y=predictions_df['Mean'],
                 yerr=predictions_df['StdDev'], fmt='o', markersize=4, ecolor='red', elinewidth=1, capsize=5, alpha=0.7)
    plt.plot(range(len(predictions_df)), predictions_df['Mean'], color='black', linewidth=0.5, alpha=0.7)
    plt.title('Predicted Mean with Uncertainty Bands', fontsize=16)
    plt.xlabel('Sample Number', fontsize=14)
    plt.ylabel('Predicted Mean str')
    plt.tight_layout()
    uncertainty_bands_path = 'uncertainty_bands.png'
    _ensure_parent_dir(uncertainty_bands_path)
    plt.savefig(_ensure_parent_dir(uncertainty_bands_path))
    plt.close()

    # 3. Histogram of Standard Deviations
    plt.figure(figsize=(12, 8))
    sns.histplot(predictions_df['StdDev'], bins=20, kde=True)
    plt.title('Distribution of Predicted Standard Deviations')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.tight_layout()
    histogram_stddev_path = 'histogram_stddev.png'
    _ensure_parent_dir(histogram_stddev_path)
    plt.savefig(_ensure_parent_dir(histogram_stddev_path))
    plt.close()

    # 4. Residual Plot
    residuals = predictions_df['Actual'] - predictions_df['Mean']
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=predictions_df['Mean'], y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Mean str')
    plt.ylabel('Residuals')
    plt.tight_layout()
    residual_plot_path = 'residual_plot.png'
    _ensure_parent_dir(residual_plot_path)
    plt.savefig(_ensure_parent_dir(residual_plot_path))
    plt.close()

    # Plot the standard deviation of the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(
        predictions_df['StdDev'],
        label='Standard Deviation',
        color='green',
        marker='o',
        linestyle='dashed'
    )
    plt.xlabel('Sample Number')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation of Predictions')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    stddev_plot_path = 'stddev_plot.png'
    _ensure_parent_dir(stddev_plot_path)
    plt.savefig(_ensure_parent_dir(stddev_plot_path))
    plt.close()

    # Plot the actual vs predicted mean with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(
        y_test,
        label='Actual str',
        color='black',
        marker='o',
        linestyle='dashed',
        linewidth=1,
        markersize=4
    )
    plt.plot(
        predictions_df['Mean'],
        label='Predicted Mean str',
        color='red',
        marker='x',
        linestyle='dashed',
        linewidth=1,
        markersize=4
    )
    confidence_interval = 1.96 * predictions_df['StdDev']
    plt.fill_between(
        np.arange(len(predictions_df)),
        predictions_df['Mean'] - confidence_interval,
        predictions_df['Mean'] + confidence_interval,
        color='blue',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    plt.xlabel('Sample Number')
    plt.ylabel(' str')
    plt.title('Predictions with Confidence Intervals')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    prediction_plot_path = 'prediction_plot.png'
    _ensure_parent_dir(prediction_plot_path)
    plt.savefig(_ensure_parent_dir(prediction_plot_path))
    plt.close()

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(_ensure_parent_dir(excel_path), engine='xlsxwriter')

    # Write the dataframe to a worksheet.
    writer
    predictions_df.to_excel(writer, sheet_name='Predictions', index=False)

    # Access the XlsxWriter workbook and worksheet objects from the writer object.
    workbook = writer.book

    # Insert the plots into new worksheets.
    worksheet_predicted_vs_actual = workbook.add_worksheet('Predicted vs Actual')
    worksheet_predicted_vs_actual.insert_image('A1', predicted_vs_actual_path)

    worksheet_uncertainty_bands = workbook.add_worksheet('Uncertainty Bands')
    worksheet_uncertainty_bands.insert_image('A1', uncertainty_bands_path)

    worksheet_histogram_stddev = workbook.add_worksheet('Histogram StdDev')
    worksheet_histogram_stddev.insert_image('A1', histogram_stddev_path)

    worksheet_residual_plot = workbook.add_worksheet('Residual Plot')
    worksheet_residual_plot.insert_image('A1', residual_plot_path)

    worksheet_stddev = workbook.add_worksheet('StdDev Plot')
    worksheet_stddev.insert_image('A1', stddev_plot_path)

    worksheet_prediction = workbook.add_worksheet('Predictions Plot')
    worksheet_prediction.insert_image('A1', prediction_plot_path)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
    print(f"Excel file saved at {excel_path}")

    # Optionally, remove the image files now that they're embedded in the Excel file
    if os.path.exists(str(predicted_vs_actual_path)): os.remove(str(predicted_vs_actual_path))
    if os.path.exists(str(uncertainty_bands_path)): os.remove(str(uncertainty_bands_path))
    if os.path.exists(str(histogram_stddev_path)): os.remove(str(histogram_stddev_path))
    if os.path.exists(str(residual_plot_path)): os.remove(str(residual_plot_path))
    if os.path.exists(str(stddev_plot_path)): os.remove(str(stddev_plot_path))
    if os.path.exists(str(prediction_plot_path)): os.remove(str(prediction_plot_path))
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def generate_and_save_crps_decomposition(
    predictions_df,
    y_true,
    excel_path,
    model_name="Model",
    n_bins=10
):

    # -------------------------------
    # Input handling
    # -------------------------------
    mu_pred = predictions_df['Mean'].values
    sigma_pred = predictions_df['StdDev'].values

    if 'Actual' in predictions_df.columns:
        y_true = predictions_df['Actual'].values
    else:
        y_true = np.asarray(y_true).ravel()

    y_true = np.asarray(y_true).ravel()

    # -------------------------------
    # Total CRPS
    # -------------------------------
    crps_values = ps.crps_gaussian(
        y_true, mu=mu_pred, sig=sigma_pred
    )
    total_crps = np.mean(crps_values)

    # -------------------------------
    # Uncertainty (climatology)
    # -------------------------------
    global_mu = np.mean(y_true)
    global_sigma = np.std(y_true)

    uncertainty = np.mean(
        ps.crps_gaussian(y_true, mu=global_mu, sig=global_sigma)
    )

    # -------------------------------
    # Reliability & Resolution
    # -------------------------------
    sort_idx = np.argsort(mu_pred)
    y_sorted = y_true[sort_idx]
    mu_sorted = mu_pred[sort_idx]
    sigma_sorted = sigma_pred[sort_idx]

    bins = np.array_split(np.arange(len(y_true)), n_bins)

    bin_results = []
    reliability_sum = 0.0

    for i, b in enumerate(bins):
        bin_y = y_sorted[b]
        bin_mu = mu_sorted[b]
        bin_sigma = sigma_sorted[b]

        bin_crps = np.mean(
            ps.crps_gaussian(bin_y, mu=bin_mu, sig=bin_sigma)
        )

        reliability_sum += bin_crps

        bin_results.append({
            "Bin": i + 1,
            "Bin_Size": len(b),
            "Mean_Pred_Mu": np.mean(bin_mu),
            "Mean_Pred_Sigma": np.mean(bin_sigma),
            "Bin_CRPS": bin_crps
        })

    reliability = reliability_sum / n_bins
    resolution = uncertainty - (total_crps - reliability)

    # -------------------------------
    # Summary table
    # -------------------------------
    summary_df = pd.DataFrame([{
        "Model": model_name,
        "Total_CRPS": total_crps,
        "Reliability": reliability,
        "Resolution": resolution,
        "Uncertainty": uncertainty,
        "Num_Bins": n_bins
    }])

    bin_df = pd.DataFrame(bin_results)

    # -------------------------------
    # Save to Excel
    # -------------------------------
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_summary = f"CRPS_Summary_{model_name}"
    sheet_bins = f"CRPS_Bins_{model_name}"

    with pd.ExcelWriter(_ensure_parent_dir(_ensure_excel_file(
        excel_path)),
        engine="openpyxl",
        mode="a" if excel_path.exists() else "w",
        if_sheet_exists="replace"
    ) as writer:
        writer
        summary_df.to_excel(writer, sheet_name=sheet_summary, index=False)
        writer
        bin_df.to_excel(writer, sheet_name=sheet_bins, index=False)

    print(f"✅ CRPS decomposition saved to:\n{excel_path}")
    print(f"   Sheets: {sheet_summary}, {sheet_bins}")

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
def generate_and_save_pit_results(
    predictions_df,
    y_true,
    excel_path,
    model_name="Model",
    n_bins=10
):
    """
    Generate PIT values, histogram, calibration diagnostics,
    and save everything to Excel with the PIT plot EMBEDDED.
    """

    # -------------------------------
    # Input handling
    # -------------------------------
    mu_pred = predictions_df['Mean'].values
    sigma_pred = predictions_df['StdDev'].values

    if 'Actual' in predictions_df.columns:
        y_true = predictions_df['Actual'].values
    else:
        y_true = np.asarray(y_true).ravel()

    y_true = np.asarray(y_true).ravel()

    sigma_pred = np.clip(sigma_pred, 1e-6, None)

    # -------------------------------
    # Compute PIT values
    # -------------------------------
    pit_values = norm.cdf(y_true, loc=mu_pred, scale=sigma_pred)

    pit_df = pd.DataFrame({"PIT_Value": pit_values})

    # -------------------------------
    # Histogram data
    # -------------------------------
    hist_density, bin_edges = np.histogram(
        pit_values, bins=n_bins, range=(0, 1), density=True
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_df = pd.DataFrame({
        "Bin_Center": bin_centers,
        "Density": hist_density
    })

    # -------------------------------
    # Calibration diagnostics
    # -------------------------------
    pit_mean = pit_values.mean()
    pit_var = pit_values.var()
    ks_stat, ks_pvalue = kstest(pit_values, 'uniform')

    if ks_pvalue > 0.05:
        diagnosis = "Well-calibrated (Uniform PIT)"
    elif pit_var < 1/12:
        diagnosis = "Over-dispersed (Under-confident)"
    elif pit_var > 1/12:
        diagnosis = "Under-dispersed (Over-confident)"
    else:
        diagnosis = "Biased or miscalibrated"

    summary_df = pd.DataFrame([{
        "Model": model_name,
        "Mean_PIT": pit_mean,
        "Variance_PIT": pit_var,
        "Ideal_Variance_(1/12)": 1/12,
        "KS_Statistic": ks_stat,
        "KS_pvalue": ks_pvalue,
        "Calibration_Diagnosis": diagnosis
    }])

    # -------------------------------
    # Create PIT plot in memory
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        pit_values,
        bins=n_bins,
        density=True,
        color="skyblue",
        edgecolor="black",
        alpha=0.75
    )
    ax.axhline(1, color="red", linestyle="--", label="Perfect Calibration")
    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT Histogram – {model_name}")
    ax.legend()
    fig.tight_layout()

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300)
    plt.close(fig)
    img_buffer.seek(0)

    # -------------------------------
    # Save to Excel
    # -------------------------------
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_values = f"PIT_Values_{model_name}"
    sheet_hist = f"PIT_Histogram_{model_name}"
    sheet_summary = f"PIT_Summary_{model_name}"
    sheet_plot = f"PIT_Plot_{model_name}"

    with pd.ExcelWriter(_ensure_parent_dir(_ensure_excel_file(
        excel_path)),
        engine="openpyxl",
        mode="a" if excel_path.exists() else "w",
        if_sheet_exists="replace"
    ) as writer:

        writer
        pit_df.to_excel(writer, sheet_name=sheet_values, index=False)
        writer
        hist_df.to_excel(writer, sheet_name=sheet_hist, index=False)
        writer
        summary_df.to_excel(writer, sheet_name=sheet_summary, index=False)

        # Create empty sheet for plot
        workbook = writer.book
        worksheet = workbook.create_sheet(sheet_plot)

        img = openpyxlImage(img_buffer)
        img.width = img.width * 0.4
        img.height = img.height * 0.4

        img.anchor = "B2"
        worksheet.add_image(img)

    print(f"✅ PIT values, histogram, summary, and EMBEDDED plot saved to:")
    print(f"   {excel_path}")

    return pit_df
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def generate_and_save_calibration_curve(
    predictions_df,
    y_true,
    excel_path,
    model_name="Model",
    n_quantiles=10
):
    """
    Generate quantile calibration curve (reliability diagram),
    compute calibration errors, and save everything to Excel
    with the plot EMBEDDED.
    """

    # -------------------------------
    # Input handling
    # -------------------------------
    mu_pred = predictions_df['Mean'].values
    sigma_pred = predictions_df['StdDev'].values

    if 'Actual' in predictions_df.columns:
        y_true = predictions_df['Actual'].values
    else:
        y_true = np.asarray(y_true).ravel()

    y_true = np.asarray(y_true).ravel()
    sigma_pred = np.clip(sigma_pred, 1e-6, None)

    # -------------------------------
    # Quantiles
    # -------------------------------
    quantiles = np.linspace(0, 1, n_quantiles + 1)

    observed_freq = []
    predicted_q = []

    for q in quantiles:
        # Predicted value at quantile q
        thresh = norm.ppf(q, loc=mu_pred, scale=sigma_pred)

        # Observed frequency
        obs = np.mean(y_true <= thresh)

        predicted_q.append(q)
        observed_freq.append(obs)

    predicted_q = np.array(predicted_q)
    observed_freq = np.array(observed_freq)

    # -------------------------------
    # Calibration metrics
    # -------------------------------
    mce = np.mean(np.abs(observed_freq - predicted_q))
    rmsce = np.sqrt(np.mean((observed_freq - predicted_q) ** 2))

    calib_df = pd.DataFrame({
        "Predicted_Quantile": predicted_q,
        "Observed_Frequency": observed_freq,
        "Absolute_Error": np.abs(observed_freq - predicted_q)
    })

    summary_df = pd.DataFrame([{
        "Model": model_name,
        "Mean_Calibration_Error (MCE)": mce,
        "RMS_Calibration_Error (RMSCE)": rmsce,
        "Num_Quantiles": n_quantiles
    }])

    # -------------------------------
    # Create calibration plot
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    ax.plot(
        predicted_q,
        observed_freq,
        'o-',
        color='teal',
        label=f"{model_name}"
    )

    ax.set_xlabel("Predicted Quantile (Confidence Level)")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(f"Quantile Calibration Curve – {model_name}")
    ax.legend()
    fig.tight_layout()

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300)
    plt.close(fig)
    img_buffer.seek(0)

    # -------------------------------
    # Save to Excel
    # -------------------------------
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_curve = f"Calibration_Curve_{model_name}"
    sheet_summary = f"Calibration_Summary_{model_name}"
    sheet_plot = f"Calibration_Plot_{model_name}"

    with pd.ExcelWriter(_ensure_parent_dir(_ensure_excel_file(
        excel_path)),
        engine="openpyxl",
        mode="a" if excel_path.exists() else "w",
        if_sheet_exists="replace"
    ) as writer:

        writer
        calib_df.to_excel(writer, sheet_name=sheet_curve, index=False)
        writer
        summary_df.to_excel(writer, sheet_name=sheet_summary, index=False)

        workbook = writer.book

        worksheet = workbook.create_sheet(sheet_plot)

        img = openpyxlImage(img_buffer)
        img.width = img.width * 0.4
        img.height = img.height * 0.4

        img.anchor = "B2"
        worksheet.add_image(img)

    print("✅ Calibration curve & metrics saved to Excel:")
    print(f"   {excel_path}")

    return calib_df
```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
# Define the path to the folder
folder_path = "/content/drive/MyDrive/Regression/pile_uncertainty_analysis"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

def plot_metrics_and_save_to_excel(writer, predictions_df, y_true, model_name):
    # Calculate metrics
    mean_pred = predictions_df['Mean']
    std_pred = predictions_df['StdDev']

    # Sharpness is the standard deviation
    sharpness = std_pred.mean()

    # CRPS for each prediction
    crps_values = ps.crps_gaussian(y_true, mean_pred, std_pred)

    # Negative Log Likelihood (NLL) for each prediction
    nll_values = -norm.logpdf(y_true, loc=mean_pred, scale=std_pred)
    nll = nll_values.mean()

    # Calibration (Validity)
    percentiles = np.linspace(0, 100, 101)
    calibration = [np.mean((y_true <= np.percentile(mean_pred + std_pred * np.random.randn(*mean_pred.shape), p))) for p in percentiles]

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'Mean Prediction': mean_pred,
        'Standard Deviation': std_pred,
        'CRPS': crps_values,
        'NLL': nll_values
    })

    # Add summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['Sharpness', 'Mean CRPS', 'Mean NLL'],
        'Value': [sharpness, crps_values.mean(), nll]
    })

    # Write the metrics and summary to the Excel sheet
    metrics_sheet_name = f'{model_name} Metrics'
    writer
    metrics_df.to_excel(writer, sheet_name=metrics_sheet_name, index=False)

    summary_sheet_name = f'{model_name} Summary'
    writer
    summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Sharpness plot
    axs[0, 0].hist(std_pred, bins=30, color='skyblue', edgecolor='black')
    axs[0, 0].set_title(f'Sharpness (StdDev) for {model_name}')
    axs[0, 0].set_xlabel('Standard Deviation')
    axs[0, 0].set_ylabel('Frequency')

    # CRPS plot (distribution)
    axs[0, 1].hist(crps_values, bins=30, color='lightgreen', edgecolor='black')
    axs[0, 1].set_title(f'CRPS Distribution for {model_name}')
    axs[0, 1].set_xlabel('CRPS')
    axs[0, 1].set_ylabel('Frequency')

    # NLL plot (distribution)
    axs[1, 0].hist(nll_values, bins=30, color='salmon', edgecolor='black')
    axs[1, 0].set_title(f'NLL Distribution for {model_name}')
    axs[1, 0].set_xlabel('NLL')
    axs[1, 0].set_ylabel('Frequency')

    # Calibration plot
    axs[1, 1].plot(percentiles, calibration, label='Calibration Curve')
    axs[1, 1].plot(percentiles, percentiles / 100, linestyle='--', label='Perfect Calibration')
    axs[1, 1].set_title(f'Calibration (Validity) for {model_name}')
    axs[1, 1].set_xlabel('Predicted Percentile')
    axs[1, 1].set_ylabel('Observed Frequency')
    axs[1, 1].legend()

    plt.tight_layout()

    # Save plots to images
    plot_image_path = f'{model_name}_plots.png'
    fig.savefig(_ensure_parent_dir(plot_image_path))
    plt.close(fig)

    # Insert the plot image into the Excel sheet
    sheet_name = f'{model_name} Plots'
    if sheet_name not in writer.sheets:  # Check if sheet already exists
        # If sheet doesn't exist, create a new one
        worksheet = writer.book.add_worksheet(sheet_name)
    else:
        # If sheet exists, get the existing sheet object to avoid duplicates
        worksheet = writer.sheets[sheet_name]
    worksheet.insert_image('A1', plot_image_path)
```

# **Class CARD**

```python
class CARDRegressor:
    def __init__(
        self,
        base_model,
        hidden_dim=128,
        T=50,
        lr=1e-3,
        epochs=200,
        n_samples=100,
        device="cpu"
    ):
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.T = T
        self.epochs = epochs
        self.n_samples = n_samples
        self.device = device

        self.mlp = None  # ← FIX
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

    def fit(self, X, y):
        # Base predictions
        y_pred = self.base_model.predict(X)
        residuals = y - y_pred

        X_t = torch.tensor(X, dtype=torch.float32)
        r_t = torch.tensor(residuals, dtype=torch.float32)

        # Build MLP HERE (now input dim is known)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1] + 1, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)

        for _ in range(self.epochs):
            t = torch.randint(0, self.T, (len(X_t),), dtype=torch.float32)
            noise = torch.randn_like(r_t)

            pred_noise = self.mlp(
                torch.cat([X_t, t.unsqueeze(1)], dim=1)
            ).squeeze()

            loss = self.loss_fn(pred_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        base_pred = self.base_model.predict(X)
        X_t = torch.tensor(X, dtype=torch.float32)

        samples = []
        for _ in range(self.n_samples):
            t = torch.zeros(len(X_t))
            noise = torch.randn(len(X_t))

            pred_noise = self.mlp(
                torch.cat([X_t, t.unsqueeze(1)], dim=1)
            ).detach().squeeze()

            samples.append(base_pred + (noise - pred_noise).numpy())

        samples = np.stack(samples, axis=1)
        return samples.mean(axis=1), samples.std(axis=1)

```

# **Tuned Parameters**

```python
best_scores_autosampler = {('Random Forest', 'MedianPruner'): {'best_score': 0.14752680231770787,
  'best_params': {'n_estimators': 200,
   'criterion': 'absolute_error',
   'max_depth': 40,
   'min_samples_split': 2,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'sqrt',
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 0.14752680231770787,
  'test_rmse': 0.38409217945397933,
  'test_corr_coef': 0.9635104517818707,
  'pruner': 'MedianPruner'},
 ('Random Forest', 'NopPruner'): {'best_score': 0.1706761150095517,
  'best_params': {'n_estimators': 200,
   'criterion': 'squared_error',
   'max_depth': None,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.2,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.1},
  'test_mse': 0.1706761150095517,
  'test_rmse': 0.4131296588355183,
  'test_corr_coef': 0.9580814021268568,
  'pruner': 'NopPruner'},
 ('Random Forest', 'PatientPruner'): {'best_score': 0.15322308337916704,
  'best_params': {'n_estimators': 500,
   'criterion': 'absolute_error',
   'max_depth': 20,
   'min_samples_split': 10,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 0.3,
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01},
  'test_mse': 0.15322308337916704,
  'test_rmse': 0.39143720234434415,
  'test_corr_coef': 0.9633540847173576,
  'pruner': 'PatientPruner'},
 ('Random Forest', 'PercentilePruner'): {'best_score': 0.12422375760416486,
  'best_params': {'n_estimators': 300,
   'criterion': 'absolute_error',
   'max_depth': 30,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.05},
  'test_mse': 0.12422375760416486,
  'test_rmse': 0.35245390848189617,
  'test_corr_coef': 0.9705945503043777,
  'pruner': 'PercentilePruner'},
 ('Random Forest',
  'SuccessiveHalvingPruner'): {'best_score': 0.12419639091145714, 'best_params': {'n_estimators': 200,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.05}, 'test_mse': 0.12419639091145714, 'test_rmse': 0.3524150832632694, 'test_corr_coef': 0.9705007038236941, 'pruner': 'SuccessiveHalvingPruner'},
 ('Random Forest', 'HyperbandPruner'): {'best_score': 0.12872488973958218,
  'best_params': {'n_estimators': 200,
   'criterion': 'absolute_error',
   'max_depth': 40,
   'min_samples_split': 10,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.05},
  'test_mse': 0.12872488973958218,
  'test_rmse': 0.35878251035910624,
  'test_corr_coef': 0.9695593686182873,
  'pruner': 'HyperbandPruner'},
 ('Random Forest', 'ThresholdPruner'): {'best_score': 0.13019523971354047,
  'best_params': {'n_estimators': 200,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 10,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.05},
  'test_mse': 0.13019523971354047,
  'test_rmse': 0.3608257747355924,
  'test_corr_coef': 0.9690740875183095,
  'pruner': 'ThresholdPruner'},
 ('Random Forest', 'WilcoxonPruner'): {'best_score': 0.15332546253750082,
  'best_params': {'n_estimators': 500,
   'criterion': 'absolute_error',
   'max_depth': 30,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 0.5,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 0.15332546253750082,
  'test_rmse': 0.3915679539205179,
  'test_corr_coef': 0.9621549031032881,
  'pruner': 'WilcoxonPruner'},
 ('Gradient Boosting', 'MedianPruner'): {'best_score': 0.11741937518416184,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.05,
   'n_estimators': 300,
   'subsample': 0.7,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 3,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.001,
   'ccp_alpha': 0.0},
  'test_mse': 0.11741937518416184,
  'test_rmse': 0.34266510645842224,
  'test_corr_coef': 0.971121024573984,
  'pruner': 'MedianPruner'},
 ('Gradient Boosting', 'NopPruner'): {'best_score': 0.12451436682247834,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.1,
   'n_estimators': 500,
   'subsample': 0.9,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.1,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 0.12451436682247834,
  'test_rmse': 0.3528659332132791,
  'test_corr_coef': 0.9711768472673782,
  'pruner': 'NopPruner'},
 ('Gradient Boosting', 'PatientPruner'): {'best_score': 0.11175210721294264,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.2,
   'n_estimators': 500,
   'subsample': 0.9,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 7,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 0.11175210721294264,
  'test_rmse': 0.3342934447651384,
  'test_corr_coef': 0.9697390644667225,
  'pruner': 'PatientPruner'},
 ('Gradient Boosting', 'PercentilePruner'): {'best_score': 0.11322335137178252,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.05,
   'n_estimators': 500,
   'subsample': 0.5,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 7,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 0.11322335137178252,
  'test_rmse': 0.33648677741002325,
  'test_corr_coef': 0.9700016394192005,
  'pruner': 'PercentilePruner'},
 ('Gradient Boosting',
  'SuccessiveHalvingPruner'): {'best_score': 0.11430625414117218, 'best_params': {'loss': 'quantile',
   'learning_rate': 0.05,
   'n_estimators': 700,
   'subsample': 0.9,
   'criterion': 'friedman_mse',
   'min_samples_split': 5,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.001,
   'ccp_alpha': 0.0}, 'test_mse': 0.11430625414117218, 'test_rmse': 0.3380920793824845, 'test_corr_coef': 0.9732713933160871, 'pruner': 'SuccessiveHalvingPruner'},
 ('Gradient Boosting', 'HyperbandPruner'): {'best_score': 0.10783128880543709,
  'best_params': {'loss': 'quantile',
   'learning_rate': 0.1,
   'n_estimators': 200,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 10,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 5,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 'log2',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 0.10783128880543709,
  'test_rmse': 0.3283767482716112,
  'test_corr_coef': 0.9732271129436233,
  'pruner': 'HyperbandPruner'},
 ('Gradient Boosting', 'ThresholdPruner'): {'best_score': 0.12000604065154852,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.1,
   'n_estimators': 500,
   'subsample': 0.5,
   'criterion': 'squared_error',
   'min_samples_split': 0.01,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.1,
   'max_depth': 5,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 'log2',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.0001,
   'ccp_alpha': 0.01},
  'test_mse': 0.12000604065154852,
  'test_rmse': 0.34641888033354723,
  'test_corr_coef': 0.9685509477926916,
  'pruner': 'ThresholdPruner'},
 ('Gradient Boosting', 'WilcoxonPruner'): {'best_score': 0.11570400469150145,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.05,
   'n_estimators': 700,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 5,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 10,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.001,
   'ccp_alpha': 0.01},
  'test_mse': 0.11570400469150145,
  'test_rmse': 0.3401529136895662,
  'test_corr_coef': 0.9728303315204233,
  'pruner': 'WilcoxonPruner'},
 ('XGBoost', 'MedianPruner'): {'best_score': 0.17216512303837542,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 3,
   'gamma': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.5,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.17216512303837542,
  'test_rmse': 0.4149278528110344,
  'test_corr_coef': 0.9579215530428703,
  'pruner': 'MedianPruner'},
 ('XGBoost', 'NopPruner'): {'best_score': 0.18603178163690948,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 0.5,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.18603178163690948,
  'test_rmse': 0.43131401743614767,
  'test_corr_coef': 0.9572878786872289,
  'pruner': 'NopPruner'},
 ('XGBoost', 'PatientPruner'): {'best_score': 0.19005962626345443,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.19005962626345443,
  'test_rmse': 0.4359582850037999,
  'test_corr_coef': 0.95722040785803,
  'pruner': 'PatientPruner'},
 ('XGBoost', 'PercentilePruner'): {'best_score': 0.1729339854509627,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 5,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.5,
   'colsample_bytree': 0.5,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.1729339854509627,
  'test_rmse': 0.41585332203910874,
  'test_corr_coef': 0.959339134792702,
  'pruner': 'PercentilePruner'},
 ('XGBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.1865918244582946,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'max_depth': 3,
   'min_child_weight': 5,
   'gamma': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.5,
   'reg_alpha': 1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.1865918244582946,
  'test_rmse': 0.4319627581844233,
  'test_corr_coef': 0.9541301674034157,
  'pruner': 'SuccessiveHalvingPruner'},
 ('XGBoost', 'HyperbandPruner'): {'best_score': 0.18021085625063793,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.18021085625063793,
  'test_rmse': 0.42451249245533157,
  'test_corr_coef': 0.9563422811945648,
  'pruner': 'HyperbandPruner'},
 ('XGBoost', 'ThresholdPruner'): {'best_score': 0.19119466715567288,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 5,
   'min_child_weight': 1,
   'gamma': 0.5,
   'subsample': 0.6,
   'colsample_bytree': 0.5,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.19119466715567288,
  'test_rmse': 0.437258124173437,
  'test_corr_coef': 0.9550669012393855,
  'pruner': 'ThresholdPruner'},
 ('XGBoost', 'WilcoxonPruner'): {'best_score': 0.18638765033677784,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_depth': 3,
   'min_child_weight': 3,
   'gamma': 0.5,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 0.18638765033677784,
  'test_rmse': 0.43172636048401986,
  'test_corr_coef': 0.9551930060748421,
  'pruner': 'WilcoxonPruner'},
 ('LightGBM', 'MedianPruner'): {'best_score': 0.19845946645149384,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': 5,
   'min_child_samples': 10,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 1,
   'min_child_weight': 0.001,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.19845946645149384,
  'test_rmse': 0.4454878970875571,
  'test_corr_coef': 0.9516679524186389,
  'pruner': 'MedianPruner'},
 ('LightGBM', 'NopPruner'): {'best_score': 0.20311618990758282,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 31,
   'max_depth': 7,
   'min_child_samples': 10,
   'subsample': 0.5,
   'colsample_bytree': 1.0,
   'reg_alpha': 0,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.20311618990758282,
  'test_rmse': 0.45068413540703073,
  'test_corr_coef': 0.9497618825773275,
  'pruner': 'NopPruner'},
 ('LightGBM', 'PatientPruner'): {'best_score': 0.19399049231918353,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': 7,
   'min_child_samples': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'min_child_weight': 1e-05,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.19399049231918353,
  'test_rmse': 0.4404435177399975,
  'test_corr_coef': 0.9545592867128037,
  'pruner': 'PatientPruner'},
 ('LightGBM', 'PercentilePruner'): {'best_score': 0.20998106141273384,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'num_leaves': 63,
   'max_depth': 3,
   'min_child_samples': 5,
   'subsample': 0.7,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'min_child_weight': 1e-05,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.20998106141273384,
  'test_rmse': 0.4582369053368943,
  'test_corr_coef': 0.9478902650213303,
  'pruner': 'PercentilePruner'},
 ('LightGBM', 'SuccessiveHalvingPruner'): {'best_score': 0.23944396516258096,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 31,
   'max_depth': -1,
   'min_child_samples': 20,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'bagging_freq': 0,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.23944396516258096,
  'test_rmse': 0.48933011879771,
  'test_corr_coef': 0.9333488584307176,
  'pruner': 'SuccessiveHalvingPruner'},
 ('LightGBM', 'HyperbandPruner'): {'best_score': 0.2456636839699559,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': 3,
   'min_child_samples': 20,
   'subsample': 0.8,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'bagging_freq': 0,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.2456636839699559,
  'test_rmse': 0.4956447154665889,
  'test_corr_coef': 0.9314533403107574,
  'pruner': 'HyperbandPruner'},
 ('LightGBM', 'ThresholdPruner'): {'best_score': 0.17946945784552895,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 63,
   'max_depth': 3,
   'min_child_samples': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 0.1,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.17946945784552895,
  'test_rmse': 0.4236383573822476,
  'test_corr_coef': 0.9542866862433039,
  'pruner': 'ThresholdPruner'},
 ('LightGBM', 'WilcoxonPruner'): {'best_score': 0.19726148184849976,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 31,
   'max_depth': -1,
   'min_child_samples': 5,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'min_child_weight': 0.01,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.19726148184849976,
  'test_rmse': 0.44414128590854934,
  'test_corr_coef': 0.952490619982741,
  'pruner': 'WilcoxonPruner'},
 ('GPBoost', 'MedianPruner'): {'best_score': 0.24690592013645007,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 20,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.5,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.24690592013645007,
  'test_rmse': 0.49689628710270123,
  'test_corr_coef': 0.9306706960046395,
  'pruner': 'MedianPruner'},
 ('GPBoost', 'NopPruner'): {'best_score': 0.24405561084366204,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 20,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.24405561084366204,
  'test_rmse': 0.4940198486332933,
  'test_corr_coef': 0.9318275529865149,
  'pruner': 'NopPruner'},
 ('GPBoost', 'PatientPruner'): {'best_score': 0.21590074033692608,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'max_depth': 5,
   'num_leaves': 31,
   'min_child_samples': 10,
   'subsample': 1.0,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 0,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.21590074033692608,
  'test_rmse': 0.46465120287902634,
  'test_corr_coef': 0.9499084350832003,
  'pruner': 'PatientPruner'},
 ('GPBoost', 'PercentilePruner'): {'best_score': 0.23954954846775026,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': -1,
   'num_leaves': 31,
   'min_child_samples': 20,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.23954954846775026,
  'test_rmse': 0.4894379924645718,
  'test_corr_coef': 0.9331925578271782,
  'pruner': 'PercentilePruner'},
 ('GPBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.21341642160636554,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': -1,
   'num_leaves': 15,
   'min_child_samples': 10,
   'subsample': 0.7,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.5,
   'reg_lambda': 1.0,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.21341642160636554,
  'test_rmse': 0.4619701522894802,
  'test_corr_coef': 0.9485463939108688,
  'pruner': 'SuccessiveHalvingPruner'},
 ('GPBoost', 'HyperbandPruner'): {'best_score': 0.21341642160636554,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': -1,
   'num_leaves': 63,
   'min_child_samples': 10,
   'subsample': 0.8,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.5,
   'reg_lambda': 1.0,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.21341642160636554,
  'test_rmse': 0.4619701522894802,
  'test_corr_coef': 0.9485463939108688,
  'pruner': 'HyperbandPruner'},
 ('GPBoost', 'ThresholdPruner'): {'best_score': 0.24647858129891828,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 20,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.5,
   'reg_lambda': 0,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.24647858129891828,
  'test_rmse': 0.49646609279881165,
  'test_corr_coef': 0.9305798035043363,
  'pruner': 'ThresholdPruner'},
 ('GPBoost', 'WilcoxonPruner'): {'best_score': 0.2149656245515983,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'max_depth': 7,
   'num_leaves': 31,
   'min_child_samples': 10,
   'subsample': 0.7,
   'colsample_bytree': 0.5,
   'reg_alpha': 0,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 0.2149656245515983,
  'test_rmse': 0.4636438552936923,
  'test_corr_coef': 0.9489329195829942,
  'pruner': 'WilcoxonPruner'},
 ('CatBoost', 'MedianPruner'): {'best_score': 0.19120194972527008,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.03,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 20,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.19120194972527008,
  'test_rmse': 0.4372664516347785,
  'test_corr_coef': 0.9487462669948028,
  'pruner': 'MedianPruner'},
 ('CatBoost', 'NopPruner'): {'best_score': 0.21955757985830573,
  'best_params': {'iterations': 200,
   'learning_rate': 0.03,
   'depth': 6,
   'l2_leaf_reg': 3,
   'border_count': 32,
   'min_data_in_leaf': 20,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.21955757985830573,
  'test_rmse': 0.4685697171801713,
  'test_corr_coef': 0.9438276613741003,
  'pruner': 'NopPruner'},
 ('CatBoost', 'PatientPruner'): {'best_score': 0.20247256526346125,
  'best_params': {'iterations': 500,
   'learning_rate': 0.1,
   'depth': 8,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 5,
   'rsm': 1.0,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.20247256526346125,
  'test_rmse': 0.44996951592686946,
  'test_corr_coef': 0.9465183445238835,
  'pruner': 'PatientPruner'},
 ('CatBoost', 'PercentilePruner'): {'best_score': 0.1876633039227785,
  'best_params': {'iterations': 500,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 1,
   'rsm': 1.0,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.1876633039227785,
  'test_rmse': 0.4332012279793058,
  'test_corr_coef': 0.9505985870988952,
  'pruner': 'PercentilePruner'},
 ('CatBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.1876686365237472,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 10,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.1876686365237472,
  'test_rmse': 0.43320738281306703,
  'test_corr_coef': 0.9505954285925091,
  'pruner': 'SuccessiveHalvingPruner'},
 ('CatBoost', 'HyperbandPruner'): {'best_score': 0.16955476359331176,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 5,
   'rsm': 0.8,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.16955476359331176,
  'test_rmse': 0.411770280123896,
  'test_corr_coef': 0.9545229535961917,
  'pruner': 'HyperbandPruner'},
 ('CatBoost', 'ThresholdPruner'): {'best_score': 0.1876633039227785,
  'best_params': {'iterations': 500,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 10,
   'rsm': 1.0,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.1876633039227785,
  'test_rmse': 0.4332012279793058,
  'test_corr_coef': 0.9505985870988952,
  'pruner': 'ThresholdPruner'},
 ('CatBoost', 'WilcoxonPruner'): {'best_score': 0.16958500552464362,
  'best_params': {'iterations': 500,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 10,
   'rsm': 0.8,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 0.16958500552464362,
  'test_rmse': 0.4118070003346757,
  'test_corr_coef': 0.9545297513334484,
  'pruner': 'WilcoxonPruner'},
 ('NGBoost', 'MedianPruner'): {'best_score': 0.10680549009394465,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.10680549009394465,
  'test_rmse': 0.32681109236674427,
  'test_corr_coef': 0.9736167792299366,
  'pruner': 'MedianPruner'},
 ('NGBoost', 'NopPruner'): {'best_score': 0.10986668754073355,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.05,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.10986668754073355,
  'test_rmse': 0.33146144201208916,
  'test_corr_coef': 0.9700480176632558,
  'pruner': 'NopPruner'},
 ('NGBoost', 'PatientPruner'): {'best_score': 0.116158798611392,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.1,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.116158798611392,
  'test_rmse': 0.3408207719775777,
  'test_corr_coef': 0.9705871785896832,
  'pruner': 'PatientPruner'},
 ('NGBoost', 'PercentilePruner'): {'best_score': 0.1050298520817627,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.1,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.1050298520817627,
  'test_rmse': 0.32408309440907695,
  'test_corr_coef': 0.9709960891496683,
  'pruner': 'PercentilePruner'},
 ('NGBoost', 'SuccessiveHalvingPruner'): {'best_score': 0.12009011085041925,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.12009011085041925,
  'test_rmse': 0.34654020091530396,
  'test_corr_coef': 0.9713784206071934,
  'pruner': 'SuccessiveHalvingPruner'},
 ('NGBoost', 'HyperbandPruner'): {'best_score': 0.10590469596111333,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.1,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.10590469596111333,
  'test_rmse': 0.3254300169946118,
  'test_corr_coef': 0.9718003402714438,
  'pruner': 'HyperbandPruner'},
 ('NGBoost', 'ThresholdPruner'): {'best_score': 0.10460175722886922,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.10460175722886922,
  'test_rmse': 0.32342194920702155,
  'test_corr_coef': 0.9758047256161577,
  'pruner': 'ThresholdPruner'},
 ('NGBoost', 'WilcoxonPruner'): {'best_score': 0.1191207604418576,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'natural_gradient': False,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.1191207604418576,
  'test_rmse': 0.34513875534610367,
  'test_corr_coef': 0.9686024362997154,
  'pruner': 'WilcoxonPruner'},
 ('TabNet', 'MedianPruner'): {'best_score': 0.3056489783917234,
  'best_params': {'n_d': 16,
   'n_a': 32,
   'n_steps': 7,
   'gamma': 1.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 3,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.3056489783917234,
  'test_rmse': 0.5528552960691644,
  'test_corr_coef': 0.9485378160034655,
  'pruner': 'MedianPruner'},
 ('TabNet', 'NopPruner'): {'best_score': 0.3060827874030739,
  'best_params': {'n_d': 8,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 2.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.3060827874030739,
  'test_rmse': 0.5532474919989009,
  'test_corr_coef': 0.9231154908956068,
  'pruner': 'NopPruner'},
 ('TabNet', 'PatientPruner'): {'best_score': 0.4051254057686129,
  'best_params': {'n_d': 8,
   'n_a': 16,
   'n_steps': 10,
   'gamma': 2.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 3,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.4051254057686129,
  'test_rmse': 0.6364946235190152,
  'test_corr_coef': 0.917530690678246,
  'pruner': 'PatientPruner'},
 ('TabNet', 'PercentilePruner'): {'best_score': 0.43986404130494644,
  'best_params': {'n_d': 64,
   'n_a': 64,
   'n_steps': 3,
   'gamma': 1.3,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 3,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.43986404130494644,
  'test_rmse': 0.6632224674307607,
  'test_corr_coef': 0.8710465294665402,
  'pruner': 'PercentilePruner'},
 ('TabNet', 'SuccessiveHalvingPruner'): {'best_score': 0.43232633658208247,
  'best_params': {'n_d': 64,
   'n_a': 64,
   'n_steps': 7,
   'gamma': 1.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 3,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.43232633658208247,
  'test_rmse': 0.657515274789934,
  'test_corr_coef': 0.9611621629185575,
  'pruner': 'SuccessiveHalvingPruner'},
 ('TabNet', 'HyperbandPruner'): {'best_score': 0.3045008097996339,
  'best_params': {'n_d': 32,
   'n_a': 64,
   'n_steps': 3,
   'gamma': 1.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 3,
   'n_independent': 3,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.3045008097996339,
  'test_rmse': 0.5518159202121972,
  'test_corr_coef': 0.9105978033389835,
  'pruner': 'HyperbandPruner'},
 ('TabNet', 'ThresholdPruner'): {'best_score': 0.28285498026470063,
  'best_params': {'n_d': 8,
   'n_a': 16,
   'n_steps': 3,
   'gamma': 1.5,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.28285498026470063,
  'test_rmse': 0.5318411231417712,
  'test_corr_coef': 0.9320610386461415,
  'pruner': 'ThresholdPruner'},
 ('TabNet', 'WilcoxonPruner'): {'best_score': 0.3852431697675478,
  'best_params': {'n_d': 64,
   'n_a': 8,
   'n_steps': 5,
   'gamma': 2.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 0.3852431697675478,
  'test_rmse': 0.6206796031508912,
  'test_corr_coef': 0.9060684683858008,
  'pruner': 'WilcoxonPruner'},
 ('HistGradientBoosting', 'MedianPruner'): {'best_score': 0.2052831927048833,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 400,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.2,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.2052831927048833,
  'test_rmse': 0.4530818830022707,
  'test_corr_coef': 0.9541777770389761,
  'pruner': 'MedianPruner'},
 ('HistGradientBoosting', 'NopPruner'): {'best_score': 0.2111932525133927,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 200,
   'max_depth': 7,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.2111932525133927,
  'test_rmse': 0.45955767049783064,
  'test_corr_coef': 0.9484835964617143,
  'pruner': 'NopPruner'},
 ('HistGradientBoosting', 'PatientPruner'): {'best_score': 0.2138351889423943,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 200,
   'max_depth': 7,
   'min_samples_leaf': 10,
   'max_leaf_nodes': None,
   'l2_regularization': 0.1,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.2138351889423943,
  'test_rmse': 0.46242317085370443,
  'test_corr_coef': 0.9479542105656841,
  'pruner': 'PatientPruner'},
 ('HistGradientBoosting',
  'PercentilePruner'): {'best_score': 0.24642635329818066, 'best_params': {'learning_rate': 0.05,
   'max_iter': 100,
   'max_depth': 7,
   'min_samples_leaf': 20,
   'max_leaf_nodes': None,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.24642635329818066, 'test_rmse': 0.4964134902459649, 'test_corr_coef': 0.9329915536835148, 'pruner': 'PercentilePruner'},
 ('HistGradientBoosting',
  'SuccessiveHalvingPruner'): {'best_score': 0.21193311006320778, 'best_params': {'learning_rate': 0.01,
   'max_iter': 200,
   'max_depth': 7,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 31,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.21193311006320778, 'test_rmse': 0.46036193376864665, 'test_corr_coef': 0.9527593540226986, 'pruner': 'SuccessiveHalvingPruner'},
 ('HistGradientBoosting',
  'HyperbandPruner'): {'best_score': 0.20941919332405332, 'best_params': {'learning_rate': 0.01,
   'max_iter': 300,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': None,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.20941919332405332, 'test_rmse': 0.45762341867965334, 'test_corr_coef': 0.9471337828054288, 'pruner': 'HyperbandPruner'},
 ('HistGradientBoosting',
  'ThresholdPruner'): {'best_score': 0.20619775435532298, 'best_params': {'learning_rate': 0.01,
   'max_iter': 300,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 31,
   'l2_regularization': 0.0,
   'max_bins': 128,
   'early_stopping': True,
   'validation_fraction': 0.2,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 0.20619775435532298, 'test_rmse': 0.454090028909822, 'test_corr_coef': 0.9538995069801086, 'pruner': 'ThresholdPruner'},
 ('HistGradientBoosting', 'WilcoxonPruner'): {'best_score': 0.2111932525133927,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 200,
   'max_depth': 5,
   'min_samples_leaf': 10,
   'max_leaf_nodes': None,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 0.2111932525133927,
  'test_rmse': 0.45955767049783064,
  'test_corr_coef': 0.9484835964617143,
  'pruner': 'WilcoxonPruner'},
 ('PGBM', 'MedianPruner'): {'best_score': 0.1648545015498252,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'max_leaves': 61,
   'min_split_gain': 1.0,
   'reg_lambda': 0.1,
   'feature_fraction': 0.5,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 10,
   'max_bin': 256,
   'distribution': 'laplace'},
  'test_mse': 0.1648545015498252,
  'test_rmse': 0.4060227845205552,
  'test_corr_coef': 0.9580263517260853,
  'pruner': 'MedianPruner'},
 ('PGBM', 'NopPruner'): {'best_score': 0.1710888428530448,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_leaves': 30,
   'min_split_gain': 1.0,
   'reg_lambda': 1.0,
   'feature_fraction': 0.5,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 5,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 0.1710888428530448,
  'test_rmse': 0.4136288709133404,
  'test_corr_coef': 0.9574036436002307,
  'pruner': 'NopPruner'},
 ('PGBM', 'PatientPruner'): {'best_score': 0.18070949091589739,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'max_leaves': 54,
   'min_split_gain': 1.0,
   'reg_lambda': 1.0,
   'feature_fraction': 0.7,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 10,
   'max_bin': 256,
   'distribution': 'laplace'},
  'test_mse': 0.18070949091589739,
  'test_rmse': 0.4250993894560393,
  'test_corr_coef': 0.9539308205804653,
  'pruner': 'PatientPruner'},
 ('PGBM', 'PercentilePruner'): {'best_score': 0.16507172976406367,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.15,
   'max_leaves': 53,
   'min_split_gain': 1.0,
   'reg_lambda': 0.1,
   'feature_fraction': 0.5,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 10,
   'max_bin': 128,
   'distribution': 'studentt'},
  'test_mse': 0.16507172976406367,
  'test_rmse': 0.4062902038741073,
  'test_corr_coef': 0.9578415965816813,
  'pruner': 'PercentilePruner'},
 ('PGBM', 'SuccessiveHalvingPruner'): {'best_score': 0.1700184465969458,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'max_leaves': 50,
   'min_split_gain': 0.5,
   'reg_lambda': 1.0,
   'feature_fraction': 0.5,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 128,
   'distribution': 'studentt'},
  'test_mse': 0.1700184465969458,
  'test_rmse': 0.41233293173956626,
  'test_corr_coef': 0.956864498835057,
  'pruner': 'SuccessiveHalvingPruner'},
 ('PGBM', 'HyperbandPruner'): {'best_score': 0.17154234019930356,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_leaves': 61,
   'min_split_gain': 0.5,
   'reg_lambda': 0.1,
   'feature_fraction': 0.5,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 10,
   'max_bin': 128,
   'distribution': 'normal'},
  'test_mse': 0.17154234019930356,
  'test_rmse': 0.414176701661626,
  'test_corr_coef': 0.9537239087902425,
  'pruner': 'HyperbandPruner'},
 ('PGBM', 'ThresholdPruner'): {'best_score': 0.1769292524000361,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 33,
   'min_split_gain': 1.0,
   'reg_lambda': 0.1,
   'feature_fraction': 0.5,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 10,
   'max_bin': 256,
   'distribution': 'normal'},
  'test_mse': 0.1769292524000361,
  'test_rmse': 0.42062959049505316,
  'test_corr_coef': 0.9562795032885706,
  'pruner': 'ThresholdPruner'},
 ('PGBM', 'WilcoxonPruner'): {'best_score': 0.19701356538623357,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_leaves': 63,
   'min_split_gain': 1.0,
   'reg_lambda': 5.0,
   'feature_fraction': 0.5,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 3,
   'max_bin': 256,
   'distribution': 'normal'},
  'test_mse': 0.19701356538623357,
  'test_rmse': 0.44386210176836854,
  'test_corr_coef': 0.954328575664314,
  'pruner': 'WilcoxonPruner'}}
```

# **Probabilistic Distribution using LGBM**

```python
best_params = get_best_model_params(best_scores_autosampler, 'LightGBM')

lgbm_model = LGBMRegressor(**best_params)
lgbm_model.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=lgbm_model,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=200
)

card_model.fit(X_train, y_train)

mean, stddev = card_model.predict(X_test)

predictions_LGBM_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_LGBM_df.head())

```

```python
generate_and_save_plots(predictions_LGBM_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx",
    model_name="LightGBM",
    n_quantiles=10
)
```

# **Probabilistic Distribution using XGBoost**

```python
best_params = get_best_model_params(best_scores_autosampler, 'XGBoost')

xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=xgb_model,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=200
)

card_model.fit(X_train, y_train)

mean, stddev = card_model.predict(X_test)

predictions_XGB_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_XGB_df.head())
```

```python
generate_and_save_plots(predictions_XGB_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx",
    model_name="XGBoost",
    n_quantiles=10
)
```

# **Probabilistic Distribution using GPBoost**


```python
best_params = get_best_model_params(best_scores_autosampler, 'GPBoost')

gpboost_model = GPBoostRegressor(**best_params)
gpboost_model.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=gpboost_model,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=200
)


card_model.fit(X_train, y_train)

mean, stddev = card_model.predict(X_test)

predictions_GPBoost_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_GPBoost_df.head())
```

```python
generate_and_save_plots(predictions_GPBoost_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx",
    model_name="GPBoost",
    n_quantiles=10
)
```

# **Probabilistic distributions using NGBoost**

```python

best_params = get_best_model_params(best_scores_autosampler, 'NGBoost')

incompatible_keys = ['Dist', 'Score']
for key in incompatible_keys:
    best_params.pop(key, None)

ngb_model = NGBRegressor(Dist=Normal, **best_params)

ngb_model.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=ngb_model,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=100
)

card_model.fit(X_train, y_train)

mean_pred, std_pred = card_model.predict(X_test)


predictions_NGB_df = pd.DataFrame({
    "Mean": mean_pred,
    "StdDev": std_pred,
    "Actual": y_test.ravel()
})

mse = mean_squared_error(y_test, mean_pred)
print(f"Mean Squared Error: {mse:.2f}")

print(predictions_NGB_df.head())
```

```python
generate_and_save_plots(predictions_NGB_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx",
    model_name="NGBoost",
    n_quantiles=10
)
```

# **Probabilistic Distribution using Gradient Boosting**

```python
best_params = get_best_model_params(best_scores_autosampler, 'GradientBoosting')

gbr_model = GradientBoostingRegressor(**best_params)
gbr_model.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=gbr_model,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=200
)

card_model.fit(X_train, y_train)

mean, stddev = card_model.predict(X_test)

predictions_GBR_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_GBR_df.head())
```

```python
generate_and_save_plots(predictions_GBR_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx",
    model_name="Gradient Boosting",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx",
    model_name="GBoosting",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx",
    model_name="GBoosting",
    n_quantiles=10
)
```

# **Probabilistic Distribution using Catboost**

```python

best_params = get_best_model_params(best_scores_autosampler, 'CatBoost')

catboost_model = CatBoostRegressor(**best_params)
catboost_model.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=catboost_model,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=200
)

card_model.fit(X_train, y_train)

mean, stddev = card_model.predict(X_test)

predictions_CatBoost_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_CatBoost_df.head())

```

```python
generate_and_save_plots(predictions_CatBoost_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="CatBoost",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="CatBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="CatBoost",
    n_quantiles=10
)
```

# **Probabilistic Distribution using HGBM**

```python


best_params = get_best_model_params(best_scores_autosampler, 'HGBR')


hgb_model = HistGradientBoostingRegressor(
    **best_params,
)
hgb_model.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=hgb_model,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=200
)

card_model.fit(X_train, y_train)

mean, stddev = card_model.predict(X_test)

predictions_HGBR_df = pd.DataFrame({
    "Mean": mean,
    "StdDev": stddev,
    "Actual": y_test.values.ravel()
})

print(predictions_HGBR_df.head())

```

```python
generate_and_save_plots(predictions_HGBR_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx",
    model_name="HGBR",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx",
    model_name="HGBR",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx",
    model_name="HGBR",
    n_quantiles=10
)
```

# **Probabilistic Distribution using TabNet**

```python
class TabNetMeanWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        self.model.fit(
            X,
            y,
            max_epochs=200,
            patience=30,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

    def predict(self, X):
        X = np.asarray(X)
        return self.model.predict(X).reshape(-1)

best_params = get_best_model_params(best_scores_autosampler, 'TabNet')

tabnet_base = TabNetMeanWrapper(
    TabNetRegressor(**best_params)
)

tabnet_base.fit(X_train, y_train)

card_model = CARDRegressor(
    base_model=tabnet_base,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=200
)

card_model.fit(X_train, y_train)

mean, stddev = card_model.predict(X_test)

predictions_TabNet_df = pd.DataFrame({
    "Mean": mean,
    "StdDev": stddev,
    "Actual": y_test.values
})

print(predictions_TabNet_df.head())

```

```python
generate_and_save_plots(predictions_TabNet_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/TabNet.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/TabNet.xlsx",
    model_name="TabNet",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/TabNet.xlsx",
    model_name="TabNet",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="TabNet",
    n_quantiles=10
)
```

# **Probabilistic distributions using PGBM**

```python
# Assuming best_params is obtained correctly
best_params = get_best_model_params(best_scores_autosampler, 'PGBM')

# Remove any conflicting parameters (safe-guard)
for key in ['Dist', 'Score']:
    best_params.pop(key, None)

# -------------------------
# Train PGBM as BASE model
# -------------------------
pgbm_model = PGBM()

def mseloss_objective(yhat, y, sample_weight=None):
    if not torch.is_tensor(yhat):
        yhat = torch.from_numpy(np.array(yhat)).float()
    if not torch.is_tensor(y):
        y = torch.from_numpy(np.array(y)).float()
    gradient = yhat - y
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    if not torch.is_tensor(yhat):
        yhat = torch.from_numpy(np.array(yhat)).float()
    if not torch.is_tensor(y):
        y = torch.from_numpy(np.array(y)).float()
    return torch.sqrt(torch.mean((yhat - y) ** 2))

X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)

pgbm_model.train(
    (X_train_np, y_train_np),
    objective=mseloss_objective,
    metric=rmseloss_metric,
    params=best_params
)

# -------------------------
# 🔧 NUMPY-SAFE WRAPPER
# -------------------------
class PGBMWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        y_pred = self.model.predict(X)
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        return y_pred

pgbm_wrapped = PGBMWrapper(pgbm_model)

# -------------------------
# CARD wrapper on top
# -------------------------
card_model = CARDRegressor(
    base_model=pgbm_wrapped,
    hidden_dim=128,
    T=50,
    epochs=200,
    n_samples=100
)

# Fit CARD (NOW SAFE)
card_model.fit(X_train_np, y_train_np)

# CARD predictions
mean_pred, std_pred = card_model.predict(X_test_np)

# Create prediction DataFrame
predictions_PGBM_df = pd.DataFrame({
    "Mean": mean_pred,
    "StdDev": std_pred,
    "Actual": y_test_np.ravel()
})

# Calculate and print MSE
mse = mean_squared_error(y_test_np, mean_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Preview results
print(predictions_PGBM_df.head())
```

```python
generate_and_save_plots(predictions_PGBM_df, y_test, "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_quantiles=10
)
```

# **Matrix Evaulation**

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

# Create a Pandas Excel writer using XlsxWriter as the engine.
folder_path = "/content/drive/MyDrive/Regression/pile_uncertainty_analysis"
excel_path = os.path.join(folder_path, "/content/drive/MyDrive/Regression/pile_uncertainty_analysis/probabilistic_distribution_card/Matrix Evaluation.xlsx")
with pd.ExcelWriter(_ensure_parent_dir(excel_path), engine='xlsxwriter') as writer:
    plot_metrics_and_save_to_excel(writer, predictions_NGB_df, y_test, 'NGBoost')
    plot_metrics_and_save_to_excel(writer, predictions_PGBM_df, y_test, 'PGBM')
    plot_metrics_and_save_to_excel(writer, predictions_LGBM_df, y_test, 'LGBM')
    plot_metrics_and_save_to_excel(writer, predictions_XGB_df, y_test, 'XGBoost')
    plot_metrics_and_save_to_excel(writer, predictions_TabNet_df, y_test, "TabNet")
    plot_metrics_and_save_to_excel(writer, predictions_HGBR_df, y_test, "HGBM")
    plot_metrics_and_save_to_excel(writer, predictions_CatBoost_df, y_test, "CatBoost")
    plot_metrics_and_save_to_excel(writer, predictions_GPBoost_df, y_test, "GPBoost")
    plot_metrics_and_save_to_excel(writer, predictions_GBR_df, y_test, "GradientBoosting")
```

