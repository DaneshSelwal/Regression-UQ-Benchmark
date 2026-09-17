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
# Go to find & replace button and replace (water_quality_ewqi_uncertainty_analysis) with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (str) with actual data label name.
```

```python
feature_names = ['pH', 'TDS', 'EC', 'TH', 'ca', 'mg', 'CO3', 'HCO3', 'NA', 'K', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE']
```

```python
train_data_path = "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/data/test.csv"
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
folder_path = "/content/drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis"

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
best_scores_autosampler = {('Random Forest', 'MedianPruner'): {'best_score': 497.1726066601312,
  'best_params': {'n_estimators': 500,
   'criterion': 'squared_error',
   'max_depth': 20,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 0.5,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 497.1726066601312,
  'test_rmse': 22.297367706976786,
  'test_corr_coef': 0.956734599665225,
  'pruner': 'MedianPruner'},
 ('Random Forest', 'NopPruner'): {'best_score': 496.77749759305095,
  'best_params': {'n_estimators': 500,
   'criterion': 'squared_error',
   'max_depth': 10,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 0.5,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 496.77749759305095,
  'test_rmse': 22.288505952464625,
  'test_corr_coef': 0.956804497169129,
  'pruner': 'NopPruner'},
 ('Random Forest', 'PatientPruner'): {'best_score': 496.754675557273,
  'best_params': {'n_estimators': 500,
   'criterion': 'squared_error',
   'max_depth': None,
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 0.5,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01},
  'test_mse': 496.754675557273,
  'test_rmse': 22.28799397786335,
  'test_corr_coef': 0.9568118135387959,
  'pruner': 'PatientPruner'},
 ('Random Forest', 'PercentilePruner'): {'best_score': 502.91913355234993,
  'best_params': {'n_estimators': 700,
   'criterion': 'friedman_mse',
   'max_depth': 30,
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 0.5,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.1},
  'test_mse': 502.91913355234993,
  'test_rmse': 22.42585859119668,
  'test_corr_coef': 0.956757212273653,
  'pruner': 'PercentilePruner'},
 ('Random Forest',
  'SuccessiveHalvingPruner'): {'best_score': 501.77203003955043, 'best_params': {'n_estimators': 300,
   'criterion': 'squared_error',
   'max_depth': 10,
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01}, 'test_mse': 501.77203003955043, 'test_rmse': 22.400268526059023, 'test_corr_coef': 0.9577060428217671, 'pruner': 'SuccessiveHalvingPruner'},
 ('Random Forest', 'HyperbandPruner'): {'best_score': 505.53942164338036,
  'best_params': {'n_estimators': 700,
   'criterion': 'squared_error',
   'max_depth': 30,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.2,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 505.53942164338036,
  'test_rmse': 22.484203824983005,
  'test_corr_coef': 0.9571879877850138,
  'pruner': 'HyperbandPruner'},
 ('Random Forest', 'ThresholdPruner'): {'best_score': 491.42244829819947,
  'best_params': {'n_estimators': 200,
   'criterion': 'squared_error',
   'max_depth': None,
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01},
  'test_mse': 491.42244829819947,
  'test_rmse': 22.168050169065378,
  'test_corr_coef': 0.9565528844994599,
  'pruner': 'ThresholdPruner'},
 ('Random Forest', 'WilcoxonPruner'): {'best_score': 491.0800216333908,
  'best_params': {'n_estimators': 200,
   'criterion': 'squared_error',
   'max_depth': 20,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 491.0800216333908,
  'test_rmse': 22.160325395476278,
  'test_corr_coef': 0.9566343173406529,
  'pruner': 'WilcoxonPruner'},
 ('Gradient Boosting', 'MedianPruner'): {'best_score': 160.39894303359338,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 500,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 160.39894303359338,
  'test_rmse': 12.664870430983232,
  'test_corr_coef': 0.9786294970853713,
  'pruner': 'MedianPruner'},
 ('Gradient Boosting', 'NopPruner'): {'best_score': 154.88069942037498,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 700,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
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
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 154.88069942037498,
  'test_rmse': 12.445107449129356,
  'test_corr_coef': 0.9774780717330619,
  'pruner': 'NopPruner'},
 ('Gradient Boosting', 'PatientPruner'): {'best_score': 319.4207439495004,
  'best_params': {'loss': 'quantile',
   'learning_rate': 0.2,
   'n_estimators': 300,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 3,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.01},
  'test_mse': 319.4207439495004,
  'test_rmse': 17.87234578754284,
  'test_corr_coef': 0.9656686397927369,
  'pruner': 'PatientPruner'},
 ('Gradient Boosting', 'PercentilePruner'): {'best_score': 167.44624369486803,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 500,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 10,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.01,
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
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 167.44624369486803,
  'test_rmse': 12.940102151639609,
  'test_corr_coef': 0.9783249697608657,
  'pruner': 'PercentilePruner'},
 ('Gradient Boosting',
  'SuccessiveHalvingPruner'): {'best_score': 209.05660592510384, 'best_params': {'loss': 'huber',
   'learning_rate': 0.05,
   'n_estimators': 300,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.001,
   'ccp_alpha': 0.0}, 'test_mse': 209.05660592510384, 'test_rmse': 14.458789919115079, 'test_corr_coef': 0.9711002789653699, 'pruner': 'SuccessiveHalvingPruner'},
 ('Gradient Boosting', 'HyperbandPruner'): {'best_score': 166.98244781777268,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.1,
   'n_estimators': 300,
   'subsample': 0.7,
   'criterion': 'squared_error',
   'min_samples_split': 10,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 3,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 20,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 166.98244781777268,
  'test_rmse': 12.92216885115547,
  'test_corr_coef': 0.9753948831674288,
  'pruner': 'HyperbandPruner'},
 ('Gradient Boosting', 'ThresholdPruner'): {'best_score': 224.603332352986,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.2,
   'n_estimators': 500,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 10,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
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
   'ccp_alpha': 0.01},
  'test_mse': 224.603332352986,
  'test_rmse': 14.98677191235611,
  'test_corr_coef': 0.9691900597572881,
  'pruner': 'ThresholdPruner'},
 ('Gradient Boosting', 'WilcoxonPruner'): {'best_score': 193.60489853112514,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.1,
   'n_estimators': 500,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 2,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 7,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 193.60489853112514,
  'test_rmse': 13.914197732213134,
  'test_corr_coef': 0.9772124930007884,
  'pruner': 'WilcoxonPruner'},
 ('XGBoost', 'MedianPruner'): {'best_score': 210.22785405443477,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 0.5,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 1,
   'reg_lambda': 1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 210.22785405443477,
  'test_rmse': 14.499236326594403,
  'test_corr_coef': 0.9716039951720425,
  'pruner': 'MedianPruner'},
 ('XGBoost', 'NopPruner'): {'best_score': 203.78744598873558,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 203.78744598873558,
  'test_rmse': 14.275414039135102,
  'test_corr_coef': 0.972214002001108,
  'pruner': 'NopPruner'},
 ('XGBoost', 'PatientPruner'): {'best_score': 232.96485275333296,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.1,
   'max_depth': 3,
   'min_child_weight': 3,
   'gamma': 0.5,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 232.96485275333296,
  'test_rmse': 15.263186192709993,
  'test_corr_coef': 0.9709826608345447,
  'pruner': 'PatientPruner'},
 ('XGBoost', 'PercentilePruner'): {'best_score': 216.04391248784393,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.6,
   'colsample_bytree': 0.5,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 216.04391248784393,
  'test_rmse': 14.69843231395253,
  'test_corr_coef': 0.9712352500641135,
  'pruner': 'PercentilePruner'},
 ('XGBoost', 'SuccessiveHalvingPruner'): {'best_score': 176.68759651311868,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.1,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 176.68759651311868,
  'test_rmse': 13.292388668449275,
  'test_corr_coef': 0.9738713880593469,
  'pruner': 'SuccessiveHalvingPruner'},
 ('XGBoost', 'HyperbandPruner'): {'best_score': 201.84220181861593,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.15,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 201.84220181861593,
  'test_rmse': 14.207117998335058,
  'test_corr_coef': 0.9728221055227199,
  'pruner': 'HyperbandPruner'},
 ('XGBoost', 'ThresholdPruner'): {'best_score': 166.87412984476197,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 166.87412984476197,
  'test_rmse': 12.917977002795832,
  'test_corr_coef': 0.9773098708759632,
  'pruner': 'ThresholdPruner'},
 ('XGBoost', 'WilcoxonPruner'): {'best_score': 201.61436815384675,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_depth': 3,
   'min_child_weight': 1,
   'gamma': 0.5,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 201.61436815384675,
  'test_rmse': 14.199097441522357,
  'test_corr_coef': 0.973330665996567,
  'pruner': 'WilcoxonPruner'},
 ('LightGBM', 'MedianPruner'): {'best_score': 213.1733412037077,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': 5,
   'min_child_samples': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 213.1733412037077,
  'test_rmse': 14.600456883389223,
  'test_corr_coef': 0.9684949027738449,
  'pruner': 'MedianPruner'},
 ('LightGBM', 'NopPruner'): {'best_score': 298.91676453032903,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': 3,
   'min_child_samples': 5,
   'subsample': 1.0,
   'colsample_bytree': 0.9,
   'reg_alpha': 1,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 298.91676453032903,
  'test_rmse': 17.289209482516227,
  'test_corr_coef': 0.9605307426357604,
  'pruner': 'NopPruner'},
 ('LightGBM', 'PatientPruner'): {'best_score': 217.41376077168775,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'reg_alpha': 1,
   'reg_lambda': 1,
   'min_child_weight': 0.001,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 217.41376077168775,
  'test_rmse': 14.744957130208542,
  'test_corr_coef': 0.9716477463223928,
  'pruner': 'PatientPruner'},
 ('LightGBM', 'PercentilePruner'): {'best_score': 223.36544632845784,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': 5,
   'min_child_samples': 1,
   'subsample': 0.6,
   'colsample_bytree': 1.0,
   'reg_alpha': 1,
   'reg_lambda': 1,
   'min_child_weight': 0.1,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 223.36544632845784,
  'test_rmse': 14.945415562253793,
  'test_corr_coef': 0.9701627442637851,
  'pruner': 'PercentilePruner'},
 ('LightGBM', 'SuccessiveHalvingPruner'): {'best_score': 298.7053099814435,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'num_leaves': 63,
   'max_depth': 3,
   'min_child_samples': 5,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'bagging_freq': 0,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 298.7053099814435,
  'test_rmse': 17.28309318326565,
  'test_corr_coef': 0.9599620787142591,
  'pruner': 'SuccessiveHalvingPruner'},
 ('LightGBM', 'HyperbandPruner'): {'best_score': 240.51515926452262,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': 3,
   'min_child_samples': 1,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 240.51515926452262,
  'test_rmse': 15.508551165873705,
  'test_corr_coef': 0.9714116638050261,
  'pruner': 'HyperbandPruner'},
 ('LightGBM', 'ThresholdPruner'): {'best_score': 306.9617775986046,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.15,
   'num_leaves': 15,
   'max_depth': 3,
   'min_child_samples': 5,
   'subsample': 0.9,
   'colsample_bytree': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 0,
   'min_child_weight': 1e-05,
   'bagging_freq': 0,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 306.9617775986046,
  'test_rmse': 17.520324700147672,
  'test_corr_coef': 0.9584020466316027,
  'pruner': 'ThresholdPruner'},
 ('LightGBM', 'WilcoxonPruner'): {'best_score': 244.83318058757223,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 1,
   'subsample': 0.6,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.01,
   'reg_lambda': 0,
   'min_child_weight': 1e-05,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 244.83318058757223,
  'test_rmse': 15.64714608443253,
  'test_corr_coef': 0.9658566522174811,
  'pruner': 'WilcoxonPruner'},
 ('GPBoost', 'MedianPruner'): {'best_score': 293.0101251729265,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 15,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 293.0101251729265,
  'test_rmse': 17.117538525527742,
  'test_corr_coef': 0.9612550530540446,
  'pruner': 'MedianPruner'},
 ('GPBoost', 'NopPruner'): {'best_score': 302.95261509197184,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.1,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 5,
   'subsample': 0.9,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.5,
   'reg_lambda': 0.1,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 302.95261509197184,
  'test_rmse': 17.40553403639118,
  'test_corr_coef': 0.9601419867851486,
  'pruner': 'NopPruner'},
 ('GPBoost', 'PatientPruner'): {'best_score': 299.07160436500203,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 0.9,
   'reg_alpha': 1.0,
   'reg_lambda': 0.1,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 299.07160436500203,
  'test_rmse': 17.29368683551897,
  'test_corr_coef': 0.9608235696686525,
  'pruner': 'PatientPruner'},
 ('GPBoost', 'PercentilePruner'): {'best_score': 302.4539295475795,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.15,
   'max_depth': 3,
   'num_leaves': 63,
   'min_child_samples': 5,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'reg_alpha': 1.0,
   'reg_lambda': 0.5,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 302.4539295475795,
  'test_rmse': 17.391202648108596,
  'test_corr_coef': 0.9600726743566298,
  'pruner': 'PercentilePruner'},
 ('GPBoost', 'SuccessiveHalvingPruner'): {'best_score': 302.95261509197184,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.1,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.6,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.5,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 302.95261509197184,
  'test_rmse': 17.40553403639118,
  'test_corr_coef': 0.9601419867851486,
  'pruner': 'SuccessiveHalvingPruner'},
 ('GPBoost', 'HyperbandPruner'): {'best_score': 298.96846756091594,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 15,
   'min_child_samples': 5,
   'subsample': 1.0,
   'colsample_bytree': 0.7,
   'reg_alpha': 1.0,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 298.96846756091594,
  'test_rmse': 17.290704657732025,
  'test_corr_coef': 0.9597344567776984,
  'pruner': 'HyperbandPruner'},
 ('GPBoost', 'ThresholdPruner'): {'best_score': 303.69100928873127,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.6,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.5,
   'reg_lambda': 0,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 303.69100928873127,
  'test_rmse': 17.426732605073486,
  'test_corr_coef': 0.9602594834479129,
  'pruner': 'ThresholdPruner'},
 ('GPBoost', 'WilcoxonPruner'): {'best_score': 302.69736998457273,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 302.69736998457273,
  'test_rmse': 17.398200193829613,
  'test_corr_coef': 0.9598410282782289,
  'pruner': 'WilcoxonPruner'},
 ('CatBoost', 'MedianPruner'): {'best_score': 368.5205933957056,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.1,
   'depth': 4,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 20,
   'rsm': 0.8,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 368.5205933957056,
  'test_rmse': 19.196890201168145,
  'test_corr_coef': 0.9564404896870229,
  'pruner': 'MedianPruner'},
 ('CatBoost', 'NopPruner'): {'best_score': 347.4429012205738,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.03,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 5,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 347.4429012205738,
  'test_rmse': 18.63982031084457,
  'test_corr_coef': 0.9614771149420601,
  'pruner': 'NopPruner'},
 ('CatBoost', 'PatientPruner'): {'best_score': 370.10393301836245,
  'best_params': {'iterations': 500,
   'learning_rate': 0.1,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 5,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 370.10393301836245,
  'test_rmse': 19.23808548214615,
  'test_corr_coef': 0.9594743814447798,
  'pruner': 'PatientPruner'},
 ('CatBoost', 'PercentilePruner'): {'best_score': 354.04068707098565,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.05,
   'depth': 4,
   'l2_leaf_reg': 7,
   'border_count': 64,
   'min_data_in_leaf': 10,
   'rsm': 0.8,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 354.04068707098565,
  'test_rmse': 18.815968937872576,
  'test_corr_coef': 0.9599972705368262,
  'pruner': 'PercentilePruner'},
 ('CatBoost', 'SuccessiveHalvingPruner'): {'best_score': 347.4429012205738,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.03,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 347.4429012205738,
  'test_rmse': 18.63982031084457,
  'test_corr_coef': 0.9614771149420601,
  'pruner': 'SuccessiveHalvingPruner'},
 ('CatBoost', 'HyperbandPruner'): {'best_score': 347.4429012205738,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.03,
   'depth': 4,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 20,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 347.4429012205738,
  'test_rmse': 18.63982031084457,
  'test_corr_coef': 0.9614771149420601,
  'pruner': 'HyperbandPruner'},
 ('CatBoost', 'ThresholdPruner'): {'best_score': 368.5205933957056,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.1,
   'depth': 4,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 10,
   'rsm': 0.8,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 368.5205933957056,
  'test_rmse': 19.196890201168145,
  'test_corr_coef': 0.9564404896870229,
  'pruner': 'ThresholdPruner'},
 ('CatBoost', 'WilcoxonPruner'): {'best_score': 365.7968617795807,
  'best_params': {'iterations': 500,
   'learning_rate': 0.05,
   'depth': 4,
   'l2_leaf_reg': 3,
   'border_count': 32,
   'min_data_in_leaf': 1,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 365.7968617795807,
  'test_rmse': 19.125816630397267,
  'test_corr_coef': 0.9606103881972815,
  'pruner': 'WilcoxonPruner'},
 ('NGBoost', 'MedianPruner'): {'best_score': 179.09111962798949,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 1.0,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 179.09111962798949,
  'test_rmse': 13.382493027384303,
  'test_corr_coef': 0.9757209713284791,
  'pruner': 'MedianPruner'},
 ('NGBoost', 'NopPruner'): {'best_score': 175.50185272796932,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 1.0,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 175.50185272796932,
  'test_rmse': 13.247711226018225,
  'test_corr_coef': 0.9761834419659058,
  'pruner': 'NopPruner'},
 ('NGBoost', 'PatientPruner'): {'best_score': 174.52057835042388,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 1.0,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 174.52057835042388,
  'test_rmse': 13.210623692711252,
  'test_corr_coef': 0.9765486479417845,
  'pruner': 'PatientPruner'},
 ('NGBoost', 'PercentilePruner'): {'best_score': 194.85172141222253,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 1.0,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 194.85172141222253,
  'test_rmse': 13.95892980898688,
  'test_corr_coef': 0.9765801058898063,
  'pruner': 'PercentilePruner'},
 ('NGBoost', 'SuccessiveHalvingPruner'): {'best_score': 186.89926194076523,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 186.89926194076523,
  'test_rmse': 13.671110486744126,
  'test_corr_coef': 0.9783926643154599,
  'pruner': 'SuccessiveHalvingPruner'},
 ('NGBoost', 'HyperbandPruner'): {'best_score': 177.91122355138324,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 1.0,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 177.91122355138324,
  'test_rmse': 13.338336611113968,
  'test_corr_coef': 0.9765296263458519,
  'pruner': 'HyperbandPruner'},
 ('NGBoost', 'ThresholdPruner'): {'best_score': 155.9848550761906,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 155.9848550761906,
  'test_rmse': 12.489389699908903,
  'test_corr_coef': 0.9803930711650164,
  'pruner': 'ThresholdPruner'},
 ('NGBoost', 'WilcoxonPruner'): {'best_score': 175.75747585355253,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 175.75747585355253,
  'test_rmse': 13.25735553772141,
  'test_corr_coef': 0.9762639080221905,
  'pruner': 'WilcoxonPruner'},
 ('TabNet', 'MedianPruner'): {'best_score': 740.3845011032141,
  'best_params': {'n_d': 16,
   'n_a': 8,
   'n_steps': 7,
   'gamma': 1.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 3,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 740.3845011032141,
  'test_rmse': 27.2100073705101,
  'test_corr_coef': 0.9230741247197396,
  'pruner': 'MedianPruner'},
 ('TabNet', 'NopPruner'): {'best_score': 661.9252456377329,
  'best_params': {'n_d': 32,
   'n_a': 8,
   'n_steps': 7,
   'gamma': 1.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 2,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 661.9252456377329,
  'test_rmse': 25.727907914125723,
  'test_corr_coef': 0.9375049301933271,
  'pruner': 'NopPruner'},
 ('TabNet', 'PatientPruner'): {'best_score': 561.1656802691135,
  'best_params': {'n_d': 8,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 561.1656802691135,
  'test_rmse': 23.688935819684122,
  'test_corr_coef': 0.9318886782102657,
  'pruner': 'PatientPruner'},
 ('TabNet', 'PercentilePruner'): {'best_score': 561.1656802691135,
  'best_params': {'n_d': 8,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 561.1656802691135,
  'test_rmse': 23.688935819684122,
  'test_corr_coef': 0.9318886782102657,
  'pruner': 'PercentilePruner'},
 ('TabNet', 'SuccessiveHalvingPruner'): {'best_score': 884.7810634293032,
  'best_params': {'n_d': 32,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 2.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 884.7810634293032,
  'test_rmse': 29.745269597522615,
  'test_corr_coef': 0.8969667948482032,
  'pruner': 'SuccessiveHalvingPruner'},
 ('TabNet', 'HyperbandPruner'): {'best_score': 788.668537544431,
  'best_params': {'n_d': 32,
   'n_a': 16,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 788.668537544431,
  'test_rmse': 28.08324300262402,
  'test_corr_coef': 0.9071967149047229,
  'pruner': 'HyperbandPruner'},
 ('TabNet', 'ThresholdPruner'): {'best_score': 609.9543017765326,
  'best_params': {'n_d': 32,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 2,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 609.9543017765326,
  'test_rmse': 24.69725291963729,
  'test_corr_coef': 0.9363069902713822,
  'pruner': 'ThresholdPruner'},
 ('TabNet', 'WilcoxonPruner'): {'best_score': 666.6390125869158,
  'best_params': {'n_d': 16,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 2.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 666.6390125869158,
  'test_rmse': 25.819353450210865,
  'test_corr_coef': 0.9148062364563666,
  'pruner': 'WilcoxonPruner'},
 ('HistGradientBoosting', 'MedianPruner'): {'best_score': 308.2494075545725,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 500,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 1.0,
   'max_bins': 255,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 308.2494075545725,
  'test_rmse': 17.55703299406174,
  'test_corr_coef': 0.9571023046674804,
  'pruner': 'MedianPruner'},
 ('HistGradientBoosting', 'NopPruner'): {'best_score': 308.8182699911407,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 308.8182699911407,
  'test_rmse': 17.57322594150376,
  'test_corr_coef': 0.9573002218442682,
  'pruner': 'NopPruner'},
 ('HistGradientBoosting', 'PatientPruner'): {'best_score': 305.50458250552157,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 0.5,
   'max_bins': 128,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 305.50458250552157,
  'test_rmse': 17.478689381802102,
  'test_corr_coef': 0.957444692058178,
  'pruner': 'PatientPruner'},
 ('HistGradientBoosting',
  'PercentilePruner'): {'best_score': 307.14533474906676, 'best_params': {'learning_rate': 0.1,
   'max_iter': 400,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.5,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 307.14533474906676, 'test_rmse': 17.52556232333407, 'test_corr_coef': 0.9571084111490522, 'pruner': 'PercentilePruner'},
 ('HistGradientBoosting',
  'SuccessiveHalvingPruner'): {'best_score': 305.50458250552157, 'best_params': {'learning_rate': 0.1,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 31,
   'l2_regularization': 0.5,
   'max_bins': 128,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 305.50458250552157, 'test_rmse': 17.478689381802102, 'test_corr_coef': 0.957444692058178, 'pruner': 'SuccessiveHalvingPruner'},
 ('HistGradientBoosting', 'HyperbandPruner'): {'best_score': 308.2494075545725,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 500,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 31,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 308.2494075545725,
  'test_rmse': 17.55703299406174,
  'test_corr_coef': 0.9571023046674804,
  'pruner': 'HyperbandPruner'},
 ('HistGradientBoosting',
  'ThresholdPruner'): {'best_score': 307.36256812194904, 'best_params': {'learning_rate': 0.1,
   'max_iter': 500,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 0.5,
   'max_bins': 128,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 307.36256812194904, 'test_rmse': 17.53175884279581, 'test_corr_coef': 0.9570747462258868, 'pruner': 'ThresholdPruner'},
 ('HistGradientBoosting', 'WilcoxonPruner'): {'best_score': 320.6122621869927,
  'best_params': {'learning_rate': 0.05,
   'max_iter': 300,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': 63,
   'l2_regularization': 1.0,
   'max_bins': 255,
   'early_stopping': False,
   'validation_fraction': 0.2,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 320.6122621869927,
  'test_rmse': 17.905648890419826,
  'test_corr_coef': 0.9564377958343118,
  'pruner': 'WilcoxonPruner'},
 ('PGBM', 'MedianPruner'): {'best_score': 320.29442429575573,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.15,
   'max_leaves': 41,
   'min_split_gain': 0.5,
   'reg_lambda': 5.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 5,
   'max_bin': 128,
   'distribution': 'studentt'},
  'test_mse': 320.29442429575573,
  'test_rmse': 17.89677133719252,
  'test_corr_coef': 0.9552618710941777,
  'pruner': 'MedianPruner'},
 ('PGBM', 'NopPruner'): {'best_score': 225.94092773243457,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.1,
   'max_leaves': 33,
   'min_split_gain': 0.1,
   'reg_lambda': 10.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 225.94092773243457,
  'test_rmse': 15.031331535577099,
  'test_corr_coef': 0.9674460933337918,
  'pruner': 'NopPruner'},
 ('PGBM', 'PatientPruner'): {'best_score': 252.11330094737534,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_leaves': 60,
   'min_split_gain': 0.0,
   'reg_lambda': 0.1,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'laplace'},
  'test_mse': 252.11330094737534,
  'test_rmse': 15.878076109761388,
  'test_corr_coef': 0.9664375924887434,
  'pruner': 'PatientPruner'},
 ('PGBM', 'PercentilePruner'): {'best_score': 204.6617165350584,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_leaves': 30,
   'min_split_gain': 0.1,
   'reg_lambda': 0.1,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 3,
   'max_bin': 128,
   'distribution': 'studentt'},
  'test_mse': 204.6617165350584,
  'test_rmse': 14.30600281472985,
  'test_corr_coef': 0.9695547928398048,
  'pruner': 'PercentilePruner'},
 ('PGBM', 'SuccessiveHalvingPruner'): {'best_score': 203.9210167801792,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.1,
   'max_leaves': 23,
   'min_split_gain': 0.5,
   'reg_lambda': 1.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'laplace'},
  'test_mse': 203.9210167801792,
  'test_rmse': 14.280091623661916,
  'test_corr_coef': 0.9723370733600075,
  'pruner': 'SuccessiveHalvingPruner'},
 ('PGBM', 'HyperbandPruner'): {'best_score': 207.01557595508848,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_leaves': 28,
   'min_split_gain': 1.0,
   'reg_lambda': 0.1,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 207.01557595508848,
  'test_rmse': 14.388035861613929,
  'test_corr_coef': 0.9697329013099916,
  'pruner': 'HyperbandPruner'},
 ('PGBM', 'ThresholdPruner'): {'best_score': 218.35010707891826,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'max_leaves': 40,
   'min_split_gain': 0.0,
   'reg_lambda': 5.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 218.35010707891826,
  'test_rmse': 14.77667442555727,
  'test_corr_coef': 0.9684724623182985,
  'pruner': 'ThresholdPruner'},
 ('PGBM', 'WilcoxonPruner'): {'best_score': 211.3068684747127,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_leaves': 44,
   'min_split_gain': 0.1,
   'reg_lambda': 5.0,
   'feature_fraction': 0.7,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 211.3068684747127,
  'test_rmse': 14.53639805710867,
  'test_corr_coef': 0.9710731342948299,
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
generate_and_save_plots(predictions_LGBM_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/LightGBM.xlsx",
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
generate_and_save_plots(predictions_XGB_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/XGBoost.xlsx",
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
generate_and_save_plots(predictions_GPBoost_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/GPBoost.xlsx",
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
generate_and_save_plots(predictions_NGB_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/NGBoost.xlsx",
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
generate_and_save_plots(predictions_GBR_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx",
    model_name="Gradient Boosting",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx",
    model_name="GBoosting",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/Gradient Boosting.xlsx",
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
generate_and_save_plots(predictions_CatBoost_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="CatBoost",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="CatBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
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
generate_and_save_plots(predictions_HGBR_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx",
    model_name="HGBR",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx",
    model_name="HGBR",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/HGBM.xlsx",
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
generate_and_save_plots(predictions_TabNet_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/TabNet.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/TabNet.xlsx",
    model_name="TabNet",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/TabNet.xlsx",
    model_name="TabNet",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/CatBoost.xlsx",
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
generate_and_save_plots(predictions_PGBM_df, y_test, "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/PGBM.xlsx",
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
folder_path = "/content/drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis"
excel_path = os.path.join(folder_path, "/content/drive/MyDrive/Regression/water_quality_ewqi_uncertainty_analysis/probabilistic_distribution_card/Matrix Evaluation.xlsx")
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

