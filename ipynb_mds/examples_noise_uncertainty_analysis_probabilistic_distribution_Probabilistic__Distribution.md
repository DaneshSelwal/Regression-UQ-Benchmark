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
```

```python
# Go to find & replace button and replace (noise_uncertainty_analysis) with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (str) with actual data label name.
```

```python
feature_names = ['A', 'B', 'C', 'D', 'O']
```

```python
train_data_path = "./drive/MyDrive/Regression/noise_uncertainty_analysis/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/noise_uncertainty_analysis/data/test.csv"
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
folder_path = "/content/drive/MyDrive/Regression/noise_uncertainty_analysis"

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

# **Tuned Parameters**

```python
best_scores_autosampler = {('Random Forest', 'MedianPruner'): {'best_score': 13.386480292958343,
  'best_params': {'n_estimators': 100,
   'criterion': 'absolute_error',
   'max_depth': 40,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 13.386480292958343,
  'test_rmse': 3.6587539262648345,
  'test_corr_coef': 0.9703857201117179,
  'pruner': 'MedianPruner'},
 ('Random Forest', 'NopPruner'): {'best_score': 13.031080279414615,
  'best_params': {'n_estimators': 700,
   'criterion': 'friedman_mse',
   'max_depth': 20,
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.1,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01},
  'test_mse': 13.031080279414615,
  'test_rmse': 3.6098587616989413,
  'test_corr_coef': 0.9707198566155737,
  'pruner': 'NopPruner'},
 ('Random Forest', 'PatientPruner'): {'best_score': 14.299335373065158,
  'best_params': {'n_estimators': 700,
   'criterion': 'squared_error',
   'max_depth': 10,
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 'log2',
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 14.299335373065158,
  'test_rmse': 3.7814462012654837,
  'test_corr_coef': 0.9686598451590471,
  'pruner': 'PatientPruner'},
 ('Random Forest', 'PercentilePruner'): {'best_score': 13.002555971496559,
  'best_params': {'n_estimators': 700,
   'criterion': 'squared_error',
   'max_depth': 30,
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01},
  'test_mse': 13.002555971496559,
  'test_rmse': 3.6059057075160132,
  'test_corr_coef': 0.9707614120536777,
  'pruner': 'PercentilePruner'},
 ('Random Forest',
  'SuccessiveHalvingPruner'): {'best_score': 13.242745794750004, 'best_params': {'n_estimators': 100,
   'criterion': 'absolute_error',
   'max_depth': 40,
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001}, 'test_mse': 13.242745794750004, 'test_rmse': 3.6390583664940035, 'test_corr_coef': 0.9705533435061875, 'pruner': 'SuccessiveHalvingPruner'},
 ('Random Forest', 'HyperbandPruner'): {'best_score': 13.032712146115928,
  'best_params': {'n_estimators': 700,
   'criterion': 'friedman_mse',
   'max_depth': 10,
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 13.032712146115928,
  'test_rmse': 3.6100847837849916,
  'test_corr_coef': 0.9707060136472022,
  'pruner': 'HyperbandPruner'},
 ('Random Forest', 'ThresholdPruner'): {'best_score': 13.098183337721803,
  'best_params': {'n_estimators': 300,
   'criterion': 'squared_error',
   'max_depth': 20,
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 13.098183337721803,
  'test_rmse': 3.619141243129619,
  'test_corr_coef': 0.9708556688744011,
  'pruner': 'ThresholdPruner'},
 ('Random Forest', 'WilcoxonPruner'): {'best_score': 13.05436043571211,
  'best_params': {'n_estimators': 300,
   'criterion': 'friedman_mse',
   'max_depth': 30,
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 13.05436043571211,
  'test_rmse': 3.613081847358583,
  'test_corr_coef': 0.970902748067831,
  'pruner': 'WilcoxonPruner'},
 ('Gradient Boosting', 'MedianPruner'): {'best_score': 10.829304123572543,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.1,
   'n_estimators': 300,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.001,
   'ccp_alpha': 0.01},
  'test_mse': 10.829304123572543,
  'test_rmse': 3.29079080519752,
  'test_corr_coef': 0.9755020526141754,
  'pruner': 'MedianPruner'},
 ('Gradient Boosting', 'NopPruner'): {'best_score': 11.661624240805503,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.1,
   'n_estimators': 500,
   'subsample': 0.7,
   'criterion': 'squared_error',
   'min_samples_split': 2,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 5,
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
   'tol': 0.001,
   'ccp_alpha': 0.0},
  'test_mse': 11.661624240805503,
  'test_rmse': 3.414912039980752,
  'test_corr_coef': 0.9744114769026063,
  'pruner': 'NopPruner'},
 ('Gradient Boosting', 'PatientPruner'): {'best_score': 9.84804317748675,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.2,
   'n_estimators': 200,
   'subsample': 1.0,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 5,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 9.84804317748675,
  'test_rmse': 3.138159202062054,
  'test_corr_coef': 0.9778974508000391,
  'pruner': 'PatientPruner'},
 ('Gradient Boosting', 'PercentilePruner'): {'best_score': 11.728091600997796,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.1,
   'n_estimators': 100,
   'subsample': 1.0,
   'criterion': 'friedman_mse',
   'min_samples_split': 0.01,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 5,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 0.5,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.01},
  'test_mse': 11.728091600997796,
  'test_rmse': 3.4246301407594073,
  'test_corr_coef': 0.9735794434938067,
  'pruner': 'PercentilePruner'},
 ('Gradient Boosting',
  'SuccessiveHalvingPruner'): {'best_score': 11.402302876275247, 'best_params': {'loss': 'huber',
   'learning_rate': 0.2,
   'n_estimators': 500,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 0.01,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 5,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.0001,
   'ccp_alpha': 0.01}, 'test_mse': 11.402302876275247, 'test_rmse': 3.37672961255047, 'test_corr_coef': 0.9771847733785521, 'pruner': 'SuccessiveHalvingPruner'},
 ('Gradient Boosting', 'HyperbandPruner'): {'best_score': 11.790788566395653,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.05,
   'n_estimators': 100,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 0.01,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 7,
   'min_impurity_decrease': 0.1,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 11.790788566395653,
  'test_rmse': 3.433771769701017,
  'test_corr_coef': 0.9735327051071399,
  'pruner': 'HyperbandPruner'},
 ('Gradient Boosting', 'ThresholdPruner'): {'best_score': 10.826701502045989,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.1,
   'n_estimators': 200,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 10,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 10.826701502045989,
  'test_rmse': 3.2903953412995812,
  'test_corr_coef': 0.9760380316337317,
  'pruner': 'ThresholdPruner'},
 ('Gradient Boosting', 'WilcoxonPruner'): {'best_score': 11.261690705814173,
  'best_params': {'loss': 'squared_error',
   'learning_rate': 0.01,
   'n_estimators': 500,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 10,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': 10,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 11.261690705814173,
  'test_rmse': 3.3558442612573924,
  'test_corr_coef': 0.9762402043381541,
  'pruner': 'WilcoxonPruner'},
 ('XGBoost', 'MedianPruner'): {'best_score': 12.971329740251232,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.05,
   'max_depth': 5,
   'min_child_weight': 1,
   'gamma': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 12.971329740251232,
  'test_rmse': 3.6015732312770252,
  'test_corr_coef': 0.9707032290601142,
  'pruner': 'MedianPruner'},
 ('XGBoost', 'NopPruner'): {'best_score': 12.677151527294752,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.15,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 12.677151527294752,
  'test_rmse': 3.560498775072778,
  'test_corr_coef': 0.9718695625297014,
  'pruner': 'NopPruner'},
 ('XGBoost', 'PatientPruner'): {'best_score': 12.423187943362256,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.15,
   'max_depth': 5,
   'min_child_weight': 5,
   'gamma': 0.1,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.9,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 12.423187943362256,
  'test_rmse': 3.5246543012559766,
  'test_corr_coef': 0.9718710858597605,
  'pruner': 'PatientPruner'},
 ('XGBoost', 'PercentilePruner'): {'best_score': 13.444006123809128,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 1,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 13.444006123809128,
  'test_rmse': 3.6666068951837647,
  'test_corr_coef': 0.9693491795386318,
  'pruner': 'PercentilePruner'},
 ('XGBoost', 'SuccessiveHalvingPruner'): {'best_score': 12.534815685757243,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 5,
   'min_child_weight': 3,
   'gamma': 0.1,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 12.534815685757243,
  'test_rmse': 3.540454163770129,
  'test_corr_coef': 0.9720865242064762,
  'pruner': 'SuccessiveHalvingPruner'},
 ('XGBoost', 'HyperbandPruner'): {'best_score': 13.01825860252241,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 13.01825860252241,
  'test_rmse': 3.608082399630365,
  'test_corr_coef': 0.9705306976359102,
  'pruner': 'HyperbandPruner'},
 ('XGBoost', 'ThresholdPruner'): {'best_score': 12.995629257279962,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_depth': 5,
   'min_child_weight': 5,
   'gamma': 0.5,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 12.995629257279962,
  'test_rmse': 3.6049451115488518,
  'test_corr_coef': 0.9708011364370677,
  'pruner': 'ThresholdPruner'},
 ('XGBoost', 'WilcoxonPruner'): {'best_score': 11.911728329119882,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 11.911728329119882,
  'test_rmse': 3.4513371798652015,
  'test_corr_coef': 0.9730112774860044,
  'pruner': 'WilcoxonPruner'},
 ('LightGBM', 'MedianPruner'): {'best_score': 11.465139524319431,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'num_leaves': 31,
   'max_depth': 7,
   'min_child_samples': 5,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 1e-05,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 11.465139524319431,
  'test_rmse': 3.3860211937197664,
  'test_corr_coef': 0.9741951359293826,
  'pruner': 'MedianPruner'},
 ('LightGBM', 'NopPruner'): {'best_score': 12.765266555529877,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'num_leaves': 15,
   'max_depth': 7,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.01,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 12.765266555529877,
  'test_rmse': 3.5728513200985397,
  'test_corr_coef': 0.9716795934742927,
  'pruner': 'NopPruner'},
 ('LightGBM', 'PatientPruner'): {'best_score': 10.593776675543129,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'num_leaves': 63,
   'max_depth': 5,
   'min_child_samples': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.9,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 0.001,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.593776675543129,
  'test_rmse': 3.2548082394425526,
  'test_corr_coef': 0.9759119827720704,
  'pruner': 'PatientPruner'},
 ('LightGBM', 'PercentilePruner'): {'best_score': 10.726692183195036,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 31,
   'max_depth': 7,
   'min_child_samples': 5,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 1,
   'min_child_weight': 0.01,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.726692183195036,
  'test_rmse': 3.2751629246794787,
  'test_corr_coef': 0.9755942859381874,
  'pruner': 'PercentilePruner'},
 ('LightGBM', 'SuccessiveHalvingPruner'): {'best_score': 12.012802816253567,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 31,
   'max_depth': 5,
   'min_child_samples': 10,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'bagging_freq': 0,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 12.012802816253567,
  'test_rmse': 3.465949049864058,
  'test_corr_coef': 0.9737012091768482,
  'pruner': 'SuccessiveHalvingPruner'},
 ('LightGBM', 'HyperbandPruner'): {'best_score': 11.27855307408184,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 5,
   'subsample': 0.9,
   'colsample_bytree': 0.9,
   'reg_alpha': 0,
   'reg_lambda': 0,
   'min_child_weight': 0.01,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 11.27855307408184,
  'test_rmse': 3.358355709879738,
  'test_corr_coef': 0.9750286702009904,
  'pruner': 'HyperbandPruner'},
 ('LightGBM', 'ThresholdPruner'): {'best_score': 10.309598332015783,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': 7,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'min_child_weight': 1e-05,
   'bagging_freq': 0,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.309598332015783,
  'test_rmse': 3.2108563237889958,
  'test_corr_coef': 0.977021422519045,
  'pruner': 'ThresholdPruner'},
 ('LightGBM', 'WilcoxonPruner'): {'best_score': 10.037746636623021,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.15,
   'num_leaves': 15,
   'max_depth': 3,
   'min_child_samples': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'min_child_weight': 1e-05,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.037746636623021,
  'test_rmse': 3.1682403060094764,
  'test_corr_coef': 0.9776264961015804,
  'pruner': 'WilcoxonPruner'},
 ('GPBoost', 'MedianPruner'): {'best_score': 10.901667280503618,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': -1,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0.5,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.901667280503618,
  'test_rmse': 3.3017672965403873,
  'test_corr_coef': 0.9759578752410472,
  'pruner': 'MedianPruner'},
 ('GPBoost', 'NopPruner'): {'best_score': 10.47862493907079,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': -1,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.9,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.5,
   'reg_lambda': 0.1,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.47862493907079,
  'test_rmse': 3.2370704254110367,
  'test_corr_coef': 0.9769317908648607,
  'pruner': 'NopPruner'},
 ('GPBoost', 'PatientPruner'): {'best_score': 10.684721268563525,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 5,
   'num_leaves': 63,
   'min_child_samples': 5,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 1.0,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.684721268563525,
  'test_rmse': 3.2687491902199417,
  'test_corr_coef': 0.9767022352368916,
  'pruner': 'PatientPruner'},
 ('GPBoost', 'PercentilePruner'): {'best_score': 12.272039819671177,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_depth': -1,
   'num_leaves': 15,
   'min_child_samples': 1,
   'subsample': 1.0,
   'colsample_bytree': 0.9,
   'reg_alpha': 1.0,
   'reg_lambda': 1.0,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 12.272039819671177,
  'test_rmse': 3.503147130748461,
  'test_corr_coef': 0.9720949658383836,
  'pruner': 'PercentilePruner'},
 ('GPBoost', 'SuccessiveHalvingPruner'): {'best_score': 10.638909732949003,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 5,
   'num_leaves': 15,
   'min_child_samples': 5,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 0.1,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.638909732949003,
  'test_rmse': 3.2617341603737424,
  'test_corr_coef': 0.9767142557939993,
  'pruner': 'SuccessiveHalvingPruner'},
 ('GPBoost', 'HyperbandPruner'): {'best_score': 10.866096507361197,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': -1,
   'num_leaves': 31,
   'min_child_samples': 10,
   'subsample': 1.0,
   'colsample_bytree': 0.9,
   'reg_alpha': 1.0,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.866096507361197,
  'test_rmse': 3.296376269081125,
  'test_corr_coef': 0.9753492057788247,
  'pruner': 'HyperbandPruner'},
 ('GPBoost', 'ThresholdPruner'): {'best_score': 11.503699657253586,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_depth': 7,
   'num_leaves': 15,
   'min_child_samples': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 0.5,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 11.503699657253586,
  'test_rmse': 3.391710432400382,
  'test_corr_coef': 0.9739972829080268,
  'pruner': 'ThresholdPruner'},
 ('GPBoost', 'WilcoxonPruner'): {'best_score': 10.866096507361197,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': -1,
   'num_leaves': 63,
   'min_child_samples': 10,
   'subsample': 0.6,
   'colsample_bytree': 1.0,
   'reg_alpha': 1.0,
   'reg_lambda': 0.1,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 10.866096507361197,
  'test_rmse': 3.296376269081125,
  'test_corr_coef': 0.9753492057788247,
  'pruner': 'WilcoxonPruner'},
 ('CatBoost', 'MedianPruner'): {'best_score': 10.919798187835573,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 5,
   'rsm': 1.0,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 10.919798187835573,
  'test_rmse': 3.3045117926609935,
  'test_corr_coef': 0.9763196998303095,
  'pruner': 'MedianPruner'},
 ('CatBoost', 'NopPruner'): {'best_score': 10.919798187835573,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 1,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 10.919798187835573,
  'test_rmse': 3.3045117926609935,
  'test_corr_coef': 0.9763196998303095,
  'pruner': 'NopPruner'},
 ('CatBoost', 'PatientPruner'): {'best_score': 11.855218020554956,
  'best_params': {'iterations': 500,
   'learning_rate': 0.03,
   'depth': 8,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 5,
   'rsm': 0.8,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 11.855218020554956,
  'test_rmse': 3.443140720411374,
  'test_corr_coef': 0.9736540447216337,
  'pruner': 'PatientPruner'},
 ('CatBoost', 'PercentilePruner'): {'best_score': 12.607187864148157,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 8,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 1,
   'rsm': 0.6,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12.607187864148157,
  'test_rmse': 3.55066020116656,
  'test_corr_coef': 0.971322712261391,
  'pruner': 'PercentilePruner'},
 ('CatBoost', 'SuccessiveHalvingPruner'): {'best_score': 11.821269468870975,
  'best_params': {'iterations': 500,
   'learning_rate': 0.03,
   'depth': 8,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 5,
   'rsm': 1.0,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 11.821269468870975,
  'test_rmse': 3.438207304522369,
  'test_corr_coef': 0.9734884293063593,
  'pruner': 'SuccessiveHalvingPruner'},
 ('CatBoost', 'HyperbandPruner'): {'best_score': 12.027038308322881,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 8,
   'l2_leaf_reg': 1,
   'border_count': 32,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12.027038308322881,
  'test_rmse': 3.4680020629063764,
  'test_corr_coef': 0.9732364988808638,
  'pruner': 'HyperbandPruner'},
 ('CatBoost', 'ThresholdPruner'): {'best_score': 11.652636892792769,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 32,
   'min_data_in_leaf': 20,
   'rsm': 1.0,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 11.652636892792769,
  'test_rmse': 3.4135958889113938,
  'test_corr_coef': 0.9747368334479797,
  'pruner': 'ThresholdPruner'},
 ('CatBoost', 'WilcoxonPruner'): {'best_score': 12.529309573247462,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 8,
   'l2_leaf_reg': 3,
   'border_count': 64,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 12.529309573247462,
  'test_rmse': 3.5396764786131882,
  'test_corr_coef': 0.9721035382553069,
  'pruner': 'WilcoxonPruner'},
 ('NGBoost', 'MedianPruner'): {'best_score': 11.787200156683458,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 11.787200156683458,
  'test_rmse': 3.433249212725965,
  'test_corr_coef': 0.9740301732040282,
  'pruner': 'MedianPruner'},
 ('NGBoost', 'NopPruner'): {'best_score': 13.01778782749815,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'natural_gradient': True,
   'minibatch_frac': 0.9,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 13.01778782749815,
  'test_rmse': 3.608017160089202,
  'test_corr_coef': 0.970563269407315,
  'pruner': 'NopPruner'},
 ('NGBoost', 'PatientPruner'): {'best_score': 12.229892523132435,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12.229892523132435,
  'test_rmse': 3.497126323588045,
  'test_corr_coef': 0.9723061220102394,
  'pruner': 'PatientPruner'},
 ('NGBoost', 'PercentilePruner'): {'best_score': 11.475107621268567,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 11.475107621268567,
  'test_rmse': 3.3874928223198593,
  'test_corr_coef': 0.974015347546475,
  'pruner': 'PercentilePruner'},
 ('NGBoost', 'SuccessiveHalvingPruner'): {'best_score': 13.357727325159095,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 13.357727325159095,
  'test_rmse': 3.6548224751907026,
  'test_corr_coef': 0.9703301488315333,
  'pruner': 'SuccessiveHalvingPruner'},
 ('NGBoost', 'HyperbandPruner'): {'best_score': 12.074201542337628,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12.074201542337628,
  'test_rmse': 3.4747951799117063,
  'test_corr_coef': 0.9733753137108688,
  'pruner': 'HyperbandPruner'},
 ('NGBoost', 'ThresholdPruner'): {'best_score': 12.049261311520954,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12.049261311520954,
  'test_rmse': 3.471204590847528,
  'test_corr_coef': 0.9731028245083783,
  'pruner': 'ThresholdPruner'},
 ('NGBoost', 'WilcoxonPruner'): {'best_score': 12.518679471941832,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 12.518679471941832,
  'test_rmse': 3.5381745960229027,
  'test_corr_coef': 0.9721688812310673,
  'pruner': 'WilcoxonPruner'},
 ('TabNet', 'MedianPruner'): {'best_score': 57.150789619831066,
  'best_params': {'n_d': 8,
   'n_a': 64,
   'n_steps': 3,
   'gamma': 1.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 57.150789619831066,
  'test_rmse': 7.55981412601071,
  'test_corr_coef': 0.9165437704275357,
  'pruner': 'MedianPruner'},
 ('TabNet', 'NopPruner'): {'best_score': 64.61811524711449,
  'best_params': {'n_d': 64,
   'n_a': 8,
   'n_steps': 7,
   'gamma': 1.3,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 64.61811524711449,
  'test_rmse': 8.038539372741448,
  'test_corr_coef': 0.9445795088788163,
  'pruner': 'NopPruner'},
 ('TabNet', 'PatientPruner'): {'best_score': 60.884687637207904,
  'best_params': {'n_d': 32,
   'n_a': 64,
   'n_steps': 10,
   'gamma': 1.5,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 60.884687637207904,
  'test_rmse': 7.802864066303341,
  'test_corr_coef': 0.9509496588137201,
  'pruner': 'PatientPruner'},
 ('TabNet', 'PercentilePruner'): {'best_score': 42.7407429742994,
  'best_params': {'n_d': 64,
   'n_a': 64,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 42.7407429742994,
  'test_rmse': 6.53764047453662,
  'test_corr_coef': 0.9597414136880744,
  'pruner': 'PercentilePruner'},
 ('TabNet', 'SuccessiveHalvingPruner'): {'best_score': 41.707776667325746,
  'best_params': {'n_d': 64,
   'n_a': 8,
   'n_steps': 10,
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
  'test_mse': 41.707776667325746,
  'test_rmse': 6.458155825568608,
  'test_corr_coef': 0.938355662576757,
  'pruner': 'SuccessiveHalvingPruner'},
 ('TabNet', 'HyperbandPruner'): {'best_score': 55.317839938826495,
  'best_params': {'n_d': 64,
   'n_a': 8,
   'n_steps': 10,
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
  'test_mse': 55.317839938826495,
  'test_rmse': 7.437596381817616,
  'test_corr_coef': 0.9574978176787599,
  'pruner': 'HyperbandPruner'},
 ('TabNet', 'ThresholdPruner'): {'best_score': 51.58994975454799,
  'best_params': {'n_d': 32,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 1.3,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 3,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 51.58994975454799,
  'test_rmse': 7.182614409429758,
  'test_corr_coef': 0.937347118124312,
  'pruner': 'ThresholdPruner'},
 ('TabNet', 'WilcoxonPruner'): {'best_score': 47.36691531108609,
  'best_params': {'n_d': 64,
   'n_a': 8,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.01,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 1,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 47.36691531108609,
  'test_rmse': 6.882362625660326,
  'test_corr_coef': 0.9297065611536723,
  'pruner': 'WilcoxonPruner'},
 ('HistGradientBoosting', 'MedianPruner'): {'best_score': 10.545840005139208,
  'best_params': {'learning_rate': 0.05,
   'max_iter': 300,
   'max_depth': 5,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.1,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 10.545840005139208,
  'test_rmse': 3.2474359123990744,
  'test_corr_coef': 0.9765957955399636,
  'pruner': 'MedianPruner'},
 ('HistGradientBoosting', 'NopPruner'): {'best_score': 9.984427233372632,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 100,
   'max_depth': None,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 31,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 9.984427233372632,
  'test_rmse': 3.159814430211469,
  'test_corr_coef': 0.97775014211043,
  'pruner': 'NopPruner'},
 ('HistGradientBoosting', 'PatientPruner'): {'best_score': 10.452850001722435,
  'best_params': {'learning_rate': 0.05,
   'max_iter': 400,
   'max_depth': 7,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.5,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 10.452850001722435,
  'test_rmse': 3.233086760624038,
  'test_corr_coef': 0.976777912723613,
  'pruner': 'PatientPruner'},
 ('HistGradientBoosting',
  'PercentilePruner'): {'best_score': 10.382128296001117, 'best_params': {'learning_rate': 0.05,
   'max_iter': 400,
   'max_depth': None,
   'min_samples_leaf': 10,
   'max_leaf_nodes': None,
   'l2_regularization': 0.5,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 10.382128296001117, 'test_rmse': 3.2221310178205225, 'test_corr_coef': 0.9769202112610341, 'pruner': 'PercentilePruner'},
 ('HistGradientBoosting',
  'SuccessiveHalvingPruner'): {'best_score': 10.550347312706231, 'best_params': {'learning_rate': 0.15,
   'max_iter': 200,
   'max_depth': 5,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 10.550347312706231, 'test_rmse': 3.248129817711452, 'test_corr_coef': 0.9766545149805603, 'pruner': 'SuccessiveHalvingPruner'},
 ('HistGradientBoosting',
  'HyperbandPruner'): {'best_score': 10.487063272105305, 'best_params': {'learning_rate': 0.05,
   'max_iter': 400,
   'max_depth': 7,
   'min_samples_leaf': 10,
   'max_leaf_nodes': None,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 10.487063272105305, 'test_rmse': 3.23837355351499, 'test_corr_coef': 0.9766194329102597, 'pruner': 'HyperbandPruner'},
 ('HistGradientBoosting', 'ThresholdPruner'): {'best_score': 10.40769142601717,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 400,
   'max_depth': 7,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 10.40769142601717,
  'test_rmse': 3.2260953838994237,
  'test_corr_coef': 0.9766945335976727,
  'pruner': 'ThresholdPruner'},
 ('HistGradientBoosting', 'WilcoxonPruner'): {'best_score': 10.318356266217195,
  'best_params': {'learning_rate': 0.05,
   'max_iter': 400,
   'max_depth': 5,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 63,
   'l2_regularization': 1.0,
   'max_bins': 64,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 10.318356266217195,
  'test_rmse': 3.2122198346653046,
  'test_corr_coef': 0.9771055983031024,
  'pruner': 'WilcoxonPruner'},
 ('PGBM', 'MedianPruner'): {'best_score': 14.496627958565755,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'max_leaves': 51,
   'min_split_gain': 0.0,
   'reg_lambda': 5.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 14.496627958565755,
  'test_rmse': 3.807443756454684,
  'test_corr_coef': 0.9690437148790706,
  'pruner': 'MedianPruner'},
 ('PGBM', 'NopPruner'): {'best_score': 14.539617922744112,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.05,
   'max_leaves': 29,
   'min_split_gain': 0.1,
   'reg_lambda': 5.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 14.539617922744112,
  'test_rmse': 3.8130850925129,
  'test_corr_coef': 0.9689935902416109,
  'pruner': 'NopPruner'},
 ('PGBM', 'PatientPruner'): {'best_score': 14.442386821156454,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.1,
   'max_leaves': 32,
   'min_split_gain': 0.1,
   'reg_lambda': 10.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'laplace'},
  'test_mse': 14.442386821156454,
  'test_rmse': 3.800314042438658,
  'test_corr_coef': 0.9704544091739867,
  'pruner': 'PatientPruner'},
 ('PGBM', 'PercentilePruner'): {'best_score': 15.433989897619893,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'max_leaves': 19,
   'min_split_gain': 0.0,
   'reg_lambda': 10.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 15.433989897619893,
  'test_rmse': 3.928611701049099,
  'test_corr_coef': 0.9668759632080998,
  'pruner': 'PercentilePruner'},
 ('PGBM', 'SuccessiveHalvingPruner'): {'best_score': 15.275256998512873,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.05,
   'max_leaves': 56,
   'min_split_gain': 0.1,
   'reg_lambda': 10.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 15.275256998512873,
  'test_rmse': 3.908357327383574,
  'test_corr_coef': 0.9672779427989531,
  'pruner': 'SuccessiveHalvingPruner'},
 ('PGBM', 'HyperbandPruner'): {'best_score': 15.167206941197199,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.15,
   'max_leaves': 26,
   'min_split_gain': 0.5,
   'reg_lambda': 5.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 15.167206941197199,
  'test_rmse': 3.8945098460778347,
  'test_corr_coef': 0.9675517690466688,
  'pruner': 'HyperbandPruner'},
 ('PGBM', 'ThresholdPruner'): {'best_score': 14.629252877302713,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_leaves': 59,
   'min_split_gain': 0.0,
   'reg_lambda': 5.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.9,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 14.629252877302713,
  'test_rmse': 3.8248206333503685,
  'test_corr_coef': 0.9701936368266324,
  'pruner': 'ThresholdPruner'},
 ('PGBM', 'WilcoxonPruner'): {'best_score': 15.950208926775849,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_leaves': 24,
   'min_split_gain': 0.5,
   'reg_lambda': 1.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 1.0,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 10,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 15.950208926775849,
  'test_rmse': 3.993771266206397,
  'test_corr_coef': 0.965241150561115,
  'pruner': 'WilcoxonPruner'}}
```

```python
best_scores_autosampler
```

```python
best_scores_autosampler
```

# **Probabilistic distributions using NGBoost**

```python
# Assuming best_params is obtained correctly
best_params = get_best_model_params(best_scores_autosampler, 'NGBoost')

# Remove any conflicting parameters from best_params
incompatible_keys = ['Dist', 'Score']
for key in incompatible_keys:
    best_params.pop(key, None)

# Initialize NGBoost model with best_params
ngb_model = NGBRegressor(Dist=Normal, **best_params)

# Fit the model
ngb_model.fit(X_train, y_train)

# Predict the distribution
pred_dist = ngb_model.pred_dist(X_test)

# Extract mean and standard deviation
predictions_NGB_df = pd.DataFrame({
    'Mean': pred_dist.loc,
    'StdDev': pred_dist.scale,
    'Actual': y_test.ravel()
})

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, pred_dist.loc)
print(f"Mean Squared Error: {mse:.2f}")

# Print the DataFrame with mean and standard deviation
print(predictions_NGB_df.head())
```

```python
generate_and_save_plots(predictions_NGB_df, y_test, "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx",
    model_name="NGBoost",
    n_quantiles=10
)
```

# **Probabilistic distributions using PGBM**

```python
# Assuming best_params is obtained correctly
best_params = get_best_model_params(best_scores_autosampler, 'PGBM')

# Remove any conflicting parameters from best_params
# Note: Adjust the incompatible keys based on PGBM's requirements
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
pgbm_model.train((X_train, y_train), objective=mseloss_objective, metric=rmseloss_metric, params = best_params)

# Predict the distribution
pred_dist = pgbm_model.predict_dist(X_test)

# Extract mean and standard deviation
# Note: PGBM might return predictions in a different format, adjust accordingly
predictions_PGBM_df = pd.DataFrame({
    'Mean': pred_dist.mean(axis=0),
    'StdDev': pred_dist.std(axis=0),
    'Actual': y_test.ravel()
})

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, predictions_PGBM_df['Mean'])
print(f"Mean Squared Error: {mse:.2f}")

# Print the DataFrame with mean and standard deviation
print(predictions_PGBM_df.head())
```

```python
generate_and_save_plots(predictions_PGBM_df, y_test, "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx",
    model_name="PGBM",
    n_quantiles=10
)
```

# **Probabilistic Distribution using GPBoost (Treeffuser)**


```python
best_params = get_best_model_params(best_scores_autosampler, 'GPBoost')

gpboost_score_engine = TreeBasedScoreModel(
    model_name='gpboost',
    model_params=best_params,
    seed=42,
    verbose=-1
)

# The fix is in the line below: '_score_model' has been corrected to 'score_model'.
model = Treeffuser(score_model=gpboost_score_engine, seed=42)
model.fit(X_train, y_train)

n_samples_for_prediction = 200
y_samples = model.sample(X_test, n_samples=n_samples_for_prediction, verbose=False)

y_samples_obj = Samples(y_samples)
mean = y_samples_obj.sample_mean()
stddev = y_samples_obj.sample_std()

predictions_GPBoost_TF_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_GPBoost_TF_df.head())

```

```python
generate_and_save_plots(predictions_GPBoost_TF_df, y_test, "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GPBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GPBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GPBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx",
    model_name="GPBoost",
    n_quantiles=10
)
```

# **Probabilistic Distribution using Gradient Boosting (Treeffuser)**

```python
best_params = get_best_model_params(best_scores_autosampler, 'GradientBoosting')

gbr_score_engine = TreeBasedScoreModel(
    model_name='gradient_boosting',
    model_params=best_params,
    seed=42,
)

model = Treeffuser(score_model=gbr_score_engine, seed=42)
model.fit(X_train, y_train)

n_samples_for_prediction = 200
y_samples = model.sample(X_test, n_samples=n_samples_for_prediction, verbose=False)

y_samples_obj = Samples(y_samples)
mean = y_samples_obj.sample_mean()
stddev = y_samples_obj.sample_std()

predictions_GBR_TF_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_GBR_TF_df.head())

```

```python
generate_and_save_plots(predictions_GBR_TF_df, y_test, "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GBR_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx",
    model_name="Gradient Boosting",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GBR_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx",
    model_name="GBoosting",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GBR_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx",
    model_name="GBoosting",
    n_quantiles=10
)
```

# **Probabilistic Distribution using Catboost (Treeffuser)**

```python
best_params = get_best_model_params(best_scores_autosampler, 'CatBoost')

catboost_score_engine = TreeBasedScoreModel(
    model_name='catboost',
    model_params=best_params,
    seed=42,
)

model = Treeffuser(score_model=catboost_score_engine, seed=42)
model.fit(X_train, y_train)

n_samples_for_prediction = 200
y_samples = model.sample(X_test, n_samples=n_samples_for_prediction, verbose=False)

y_samples_obj = Samples(y_samples)
mean = y_samples_obj.sample_mean()
stddev = y_samples_obj.sample_std()

predictions_CatBoost_TF_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_CatBoost_TF_df.head())
```

```python
generate_and_save_plots(predictions_CatBoost_TF_df, y_test, "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_CatBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx",
    model_name="CatBoost",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_CatBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx",
    model_name="CatBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_CatBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx",
    model_name="CatBoost",
    n_quantiles=10
)
```

# **Probabilistic Distribution using XGBoost (Treeffuser)**

```python
best_params = get_best_model_params(best_scores_autosampler, 'XGBoost')

xgb_score_engine = TreeBasedScoreModel(
    model_name='xgboost',
    model_params=best_params,
    seed=42,
    verbose=-1
)

model = Treeffuser(score_model=xgb_score_engine, seed=42)
model.fit(X_train, y_train)

n_samples_for_prediction = 200
y_samples = model.sample(X_test, n_samples=n_samples_for_prediction, verbose=False)

y_samples_obj = Samples(y_samples)
mean = y_samples_obj.sample_mean()
stddev = y_samples_obj.sample_std()

predictions_XGB_TF_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_XGB_TF_df.head())
```

```python
generate_and_save_plots(predictions_XGB_TF_df, y_test, "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_XGB_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_XGB_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_XGB_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx",
    model_name="XGBoost",
    n_quantiles=10
)
```

# **Probabilistic Distribution using LGBM (Treeffuser)**

```python
best_params = get_best_model_params(best_scores_autosampler, 'LightGBM')

lgbm_score_engine = TreeBasedScoreModel(
    model_name='lightgbm',
    model_params=best_params,
    seed=42,
    verbose=-1
)

model = Treeffuser(score_model=lgbm_score_engine, seed=42)
model.fit(X_train, y_train)

n_samples_for_prediction = 200
y_samples = model.sample(X_test, n_samples=n_samples_for_prediction, verbose=False)

y_samples_obj = Samples(y_samples)
mean = y_samples_obj.sample_mean()
stddev = y_samples_obj.sample_std()

predictions_LGBM_TF_df = pd.DataFrame({
    'Mean': mean,
    'StdDev': stddev,
    'Actual': y_test
})

print(predictions_LGBM_TF_df.head())

```

```python
generate_and_save_plots(predictions_LGBM_TF_df, y_test, "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_LGBM_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_LGBM_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_LGBM_TF_df,
    y_test,
    "./drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx",
    model_name="LightGBM",
    n_quantiles=10
)
```

# **IBUG Installation**

```python
# Clone the repository
!git clone https://github.com/jjbrophy47/ibug.git
%cd ibug

# Build extensions if needed
!python setup.py build_ext --inplace

# Install the package
!pip install .

# Verify installation
try:
    from ibug import IBUGWrapper
    print("IBUGWrapper imported successfully!")
except ImportError as e:
    print("Error importing IBUGWrapper:", e)
```

```python
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

```

# **Probabilistic Distributions using LGBM (IBUG)**

```python
# y_train = y_train.to_numpy()
# y_val = y_val.to_numpy()
```

```python
# Assume hyperparameter tuning has been done and best_params are obtained
best_params = get_best_model_params(best_scores_autosampler, 'LightGBM')

# --- Ensure IBUG compatibility ---
best_params['reg_alpha'] = 0
best_params['class_weight'] = None
best_params['boosting_type'] = 'gbdt'

# Initialize and train the LightGBM regressor using the best_params
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)

# Extend the trained model into a probabilistic predictor using IBUGWrapper
prob_model = IBUGWrapper().fit(model, X_train, y_train, X_val=X_val, y_val=y_val)

# Use the model to predict the probabilistic distribution on the test data.
location, scale = prob_model.pred_dist(X_test)

# Optionally, to also get the k-nearest training indices and their values used in modeling
location2, scale2, train_idxs, train_vals = prob_model.pred_dist(X_test, return_kneighbors=True)

dev = scale**0.5
# Create a DataFrame for clarity
predictions_LGBM_df = pd.DataFrame({
    'Mean': location,
    'StdDev': dev,
    'Actual': y_test,
})

print(predictions_LGBM_df.head())
```

```python
%cd ..
```

```python
%cd ..
```

```python
generate_and_save_plots(predictions_LGBM_df, y_test, "./content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_LGBM_df,
    y_test,
    "/content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx",
    model_name="LGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_LGBM_df,
    y_test,
    "/content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx",
    model_name="LGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_LGBM_df,
    y_test,
    "/content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx",
    model_name="LGBM",
    n_quantiles=10
)
```

# **Probabilistic Distributions using XGBoost (IBUG)**

```python
# Assume hyperparameter tuning has been done and best_params are obtained
best_params = get_best_model_params(best_scores_autosampler, 'XGBoost')

best_params['reg_alpha'] = 0
best_params['scale_pos_weight'] = 1
best_params['base_score'] = 0.5

# Initialize and train the XGBoost regressor using the best_params
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# Extend the trained model into a probabilistic predictor using IBUGWrapper
prob_model = IBUGWrapper().fit(model, X_train, y_train, X_val=X_val, y_val=y_val)

# Use the model to predict the probabilistic distribution on the test data.
# This returns the predicted mean (location) and uncertainty (scale, variance proxy)
location, scale = prob_model.pred_dist(X_test)

# Optionally, to also get the k-nearest training indices and their values used in modeling
location, scale, train_idxs, train_vals = prob_model.pred_dist(X_test, return_kneighbors=True)

# Calculate the standard deviation from the variance (scale)
std_dev = scale ** 0.5

# Create a DataFrame for clarity
predictions_XGB_df = pd.DataFrame({
    'Mean': location,
    'StdDev': std_dev,
    'Actual': y_test
})

print(predictions_XGB_df.head())
```

```python
generate_and_save_plots(predictions_LGBM_df, y_test, "./content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_XGB_df,
    y_test,
    "/content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx",
    model_name="XGB",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_XGB_df,
    y_test,
    "/content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx",
    model_name="XGB",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_XGB_df,
    y_test,
    "/content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx",
    model_name="XGB",
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
folder_path = "/content/drive/MyDrive/Regression/noise_uncertainty_analysis"
excel_path = os.path.join(folder_path, "/content/drive/MyDrive/Regression/noise_uncertainty_analysis/probabilistic_distribution/Matrix Evaluation.xlsx")
with pd.ExcelWriter(_ensure_parent_dir(excel_path), engine='xlsxwriter') as writer:
    plot_metrics_and_save_to_excel(writer, predictions_NGB_df, y_test, 'NGBoost')
    plot_metrics_and_save_to_excel(writer, predictions_PGBM_df, y_test, 'PGBM')
    plot_metrics_and_save_to_excel(writer, predictions_LGBM_df, y_test, 'LGBM')
    plot_metrics_and_save_to_excel(writer, predictions_XGB_df, y_test, 'XGBoost')
    plot_metrics_and_save_to_excel(writer, predictions_LGBM_TF_df, y_test, "LightGBM_TF")
    plot_metrics_and_save_to_excel(writer, predictions_XGB_TF_df, y_test, "XGBoost_TF")
    plot_metrics_and_save_to_excel(writer, predictions_CatBoost_TF_df, y_test, "CatBoost_TF")
    plot_metrics_and_save_to_excel(writer, predictions_GPBoost_TF_df, y_test, "GPBoost_TF")
    plot_metrics_and_save_to_excel(writer, predictions_GBR_TF_df, y_test, "GradientBoosting_TF")
```

