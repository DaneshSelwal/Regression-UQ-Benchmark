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
folder_path = "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis"

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
generate_and_save_plots(predictions_NGB_df, y_test, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/NGBoost_Prob.xlsx",
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
generate_and_save_plots(predictions_PGBM_df, y_test, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/PGBM_Prob.xlsx",
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
generate_and_save_plots(predictions_GPBoost_TF_df, y_test, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GPBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GPBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GPBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GPBoost_Prob_TF.xlsx",
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
generate_and_save_plots(predictions_GBR_TF_df, y_test, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GBR_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx",
    model_name="Gradient Boosting",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GBR_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx",
    model_name="GBoosting",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GBR_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/GBoosting_Prob_TF.xlsx",
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
generate_and_save_plots(predictions_CatBoost_TF_df, y_test, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_CatBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx",
    model_name="CatBoost",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_CatBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx",
    model_name="CatBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_CatBoost_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/CatBoost_Prob_TF.xlsx",
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
generate_and_save_plots(predictions_XGB_TF_df, y_test, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_XGB_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_XGB_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_XGB_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob_TF.xlsx",
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
generate_and_save_plots(predictions_LGBM_TF_df, y_test, "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_LGBM_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_LGBM_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_LGBM_TF_df,
    y_test,
    "./drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob_TF.xlsx",
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
generate_and_save_plots(predictions_LGBM_df, y_test, "./content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_LGBM_df,
    y_test,
    "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx",
    model_name="LGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_LGBM_df,
    y_test,
    "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx",
    model_name="LGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_LGBM_df,
    y_test,
    "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/LGBM_Prob.xlsx",
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
generate_and_save_plots(predictions_LGBM_df, y_test, "./content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_XGB_df,
    y_test,
    "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx",
    model_name="XGB",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_XGB_df,
    y_test,
    "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx",
    model_name="XGB",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_XGB_df,
    y_test,
    "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/XGB_Prob.xlsx",
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
folder_path = "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis"
excel_path = os.path.join(folder_path, "/content/drive/MyDrive/Regression/concrete_uncertainty_analysis/probabilistic_distribution/Matrix Evaluation.xlsx")
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

