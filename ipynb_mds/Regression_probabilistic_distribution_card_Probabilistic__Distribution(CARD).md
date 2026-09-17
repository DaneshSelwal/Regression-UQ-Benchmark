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
# Go to find & replace button and replace () with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (Y_Label) with actual data label name.
```

```python
feature_names = ['Qt', 'Qt-1', 'St-1']
```

```python
train_data_path = "./drive/MyDrive/Regression/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/data/test.csv"
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
def generate_and_save_plots(predictions_df, y_test, excel_path):
    # 1. Predicted vs. Actual Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Actual', y='Mean', data=predictions_df)
    plt.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
             [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
             color='red', linestyle='--')
    plt.title('Predicted Y_Label vs. Actual Y_Label')
    plt.xlabel('Actual Y_Label')
    plt.ylabel('Predicted Mean Y_Label')
    plt.tight_layout()
    predicted_vs_actual_path = 'predicted_vs_actual.png'
    plt.savefig(predicted_vs_actual_path)
    plt.close()

    # 2. Uncertainty Bands
    plt.figure(figsize=(12, 8))
    plt.errorbar(x=range(len(predictions_df)), y=predictions_df['Mean'],
                 yerr=predictions_df['StdDev'], fmt='o', markersize=4, ecolor='red', elinewidth=1, capsize=5, alpha=0.7)
    plt.plot(range(len(predictions_df)), predictions_df['Mean'], color='black', linewidth=0.5, alpha=0.7)
    plt.title('Predicted Mean with Uncertainty Bands', fontsize=16)
    plt.xlabel('Sample Number', fontsize=14)
    plt.ylabel('Predicted Mean Y_Label')
    plt.tight_layout()
    uncertainty_bands_path = 'uncertainty_bands.png'
    plt.savefig(uncertainty_bands_path)
    plt.close()

    # 3. Histogram of Standard Deviations
    plt.figure(figsize=(12, 8))
    sns.histplot(predictions_df['StdDev'], bins=20, kde=True)
    plt.title('Distribution of Predicted Standard Deviations')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.tight_layout()
    histogram_stddev_path = 'histogram_stddev.png'
    plt.savefig(histogram_stddev_path)
    plt.close()

    # 4. Residual Plot
    residuals = predictions_df['Actual'] - predictions_df['Mean']
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=predictions_df['Mean'], y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Mean Y_Label')
    plt.ylabel('Residuals')
    plt.tight_layout()
    residual_plot_path = 'residual_plot.png'
    plt.savefig(residual_plot_path)
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
    plt.savefig(stddev_plot_path)
    plt.close()

    # Plot the actual vs predicted mean with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(
        y_test,
        label='Actual Y_Label',
        color='black',
        marker='o',
        linestyle='dashed',
        linewidth=1,
        markersize=4
    )
    plt.plot(
        predictions_df['Mean'],
        label='Predicted Mean Y_Label',
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
    plt.ylabel(' Y_Label')
    plt.title('Predictions with Confidence Intervals')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    prediction_plot_path = 'prediction_plot.png'
    plt.savefig(prediction_plot_path)
    plt.close()

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

    # Write the dataframe to a worksheet.
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
    os.remove(predicted_vs_actual_path)
    os.remove(uncertainty_bands_path)
    os.remove(histogram_stddev_path)
    os.remove(residual_plot_path)
    os.remove(stddev_plot_path)
    os.remove(prediction_plot_path)
```

```python
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

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
        mode="a" if excel_path.exists() else "w",
        if_sheet_exists="replace"
    ) as writer:
        summary_df.to_excel(writer, sheet_name=sheet_summary, index=False)
        bin_df.to_excel(writer, sheet_name=sheet_bins, index=False)

    print(f"✅ CRPS decomposition saved to:\n{excel_path}")
    print(f"   Sheets: {sheet_summary}, {sheet_bins}")

    return summary_df
```

```python
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

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
        mode="a" if excel_path.exists() else "w",
        if_sheet_exists="replace"
    ) as writer:

        pit_df.to_excel(writer, sheet_name=sheet_values, index=False)
        hist_df.to_excel(writer, sheet_name=sheet_hist, index=False)
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

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
        mode="a" if excel_path.exists() else "w",
        if_sheet_exists="replace"
    ) as writer:

        calib_df.to_excel(writer, sheet_name=sheet_curve, index=False)
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
# Define the path to the folder
folder_path = "/content/drive/MyDrive/Regression/"

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
    metrics_df.to_excel(writer, sheet_name=metrics_sheet_name, index=False)

    summary_sheet_name = f'{model_name} Summary'
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
    fig.savefig(plot_image_path)
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
best_scores_autosampler = {('Random Forest', 'MedianPruner'): {'best_score': 125647.84335013448,
  'best_params': {'n_estimators': 700,
   'criterion': 'absolute_error',
   'max_depth': 20,
   'min_samples_split': 0.01,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 'sqrt',
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.1},
  'test_mse': 125647.84335013448,
  'test_rmse': 354.4683954178912,
  'test_corr_coef': 0.9579299217249909,
  'pruner': 'MedianPruner'},
 ('Random Forest', 'NopPruner'): {'best_score': 115703.5673107638,
  'best_params': {'n_estimators': 300,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 10,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 1.0,
   'max_leaf_nodes': 200,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01},
  'test_mse': 115703.5673107638,
  'test_rmse': 340.1522707711413,
  'test_corr_coef': 0.9615119752252415,
  'pruner': 'NopPruner'},
 ('Random Forest', 'PatientPruner'): {'best_score': 115629.12224649779,
  'best_params': {'n_estimators': 300,
   'criterion': 'absolute_error',
   'max_depth': 40,
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.2,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.1},
  'test_mse': 115629.12224649779,
  'test_rmse': 340.04282413616346,
  'test_corr_coef': 0.9615056996911798,
  'pruner': 'PatientPruner'},
 ('Random Forest', 'PercentilePruner'): {'best_score': 117503.65849868047,
  'best_params': {'n_estimators': 200,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 5,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 1.0,
   'max_leaf_nodes': 50,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 117503.65849868047,
  'test_rmse': 342.78806644730275,
  'test_corr_coef': 0.9609173489911242,
  'pruner': 'PercentilePruner'},
 ('Random Forest',
  'SuccessiveHalvingPruner'): {'best_score': 126584.35298970548, 'best_params': {'n_estimators': 200,
   'criterion': 'absolute_error',
   'max_depth': 20,
   'min_samples_split': 0.01,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 0.3,
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.01}, 'test_mse': 126584.35298970548, 'test_rmse': 355.7869488748927, 'test_corr_coef': 0.9595107925559148, 'pruner': 'SuccessiveHalvingPruner'},
 ('Random Forest', 'HyperbandPruner'): {'best_score': 125243.55676457845,
  'best_params': {'n_estimators': 500,
   'criterion': 'absolute_error',
   'max_depth': 40,
   'min_samples_split': 5,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 'sqrt',
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.01,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 125243.55676457845,
  'test_rmse': 353.8976642542,
  'test_corr_coef': 0.9581547133700536,
  'pruner': 'HyperbandPruner'},
 ('Random Forest', 'ThresholdPruner'): {'best_score': 124854.97625512879,
  'best_params': {'n_estimators': 700,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 0.01,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.01,
   'max_features': 'sqrt',
   'max_leaf_nodes': 100,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.0},
  'test_mse': 124854.97625512879,
  'test_rmse': 353.348236524719,
  'test_corr_coef': 0.9581803650792393,
  'pruner': 'ThresholdPruner'},
 ('Random Forest', 'WilcoxonPruner'): {'best_score': 127823.90890176565,
  'best_params': {'n_estimators': 500,
   'criterion': 'absolute_error',
   'max_depth': None,
   'min_samples_split': 5,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_features': 0.5,
   'max_leaf_nodes': None,
   'min_impurity_decrease': 0.0,
   'n_jobs': -1,
   'random_state': 42,
   'verbose': 0,
   'warm_start': False,
   'ccp_alpha': 0.001},
  'test_mse': 127823.90890176565,
  'test_rmse': 357.5246969116478,
  'test_corr_coef': 0.9601312789260384,
  'pruner': 'WilcoxonPruner'},
 ('Gradient Boosting', 'MedianPruner'): {'best_score': 98151.97595540313,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.1,
   'n_estimators': 300,
   'subsample': 1.0,
   'criterion': 'friedman_mse',
   'min_samples_split': 0.01,
   'min_samples_leaf': 3,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 10,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': 'sqrt',
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 98151.97595540313,
  'test_rmse': 313.2921575070195,
  'test_corr_coef': 0.9675686886787896,
  'pruner': 'MedianPruner'},
 ('Gradient Boosting', 'NopPruner'): {'best_score': 106575.03780853433,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.1,
   'n_estimators': 100,
   'subsample': 0.7,
   'criterion': 'friedman_mse',
   'min_samples_split': 5,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 5,
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
   'ccp_alpha': 0.001},
  'test_mse': 106575.03780853433,
  'test_rmse': 326.45832476525135,
  'test_corr_coef': 0.9653413824999147,
  'pruner': 'NopPruner'},
 ('Gradient Boosting', 'PatientPruner'): {'best_score': 104595.41518900798,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.2,
   'n_estimators': 500,
   'subsample': 1.0,
   'criterion': 'squared_error',
   'min_samples_split': 5,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 10,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.0001,
   'ccp_alpha': 0.0},
  'test_mse': 104595.41518900798,
  'test_rmse': 323.4121444674086,
  'test_corr_coef': 0.965322863260212,
  'pruner': 'PatientPruner'},
 ('Gradient Boosting', 'PercentilePruner'): {'best_score': 99725.80067490121,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.05,
   'n_estimators': 300,
   'subsample': 0.9,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 7,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.1,
   'verbose': 0,
   'max_leaf_nodes': 50,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': None,
   'tol': 0.001,
   'ccp_alpha': 0.001},
  'test_mse': 99725.80067490121,
  'test_rmse': 315.79392121271303,
  'test_corr_coef': 0.9695611559059057,
  'pruner': 'PercentilePruner'},
 ('Gradient Boosting',
  'SuccessiveHalvingPruner'): {'best_score': 115388.14082603931, 'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.2,
   'n_estimators': 700,
   'subsample': 0.9,
   'criterion': 'friedman_mse',
   'min_samples_split': 0.01,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.0,
   'max_depth': 5,
   'min_impurity_decrease': 0.0,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.5,
   'verbose': 0,
   'max_leaf_nodes': 30,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.001,
   'ccp_alpha': 0.01}, 'test_mse': 115388.14082603931, 'test_rmse': 339.6882995130084, 'test_corr_coef': 0.9613694345764211, 'pruner': 'SuccessiveHalvingPruner'},
 ('Gradient Boosting', 'HyperbandPruner'): {'best_score': 94870.04457542315,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.2,
   'n_estimators': 200,
   'subsample': 0.5,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 5,
   'min_weight_fraction_leaf': 0.05,
   'max_depth': 7,
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
   'tol': 0.001,
   'ccp_alpha': 0.0},
  'test_mse': 94870.04457542315,
  'test_rmse': 308.0098124661342,
  'test_corr_coef': 0.9693954848146131,
  'pruner': 'HyperbandPruner'},
 ('Gradient Boosting', 'ThresholdPruner'): {'best_score': 104617.67213635288,
  'best_params': {'loss': 'huber',
   'learning_rate': 0.1,
   'n_estimators': 700,
   'subsample': 0.9,
   'criterion': 'squared_error',
   'min_samples_split': 2,
   'min_samples_leaf': 0.01,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 7,
   'min_impurity_decrease': 0.01,
   'init': None,
   'random_state': 42,
   'max_features': None,
   'alpha': 0.9,
   'verbose': 0,
   'max_leaf_nodes': None,
   'warm_start': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'tol': 0.0001,
   'ccp_alpha': 0.001},
  'test_mse': 104617.67213635288,
  'test_rmse': 323.4465522097165,
  'test_corr_coef': 0.9670218469145725,
  'pruner': 'ThresholdPruner'},
 ('Gradient Boosting', 'WilcoxonPruner'): {'best_score': 115965.93616321213,
  'best_params': {'loss': 'absolute_error',
   'learning_rate': 0.2,
   'n_estimators': 200,
   'subsample': 0.9,
   'criterion': 'friedman_mse',
   'min_samples_split': 10,
   'min_samples_leaf': 1,
   'min_weight_fraction_leaf': 0.01,
   'max_depth': 10,
   'min_impurity_decrease': 0.0,
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
   'ccp_alpha': 0.0},
  'test_mse': 115965.93616321213,
  'test_rmse': 340.5377162124808,
  'test_corr_coef': 0.9661622534830681,
  'pruner': 'WilcoxonPruner'},
 ('XGBoost', 'MedianPruner'): {'best_score': 144897.269725111,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 3,
   'gamma': 0.5,
   'subsample': 0.8,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 144897.269725111,
  'test_rmse': 380.65373993317206,
  'test_corr_coef': 0.9645849926785167,
  'pruner': 'MedianPruner'},
 ('XGBoost', 'NopPruner'): {'best_score': 112489.26747933708,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 5,
   'min_child_weight': 1,
   'gamma': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 112489.26747933708,
  'test_rmse': 335.3941971461896,
  'test_corr_coef': 0.96564448358986,
  'pruner': 'NopPruner'},
 ('XGBoost', 'PatientPruner'): {'best_score': 126101.16516872347,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.5,
   'reg_alpha': 0.01,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 126101.16516872347,
  'test_rmse': 355.1072586821107,
  'test_corr_coef': 0.9614123668357902,
  'pruner': 'PatientPruner'},
 ('XGBoost', 'PercentilePruner'): {'best_score': 136726.62651782745,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 136726.62651782745,
  'test_rmse': 369.76563728641344,
  'test_corr_coef': 0.957646899373698,
  'pruner': 'PercentilePruner'},
 ('XGBoost', 'SuccessiveHalvingPruner'): {'best_score': 113753.5246155919,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.15,
   'max_depth': 5,
   'min_child_weight': 1,
   'gamma': 0.5,
   'subsample': 0.5,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.1,
   'reg_lambda': 5,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 113753.5246155919,
  'test_rmse': 337.2736642781228,
  'test_corr_coef': 0.966771646445439,
  'pruner': 'SuccessiveHalvingPruner'},
 ('XGBoost', 'HyperbandPruner'): {'best_score': 133430.5131094733,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.1,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 133430.5131094733,
  'test_rmse': 365.2814163209967,
  'test_corr_coef': 0.9606698458763485,
  'pruner': 'HyperbandPruner'},
 ('XGBoost', 'ThresholdPruner'): {'best_score': 115799.52745518631,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0,
   'subsample': 0.7,
   'colsample_bytree': 0.9,
   'colsample_bylevel': 0.7,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 115799.52745518631,
  'test_rmse': 340.2932962242811,
  'test_corr_coef': 0.9654423196285966,
  'pruner': 'ThresholdPruner'},
 ('XGBoost', 'WilcoxonPruner'): {'best_score': 123304.66533714374,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 7,
   'min_child_weight': 1,
   'gamma': 0.5,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'colsample_bylevel': 0.9,
   'reg_alpha': 0.01,
   'reg_lambda': 10,
   'objective': 'reg:squarederror',
   'random_state': 42,
   'n_jobs': -1},
  'test_mse': 123304.66533714374,
  'test_rmse': 351.14764036960827,
  'test_corr_coef': 0.9635628198182367,
  'pruner': 'WilcoxonPruner'},
 ('LightGBM', 'MedianPruner'): {'best_score': 86044.08662798398,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 31,
   'max_depth': 5,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'min_child_weight': 1e-05,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 86044.08662798398,
  'test_rmse': 293.332723418278,
  'test_corr_coef': 0.9714165057095582,
  'pruner': 'MedianPruner'},
 ('LightGBM', 'NopPruner'): {'best_score': 94477.59834052388,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 31,
   'max_depth': -1,
   'min_child_samples': 10,
   'subsample': 0.7,
   'colsample_bytree': 0.5,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 1e-05,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 94477.59834052388,
  'test_rmse': 307.3720845173222,
  'test_corr_coef': 0.969908476703405,
  'pruner': 'NopPruner'},
 ('LightGBM', 'PatientPruner'): {'best_score': 87830.23792054132,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 31,
   'max_depth': 7,
   'min_child_samples': 1,
   'subsample': 0.6,
   'colsample_bytree': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 0.1,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 87830.23792054132,
  'test_rmse': 296.36166742772474,
  'test_corr_coef': 0.9721975978240425,
  'pruner': 'PatientPruner'},
 ('LightGBM', 'PercentilePruner'): {'best_score': 94715.31767890525,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 63,
   'max_depth': 5,
   'min_child_samples': 1,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 0.1,
   'bagging_freq': 5,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 94715.31767890525,
  'test_rmse': 307.7585379463992,
  'test_corr_coef': 0.9691775251113111,
  'pruner': 'PercentilePruner'},
 ('LightGBM', 'SuccessiveHalvingPruner'): {'best_score': 86867.50262566945,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.1,
   'reg_lambda': 10,
   'min_child_weight': 0.01,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 86867.50262566945,
  'test_rmse': 294.7329344095591,
  'test_corr_coef': 0.9720684448833871,
  'pruner': 'SuccessiveHalvingPruner'},
 ('LightGBM', 'HyperbandPruner'): {'best_score': 88709.00101436477,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 1,
   'subsample': 0.9,
   'colsample_bytree': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 10,
   'min_child_weight': 0.001,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 88709.00101436477,
  'test_rmse': 297.84056307757135,
  'test_corr_coef': 0.9712796302011322,
  'pruner': 'HyperbandPruner'},
 ('LightGBM', 'ThresholdPruner'): {'best_score': 99979.74519447866,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'num_leaves': 15,
   'max_depth': 3,
   'min_child_samples': 5,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'reg_alpha': 1,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 99979.74519447866,
  'test_rmse': 316.1957387354843,
  'test_corr_coef': 0.9671420731714029,
  'pruner': 'ThresholdPruner'},
 ('LightGBM', 'WilcoxonPruner'): {'best_score': 100736.29445203877,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'num_leaves': 15,
   'max_depth': -1,
   'min_child_samples': 5,
   'subsample': 0.5,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'min_child_weight': 1e-05,
   'bagging_freq': 1,
   'objective': 'regression',
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 100736.29445203877,
  'test_rmse': 317.3898146633549,
  'test_corr_coef': 0.9693661922645916,
  'pruner': 'WilcoxonPruner'},
 ('GPBoost', 'MedianPruner'): {'best_score': 112096.46959308149,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 10,
   'subsample': 1.0,
   'colsample_bytree': 1.0,
   'reg_alpha': 1.0,
   'reg_lambda': 1.0,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 112096.46959308149,
  'test_rmse': 334.80810861310016,
  'test_corr_coef': 0.9663622360328146,
  'pruner': 'MedianPruner'},
 ('GPBoost', 'NopPruner'): {'best_score': 111537.3357855164,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.8,
   'colsample_bytree': 0.9,
   'reg_alpha': 0,
   'reg_lambda': 0.5,
   'min_child_weight': 0.001,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 111537.3357855164,
  'test_rmse': 333.9720583903935,
  'test_corr_coef': 0.9658655126335507,
  'pruner': 'NopPruner'},
 ('GPBoost', 'PatientPruner'): {'best_score': 115032.42871913446,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': -1,
   'num_leaves': 63,
   'min_child_samples': 20,
   'subsample': 0.7,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.5,
   'reg_lambda': 0.1,
   'min_child_weight': 1e-05,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 115032.42871913446,
  'test_rmse': 339.1643093238651,
  'test_corr_coef': 0.9641114404740005,
  'pruner': 'PatientPruner'},
 ('GPBoost', 'PercentilePruner'): {'best_score': 114200.00943677705,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.9,
   'colsample_bytree': 1.0,
   'reg_alpha': 0,
   'reg_lambda': 1.0,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 114200.00943677705,
  'test_rmse': 337.9349189367341,
  'test_corr_coef': 0.9651040557662787,
  'pruner': 'PercentilePruner'},
 ('GPBoost', 'SuccessiveHalvingPruner'): {'best_score': 121675.81844833051,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 15,
   'min_child_samples': 10,
   'subsample': 0.9,
   'colsample_bytree': 0.5,
   'reg_alpha': 0.5,
   'reg_lambda': 1.0,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 121675.81844833051,
  'test_rmse': 348.820610698867,
  'test_corr_coef': 0.9629349493155991,
  'pruner': 'SuccessiveHalvingPruner'},
 ('GPBoost', 'HyperbandPruner'): {'best_score': 115913.92866273696,
  'best_params': {'n_estimators': 300,
   'learning_rate': 0.01,
   'max_depth': 7,
   'num_leaves': 31,
   'min_child_samples': 20,
   'subsample': 0.6,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.5,
   'reg_lambda': 0.5,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 115913.92866273696,
  'test_rmse': 340.4613467968676,
  'test_corr_coef': 0.9639397986834962,
  'pruner': 'HyperbandPruner'},
 ('GPBoost', 'ThresholdPruner'): {'best_score': 120416.85819599114,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 1.0,
   'colsample_bytree': 0.7,
   'reg_alpha': 0,
   'reg_lambda': 0.5,
   'min_child_weight': 0.01,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 120416.85819599114,
  'test_rmse': 347.0113228642419,
  'test_corr_coef': 0.9627972779246036,
  'pruner': 'ThresholdPruner'},
 ('GPBoost', 'WilcoxonPruner'): {'best_score': 114811.89688417579,
  'best_params': {'n_estimators': 400,
   'learning_rate': 0.01,
   'max_depth': 3,
   'num_leaves': 31,
   'min_child_samples': 5,
   'subsample': 0.9,
   'colsample_bytree': 1.0,
   'reg_alpha': 0.5,
   'reg_lambda': 1.0,
   'min_child_weight': 0.1,
   'random_state': 42,
   'n_jobs': -1,
   'verbose': -1},
  'test_mse': 114811.89688417579,
  'test_rmse': 338.8390427388435,
  'test_corr_coef': 0.9649777410550676,
  'pruner': 'WilcoxonPruner'},
 ('CatBoost', 'MedianPruner'): {'best_score': 101537.83907013081,
  'best_params': {'iterations': 200,
   'learning_rate': 0.03,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 1,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 101537.83907013081,
  'test_rmse': 318.65002600051804,
  'test_corr_coef': 0.9666880340359085,
  'pruner': 'MedianPruner'},
 ('CatBoost', 'NopPruner'): {'best_score': 109091.92676281127,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 64,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 109091.92676281127,
  'test_rmse': 330.2906701116628,
  'test_corr_coef': 0.9637561849388776,
  'pruner': 'NopPruner'},
 ('CatBoost', 'PatientPruner'): {'best_score': 111987.22701024539,
  'best_params': {'iterations': 200,
   'learning_rate': 0.1,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 64,
   'min_data_in_leaf': 1,
   'rsm': 0.8,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 111987.22701024539,
  'test_rmse': 334.64492676603567,
  'test_corr_coef': 0.9630756672808098,
  'pruner': 'PatientPruner'},
 ('CatBoost', 'PercentilePruner'): {'best_score': 101537.83907013081,
  'best_params': {'iterations': 200,
   'learning_rate': 0.03,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 20,
   'rsm': 1.0,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 101537.83907013081,
  'test_rmse': 318.65002600051804,
  'test_corr_coef': 0.9666880340359085,
  'pruner': 'PercentilePruner'},
 ('CatBoost', 'SuccessiveHalvingPruner'): {'best_score': 108300.3730298083,
  'best_params': {'iterations': 200,
   'learning_rate': 0.05,
   'depth': 10,
   'l2_leaf_reg': 3,
   'border_count': 64,
   'min_data_in_leaf': 10,
   'rsm': 0.6,
   'bagging_temperature': 0,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 108300.3730298083,
  'test_rmse': 329.09022019775716,
  'test_corr_coef': 0.9640724970413321,
  'pruner': 'SuccessiveHalvingPruner'},
 ('CatBoost', 'HyperbandPruner'): {'best_score': 118902.81904556698,
  'best_params': {'iterations': 500,
   'learning_rate': 0.03,
   'depth': 10,
   'l2_leaf_reg': 5,
   'border_count': 64,
   'min_data_in_leaf': 1,
   'rsm': 0.6,
   'bagging_temperature': 1,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 118902.81904556698,
  'test_rmse': 344.8228806874146,
  'test_corr_coef': 0.9603030735997221,
  'pruner': 'HyperbandPruner'},
 ('CatBoost', 'ThresholdPruner'): {'best_score': 106739.36867174642,
  'best_params': {'iterations': 200,
   'learning_rate': 0.03,
   'depth': 6,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 5,
   'rsm': 1.0,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 106739.36867174642,
  'test_rmse': 326.709915172078,
  'test_corr_coef': 0.9645246862965655,
  'pruner': 'ThresholdPruner'},
 ('CatBoost', 'WilcoxonPruner'): {'best_score': 112522.21218165623,
  'best_params': {'iterations': 1000,
   'learning_rate': 0.01,
   'depth': 10,
   'l2_leaf_reg': 1,
   'border_count': 64,
   'min_data_in_leaf': 5,
   'rsm': 0.8,
   'bagging_temperature': 10,
   'random_seed': 42,
   'verbose': 0},
  'test_mse': 112522.21218165623,
  'test_rmse': 335.44330695611774,
  'test_corr_coef': 0.9641808624579986,
  'pruner': 'WilcoxonPruner'},
 ('NGBoost', 'MedianPruner'): {'best_score': 139059.60517846112,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.03,
   'natural_gradient': True,
   'minibatch_frac': 0.7,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 139059.60517846112,
  'test_rmse': 372.90696584867,
  'test_corr_coef': 0.9560637639116014,
  'pruner': 'MedianPruner'},
 ('NGBoost', 'NopPruner'): {'best_score': 144252.83582636862,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 144252.83582636862,
  'test_rmse': 379.80631356833527,
  'test_corr_coef': 0.9655551114748977,
  'pruner': 'NopPruner'},
 ('NGBoost', 'PatientPruner'): {'best_score': 129173.46426120309,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.05,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 129173.46426120309,
  'test_rmse': 359.4071010166648,
  'test_corr_coef': 0.9630813831659504,
  'pruner': 'PatientPruner'},
 ('NGBoost', 'PercentilePruner'): {'best_score': 170535.12440821258,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 1.0,
   'col_sample': 1.0,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 170535.12440821258,
  'test_rmse': 412.9589863512024,
  'test_corr_coef': 0.956531348075053,
  'pruner': 'PercentilePruner'},
 ('NGBoost', 'SuccessiveHalvingPruner'): {'best_score': 150107.8515952292,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 150107.8515952292,
  'test_rmse': 387.4375454124564,
  'test_corr_coef': 0.9648418504016317,
  'pruner': 'SuccessiveHalvingPruner'},
 ('NGBoost', 'HyperbandPruner'): {'best_score': 159522.47310411773,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.7,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 159522.47310411773,
  'test_rmse': 399.402645339409,
  'test_corr_coef': 0.964985616741838,
  'pruner': 'HyperbandPruner'},
 ('NGBoost', 'ThresholdPruner'): {'best_score': 167575.8434276094,
  'best_params': {'n_estimators': 200,
   'learning_rate': 0.01,
   'natural_gradient': True,
   'minibatch_frac': 0.9,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 167575.8434276094,
  'test_rmse': 409.36028560133843,
  'test_corr_coef': 0.9636766208312282,
  'pruner': 'ThresholdPruner'},
 ('NGBoost', 'WilcoxonPruner'): {'best_score': 152421.19764256236,
  'best_params': {'n_estimators': 1000,
   'learning_rate': 0.05,
   'natural_gradient': True,
   'minibatch_frac': 0.5,
   'col_sample': 0.9,
   'Dist': ngboost.distns.normal.Normal,
   'Score': ngboost.scores.LogScore,
   'random_state': 42,
   'verbose': 0},
  'test_mse': 152421.19764256236,
  'test_rmse': 390.4115746780087,
  'test_corr_coef': 0.9569635429927064,
  'pruner': 'WilcoxonPruner'},
 ('TabNet', 'MedianPruner'): {'best_score': 320466.64675731334,
  'best_params': {'n_d': 64,
   'n_a': 8,
   'n_steps': 5,
   'gamma': 2.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 320466.64675731334,
  'test_rmse': 566.0977360468008,
  'test_corr_coef': 0.9590577932813041,
  'pruner': 'MedianPruner'},
 ('TabNet', 'NopPruner'): {'best_score': 262884.08066848107,
  'best_params': {'n_d': 32,
   'n_a': 64,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 262884.08066848107,
  'test_rmse': 512.7222256431654,
  'test_corr_coef': 0.9737017384987599,
  'pruner': 'NopPruner'},
 ('TabNet', 'PatientPruner'): {'best_score': 234676.12036911483,
  'best_params': {'n_d': 32,
   'n_a': 64,
   'n_steps': 10,
   'gamma': 1.5,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 3,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 234676.12036911483,
  'test_rmse': 484.43381422967866,
  'test_corr_coef': 0.9577097500704689,
  'pruner': 'PatientPruner'},
 ('TabNet', 'PercentilePruner'): {'best_score': 356553.65101214184,
  'best_params': {'n_d': 64,
   'n_a': 64,
   'n_steps': 5,
   'gamma': 1.5,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 1,
   'n_independent': 3,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 356553.65101214184,
  'test_rmse': 597.1211359616589,
  'test_corr_coef': 0.9440491878609079,
  'pruner': 'PercentilePruner'},
 ('TabNet', 'SuccessiveHalvingPruner'): {'best_score': 115329.83105861585,
  'best_params': {'n_d': 64,
   'n_a': 64,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'sparsemax',
   'n_shared': 2,
   'n_independent': 3,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 115329.83105861585,
  'test_rmse': 339.6024603247389,
  'test_corr_coef': 0.9695843406490969,
  'pruner': 'SuccessiveHalvingPruner'},
 ('TabNet', 'HyperbandPruner'): {'best_score': 234574.8695731652,
  'best_params': {'n_d': 32,
   'n_a': 32,
   'n_steps': 7,
   'gamma': 1.0,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 3,
   'n_independent': 3,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 234574.8695731652,
  'test_rmse': 484.32929869373504,
  'test_corr_coef': 0.9585848067836743,
  'pruner': 'HyperbandPruner'},
 ('TabNet', 'ThresholdPruner'): {'best_score': 197564.61639049702,
  'best_params': {'n_d': 64,
   'n_a': 64,
   'n_steps': 10,
   'gamma': 1.5,
   'lambda_sparse': 0.001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 2,
   'n_independent': 3,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 197564.61639049702,
  'test_rmse': 444.48241403962993,
  'test_corr_coef': 0.9557576961290063,
  'pruner': 'ThresholdPruner'},
 ('TabNet', 'WilcoxonPruner'): {'best_score': 136359.22694099968,
  'best_params': {'n_d': 64,
   'n_a': 64,
   'n_steps': 10,
   'gamma': 1.0,
   'lambda_sparse': 0.0001,
   'optimizer_params': {'lr': 0.02},
   'mask_type': 'entmax',
   'n_shared': 3,
   'n_independent': 2,
   'scheduler_params': {'step_size': 10, 'gamma': 0.9},
   'scheduler_fn': torch.optim.lr_scheduler.StepLR,
   'seed': 42,
   'verbose': 0},
  'test_mse': 136359.22694099968,
  'test_rmse': 369.26850250325936,
  'test_corr_coef': 0.9572939602193796,
  'pruner': 'WilcoxonPruner'},
 ('HistGradientBoosting', 'MedianPruner'): {'best_score': 97200.1226064393,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 500,
   'max_depth': None,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.1,
   'max_bins': 255,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 97200.1226064393,
  'test_rmse': 311.7693419925046,
  'test_corr_coef': 0.9718101341850087,
  'pruner': 'MedianPruner'},
 ('HistGradientBoosting', 'NopPruner'): {'best_score': 121844.6068738407,
  'best_params': {'learning_rate': 0.01,
   'max_iter': 400,
   'max_depth': 3,
   'min_samples_leaf': 5,
   'max_leaf_nodes': None,
   'l2_regularization': 0.0,
   'max_bins': 255,
   'early_stopping': False,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 121844.6068738407,
  'test_rmse': 349.06246844059405,
  'test_corr_coef': 0.9607058811367111,
  'pruner': 'NopPruner'},
 ('HistGradientBoosting', 'PatientPruner'): {'best_score': 98723.42448097974,
  'best_params': {'learning_rate': 0.15,
   'max_iter': 400,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 63,
   'l2_regularization': 0.5,
   'max_bins': 255,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 98723.42448097974,
  'test_rmse': 314.20283970865023,
  'test_corr_coef': 0.9672188283074789,
  'pruner': 'PatientPruner'},
 ('HistGradientBoosting',
  'PercentilePruner'): {'best_score': 98723.42448097974, 'best_params': {'learning_rate': 0.15,
   'max_iter': 500,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 63,
   'l2_regularization': 0.5,
   'max_bins': 255,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 98723.42448097974, 'test_rmse': 314.20283970865023, 'test_corr_coef': 0.9672188283074789, 'pruner': 'PercentilePruner'},
 ('HistGradientBoosting',
  'SuccessiveHalvingPruner'): {'best_score': 112723.87184121429, 'best_params': {'learning_rate': 0.15,
   'max_iter': 100,
   'max_depth': None,
   'min_samples_leaf': 20,
   'max_leaf_nodes': 31,
   'l2_regularization': 0.0,
   'max_bins': 128,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 112723.87184121429, 'test_rmse': 335.7437591992058, 'test_corr_coef': 0.963128045661365, 'pruner': 'SuccessiveHalvingPruner'},
 ('HistGradientBoosting',
  'HyperbandPruner'): {'best_score': 101709.92992105405, 'best_params': {'learning_rate': 0.05,
   'max_iter': 200,
   'max_depth': 3,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 1.0,
   'max_bins': 255,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 10,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0}, 'test_mse': 101709.92992105405, 'test_rmse': 318.91994280862093, 'test_corr_coef': 0.9688923783881614, 'pruner': 'HyperbandPruner'},
 ('HistGradientBoosting', 'ThresholdPruner'): {'best_score': 98801.3336264882,
  'best_params': {'learning_rate': 0.15,
   'max_iter': 300,
   'max_depth': 5,
   'min_samples_leaf': 10,
   'max_leaf_nodes': None,
   'l2_regularization': 0.1,
   'max_bins': 255,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 5,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 98801.3336264882,
  'test_rmse': 314.3267943184103,
  'test_corr_coef': 0.9700348280080318,
  'pruner': 'ThresholdPruner'},
 ('HistGradientBoosting', 'WilcoxonPruner'): {'best_score': 104614.56815861314,
  'best_params': {'learning_rate': 0.1,
   'max_iter': 500,
   'max_depth': None,
   'min_samples_leaf': 10,
   'max_leaf_nodes': 15,
   'l2_regularization': 0.5,
   'max_bins': 255,
   'early_stopping': True,
   'validation_fraction': 0.1,
   'n_iter_no_change': 15,
   'loss': 'squared_error',
   'random_state': 42,
   'verbose': 0},
  'test_mse': 104614.56815861314,
  'test_rmse': 323.4417538887228,
  'test_corr_coef': 0.9706866758128522,
  'pruner': 'WilcoxonPruner'},
 ('PGBM', 'MedianPruner'): {'best_score': 92073.78363045836,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 52,
   'min_split_gain': 1.0,
   'reg_lambda': 5.0,
   'feature_fraction': 0.9,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 3,
   'max_bin': 128,
   'distribution': 'laplace'},
  'test_mse': 92073.78363045836,
  'test_rmse': 303.4366220983525,
  'test_corr_coef': 0.9693307274788633,
  'pruner': 'MedianPruner'},
 ('PGBM', 'NopPruner'): {'best_score': 91450.71937529996,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_leaves': 42,
   'min_split_gain': 0.0,
   'reg_lambda': 0.1,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.2,
   'min_data_in_leaf': 10,
   'max_bin': 128,
   'distribution': 'normal'},
  'test_mse': 91450.71937529996,
  'test_rmse': 302.40819991412263,
  'test_corr_coef': 0.9696453678440501,
  'pruner': 'NopPruner'},
 ('PGBM', 'PatientPruner'): {'best_score': 88349.57707765196,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 40,
   'min_split_gain': 0.5,
   'reg_lambda': 5.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 10,
   'max_bin': 64,
   'distribution': 'normal'},
  'test_mse': 88349.57707765196,
  'test_rmse': 297.2365675310694,
  'test_corr_coef': 0.971513544690967,
  'pruner': 'PatientPruner'},
 ('PGBM', 'PercentilePruner'): {'best_score': 90954.58096772693,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_leaves': 44,
   'min_split_gain': 0.5,
   'reg_lambda': 1.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 10,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 90954.58096772693,
  'test_rmse': 301.5867718712592,
  'test_corr_coef': 0.9710871704344548,
  'pruner': 'PercentilePruner'},
 ('PGBM', 'SuccessiveHalvingPruner'): {'best_score': 88349.46474146072,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 49,
   'min_split_gain': 0.0,
   'reg_lambda': 5.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 10,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 88349.46474146072,
  'test_rmse': 297.23637856335944,
  'test_corr_coef': 0.9715135648364278,
  'pruner': 'SuccessiveHalvingPruner'},
 ('PGBM', 'HyperbandPruner'): {'best_score': 90948.86630655531,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_leaves': 28,
   'min_split_gain': 0.1,
   'reg_lambda': 1.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.5,
   'tree_correlation': 0.3,
   'min_data_in_leaf': 10,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 90948.86630655531,
  'test_rmse': 301.57729739911673,
  'test_corr_coef': 0.9710884818595122,
  'pruner': 'HyperbandPruner'},
 ('PGBM', 'ThresholdPruner'): {'best_score': 88349.46474146072,
  'best_params': {'n_estimators': 500,
   'learning_rate': 0.01,
   'max_leaves': 57,
   'min_split_gain': 0.1,
   'reg_lambda': 5.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.1,
   'min_data_in_leaf': 10,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 88349.46474146072,
  'test_rmse': 297.23637856335944,
  'test_corr_coef': 0.9715135648364278,
  'pruner': 'ThresholdPruner'},
 ('PGBM', 'WilcoxonPruner'): {'best_score': 86773.2618088813,
  'best_params': {'n_estimators': 100,
   'learning_rate': 0.05,
   'max_leaves': 34,
   'min_split_gain': 0.0,
   'reg_lambda': 10.0,
   'feature_fraction': 1.0,
   'bagging_fraction': 0.7,
   'tree_correlation': 0.0,
   'min_data_in_leaf': 10,
   'max_bin': 64,
   'distribution': 'studentt'},
  'test_mse': 86773.2618088813,
  'test_rmse': 294.57301609088586,
  'test_corr_coef': 0.9713851074419442,
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
generate_and_save_plots(predictions_LGBM_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/LightGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/LightGBM.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/LightGBM.xlsx",
    model_name="LightGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_LGBM_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/LightGBM.xlsx",
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
generate_and_save_plots(predictions_XGB_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/XGBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/XGBoost.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/XGBoost.xlsx",
    model_name="XGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_XGB_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/XGBoost.xlsx",
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
generate_and_save_plots(predictions_GPBoost_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/GPBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/GPBoost.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/GPBoost.xlsx",
    model_name="GPBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GPBoost_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/GPBoost.xlsx",
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
generate_and_save_plots(predictions_NGB_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/NGBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/NGBoost.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/NGBoost.xlsx",
    model_name="NGBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_NGB_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/NGBoost.xlsx",
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
generate_and_save_plots(predictions_GBR_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/Gradient Boosting.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/Gradient Boosting.xlsx",
    model_name="Gradient Boosting",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/Gradient Boosting.xlsx",
    model_name="GBoosting",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_GBR_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/Gradient Boosting.xlsx",
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
generate_and_save_plots(predictions_CatBoost_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/CatBoost.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="CatBoost",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/CatBoost.xlsx",
    model_name="CatBoost",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_CatBoost_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/CatBoost.xlsx",
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
generate_and_save_plots(predictions_HGBR_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/HGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/HGBM.xlsx",
    model_name="HGBR",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/HGBM.xlsx",
    model_name="HGBR",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_HGBR_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/HGBM.xlsx",
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
generate_and_save_plots(predictions_TabNet_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/TabNet.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/TabNet.xlsx",
    model_name="TabNet",
    n_bins=10
)

```

```python
generate_and_save_pit_results(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/TabNet.xlsx",
    model_name="TabNet",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_TabNet_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/CatBoost.xlsx",
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
generate_and_save_plots(predictions_PGBM_df, y_test, "./drive/MyDrive/Regression/probabilistic_distribution_card/PGBM.xlsx")
```

```python
generate_and_save_crps_decomposition(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_pit_results(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_bins=10
)
```

```python
generate_and_save_calibration_curve(
    predictions_PGBM_df,
    y_test,
    "./drive/MyDrive/Regression/probabilistic_distribution_card/PGBM.xlsx",
    model_name="PGBM",
    n_quantiles=10
)
```

# **Matrix Evaulation**

```python
# Create a Pandas Excel writer using XlsxWriter as the engine.
folder_path = "/content/drive/MyDrive/Regression/"
excel_path = os.path.join(folder_path, "/content/drive/MyDrive/Regression/probabilistic_distribution_card/Matrix Evaluation.xlsx")
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
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

