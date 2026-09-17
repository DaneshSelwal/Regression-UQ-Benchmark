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
from sklearn.model_selection import train_test_split
from typing_extensions import TypedDict
from typing import Union
from sklearn.model_selection import KFold
from PIL import Image as PImage
from openpyxl.utils.dataframe import dataframe_to_rows
from pytorch_tabnet import TabNetRegressor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_numpy_2d(x):
    arr = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32)

def _to_numpy_1d(y):
    arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    return arr.reshape(-1).astype(np.float32)

def _ensure_excel_workbook(excel_file_path):
    if not excel_file_path:
        return
    os.makedirs(os.path.dirname(excel_file_path), exist_ok=True)
    if not os.path.exists(excel_file_path):
        wb = Workbook()
        wb.active.title = "Init"
        wb.save(_ensure_parent_dir(excel_file_path))

def save_plot_to_excel(fig, excel_file_path, sheet_name):
    _ensure_excel_workbook(excel_file_path)
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = Image(PImage.open(buf))
        workbook = load_workbook(_ensure_excel_file(excel_file_path))
        base_name = sheet_name
        i = 1
        while sheet_name in workbook.sheetnames:
            sheet_name = f"{base_name}_{i}"
            i += 1
        worksheet = workbook.create_sheet(title=sheet_name)
        worksheet.add_image(img, "A1")
        workbook.save(_ensure_parent_dir(excel_file_path))

def save_values_to_excel(results_dict, excel_file_path, sheet_name):
    _ensure_excel_workbook(excel_file_path)
    df = pd.DataFrame(results_dict)
    workbook = load_workbook(_ensure_excel_file(excel_file_path))
    base_name = sheet_name
    i = 1
    while sheet_name in workbook.sheetnames:
        sheet_name = f"{base_name}_{i}"
        i += 1
    worksheet = workbook.create_sheet(title=sheet_name)
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            worksheet.cell(row=r_idx, column=c_idx, value=value)
    workbook.save(_ensure_parent_dir(excel_file_path))

def save_dataframe_to_excel(df, excel_file_path, sheet_name):
    _ensure_excel_workbook(excel_file_path)
    workbook = load_workbook(_ensure_excel_file(excel_file_path))
    base_name = sheet_name
    i = 1
    while sheet_name in workbook.sheetnames:
        sheet_name = f"{base_name}_{i}"
        i += 1
    worksheet = workbook.create_sheet(title=sheet_name)
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            worksheet.cell(row=r_idx, column=c_idx, value=value)
    workbook.save(_ensure_parent_dir(excel_file_path))

class AlphaNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def interval_size(scores, alpha, eps=1e-6):
    n = scores.numel()
    denominator = torch.clamp(alpha * (n + 1) - 1.0, min=eps)
    return 2.0 * scores.sum() / denominator

def _compute_scores_without_index(base_model, X_calib, y_calib, holdout_idx):
    loo_X = np.delete(X_calib, holdout_idx, axis=0)
    loo_y = np.delete(y_calib, holdout_idx, axis=0)
    return np.abs(base_model.predict(loo_X) - loo_y)

def _build_loo_feature_matrix(base_model, X_calib, y_calib):
    rows = []
    for idx in range(len(X_calib)):
        scores = _compute_scores_without_index(base_model, X_calib, y_calib, idx)
        rows.append([scores.sum()])
    return torch.tensor(np.asarray(rows, dtype=np.float32))

def _resolve_model_params(model_class, best_params):
    params = dict(best_params or {})
    if "TabNet" in model_class.__name__ and "verbose" in params:
        params.pop("verbose", None)
    try:
        model_class(**params)
        return params
    except Exception:
        return {}

def _train_alpha_net(
    base_model,
    X_calib,
    y_calib,
    lambdas=(10, 20, 50),
    num_runs=3,
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    alpha_clip=1e-3,
    seed=42,
):
    X_train_alpha = _build_loo_feature_matrix(base_model, X_calib, y_calib)
    all_results = {}
    for lambda_reg in lambdas:
        lambda_losses = []
        lambda_sizes = []
        lambda_alphas = []
        lambda_models = []
        for run_idx in range(num_runs):
            local_seed = seed + run_idx
            torch.manual_seed(local_seed)
            np.random.seed(local_seed)
            random.seed(local_seed)
            dataset = TensorDataset(X_train_alpha, torch.arange(len(X_train_alpha), dtype=torch.long))
            loader = DataLoader(dataset, batch_size=min(batch_size, len(X_train_alpha)), shuffle=True)
            alpha_net = AlphaNet(input_dim=X_train_alpha.shape[1]).to(DEVICE)
            optimizer = optim.Adam(alpha_net.parameters(), lr=learning_rate)
            all_losses = []
            all_sizes = []
            all_alphas = []
            for _ in range(epochs):
                epoch_losses = []
                epoch_sizes = []
                epoch_alphas = []
                alpha_net.train()
                for x_batch, idx_batch in loader:
                    x_batch = x_batch.to(DEVICE)
                    alpha_pred = torch.clamp(alpha_net(x_batch), min=alpha_clip, max=1.0 - alpha_clip)
                    batch_sizes = []
                    for j, idx in enumerate(idx_batch.tolist()):
                        scores_np = _compute_scores_without_index(base_model, X_calib, y_calib, idx)
                        scores_tensor = torch.tensor(scores_np, dtype=torch.float32, device=DEVICE)
                        batch_sizes.append(interval_size(scores_tensor, alpha_pred[j]))
                    batch_sizes = torch.stack(batch_sizes)
                    loss = (batch_sizes + lambda_reg * alpha_pred).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.item()))
                    epoch_sizes.append(float(batch_sizes.mean().item()))
                    epoch_alphas.append(float(alpha_pred.mean().item()))
                all_losses.append(float(np.mean(epoch_losses)))
                all_sizes.append(float(np.mean(epoch_sizes)))
                all_alphas.append(float(np.mean(epoch_alphas)))
            alpha_net.eval()
            lambda_losses.append(np.asarray(all_losses, dtype=np.float32))
            lambda_sizes.append(np.asarray(all_sizes, dtype=np.float32))
            lambda_alphas.append(np.asarray(all_alphas, dtype=np.float32))
            lambda_models.append(alpha_net)
        all_results[lambda_reg] = {
            "all_losses": np.vstack(lambda_losses),
            "all_sizes": np.vstack(lambda_sizes),
            "all_alphas": np.vstack(lambda_alphas),
            "models": lambda_models,
        }
    return all_results

def _moving_average(values, window=10):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="valid")

def _smooth_runs(run_matrix, window=10):
    smoothed_runs = [_moving_average(run_curve, window=window) for run_curve in np.asarray(run_matrix)]
    if not smoothed_runs:
        return np.array([]), np.array([])
    min_len = min(len(curve) for curve in smoothed_runs)
    if min_len == 0:
        return np.array([]), np.array([])
    stacked = np.asarray([curve[:min_len] for curve in smoothed_runs], dtype=float)
    return stacked.mean(axis=0), stacked.std(axis=0)

def plot_acp_training_dynamics(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for idx, (lambda_reg, result_dict) in enumerate(sorted(all_results.items(), key=lambda item: item[0])):
        color = colors[idx % len(colors)]
        loss_mean, loss_std = _smooth_runs(result_dict["all_losses"], window=10)
        size_mean, size_std = _smooth_runs(result_dict["all_sizes"], window=10)
        alpha_mean, alpha_std = _smooth_runs(result_dict["all_alphas"], window=10)
        if loss_mean.size == 0:
            continue
        epochs_axis = np.arange(1, len(loss_mean) + 1)
        axes[0].plot(epochs_axis, loss_mean, color=color, label=f"lambda={lambda_reg}")
        axes[0].fill_between(epochs_axis, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.2)
        axes[1].plot(epochs_axis, size_mean, color=color, label=f"lambda={lambda_reg}")
        axes[1].fill_between(epochs_axis, size_mean - size_std, size_mean + size_std, color=color, alpha=0.2)
        axes[2].plot(epochs_axis, alpha_mean, color=color, label=f"lambda={lambda_reg}")
        axes[2].fill_between(epochs_axis, alpha_mean - alpha_std, alpha_mean + alpha_std, color=color, alpha=0.2)
    axes[0].set_title("Smoothed Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Smoothed Mean Size")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Size")
    axes[2].set_title(r"Smoothed Mean $\tilde{\alpha}$")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel(r"Mean $\tilde{\alpha}$")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    return fig

def _select_best_alpha_net(all_results):
    best_lambda = None
    best_run_idx = None
    best_loss = float("inf")
    for lambda_reg, result_dict in all_results.items():
        final_losses = result_dict["all_losses"][:, -1]
        run_idx = int(np.argmin(final_losses))
        run_loss = float(final_losses[run_idx])
        if run_loss < best_loss:
            best_loss = run_loss
            best_lambda = lambda_reg
            best_run_idx = run_idx
    return best_lambda, best_run_idx, all_results[best_lambda]["models"][best_run_idx]

def evaluate_acp(model, alpha_net, X_calib, y_calib, X_test, y_test):
    X_calib_np = _to_numpy_2d(X_calib)
    y_calib_np = _to_numpy_1d(y_calib)
    X_test_np = _to_numpy_2d(X_test)
    y_test_np = _to_numpy_1d(y_test)
    calib_scores = np.abs(model.predict(X_calib_np) - y_calib_np)
    test_feature = torch.tensor([[calib_scores.sum()]], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        alpha_hat = float(torch.clamp(alpha_net(test_feature), min=1e-3, max=1.0 - 1e-3).item())
    score_tensor = torch.tensor(calib_scores, dtype=torch.float32, device=DEVICE)
    interval_width = float(interval_size(score_tensor, torch.tensor(alpha_hat, device=DEVICE)).item())
    y_pred = np.asarray(model.predict(X_test_np)).reshape(-1)
    lower = y_pred - 0.5 * interval_width
    upper = y_pred + 0.5 * interval_width
    coverage = float(np.mean((y_test_np >= lower) & (y_test_np <= upper)))
    rmse = float(np.sqrt(mean_squared_error(y_test_np, y_pred)))
    mae = float(np.mean(np.abs(y_test_np - y_pred)))
    return {
        "X_test": X_test_np,
        "y_test": y_test_np,
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
        "alpha": alpha_hat,
        "interval_width": interval_width,
        "coverage": coverage,
        "rmse": rmse,
        "mae": mae,
    }

def conformal_predictions_ACP(
    model_class,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path,
    alpha=0.1,
    lambdas=(10, 20, 50),
    num_runs=3,
    epochs=50,
):
    X_train_np = _to_numpy_2d(X_train)
    y_train_np = _to_numpy_1d(y_train)
    X_test_np = _to_numpy_2d(X_test)
    y_test_np = _to_numpy_1d(y_test)
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train_np,
        y_train_np,
        test_size=0.35,
        random_state=42,
    )
    safe_params = _resolve_model_params(model_class, best_params)
    model = model_class(**safe_params)
    model.fit(X_fit, y_fit)
    all_results = _train_alpha_net(
        base_model=model,
        X_calib=X_calib,
        y_calib=y_calib,
        lambdas=lambdas,
        num_runs=num_runs,
        epochs=epochs,
    )
    best_lambda, best_run_idx, alpha_net = _select_best_alpha_net(all_results)
    eval_bundle = evaluate_acp(
        model=model,
        alpha_net=alpha_net,
        X_calib=X_calib,
        y_calib=y_calib,
        X_test=X_test_np,
        y_test=y_test_np,
    )
    y_true = eval_bundle["y_test"]
    y_pred = eval_bundle["y_pred"]
    lower = eval_bundle["lower"]
    upper = eval_bundle["upper"]
    print(f"{model_name} ACP best lambda: {best_lambda}")
    print(f"{model_name} ACP best run: {best_run_idx + 1}")
    print(f"{model_name} ACP alpha: {eval_bundle['alpha']:.4f}")
    print(f"{model_name} ACP width: {eval_bundle['interval_width']:.4f}")
    print(f"{model_name} ACP coverage: {eval_bundle['coverage'] * 100:.2f}%")
    print(f"{model_name} ACP RMSE: {eval_bundle['rmse']:.4f}")
    print(f"{model_name} ACP MAE: {eval_bundle['mae']:.4f}")
    fig_pred, axs = plt.subplots(2, 1, figsize=(11, 12))
    idx = np.arange(len(y_true))
    axs[0].scatter(idx, y_true, label="True", color="blue", s=12, alpha=0.7)
    axs[0].fill_between(idx, lower, upper, color="gray", alpha=0.45, label="ACP Interval")
    axs[0].plot(idx, y_pred, color="red", linewidth=1.2, label="Prediction")
    axs[0].set_title(f"ACP Intervals - {model_name}")
    axs[0].set_xlabel("Sample Number")
    axs[0].set_ylabel("str")
    axs[0].legend(loc="upper left", fontsize="small", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    abs_err = np.abs(y_true - y_pred)
    widths = upper - lower
    axs[1].scatter(widths, abs_err, s=14, alpha=0.7, color="teal", label="Samples")
    axs[1].axvline(np.mean(widths), linestyle="--", color="gray", label="Mean Width")
    axs[1].axhline(np.mean(abs_err), linestyle="--", color="orange", label="Mean Absolute Error")
    axs[1].set_title(f"ACP Width vs Absolute Error - {model_name}")
    axs[1].set_xlabel("Interval Width")
    axs[1].set_ylabel("Absolute Error")
    axs[1].legend(loc="upper left", fontsize="small", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    plt.tight_layout()
    fig_dynamics = plot_acp_training_dynamics(all_results)
    if excel_file_path:
        save_plot_to_excel(fig_pred, excel_file_path, "conformal_predictions_ACP")
        save_plot_to_excel(fig_dynamics, excel_file_path, "acp_training_dynamics")
        save_values_to_excel({
                "ACP_true": y_true,
                "ACP_pred": y_pred,
                "ACP_lower": lower,
                "ACP_upper": upper,
                "ACP_alpha": np.full(len(y_true), eval_bundle["alpha"]),
                "ACP_interval_width": np.full(len(y_true), eval_bundle["interval_width"]),
                "ACP_is_covered": ((y_true >= lower) & (y_true <= upper)).astype(int),
                "ACP_best_lambda": np.full(len(y_true), best_lambda),
                "ACP_best_run": np.full(len(y_true), best_run_idx + 1),
            },
            excel_file_path,
            "conformal_predictions_ACP_values",
        )
        training_summary_rows = []
        training_curve_rows = []
        for lambda_reg, result_dict in sorted(all_results.items(), key=lambda item: item[0]):
            final_losses = result_dict["all_losses"][:, -1]
            final_sizes = result_dict["all_sizes"][:, -1]
            final_alphas = result_dict["all_alphas"][:, -1]
            training_summary_rows.append(
                {
                    "lambda": lambda_reg,
                    "runs": result_dict["all_losses"].shape[0],
                    "best_run": int(np.argmin(final_losses)) + 1,
                    "final_loss_mean": float(np.mean(final_losses)),
                    "final_loss_std": float(np.std(final_losses)),
                    "final_size_mean": float(np.mean(final_sizes)),
                    "final_size_std": float(np.std(final_sizes)),
                    "final_alpha_mean": float(np.mean(final_alphas)),
                    "final_alpha_std": float(np.std(final_alphas)),
                }
            )
            mean_loss_curve = result_dict["all_losses"].mean(axis=0)
            mean_size_curve = result_dict["all_sizes"].mean(axis=0)
            mean_alpha_curve = result_dict["all_alphas"].mean(axis=0)
            for epoch_idx in range(len(mean_loss_curve)):
                training_curve_rows.append(
                    {
                        "lambda": lambda_reg,
                        "epoch": epoch_idx + 1,
                        "mean_loss": float(mean_loss_curve[epoch_idx]),
                        "mean_size": float(mean_size_curve[epoch_idx]),
                        "mean_alpha": float(mean_alpha_curve[epoch_idx]),
                    }
                )
        save_dataframe_to_excel(pd.DataFrame(training_summary_rows), excel_file_path, "acp_training_summary")
        save_dataframe_to_excel(pd.DataFrame(training_curve_rows), excel_file_path, "acp_training_curves")
    plt.close(fig_pred)
    plt.close(fig_dynamics)
    eval_bundle["model"] = model
    eval_bundle["alpha_net"] = alpha_net
    eval_bundle["all_results"] = all_results
    eval_bundle["best_lambda"] = best_lambda
    eval_bundle["best_run"] = best_run_idx + 1
    return eval_bundle

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

def _train_alpha_net_multi(
    base_model,
    X_calib,
    y_calib,
    X_train_alpha,
    lambdas=(10, 20, 50),
    num_runs=3,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    alpha_clip=1e-3,
    seed=42,
):
    Dataset = TensorDataset(X_train_alpha, torch.arange(len(X_train_alpha), dtype=torch.long))
    loader = DataLoader(Dataset, batch_size=min(batch_size, len(X_train_alpha)), shuffle=True)

    all_results = {}
    for lambda_reg in lambdas:
        lambda_losses = []
        lambda_sizes = []
        lambda_alphas = []
        lambda_models = []

        for run_idx in range(num_runs):
            torch.manual_seed(seed + run_idx)
            np.random.seed(seed + run_idx)

            alpha_net = AlphaNet(input_dim=X_train_alpha.shape[1]).to(DEVICE)
            optimizer = optim.Adam(alpha_net.parameters(), lr=lr)

            run_losses = []
            run_sizes = []
            run_alphas = []

            for _ in range(epochs):
                alpha_net.train()
                epoch_losses = []
                epoch_sizes = []
                epoch_alphas = []

                for x_batch, idx_batch in loader:
                    x_batch = x_batch.to(DEVICE)
                    alpha_pred = torch.clamp(alpha_net(x_batch), min=alpha_clip, max=1.0 - alpha_clip)

                    batch_sizes = []
                    for j, idx in enumerate(idx_batch.tolist()):
                        scores_np = _compute_scores_without_index(base_model, X_calib, y_calib, idx)
                        scores_tensor = torch.tensor(scores_np, dtype=torch.float32, device=DEVICE)
                        batch_sizes.append(interval_size(scores_tensor, alpha_pred[j]))

                    batch_sizes = torch.stack(batch_sizes)
                    loss = (batch_sizes + float(lambda_reg) * alpha_pred).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(float(loss.item()))
                    epoch_sizes.append(float(batch_sizes.mean().item()))
                    epoch_alphas.append(float(alpha_pred.mean().item()))

                run_losses.append(float(np.mean(epoch_losses)))
                run_sizes.append(float(np.mean(epoch_sizes)))
                run_alphas.append(float(np.mean(epoch_alphas)))

            alpha_net.eval()
            lambda_losses.append(run_losses)
            lambda_sizes.append(run_sizes)
            lambda_alphas.append(run_alphas)
            lambda_models.append(alpha_net)

        all_results[float(lambda_reg)] = {
            "all_losses": np.asarray(lambda_losses, dtype=np.float32),
            "all_sizes": np.asarray(lambda_sizes, dtype=np.float32),
            "all_alphas": np.asarray(lambda_alphas, dtype=np.float32),
            "models": lambda_models,
        }

    return all_results


def _moving_average(values, window=10):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def _smooth_runs(run_matrix, window=10):
    run_matrix = np.asarray(run_matrix, dtype=np.float32)
    if run_matrix.ndim == 1:
        run_matrix = run_matrix.reshape(1, -1)
    if run_matrix.size == 0:
        return np.asarray([]), np.asarray([])

    smoothed_runs = [_moving_average(run, window=window) for run in run_matrix]
    min_len = min(len(curve) for curve in smoothed_runs)
    if min_len == 0:
        return np.asarray([]), np.asarray([])

    trimmed = np.vstack([curve[:min_len] for curve in smoothed_runs])
    return trimmed.mean(axis=0), trimmed.std(axis=0)


def plot_acp_training_dynamics(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if not all_results:
        for ax in axes:
            ax.text(0.5, 0.5, "No training results", ha="center", va="center")
            ax.axis("off")
        return fig

    sorted_results = sorted(all_results.items(), key=lambda item: item[0])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_results)))

    for color, (lambda_reg, result) in zip(colors, sorted_results):
        loss_mean, loss_std = _smooth_runs(result["all_losses"], window=10)
        size_mean, size_std = _smooth_runs(result["all_sizes"], window=10)
        alpha_mean, alpha_std = _smooth_runs(result["all_alphas"], window=10)

        if loss_mean.size:
            x_loss = np.arange(1, loss_mean.size + 1)
            axes[0].plot(x_loss, loss_mean, color=color, label=f"lambda={lambda_reg:g}")
            axes[0].fill_between(x_loss, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.2)

        if size_mean.size:
            x_size = np.arange(1, size_mean.size + 1)
            axes[1].plot(x_size, size_mean, color=color, label=f"lambda={lambda_reg:g}")
            axes[1].fill_between(x_size, size_mean - size_std, size_mean + size_std, color=color, alpha=0.2)

        if alpha_mean.size:
            x_alpha = np.arange(1, alpha_mean.size + 1)
            axes[2].plot(x_alpha, alpha_mean, color=color, label=f"lambda={lambda_reg:g}")
            axes[2].fill_between(x_alpha, alpha_mean - alpha_std, alpha_mean + alpha_std, color=color, alpha=0.2)

    axes[0].set_title("Smoothed Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Smoothed Mean Size")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Size")

    axes[2].set_title("Smoothed Mean Alpha-tilde")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Alpha-tilde")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize="small")

    plt.tight_layout()
    return fig


def _select_best_alpha_net(all_results):
    best_choice = None
    for lambda_reg, result in all_results.items():
        losses = result["all_losses"]
        if losses.size == 0:
            continue
        final_losses = losses[:, -1]
        run_idx = int(np.argmin(final_losses))
        final_loss = float(final_losses[run_idx])
        if best_choice is None or final_loss < best_choice["final_loss"]:
            best_choice = {
                "lambda": lambda_reg,
                "run_idx": run_idx,
                "final_loss": final_loss,
                "model": result["models"][run_idx],
            }

    if best_choice is None:
        raise RuntimeError("AlphaNet training produced no valid runs.")

    return best_choice


def evaluate_acp(model, alpha_net, X_calib, y_calib, X_test, y_test):
    X_calib_np = _to_numpy_2d(X_calib)
    y_calib_np = _to_numpy_1d(y_calib)
    X_test_np = _to_numpy_2d(X_test)
    y_test_np = _to_numpy_1d(y_test)

    calib_preds = np.asarray(model.predict(X_calib_np)).reshape(-1)
    calib_scores = np.abs(calib_preds - y_calib_np)

    feature = torch.tensor([[calib_scores.sum()]], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        alpha_hat = float(torch.clamp(alpha_net(feature), min=1e-3, max=1.0 - 1e-3).item())

    scores_tensor = torch.tensor(calib_scores, dtype=torch.float32, device=DEVICE)
    interval_width = float(
        interval_size(scores_tensor, torch.tensor(alpha_hat, dtype=torch.float32, device=DEVICE)).item()
    )

    y_pred = np.asarray(model.predict(X_test_np)).reshape(-1)
    lower = y_pred - 0.5 * interval_width
    upper = y_pred + 0.5 * interval_width

    coverage = float(np.mean((y_test_np >= lower) & (y_test_np <= upper)))
    rmse = float(np.sqrt(mean_squared_error(y_test_np, y_pred)))
    mae = float(np.mean(np.abs(y_test_np - y_pred)))

    return {
        "coverage": coverage,
        "interval_width": interval_width,
        "alpha_hat": alpha_hat,
        "rmse": rmse,
        "mae": mae,
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
        "y_test": y_test_np,
    }


def run_full_acp_pipeline(
    model_class,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    excel_file_path,
    lambdas=(10, 20, 50),
    num_runs=3,
    epochs=50,
):
    X_train_np = _to_numpy_2d(X_train)
    y_train_np = _to_numpy_1d(y_train)
    X_test_np = _to_numpy_2d(X_test)
    y_test_np = _to_numpy_1d(y_test)

    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train_np, y_train_np, test_size=0.35, random_state=42
    )

    safe_params = _resolve_model_params(model_class, best_params)
    model = model_class(**safe_params)
    model.fit(X_fit, y_fit)

    X_train_alpha = _build_loo_feature_matrix(model, X_calib, y_calib)
    all_results = _train_alpha_net_multi(
        base_model=model,
        X_calib=X_calib,
        y_calib=y_calib,
        X_train_alpha=X_train_alpha,
        lambdas=lambdas,
        num_runs=num_runs,
        epochs=epochs,
    )

    dynamics_fig = plot_acp_training_dynamics(all_results)
    dynamics_fig.suptitle(f"ACP Training Dynamics - {model_name}")
    dynamics_fig.tight_layout(rect=[0, 0, 1, 0.95])
    if excel_file_path:
        save_plot_to_excel(dynamics_fig, excel_file_path, "acp_training_dynamics")
    plt.close(dynamics_fig)

    best_choice = _select_best_alpha_net(all_results)
    metrics = evaluate_acp(
        model=model,
        alpha_net=best_choice["model"],
        X_calib=X_calib,
        y_calib=y_calib,
        X_test=X_test_np,
        y_test=y_test_np,
    )

    y_true = metrics["y_test"]
    y_pred = metrics["y_pred"]
    lower = metrics["lower"]
    upper = metrics["upper"]

    print(f"{model_name} best lambda: {best_choice['lambda']}")
    print(f"{model_name} best run: {best_choice['run_idx']}")
    print(f"{model_name} ACP alpha_hat: {metrics['alpha_hat']:.4f}")
    print(f"{model_name} ACP width: {metrics['interval_width']:.4f}")
    print(f"{model_name} ACP coverage: {metrics['coverage'] * 100:.2f}%")
    print(f"{model_name} ACP RMSE: {metrics['rmse']:.4f}")
    print(f"{model_name} ACP MAE: {metrics['mae']:.4f}")

    fig, axs = plt.subplots(2, 1, figsize=(11, 12))
    idx = np.arange(len(y_true))

    axs[0].scatter(idx, y_true, label="True", color="blue", s=12, alpha=0.7)
    axs[0].fill_between(idx, lower, upper, color="gray", alpha=0.45, label="ACP Interval")
    axs[0].plot(idx, y_pred, color="red", linewidth=1.2, label="Prediction")
    axs[0].set_title(f"ACP Intervals - {model_name}")
    axs[0].set_xlabel("Sample Number")
    axs[0].set_ylabel("str")
    axs[0].legend(loc="upper left", fontsize="small", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    abs_err = np.abs(y_true - y_pred)
    widths = upper - lower
    axs[1].scatter(widths, abs_err, s=14, alpha=0.7, color="teal", label="Samples")
    axs[1].axvline(np.mean(widths), linestyle="--", color="gray", label="Mean Width")
    axs[1].axhline(np.mean(abs_err), linestyle="--", color="orange", label="Mean Absolute Error")
    axs[1].set_title(f"ACP Width vs Absolute Error - {model_name}")
    axs[1].set_xlabel("Interval Width")
    axs[1].set_ylabel("Absolute Error")
    axs[1].legend(loc="upper left", fontsize="small", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    plt.tight_layout()

    if excel_file_path:
        save_plot_to_excel(fig, excel_file_path, "conformal_predictions_ACP")

        save_values_to_excel({
                "ACP_true": y_true,
                "ACP_pred": y_pred,
                "ACP_lower": lower,
                "ACP_upper": upper,
                "ACP_alpha": np.full(len(y_true), metrics["alpha_hat"]),
                "ACP_interval_width": np.full(len(y_true), metrics["interval_width"]),
                "ACP_is_covered": ((y_true >= lower) & (y_true <= upper)).astype(int),
            },
            excel_file_path,
            "conformal_predictions_ACP_values",
        )

        save_values_to_excel({
                "model_name": [model_name],
                "best_lambda": [best_choice["lambda"]],
                "best_run": [best_choice["run_idx"]],
                "best_final_loss": [best_choice["final_loss"]],
                "coverage": [metrics["coverage"]],
                "interval_width": [metrics["interval_width"]],
                "rmse": [metrics["rmse"]],
                "mae": [metrics["mae"]],
                "alpha_hat": [metrics["alpha_hat"]],
            },
            excel_file_path,
            "acp_final_metrics",
        )

        summary = {
            "lambda": [],
            "mean_final_loss": [],
            "std_final_loss": [],
            "mean_final_size": [],
            "std_final_size": [],
            "mean_final_alpha": [],
            "std_final_alpha": [],
        }
        curve_data = {
            "lambda": [],
            "run": [],
            "epoch": [],
            "loss": [],
            "size": [],
            "alpha_tilde": [],
        }

        for lambda_reg, result in sorted(all_results.items(), key=lambda item: item[0]):
            losses = result["all_losses"]
            sizes = result["all_sizes"]
            alphas = result["all_alphas"]

            summary["lambda"].append(lambda_reg)
            summary["mean_final_loss"].append(float(losses[:, -1].mean()))
            summary["std_final_loss"].append(float(losses[:, -1].std()))
            summary["mean_final_size"].append(float(sizes[:, -1].mean()))
            summary["std_final_size"].append(float(sizes[:, -1].std()))
            summary["mean_final_alpha"].append(float(alphas[:, -1].mean()))
            summary["std_final_alpha"].append(float(alphas[:, -1].std()))

            n_runs, n_epochs = losses.shape
            for run_idx in range(n_runs):
                for epoch_idx in range(n_epochs):
                    curve_data["lambda"].append(lambda_reg)
                    curve_data["run"].append(run_idx)
                    curve_data["epoch"].append(epoch_idx + 1)
                    curve_data["loss"].append(float(losses[run_idx, epoch_idx]))
                    curve_data["size"].append(float(sizes[run_idx, epoch_idx]))
                    curve_data["alpha_tilde"].append(float(alphas[run_idx, epoch_idx]))

        save_values_to_excel(summary, excel_file_path, "acp_training_summary")
        save_values_to_excel(curve_data, excel_file_path, "acp_training_curves")

    plt.close(fig)

    return {
        "model": model,
        "all_results": all_results,
        "best_choice": best_choice,
        "metrics": metrics,
    }

```

```python
# Unified ACP execution entry point:
# run_full_acp_pipeline(model_class, best_params, X_train, y_train, X_test, y_test, model_name, excel_file_path)

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
run_full_acp_pipeline(
    LGBMRegressor,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "LightGBM",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/LightGBM.xlsx",
)

```

# **Conformal Predictions with XGBoost**


```python
best_params = get_best_model_params(best_scores_autosampler, 'XGBoost')

```

```python
run_full_acp_pipeline(
    XGBRegressor,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "XGBoost",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/XGBoost.xlsx",
)

```

# **Conformal Predictions with GPBoost**


```python
best_params = get_best_model_params(best_scores_autosampler, 'GPBoost')

```

```python
run_full_acp_pipeline(
    GPBoostRegressor,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "GPBoost",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/GPBoost.xlsx",
)

```

# **Conformal Predictions with NGBoost**


```python
best_params = get_best_model_params(best_scores_autosampler, 'NGBoost')

```

```python
run_full_acp_pipeline(
    NGBRegressor,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "NGBoost",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/NGBoost.xlsx",
)

```

# **Conformal Predictions with Gradient Boosting**


```python
best_params = get_best_model_params(best_scores_autosampler, 'GradientBoosting')

```

```python
run_full_acp_pipeline(
    GradientBoostingRegressor,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "GradientBoosting",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/Gradient Boosting.xlsx",
)

```

# **Conformal Predictions with CatBoost**


```python
best_params = get_best_model_params(best_scores_autosampler, 'CatBoost')

```

```python
run_full_acp_pipeline(
    CatBoostRegressor,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "CatBoost",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/CatBoost.xlsx",
)

```

# **Conformal Predictions with HistGradientBoosting**


```python
best_params = get_best_model_params(best_scores_autosampler, 'HGBR')

```

```python
run_full_acp_pipeline(
    HistGradientBoostingRegressor,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "HGBR",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/HGBM.xlsx",
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

# Add actual target values to the DataFrame
predictions_PGBM_df['Actual'] = y_test.ravel()

# Print the DataFrame with predictions for each quantile
print(predictions_PGBM_df.head())

```

```python
# ACP execution for PGBM
best_params = get_best_model_params(best_scores_autosampler, "PGBM")
run_full_acp_pipeline(
    model_class=PGBMWrapper,
    best_params=best_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="PGBM",
    excel_file_path="./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/PGBM.xlsx",
)

```

# **Conformal Predictions with TabNet**


```python
best_params = get_best_model_params(best_scores_autosampler, 'TabNet')
print(best_params)

```

```python
class TabNetRegressorCP(TabNetRegressor):
    def fit(self, X, y, *args, **kwargs):
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        return super().fit(X, y_arr, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        preds = super().predict(X, *args, **kwargs)
        return np.asarray(preds).reshape(-1)


best_params = get_best_model_params(best_scores_autosampler, "TabNet")
if isinstance(best_params, dict) and "verbose" in best_params:
    best_params = dict(best_params)
    best_params.pop("verbose", None)

run_full_acp_pipeline(
    TabNetRegressorCP,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    "TabNet",
    "./drive/MyDrive/Regression/scour_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/TabNet.xlsx",
)

```

