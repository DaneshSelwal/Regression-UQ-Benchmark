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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/LightGBM.xlsx",
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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/XGBoost.xlsx",
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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/GPBoost.xlsx",
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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/NGBoost.xlsx",
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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/Gradient Boosting.xlsx",
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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/CatBoost.xlsx",
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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/HGBM.xlsx",
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
    excel_file_path="./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/PGBM.xlsx",
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
    "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/TabNet.xlsx",
)

```

