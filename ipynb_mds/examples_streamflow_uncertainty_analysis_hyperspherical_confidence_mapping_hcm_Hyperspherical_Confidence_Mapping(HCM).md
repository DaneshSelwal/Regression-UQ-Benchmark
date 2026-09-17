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
import io
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import properscoring as ps
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

```

```python
# Go to find & replace button and replace (streamflow_uncertainty_analysis) with your folder name. Rename your train and test Dataset as train.csv and test.csv.
# Modify the names of the feature in the below cell.
# Replace (str) with actual data label name.
```

```python
feature_names = ['A', 'B', 'C', 'D', 'O']

```

```python
train_data_path = "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/data/train.csv"
test_data_path = "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/data/test.csv"
output_folder = "./drive/MyDrive/Regression/streamflow_uncertainty_analysis/hyperspherical_confidence_mapping_hcm"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_demo_dataframe(n_rows, feature_names, seed=42, is_train=True):
    rng = np.random.default_rng(seed + (0 if is_train else 99))
    data = {name: rng.normal(loc=0.0, scale=1.0, size=n_rows) for name in feature_names}
    frame = pd.DataFrame(data)
    nonlinear_term = 0.8 * np.sin(frame[feature_names[0]].values)
    interaction_term = 0.5 * frame[feature_names[1]].values * frame[feature_names[2]].values
    trend_term = 0.3 * frame[feature_names[0]].values ** 2
    noise = rng.normal(loc=0.0, scale=0.35 if is_train else 0.4, size=n_rows)
    frame['Target'] = 12.0 + 3.2 * frame[feature_names[0]].values - 1.7 * frame[feature_names[1]].values + 2.1 * frame[feature_names[2]].values + nonlinear_term + interaction_term - trend_term + noise
    return frame


def read_csv_from_candidates(path_candidates):
    for path in path_candidates:
        path = Path(path)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if not df.empty:
                print(f"Loaded data from: {path}")
                return df
        except Exception:
            continue
    return pd.DataFrame()
random_seed = 42

set_seed(random_seed)
train_demo = build_demo_dataframe(240, feature_names, seed=random_seed, is_train=True)
test_demo = build_demo_dataframe(80, feature_names, seed=random_seed, is_train=False)

train_candidates = [
    train_data_path,
    "/content/drive/MyDrive/Regression/streamflow_uncertainty_analysis/data/train.csv",
    "streamflow_uncertainty_analysis/data/train.csv"
]
test_candidates = [
    test_data_path,
    "/content/drive/MyDrive/Regression/streamflow_uncertainty_analysis/data/test.csv",
    "streamflow_uncertainty_analysis/data/test.csv"
]

train_data = read_csv_from_candidates(train_candidates)
test_data = read_csv_from_candidates(test_candidates)

if train_data.empty:
    print("Warning: training CSV is missing or empty. A generated demo Dataset will be used so the notebook can still run in Google Colab.")
    train_data = train_demo.copy()
if test_data.empty:
    print("Warning: testing CSV is missing or empty. A generated demo Dataset will be used so the notebook can still run in Google Colab.")
    test_data = test_demo.copy()

```

```python
print("\nShape of training data:", train_data.shape)
print("First 5 rows of training data:\n", train_data.head(5))
print("\nShape of test data:", test_data.shape)
print("First 5 rows of test data:\n", test_data.head(5))
```

```python
X_full = train_data.iloc[:, :-1]

y_full = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]

y_test = test_data.iloc[:, -1]

x_test = X_test.copy()

print("\nShape of X_full:", X_full.shape)

print("Shape of y_full:", y_full.shape)

print("\nShape of X_test:", X_test.shape)

print("Shape of y_test:", y_test.shape)
```

```python
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(

    X_full,

    y_full,

    test_size=0.20,

    random_state=42

)

print("\nTraining Shape:", X_train_raw.shape)

print("Validation Shape:", X_val_raw.shape)
```

```python
# Apply z-score normalization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)

X_val = scaler.transform(X_val_raw)

X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# Print the first five rows of the normalized data
print("\nFirst five rows of normalized X_train:")
print(X_train[:5])

print("\nFirst five rows of normalized X_test:")
print(X_test[:5])
```

```python


target_scaler = StandardScaler()

y_train = target_scaler.fit_transform(

    y_train_raw.values.reshape(-1, 1)

).astype(np.float32)

y_val = target_scaler.transform(

    y_val_raw.values.reshape(-1, 1)

).astype(np.float32)

y_test_array = y_test.values.reshape(-1, 1).astype(np.float32)

print("\nTarget scaling completed.")
```

# **Functions:**


```python


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
def expand_scalar_targets(y_array):
    y_array = np.asarray(y_array, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([y_array, y_array], axis=1)


def create_dataloader(X_array, y_array, batch_size=64, shuffle=True):
    features = torch.tensor(X_array, dtype=torch.float32)
    expanded_targets = torch.tensor(expand_scalar_targets(y_array), dtype=torch.float32)
    Dataset = TensorDataset(features, expanded_targets)
    return DataLoader(Dataset, batch_size=batch_size, shuffle=shuffle)


class HCMUCIRegressor(nn.Module):
    """Paper-style tabular regression backbone: 3 hidden layers of width 20 with LeakyReLU."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 2)   # direction d
        self.fc5 = nn.Linear(20, 1)   # magnitude R
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        d = self.fc4(x)
        R = self.fc5(x)
        return R, d


def train_hcm_model(X_train, y_train, X_val, y_val, training_config):
    set_seed(training_config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HCMUCIRegressor(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    train_loader = create_dataloader(X_train, y_train, batch_size=training_config['batch_size'], shuffle=True)
    history = []
    best_state = None
    best_val_loss = np.inf
    patience_counter = 0

    y_val_expanded = expand_scalar_targets(y_val)

    for epoch in range(training_config['epochs']):
        model.train()
        epoch_loss = []
        epoch_r_loss = []
        epoch_d_loss = []

        for x_batch, y_expanded in train_loader:
            x_batch = x_batch.to(device)
            y_expanded = y_expanded.to(device)

            optimizer.zero_grad()
            pred_R, pred_d = model(x_batch)

            R_target = torch.sqrt(torch.sum(y_expanded ** 2, dim=1, keepdim=True))
            d_target = y_expanded / (R_target + 1e-8)

            d_loss = criterion(pred_R * d_target, y_expanded)
            R_loss = criterion(R_target * pred_d, y_expanded)
            loss = d_loss + R_loss

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_r_loss.append(R_loss.item())
            epoch_d_loss.append(d_loss.item())

        val_outputs = predict_hcm_outputs(model, X_val, device)
        val_loss = mean_squared_error(y_val.reshape(-1), val_outputs['mean_prediction'])
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(np.mean(epoch_loss)),
            'R_loss': float(np.mean(epoch_r_loss)),
            'd_loss': float(np.mean(epoch_d_loss)),
            'val_mse': float(val_loss)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % training_config['print_every'] == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{training_config['epochs']} | Loss: {np.mean(epoch_loss):.5f} | R Loss: {np.mean(epoch_r_loss):.5f} | d Loss: {np.mean(epoch_d_loss):.5f} | Val MSE: {val_loss:.5f}")

        if patience_counter >= training_config['patience']:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), device


@torch.no_grad()
def predict_hcm_outputs(model, X_array, device):
    model.eval()
    model.to(device)

    x_tensor = torch.tensor(X_array, dtype=torch.float32, device=device)
    pred_R, pred_d = model(x_tensor)

    d_norm_sq = torch.sum(pred_d ** 2, dim=1)
    d_norm = torch.sqrt(torch.clamp(d_norm_sq, min=1e-12))

    # Public GitHub regression example uses this as the uncertainty band width.
    sigma_hat = torch.sqrt(torch.abs(d_norm_sq - 1.0)) * torch.abs(pred_R.squeeze(-1))
    raw_u = torch.abs(pred_R.squeeze(-1)) * torch.abs(d_norm - 1.0)
    pred_y = (pred_R * pred_d)[:, 0]

    return {
        'mean_prediction': pred_y.cpu().numpy(),
        'pred_R': pred_R.squeeze(-1).cpu().numpy(),
        'pred_d': pred_d.cpu().numpy(),
        'd_norm_sq': d_norm_sq.cpu().numpy(),
        'd_norm': d_norm.cpu().numpy(),
        'sigma_hat_raw': sigma_hat.cpu().numpy(),
        'raw_uncertainty_score': raw_u.cpu().numpy(),
    }

```

```python
def search_temperature_scale(raw_sigma, absolute_errors, grid=None):
    raw_sigma = np.clip(np.asarray(raw_sigma, dtype=float), 1e-8, None)
    absolute_errors = np.asarray(absolute_errors, dtype=float)

    if grid is None:
        grid = np.logspace(-2, 2, 400)

    target_coverage = np.array([0.68, 0.95, 0.997])
    best_scale = 1.0
    best_loss = np.inf

    for scale in grid:
        sigma_scaled = raw_sigma * scale
        coverage = np.array([
            np.mean(absolute_errors <= sigma_scaled),
            np.mean(absolute_errors <= 2.0 * sigma_scaled),
            np.mean(absolute_errors <= 3.0 * sigma_scaled)
        ])
        loss = np.sum((coverage - target_coverage) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_scale = scale

    return float(best_scale)


def normalize_confidence(confidence_values):
    confidence_values = np.asarray(confidence_values, dtype=float)
    cmin, cmax = confidence_values.min(), confidence_values.max()
    if np.isclose(cmin, cmax):
        return np.ones_like(confidence_values)
    return (confidence_values - cmin) / (cmax - cmin)


def create_predictions_dataframe(output_dict, y_true, sigma_scale, target_scaler):
    y_true = np.asarray(y_true).reshape(-1)
    mean_prediction_scaled = np.asarray(output_dict['mean_prediction'], dtype=float).reshape(-1, 1)
    mean_prediction = target_scaler.inverse_transform(mean_prediction_scaled).reshape(-1)

    sigma_raw_scaled = np.clip(np.asarray(output_dict['sigma_hat_raw'], dtype=float), 1e-8, None)
    sigma_calibrated_scaled = sigma_raw_scaled * sigma_scale
    sigma_raw = sigma_raw_scaled * float(target_scaler.scale_[0])
    sigma_calibrated = sigma_calibrated_scaled * float(target_scaler.scale_[0])
    raw_uncertainty = np.asarray(output_dict['raw_uncertainty_score'], dtype=float) * float(target_scaler.scale_[0])
    absolute_error = np.abs(y_true - mean_prediction)

    confidence = np.exp(-sigma_calibrated_scaled)
    confidence_normalized = normalize_confidence(confidence)

    predictions_df = pd.DataFrame({
        'Mean': mean_prediction,
        'Mean_Scaled': mean_prediction_scaled.reshape(-1),
        'Sigma_Hat_Raw': sigma_raw,
        'Sigma_Hat_Raw_Scaled': sigma_raw_scaled,
        'Sigma_Hat': sigma_calibrated,
        'Sigma_Hat_Scaled': sigma_calibrated_scaled,
        'StdDev': sigma_calibrated,
        'Confidence': confidence,
        'Confidence_Normalized': confidence_normalized,
        'RawUncertainty': raw_uncertainty,
        'Magnitude_R': output_dict['pred_R'],
        'Direction_Norm': output_dict['d_norm'],
        'Direction_Norm_Squared': output_dict['d_norm_sq'],
        'Absolute_Error': absolute_error,
        'Actual': y_true
    })

    for k in [1, 2, 3]:
        predictions_df[f'Lower_{k}Sigma'] = predictions_df['Mean'] - k * predictions_df['Sigma_Hat']
        predictions_df[f'Upper_{k}Sigma'] = predictions_df['Mean'] + k * predictions_df['Sigma_Hat']
        predictions_df[f'Coverage_{k}Sigma'] = (
            (predictions_df['Actual'] >= predictions_df[f'Lower_{k}Sigma']) &
            (predictions_df['Actual'] <= predictions_df[f'Upper_{k}Sigma'])
        ).astype(int)

    z_95 = 1.96
    predictions_df['Lower_95'] = predictions_df['Mean'] - z_95 * predictions_df['Sigma_Hat']
    predictions_df['Upper_95'] = predictions_df['Mean'] + z_95 * predictions_df['Sigma_Hat']
    predictions_df['Coverage_95'] = (
        (predictions_df['Actual'] >= predictions_df['Lower_95']) &
        (predictions_df['Actual'] <= predictions_df['Upper_95'])
    ).astype(int)

    return predictions_df


def compute_regression_ece(sigma_values, absolute_errors, n_bins=10):
    sigma_values = np.asarray(sigma_values, dtype=float)
    absolute_errors = np.asarray(absolute_errors, dtype=float)
    quantile_edges = np.quantile(sigma_values, np.linspace(0, 1, n_bins + 1))
    quantile_edges[0] -= 1e-8
    quantile_edges[-1] += 1e-8

    total = len(sigma_values)
    ece = 0.0
    rows = []
    for idx in range(n_bins):
        lower = quantile_edges[idx]
        upper = quantile_edges[idx + 1]
        mask = (sigma_values >= lower) & (sigma_values < upper)
        if not np.any(mask):
            continue
        mean_sigma = sigma_values[mask].mean()
        mean_error = absolute_errors[mask].mean()
        weight = mask.sum() / total
        ece += weight * abs(mean_sigma - mean_error)
        rows.append({
            'Bin': idx + 1,
            'Bin_Size': int(mask.sum()),
            'Mean_Sigma': mean_sigma,
            'Mean_Absolute_Error': mean_error,
            'Absolute_Gap': abs(mean_sigma - mean_error)
        })
    return float(ece), pd.DataFrame(rows)


def compute_paper_metrics(predictions_df, model_name='HCM'):
    absolute_error = predictions_df['Absolute_Error'].values
    sigma = np.clip(predictions_df['Sigma_Hat'].values, 1e-8, None)

    coverage_1 = np.mean(absolute_error <= sigma)
    coverage_2 = np.mean(absolute_error <= 2.0 * sigma)
    coverage_3 = np.mean(absolute_error <= 3.0 * sigma)
    coverage_95 = predictions_df['Coverage_95'].mean()
    average_95_width = (predictions_df['Upper_95'] - predictions_df['Lower_95']).mean()
    ece_reg, ece_bins_df = compute_regression_ece(sigma, absolute_error, n_bins=10)

    pearson_value = pearsonr(sigma, absolute_error)[0]
    spearman_value = spearmanr(sigma, absolute_error)[0]
    rmse = np.sqrt(mean_squared_error(predictions_df['Actual'], predictions_df['Mean']))
    mae = mean_absolute_error(predictions_df['Actual'], predictions_df['Mean'])

    summary_df = pd.DataFrame([{
        'Model': model_name,
        'Cov@1Sigma': coverage_1,
        'Cov@2Sigma': coverage_2,
        'Cov@3Sigma': coverage_3,
        'Coverage_95Interval': coverage_95,
        'Average_95Interval_Width': average_95_width,
        'Pearson': pearson_value,
        'Spearman': spearman_value,
        'ECE_reg': ece_reg,
        'RMSE': rmse,
        'MAE': mae
    }])
    return summary_df, ece_bins_df

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def insert_figure_into_worksheet(writer, sheet_name, figure, image_cell='H2'):

    image_buffer = io.BytesIO()

    figure.savefig(

        image_buffer,

        format='png',

        dpi=200,

        bbox_inches='tight'

    )

    image_buffer.seek(0)

    worksheet = writer.sheets[sheet_name]

    worksheet.insert_image(

        image_cell,

        'plot.png',

        {'image_data': image_buffer}

    )

    plt.close(figure)


def generate_and_save_plots(predictions_df, excel_path):
    with pd.ExcelWriter(_ensure_parent_dir(excel_path), engine='xlsxwriter') as writer:
        writer
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)

        ordered_df = predictions_df.sort_values('Actual').reset_index(drop=True)

        fig0, ax0 = plt.subplots(figsize=(12, 8))
        ax0.plot(ordered_df.index, ordered_df['Mean'], color='black', label='Prediction', linewidth=1.2)
        ax0.scatter(ordered_df.index, ordered_df['Actual'], color='darkorange', s=18, alpha=0.8, label='Actual')
        ax0.fill_between(
            ordered_df.index,
            ordered_df['Lower_95'],
            ordered_df['Upper_95'],
            color='skyblue',
            alpha=0.25,
            label='95% interval'
        )
        ax0.set_title('HCM Prediction with 95% Interval')
        ax0.set_xlabel('Ordered Test Sample')
        ax0.set_ylabel('Target Value')
        ax0.legend(loc='best')
        writer
        ordered_df.to_excel(writer, sheet_name='Bands_95Coverage', index=False)
        insert_figure_into_worksheet(writer, 'Bands_95Coverage', fig0)

        fig1, ax1 = plt.subplots(figsize=(12, 8))
        ax1.plot(ordered_df.index, ordered_df['Mean'], color='black', label='Prediction', linewidth=1.2)
        ax1.scatter(ordered_df.index, ordered_df['Actual'], color='darkorange', s=18, alpha=0.8, label='Actual')
        for k, alpha in zip([3, 2, 1], [0.10, 0.16, 0.24]):
            ax1.fill_between(
                ordered_df.index,
                ordered_df[f'Lower_{k}Sigma'],
                ordered_df[f'Upper_{k}Sigma'],
                color='skyblue',
                alpha=alpha,
                label=f'{k}σ interval'
            )
        ax1.set_title('HCM Prediction with 1σ, 2σ, 3σ Bands')
        ax1.set_xlabel('Ordered Test Sample')
        ax1.set_ylabel('Target Value')
        ax1.legend(loc='best')
        insert_figure_into_worksheet(writer, 'Bands_95Coverage', fig1, image_cell='J2')

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='Sigma_Hat', y='Absolute_Error', data=predictions_df, ax=ax2)
        ax2.set_title('HCM Uncertainty vs Absolute Error')
        ax2.set_xlabel('Calibrated Sigma_Hat')
        ax2.set_ylabel('Absolute Error')
        pd.DataFrame({'Sigma_Hat': predictions_df['Sigma_Hat'], 'Absolute_Error': predictions_df['Absolute_Error']}).to_excel(writer, sheet_name='Uncertainty_vs_Error', index=False)
        insert_figure_into_worksheet(writer, 'Uncertainty_vs_Error', fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 7))
        sns.histplot(predictions_df['Confidence_Normalized'], bins=20, kde=True, ax=ax3)
        ax3.set_title('Distribution of Normalized Confidence')
        ax3.set_xlabel('Normalized Confidence')
        ax3.set_ylabel('Frequency')
        pd.DataFrame({'Confidence_Normalized': predictions_df['Confidence_Normalized']}).to_excel(writer, sheet_name='Confidence_Distribution', index=False)
        insert_figure_into_worksheet(writer, 'Confidence_Distribution', fig3)

        fig4, ax4 = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='Actual', y='Mean', data=predictions_df, ax=ax4)
        diagonal_min = min(predictions_df['Actual'].min(), predictions_df['Mean'].min())
        diagonal_max = max(predictions_df['Actual'].max(), predictions_df['Mean'].max())
        ax4.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle='--', color='red')
        ax4.set_title('Predicted str vs Actual str')
        ax4.set_xlabel('Actual str')
        ax4.set_ylabel('Predicted Mean str')
        pd.DataFrame({'Actual': predictions_df['Actual'], 'Mean': predictions_df['Mean']}).to_excel(writer, sheet_name='Predicted_vs_Actual', index=False)
        insert_figure_into_worksheet(writer, 'Predicted_vs_Actual', fig4)

```

```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
def generate_and_save_paper_metrics(predictions_df, excel_path, model_name='HCM'):
    paper_metrics_df, ece_bins_df = compute_paper_metrics(predictions_df, model_name=model_name)

    coverage_df = pd.DataFrame([
        {'Scale': '1Sigma', 'Coverage': predictions_df['Coverage_1Sigma'].mean(), 'Target': 0.68},
        {'Scale': '95PercentInterval', 'Coverage': predictions_df['Coverage_95'].mean(), 'Target': 0.95},
        {'Scale': '2Sigma', 'Coverage': predictions_df['Coverage_2Sigma'].mean(), 'Target': 0.95},
        {'Scale': '3Sigma', 'Coverage': predictions_df['Coverage_3Sigma'].mean(), 'Target': 0.997},
    ])

    confidence_bins = pd.qcut(predictions_df['Confidence_Normalized'], q=min(10, len(predictions_df)), duplicates='drop')
    confidence_bin_df = predictions_df.assign(Confidence_Bin=confidence_bins).groupby('Confidence_Bin', observed=False).agg(
        Mean_Confidence=('Confidence_Normalized', 'mean'),
        Mean_Absolute_Error=('Absolute_Error', 'mean'),
        Mean_Sigma=('Sigma_Hat', 'mean'),
        Count=('Confidence_Normalized', 'size')
    ).reset_index()

    percentile_df = pd.DataFrame({
        'Percentile': [50, 75, 90, 95, 99],
        'Sigma_Hat_Threshold': np.percentile(predictions_df['Sigma_Hat'], [50, 75, 90, 95, 99]),
        'RawUncertainty_Threshold': np.percentile(predictions_df['RawUncertainty'], [50, 75, 90, 95, 99])
    })

    if not os.path.exists(excel_path):
        pd.DataFrame().to_excel(excel_path)
    with pd.ExcelWriter(_ensure_parent_dir(_ensure_excel_file(excel_path)), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        writer
        paper_metrics_df.to_excel(writer, sheet_name='Paper_Metrics', index=False)
        writer
        coverage_df.to_excel(writer, sheet_name='Coverage_Details', index=False)
        writer
        ece_bins_df.to_excel(writer, sheet_name='ECE_Bins', index=False)
        writer
        confidence_bin_df.to_excel(writer, sheet_name='Confidence_Bins', index=False)
        writer
        percentile_df.to_excel(writer, sheet_name='Uncertainty_Thresholds', index=False)


def generate_and_save_secondary_metrics(predictions_df, excel_path, model_name='HCM'):
    mu_pred = predictions_df['Mean'].values
    sigma_pred = np.clip(predictions_df['Sigma_Hat'].values, 1e-8, None)
    y_true = predictions_df['Actual'].values

    crps_values = ps.crps_gaussian(y_true, mu=mu_pred, sig=sigma_pred)
    nll_values = -0.5 * np.log(2 * np.pi * sigma_pred**2) - ((y_true - mu_pred) ** 2) / (2 * sigma_pred**2)
    secondary_df = pd.DataFrame({
        'Mean': mu_pred,
        'Sigma_Hat': sigma_pred,
        'CRPS': crps_values,
        'LogLikelihood': nll_values
    })

    summary_df = pd.DataFrame([{
        'Model': model_name,
        'Mean_CRPS': float(np.mean(crps_values)),
        'Mean_Negative_LogLikelihood': float(np.mean(-nll_values)),
        'Coverage_95Interval': float(predictions_df['Coverage_95'].mean())
    }])

    if not os.path.exists(excel_path):
        pd.DataFrame().to_excel(excel_path)
    with pd.ExcelWriter(_ensure_parent_dir(_ensure_excel_file(excel_path)), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        writer
        secondary_df.to_excel(writer, sheet_name='Secondary_Compatibility', index=False)
        writer
        summary_df.to_excel(writer, sheet_name='Secondary_Summary', index=False)

```

```python
# Define the path to the folder
folder_path = "/content/drive/MyDrive/Regression/streamflow_uncertainty_analysis"
if 'google.colab' in sys.modules:
    os.makedirs(folder_path, exist_ok=True)

if 'google.colab' in sys.modules:
    output_candidates = [
        Path(output_folder),
        Path('/content/drive/MyDrive/Regression/streamflow_uncertainty_analysis/hyperspherical_confidence_mapping_hcm')
    ]
else:
    output_candidates = [
        Path('streamflow_uncertainty_analysis/hyperspherical_confidence_mapping_hcm'),
        Path(output_folder)
    ]

for candidate in output_candidates:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        resolved_output_folder = candidate
        break
    except Exception:
        continue
else:
    raise RuntimeError('Could not create an output folder for HCM results.')

```

```python
def build_matrix_evaluation(predictions_df, model_name='HCM'):
    paper_metrics_df, _ = compute_paper_metrics(predictions_df, model_name=model_name)
    matrix_df = paper_metrics_df.copy()
    matrix_df['Mean_Sigma_Hat'] = predictions_df['Sigma_Hat'].mean()
    matrix_df['Mean_RawUncertainty'] = predictions_df['RawUncertainty'].mean()
    matrix_df['Coverage_95Interval'] = predictions_df['Coverage_95'].mean()
    matrix_df['Average_95Interval_Width'] = (predictions_df['Upper_95'] - predictions_df['Lower_95']).mean()
    return matrix_df

```

# **Hyperspherical Confidence Mapping (HCM)**


```python
# Paper-style tabular regression setup from the HCM paper appendix.
training_config = {
    'seed': random_seed,
    'epochs': 200,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'patience': 40,
    'print_every': 20
}
training_config

```

```python
# Train model
model, training_history_df, device = train_hcm_model(
    X_train,
    y_train,
    X_val,
    y_val,
    training_config
)

# Validation outputs
validation_outputs = predict_hcm_outputs(
    model,
    X_val,
    device
)

# Temperature scaling
validation_scale = search_temperature_scale(
    validation_outputs['sigma_hat_raw'],
    np.abs(
        y_val.reshape(-1)
        - validation_outputs['mean_prediction']
    )
)

print(f"Validation temperature scale: {validation_scale:.6f}")

# Test prediction
test_outputs = predict_hcm_outputs(
    model,
    X_test_scaled,
    device
)

# Build prediction dataframe
predictions_HCM_df = create_predictions_dataframe(
    test_outputs,
    y_test_array,
    validation_scale,
    target_scaler
)

print(predictions_HCM_df.head())
```

```python

hcm_excel_path = str(resolved_output_folder / 'HCM.xlsx')

generate_and_save_plots(

    predictions_HCM_df,

    hcm_excel_path

)

print(f"Saved detailed HCM outputs to: {hcm_excel_path}")

generate_and_save_paper_metrics(

    predictions_HCM_df,

    hcm_excel_path,

    model_name='HCM'

)

generate_and_save_secondary_metrics(

    predictions_HCM_df,

    hcm_excel_path,

    model_name='HCM'

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
generate_and_save_paper_metrics(predictions_HCM_df, hcm_excel_path, model_name='HCM')
if not os.path.exists(hcm_excel_path):
    pd.DataFrame().to_excel(hcm_excel_path)
with pd.ExcelWriter(_ensure_parent_dir(_ensure_excel_file(hcm_excel_path)), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    writer
    training_history_df.to_excel(writer, sheet_name='Training_History', index=False)

```

```python
generate_and_save_secondary_metrics(predictions_HCM_df, hcm_excel_path, model_name='HCM')

```

# **Matrix Evaulation**


```python
import os

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

import os
matrix_evaluation_df = build_matrix_evaluation(predictions_HCM_df, model_name='HCM')
matrix_excel_path = resolved_output_folder / 'Matrix Evaluation.xlsx'
with pd.ExcelWriter(_ensure_parent_dir(matrix_excel_path), engine='xlsxwriter') as writer:
    writer
    matrix_evaluation_df.to_excel(writer, sheet_name='Matrix Evaluation', index=False)

print(matrix_evaluation_df)
print(f"Saved matrix evaluation to: {matrix_excel_path}")

```

