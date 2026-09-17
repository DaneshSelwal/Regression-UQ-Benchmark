# **Start**

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
import gpboost
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
from pytorch_tabnet import TabNetRegressor
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

# **Functions**

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


```

```python
def plot_best_scores(best_scores_ran, excel_file_path):
    # Extract the best pruner for each model based on RMSE and correlation coefficient
    best_rmse_scores = {}
    best_corr_coef_scores = {}

    for (model_name, pruner_name), scores in best_scores_ran.items():
        # Initialize if not already present
        if model_name not in best_rmse_scores:
            best_rmse_scores[model_name] = (scores['test_rmse'], pruner_name)
        if model_name not in best_corr_coef_scores:
            best_corr_coef_scores[model_name] = (scores['test_corr_coef'], pruner_name)

        # Update if better scores are found
        if scores['test_rmse'] < best_rmse_scores[model_name][0]:
            best_rmse_scores[model_name] = (scores['test_rmse'], pruner_name)
        if scores['test_corr_coef'] > best_corr_coef_scores[model_name][0]:
            best_corr_coef_scores[model_name] = (scores['test_corr_coef'], pruner_name)

    # Prepare data for plotting
    model_names_rmse = [f"{model} ({pruner})" for model, (rmse, pruner) in best_rmse_scores.items()]
    rmse_values = [rmse for rmse, _ in best_rmse_scores.values()]

    model_names_corr = [f"{model} ({pruner})" for model, (corr, pruner) in best_corr_coef_scores.items()]
    corr_values = [corr for corr, _ in best_corr_coef_scores.values()]

    # Plot RMSE
    plt.figure(figsize=(12, 6))
    bars_rmse = plt.bar(model_names_rmse, rmse_values, color='skyblue')

    # Highlight the best model
    best_rmse_index = np.argmin(rmse_values)
    bars_rmse[best_rmse_index].set_color('orange')

    # Annotate the bars with the RMSE scores
    for i, bar in enumerate(bars_rmse):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
                 f'{rmse_values[i]:.5f}', ha='center', va='bottom', color='black')

    # Add labels and title for the RMSE bar chart
    plt.xlabel('Model (Pruner)')
    plt.ylabel('Test RMSE')
    plt.title('Best Test RMSE for Each Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    rmse_image_path = 'rmse_plot.png'
    plt.savefig(rmse_image_path)
    plt.close()

    # Plot Correlation Coefficient
    plt.figure(figsize=(12, 6))
    bars_corr = plt.bar(model_names_corr, corr_values, color='lightgreen')

    # Highlight the best model
    best_corr_index = np.argmax(corr_values)
    bars_corr[best_corr_index].set_color('orange')

    # Annotate the bars with the correlation coefficient scores
    for i, bar in enumerate(bars_corr):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
                 f'{corr_values[i]:.5f}', ha='center', va='bottom', color='black')

    # Add labels and title for the correlation coefficient bar chart
    plt.xlabel('Model (Pruner)')
    plt.ylabel('Correlation Coefficient')
    plt.title('Best Correlation Coefficient for Each Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    corr_image_path = 'corr_plot.png'
    plt.savefig(corr_image_path)
    plt.close()

    # Load the existing Excel file
    workbook = load_workbook(excel_file_path)

    # Create a new sheet for the plots
    sheet_name = 'Best Models Plots'
    if sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
    else:
        sheet = workbook.create_sheet(title=sheet_name)

    # Insert images into the new Excel sheet
    img_rmse = Image(rmse_image_path)
    img_corr = Image(corr_image_path)

    # Insert images
    sheet.add_image(img_rmse, 'A1')
    sheet.add_image(img_corr, 'A20')  # Adjust the position as needed

    # Save the workbook
    workbook.save(excel_file_path)

    # Clean up the image files
    os.remove(rmse_image_path)
    os.remove(corr_image_path)

# Example usage
# plot_best_scores(best_scores_ran, 'path_to_your_excel_file.xlsx')
```

```python
def generate_interpretml_explanations_summary_pruners(
    results_dict, X_train, y_train, X_test, feature_names, instance_indices=None, excel_file_path=None
):
    if instance_indices is None:
        instance_indices = range(len(X_test))
    elif isinstance(instance_indices, int):
        instance_indices = [instance_indices]

    valid_indices = [idx for idx in instance_indices if 0 <= idx < len(X_test)]
    if not valid_indices:
        print("No valid instance indices provided.")
        return

    if isinstance(X_test, pd.DataFrame):
        instances_to_explain = X_test.iloc[valid_indices]
    else:
        instances_to_explain = X_test[valid_indices]

    best_model_pruners = {}
    for model_key, model_info in results_dict.items():
        if isinstance(model_key, tuple):
            model_name, pruner_name = model_key
        else:
            model_name = model_key
            pruner_name = None

        best_score = model_info.get('best_score')
        if best_score is None:
            print(f"No 'best_score' found for {model_key}. Skipping this combination.")
            continue

        if model_name not in best_model_pruners:
            best_model_pruners[model_name] = {
                'pruner_name': pruner_name,
                'model_info': model_info,
                'best_score': best_score
            }
        else:
            current_best_score = best_model_pruners[model_name]['best_score']
            if best_score < current_best_score:
                best_model_pruners[model_name] = {
                    'pruner_name': pruner_name,
                    'model_info': model_info,
                    'best_score': best_score
                }

    for model_name, info in best_model_pruners.items():
        pruner_name = info['pruner_name']
        model_info = info['model_info']
        best_params = dict(model_info['best_params'])  # don't mutate original!
        model_class = model_classes.get(model_name)

        if model_class is None:
            print(f"Model {model_name} is not supported or not available.")
            continue

        if model_name == 'CatBoost':
            best_params['verbose'] = 0

        # ------- Main model fit logic ---------
        if model_name == "TabNet":
            # TabNet: reshape y, fit, flatten pred for LIME/SHAP, etc.
            y_train_tabnet = np.array(y_train).reshape(-1, 1)
            try:
                model = model_class(**{k: v for k, v in best_params.items() if k != "verbose"})
            except TypeError:
                model = model_class()
            model.fit(np.array(X_train), y_train_tabnet, max_epochs=100, patience=10, batch_size=1024, eval_set=[(np.array(X_train), y_train_tabnet)])
            def predict_fn(data):
                preds = model.predict(np.array(data))
                # flatten for interpreters
                return preds.flatten()
        else:
            try:
                model = model_class(**best_params)
            except TypeError:
                model = model_class()
            model.fit(X_train, y_train)
            def predict_fn(data):
                return model.predict(data)

        if isinstance(X_train, pd.DataFrame):
            data_for_explainer = X_train.values
        else:
            data_for_explainer = X_train

        if isinstance(instances_to_explain, pd.DataFrame):
            data_for_explanation = instances_to_explain.values
        else:
            data_for_explanation = instances_to_explain

        # Generate LIME explanations
        lime_explainer = LimeTabular(
            predict_fn,
            data=data_for_explainer,
            feature_names=feature_names,
            random_state=1,
            mode='regression'
        )
        lime_explanation = lime_explainer.explain_local(data_for_explanation)

        feature_importances_lime = {}
        num_instances = len(valid_indices)
        for idx in range(num_instances):
            explanation = lime_explanation.data(idx)
            for feature_name, feature_score in zip(explanation['names'], explanation['scores']):
                feature_importances_lime[feature_name] = feature_importances_lime.get(feature_name, 0) + abs(feature_score)
        feature_importances_lime = {k: v / num_instances for k, v in feature_importances_lime.items()}
        feature_importances_lime = {k: round(v, 3) for k, v in feature_importances_lime.items()}

        # Generate SHAP explanations using ShapKernel
        try:
            shap_explainer = ShapKernel(predict_fn, data_for_explainer, feature_names=feature_names)
            shap_explanation = shap_explainer.explain_local(data_for_explanation)

            feature_importances_shap = {}
            for idx in range(num_instances):
                explanation = shap_explanation.data(idx)
                for feature_name, feature_score in zip(explanation['names'], explanation['scores']):
                    feature_importances_shap[feature_name] = feature_importances_shap.get(feature_name, 0) + abs(feature_score)

            feature_importances_shap = {k: v / num_instances for k, v in feature_importances_shap.items()}
            feature_importances_shap = {k: round(v, 3) for k, v in feature_importances_shap.items()}
        except Exception as e:
            print(f"Could not compute SHAP values for model {model_name}: {e}")
            feature_importances_shap = {}

        # Plot LIME and SHAP feature importances side by side
        fig, axes = plt.subplots(1, 2, figsize=(34, 36))

        # Plot LIME feature importances
        lime_importances_df = pd.DataFrame.from_dict(
            feature_importances_lime, orient='index', columns=['importance']
        )
        lime_importances_df.sort_values(by='importance', ascending=False, inplace=True)
        lime_importances_df.plot(kind='bar', legend=False, color='skyblue', ax=axes[0])
        axes[0].set_title(f"LIME Feature Importances for {model_name}")
        axes[0].set_ylabel("Average Absolute Importance Score")
        axes[0].set_xlabel("Features")
        axes[0].tick_params(axis='x', rotation=45)

        for p in axes[0].patches:
            height = p.get_height()
            axes[0].annotate(f'{height:.3f}',
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=8)

        # Plot SHAP feature importances
        shap_importances_df = pd.DataFrame.from_dict(
            feature_importances_shap, orient='index', columns=['importance']
        )
        shap_importances_df.sort_values(by='importance', ascending=False, inplace=True)
        shap_importances_df.plot(kind='bar', legend=False, color='orange', ax=axes[1])
        axes[1].set_title(f"SHAP Feature Importances for {model_name}")
        axes[1].set_ylabel("Average Absolute SHAP Value")
        axes[1].set_xlabel("Features")
        axes[1].tick_params(axis='x', rotation=45)

        for p in axes[1].patches:
            height = p.get_height()
            axes[1].annotate(f'{height:.3f}',
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Save plots as images
        image_path = f'feature_importances_{model_name}.png'
        fig.savefig(image_path)
        plt.close(fig)

        # Optionally insert images and scores into an Excel file
        if excel_file_path:
            workbook = load_workbook(excel_file_path)
            sheet_name = f'{model_name} Explanations'
            if sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
            else:
                sheet = workbook.create_sheet(title=sheet_name)

            # Insert images into the new Excel sheet
            img = Image(image_path)
            sheet.add_image(img, 'A1')

            # Create a new sheet for feature importance scores
            scores_sheet_name = f'{model_name} Scores'
            if scores_sheet_name in workbook.sheetnames:
                scores_sheet = workbook[scores_sheet_name]
            else:
                scores_sheet = workbook.create_sheet(title=scores_sheet_name)

            # Write LIME scores
            scores_sheet.append(['Feature', 'LIME Importance'])
            for feature, importance in feature_importances_lime.items():
                scores_sheet.append([feature, importance])

            # Write SHAP scores if available
            if feature_importances_shap:
                scores_sheet.append(['Feature', 'SHAP Importance'])
                for feature, importance in feature_importances_shap.items():
                    scores_sheet.append([feature, importance])

            # Save the workbook
            workbook.save(excel_file_path)

            # Clean up the image file
            os.remove(image_path)
```

# **Hyperparameter tuning using Autosampler by Optuna**

```python
import joblib

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
    loss = torch.sqrt(torch.mean((yhat - y) ** 2))
    return loss


def hyperparameter_tuning_all(X_train, y_train, X_test, y_test, excel_path):

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # ================= NEW =================
    model_save_dir = "./drive/MyDrive/Regression/hyperparameter_tuning/models"
    os.makedirs(model_save_dir, exist_ok=True)
    best_rmse_tracker = {}
    # =======================================

    models = {
        'Random Forest': (RandomForestRegressor, {
            'n_estimators': [100, 200, 300, 500, 700],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 0.01],
            'min_samples_leaf': [1, 3, 5, 0.01],
            'min_weight_fraction_leaf': [0.0, 0.01, 0.1, 0.2],
            'max_features': [1.0, 'sqrt', 'log2', 0.3, 0.5],
            'max_leaf_nodes': [None, 50, 100, 200],
            'min_impurity_decrease': [0.0, 0.01, 0.1, 0.2],
            'n_jobs': [-1],
            'random_state': [42],
            'verbose': [0],
            'warm_start': [False],
            'ccp_alpha': [0.0, 0.001, 0.01, 0.05, 0.1]
        }),
        'Gradient Boosting': (GradientBoostingRegressor, {
            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 500, 700],
            'subsample': [1.0, 0.9, 0.7, 0.5],
            'criterion': ['friedman_mse', 'squared_error'],
            'min_samples_split': [2, 5, 10, 0.01],
            'min_samples_leaf': [1, 3, 5, 0.01],
            'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 10],
            'min_impurity_decrease': [0.0, 0.01, 0.1],
            'init': [None],
            'random_state': [42],
            'max_features': [None, 'sqrt', 'log2', 0.5],
            'alpha': [0.9, 0.5, 0.1],
            'verbose': [0],
            'max_leaf_nodes': [None, 10, 30, 50],
            'warm_start': [False],
            'validation_fraction': [0.1],
            'n_iter_no_change': [None, 10, 20],
            'tol': [1e-4, 1e-3],
            'ccp_alpha': [0.0, 0.001, 0.01]
        }),
        'XGBoost': (XGBRegressor, {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.5, 1],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'colsample_bylevel': [0.5, 0.7, 0.9],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0.1, 1, 5, 10],
            'objective': ['reg:squarederror'],
            'random_state': [42],
            'n_jobs': [-1]
        }),
        'LightGBM': (LGBMRegressor, {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'num_leaves': [15, 31, 63],
            'max_depth': [3, 5, 7, -1],
            'min_child_samples': [1, 5, 10, 20],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.1, 1, 10],
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1],
            'bagging_freq': [0, 1, 5],
            'objective': ['regression'],
            'random_state': [42],
            'n_jobs': [-1],
            'verbose': [-1]
        }),
        'GPBoost': (GPBoostRegressor, {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [15, 31, 63],
            'min_child_samples': [1, 5, 10, 20],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1],
            'random_state': [42],
            'n_jobs': [-1],
            'verbose': [-1]
        }),
        'CatBoost': (CatBoostRegressor, {
            'iterations': [200, 500, 1000],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128],
            'min_data_in_leaf': [1, 5, 10, 20],
            'rsm': [0.6, 0.8, 1.0],
            'bagging_temperature': [0, 1, 10],
            'random_seed': [42],
            'verbose': [0]
        }),
        'NGBoost': (NGBRegressor, {
            'n_estimators': [200, 500, 1000],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'natural_gradient': [True, False],
            'minibatch_frac': [0.5, 0.7, 0.9, 1.0],
            'col_sample': [0.5, 0.7, 0.9, 1.0],
            'Dist': [Normal],
            'Score': [LogScore],
            'random_state': [42],
            'verbose': [0]
        }),
        'TabNet': (TabNetRegressor, {
            'n_d': [8, 16, 32, 64],
            'n_a': [8, 16, 32, 64],
            'n_steps': [3, 5, 7, 10],
            'gamma': [1.0, 1.3, 1.5, 2.0],
            'lambda_sparse': [1e-4, 1e-3, 1e-2],
            'optimizer_params': [{'lr': 2e-2}], # Fixed learning rate as recommended
            'mask_type': ['sparsemax', 'entmax'],
            'n_shared': [1, 2, 3],
            'n_independent': [1, 2, 3],
            'scheduler_params': [{"step_size": 10, "gamma": 0.9}],
            'scheduler_fn': [torch.optim.lr_scheduler.StepLR],
            'seed': [42],
            'verbose': [0]
        }),
        'HistGradientBoosting': (HistGradientBoostingRegressor, {
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_iter': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 7, None],
            'min_samples_leaf': [5, 10, 20],
            'max_leaf_nodes': [15, 31, 63, None],
            'l2_regularization': [0.0, 0.1, 0.5, 1.0],
            'max_bins': [64, 128, 255],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.2],
            'n_iter_no_change': [5, 10, 15],
            'loss': ['squared_error'],
            'random_state': [42],
            'verbose': [0]
        }),
        'PGBM': (PGBM, {})
    }

    pruners = [
        optuna.pruners.MedianPruner(),
        optuna.pruners.NopPruner(),
        optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=3),
        optuna.pruners.PercentilePruner(25.0),
        optuna.pruners.SuccessiveHalvingPruner(),
        optuna.pruners.HyperbandPruner(),
        optuna.pruners.ThresholdPruner(lower=0.1),
        optuna.pruners.WilcoxonPruner()
    ]

    best_scores = {}
    predictions_df = pd.DataFrame({'Actual': y_test})
    timing_records = []

    for model_name, (model_class, param_space) in models.items():

        rmse_trial_history = {p.__class__.__name__: [] for p in pruners}

        for pruner in pruners:
            pruner_name = pruner.__class__.__name__
            print(f"Running Optuna for {model_name} with {pruner_name}...")
            start_time = time.time()

            best_rmse_tracker[(model_name, pruner_name)] = np.inf

            sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
            study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

            if model_name == 'PGBM':

                def pgbm_objective(trial):
                    params = {
                            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
                            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.15]),
                            'max_leaves': trial.suggest_int('max_leaves', 15, 63),
                            'min_split_gain': trial.suggest_categorical('min_split_gain', [0.0, 0.1, 0.5, 1.0]),
                            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.1, 1.0, 5.0, 10.0]),
                            'feature_fraction': trial.suggest_categorical('feature_fraction', [0.5, 0.7, 0.9, 1.0]),
                            'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.5, 0.7, 0.9, 1.0]),
                            'tree_correlation': trial.suggest_categorical('tree_correlation', [0.0, 0.1, 0.2, 0.3]),
                            'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [3, 5, 10, 20]),
                            'max_bin': trial.suggest_categorical('max_bin', [64, 128, 256]),
                            'distribution': trial.suggest_categorical('distribution', ['normal', 'studentt', 'laplace']),
                            'objective': 'mse',
                            'metric': 'rmse',
                            'random_state': 42,
                            'verbose': 0
                        }

                    model = PGBM()
                    model.train((X_train, y_train),
                                objective=mseloss_objective,
                                metric=rmseloss_metric,
                                params=params)

                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)

                    rmse_trial_history[pruner_name].append(rmse)

                    # ===== SAVE BEST MODEL =====
                    if rmse < best_rmse_tracker[(model_name, pruner_name)]:
                        best_rmse_tracker[(model_name, pruner_name)] = rmse
                        save_path = os.path.join(
                            model_save_dir,
                            f"{model_name}_{pruner_name}_BEST.pkl"
                        )
                        joblib.dump(model, save_path)

                    return mse

                study.optimize(pgbm_objective, n_trials=50)

            else:

                def objective(trial):
                    params = {}
                    for key, values in param_space.items():
                        params[key] = trial.suggest_categorical(key, values)

                    model = model_class(**params)

                    if model_name == 'TabNet':
                        model.fit(X_train, y_train.reshape(-1, 1))
                    else:
                        model.fit(X_train, y_train)


                    y_pred = model.predict(X_test)

                    if model_name == 'TabNet':
                        y_pred = y_pred.ravel()

                    mse = mean_squared_error(y_test, y_pred)

                    rmse = np.sqrt(mse)

                    rmse_trial_history[pruner_name].append(rmse)

                    # ===== SAVE BEST MODEL =====
                    if rmse < best_rmse_tracker[(model_name, pruner_name)]:
                        best_rmse_tracker[(model_name, pruner_name)] = rmse
                        save_path = os.path.join(
                            model_save_dir,
                            f"{model_name}_{pruner_name}_BEST.pkl"
                        )
                        joblib.dump(model, save_path)

                    return mse

                study.optimize(objective, n_trials=50)

            elapsed_time = time.time() - start_time

            # Load frozen model (NO RETRAIN)
            best_model = joblib.load(
                os.path.join(model_save_dir, f"{model_name}_{pruner_name}_BEST.pkl")
            )

            y_pred = best_model.predict(X_test)

            if model_name == 'TabNet':
                y_pred = y_pred.ravel()

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            corr_coef = np.corrcoef(y_test, y_pred)[0, 1]


            predictions_df[f'{model_name}_{pruner_name}_Predicted'] = y_pred

            best_scores[(model_name, pruner_name)] = {
                'best_score': mse,
                'best_params': study.best_params,
                'test_mse': mse,
                'test_rmse': rmse,
                'test_corr_coef': corr_coef,
                'pruner': pruner_name
            }

            timing_records.append({
                'Model': model_name,
                'Pruner': pruner_name,
                'Tuning_Time_Seconds': elapsed_time
            })

        # RMSE plots & Excel writing (UNCHANGED)
        rmse_df = pd.DataFrame(rmse_trial_history)
        rmse_df.insert(0, "Trial", np.arange(1, len(rmse_df) + 1))

        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            rmse_df.to_excel(writer, sheet_name=f"RMSE_Trials_{model_name}", index=False)
        # ================= SAVE RMSE PLOT =================
        plot_dir = os.path.dirname(excel_path)
        plot_path = os.path.join(plot_dir, f"RMSE_Trials_{model_name}.png")

        plt.figure(figsize=(10, 6))
        for pruner_name, values in rmse_trial_history.items():
            if len(values) > 0:   # <-- important safety check
                plt.plot(values, label=pruner_name, linewidth=2)

        plt.title(f"RMSE Variation Over Trials\n{model_name}")
        plt.xlabel("Trial Number")
        plt.ylabel("RMSE")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()

        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        # ================= INSERT PLOT INTO EXCEL =================
        wb = load_workbook(excel_path)
        ws = wb[f"RMSE_Trials_{model_name}"]

        img = Image(plot_path)
        img.anchor = "J2"
        ws.add_image(img)

        wb.save(excel_path)

    timing_df = pd.DataFrame(timing_records)

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        timing_df.to_excel(writer, sheet_name='Tuning_Time', index=False)

    return best_scores


best_scores_autosampler = hyperparameter_tuning_all(X_train, y_train, X_test, y_test, "./drive/MyDrive/Regression/hyperparameter_tuning/test.xlsx")

```

```python
best_scores_autosampler
```

# **Best Model Analysis**

```python
def get_best_models_and_predict(best_scores_autosampler, X_train, y_train, X_test, y_test, file_path):
    # Convert input data to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Mapping for model creation based on dictionary keys
    model_mapping = {
        'Random Forest': RandomForestRegressor,
        'Gradient Boosting': GradientBoostingRegressor,
        'XGBoost': XGBRegressor,
        'LightGBM': LGBMRegressor,
        'CatBoost': CatBoostRegressor,
        'GPBoost': GPBoostRegressor,
        'NGBoost': NGBRegressor,
        'TabNet': TabNetRegressor,
        'HistGradientBoosting': HistGradientBoostingRegressor,
        'PGBM': PGBM  # PGBM is handled separately
    }

    # Dictionary to store the best model for each type
    best_models = {}

    # Iterate over the dictionary to find the best pruner for each model type
    for (model_name, pruner), params in best_scores_autosampler.items():
        current_score = params.get('test_mse', np.inf)
        if model_name not in best_models or current_score < best_models[model_name]['score']:
            best_models[model_name] = {
                'score': current_score,
                'params': params['best_params'],
                'pruner': pruner
            }

    # Prepare a DataFrame to store predictions
    df = pd.read_csv(file_path)

    # Iterate over the best models to train and predict
    for model_name, model_info in best_models.items():
        best_params = model_info['params']
        model_class = model_mapping.get(model_name)

        if model_class is None:
            print(f"Model {model_name} is not supported or not available.")
            continue

        # Handle specific parameters or settings for model if needed
        if model_name == 'CatBoost':
            best_params.pop('verbose', None)  # Remove 'verbose' for CatBoost

        # Create an instance of the best model with the best parameters
        if model_name == 'PGBM':
            model = model_class()
            model.train((X_train, y_train), objective=mseloss_objective, metric=rmseloss_metric, params=best_params)
            predictions = model.predict(X_test)
        elif model_name == 'TabNet':
            model = model_class(**best_params)
            model.fit(X_train, y_train.reshape(-1, 1))
            predictions = model.predict(X_test)
            predictions = predictions.ravel()
        else:
            model = model_class(**best_params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        # Add predictions to the DataFrame
        df[f'{model_name} Predictions'] = predictions

        # Plot actual vs. predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red', lw=2)
        plt.xlabel("Actual Y_Label")
        plt.ylabel("Predicted Y_Label")
        plt.title(f"Actual vs. Predicted Values ({model_name})")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot temporarily
        plot_path = f'temp_plot_{model_name}.png'
        plt.savefig(plot_path)
        plt.close()

    # ✅ NEW OUTPUT DIRECTORY
    output_dir = "./drive/MyDrive/Regression/hyperparameter_tuning/"
    os.makedirs(output_dir, exist_ok=True)

    output_excel_path = os.path.join(
        output_dir,
        os.path.basename(file_path).replace('.csv', '_results.xlsx')
    )

    # Save predictions and plots to Excel
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        # Write data to Excel
        df.to_excel(writer, sheet_name='data', index=False)

        # Get the xlsxwriter objects
        workbook = writer.book

        # Insert each plot into a separate worksheet
        for model_name in best_models.keys():
            short_model_name = ''.join([word[0] for word in model_name.split()])
            sheet_name = f'{short_model_name}_Plot'

            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet
            plot_path = f'temp_plot_{model_name}.png'
            worksheet.insert_image('A1', plot_path)

    # Clean up temporary plot files
    for model_name in best_models.keys():
        os.remove(f'temp_plot_{model_name}.png')

    return df, best_models

# Call the function
df, best_models = get_best_models_and_predict(best_scores_autosampler, X_train, y_train, X_test, y_test, "./drive/MyDrive/Regression/data/test.csv")
```

```python
plot_best_scores(best_scores_autosampler,"./drive/MyDrive/Regression/hyperparameter_tuning/test_results.xlsx")
```

```python
generate_interpretml_explanations_summary_pruners(best_scores_autosampler, X_train, y_train, x_test, feature_names,excel_file_path = "./drive/MyDrive/Regression/hyperparameter_tuning/test_results.xlsx")
```

