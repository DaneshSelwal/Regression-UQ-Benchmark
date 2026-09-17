# TabFM Codebase Part 2

## File: examples/classification_example.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example showing how to run classification with TabFM v1.0.0."""

import numpy as np
import pandas as pd
import tabfm


def run_example(model=None) -> np.ndarray:
  """Generates dummy data and runs classification."""
  if model is None:
    # Option A: JAX Backend (default)
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")

    # Option B: PyTorch Backend
    # model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")

  # 2. Initialize scikit-learn compatible classifier
  clf = tabfm.TabFMClassifier(model=model)

  # 3. Generate dummy dataset (with mixed column types)
  # Context (training) data
  X_train = pd.DataFrame({
      "num_feat_1": [1.5, 2.5, 3.5, 4.5, 5.5],
      "cat_feat_1": ["A", "B", "A", "B", "C"],
      "num_feat_2": [10.0, 20.0, 10.0, 30.0, 20.0],
  })
  y_train = np.array(["yes", "no", "yes", "no", "yes"])

  # Test samples
  X_test = pd.DataFrame({
      "num_feat_1": [2.0, 4.0],
      "cat_feat_1": ["B", "A"],
      "num_feat_2": [15.0, 25.0],
  })

  # 4. Fit the classifier (prepares internal data transformers)
  clf.fit(X_train, y_train)

  # 5. Predict class probabilities
  probs = clf.predict_proba(X_test)
  return probs


if __name__ == "__main__":
  print("Running TabFM classification model... (Note: compilation and model execution may take a few minutes on first run)")
  predictions = run_example()
  print("Classification predictions:\n", predictions)

```

## File: examples/regression_example.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example showing how to run regression with TabFM v1.0.0."""

import numpy as np
import pandas as pd
import tabfm


def run_example(model=None) -> np.ndarray:
  """Generates dummy data and runs regression."""
  if model is None:
    # Option A: JAX Backend (default)
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="regression")

    # Option B: PyTorch Backend
    # model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression")

  # 2. Initialize scikit-learn compatible regressor
  reg = tabfm.TabFMRegressor(model=model)

  # 3. Generate dummy dataset
  X_train = pd.DataFrame({
      "num_feat_1": [1.5, 2.5, 3.5, 4.5, 5.5],
      "cat_feat_1": ["A", "B", "A", "B", "C"],
  })
  y_train = np.array([10.5, 20.0, 11.0, 29.5, 21.0])

  X_test = pd.DataFrame({
      "num_feat_1": [2.0, 4.0],
      "cat_feat_1": ["B", "A"],
  })

  # 4. Fit and predict
  reg.fit(X_train, y_train)
  preds = reg.predict(X_test)
  return preds


if __name__ == "__main__":
  print("Running TabFM regression model... (Note: compilation and model execution may take a few minutes on first run)")
  predictions = run_example()
  print("Regression predictions:\n", predictions)

```

## File: examples/tabarena_classification_example.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classification on a real TabArena task with TabFM v1.0.0.

Loads repeat-0 / fold-0 of the TabArena ``maternal_health_risk`` task (a small
3-class dataset with string labels), then runs inference twice:

  * default  -- ``TabFMClassifier(model=...)`` (uniform logit averaging)
  * ensemble -- ``TabFMClassifier.ensemble(model=...)`` (feature crosses / SVD
    features + NNLS-weighted blending + probability calibration)

and reports ROC AUC and log loss for each so the two presets can be compared.

Requires the optional ``openml`` dependency: ``pip install tabfm[examples]``.
"""

import numpy as np
import tabfm

# OpenML task id for the TabArena "maternal_health_risk" classification task
# (1014 rows, 6 numerical features, 3-class string target).
TASK_ID = 363685
SEED = 0


def _load_fold_0(task_id):
  """Returns (X_train, y_train, X_test, y_test) for repeat-0 / fold-0.

  Note: both X and y are passed to TabFM as raw OpenML values -- including the
  string class labels, which TabFM label-encodes internally (and exposes via
  ``clf.classes_``) -- so no manual coercion is needed here.
  """
  import openml  # pylint: disable=g-import-not-at-top

  task = openml.tasks.get_task(task_id)
  dataset = task.get_dataset()
  x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
  split = task.download_split().split[0][0][0]  # repeat 0, fold 0, sample 0

  x_train = x.iloc[split.train].copy()
  x_test = x.iloc[split.test].copy()
  y_train = y.iloc[split.train]  # raw labels -- TabFM encodes them internally
  y_test = y.iloc[split.test]
  return x_train, y_train, x_test, y_test


def _evaluate(clf, x_train, y_train, x_test, y_test):
  """Fits ``clf`` on raw labels and returns (roc_auc, log_loss) on test fold."""
  from sklearn.metrics import log_loss, roc_auc_score  # pylint: disable=g-import-not-at-top

  # Fit directly on the raw string/category target -- TabFM encodes it
  # internally and exposes the original labels via ``clf.classes_``.
  clf.fit(x_train, y_train)
  proba = np.asarray(clf.predict_proba(x_test))

  # clf.classes_ is already sorted (sklearn convention) and proba columns are
  # aligned to it, so it can be used directly as the metric label order. Only
  # the sklearn metrics need y_test as concrete labels.
  labels = np.asarray(clf.classes_)
  y_test = np.asarray(y_test)

  ll = log_loss(y_test, proba, labels=labels)
  if len(labels) == 2:
    # Binary AUC needs a 0/1 indicator; treat the second sorted label as the
    # positive class (matching proba[:, 1]).
    y_true_bin = (y_test == labels[1]).astype(int)
    auc = roc_auc_score(y_true_bin, proba[:, 1])
  else:
    auc = roc_auc_score(
        y_test, proba, multi_class="ovr", labels=labels, average="macro"
    )
  return auc, ll


def run_example(model=None):
  """Runs default and ensemble classification on TabArena fold 0.

  Args:
    model: An optional pre-loaded TabFM classification model. Loaded from
      Hugging Face when ``None``.

  Returns:
    A dict mapping "default"/"ensemble" to a (roc_auc, log_loss) tuple.
  """
  if model is None:
    # Option A: JAX Backend (default)
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")

    # Option B: PyTorch Backend
    # model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")

  x_train, y_train, x_test, y_test = _load_fold_0(TASK_ID)

  results = {}
  results["default"] = _evaluate(
      tabfm.TabFMClassifier(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  results["ensemble"] = _evaluate(
      tabfm.TabFMClassifier.ensemble(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  return results


if __name__ == "__main__":
  print(
      "Running TabFM classification on TabArena maternal_health_risk (fold 0)..."
      " (Note: compilation and model execution may take a few minutes on"
      " first run)"
  )
  scores = run_example()
  print(f"\n{'preset':<10} {'ROC_AUC':>9} {'log_loss':>9}")
  for preset, (auc, ll) in scores.items():
    print(f"{preset:<10} {auc:>9.5f} {ll:>9.5f}")

```

## File: examples/tabarena_regression_example.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression on a real TabArena task with TabFM v1.0.0.

Loads repeat-0 / fold-0 of the TabArena ``airfoil_self_noise`` task (a small
dataset with a mix of numerical and categorical columns), then runs inference
twice:

  * default  -- ``TabFMRegressor(model=...)`` (uniform averaging)
  * ensemble -- ``TabFMRegressor.ensemble(model=...)`` (feature crosses / SVD
    features + NNLS-weighted blending)

and reports RMSE, R^2 and MAE for each so the two presets can be compared.

Requires the optional ``openml`` dependency: ``pip install tabfm[examples]``.
"""

import numpy as np
import tabfm

# OpenML task id for the TabArena "airfoil_self_noise" regression task
# (1503 rows, 4 numerical + 1 categorical features).
TASK_ID = 363612
SEED = 0


def _load_fold_0(task_id):
  """Returns (X_train, y_train, X_test, y_test) for repeat-0 / fold-0.

  Note: both X and y are passed to TabFM as raw OpenML values. TabFM routes
  each feature column by dtype internally (pandas ``category`` columns are
  handled natively by ``CategoricalOrdinalEncoder``) and coerces the target to
  numeric itself (``check_array(y, dtype="numeric")``), so no manual coercion is
  needed here.
  """
  import openml  # pylint: disable=g-import-not-at-top

  task = openml.tasks.get_task(task_id)
  dataset = task.get_dataset()
  x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
  split = task.download_split().split[0][0][0]  # repeat 0, fold 0, sample 0

  x_train = x.iloc[split.train].copy()
  x_test = x.iloc[split.test].copy()
  y_train = y.iloc[split.train]  # raw target -- TabFM coerces to numeric itself
  y_test = y.iloc[split.test]
  return x_train, y_train, x_test, y_test


def _evaluate(reg, x_train, y_train, x_test, y_test):
  """Fits ``reg`` and returns (rmse, r2, mae) on the test fold."""
  from sklearn.metrics import (  # pylint: disable=g-import-not-at-top, g-multiple-import
      mean_absolute_error,
      mean_squared_error,
      r2_score,
  )

  reg.fit(x_train, y_train)
  pred = np.asarray(reg.predict(x_test), dtype=float).ravel()
  rmse = mean_squared_error(y_test, pred) ** 0.5
  return rmse, r2_score(y_test, pred), mean_absolute_error(y_test, pred)


def run_example(model=None):
  """Runs default and ensemble regression on TabArena fold 0.

  Args:
    model: An optional pre-loaded TabFM regression model. Loaded from Hugging
      Face when ``None``.

  Returns:
    A dict mapping "default"/"ensemble" to an (rmse, r2, mae) tuple.
  """
  if model is None:
    # Option A: JAX Backend (default)
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="regression")

    # Option B: PyTorch Backend
    # model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression")

  x_train, y_train, x_test, y_test = _load_fold_0(TASK_ID)

  results = {}
  results["default"] = _evaluate(
      tabfm.TabFMRegressor(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  results["ensemble"] = _evaluate(
      tabfm.TabFMRegressor.ensemble(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  return results


if __name__ == "__main__":
  print(
      "Running TabFM regression on TabArena airfoil_self_noise"
      " (fold 0)... (Note: compilation and model execution may take a few"
      " minutes on first run)"
  )
  scores = run_example()
  print(f"\n{'preset':<10} {'RMSE':>12} {'R2':>9} {'MAE':>12}")
  for preset, (rmse, r2, mae) in scores.items():
    print(f"{preset:<10} {rmse:>12.5g} {r2:>9.5f} {mae:>12.5g}")

```

## File: pyproject.toml

```
[project]
# Project metadata. Available keys are documented at:
# https://packaging.python.org/en/latest/specifications/declaring-project-metadata
name = "tabfm"
description = ""
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
authors = [{name = "tabfm authors", email="tabfm@google.com"}]
classifiers = [  # List of https://pypi.org/classifiers/
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]
keywords = []

# pip dependencies of the project
# Installed locally with `pip install -e .`
# Pinned versions for the full reproducible environment are in requirements.txt.
# Only the deps with known incompatibilities are constrained; the rest float.
#   - flax>=0.12.7 provides nnx.ModelAndOptimizer (used in checkpointing.py);
#     it also requires Python >=3.11.
#   - jaxtyping<0.3 / typeguard<3: jaxtyping 0.2.x integrates with typeguard 2.x;
#     newer typeguard (4.x) rejects jaxtyping's shape-string annotations.
# requirements.txt remains the full pinned lock for a reproducible environment.
dependencies = [
    "absl-py",
    "jaxtyping<0.3",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "typeguard<3",
    "huggingface-hub",
]

# `version` is automatically set by flit to use `tabfm.__version__`
dynamic = ["version"]

[project.urls]
homepage = "https://github.com/google-research/tabfm"
repository = "https://github.com/google-research/tabfm"
changelog = "https://github.com/google-research/tabfm/blob/main/CHANGELOG.md"
# documentation = ""

[project.optional-dependencies]
# Backend extras
jax = [
    "chex",
    "einops",
    "flax>=0.12.7",
    "jax",
    "jaxlib",
    "optax",
    "orbax-checkpoint",
]
pytorch = [
    "torch",
]

# Development deps (unittest, linting, formating,...)
# Installed through `pip install -e .[dev]`
dev = [
    "pytest",
    "pytest-xdist",
    "pylint>=2.6.0",
    "pyink",
]

# Dependencies for the runnable scripts under examples/. The TabArena examples
# pull datasets from OpenML; scikit-learn (a core dep) supplies the metrics.
# Installed through `pip install tabfm[examples]`.
examples = [
    "openml",
]

# GPU support. The base `jax`/`jaxlib` deps are CPU-only; install this extra
# (`pip install tabfm[cuda]`) to pull the CUDA 12 plugin + NVIDIA libraries for
# GPU execution. Kept optional so CPU-only / macOS / CI installs stay light.
cuda = [
    "jax[cuda12]",
]

[tool.pyink]
# Formatting configuration to follow Google style-guide
line-length = 80
unstable = true
pyink-indentation = 2
pyink-use-majority-quotes = true

[build-system]
# Build system specify which backend is used to build/install the project (flit,
# poetry, setuptools,...). All backends are supported by `pip install`
requires = ["flit_core >=3.8,<4"]
build-backend = "flit_core.buildapi"

[tool.flit.sdist]
# Flit specific options (files to exclude from the PyPI package).
# If using another build backend (setuptools, poetry), you can remove this
# section.
exclude = [
  # Do not release tests files on PyPI
  "**/*_test.py",
]

```

## File: requirements.txt

```
#
# This file is autogenerated by pip-compile with Python 3.11
# by the following command:
#
#    pip-compile --allow-unsafe --extra-index-url=https://download.pytorch.org/whl/cpu --extra=jax --extra=pytorch --index-url=https://pypi.org/simple/ --output-file=requirements.txt pyproject.toml
#
--extra-index-url https://download.pytorch.org/whl/cpu

absl-py==2.4.0
    # via
    #   chex
    #   optax
    #   orbax-checkpoint
    #   tabfm (pyproject.toml)
aiofiles==23.2.1
    # via orbax-checkpoint
annotated-doc==0.0.4
    # via typer
anyio==4.14.1
    # via httpx
certifi==2026.6.17
    # via
    #   httpcore
    #   httpx
chex==0.1.92
    # via tabfm (pyproject.toml)
click==8.4.2
    # via
    #   huggingface-hub
    #   typer
einops==0.8.2
    # via tabfm (pyproject.toml)
etils[epath,epy]==1.7.0
    # via orbax-checkpoint
filelock==3.29.4
    # via
    #   huggingface-hub
    #   torch
flax==0.12.7
    # via tabfm (pyproject.toml)
fsspec==2024.6.0
    # via
    #   etils
    #   huggingface-hub
    #   torch
h11==0.16.0
    # via httpcore
hf-xet==1.5.1
    # via huggingface-hub
httpcore==1.0.9
    # via httpx
httpx==0.28.1
    # via huggingface-hub
huggingface-hub==1.21.0
    # via tabfm (pyproject.toml)
humanize==4.9.0
    # via orbax-checkpoint
idna==3.18
    # via
    #   anyio
    #   httpx
importlib-resources==6.4.0
    # via etils
jax==0.10.1
    # via
    #   chex
    #   flax
    #   optax
    #   orbax-checkpoint
    #   tabfm (pyproject.toml)
jaxlib==0.10.1
    # via
    #   chex
    #   jax
    #   optax
    #   tabfm (pyproject.toml)
jaxtyping==0.2.25
    # via tabfm (pyproject.toml)
jinja2==3.1.6
    # via torch
joblib==1.4.2
    # via scikit-learn
markdown-it-py==3.0.0
    # via rich
markupsafe==3.0.3
    # via jinja2
mdurl==0.1.2
    # via markdown-it-py
ml-dtypes==0.5.0
    # via
    #   jax
    #   jaxlib
    #   tensorstore
mpmath==1.3.0
    # via sympy
msgpack==1.2.1
    # via
    #   flax
    #   orbax-checkpoint
networkx==3.6.1
    # via torch
numpy==2.2.0
    # via
    #   chex
    #   flax
    #   jax
    #   jaxlib
    #   jaxtyping
    #   ml-dtypes
    #   opt-einsum
    #   optax
    #   orbax-checkpoint
    #   pandas
    #   scikit-learn
    #   scipy
    #   tabfm (pyproject.toml)
    #   tensorstore
    #   treescope
opt-einsum==3.3.0
    # via jax
optax==0.2.8
    # via
    #   flax
    #   tabfm (pyproject.toml)
orbax-checkpoint==0.12.0
    # via
    #   flax
    #   tabfm (pyproject.toml)
packaging==26.2
    # via huggingface-hub
pandas==2.2.3
    # via tabfm (pyproject.toml)
prometheus-client==0.20.0
    # via orbax-checkpoint
protobuf==5.29.6
    # via orbax-checkpoint
psutil==5.9.8
    # via orbax-checkpoint
pygments==2.20.0
    # via rich
python-dateutil==2.9.0.post0
    # via pandas
pytz==2024.1
    # via pandas
pyyaml==6.0.1
    # via
    #   flax
    #   huggingface-hub
    #   orbax-checkpoint
rich==13.7.1
    # via
    #   flax
    #   typer
scikit-learn==1.6.0
    # via tabfm (pyproject.toml)
scipy==1.17.1
    # via
    #   jax
    #   jaxlib
    #   scikit-learn
    #   tabfm (pyproject.toml)
shellingham==1.5.4
    # via typer
simplejson==3.19.2
    # via orbax-checkpoint
six==1.16.0
    # via python-dateutil
sympy==1.14.0
    # via torch
tensorstore==0.1.84
    # via
    #   flax
    #   orbax-checkpoint
threadpoolctl==3.5.0
    # via scikit-learn
toolz==1.1.0
    # via chex
torch==2.12.1+cpu
    # via tabfm (pyproject.toml)
tqdm==4.68.3
    # via huggingface-hub
treescope==0.1.10
    # via flax
typeguard==2.13.3
    # via
    #   jaxtyping
    #   tabfm (pyproject.toml)
typer==0.24.2
    # via huggingface-hub
typing-extensions==4.15.0
    # via
    #   anyio
    #   chex
    #   etils
    #   flax
    #   huggingface-hub
    #   jaxtyping
    #   orbax-checkpoint
    #   torch
tzdata==2024.1
    # via pandas
uvloop==0.19.0
    # via orbax-checkpoint
zipp==4.1.0
    # via etils

# The following packages are considered to be unsafe in a requirements file:
setuptools==81.0.0
    # via torch

```

## File: tabfm/BUILD

```
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "tabfm",
    srcs = ["__init__.py"],
    deps = [
        "//tabfm/src:classifier_and_regressor",
        "//tabfm/src:tabfm_v1_0_0",
    ],
)

```

## File: tabfm/__init__.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tabfm API."""

try:
  from tabfm.src.jax import tabfm_v1_0_0 as tabfm_v1_0_0_jax
  tabfm_v1_0_0 = tabfm_v1_0_0_jax
except ImportError:
  # JAX is not installed or incomplete, tabfm_v1_0_0_jax is not available.
  pass

try:
  from tabfm.src.pytorch import tabfm_v1_0_0 as tabfm_v1_0_0_pytorch
except ImportError:
  # PyTorch is not installed or incomplete, tabfm_v1_0_0_pytorch is not available.
  pass

from tabfm.src.classifier_and_regressor import TabFMClassifier, TabFMRegressor

# A new PyPI release will be pushed every time `__version__` is increased.
# When changing this, also update the CHANGELOG.md.
__version__ = '1.0.1'

```

## File: tabfm/src/BUILD

```
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tabfm:__subpackages__"])

py_library(
    name = "classifier_and_regressor",
    srcs = ["classifier_and_regressor.py"],
    visibility = ["//tabfm:__subpackages__"],
    deps = [
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//jaxtyping",
        "@pip//numpy",
        "@pip//pandas",
        "@pip//scikit_learn",
    ],
)

py_test(
    name = "classifier_and_regressor_test",
    srcs = ["classifier_and_regressor_test.py"],
    deps = [
        ":classifier_and_regressor",
        "//tabfm/src/jax:model",
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//numpy",
        "@pip//pandas",
    ],
)

py_test(
    name = "classifier_and_regressor_multidevice_test",
    srcs = ["classifier_and_regressor_multidevice_test.py"],
    deps = [
        ":classifier_and_regressor",
        "//tabfm/src/jax:model",
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//numpy",
    ],
)

py_library(
    name = "torch_convert",
    srcs = ["hugging_face/torch_convert.py"],
    deps = [
        "//tabfm/src/jax:model",
        "//tabfm/src/pytorch:model",
        "@pip//numpy",
        "@pip//torch",
        "@pip//flax",
        "@pip//jax",
    ],
)

```

## File: tabfm/src/__init__.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tabfm src package

```

