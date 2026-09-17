# TabFM Codebase Part 3

## File: tabfm/src/classifier_and_regressor.py

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

"""Classifier and Regressor interfaces for TabFM.

This module provides scikit-learn compatible classifier and regressor classes
built on top of the TabFM (Tabular Foundation Model) foundation model, as
well as the data preprocessing and ensemble-generation utilities they rely on.

Key classes:
  - CategoricalOrdinalEncoder: Ordinal encoding by appearance or frequency.
  - TransformToNumerical: Mixed-type DataFrame --> numeric array pipeline.
  - EnsembleGenerator: Creates diverse data views for ensemble inference.
  - TabFMClassifier: sklearn-compatible TabFM classifier.
  - TabFMRegressor:  sklearn-compatible TabFM regressor.
"""

import collections
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging
import numpy as np

try:
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.experimental import multihost_utils
  from jax.sharding import NamedSharding, PartitionSpec
  HAS_JAX = True
  Array = jax.Array
except ImportError:
  HAS_JAX = False
  Array = np.ndarray  # Fallback for type annotation parser

try:
  import torch
  from tabfm.src.pytorch.model import move_cache_to_device
  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False
import pandas as pd
import scipy.optimize as opt
import scipy.special
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

import jaxtyping as jt
import typeguard

jt.typed = jt.jaxtyped(typechecker=typeguard.typechecked)

# pylint: disable=invalid-name

# Single source of truth for the default ensemble seed. Threaded into both
# estimators and, from there, into every stochastic component (KFold OOF
# cross-fit, SVD, quantile normalizer, feature crosses / row subsampling,
# class shift). Type-detection heuristics are deliberately NOT seeded from
# this -- column-type detection must stay stable across ensemble seeds so the
# feature schema doesn't change when only the model seed varies.
_DEFAULT_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------


class CategoricalOrdinalEncoder(BaseEstimator, TransformerMixin):
  """Ordinal Encoder that assigns indices based on order of appearance or frequency.

  Args:
    dtype: Output dtype for encoded values.
    handle_unknown: How to handle unknown categories at transform time. Only
      ``"use_encoded_value"`` is currently supported.
    unknown_value: Encoded value used for unknown categories.
    encoded_missing_value: Encoded value used for NaN / missing values.
    min_frequency: Categories appearing fewer than this many times in the
      training data are treated as unknown.
    mode: Encoding order strategy.
      - ``"appearance"``: categories are ordered by first occurrence in the
        data (original behaviour).
      - ``"frequency"``: categories are sorted by descending frequency, so
        the most common category receives index 0.
      - ``"alphabetical"``: categories are sorted ascending (alphabetically),
        matching scikit-learn's ``LabelEncoder`` convention.
  """

  categories_: List[np.ndarray]

  def __init__(
      self,
      dtype=np.float64,
      handle_unknown: str = "use_encoded_value",
      unknown_value: int = -1,
      encoded_missing_value: int = -1,
      min_frequency: int = 1,
      mode: str = "appearance",
  ):
    self.dtype = dtype
    self.handle_unknown = handle_unknown
    self.unknown_value = unknown_value
    self.encoded_missing_value = encoded_missing_value
    self.min_frequency = min_frequency
    self.mode = mode

  def fit(self, X: Any, y: Any = None) -> "CategoricalOrdinalEncoder":
    """Fits the encoder to the data.

    Args:
      X: Input array-like of shape (n_samples, n_features).
      y: Ignored. Kept for sklearn compatibility.

    Returns:
      self.
    """
    if hasattr(X, "iloc"):
      X = X.values
    self.categories_ = []
    n_features = X.shape[1]

    for i in range(n_features):
      col = X[:, i]

      # Count occurrences to filter rare categories
      counts = pd.Series(col).value_counts()
      rare_cats = counts[counts < self.min_frequency].index

      # Determine the candidate categories in the order dictated by the mode.
      if self.mode == "frequency":
        # Descending frequency. value_counts is already frequency-sorted, with
        # ties broken by appearance order via pandas' stable sort.
        candidates = counts.index.tolist()
      elif self.mode == "appearance":
        # Order of first appearance.
        candidates = pd.unique(col)
      elif self.mode == "alphabetical":
        # Order of first appearance; sorted below once NaNs are removed.
        candidates = pd.unique(col)
      else:
        raise ValueError(
            f"Unknown mode: {self.mode!r}. Expected one of 'appearance', "
            "'alphabetical', or 'frequency'."
        )

      # Drop NaNs, the literal string "nan" (to match TF behavior), and rare
      # categories.
      uniques = [
          u
          for u in candidates
          if not pd.isna(u) and str(u) != "nan" and u not in rare_cats
      ]
      if self.mode == "alphabetical":
        # Sort categories so encoding matches sklearn's LabelEncoder convention
        # (classes ordered ascending / alphabetically).
        uniques = sorted(uniques)

      self.categories_.append(np.array(uniques))

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transforms the data using the fitted encoder.

    Args:
      X: Input array-like of shape (n_samples, n_features).

    Returns:
      Encoded array of shape (n_samples, n_features) with dtype ``self.dtype``.
    """
    check_is_fitted(self)
    if hasattr(X, "iloc"):
      X = X.values
    n_samples, n_features = X.shape
    X_out = np.full(
        (n_samples, n_features), self.unknown_value, dtype=self.dtype
    )

    for i in range(n_features):
      col = X[:, i]
      cats = self.categories_[i]
      cat_to_idx = {c: idx for idx, c in enumerate(cats)}
      mapped = pd.Series(col).map(cat_to_idx)

      # Convert to numeric to avoid Categorical/fillna issues
      if mapped.dtype.name == "category":
        mapped = mapped.astype(object)

      # Fill NaNs/Unknowns with unknown_value
      X_out[:, i] = mapped.fillna(self.unknown_value).values.astype(self.dtype)

    return X_out

  def inverse_transform(self, X: np.ndarray) -> np.ndarray:
    """Decodes indices back to original categorical values.

    Args:
      X: Encoded array of shape (n_samples, n_features).

    Returns:
      Decoded object array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    if hasattr(X, "iloc"):
      X = X.values
    n_samples, n_features = X.shape
    X_out = np.empty((n_samples, n_features), dtype=object)

    for i in range(n_features):
      col = X[:, i]
      cats = self.categories_[i]
      valid_mask = col != self.unknown_value
      if np.any(valid_mask):
        indices = col[valid_mask].astype(int)
        X_out[valid_mask, i] = cats[indices]

    return X_out


def _looks_like_datetime(X: pd.Series) -> bool:
  """Checks if a text-typed pandas Series looks like datetime data.

  Considers object- and string-dtype (pandas>=3) columns. Uses a lenient
  ``errors="coerce"`` parse with a threshold (see below): a column is treated
  as datetime as long as a meaningful fraction of its values parse as dates,
  so partially-date columns (dates mixed with some non-date values) are still
  detected on purpose.

  Args:
    X: A pandas Series whose dtype may or may not be object/string.

  Returns:
    True if the series looks like datetime data stored as text.
  """
  # Accept object dtype and the pandas string dtype (incl. the pyarrow-backed
  # default in pandas>=3); otherwise date-as-text columns load as 'string',
  # fail this object-only gate, and silently fall through to categorical.
  if not (pd.api.types.is_object_dtype(X.dtype)
          or isinstance(X.dtype, pd.StringDtype)):
    return False
  if X.isnull().all():
    return False
  try:
    pd.to_numeric(X)
  except (ValueError, TypeError):
    try:
      if len(X) > 500:
        # Subsample only to keep the datetime parse-check fast. The fixed seed
        # is deliberate and independent of the ensemble seed: this is a
        # type-detection heuristic, so it must not depend on -- or perturb --
        # the model's random_state (column-type detection must stay stable
        # across ensemble seeds). A random sample is used over .head() so a
        # sorted / front-loaded column is still represented.
        X = X.sample(n=500, random_state=0)
      result = pd.to_datetime(X, errors="coerce", format="mixed")
      if result.isnull().mean() > 0.8:
        return False
      return True
    except Exception:  # pylint: disable=broad-except
      return False
  return False


class DatetimeTransformer(BaseEstimator, TransformerMixin):
  """Transformer that converts raw datetime columns into numeric features.

  Each datetime column is expanded into derived features (year, month, day,
  dayofweek) in addition to its Unix-nanosecond integer representation.
  """

  features_in: List[int]
  _fillna_map: Dict[int, Any]

  def __init__(self, features: Optional[List[str]] = None):
    """Initialises the transformer.

    Args:
      features: List of datetime sub-fields to extract per column.
        Defaults to ``['year', 'month', 'day', 'dayofweek']``.
    """
    if features is None:
      self.features = ["year", "month", "day", "dayofweek"]
    else:
      self.features = features
    self.features_in = []
    self._fillna_map = {}

  def fit(self, X: Any, y: Any = None) -> "DatetimeTransformer":
    """Fits the transformer, recording fill-in values for missing datetimes.

    Args:
      X: DataFrame or array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    if not isinstance(X, pd.DataFrame):
      X = pd.DataFrame(X)
    # Track columns by position: non-string column names would break the
    # synthetic "<column>.<part>" labels built in transform().
    self.features_in = list(range(X.shape[1]))
    for pos in self.features_in:
      series = pd.to_datetime(
          X.iloc[:, pos], utc=True, errors="coerce", format="mixed"
      )
      self._fillna_map[pos] = series.mean()
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transforms datetime columns into numeric arrays.

    Args:
      X: DataFrame or array-like of shape (n_samples, n_features).

    Returns:
      Numeric array of shape (n_samples, n_features * (1 + len(self.features))).
    """
    if not isinstance(X, pd.DataFrame):
      X = pd.DataFrame(X)
    X_datetime = pd.DataFrame(index=X.index)
    for pos in self.features_in:
      series = pd.to_datetime(
          X.iloc[:, pos].copy(), utc=True, errors="coerce", format="mixed"
      )
      broken_idx = series[
          (series == "NaT") | series.isna() | series.isnull()
      ].index
      if len(broken_idx) > 0:
        series.loc[broken_idx] = self._fillna_map[pos]
      X_datetime[f"{pos}"] = pd.to_numeric(series)
      for dt_feature in self.features:
        X_datetime[f"{pos}.{dt_feature}"] = getattr(
            series.dt, dt_feature
        ).astype(np.int64)
    return X_datetime.values


class TransformToNumerical(TransformerMixin, BaseEstimator):
  """Transforms non-numerical data in a DataFrame to numerical representations.

  Automatically detects categorical, datetime, and numeric columns, applying
  appropriate encoding for each type via a sklearn ``ColumnTransformer``.
  Falls through to an identity transform for non-DataFrame inputs.

  Attributes:
    tfm_: The fitted ``ColumnTransformer`` (or ``FunctionTransformer`` for
      non-DataFrame inputs).
  """

  tfm_: Union[ColumnTransformer, FunctionTransformer]

  def __init__(
      self,
      verbose: bool = False,
      cat_encoder_mode: str = "appearance",
      min_cat_frequency: int = 2,
  ):
    """Initialises the transformer.

    Args:
      verbose: Whether to print column-classification information.
      cat_encoder_mode: Encoding mode passed to ``CategoricalOrdinalEncoder``
        (``"appearance"`` or ``"frequency"``).
      min_cat_frequency: Minimum frequency for a category to be kept distinct;
        rarer categories are encoded as unknown.
    """
    self.verbose = verbose
    self.cat_encoder_mode = cat_encoder_mode
    self.min_cat_frequency = min_cat_frequency

  def fit(self, X: Any, y: Any = None) -> "TransformToNumerical":
    """Configure transformers for different column types in the input data.

    Args:
      X: Array-like of shape (n_samples, n_features).  If a DataFrame, column
        types are used to determine the appropriate transformation.
      y: Ignored.

    Returns:
      self.
    """
    if not hasattr(
        X, "columns"
    ):  # proxy way to check whether X is a dataframe without importing pandas
      # no dataframe
      self.tfm_ = FunctionTransformer()
      return self

    if X.columns.duplicated().any():
      # sklearn's ColumnTransformer rejects duplicate column names; fail fast
      # here with an actionable message instead of crashing mid-pipeline.
      raise ValueError(
          "X contains duplicate column names:"
          f" {X.columns[X.columns.duplicated()].unique().tolist()}. Rename or"
          " deduplicate them before fitting, e.g. with"
          " X.columns = range(X.shape[1])."
      )

    datetime_pos = []
    cat_pos = []
    numeric_pos = []

    # Classify columns by position so column labels (which may be ints or
    # other non-strings) are never used for lookups.
    for pos in range(X.shape[1]):
      series = X.iloc[:, pos]
      if pd.api.types.is_datetime64_any_dtype(series.dtype):
        datetime_pos.append(pos)
      elif _looks_like_datetime(series):
        datetime_pos.append(pos)
      elif pd.api.types.is_numeric_dtype(series.dtype):
        numeric_pos.append(pos)
      else:
        # fallback to categorical if unknown
        cat_pos.append(pos)

    self.tfm_ = ColumnTransformer(
        transformers=[
            (
                "categorical",
                CategoricalOrdinalEncoder(
                    dtype=np.int64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    min_frequency=self.min_cat_frequency,
                    mode=self.cat_encoder_mode,
                ),
                cat_pos,
            ),
            ("continuous", SimpleImputer(), numeric_pos),
            ("datetime", DatetimeTransformer(), datetime_pos),
        ]
    )
    self.tfm_.fit(X)

    if self.verbose:
      selected_cols = []
      for name, tfm, pos in self.tfm_.transformers_:
        if tfm != "drop":
          cols = list(X.columns[pos])
          selected_cols.extend(cols)
          print(f"Columns classified as {name}: {cols}")
      dropped_cols = set(X.columns).difference(set(selected_cols))
      if len(dropped_cols) >= 1:
        print(
            "The following columns are not used due to their data type:"
            f" {list(dropped_cols)}"
        )

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transform features using the fitted transformer.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Numeric array of shape (n_samples, n_features_out).
    """
    return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
  """Removes features with only one unique value in the training set.

  Attributes:
    n_features_in_ : int
    Number of features in the training data.

    n_features_out_ : int
    Number of features after filtering.

    features_to_keep_ : ndarray
    Boolean mask for features to keep.
  """

  features_to_keep_: np.ndarray
  n_features_out_: int
  n_features_in_: int

  def __init__(self, threshold: int = 1):
    """Initialises the filter.

    Notes
    -----
    1. Features with unique values <= threshold are removed.
    2. When the input dataset has very few samples (n_samples <= threshold), all
       features are preserved
       regardless of their unique value counts. This is a safety mechanism
       because:
       - With few samples, it's difficult to reliably assess feature variability
       - A feature might appear constant in few samples but vary in the complete
       dataset

    Args:
      threshold: Features with at most this many unique values are removed.
    """
    self.threshold = threshold

  def fit(self, X: Any, y: Any = None) -> "UniqueFeatureFilter":
    """Learn which features to keep based on unique value counts.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)
    self.n_features_in_ = X.shape[1]
    if X.shape[0] <= self.threshold:
      self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
    else:
      # For each feature, check if it has more than threshold unique values
      self.features_to_keep_ = np.array([
          len(np.unique(X[:, i])) > self.threshold
          for i in range(self.n_features_in_)
      ])

    self.n_features_out_ = np.sum(self.features_to_keep_)

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Filter features according to unique value counts.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Array of shape (n_samples, n_features_out_).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
  """Clips extreme values based on training data distribution.

  This implementation uses a two-stage Z-score based approach to identify and
  clip outliers:
  1. First stage: Identify values beyond z standard deviations and mark as
  missing
  2. Second stage: Recompute statistics without outliers for more robust bounds
  3. Final stage: Apply log-based clipping to maintain data distribution

  Attributes:
    means_: Per-feature means after outlier removal.
    stds_: Per-feature standard deviations after outlier removal.
    lower_bounds_: Per-feature clipping lower bounds.
    upper_bounds_: Per-feature clipping upper bounds.
  """

  means_: np.ndarray
  stds_: np.ndarray
  lower_bounds_: np.ndarray
  upper_bounds_: np.ndarray

  def __init__(self, threshold: float = 4.0):
    """Initialises the remover.

    Args:
      threshold: Values beyond this many standard deviations are outliers.
    """
    self.threshold = threshold

  def fit(self, X: Any, y: Any = None) -> "OutlierRemover":
    """Learn clipping bounds from training data.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)

    # First stage: Identify outliers using initial statistics
    self.means_ = np.nanmean(X, axis=0)
    self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0)

    # Ensure standard deviations are not zero
    self.stds_ = np.maximum(self.stds_, 1e-6)

    # Create a clean copy with outliers replaced by NaN
    X_clean = X.copy()
    lower_bounds = self.means_ - self.threshold * self.stds_
    upper_bounds = self.means_ + self.threshold * self.stds_

    # Create masks for values outside bounds
    lower_mask = X < lower_bounds[np.newaxis, :]
    upper_mask = X > upper_bounds[np.newaxis, :]
    outlier_mask = np.logical_or(lower_mask, upper_mask)

    # Set outliers to NaN
    X_clean[outlier_mask] = np.nan

    # Second stage: Recompute statistics without outliers
    self.means_ = np.nanmean(X_clean, axis=0)
    self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

    # Ensure standard deviations are not zero
    self.stds_ = np.maximum(self.stds_, 1e-6)

    # Compute final bounds
    self.lower_bounds_ = self.means_ - self.threshold * self.stds_
    self.upper_bounds_ = self.means_ + self.threshold * self.stds_

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Clip values based on learned bounds with log-based adjustments.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Clipped array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
    X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)
    return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
  """Standard scaling with clipping.

  This scaler computes the mean and standard deviation of the training data,
  adds a small epsilon to the standard deviation to avoid division by zero,
  and clips the transformed values to a reasonable range.

  Attributes:
    mean_ : ndarray of shape (n_features,)
      The mean value for each feature in the training set.

    scale_ : ndarray of shape (n_features,)
      The standard deviation for each feature in the training set with epsilon
      added.
  """

  mean_: np.ndarray
  scale_: np.ndarray

  def __init__(
      self,
      clip_min: float = -100,
      clip_max: float = 100,
      epsilon: float = 1e-6,
  ):
    """Initialises the scaler.

    Args:
      clip_min: Lower bound for clipping scaled values.
      clip_max: Upper bound for clipping scaled values.
      epsilon: Small constant added to the standard deviation to avoid
        division by zero.
    """
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.epsilon = epsilon

  def fit(self, X: Any, y: Any = None) -> "CustomStandardScaler":
    """Compute the mean and std to be used for scaling.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)
    self.mean_ = np.mean(X, axis=0)
    self.scale_ = np.std(X, axis=0) + self.epsilon
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Standardize features by removing the mean and scaling to unit variance.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Scaled and clipped array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    X_scaled = (X - self.mean_) / self.scale_
    return np.clip(X_scaled, self.clip_min, self.clip_max)


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
  """Quantile transformer adapted for tabular deep learning models.

  This implementation is based on research from the RTDL group and adds noise to
  training
  data before applying quantile transformation, improving robustness and
  generalization.
  It also dynamically adjusts the number of quantiles based on data size.

  Attributes:
    normalizer_: The fitted ``QuantileTransformer``.

  Notes:
    Adapted from
    https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
    following
    https://github.com/dholzmueller/pytabkit/blob/949bf81e3964f65a33dd2c252c3713c239c17b2d/pytabkit/models/utils.py#L431
  """

  normalizer_: QuantileTransformer

  def __init__(
      self,
      noise: float = 1e-3,
      n_quantiles: int = 1000,
      subsample: int = 1_000_000_000,
      output_distribution: str = "normal",
      random_state: Optional[int] = None,
  ):
    """Initialises the transformer.

    Args:
      noise: Relative magnitude of Gaussian noise to add. Set to 0 to disable.
      n_quantiles: Maximum number of quantiles. Actual number is determined
        dynamically as ``min(n_samples // 30, n_quantiles)``, with a floor of
        10.
      subsample: Maximum samples used to estimate quantiles.
      output_distribution: Target marginal distribution (``"uniform"`` or
        ``"normal"``).
      random_state: Seed for reproducibility.
    """
    self.noise = noise
    self.n_quantiles = n_quantiles
    self.subsample = subsample
    self.output_distribution = output_distribution
    self.random_state = random_state

  def fit(self, X: Any, y: Any = None) -> "RTDLQuantileTransformer":
    """Fit the quantile transformer to training data with optional noise.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    # Calculate the number of quantiles based on data size
    n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

    # Initialize QuantileTransformer
    normalizer = QuantileTransformer(
        output_distribution=self.output_distribution,
        n_quantiles=n_quantiles,
        subsample=self.subsample,
        random_state=self.random_state,
    )

    # Add noise if required
    X_modified = self._add_noise(X) if self.noise > 0 else X

    # Fit the normalizer
    normalizer.fit(X_modified)

    # Show that it's fitted
    self.normalizer_ = normalizer

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transform data using the fitted quantile transformer.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Transformed array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    return self.normalizer_.transform(X)

  def _add_noise(self, X: np.ndarray) -> np.ndarray:
    """Add noise to the input data proportional to feature standard deviations.

    The noise magnitude is controlled by the 'noise' parameter and is scaled
    inversely to the standard deviation of each feature to ensure
    consistent noise levels across features of different scales.

    Args:
      X: Array of shape (n_samples, n_features).

    Returns:
      Noisy array of shape (n_samples, n_features).
      The input data with added Gaussian noise.
    """
    stds = np.std(X, axis=0, keepdims=True)
    noise_std = self.noise / np.maximum(stds, self.noise)
    rng = np.random.default_rng(self.random_state)
    return X + noise_std * rng.standard_normal(X.shape)


class FeatureShuffler:
  """Generates feature permutations for ensemble creation.

  Attributes:
    rng_: The random number generator used for shuffling.
  """

  rng_: random.Random

  def __init__(
      self,
      n_features: int,
      method: str = "random",
      random_state: Optional[int] = None,
  ):
    """Initialises the shuffler.

    Args:
      n_features: Number of features to shuffle.
      method: Shuffling strategy: ``"random"`` or ``"none"``.
      random_state: Seed for reproducibility.
    """
    self.n_features = n_features
    self.method = method
    self.random_state = random_state
    self.rng_ = random.Random(self.random_state)

  def shuffle(self, n_estimators: int) -> List[np.ndarray]:
    """Generates a list of feature-shuffle patterns.

    Args:
      n_estimators: Number of shuffle patterns to generate.

    Returns:
      List of ``n_estimators`` arrays, each of length ``self.n_features``,
      representing a permutation of feature indices.
    """
    self.rng_ = random.Random(self.random_state)
    feature_indices = list(range(self.n_features))

    if self.method == "none" or n_estimators == 1:
      shuffle_patterns = [feature_indices]
    elif self.method == "random":
      if self.n_features <= 5:
        all_perms = [
            list(perm) for perm in itertools.permutations(feature_indices)
        ]
        shuffle_patterns = self.rng_.sample(
            all_perms, min(n_estimators, len(all_perms))
        )
      else:
        shuffle_patterns = [
            self.rng_.sample(feature_indices, self.n_features)
            for _ in range(n_estimators)
        ]
    else:
      raise ValueError(
          f"Unknown method: {self.method}. Use 'random' or 'none'."
      )

    return [np.array(p) for p in shuffle_patterns]


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
  """Preprocessing pipeline combining scaling, normalization, and outlier handling.

  Attributes:
    standard_scaler_: Fitted ``CustomStandardScaler``.
    normalizer_: Fitted normalization transformer (or ``None``).
    outlier_remover_: Fitted ``OutlierRemover``.
    X_transformed_: Cached transformed training data.
    X_min_: Per-feature minimum of scaled data (used for clipping at transform).
    X_max_: Per-feature maximum of scaled data (used for clipping at transform).
  """

  n_features_in_: int
  standard_scaler_: Optional[CustomStandardScaler]
  normalizer_: Optional[Any]
  outlier_remover_: Optional[OutlierRemover]
  X_transformed_: np.ndarray
  X_min_: np.ndarray
  X_max_: np.ndarray

  def __init__(
      self,
      normalization_method: str = "none",
      outlier_threshold: float = 4.0,
      random_state: Optional[int] = None,
  ):
    """Initialises the pipeline.

    Args:
      normalization_method: Normalization strategy: ``"none"``, ``"power"``,
        ``"quantile"``, ``"quantile_rtdl"``, or ``"robust"``.
      outlier_threshold: Z-score threshold for outlier detection.
      random_state: Seed for reproducible normalization.
    """
    self.normalization_method = normalization_method
    self.outlier_threshold = outlier_threshold
    self.random_state = random_state

  def fit(self, X: Any, y: Any = None) -> "PreprocessingPipeline":
    """Fit the preprocessing pipeline.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X, ensure_min_features=0)
    # If there are no features, there's nothing to preprocess.
    if self.n_features_in_ == 0:
      self.standard_scaler_ = None
      self.normalizer_ = None
      self.outlier_remover_ = None
      self.X_transformed_ = X
      return self
    # 1. Apply standard scaling
    self.standard_scaler_ = CustomStandardScaler()
    X_scaled = self.standard_scaler_.fit_transform(X)

    # 2. Apply normalization
    if self.normalization_method != "none":
      if self.normalization_method == "power":
        self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
      elif self.normalization_method == "quantile":
        self.normalizer_ = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
      elif self.normalization_method == "quantile_rtdl":
        self.normalizer_ = Pipeline([
            (
                "quantile_rtdl",
                RTDLQuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                ),
            ),
            ("std", StandardScaler()),
        ])
      elif self.normalization_method == "robust":
        self.normalizer_ = RobustScaler(unit_variance=True)
      else:
        raise ValueError(
            f"Unknown normalization method: {self.normalization_method}"
        )
      self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
      self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
      X_normalized = self.normalizer_.fit_transform(X_scaled)
    else:
      self.normalizer_ = None
      X_normalized = X_scaled

    # 3. Handle outliers
    self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
    self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Apply the preprocessing pipeline.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Preprocessed array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False, copy=True, ensure_min_features=0)
    if self.n_features_in_ == 0:
      return X
    if self.standard_scaler_ is not None:
      X = self.standard_scaler_.transform(X)
    if self.normalizer_ is not None:
      try:
        # this can fail in rare cases if there is an outlier in X that was not present in fit()
        X = self.normalizer_.transform(X)
      except ValueError:
        # clip values to train min/max
        X = np.clip(X, self.X_min_, self.X_max_)
        X = self.normalizer_.transform(X)
    if self.outlier_remover_ is not None:
      X = self.outlier_remover_.transform(X)
    return X


class EnsembleGenerator(TransformerMixin, BaseEstimator):
  """Generate diverse ensemble variants for robust tabular prediction with TabFM.

  Creates multiple views of a dataset by combining different feature shuffles,
  class-label shifts, categorical value permutations, and normalization methods.

  Attributes:
    norm_methods_: List of normalization methods in use.
    cat_features_: Indices of categorical features after unique filtering.
    cat_values_: Unique values per categorical feature column index.
    X_: Training features after unique filtering (cached).
    y_: Training labels (cached).
    n_features_in_: Number of features after unique filtering.
    n_classes_: Number of unique classes (classification only).
    rng_: Random-number generator.
    ensemble_configs_: Ordered dict mapping norm method -> list of
      (shuffle_pattern, shift_offset, cat_perm) tuples.
    feature_shuffle_patterns_: Ordered dict mapping norm method -> shuffle
      patterns.
    class_shift_offsets_: Ordered dict mapping norm method -> shift offsets.
    cat_permutations_: Ordered dict mapping norm method -> cat permutation
      dicts.
    row_subsample_patterns_: Ordered dict mapping norm method -> row subsample
      patterns (used for max_num_rows).
    preprocessors_: Fitted ``PreprocessingPipeline`` per norm method.
    unique_filter_: Fitted ``UniqueFeatureFilter`` used to remove duplicate or
      constant features.
    n_original_features_: Number of input features before feature augmentation
      crosses or SVD.
    cross_pairs_: List of feature index pairs generated for feature crosses.
    cross_pool_start_: Starting column index in ``X_`` for the feature crosses
      pool.
    cross_pool_end_: Ending column index in ``X_`` for the feature crosses pool.
    svd_pipeline_: Fitted SVD pipeline used to generate SVD structural features.
    svd_pool_start_: Starting column index in ``X_`` for the SVD features pool.
    svd_pool_end_: Ending column index in ``X_`` for the SVD features pool.
    k_crosses_list_: List of feature cross counts to sample per ensemble member.
    k_svd_list_: List of SVD feature counts to sample per ensemble member.
  """

  norm_methods_: List[str]
  cat_features_: np.ndarray
  cat_values_: Dict[int, np.ndarray]
  X_: np.ndarray
  y_: np.ndarray
  n_features_in_: int
  n_classes_: int
  rng_: random.Random
  ensemble_configs_: collections.OrderedDict
  feature_shuffle_patterns_: collections.OrderedDict
  class_shift_offsets_: collections.OrderedDict
  cat_permutations_: collections.OrderedDict
  row_subsample_patterns_: collections.OrderedDict
  preprocessors_: Dict[str, PreprocessingPipeline]
  unique_filter_: UniqueFeatureFilter
  n_original_features_: int
  cross_pairs_: List[Any]
  cross_pool_start_: int
  cross_pool_end_: int
  svd_pipeline_: Any
  svd_pool_start_: int
  svd_pool_end_: int
  k_crosses_list_: List[int]
  k_svd_list_: List[int]

  def __init__(
      self,
      n_estimators: int,
      norm_methods: Union[str, List[str], None] = None,
      feat_shuffle_method: str = "random",
      class_shift: bool = True,
      cat_features: Optional[List[int]] = None,
      permute_categorical: bool = False,
      outlier_threshold: float = 4.0,
      max_num_features: Optional[int] = 500,
      max_num_rows: Optional[int] = None,
      n_feature_crosses: Union[int, str] = 0,
      n_svd_features: Union[int, str] = 0,
      total_svd_pool: Optional[int] = None,
      random_state: Optional[int] = None,
      task: str = "classification",
  ):
    """Initialises the generator.

    Args:
      n_estimators: Number of ensemble members to generate.
      norm_methods: Normalization method(s) to use. If ``None``, defaults to
        ``["none", "power"]``. May be a single string or a list.
      feat_shuffle_method: Feature-permutation strategy (``"random"`` or
        ``"none"``).
      class_shift: Whether to apply random class-label shifts (classification
        only).
      cat_features: Indices of categorical features in the *encoded* input.
      permute_categorical: Whether to randomly permute categorical values across
        ensemble members.
      outlier_threshold: Z-score threshold forwarded to``OutlierRemover``.
      max_num_features: Maximum number of features to subsample per ensemble
        member.
      max_num_rows: Maximum number of rows to subsample per ensemble member.
      n_feature_crosses: ``"sqrt"`` to add sqrt(n_features) random feature
        crosses per ensemble member, or ``0`` to disable.
      n_svd_features: ``"sqrt"`` to add sqrt(n_features) random SVD features per
        ensemble member, or ``0`` to disable.
      total_svd_pool: Total pool size of SVD features to generate.
      task: Either ``"classification"`` or ``"regression"``.
    """
    self.n_estimators = n_estimators
    self.norm_methods = norm_methods
    self.feat_shuffle_method = feat_shuffle_method
    self.class_shift = class_shift
    self.cat_features = cat_features
    self.permute_categorical = permute_categorical
    self.outlier_threshold = outlier_threshold
    self.max_num_features = max_num_features
    self.max_num_rows = max_num_rows
    self.n_feature_crosses = n_feature_crosses
    self.n_svd_features = n_svd_features
    self.total_svd_pool = total_svd_pool
    self.random_state = random_state
    self.task = task

  def _get_n_features_to_add(
      self, n_features_requested: Union[int, str, None], n_cols: int
  ) -> int:
    if n_features_requested in (0, None):
      return 0
    if (
        isinstance(n_features_requested, str)
        and n_features_requested.lower() == "sqrt"
    ):
      return max(1, int(np.sqrt(n_cols)))
    raise ValueError(
        f"Invalid requested number of features: {n_features_requested!r}."
        " Expected 0 (disabled) or 'sqrt'."
    )

  def _get_member_n_features_list(
      self, n_features_requested: Any, n_cols: int
  ) -> List[int]:
    """Resolves the number of features to add for each ensemble member.

    Uses the "split" allocation: even-indexed members get no added features
    while odd-indexed members get the full ``k_max``, yielding a diverse mix
    of augmented and non-augmented views.
    """
    k_max = self._get_n_features_to_add(n_features_requested, n_cols)
    return [0 if i % 2 == 0 else k_max for i in range(self.n_estimators)]

  def fit(self, X: Any, y: Any) -> "EnsembleGenerator":
    """Fit the ensemble generator to training data.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Label array of shape (n_samples,).

    Returns:
      self.
    """
    validate_data(self, X, y)

    if self.norm_methods is None:
      self.norm_methods_ = ["none", "power"]
    elif isinstance(self.norm_methods, str):
      self.norm_methods_ = [self.norm_methods]
    else:
      self.norm_methods_ = list(self.norm_methods)

    self.unique_filter_ = UniqueFeatureFilter()
    X = self.unique_filter_.fit_transform(X)

    # Update categorical features indices after filtering
    if self.cat_features is not None:
      # Create mask of original features
      mask = np.zeros(self.unique_filter_.n_features_in_, dtype=bool)
      mask[self.cat_features] = True
      # Filter
      mask = mask[self.unique_filter_.features_to_keep_]
      self.cat_features_ = np.where(mask)[0]
      self.cat_values_ = {
          idx: np.unique(X[:, idx]) for idx in self.cat_features_
      }
    else:
      self.cat_features_ = np.array([], dtype=np.int64)
      self.cat_values_ = {}

    self.rng_ = random.Random(self.random_state)

    n_original_features = X.shape[1]
    self.n_original_features_ = n_original_features
    current_idx = n_original_features

    n_cols_expected = n_original_features
    if self.max_num_features is not None:
      n_cols_expected = min(n_cols_expected, self.max_num_features)

    self.k_crosses_list_ = self._get_member_n_features_list(
        self.n_feature_crosses, n_cols_expected
    )
    self.k_svd_list_ = self._get_member_n_features_list(
        self.n_svd_features, n_cols_expected
    )

    if sum(self.k_crosses_list_) > 0:
      num_features = [
          i for i in range(n_original_features) if i not in self.cat_features_
      ]
      if len(num_features) >= 2:
        max_possible_pairs = len(num_features) * (len(num_features) - 1) // 2
        pool_size = min(sum(self.k_crosses_list_), max_possible_pairs)

        if max_possible_pairs <= pool_size:
          self.cross_pairs_ = list(itertools.combinations(num_features, 2))
        else:
          # Combinatorial unranking to sample uniformly from combinations.
          # Uses combinadics representation to map an integer to a combination.
          # Idea: assume ordering (0,1); (0,2); (1,2); (0,3); ...
          # There are j(j-1)/2 elements before (0,j).
          # Need largest j such that j(j-1)/2 <= m; solve w/ quadratic formula.
          # Compute residual to get index i within the j-th block.
          sampled_indices = self.rng_.sample(
              range(max_possible_pairs), pool_size
          )
          self.cross_pairs_ = []
          for m in sampled_indices:
            j = (1 + math.isqrt(1 + 8 * m)) // 2
            i = m - j * (j - 1) // 2
            self.cross_pairs_.append((num_features[i], num_features[j]))

        X = _append_cross_features(X, self.cross_pairs_)

        self.cross_pool_start_ = current_idx
        self.cross_pool_end_ = X.shape[1]
        current_idx = X.shape[1]

    if sum(self.k_svd_list_) > 0:
      transformers = []
      if len(self.cat_features_) > 0:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            self.cat_features_,
        ))
      num_features = [
          i for i in range(n_original_features) if i not in self.cat_features_
      ]
      if num_features:
        transformers.append(("num", StandardScaler(), num_features))

      if transformers:
        preprocessor = ColumnTransformer(transformers)
        X_prep = preprocessor.fit_transform(X[:, :n_original_features])
        n_features_prep = X_prep.shape[1]
        n_samples = X.shape[0]
        max_possible_svd = min(n_samples, n_features_prep) - 1

        pool_size = self.total_svd_pool
        if pool_size is None:
          pool_size = sum(self.k_svd_list_)
        pool_size = min(pool_size, max_possible_svd)

        if pool_size > 0:
          self.svd_pipeline_ = Pipeline([
              ("prep", preprocessor),
              (
                  "svd",
                  TruncatedSVD(
                      n_components=pool_size, random_state=self.random_state
                  ),
              ),
          ])
          X = _append_svd_features(
              X, n_original_features, self.svd_pipeline_, is_train=True
          )

          self.svd_pool_start_ = current_idx
          self.svd_pool_end_ = X.shape[1]

    self.X_ = X
    self.y_ = y
    self.n_features_in_ = X.shape[1]
    if self.task == "classification":
      self.n_classes_ = len(np.unique(y))
    else:
      self.n_classes_ = 0

    # Generate and unpack all ensemble components
    (
        self.ensemble_configs_,
        self.feature_shuffle_patterns_,
        self.class_shift_offsets_,
        self.cat_permutations_,
        self.row_subsample_patterns_,
    ) = self._generate_ensemble()

    self.preprocessors_ = {}
    for norm_method in self.ensemble_configs_:
      if norm_method not in self.preprocessors_:
        preprocessor = PreprocessingPipeline(
            normalization_method=norm_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        preprocessor.fit(X)
        self.preprocessors_[norm_method] = preprocessor

    return self

  def _generate_ensemble(
      self,
  ) -> Tuple[
      collections.OrderedDict,
      collections.OrderedDict,
      collections.OrderedDict,
      collections.OrderedDict,
      collections.OrderedDict,
  ]:
    """Create diverse ensemble configurations grouped by normalization method.

    Returns:
      A 5-tuple of OrderedDicts:
        (ensemble_configs, feature_shuffle_patterns, class_shift_offsets,
         cat_permutations_grouped, row_subsample_patterns_grouped).
    """
    n_cols = self.n_original_features_
    if self.max_num_features is not None:
      n_cols = min(n_cols, self.max_num_features)

    any_crosses = any(k > 0 for k in self.k_crosses_list_) and hasattr(
        self, "cross_pairs_"
    )
    any_svd = any(k > 0 for k in self.k_svd_list_) and hasattr(
        self, "svd_pipeline_"
    )

    is_subsampling = n_cols < self.n_original_features_
    if any_crosses or any_svd or is_subsampling:
      shuffle_patterns = []
      for idx in range(self.n_estimators):
        # Subsample original features matching max_num_features
        cols = self.rng_.sample(range(self.n_original_features_), n_cols)

        k_cross = self.k_crosses_list_[idx]
        if k_cross > 0 and hasattr(self, "cross_pool_start_"):
          pool_start = self.cross_pool_start_
          pool_end = self.cross_pool_end_
          pool_size = pool_end - pool_start
          k = min(k_cross, pool_size)
          selected_crosses = self.rng_.sample(range(pool_start, pool_end), k)
          cols.extend(selected_crosses)

        k_svd = self.k_svd_list_[idx]
        if k_svd > 0 and hasattr(self, "svd_pool_start_"):
          pool_start = self.svd_pool_start_
          pool_end = self.svd_pool_end_
          pool_size = pool_end - pool_start
          k = min(k_svd, pool_size)
          selected_svd = self.rng_.sample(range(pool_start, pool_end), k)
          cols.extend(selected_svd)

        shuffle_pattern = np.array(self.rng_.sample(cols, len(cols)))
        shuffle_patterns.append(shuffle_pattern)
    else:
      shuffler = FeatureShuffler(
          n_features=self.n_features_in_,
          method=self.feat_shuffle_method,
          random_state=self.random_state,
      )
      shuffle_patterns = shuffler.shuffle(self.n_estimators)

      if len(shuffle_patterns) < self.n_estimators:
        num_cycles = (self.n_estimators + len(shuffle_patterns) - 1) // len(
            shuffle_patterns
        )
        shuffle_patterns = (shuffle_patterns * num_cycles)[: self.n_estimators]

    # 2. Generate Class Shifts
    if (
        self.task == "classification"
        and self.class_shift
        and self.n_estimators > 1
        and self.n_classes_ > 1
    ):
      base_offsets = self.rng_.sample(range(self.n_classes_), self.n_classes_)
      num_cycles = (
          self.n_estimators + len(base_offsets) - 1
      ) // len(base_offsets)
      shift_offsets = (base_offsets * num_cycles)[: self.n_estimators]
    else:
      shift_offsets = [0] * self.n_estimators

    # 3. Generate Categorical Permutations
    cat_permutations: List[Optional[Dict[int, Dict[Any, Any]]]] = []
    if self.permute_categorical and len(self.cat_features_) > 0:
      for _ in range(self.n_estimators):
        perm_dict = {
            col_idx: dict(
                zip(vals, self.rng_.sample(list(vals), len(vals)))
            )
            for col_idx, vals in self.cat_values_.items()
        }
        cat_permutations.append(perm_dict)
    else:
      cat_permutations = [None] * self.n_estimators

    # 4. Generate Row Subsample Patterns
    n_rows = self.X_.shape[0]
    if self.max_num_rows is not None:
      n_rows = min(n_rows, self.max_num_rows)

    if n_rows < self.X_.shape[0]:
      row_subsample_patterns = [
          np.array(self.rng_.sample(range(self.X_.shape[0]), n_rows))
          for _ in range(self.n_estimators)
      ]
    else:
      row_subsample_patterns = [None] * self.n_estimators

    # 5. Combine into configurations
    shuffle_shift_cat_configs = list(
        zip(
            shuffle_patterns,
            shift_offsets,
            cat_permutations,
            row_subsample_patterns,
        )
    )
    self.rng_.shuffle(shuffle_shift_cat_configs)

    # 6. Assign Normalization Methods
    num_cycles = (
        self.n_estimators + len(self.norm_methods_) - 1
    ) // len(self.norm_methods_)
    norm_methods_for_estimators = (self.norm_methods_ * num_cycles)[: self.n_estimators]
    full_configs = list(zip(norm_methods_for_estimators, shuffle_shift_cat_configs))

    # Group by normalization method and separate components
    ensemble_configs: collections.OrderedDict = collections.OrderedDict()
    feature_shuffle_patterns: collections.OrderedDict = collections.OrderedDict()
    class_shift_offsets_dict: collections.OrderedDict = collections.OrderedDict()
    cat_permutations_grouped: collections.OrderedDict = collections.OrderedDict()
    row_subsample_patterns_grouped: collections.OrderedDict = (
        collections.OrderedDict()
    )

    for norm_method in self.norm_methods_:
      configs = [config for norm, config in full_configs if norm == norm_method]
      if configs:
        ensemble_configs[norm_method] = configs
        feature_shuffle_patterns[norm_method] = [c[0] for c in configs]
        class_shift_offsets_dict[norm_method] = [c[1] for c in configs]
        cat_permutations_grouped[norm_method] = [c[2] for c in configs]
        row_subsample_patterns_grouped[norm_method] = [c[3] for c in configs]

    return (
        ensemble_configs,
        feature_shuffle_patterns,
        class_shift_offsets_dict,
        cat_permutations_grouped,
        row_subsample_patterns_grouped,
    )

  def transform(self, X: Any) -> collections.OrderedDict:
    """Generate ensemble data views for test samples.

    Args:
      X: Array-like of shape (n_test_samples, n_features).

    Returns:
      OrderedDict mapping each norm_method to a tuple
      ``(Xs, ys)`` where ``Xs`` has shape
      ``(n_configs, n_train + n_test, n_features)`` and ``ys`` has shape
      ``(n_configs, n_train)``.
    """
    check_is_fitted(self, ["ensemble_configs_"])
    X = self.unique_filter_.transform(X)

    if hasattr(self, "cross_pairs_") and self.cross_pairs_:
      X = _append_cross_features(X, self.cross_pairs_)

    if hasattr(self, "svd_pipeline_") and self.svd_pipeline_:
      X = _append_svd_features(
          X, self.n_original_features_, self.svd_pipeline_, is_train=False
      )

    data, _ = self._transform_features(X_test=X)
    return data

  def transform_fold(
      self, train_fold: np.ndarray, val_fold: np.ndarray
  ) -> Tuple[collections.OrderedDict, List[np.ndarray]]:
    """Generate ensemble data views for a fold-based split of training data.

    Args:
      train_fold: 1D array of training row indices (relative to row subsample).
      val_fold: 1D array of validation row indices (relative to row subsample).

    Returns:
      A tuple containing:
        - data: Ordered dictionary mapping normalization string keys to
          (X_ensemble, y_ensemble) arrays formatted identically to transform().
        - val_indices_list: List of absolute sample indices in the training set
          corresponding to validation query rows for each ensemble member.
          For example, if our row subsample for member i is [0, 2, 4, 6, 8] and
          our val_fold is [1, 3], then the ith element of val_indices_list will
          be [2, 6].
    """
    return self._transform_features(
        X_test=None, train_fold=train_fold, val_fold=val_fold
    )

  def _transform_features(
      self,
      X_test: Optional[np.ndarray],
      train_fold: Optional[np.ndarray] = None,
      val_fold: Optional[np.ndarray] = None,
  ) -> Tuple[collections.OrderedDict, List[np.ndarray]]:
    """Shared helper to construct transformed feature and target dictionaries.

    Handles feature formatting, categorical value permutations, scaling, and
    padding for both standard test inference (transform) and out-of-fold
    cross-validation (transform_fold).

    Args:
      X_test: 2D feature array of test queries of shape (n_samples, n_features).
        If None, the evaluation test queries are taken from self.X_[val_fold]
        during cross-validation.
      train_fold: Optional array of indices selecting in-context training rows
        during cross-validation. If None, all active training rows are used.
      val_fold: Optional array of indices selecting evaluation validation rows
        during cross-validation.

    Returns:
      A tuple containing:
        - data: Ordered dictionary mapping normalization keys to transformed
          feature and target batches.
        - val_indices_list: List of absolute validation row indices per config.
    """
    y = self.y_
    N = len(y)

    # Find max_features across all ensemble configs so that we can pad all
    # feature matrices to this width to feed a regular tensor into the model.
    max_features = 0
    for (
        norm_method,
        shuffle_shift_cat_configs,
    ) in self.ensemble_configs_.items():
      for shuffle_pattern, _, _, _ in shuffle_shift_cat_configs:
        max_features = max(max_features, len(shuffle_pattern))

    data: collections.OrderedDict = collections.OrderedDict()
    val_indices_list = []

    for (
        norm_method,
        shuffle_shift_cat_configs,
    ) in self.ensemble_configs_.items():
      preprocessor = self.preprocessors_[norm_method]
      X_ensemble = []
      y_ensemble = []

      for (
          shuffle_pattern,
          shift_offset,
          cat_perm,
          row_sub_pattern,
      ) in shuffle_shift_cat_configs:
        in_bag_idx = (
            row_sub_pattern if row_sub_pattern is not None else np.arange(N)
        )

        if train_fold is not None and val_fold is not None:
          train_idx = in_bag_idx[train_fold]
          val_idx = in_bag_idx[val_fold]
          val_indices_list.append(val_idx)
        else:
          train_idx = in_bag_idx
          val_idx = None

        y_to_use = y[train_idx]
        if cat_perm:
          # If we have categorical permutations, we must apply them before preprocessing
          # Note: self.X_ is the fitted training data. X_test is the test data.
          # We need to construct the full dataset (Train + Test)
          X_train_to_use = self.X_[train_idx]
          X_test_to_use = self.X_[val_idx] if val_idx is not None else X_test
          X_full = np.concatenate([X_train_to_use, X_test_to_use], axis=0)

          # Apply value permutation
          _apply_categorical_permutation(X_full, cat_perm)
          X_variant_instance = preprocessor.transform(X_full)
        else:
          X_train_trans = preprocessor.X_transformed_[train_idx]
          X_test_trans = (
              preprocessor.X_transformed_[val_idx]
              if val_idx is not None
              else preprocessor.transform(X_test)
          )
          X_variant_instance = np.concatenate(
              [X_train_trans, X_test_trans], axis=0
          )

        # Apply feature shuffling
        shuffled_cols = X_variant_instance[:, shuffle_pattern]
        shuffled_cols = _pad_features(shuffled_cols, max_features)
        X_ensemble.append(shuffled_cols)

        # Apply class shifting
        if self.task == "classification":
          y_ensemble.append((y_to_use + shift_offset) % self.n_classes_)
        else:
          y_ensemble.append(y_to_use)

      data[norm_method] = (
          np.stack(X_ensemble, axis=0),
          np.stack(y_ensemble, axis=0),
      )

    return data, val_indices_list

  def prepare_ensemble_tensors(
      self, data: collections.OrderedDict
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Prepare batched ensemble tensors and flat configs from transformed data.

    Args:
      data: Ordered dictionary mapping normalization string keys to (Xs, ys)
        sequence tensors (e.g., returned by transform() or transform_fold()).

    Returns:
      Tuple containing:
        - Xs_all: Concatenated feature array across all ensemble configs.
        - ys_all: Concatenated array of sequence target labels.
        - cat_masks_all: Stacked boolean masks indicating categorical indices.
        - ds_all: Array indicating sequence lengths per config.
        - configs_flat: Flattened list of ensemble config tuples.
    """
    max_features = max(Xs.shape[-1] for _, (Xs, _) in data.items())
    Xs_all, ys_all, cat_masks_all, ds_all = [], [], [], []
    configs_flat = []

    for norm_method, (Xs, ys) in data.items():
      Xs_all.append(Xs)
      ys_all.append(ys)
      configs = self.ensemble_configs_[norm_method]
      for config in configs:
        configs_flat.append(config)
        shuffle_pattern, _, _, _ = config
        mask = np.zeros(self.n_features_in_, dtype=np.bool_)
        if hasattr(self, "cat_features_"):
          mask[self.cat_features_] = True
        ds_all.append(len(shuffle_pattern))
        cat_mask = _pad_cat_mask(mask[shuffle_pattern], max_features)
        cat_masks_all.append(cat_mask)

    Xs_all = np.concatenate(Xs_all, axis=0)
    ys_all = np.concatenate(ys_all, axis=0)
    cat_masks_all = np.stack(cat_masks_all, axis=0)
    ds_all = np.array(ds_all, dtype=np.int32)
    return Xs_all, ys_all, cat_masks_all, ds_all, configs_flat

  def prepare_test_tensors(
      self, data: collections.OrderedDict
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Like prepare_ensemble_tensors(), but for test-only data (no labels).

    Used by the cache_context=True decode path (see transform_test_only()),
    where data maps norm_method -> (Xs, None) instead of (Xs, ys).

    Args:
      data: Ordered dictionary mapping normalization string keys to Xs
        tensors (e.g., returned by transform_test_only()).

    Returns:
      Tuple containing:
        - Xs_all: Concatenated feature array across all ensemble configs.
        - cat_masks_all: Stacked boolean masks indicating categorical indices.
        - ds_all: Array indicating sequence lengths per config.
        - configs_flat: Flattened list of ensemble config tuples.
    """
    max_features = max(Xs.shape[-1] for _, (Xs, _) in data.items())
    Xs_all, cat_masks_all, ds_all = [], [], []
    configs_flat = []

    for norm_method, (Xs, _) in data.items():
      Xs_all.append(Xs)
      configs = self.ensemble_configs_[norm_method]
      for config in configs:
        configs_flat.append(config)
        shuffle_pattern, _, _, _ = config
        mask = np.zeros(self.n_features_in_, dtype=np.bool_)
        if hasattr(self, "cat_features_"):
          mask[self.cat_features_] = True
        ds_all.append(len(shuffle_pattern))
        cat_mask = _pad_cat_mask(mask[shuffle_pattern], max_features)
        cat_masks_all.append(cat_mask)

    Xs_all = np.concatenate(Xs_all, axis=0)
    cat_masks_all = np.stack(cat_masks_all, axis=0)
    ds_all = np.array(ds_all, dtype=np.int32)
    return Xs_all, cat_masks_all, ds_all, configs_flat

  def transform_context_only(self) -> collections.OrderedDict:
    """Builds per-member context-only (training-row) feature/label tensors.

    Used to build a cache_context=True prefill cache at fit() time. This
    mirrors exactly the "training rows" half of what _transform_features()
    computes for the concatenated train+test tensor (same preprocessing,
    same categorical-permutation handling, same shuffle pattern and
    class-shift offset per config) but omits ever concatenating test rows.

    This is correct because every preprocessor.transform() step, including
    _apply_categorical_permutation(), is row-independent: each output row is
    computed from that row's values and fit-time statistics, not from other
    rows in the batch.

    Returns:
      OrderedDict mapping each norm_method to a tuple (Xs, ys) with the
      same shapes/semantics as transform()'s "training" portion.
    """
    check_is_fitted(self, ["ensemble_configs_"])
    y = self.y_
    N = len(y)

    max_features = 0
    for (
        norm_method,
        shuffle_shift_cat_configs,
    ) in self.ensemble_configs_.items():
      for shuffle_pattern, _, _, _ in shuffle_shift_cat_configs:
        max_features = max(max_features, len(shuffle_pattern))

    data: collections.OrderedDict = collections.OrderedDict()
    for (
        norm_method,
        shuffle_shift_cat_configs,
    ) in self.ensemble_configs_.items():
      preprocessor = self.preprocessors_[norm_method]
      X_ensemble = []
      y_ensemble = []

      for (
          shuffle_pattern,
          shift_offset,
          cat_perm,
          row_sub_pattern,
      ) in shuffle_shift_cat_configs:
        in_bag_idx = (
            row_sub_pattern if row_sub_pattern is not None else np.arange(N)
        )
        train_idx = in_bag_idx
        y_to_use = y[train_idx]

        if cat_perm:
          X_train_to_use = self.X_[train_idx].copy()
          _apply_categorical_permutation(X_train_to_use, cat_perm)
          X_variant_instance = preprocessor.transform(X_train_to_use)
        else:
          X_variant_instance = preprocessor.X_transformed_[train_idx]

        shuffled_cols = X_variant_instance[:, shuffle_pattern]
        shuffled_cols = _pad_features(shuffled_cols, max_features)
        X_ensemble.append(shuffled_cols)

        if self.task == "classification":
          y_ensemble.append((y_to_use + shift_offset) % self.n_classes_)
        else:
          y_ensemble.append(y_to_use)

      data[norm_method] = (
          np.stack(X_ensemble, axis=0),
          np.stack(y_ensemble, axis=0),
      )

    return data

  def transform_test_only(self, X: Any) -> collections.OrderedDict:
    """Builds per-member test-only feature tensors (no context rows).

    Used for the cache_context=True decode path at predict_proba() time.
    Mirrors exactly the "test rows" half of what _transform_features()
    computes for the concatenated train+test tensor (see
    transform_context_only() for why omitting the concatenation is safe).

    Args:
      X: Array-like of shape (n_test_samples, n_features).

    Returns:
      OrderedDict mapping each norm_method to a tuple (Xs, None) where
      Xs has shape (n_configs, n_test, n_features).
    """
    check_is_fitted(self, ["ensemble_configs_"])
    X = self.unique_filter_.transform(X)

    if hasattr(self, "cross_pairs_") and self.cross_pairs_:
      X = _append_cross_features(X, self.cross_pairs_)

    if hasattr(self, "svd_pipeline_") and self.svd_pipeline_:
      X = _append_svd_features(
          X, self.n_original_features_, self.svd_pipeline_, is_train=False
      )

    max_features = 0
    for (
        norm_method,
        shuffle_shift_cat_configs,
    ) in self.ensemble_configs_.items():
      for shuffle_pattern, _, _, _ in shuffle_shift_cat_configs:
        max_features = max(max_features, len(shuffle_pattern))

    data: collections.OrderedDict = collections.OrderedDict()
    for (
        norm_method,
        shuffle_shift_cat_configs,
    ) in self.ensemble_configs_.items():
      preprocessor = self.preprocessors_[norm_method]
      X_ensemble = []

      for (
          shuffle_pattern,
          _,
          cat_perm,
          _,
      ) in shuffle_shift_cat_configs:
        if cat_perm:
          X_test_to_use = X.copy()
          _apply_categorical_permutation(X_test_to_use, cat_perm)
          X_variant_instance = preprocessor.transform(X_test_to_use)
        else:
          X_variant_instance = preprocessor.transform(X)

        shuffled_cols = X_variant_instance[:, shuffle_pattern]
        shuffled_cols = _pad_features(shuffled_cols, max_features)
        X_ensemble.append(shuffled_cols)

      data[norm_method] = (np.stack(X_ensemble, axis=0), None)

    return data


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _apply_categorical_permutation(
    X_full: np.ndarray, cat_perm: Dict[int, Dict[Any, Any]]
) -> None:
  """Apply value permutations to categorical features in place.

  Args:
    X_full: 2D array of shape (n_samples, n_features) whose categorical columns
      will be permuted in place.
    cat_perm: Dictionary mapping categorical column indices to category value
      replacement mappings (e.g. ``{col_idx: {old_val: new_val}}``).
  """
  for col, mapping in cat_perm.items():
    col_vals = X_full[:, col]
    u_vals, inverse = np.unique(col_vals, return_inverse=True)
    mapped_u_vals = u_vals.copy()
    for i, val in enumerate(u_vals):
      if val in mapping:
        mapped_u_vals[i] = mapping[val]
    X_full[:, col] = mapped_u_vals[inverse]


def _append_cross_features(
    X: np.ndarray, cross_pairs: List[Tuple[int, int]]
) -> np.ndarray:
  """Append multiplicative feature crosses to feature matrix X along axis 1."""
  if not cross_pairs:
    return X
  new_cols = [X[:, i] * X[:, j] for i, j in cross_pairs]
  return np.concatenate([X, np.stack(new_cols, axis=1)], axis=1)


def _append_svd_features(
    X: np.ndarray,
    n_original_features: int,
    svd_pipeline: Optional[Pipeline],
    is_train: bool = False,
) -> np.ndarray:
  """Append TruncatedSVD components to feature matrix X along axis 1."""
  if not svd_pipeline:
    return X
  X_orig = X[:, :n_original_features]
  svd_feats = (
      svd_pipeline.fit_transform(X_orig)
      if is_train
      else svd_pipeline.transform(X_orig)
  )
  return np.concatenate([X, svd_feats], axis=1)


@jt.typed
def _pad_batch_to_multiple_of(
    x: Array | np.ndarray,
    divisor: int,
    constant_value: Union[int, float, np.number] = 0,
) -> Array | np.ndarray:
  """Pad axis 0 of array (at the end) to a multiple of ``divisor``.

  Args:
    x: Input array of any shape.
    divisor: Target multiple for axis 0.
    constant_value: Value to pad with. Defaults to 0.

  Returns:
    Array whose first dimension is the smallest multiple of ``divisor`` that
    is >= ``x.shape[0]``.  If ``divisor <= 1`` or no padding is required,
    the original array is returned unchanged.
  """
  if divisor <= 1:
    return x
  pad_size = (divisor - (x.shape[0] % divisor)) % divisor
  if pad_size == 0:
    return x
  pad_width = ((0, pad_size),) + ((0, 0),) * (x.ndim - 1)
  return np.pad(x, pad_width, constant_values=constant_value)


def _pad_features(X: np.ndarray, target_features: int) -> np.ndarray:
  """Pad a 2D feature matrix X along axis 1 (columns) to target_features with zeros."""
  if X.shape[1] < target_features:
    pad_cols = target_features - X.shape[1]
    return np.pad(X, ((0, 0), (0, pad_cols)), constant_values=0)
  return X


def _pad_cat_mask(cat_mask: np.ndarray, target_features: int) -> np.ndarray:
  """Pad a 1D categorical mask array to target_features with False entries."""
  if cat_mask.shape[0] < target_features:
    pad_cols = target_features - cat_mask.shape[0]
    return np.pad(cat_mask, (0, pad_cols), constant_values=False)
  return cat_mask


def _predict_step_pytorch(
    model: Any,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    train_size_val: int,
    ds_batch_val: Optional[np.ndarray],
    cat_mask_batch: Optional[np.ndarray],
) -> np.ndarray:
  """Runs PyTorch forward pass and returns numpy array."""
  if not HAS_TORCH:
    raise ImportError("PyTorch is required to run a PyTorch model.")

  device = next(model.parameters()).device

  X_t = torch.from_numpy(X_batch).to(device, dtype=torch.float32)
  y_t = torch.from_numpy(y_batch).to(device)
  if y_t.dtype == torch.float64:
    y_t = y_t.to(torch.float32)

  batch_size = X_batch.shape[0]
  train_size_t = torch.full(
      (batch_size,), train_size_val, dtype=torch.long, device=device
  )

  if ds_batch_val is not None:
    d_t = torch.from_numpy(ds_batch_val).to(device)
  else:
    d_t = torch.full(
        (batch_size,), X_batch.shape[-1], dtype=torch.long, device=device
    )

  cat_mask_t = (
      torch.from_numpy(cat_mask_batch).to(device)
      if cat_mask_batch is not None
      else None
  )

  with torch.no_grad():
    out_t = model(X_t, y_t, train_size_t, cat_mask=cat_mask_t, d=d_t)

  return out_t.float().cpu().numpy()  # upcast: numpy has no bfloat16


def _build_context_cache_pytorch(
    model: Any,
    ensemble_generator: "EnsembleGenerator",
    batch_size: Optional[int],
    maybe_quantize_kv_cache: bool = True,
    keep_cache_on_device: bool = True,
) -> Tuple[List[Any], int]:
  """Prefills each ensemble member's in-context (training) rows once.

  Shared by TabFMClassifier._build_context_cache() and
  TabFMRegressor._build_context_cache() (cache_context=True). Runs
  model.prefill() once per ensemble member, batched the same way
  _batch_forward() batches the uncached forward pass (via batch_size).

  Args:
    model: PyTorch TabFM nn.Module.
    ensemble_generator: Fitted EnsembleGenerator.
    batch_size: Number of ensemble members per model.prefill() call (None
      or 0 means all at once).
    maybe_quantize_kv_cache: If True (default) and the cache supports it
      (ICLearningCache.quantize()), int8-quantize the ICL K/V right after
      prefill to reduce the resident memory of holding every ensemble
      member's cache on-device at once.
    keep_cache_on_device: If True (default), keep each cache on the model's
      device. If False, move it to CPU right after building (traded for
      lower steady-state device memory at the cost of a transfer on every
      predict call; see _decode_context_cache_pytorch()).

  Returns:
    Tuple of (caches, n_members): caches is the list of per-batch cache
    dicts returned by model.prefill(), and n_members is the total number
    of ensemble members (used later to detect a stale cache).
  """
  data_ctx = ensemble_generator.transform_context_only()
  Xs_ctx, ys_ctx, cat_masks_ctx, ds_ctx, _ = (
      ensemble_generator.prepare_ensemble_tensors(data_ctx)
  )

  n_members = Xs_ctx.shape[0]
  batch_size_per_process = batch_size or n_members
  n_batches = math.ceil(n_members / batch_size_per_process)
  if n_batches > 1:
    Xs_split = np.array_split(Xs_ctx, n_batches)
    ys_split = np.array_split(ys_ctx, n_batches)
    cat_masks_split = np.array_split(cat_masks_ctx, n_batches)
    ds_split = np.array_split(ds_ctx, n_batches)
  else:
    Xs_split = [Xs_ctx]
    ys_split = [ys_ctx]
    cat_masks_split = [cat_masks_ctx]
    ds_split = [ds_ctx]

  device = next(model.parameters()).device
  caches = []
  with torch.no_grad():
    for X_batch, y_batch, cat_mask_batch, ds_batch in zip(
        Xs_split, ys_split, cat_masks_split, ds_split
    ):
      X_t = torch.from_numpy(X_batch).to(device, dtype=torch.float32)
      y_t = torch.from_numpy(y_batch).to(device)
      if y_t.dtype == torch.float64:
        y_t = y_t.to(torch.float32)
      cat_mask_t = torch.from_numpy(cat_mask_batch).to(device)
      d_t = torch.from_numpy(ds_batch).to(device)
      _, cache = model.prefill(X_t, y_t, cat_mask=cat_mask_t, d=d_t)
      icl_cache = cache.get("icl")
      if maybe_quantize_kv_cache and hasattr(icl_cache, "quantize"):
        cache["icl"] = icl_cache.quantize()
      if not keep_cache_on_device:
        cache = move_cache_to_device(cache, "cpu")
      caches.append(cache)

  return caches, n_members


def _decode_context_cache_pytorch(
    model: Any,
    ensemble_generator: "EnsembleGenerator",
    X_transformed: np.ndarray,
    context_caches: List[Any],
    expected_n_members: int,
    batch_size: Optional[int],
    keep_cache_on_device: bool = True,
) -> np.ndarray:
  """Decodes test rows against a context cache from _build_context_cache_pytorch().

  Shared by TabFMClassifier._predict_proba_internal_cached() and
  TabFMRegressor._predict_internal_cached(). Builds test-only feature
  tensors (via transform_test_only(), which does not concatenate the
  training context) and calls model.decode() per ensemble-member batch
  against its cached context, using the same batching as
  _build_context_cache_pytorch()/_batch_forward(). model.decode() dequantizes
  any int8-quantized K/V internally (see MultiheadAttention.forward()).

  Args:
    model: PyTorch TabFM nn.Module.
    ensemble_generator: Fitted EnsembleGenerator.
    X_transformed: Already-X_encoder_-transformed test features.
    context_caches: List of per-batch cache dicts from
      _build_context_cache_pytorch().
    expected_n_members: Total ensemble member count recorded at fit() time.
    batch_size: Number of ensemble members per model.decode() call.
    keep_cache_on_device: If False, each batch's cache was moved to CPU by
      _build_context_cache_pytorch(); move it back to the model's device for
      this call.

  Returns:
    np.ndarray of shape (n_members, n_test, output_dim).

  Raises:
    RuntimeError: If the ensemble member count changed since fit(), or the
      batch grouping does not match the cached batches.
  """
  data_test = ensemble_generator.transform_test_only(X_transformed)
  Xs_test, cat_masks_test, ds_test, _ = (
      ensemble_generator.prepare_test_tensors(data_test)
  )

  n_members = Xs_test.shape[0]
  if n_members != expected_n_members:
    raise RuntimeError(
        f"Number of ensemble members changed between fit() ({expected_n_members})"
        f" and predict() ({n_members}); the context cache is stale."
    )

  batch_size_per_process = batch_size or n_members
  n_batches = math.ceil(n_members / batch_size_per_process)
  if n_batches > 1:
    Xs_split = np.array_split(Xs_test, n_batches)
    cat_masks_split = np.array_split(cat_masks_test, n_batches)
    ds_split = np.array_split(ds_test, n_batches)
  else:
    Xs_split = [Xs_test]
    cat_masks_split = [cat_masks_test]
    ds_split = [ds_test]

  if len(Xs_split) != len(context_caches):
    raise RuntimeError(
        "Batch grouping mismatch between fit()-time context cache"
        f" ({len(context_caches)} groups) and predict()"
        f" ({len(Xs_split)} groups); this should not happen because both"
        " use the same total member count and batch_size."
    )

  device = next(model.parameters()).device
  outputs = []
  with torch.no_grad():
    for X_batch, cat_mask_batch, ds_batch, cache in zip(
        Xs_split, cat_masks_split, ds_split, context_caches
    ):
      X_t = torch.from_numpy(X_batch).to(device, dtype=torch.float32)
      cat_mask_t = torch.from_numpy(cat_mask_batch).to(device)
      d_t = torch.from_numpy(ds_batch).to(device)
      if not keep_cache_on_device:
        cache = move_cache_to_device(cache, device)
      out_t = model.decode(X_t, cache, cat_mask=cat_mask_t, d=d_t)
      outputs.append(out_t.float().cpu().numpy())
  return np.concatenate(outputs, axis=0)


# Compiled predict step functions memoized on the estimators by
# _batch_forward(). They close over nnx.jit state and cannot be pickled.
_COMPILED_PREDICT_CACHE_ATTRS = (
    "_predict_step_compiled_with_cat",
    "_predict_step_compiled_no_cat",
)
def _check_classifier_output_dim(output_dim: int, n_classes: int) -> None:
  """Validates that the model produces logits for all target classes.

  Args:
    output_dim: Size of the model output's last axis.
    n_classes: Number of classes seen during fit.

  Raises:
    ValueError: If the model outputs fewer values per row than there are
      classes, which usually means a regression checkpoint is being used for
      classification.
  """
  if output_dim < n_classes:
    raise ValueError(
        f"The model returned {output_dim} value(s) per row, but this"
        f" TabFMClassifier was fit on {n_classes} classes. This usually means"
        " a regression checkpoint is being used for classification. Load the"
        " classification weights instead, e.g."
        " `tabfm_v1_0_0.load(model_type='classification')`."
    )


def _check_regressor_output_dim(output_dim: int) -> None:
  """Validates that the model produces scalar regression outputs.

  Args:
    output_dim: Size of the model output's last axis.

  Raises:
    ValueError: If the model outputs more than one value per row, which
      usually means a classification checkpoint is being used for regression.
  """
  if output_dim != 1:
    raise ValueError(
        f"The model returned {output_dim} values per row, but TabFMRegressor"
        " expects a single regression output. This usually means a"
        " classification checkpoint is being used for regression; note that"
        " `tabfm_v1_0_0.load()` defaults to the classification weights. Load"
        " the regression weights instead with"
        " `tabfm_v1_0_0.load(model_type='regression')`."
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TabFMClassifier(ClassifierMixin, BaseEstimator):
  """TabFM (Tabular Foundation Model) classifier with scikit-learn interface.

  The TabFM model is a pre-trained foundation model that makes predictions via
  in-context learning: at inference time it is shown the training data as
  context and then asked to predict on test samples.  This class wraps the model
  with scikit-learn's ``ClassifierMixin`` / ``BaseEstimator`` interface,
  handling data pre-processing and ensemble aggregation automatically.

  Attributes:
    y_encoder_: Fitted ``CategoricalOrdinalEncoder`` for class labels.
    classes_: Array of original class labels.
    n_classes_: Number of unique classes.
    X_encoder_: Fitted ``TransformToNumerical`` for input features.
    ensemble_generator_: Fitted ``EnsembleGenerator``.
    active_calibration_method_: Resolved calibration method name ("platt" for
      binary, "vector" for multiclass, or None).
    ensemble_weights_: Blending weights for ensemble members computed via NNLS.
    calibration_lambda: L2 regularization strength for calibration parameter
      scaling.
  """

  y_encoder_: CategoricalOrdinalEncoder
  classes_: np.ndarray
  n_classes_: int
  X_encoder_: TransformToNumerical
  ensemble_generator_: EnsembleGenerator
  active_calibration_method_: Optional[str]
  ensemble_weights_: np.ndarray
  calibration_lambda: float

  def __init__(
      self,
      model: Any,
      n_estimators: int = 32,
      norm_methods: Optional[Union[str, List[str]]] = None,
      feat_shuffle_method: str = "random",
      class_shift: bool = True,
      permute_categorical: bool = False,
      outlier_threshold: float = 4.0,
      max_num_features: Optional[int] = 500,
      max_num_rows: Optional[int] = None,
      softmax_temperature: float = 0.9,
      average_logits: bool = True,
      use_amp: bool = True,
      batch_size: Optional[int] = 1,
      random_state: Optional[int] = _DEFAULT_RANDOM_STATE,
      verbose: bool = False,
      cat_encoder_mode: str = "appearance",
      binary_calibration_method: Optional[str] = None,
      multiclass_calibration_method: Optional[str] = None,
      num_folds_for_cv: int = 5,
      n_feature_crosses: Union[int, str] = 0,
      n_svd_features: Union[int, str] = 0,
      total_svd_pool: Optional[int] = None,
      enable_nnls: bool = False,
      nnls_beta: float = 0.75,
      calibration_lambda: float = 1e-2,
      min_rows_for_single_val_split: int = 2000,
      cache_context: bool = False,
      maybe_quantize_kv_cache: bool = True,
      keep_cache_on_device: bool = True,
  ):
    """Initialises the classifier.

    Args:
      model: Pre-trained TabFM model (NNX module).
      n_estimators: Number of ensemble members.
      norm_methods: Normalization method(s) for the ensemble. Defaults to
        ``["none", "power"]``.
      feat_shuffle_method: Feature-permutation strategy for the ensemble.
      class_shift: Whether to apply random class-label shifts.
      permute_categorical: Whether to randomly permute categorical values.
      outlier_threshold: Z-score threshold for outlier clipping.
      max_num_features: Maximum number of features to subsample per ensemble
        member.
      max_num_rows: Maximum number of rows to subsample per ensemble member.
      softmax_temperature: Temperature applied before the final softmax.
      average_logits: If True, average logits before applying softmax; otherwise
        average probabilities.
      use_amp: Whether to use automatic mixed precision (currently informational
        only).
      batch_size: Number of ensemble members to forward through the model at
        once.  ``None`` or 0 means all at once.
      random_state: Seed for ensemble randomness.
      verbose: Whether to print informational messages.
      cat_encoder_mode: Categorical encoding order (``"appearance"`` or
        ``"frequency"``).
      binary_calibration_method: Calibration method for binary problems
        (``None`` or ``"platt"``).
      multiclass_calibration_method: Calibration method for multiclass problems
        (``None`` or ``"vector"``).
      num_folds_for_cv: Number of folds for out-of-fold predictions.
      n_feature_crosses: ``"sqrt"`` to add sqrt(n_features) random feature
        crosses per ensemble member, or ``0`` to disable.
      n_svd_features: ``"sqrt"`` to add sqrt(n_features) random SVD features per
        ensemble member, or ``0`` to disable.
      total_svd_pool: Total pool size of SVD features to generate.
      enable_nnls: Whether to enable NNLS weighted ensemble.
      nnls_beta: Blending weight for NNLS.
      calibration_lambda: L2 regularization strength for calibration parameter
        scaling.
      min_rows_for_single_val_split: Minimum validation rows required to allow
        learning ensemble/calibration weights on a single train/val split
        instead of full CV. 0 means always doing full CV.
      cache_context: If True, encode the in-context (training) rows once per
        ensemble member at fit() time and reuse that cache (per-layer K/V
        and column-embedder induced-point reprs) on every predict_proba()
        call, instead of re-encoding the full context on every call.
        With maybe_quantize_kv_cache=False, the cached path is a pure speedup:
        results are numerically identical (within floating point) to
        cache_context=False. Only supported on the PyTorch backend; raises
        NotImplementedError for JAX models.
      maybe_quantize_kv_cache: If True (default) and cache_context=True,
        int8-quantize each ensemble member's cached ICL K/V right after
        fit() builds it (per-tensor symmetric quantization). This lets every
        member's cache stay resident on the GPU at once without exhausting
        device memory; it introduces small int8 rounding error, typically
        negligible for reported metrics. Ignored when cache_context=False.
      keep_cache_on_device: If True (default), keep every ensemble member's
        context cache on the model's device between predict_proba() calls. If
        False, caches are moved to CPU after fit() and transferred back to the
        device on every predict_proba() call, trading latency for lower
        steady-state device memory. Ignored when cache_context=False.
    """
    self.model = model
    self.n_estimators = n_estimators
    self.norm_methods = norm_methods
    self.feat_shuffle_method = feat_shuffle_method
    self.class_shift = class_shift
    self.permute_categorical = permute_categorical
    self.outlier_threshold = outlier_threshold
    self.max_num_features = max_num_features
    self.max_num_rows = max_num_rows
    self.softmax_temperature = softmax_temperature
    self.average_logits = average_logits
    self.use_amp = use_amp
    self.batch_size = batch_size
    self.random_state = random_state
    self.verbose = verbose
    self.cat_encoder_mode = cat_encoder_mode
    self.binary_calibration_method = binary_calibration_method
    self.multiclass_calibration_method = multiclass_calibration_method
    self.num_folds_for_cv = num_folds_for_cv
    self.n_feature_crosses = n_feature_crosses
    self.n_svd_features = n_svd_features
    self.total_svd_pool = total_svd_pool
    self.enable_nnls = enable_nnls
    self.nnls_beta = nnls_beta
    self.calibration_lambda = calibration_lambda
    self.min_rows_for_single_val_split = min_rows_for_single_val_split
    self.cache_context = cache_context
    self.maybe_quantize_kv_cache = maybe_quantize_kv_cache
    self.keep_cache_on_device = keep_cache_on_device
    if self.average_logits and self.enable_nnls:
      raise ValueError("average_logits and enable_nnls cannot both be True.")
    if self.max_num_rows is not None and self.enable_nnls:
      raise ValueError(
          "max_num_rows and enable_nnls cannot both be set at this time."
      )

  @classmethod
  def ensemble(cls, model: Any, **overrides: Any) -> "TabFMClassifier":
    """Constructs a classifier with the "ensemble" preset.

    Enables the heavier ensembling/calibration features on top of the default
    configuration: square-root feature-cross and SVD schedules, NNLS-weighted
    blending, probability (rather than logit) averaging, and per-problem
    calibration. Any keyword in ``overrides`` takes precedence over the preset.

    Args:
      model: Pre-trained TabFM model (NNX module).
      **overrides: Constructor arguments that override the preset.

    Returns:
      A configured ``TabFMClassifier`` instance.
    """
    params = dict(
        n_estimators=32,
        average_logits=False,
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        enable_nnls=True,
        binary_calibration_method="platt",
        multiclass_calibration_method="vector",
    )
    params.update(overrides)
    return cls(model, **params)

  def _more_tags(self):
    """Mark classifier as non-deterministic to bypass certain sklearn tests."""
    return dict(non_deterministic=True)

  def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.non_deterministic = True
    return tags

  def fit(self, X: Any, y: Any) -> "TabFMClassifier":
    """Fit the classifier to training data.

    Prepares the model for prediction by:
    1. Encoding class labels using LabelEncoder
    2. Converting input features to numerical values
    3. Fitting the ensemble generator to create transformed dataset views
    4. Loading the pre-trained TabFM model

    The model itself is not trained on the data; it uses in-context learning
    at inference time. This method only prepares the data transformations.

    Args:
      X: Training features of shape (n_samples, n_features).
      y: Training class labels of shape (n_samples,).

    Returns:
      self.

    Raises:
      ValueError
        If the number of classes exceeds the model's maximum supported classes.
    """
    check_classification_targets(y)

    # Encode class labels
    self.y_encoder_ = CategoricalOrdinalEncoder(
        dtype=np.int64, mode="alphabetical"
    )
    # Reshape for CategoricalOrdinalEncoder
    y_2d = y.reshape(-1, 1) if isinstance(y, np.ndarray) else np.array(y).reshape(-1, 1)
    y_encoded = self.y_encoder_.fit_transform(y_2d)
    y = y_encoded.flatten()

    # CategoricalOrdinalEncoder stores categories in a list of arrays
    self.classes_ = self.y_encoder_.categories_[0]
    self.n_classes_ = len(self.classes_)
    y_orig = y.copy()
    self.active_calibration_method_ = (
        self.binary_calibration_method
        if self.n_classes_ == 2
        else self.multiclass_calibration_method
    )

    if self.n_classes_ > self.model.max_classes:
      raise ValueError(
          f"The number of classes ({self.n_classes_}) exceeds the maximum number"
          f" of classes ({self.model.max_classes}) supported by the model."
      )

    #  Transform input features
    self.X_encoder_ = TransformToNumerical(
        verbose=self.verbose, cat_encoder_mode=self.cat_encoder_mode
    )
    X = self.X_encoder_.fit_transform(X)

    # Identify categorical feature indices (assumes OrdinalEncoder is the first transformer)
    # The ColumnTransformer in TransformToNumerical puts categorical features first.
    if hasattr(self.X_encoder_.tfm_, "transformers_"):
      n_cat = len(getattr(self.X_encoder_.tfm_, "transformers_")[0][2])
    else:
      n_cat = 0
    cat_features = list(range(n_cat))

    # Fit ensemble generator to create multiple dataset views
    self.ensemble_generator_ = EnsembleGenerator(
        n_estimators=self.n_estimators,
        norm_methods=self.norm_methods or ["none", "power"],
        feat_shuffle_method=self.feat_shuffle_method,
        class_shift=self.class_shift,
        cat_features=cat_features,
        permute_categorical=self.permute_categorical,
        outlier_threshold=self.outlier_threshold,
        max_num_features=self.max_num_features,
        max_num_rows=self.max_num_rows,
        random_state=self.random_state,
        n_feature_crosses=self.n_feature_crosses,
        n_svd_features=self.n_svd_features,
        total_svd_pool=self.total_svd_pool,
    )
    self.ensemble_generator_.fit(X, y)

    if self.cache_context:
      self._build_context_cache()

    oof_probs_fit = None
    if self.enable_nnls or (
        self.active_calibration_method_ is not None
        and self.active_calibration_method_ != "none"
    ):
      oof_probs = self.predict_oof_proba(cv=self.num_folds_for_cv)
      val_idx = getattr(self, "oof_val_indices_", None)
      if val_idx is not None:
        oof_probs_fit = oof_probs[:, val_idx, :]
        y_orig_fit = y_orig[val_idx]
        y_fit = y[val_idx]
      else:
        oof_probs_fit = oof_probs
        y_orig_fit = y_orig
        y_fit = y

    if self.enable_nnls and oof_probs_fit is not None:
      n_classes = self.n_classes_
      n_est, n_tr, _ = oof_probs_fit.shape

      # Convert y_orig_fit to one-hot targets
      y_one_hot = np.zeros((n_tr, n_classes))
      y_one_hot[np.arange(n_tr), y_orig_fit] = 1.0

      # Flatten along classification dimensions
      oof_flat = oof_probs_fit.reshape(n_est, n_tr * n_classes)

      y_one_hot_flat = y_one_hot.flatten()

      weights, _ = opt.nnls(oof_flat.T, y_one_hot_flat)
      sum_weights = np.sum(weights)
      if sum_weights > 0:
        weights = weights / sum_weights
      else:
        weights = np.ones(n_est) / n_est

      avg_weights = np.ones(n_est) / n_est
      self.ensemble_weights_ = (
          self.nnls_beta * weights + (1.0 - self.nnls_beta) * avg_weights
      )

    if (
        self.active_calibration_method_ is not None
        and self.active_calibration_method_ != "none"
        and oof_probs_fit is not None
    ):
      if self.enable_nnls:
        P = np.tensordot(self.ensemble_weights_, oof_probs_fit, axes=(0, 0))
      else:
        P = np.mean(oof_probs_fit, axis=0)
      assert P.shape == (len(y_fit), self.n_classes_), (
          f"Expected calibration input shape {(len(y_fit), self.n_classes_)},"
          f" got {P.shape}"
      )
      self._fit_calibration(P, y_fit)

    return self

  def _build_context_cache(self) -> None:
    """Prefills and caches each ensemble member's in-context (training) rows.

    Called from fit() when cache_context=True. Runs model.prefill()
    once per ensemble member (batched the same way _batch_forward() batches
    the uncached forward pass, via self.batch_size) and stores the
    resulting caches on self.context_caches_, alongside the per-member
    categorical masks / active-feature counts needed to later decode test
    rows against those caches (see _predict_proba_internal_cached()).

    Only supported on the PyTorch backend: the JAX backend's prefill()/
    decode() are not wired up here (this classifier's caching path assumes
    the PyTorch TabFM.prefill()/decode() cache/tensor contracts).

    Raises:
      NotImplementedError: If self.model is not a PyTorch nn.Module.
    """
    is_torch = HAS_TORCH and isinstance(self.model, torch.nn.Module)
    if not is_torch:
      raise NotImplementedError(
          "cache_context=True is only implemented for the PyTorch TabFM"
          " backend; the JAX backend's prefill()/decode() are not wired up"
          " here yet."
      )

    self.context_caches_, self._context_n_members_ = (
        _build_context_cache_pytorch(
            self.model, self.ensemble_generator_, self.batch_size,
            maybe_quantize_kv_cache=self.maybe_quantize_kv_cache,
            keep_cache_on_device=self.keep_cache_on_device,
        )
    )

  def _predict_proba_internal_cached(
      self, X_transformed: np.ndarray
  ) -> np.ndarray:
    """cache_context=True counterpart of _predict_proba_internal().

    Builds test-only feature tensors (via transform_test_only(), which does
    not concatenate the training context) and decodes each ensemble member's
    test chunk against its cached context, built once by
    _build_context_cache() at fit() time. It reuses the same batching,
    class-shift correction, and output-dim checks as the uncached path.

    Args:
      X_transformed: Already-X_encoder_-transformed test features.

    Returns:
      Array of shape (n_estimators_total, n_test_samples, n_classes) of raw
      (class-shift-corrected) logits, matching _predict_proba_internal()'s
      return contract.
    """
    if not hasattr(self, "context_caches_"):
      raise RuntimeError(
          "cache_context=True but no context cache was found; call fit()"
          " first."
      )

    outputs = _decode_context_cache_pytorch(
        self.model, self.ensemble_generator_, X_transformed,
        self.context_caches_, self._context_n_members_, self.batch_size,
        keep_cache_on_device=self.keep_cache_on_device,
    )

    _check_classifier_output_dim(outputs.shape[-1], self.n_classes_)
    outputs = outputs[..., :self.n_classes_]

    class_shift_offsets = []
    for offsets in self.ensemble_generator_.class_shift_offsets_.values():
      class_shift_offsets.extend(offsets)

    all_logits = []
    for i, offset in enumerate(class_shift_offsets):
      out = outputs[i]
      out = np.concatenate([out[..., offset:], out[..., :offset]], axis=-1)
      all_logits.append(out)

    return np.stack(all_logits, axis=0)

  @jt.typed
  def _batch_forward(
      self,
      Xs: jt.Float[Array | np.ndarray, "B T H"],
      ys: jt.Shaped[Array | np.ndarray, "B T_train"],
      cat_masks: Optional[jt.Bool[Array | np.ndarray, "B H"]] = None,
      ds: Optional[jt.Int[Array | np.ndarray, "B"]] = None,
  ) -> jt.Float[Array | np.ndarray, "B T_test K"]:
    """Process model forward passes in batches to manage memory efficiently.

    This method handles the batched inference through the TabFM model,
    dividing the ensemble members into smaller batches to avoid out-of-memory
    errors.

    Parameters
    ----------
    Xs : np.ndarray
        Input features of shape (n_datasets, n_samples, n_features), where
        n_datasets
        is the number of ensemble members.

    ys : np.ndarray
        Training labels of shape (n_datasets, train_size), where train_size is
        the
        number of samples used for in-context learning.

    cat_masks : np.ndarray or None, optional
        Boolean mask of shape (n_datasets, n_features) indicating which features
        are categorical (True) vs. numerical (False). The model uses this to
        apply feature-type-specific processing: for example, categorical
        features
        may use different Fourier frequencies or random embeddings compared to
        numerical features. If None, all features are treated as numerical.

    ds : np.ndarray or None, optional
        Array of shape (n_datasets,) indicating the number of active features
        per
        ensemble member, ignoring padding features.

    Returns
    -------
    np.ndarray
        Model outputs (logits or probabilities) of shape (n_datasets, test_size,
        n_classes)
        where test_size = n_samples - train_size.
    """
    is_torch = HAS_TORCH and isinstance(self.model, torch.nn.Module)

    if is_torch:
      # --- PyTorch execution path ---
      batch_size_per_process = self.batch_size or Xs.shape[0]
      n_batches = math.ceil(Xs.shape[0] / batch_size_per_process)
      if n_batches > 1:
        Xs_split = np.array_split(Xs, n_batches)
        ys_split = np.array_split(ys, n_batches)
        cat_masks_split = (
            np.array_split(cat_masks, n_batches)
            if cat_masks is not None
            else [None] * n_batches
        )
        ds_split = (
            np.array_split(ds, n_batches) if ds is not None else [None] * n_batches
        )
      else:
        Xs_split = [Xs]
        ys_split = [ys]
        cat_masks_split = [cat_masks]
        ds_split = [ds]

      outputs = []
      for X_batch, y_batch, cat_mask_batch, ds_batch_val in zip(
          Xs_split, ys_split, cat_masks_split, ds_split
      ):
        orig_batch_size = X_batch.shape[0]
        orig_seq_len = X_batch.shape[1]
        train_size_val = y_batch.shape[1]

        # Pad y to match X length along sequence dimension if needed
        if y_batch.shape[1] < X_batch.shape[1]:
          y_batch = np.pad(
              y_batch,
              ((0, 0), (0, X_batch.shape[1] - y_batch.shape[1])),
              mode="constant",
              constant_values=-100.0,
          )

        out = _predict_step_pytorch(
            self.model,
            X_batch,
            y_batch,
            train_size_val,
            ds_batch_val,
            cat_mask_batch,
        )
        # Slice output to keep only test predictions and unpadded batch.
        out = out[:orig_batch_size, train_size_val:orig_seq_len, :]
        outputs.append(out)
      return np.concatenate(outputs, axis=0)

    else:
      if not HAS_JAX:
        raise ImportError("JAX is required to run a JAX model.")
      # --- JAX execution path ---
      mesh = jax.sharding.get_mesh()
      if mesh and "data" in mesh.axis_names:
        num_data_shards = mesh.axis_sizes[mesh.axis_names.index("data")]
        data_sharding = NamedSharding(mesh, PartitionSpec("data"))
      else:
        num_data_shards = 1
        data_sharding = None

      num_classes = self.n_classes_

      _has_compiled_attr = (
          "_predict_step_compiled_with_cat"
          if (cat_masks is not None and hasattr(self.model, "cell_embedder"))
          else "_predict_step_compiled_no_cat"
      )

      if not hasattr(self, _has_compiled_attr):
        if cat_masks is not None and hasattr(self.model, "cell_embedder"):

          @nnx.jit(
              in_shardings=(
                  None,
                  data_sharding,
                  data_sharding,
                  data_sharding,
                  data_sharding,
                  data_sharding,
              ),
              out_shardings=data_sharding,
          )
          def _predict_step_fn(model, X, y, train_size, d, cat_mask):
            return model(
                X,
                y,
                train_size=train_size,
                d=d,
                cat_mask=cat_mask,
                num_classes=num_classes,
            )

        else:

          @nnx.jit(
              in_shardings=(
                  None,
                  data_sharding,
                  data_sharding,
                  data_sharding,
                  data_sharding,
              ),
              out_shardings=data_sharding,
          )
          def _predict_step_fn(model, X, y, train_size, d):
            return model(
                X,
                y,
                train_size=train_size,
                d=d,
                num_classes=num_classes,
            )

        setattr(self, _has_compiled_attr, _predict_step_fn)

      _predict_step_compiled = getattr(self, _has_compiled_attr)

      batch_size_per_process = self.batch_size or Xs.shape[0]
      n_batches = math.ceil(Xs.shape[0] / batch_size_per_process)
      if n_batches > 1:
        Xs_split = np.array_split(Xs, n_batches)
        ys_split = np.array_split(ys, n_batches)
        if cat_masks is not None:
          cat_masks_split = np.array_split(cat_masks, n_batches)
        if ds is not None:
          ds_split = np.array_split(ds, n_batches)
      else:
        Xs_split = [Xs]
        ys_split = [ys]
        if cat_masks is not None:
          cat_masks_split = [cat_masks]
        if ds is not None:
          ds_split = [ds]

      outputs = []
      cat_masks_iter = (
          cat_masks_split if cat_masks is not None else [None] * len(Xs_split)
      )
      ds_iter = ds_split if ds is not None else [None] * len(Xs_split)
      for X_batch, y_batch, cat_mask_batch, ds_batch_val in zip(
          Xs_split, ys_split, cat_masks_iter, ds_iter
      ):
        orig_batch_size = X_batch.shape[0]
        orig_seq_len = X_batch.shape[1]

        # Follow prefill(): pad sequence length T (n_row) to a multiple of 128
        # with -100. Padded rows fall past train_size and are sliced off below.
        _block_size = 128
        _T_full = X_batch.shape[1]
        _pad_len = ((_T_full - 1) // _block_size + 1) * _block_size - _T_full
        if _pad_len > 0:
          X_batch = np.pad(
              X_batch, ((0, 0), (0, _pad_len), (0, 0)), constant_values=-100.0
          )

        X_batch = _pad_batch_to_multiple_of(X_batch, num_data_shards)
        y_batch = _pad_batch_to_multiple_of(y_batch, num_data_shards)

        X_batch = jax.device_put(
            jnp.array(X_batch, dtype=jnp.float32), data_sharding
        )
        y_batch = jax.device_put(
            jnp.array(y_batch, dtype=jnp.float32), data_sharding
        )
        batch_size_padded = X_batch.shape[0]
        train_size_val = y_batch.shape[1]
        train_size = jax.device_put(
            jnp.repeat(train_size_val, batch_size_padded), data_sharding
        )

        if ds_batch_val is not None:
          ds_batch = _pad_batch_to_multiple_of(
              ds_batch_val, num_data_shards, constant_value=X_batch.shape[-1]
          )
        else:
          ds_batch = np.full(
              (batch_size_padded,), X_batch.shape[-1], dtype=np.int32
          )

        d_batch = jax.device_put(
            jnp.array(ds_batch, dtype=np.int32),
            data_sharding,
        )

        # Pad y to match X length along sequence dimension
        if y_batch.shape[1] < X_batch.shape[1]:
          y_batch = jnp.pad(
              y_batch,
              ((0, 0), (0, X_batch.shape[1] - y_batch.shape[1])),
              constant_values=-100.0,
          )

        if jax.process_index() == 0:
          logging.info("X_batch shape: %s", X_batch.shape)
          logging.info("y_batch shape: %s", y_batch.shape)
          logging.info("train_size: %s", train_size_val)

        # No gradient calculation needed for inference
        if cat_mask_batch is not None and hasattr(self.model, "cell_embedder"):
          cat_mask_batch = _pad_batch_to_multiple_of(
              cat_mask_batch, num_data_shards
          )
          cat_mask_batch = jax.device_put(
              jnp.array(cat_mask_batch, dtype=jnp.bool_), data_sharding
          )
          out = _predict_step_compiled(
              self.model, X_batch, y_batch, train_size, d_batch, cat_mask_batch
          )
        else:
          out = _predict_step_compiled(
              self.model, X_batch, y_batch, train_size, d_batch
          )

        # Slice output to keep only test predictions and unpadded batch.
        out = out[:orig_batch_size, train_size_val:orig_seq_len, :]
        out = multihost_utils.process_allgather(out, tiled=True)
        outputs.append(out)

      return np.concatenate(outputs, axis=0)

  def __getstate__(self):
    """Drops memoized compiled predict functions from the pickled state.

    The first predict memoizes nnx.jit-compiled step functions on the
    estimator (see _batch_forward()). Those closures cannot be pickled; they
    are pure caches and are rebuilt lazily on the next predict.
    """
    state = dict(super().__getstate__())
    for attr in _COMPILED_PREDICT_CACHE_ATTRS:
      state.pop(attr, None)
    return state

  @jt.typed
  def predict_oof_proba(self, cv: int = 5) -> jt.Float[Array | np.ndarray, "E N K"]:
    """Perform out-of-fold predictions on the training set for each ensemble member."""
    check_is_fitted(self)

    N = self.ensemble_generator_.X_.shape[0]
    n_classes = self.n_classes_

    all_configs = []
    for (
        norm_method,
        configs,
    ) in self.ensemble_generator_.ensemble_configs_.items():
      for config in configs:
        all_configs.append((norm_method, config))
    n_estimators = len(all_configs)

    kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
    # Take the number of rows from the first config if max_num_rows is given,
    # or use N otherwise.
    _, _, _, first_row_sub_pattern = all_configs[0][1]
    n_rows = (
        len(first_row_sub_pattern) if first_row_sub_pattern is not None else N
    )
    folds_base = list(kf.split(np.arange(n_rows)))

    if (
        getattr(self, "min_rows_for_single_val_split", 0) > 0
        and len(folds_base[0][1]) >= self.min_rows_for_single_val_split
    ):
      folds_to_run = folds_base[:1]
    else:
      folds_to_run = folds_base

    outputs_oof = np.zeros((n_estimators, N, n_classes))

    self.oof_val_indices_ = None
    for fold_idx, (train_fold, val_fold) in enumerate(folds_to_run):
      data_fold, val_indices_list = self.ensemble_generator_.transform_fold(
          train_fold, val_fold
      )
      if fold_idx == 0 and len(folds_to_run) == 1:
        self.oof_val_indices_ = val_indices_list[0]
      (
          Xs_batch,
          ys_batch,
          cat_masks_batch,
          ds_batch,
          configs_flat,
      ) = self.ensemble_generator_.prepare_ensemble_tensors(data_fold)

      out = self._batch_forward(
          Xs_batch, ys_batch, cat_masks_batch, ds=ds_batch
      )
      _check_classifier_output_dim(out.shape[-1], n_classes)
      out = out[..., :n_classes]

      for i, (_, shift_offset, _, _) in enumerate(configs_flat):
        out_i = out[i]
        out_i = np.concatenate(
            [out_i[..., shift_offset:], out_i[..., :shift_offset]], axis=-1
        )
        out_i = self.softmax(
            out_i, axis=-1, temperature=self.softmax_temperature
        )
        outputs_oof[i, val_indices_list[i]] = out_i

    return outputs_oof

  @jt.typed
  def _fit_calibration(
      self,
      P: jt.Float[Array | np.ndarray, "N K"],
      y: jt.Int[Array | np.ndarray, "N"],
  ):
    """Fit calibration model on out-of-fold predictions.

    Args:
      P: Out-of-fold probability predictions of shape (N, K), where N is the
        number of training samples and K is the number of classes.
      y: Ground truth targets of shape (N,). Note that y is not one-hot encoded,
        but contains the actual class numeric/integer labels.
    """
    K = self.n_classes_
    N = P.shape[0]
    eps = 1e-15

    if self.verbose:
      print(
          f"Fitting calibration model ({self.active_calibration_method_}) with"
          f" {K} classes..."
      )

    if K == 2:
      if self.active_calibration_method_ == "platt":
        z = np.log((P[:, 1] + eps) / (P[:, 0] + eps))

        def loss(params):
          A, B = params
          p1 = scipy.special.expit(A * z + B)
          p = np.column_stack([1 - p1, p1])
          nll = -np.sum(np.log(p[np.arange(N), y] + eps)) / N
          reg = self.calibration_lambda * ((A - 1.0) ** 2 + B**2)
          return nll + reg

        res = opt.minimize(
            loss,
            np.array([1.0, 0.0]),
            bounds=[(0.8, 1.2), (-1.0, 1.0)],
            method="L-BFGS-B",
        )
        self.calibration_params_ = {"A": res.x[0], "B": res.x[1]}
      else:
        raise ValueError(
            "Unknown binary_calibration_method:"
            f" {self.active_calibration_method_!r}. Expected None or 'platt'."
        )

    else:
      if self.active_calibration_method_ == "vector":
        z = np.log(P + eps)

        def loss(params):
          w = params[:K]
          b = params[K:]
          z_prime = w * z + b
          p = scipy.special.softmax(z_prime, axis=1)
          nll = -np.sum(np.log(p[np.arange(N), y] + eps)) / N
          reg = self.calibration_lambda * (
              np.sum((w - 1.0) ** 2) + np.sum(b**2)
          )
          return nll + reg

        init_params = np.concatenate([np.ones(K), np.zeros(K)])
        res = opt.minimize(
            loss,
            init_params,
            bounds=[(0.8, 1.2)] * K + [(-1.0, 1.0)] * K,
            method="L-BFGS-B",
        )
        self.calibration_params_ = {"w": res.x[:K], "b": res.x[K:]}
      else:
        raise ValueError(
            "Unknown multiclass_calibration_method:"
            f" {self.active_calibration_method_!r}. Expected None or 'vector'."
        )

  @jt.typed
  def _apply_calibration(
      self, P: jt.Float[Array | np.ndarray, "N K"]
  ) -> jt.Float[Array | np.ndarray, "N K"]:
    """Apply calibration model to predictions.

    Args:
      P: Uncalibrated predicted probabilities of shape (N, K), where N is the
        number of samples and K is the number of classes.

    Returns:
      Calibrated predicted probabilities of shape (N, K).
    """
    if not hasattr(self, "calibration_params_") or not self.calibration_params_:
      return P

    K = self.n_classes_
    eps = 1e-15

    if self.active_calibration_method_ == "platt" and K == 2:
      A = self.calibration_params_["A"]
      B = self.calibration_params_["B"]
      z = np.log((P[:, 1] + eps) / (P[:, 0] + eps))
      p1 = scipy.special.expit(A * z + B)
      return np.column_stack([1 - p1, p1])

    elif self.active_calibration_method_ == "vector":
      w = self.calibration_params_["w"]
      b = self.calibration_params_["b"]
      z = np.log(P + eps)
      z_prime = w * z + b
      return scipy.special.softmax(z_prime, axis=1)

    return P

  @jt.typed
  def _predict_proba_internal(self, X: Any) -> jt.Float[Array | np.ndarray, "E T K"]:
    """Predict class probabilities for test samples."""
    check_is_fitted(self)
    if isinstance(X, np.ndarray) and len(X.shape) == 1:
      raise ValueError(
          "The provided input X is one-dimensional. Reshape your data."
      )

    X_transformed = self.X_encoder_.transform(X)

    if getattr(self, "cache_context", False):
      return self._predict_proba_internal_cached(X_transformed)

    data = self.ensemble_generator_.transform(X_transformed)
    (
        Xs_all,
        ys_all,
        cat_masks_all,
        ds_all,
        _,
    ) = self.ensemble_generator_.prepare_ensemble_tensors(data)

    outputs = self._batch_forward(Xs_all, ys_all, cat_masks_all, ds=ds_all)
    _check_classifier_output_dim(outputs.shape[-1], self.n_classes_)
    outputs = outputs[..., :self.n_classes_]

    # Extract class shift offsets from ensemble generator
    class_shift_offsets = []
    for offsets in self.ensemble_generator_.class_shift_offsets_.values():
      class_shift_offsets.extend(offsets)

    # Correct for class shifts and return raw logits for all estimators
    all_logits = []
    for i, offset in enumerate(class_shift_offsets):
      out = outputs[i]
      out = np.concatenate([out[..., offset:], out[..., :offset]], axis=-1)
      all_logits.append(out)

    return np.stack(all_logits, axis=0)

  def _process_logits(self, logits_all: np.ndarray):
    probs_all = self.softmax(
        logits_all, axis=-1, temperature=self.softmax_temperature
    )

    if self.enable_nnls:
      probs = np.tensordot(self.ensemble_weights_, probs_all, axes=(0, 0))
    elif self.average_logits:
      avg_logits = np.mean(logits_all, axis=0)
      probs = self.softmax(
          avg_logits, axis=-1, temperature=self.softmax_temperature
      )
    else:
      probs = np.mean(probs_all, axis=0)

    if self.active_calibration_method_ is not None:
      probs = self._apply_calibration(probs)

    return probs

  @jt.typed
  def predict_proba(self, X: Any) -> jt.Float[Array | np.ndarray, "T K"]:
    """Predict class probabilities for test samples.

    Applies the ensemble of TabFM models to make predictions, with each
    ensemble
    member providing predictions that are then averaged. The method:
    1. Transforms input data using the fitted encoders
    2. Applies the ensemble generator to create multiple views
    3. Forwards each view through the model
    4. Corrects for class shifts
    5. Averages predictions across ensemble members

    Args:
      X: array-like of shape (n_samples, n_features). Test samples for
        prediction.

    Returns:
      np.ndarray of shape (n_samples, n_classes)
        Class probabilities for each test sample.
    """
    check_is_fitted(self)

    logits = self._predict_proba_internal(X)
    return self._process_logits(logits)

  @jt.typed
  def predict(self, X: Any) -> np.ndarray:
    """Predict class labels for test samples.

    Uses predict_proba to get class probabilities and returns the class with
    the highest probability for each sample.

    Args:
      X : array-like of shape (n_samples, n_features)
          Test samples for prediction.

    Returns:
      array-like of shape (n_samples,)
        Predicted class labels for each test sample.
    """
    proba = self.predict_proba(X)
    y = np.argmax(proba[:, : self.n_classes_], axis=1)
    y_2d = y.reshape(-1, 1)
    y_decoded = self.y_encoder_.inverse_transform(y_2d)
    return y_decoded.flatten().astype(self.classes_.dtype)

  @jt.typed
  @staticmethod
  def softmax(
      x: np.ndarray, axis: int = -1, temperature: float = 0.9
  ) -> np.ndarray:
    """Compute temperature-scaled softmax.

    Args:
      x: Input logit array of any shape.
      axis: Axis along which to compute softmax.
      temperature: Scaling factor applied before the softmax; values < 1
        produce a sharper distribution.

    Returns:
      Softmax probabilities with the same shape as ``x``.
    """
    x = x / temperature
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    # Compute softmax
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


class TabFMRegressor(RegressorMixin, BaseEstimator):
  """TabFM (Tabular Foundation Model) regressor with scikit-learn interface.

  The pre-trained TabFM model is used for in-context regression.  Target
  values are standardized before being passed to the model and are
  inverse-transformed on output.

  Attributes:
    X_encoder_: Fitted ``TransformToNumerical`` for input features.
    y_scaler_: Fitted ``StandardScaler`` for target standardization.
    ensemble_generator_: Fitted ``EnsembleGenerator``.
    y_oof_scaled_: Out-of-fold scaled target predictions for the training set.
    ensemble_weights_: Blending weights for ensemble members computed via NNLS.
  """

  X_encoder_: TransformToNumerical
  y_scaler_: StandardScaler
  ensemble_generator_: EnsembleGenerator
  y_oof_scaled_: np.ndarray
  ensemble_weights_: np.ndarray

  def __init__(
      self,
      model: Any,
      n_estimators: int = 32,
      norm_methods: Optional[Union[str, List[str]]] = None,
      feat_shuffle_method: str = "random",
      permute_categorical: bool = False,
      outlier_threshold: float = 4.0,
      max_num_features: Optional[int] = 500,
      max_num_rows: Optional[int] = None,
      use_amp: bool = True,
      batch_size: Optional[int] = 1,
      random_state: Optional[int] = _DEFAULT_RANDOM_STATE,
      verbose: bool = False,
      cat_encoder_mode: str = "appearance",
      num_folds_for_cv: int = 5,
      n_feature_crosses: Union[int, str] = 0,
      n_svd_features: Union[int, str] = 0,
      total_svd_pool: Optional[int] = None,
      enable_nnls: bool = False,
      nnls_beta: float = 0.75,
      min_rows_for_single_val_split: int = 2000,
      cache_context: bool = False,
      maybe_quantize_kv_cache: bool = True,
      keep_cache_on_device: bool = True,
  ):
    """Initialises the regressor.

    Args:
      model: Pre-trained TabFM model (NNX module).
      n_estimators: Number of ensemble members.
      norm_methods: Normalization method(s) for the ensemble.  Defaults to
        ``["none", "power"]``.
      feat_shuffle_method: Feature-permutation strategy for the ensemble.
      permute_categorical: Whether to randomly permute categorical values.
      outlier_threshold: Z-score threshold for outlier clipping.
      max_num_features: Maximum number of features to subsample per ensemble
        member.
      max_num_rows: Maximum number of rows to subsample per ensemble member.
      use_amp: Whether to use automatic mixed precision (informational only).
      batch_size: Number of ensemble members to forward at once.  ``None`` or 0
        means all at once.
      random_state: Seed for ensemble randomness.
      verbose: Whether to print informational messages.
      cat_encoder_mode: Categorical encoding order (``"appearance"`` or
        ``"frequency"``).
      num_folds_for_cv: Number of folds for out-of-fold predictions.
      n_feature_crosses: ``"sqrt"`` to add sqrt(n_features) random feature
        crosses per ensemble member, or ``0`` to disable.
      n_svd_features: ``"sqrt"`` to add sqrt(n_features) random SVD features per
        ensemble member, or ``0`` to disable.
      total_svd_pool: Total pool size of SVD features to generate.
      enable_nnls: Whether to enable NNLS weighted ensemble.
      nnls_beta: Blending weight for NNLS.
      min_rows_for_single_val_split: Minimum validation rows required to allow
        learning ensemble weights on a single train/val split instead of full
        CV. 0 means always doing full CV.
      cache_context: If True, encode the in-context (training) rows once per
        ensemble member at fit() time and reuse that cache (per-layer K/V
        and column-embedder induced-point reprs) on every predict() call,
        instead of re-encoding the full context on every call.
        With maybe_quantize_kv_cache=False, the cached path is a pure speedup:
        results are numerically identical (within floating point) to
        cache_context=False. Only supported on the PyTorch backend; raises
        NotImplementedError for JAX models.
      maybe_quantize_kv_cache: If True (default) and cache_context=True,
        int8-quantize each ensemble member's cached ICL K/V right after
        fit() builds it (per-tensor symmetric quantization). This lets every
        member's cache stay resident on the GPU at once without exhausting
        device memory; it introduces small int8 rounding error, typically
        negligible for reported metrics. Ignored when cache_context=False.
      keep_cache_on_device: If True (default), keep every ensemble member's
        context cache on the model's device between predict() calls. If False,
        caches are moved to CPU after fit() and transferred back to the device
        on every predict() call, trading latency for lower steady-state
        device memory. Ignored when cache_context=False.
    """
    self.model = model
    self.n_estimators = n_estimators
    self.norm_methods = norm_methods
    self.feat_shuffle_method = feat_shuffle_method
    self.permute_categorical = permute_categorical
    self.outlier_threshold = outlier_threshold
    self.max_num_features = max_num_features
    self.max_num_rows = max_num_rows
    self.use_amp = use_amp
    self.batch_size = batch_size
    self.random_state = random_state
    self.verbose = verbose
    self.cat_encoder_mode = cat_encoder_mode
    self.num_folds_for_cv = num_folds_for_cv
    self.n_feature_crosses = n_feature_crosses
    self.n_svd_features = n_svd_features
    self.total_svd_pool = total_svd_pool
    self.enable_nnls = enable_nnls
    self.nnls_beta = nnls_beta
    self.min_rows_for_single_val_split = min_rows_for_single_val_split
    self.cache_context = cache_context
    self.maybe_quantize_kv_cache = maybe_quantize_kv_cache
    self.keep_cache_on_device = keep_cache_on_device
    if self.max_num_rows is not None and self.enable_nnls:
      raise ValueError(
          "max_num_rows and enable_nnls cannot both be set at this time."
      )

  @classmethod
  def ensemble(cls, model: Any, **overrides: Any) -> "TabFMRegressor":
    """Constructs a regressor with the "ensemble" preset.

    Enables the heavier ensembling features on top of the default configuration:
    square-root feature-cross and SVD schedules and NNLS-weighted blending. Any
    keyword in ``overrides`` takes precedence over the preset.

    Args:
      model: Pre-trained TabFM model (NNX module).
      **overrides: Constructor arguments that override the preset.

    Returns:
      A configured ``TabFMRegressor`` instance.
    """
    params = dict(
        n_estimators=32,
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        enable_nnls=True,
    )
    params.update(overrides)
    return cls(model, **params)

  def _more_tags(self):
    """Mark regressor as non-deterministic to bypass certain sklearn tests."""
    return dict(non_deterministic=True)

  def fit(self, X: Any, y: Any) -> "TabFMRegressor":
    """Fit the regressor to training data.

    Prepares the model for prediction by:
    1. Converting input features to numerical values
    2. Fitting the ensemble generator to create transformed dataset views

    The model itself is not trained on the data; it uses in-context learning
    at inference time. This method only prepares the data transformations.

    Args:
      X : array-like of shape (n_samples, n_features)
          Training input data.

      y : array-like of shape (n_samples,)
          Training target values.

    Returns:
      self : TabFMRegressor
          Fitted regressor instance.
    """
    y_orig = np.array(y).copy()
    y = check_array(y, ensure_2d=False, dtype="numeric")
    self.X_encoder_ = TransformToNumerical(
        verbose=self.verbose, cat_encoder_mode=self.cat_encoder_mode
    )
    X = self.X_encoder_.fit_transform(X)

    if hasattr(self.X_encoder_.tfm_, "transformers_"):
      n_cat = len(getattr(self.X_encoder_.tfm_, "transformers_")[0][2])
    else:
      n_cat = 0
    cat_features = list(range(n_cat))

    self.y_scaler_ = StandardScaler()
    # Keep ndarray output even if the user set a global
    # sklearn.set_config(transform_output="pandas"): otherwise fit_transform
    # returns a DataFrame, which has no .flatten().
    self.y_scaler_.set_output(transform="default")
    y = self.y_scaler_.fit_transform(y.reshape(-1, 1)).flatten()

    self.ensemble_generator_ = EnsembleGenerator(
        n_estimators=self.n_estimators,
        norm_methods=self.norm_methods or ["none", "power"],
        feat_shuffle_method=self.feat_shuffle_method,
        class_shift=False,
        cat_features=cat_features,
        permute_categorical=self.permute_categorical,
        outlier_threshold=self.outlier_threshold,
        max_num_features=self.max_num_features,
        max_num_rows=self.max_num_rows,
        random_state=self.random_state,
        task="regression",
        n_feature_crosses=self.n_feature_crosses,
        n_svd_features=self.n_svd_features,
        total_svd_pool=self.total_svd_pool,
    )
    self.ensemble_generator_.fit(X, y)

    if self.cache_context:
      self._build_context_cache()

    if self.enable_nnls:
      self.y_oof_scaled_ = self._compute_oof_preds_scaled(
          cv=self.num_folds_for_cv
      )
      val_idx = getattr(self, "oof_val_indices_", None)
      y_oof_scaled_fit = (
          self.y_oof_scaled_[:, val_idx]
          if val_idx is not None
          else self.y_oof_scaled_
      )
      y_orig_fit = y_orig[val_idx] if val_idx is not None else y_orig
      n_est, n_tr = y_oof_scaled_fit.shape
      y_oof = np.zeros((n_est, n_tr))
      for i in range(n_est):
        y_oof[i, :] = self._inverse_transform_y(y_oof_scaled_fit[i])

      weights, _ = opt.nnls(y_oof.T, y_orig_fit)
      sum_weights = np.sum(weights)
      if sum_weights > 0:
        weights = weights / sum_weights
      else:
        weights = np.ones(n_est) / n_est

      avg_weights = np.ones(n_est) / n_est
      self.ensemble_weights_ = (
          self.nnls_beta * weights + (1.0 - self.nnls_beta) * avg_weights
      )

    return self

  def _build_context_cache(self) -> None:
    """Prefills and caches each ensemble member's in-context (training) rows.

    Called from fit() when cache_context=True. Mirrors
    TabFMClassifier._build_context_cache() via the shared
    _build_context_cache_pytorch() helper: runs model.prefill() once
    per ensemble member (batched the same way _batch_forward() batches the
    uncached forward pass, via self.batch_size) and stores the resulting
    caches on self.context_caches_ for later decoding of test rows (see
    _predict_internal_cached()).

    Only supported on the PyTorch backend: the JAX backend's prefill()/
    decode() are not wired up here (this regressor's caching path assumes
    the PyTorch TabFM.prefill()/decode() cache/tensor contracts).

    Raises:
      NotImplementedError: If self.model is not a PyTorch nn.Module.
    """
    is_torch = HAS_TORCH and isinstance(self.model, torch.nn.Module)
    if not is_torch:
      raise NotImplementedError(
          "cache_context=True is only implemented for the PyTorch TabFM"
          " backend; the JAX backend's prefill()/decode() are not wired up"
          " here yet."
      )

    self.context_caches_, self._context_n_members_ = (
        _build_context_cache_pytorch(
            self.model, self.ensemble_generator_, self.batch_size,
            maybe_quantize_kv_cache=self.maybe_quantize_kv_cache,
            keep_cache_on_device=self.keep_cache_on_device,
        )
    )

  def _predict_internal_cached(
      self, X_transformed: np.ndarray
  ) -> np.ndarray:
    """cache_context=True counterpart of _predict_internal().

    Decodes each ensemble member's test chunk against its cached context
    (built once by _build_context_cache() at fit() time) via the shared
    _decode_context_cache_pytorch() helper, then squeezes the trailing
    singleton output dimension exactly as _predict_internal() does.

    Args:
      X_transformed: Already-X_encoder_-transformed test features.

    Returns:
      Array of shape (n_estimators_total, n_test_samples) of scaled
      predictions, matching _predict_internal()'s return contract.
    """
    if not hasattr(self, "context_caches_"):
      raise RuntimeError(
          "cache_context=True but no context cache was found; call fit()"
          " first."
      )

    outputs = _decode_context_cache_pytorch(
        self.model, self.ensemble_generator_, X_transformed,
        self.context_caches_, self._context_n_members_, self.batch_size,
        keep_cache_on_device=self.keep_cache_on_device,
    )
    _check_regressor_output_dim(outputs.shape[-1])
    return outputs.squeeze(-1)

  @jt.typed
  def _batch_forward(
      self,
      Xs: jt.Float[Array | np.ndarray, "B T H"],
      ys: jt.Shaped[Array | np.ndarray, "B T_train"],
      cat_masks: Optional[jt.Bool[Array | np.ndarray, "B H"]] = None,
      ds: Optional[jt.Int[Array | np.ndarray, "B"]] = None,
  ) -> jt.Float[Array | np.ndarray, "B T_test L_out"]:
    """Process model forward passes in batches to manage memory efficiently.

    Args:
      Xs: Input features of shape (n_datasets, n_samples, n_features).
      ys: Training labels of shape (n_datasets, train_size).
      cat_masks: Optional boolean mask of shape (n_datasets, n_features)
        indicating which features are categorical (True) vs. numerical (False).
        The model uses this to apply feature-type-specific processing: for
        example, categorical features may use different Fourier frequencies or
        random embeddings compared to numerical features. If None, all features
        are treated as numerical.
      ds: Optional array of shape (n_datasets,) indicating the number of active
        features per ensemble member, ignoring padding features.

    Returns:
      Model outputs of shape (n_datasets, n_test, output_dim).
    """
    is_torch = HAS_TORCH and isinstance(self.model, torch.nn.Module)

    if is_torch:
      # --- PyTorch execution path ---
      batch_size_per_process = getattr(self, "batch_size", 1) or Xs.shape[0]
      n_batches = math.ceil(Xs.shape[0] / batch_size_per_process)
      if n_batches > 1:
        Xs_split = np.array_split(Xs, n_batches)
        ys_split = np.array_split(ys, n_batches)
        cat_masks_split = (
            np.array_split(cat_masks, n_batches)
            if cat_masks is not None
            else [None] * n_batches
        )
        ds_split = (
            np.array_split(ds, n_batches) if ds is not None else [None] * n_batches
        )
      else:
        Xs_split = [Xs]
        ys_split = [ys]
        cat_masks_split = [cat_masks]
        ds_split = [ds]

      outputs = []
      for X_batch, y_batch, cat_mask_batch, ds_batch_val in zip(
          Xs_split, ys_split, cat_masks_split, ds_split
      ):
        orig_batch_size = X_batch.shape[0]
        orig_seq_len = X_batch.shape[1]
        train_size_val = y_batch.shape[1]

        # Pad y to match X length along sequence dimension if needed
        if y_batch.shape[1] < X_batch.shape[1]:
          y_batch = np.pad(
              y_batch,
              ((0, 0), (0, X_batch.shape[1] - y_batch.shape[1])),
              mode="constant",
              constant_values=-100.0,
          )

        out = _predict_step_pytorch(
            self.model,
            X_batch,
            y_batch,
            train_size_val,
            ds_batch_val,
            cat_mask_batch,
        )
        # Slice output to keep only test predictions and unpadded batch.
        out = out[:orig_batch_size, train_size_val:orig_seq_len, :]
        outputs.append(out)
      return np.concatenate(outputs, axis=0)

    else:
      if not HAS_JAX:
        raise ImportError("JAX is required to run a JAX model.")
      # --- JAX execution path ---
      mesh = jax.sharding.get_mesh()
      if mesh and "data" in mesh.axis_names:
        num_data_shards = mesh.axis_sizes[mesh.axis_names.index("data")]
        data_sharding = NamedSharding(mesh, PartitionSpec("data"))
      else:
        num_data_shards = 1
        data_sharding = None

      _has_compiled_attr = (
          "_predict_step_compiled_with_cat"
          if (cat_masks is not None and hasattr(self.model, "cell_embedder"))
          else "_predict_step_compiled_no_cat"
      )

      if not hasattr(self, _has_compiled_attr):
        if cat_masks is not None and hasattr(self.model, "cell_embedder"):

          @nnx.jit(
              in_shardings=(
                  None,
                  data_sharding,
                  data_sharding,
                  data_sharding,
                  data_sharding,
                  data_sharding,
              ),
              out_shardings=data_sharding,
          )
          def _predict_step_fn(model, X, y, train_size, d, cat_mask):
            return model(
                X,
                y,
                train_size=train_size,
                d=d,
                cat_mask=cat_mask,
            )

        else:

          @nnx.jit(
              in_shardings=(
                  None,
                  data_sharding,
                  data_sharding,
                  data_sharding,
                  data_sharding,
              ),
              out_shardings=data_sharding,
          )
          def _predict_step_fn(model, X, y, train_size, d):
            return model(
                X,
                y,
                train_size=train_size,
                d=d,
            )

        setattr(self, _has_compiled_attr, _predict_step_fn)

      _predict_step_compiled = getattr(self, _has_compiled_attr)
      batch_size_per_process = getattr(self, "batch_size", 1) or Xs.shape[0]
      n_batches = math.ceil(Xs.shape[0] / batch_size_per_process)
      if n_batches > 1:
        Xs_split = np.array_split(Xs, n_batches)
        ys_split = np.array_split(ys, n_batches)
        if cat_masks is not None:
          cat_masks_split = np.array_split(cat_masks, n_batches)
        if ds is not None:
          ds_split = np.array_split(ds, n_batches)
      else:
        Xs_split, ys_split = [Xs], [ys]
        if cat_masks is not None:
          cat_masks_split = [cat_masks]
        if ds is not None:
          ds_split = [ds]

      outputs = []
      cat_masks_iter = (
          cat_masks_split if cat_masks is not None else [None] * len(Xs_split)
      )
      ds_iter = ds_split if ds is not None else [None] * len(Xs_split)
      for X_batch, y_batch, cat_mask_batch, ds_batch_val in zip(
          Xs_split, ys_split, cat_masks_iter, ds_iter
      ):
        orig_batch_size = X_batch.shape[0]
        orig_seq_len = X_batch.shape[1]

        # Follow prefill(): pad sequence length T (n_row) to a multiple of 128
        # with -100. Padded rows fall past train_size and are sliced off below.
        _block_size = 128
        _T_full = X_batch.shape[1]
        _pad_len = ((_T_full - 1) // _block_size + 1) * _block_size - _T_full
        if _pad_len > 0:
          X_batch = np.pad(
              X_batch, ((0, 0), (0, _pad_len), (0, 0)), constant_values=-100.0
          )

        X_batch = _pad_batch_to_multiple_of(X_batch, num_data_shards)
        y_batch = _pad_batch_to_multiple_of(y_batch, num_data_shards)

        X_batch = jax.device_put(
            jnp.array(X_batch, dtype=jnp.float32), data_sharding
        )
        y_batch = jax.device_put(
            jnp.array(y_batch, dtype=jnp.float32), data_sharding
        )
        batch_size_padded = X_batch.shape[0]
        train_size_val = y_batch.shape[1]
        train_size = jax.device_put(
            jnp.repeat(train_size_val, batch_size_padded), data_sharding
        )

        if ds_batch_val is not None:
          ds_batch = _pad_batch_to_multiple_of(
              ds_batch_val, num_data_shards, constant_value=X_batch.shape[-1]
          )
        else:
          ds_batch = np.full(
              (batch_size_padded,), X_batch.shape[-1], dtype=np.int32
          )

        d_batch = jax.device_put(
            jnp.array(ds_batch, dtype=jnp.int32), data_sharding
        )

        if y_batch.shape[1] < X_batch.shape[1]:
          y_batch = jnp.pad(
              y_batch,
              ((0, 0), (0, X_batch.shape[1] - y_batch.shape[1])),
              constant_values=-100.0,
          )

        if jax.process_index() == 0:
          logging.info("X_batch shape: %s", X_batch.shape)
          logging.info("y_batch shape: %s", y_batch.shape)
          logging.info("train_size: %s", train_size_val)

        if cat_mask_batch is not None and hasattr(self.model, "cell_embedder"):
          cat_mask_batch = _pad_batch_to_multiple_of(
              cat_mask_batch, num_data_shards
          )
          cat_mask_batch = jax.device_put(
              jnp.array(cat_mask_batch, dtype=jnp.bool_), data_sharding
          )
          out = _predict_step_compiled(
              self.model, X_batch, y_batch, train_size, d_batch, cat_mask_batch
          )
        else:
          out = _predict_step_compiled(
              self.model, X_batch, y_batch, train_size, d_batch
          )

        out = out[:orig_batch_size, train_size_val:orig_seq_len, :]
        out = multihost_utils.process_allgather(out, tiled=True)
        outputs.append(out)

      return np.concatenate(outputs, axis=0)

  def _inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
    """Inverse transform target values."""
    return self.y_scaler_.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

  def __getstate__(self):
    """Drops memoized compiled predict functions from the pickled state.

    The first predict memoizes nnx.jit-compiled step functions on the
    estimator (see _batch_forward()). Those closures cannot be pickled; they
    are pure caches and are rebuilt lazily on the next predict.
    """
    state = dict(super().__getstate__())
    for attr in _COMPILED_PREDICT_CACHE_ATTRS:
      state.pop(attr, None)
    return state

  @jt.typed
  def predict_oof(self, cv: int = 5) -> jt.Float[Array | np.ndarray, "E N"]:
    """Perform out-of-fold predictions on the training set for each ensemble member."""
    check_is_fitted(self)

    outputs_oof_scaled = self._compute_oof_preds_scaled(cv=cv)
    n_estimators, N = outputs_oof_scaled.shape

    outputs_oof = np.zeros((n_estimators, N))
    for i in range(n_estimators):
      outputs_oof[i, :] = self._inverse_transform_y(outputs_oof_scaled[i])

    return outputs_oof

  def _compute_oof_preds_scaled(
      self,
      cv: int = 5,
  ) -> np.ndarray:
    """Helper to compute OOF predictions in the scaled space."""
    check_is_fitted(self)

    N = self.ensemble_generator_.X_.shape[0]

    all_configs = []
    for (
        norm_method,
        configs,
    ) in self.ensemble_generator_.ensemble_configs_.items():
      for config in configs:
        all_configs.append((norm_method, config))
    n_estimators = len(all_configs)

    kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
    # Take the number of rows from the first config if max_num_rows is given,
    # or use N otherwise.
    _, _, _, first_row_sub_pattern = all_configs[0][1]
    n_rows = (
        len(first_row_sub_pattern) if first_row_sub_pattern is not None else N
    )
    folds_base = list(kf.split(np.arange(n_rows)))

    outputs_oof = np.zeros((n_estimators, N))

    if (
        getattr(self, "min_rows_for_single_val_split", 0) > 0
        and len(folds_base[0][1]) >= self.min_rows_for_single_val_split
    ):
      folds_to_run = folds_base[:1]
    else:
      folds_to_run = folds_base

    self.oof_val_indices_ = None
    for fold_idx, (train_fold, val_fold) in enumerate(folds_to_run):
      data_fold, val_indices_list = self.ensemble_generator_.transform_fold(
          train_fold, val_fold
      )
      if fold_idx == 0 and len(folds_to_run) == 1:
        self.oof_val_indices_ = val_indices_list[0]
      (
          Xs_batch,
          ys_batch,
          cat_masks_batch,
          ds_batch,
          _,
      ) = self.ensemble_generator_.prepare_ensemble_tensors(data_fold)

      out = self._batch_forward(
          Xs_batch, ys_batch, cat_masks_batch, ds=ds_batch
      )
      _check_regressor_output_dim(out.shape[-1])

      preds = out.squeeze(-1)

      for i in range(n_estimators):
        outputs_oof[i, val_indices_list[i]] = preds[i]

    return outputs_oof

  @jt.typed
  def _predict_internal(self, X: Any) -> jt.Float[Array | np.ndarray, "E T"]:
    """Predict regression target for test samples."""
    check_is_fitted(self)
    if isinstance(X, np.ndarray) and len(X.shape) == 1:
      raise ValueError("The provided input X is one-dimensional. Reshape your data.")

    X_transformed = self.X_encoder_.transform(X)

    if getattr(self, "cache_context", False):
      return self._predict_internal_cached(X_transformed)

    data = self.ensemble_generator_.transform(X_transformed)
    (
        Xs_all,
        ys_all,
        cat_masks_all,
        ds_all,
        _,
    ) = self.ensemble_generator_.prepare_ensemble_tensors(data)

    output = self._batch_forward(Xs_all, ys_all, cat_masks_all, ds=ds_all)
    _check_regressor_output_dim(output.shape[-1])
    predictions = output.squeeze(-1)
    return predictions

  @jt.typed
  def _combine_predictions(
      self, predictions_scaled: jt.Float[Array | np.ndarray, "E T"]
  ) -> jt.Float[Array | np.ndarray, "T"]:
    """Combine scaled predictions from ensemble members into final prediction."""
    n_est = predictions_scaled.shape[0]
    if self.enable_nnls:
      test_preds = np.zeros((n_est, predictions_scaled.shape[1]))
      for i in range(n_est):
        test_preds[i, :] = self._inverse_transform_y(predictions_scaled[i])
      return np.dot(self.ensemble_weights_, test_preds)
    else:
      avg_predictions = np.mean(predictions_scaled, axis=0)
      return self._inverse_transform_y(avg_predictions)

  @jt.typed
  def predict(self, X: Any) -> jt.Float[Array | np.ndarray, "T"]:
    """Predict regression target for test samples.

    Applies the ensemble of TabFM models to make predictions, with each
    ensemble member providing predictions that are then averaged.

    Args:
      X: array-like of shape (n_samples, n_features). Test samples for
        prediction.

    Returns:
      np.ndarray of shape (n_samples,)
          Predicted target values for each test sample.
    """
    predictions = self._predict_internal(X)
    return self._combine_predictions(predictions)

```

## File: tabfm/src/classifier_and_regressor_multidevice_test.py

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

"""Regression tests for predict on a host with more than one visible device.

These run on simulated CPU devices, so they exercise the multi-device forward
path in CI without needing GPUs. The device count must be requested before JAX
initializes, which is why this lives in its own module.
"""

import os

# Request two host devices before JAX initializes so jax.devices() returns more
# than one. Kept in its own test module so it does not affect other tests.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from absl.testing import absltest
import numpy as np

try:
  from flax import nnx
  from tabfm.src.jax import model as tabfm_model

  HAS_JAX = True
except ImportError:
  HAS_JAX = False

from tabfm.src.classifier_and_regressor import TabFMClassifier
from tabfm.src.classifier_and_regressor import TabFMRegressor

# pylint: disable=invalid-name


def _tiny_model(loss):
  return tabfm_model.TabFM(
      loss=loss,
      max_classes=2,
      embed_dim=8,
      col_num_blocks=1,
      col_nhead=2,
      col_num_inds=8,
      row_num_blocks=1,
      row_nhead=2,
      row_num_cls=1,
      icl_num_blocks=1,
      icl_nhead=2,
      rngs=nnx.Rngs(0),
  )


class MultiDevicePredictTest(absltest.TestCase):
  """predict must work when more than one device is visible and no mesh is set.

  Without a user-configured mesh the default batch is 1, but the JAX forward
  path built a data sharding over every visible device. That forced a batch of
  size 1 into an N-way shard (IndivisibleError) and placed the inputs on all
  devices while the weights stayed on device 0 (device mismatch). The forward
  path now keeps the sharding it derives from the active mesh (none by default,
  so the call is replicated), which is correct for one or many devices.
  """

  def setUp(self):
    super().setUp()
    if not HAS_JAX:
      self.skipTest("JAX is required for this test.")
    import jax  # pylint: disable=g-import-not-at-top

    if len(jax.devices()) < 2:
      self.skipTest("Requires at least two (simulated) devices.")

  def test_classifier_predict_multi_device(self):
    clf = TabFMClassifier(model=_tiny_model("cross_entropy"), n_estimators=2)
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    y = rng.randint(0, 2, size=20)
    clf.fit(X, y)
    preds = clf.predict(X[:5])
    self.assertEqual(np.asarray(preds).shape, (5,))

  def test_regressor_predict_multi_device(self):
    reg = TabFMRegressor(model=_tiny_model("mse"), n_estimators=2)
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    y = rng.rand(20)
    reg.fit(X, y)
    preds = reg.predict(X[:5])
    self.assertEqual(np.asarray(preds).shape, (5,))


if __name__ == "__main__":
  absltest.main()

```

## File: tabfm/src/classifier_and_regressor_pytorch_test.py

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

import pickle
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import torch

from tabfm.src.pytorch import model as pytorch_model
from tabfm.src.classifier_and_regressor import TabFMClassifier, TabFMRegressor


class PyTorchClassifierRegressorTest(unittest.TestCase):

  def test_classifier_fit_predict(self):
    np.random.seed(42)
    # Instantiate small PyTorch model
    model = pytorch_model.TabFM(
        embed_dim=8,
        max_classes=3,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=2,
        is_classifier=True
    )
    
    clf = TabFMClassifier(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42
    )

    X = np.random.rand(10, 3)
    y = np.random.randint(0, 3, size=10)

    clf.fit(X, y)
    
    # Predict
    preds = clf.predict(X)
    self.assertEqual(preds.shape, (10,))
    self.assertTrue(np.all(preds >= 0) and np.all(preds < 3))

    # Predict proba
    probs = clf.predict_proba(X)
    self.assertEqual(probs.shape, (10, 3))
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)

    # cache_context=True should be a pure speedup: predict_proba should match
    # the uncached path exactly. maybe_quantize_kv_cache=False isolates this
    # from the (lossy by design) int8 KV-cache quantization, which is tested
    # separately in model_test.py.
    cached_clf = TabFMClassifier(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42,
        cache_context=True,
        maybe_quantize_kv_cache=False,
    )
    cached_clf.fit(X, y)
    probs_cached = cached_clf.predict_proba(X)
    self.assertEqual(probs_cached.shape, (10, 3))
    np.testing.assert_allclose(np.sum(probs_cached, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(probs_cached, probs, rtol=1e-5, atol=1e-6)

  def test_regressor_fit_predict(self):
    np.random.seed(42)
    # Instantiate small PyTorch model
    model = pytorch_model.TabFM(
        embed_dim=8,
        max_classes=1, # Regressor might ignore this or use 1
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=2,
        is_classifier=False
    )
    
    reg = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    reg.fit(X, y)

    # Predict
    preds = reg.predict(X)
    self.assertEqual(preds.shape, (10,))

    # cache_context=True should be a pure speedup: predict should match the
    # uncached path exactly. maybe_quantize_kv_cache=False isolates this from
    # the (lossy by design) int8 KV-cache quantization, which is tested
    # separately in model_test.py.
    cached_reg = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42,
        cache_context=True,
        maybe_quantize_kv_cache=False,
    )
    cached_reg.fit(X, y)
    preds_cached = cached_reg.predict(X)
    self.assertEqual(preds_cached.shape, (10,))
    np.testing.assert_allclose(preds_cached, preds, rtol=1e-5, atol=1e-6)


class PyTorchModelPickleTest(unittest.TestCase):
  """The PyTorch model must be picklable.

  AutoGluon / TabArena save the fitted estimator (which holds the model) with
  stdlib pickle. The encode/decode heads are gelu ``MLP``s whose activation was
  previously a lambda from ``get_activation`` -- unpicklable -- so this guards
  against that regression. The regression path builds three such heads
  (y_embedder, y_encoder, decoder), so it exercises the activation most.
  """

  def _tiny_model(self, is_classifier):
    return pytorch_model.TabFM(
        embed_dim=8,
        max_classes=3,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=2,
        is_classifier=is_classifier,
    )

  def test_classifier_model_pickle_round_trip(self):
    restored = pickle.loads(pickle.dumps(self._tiny_model(is_classifier=True)))
    self.assertIsInstance(restored, pytorch_model.TabFM)

  def test_regressor_model_pickle_round_trip(self):
    restored = pickle.loads(pickle.dumps(self._tiny_model(is_classifier=False)))
    self.assertIsInstance(restored, pytorch_model.TabFM)

  def test_pickle_preserves_forward_output(self):
    # The unpickled model must still run and produce identical outputs: the
    # promoted _gelu_tanh is numerically identical to the previous lambda.
    torch.manual_seed(0)
    model = self._tiny_model(is_classifier=True).eval()
    x = torch.randn(2, 4, 6)  # [B, T, H]
    y = torch.randint(0, 3, (2, 4))  # [B, T]
    train_size = torch.tensor([2, 3])  # [B]
    with torch.no_grad():
      before = model(x, y, train_size)
      restored = pickle.loads(pickle.dumps(model)).eval()
      after = restored(x, y, train_size)
    torch.testing.assert_close(before, after)


if __name__ == "__main__":
  unittest.main()

```

## File: tabfm/src/classifier_and_regressor_test.py

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

import pickle
import unittest
from unittest import mock
from absl.testing import absltest
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.exceptions import NotFittedError

try:
  from flax import nnx
  from tabfm.src.jax import model as tabfm_model
  HAS_JAX = True
except ImportError:
  HAS_JAX = False
from tabfm.src.classifier_and_regressor import _looks_like_datetime
from tabfm.src.classifier_and_regressor import EnsembleGenerator
from tabfm.src.classifier_and_regressor import TabFMClassifier
from tabfm.src.classifier_and_regressor import TabFMRegressor
from tabfm.src.classifier_and_regressor import TransformToNumerical

# pylint: disable=invalid-name

class EnsembleGeneratorTest(absltest.TestCase):

  def test_permute_categorical_structure(self):
    # Create a simple dataset with one categorical column and one numerical
    X = pd.DataFrame({
        "cat": ["A", "B", "A", "B", "C"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y = np.array([0, 1, 0, 1, 0])

    # Pre-encode using TransformToNumerical as the Classifier does
    encoder = TransformToNumerical(min_cat_frequency=1)
    X_enc = encoder.fit_transform(X)
    # "cat" should be column 0 (value 0, 1, 2), "num" column 1.

    cat_features = [0]
    n_estimators = 10

    generator = EnsembleGenerator(
        n_estimators=n_estimators,
        norm_methods=["none"],  # Use none to minimize transformation noise
        cat_features=cat_features,
        permute_categorical=True,
        random_state=42
    )

    generator.fit(X_enc, y)

    # Check if permutations are generated
    self.assertLen(generator.cat_permutations_, 1) # One group for "none" norm
    perms = generator.cat_permutations_["none"]
    self.assertLen(perms, n_estimators)

    # Verify structure of permutations
    for perm in perms:
      self.assertIn(0, perm)  # Column 0 should be in the permutation dict
      mapping = perm[0]
      # Original values are 0, 1, 2
      original_values = set(range(3))
      self.assertEqual(set(mapping.keys()), original_values)
      self.assertEqual(set(mapping.values()), original_values)

    # Verify that we actually have different permutations
    # It's statistically improbable that 10 estimators all have identity
    # permutation
    identity_count = 0
    for perm in perms:
      mapping = perm[0]
      is_identity = all(k == v for k, v in mapping.items())
      if is_identity:
        identity_count += 1

    self.assertLess(
        identity_count, n_estimators, "Categorical permutations should vary."
    )

  def test_permute_categorical_application(self):
    # Test that transform actually changes the data
    # X has 2 samples, 1 cat feature with values 0 and 1.
    X_enc = np.array([[0, 10.0], [1, 20.0]])
    y = np.array([0, 1])
    cat_features = [0]

    # Force a specific permutation logic by mocking or just checking output
    # We rely on random_state for reproducibility
    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        cat_features=cat_features,
        permute_categorical=True,
        # Disable feature shuffling to isolate value permutation
        feat_shuffle_method="none",
        random_state=42,
    )

    generator.fit(X_enc, y)
    data = generator.transform(X_enc)

    # Extract the transformed batch for "none" normalization
    X_out, _ = data["none"]
    # X_out shape: (n_estimators, n_samples, n_features)

    # We expect some estimators to swap 0 and 1 in the first column
    # Since PreprocessingPipeline with "none" still does StandardScaler,
    # the values won't be exactly 0 and 1, but they will be distinct.
    # However, if 0->1 and 1->0 (swap), the resulting standardized values
    # should effectively flip signs (if mean centered) or at least change.

    # Let's inspect the raw permutations to know what to expect
    perms = generator.cat_permutations_["none"]

    # Find an estimator that swaps 0 and 1
    swap_idx = -1
    for i, perm in enumerate(perms):
      mapping = perm[0]
      if mapping.get(0) == 1 and mapping.get(1) == 0:
        swap_idx = i
        break

    if swap_idx != -1:
      # Compare with an estimator that (hopefully) didn't swap or at least is
      # different
      # Actually, let's just compare X_out[swap_idx] vs input logic.
      # Input col 0: [0, 1]
      # Swapped col 0: [1, 0]
      # Standard Scaler on [0, 1] -> [-1, 1] (roughly)
      # Standard Scaler on [1, 0] -> [1, -1]
      # So the output values should be inverted relative to each other?
      # WAIT: The StandardScaler is fitted on the permuted TRAINING data.
      # If we permute train and test consistently, the distribution stats might
      # remain similar (since it's just relabeling), but the *instances* change
      # values.
      # 0 becomes 1.

      # Let's verify that X_out[swap_idx, 0, 0] (Sample 0, Feat 0)
      # is different from X_out[identity_idx, 0, 0] if we find an identity one.

      # Easier check: In the swapped estimator, Sample 0 (was 0->1) should look
      # like Sample 1 (was 1)
      # from a non-swapped estimator? No, Sample 1 in non-swapped is 1. Sample 0
      # in swapped is 1.
      # So yes, X_out[swap_idx, 0, 0] should be close to
      # X_out[non_swapped, 1, 0].
      pass

    # Assert outputs are not all identical across estimators for col 0.
    col0_values = X_out[:, 0, 0] # (n_estimators,)
    self.assertGreater(
        np.std(col0_values),
        1e-6,
        "Categorical values should vary across estimators due to permutation",
    )

    # Verify col 1 (numerical) does NOT vary (feat_shuffle="none")
    # Actually PreprocessingPipeline adds noise? No, only RTDLQuantile does.
    # CustomStandardScaler is deterministic.
    # So col 1 should be identical across estimators.
    col1_values = X_out[:, 0, 1]
    self.assertLess(
        np.std(col1_values),
        1e-6,
        "Numerical values should not vary if shuffling is off",
    )

  def test_permute_categorical_false(self):
    X_enc = np.array([[0, 10.0], [1, 20.0], [2, 30.0]])
    y = np.array([0, 1, 0])
    cat_features = [0]

    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        cat_features=cat_features,
        permute_categorical=False, # DISABLED
        feat_shuffle_method="none",
        random_state=42
    )

    generator.fit(X_enc, y)

    # Check permutations are None
    perms = generator.cat_permutations_["none"]
    for perm in perms:
      self.assertIsNone(perm)

    data = generator.transform(X_enc)
    X_out, _ = data["none"]

    # Check that outputs are identical across estimators for col 0
    col0_values = X_out[:, 0, 0]
    self.assertLess(
        np.std(col0_values),
        1e-6,
        "Categorical values should be identical if permutation is disabled",
    )

  def test_row_and_col_subsampling(self):
    X_enc = np.array([[0, 10.0], [1, 20.0], [2, 30.0], [3, 40.0]])
    y = np.array([0, 1, 0, 1])

    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        max_num_rows=2,
        max_num_features=1,
        random_state=42,
    )

    generator.fit(X_enc, y)

    feat_patterns = generator.feature_shuffle_patterns_["none"]
    for p in feat_patterns:
      self.assertLen(p, 1)

    row_patterns = generator.row_subsample_patterns_["none"]
    for p in row_patterns:
      self.assertLen(p, 2)

    X_test = np.array([[4, 50.0], [5, 60.0]])
    data = generator.transform(X_test)
    X_out, y_out = data["none"]

    self.assertEqual(X_out.shape, (5, 4, 1))
    self.assertEqual(y_out.shape, (5, 2))

  def test_max_num_features(self):
    X_enc = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        max_num_features=2,
        random_state=42,
    )

    generator.fit(X_enc, y)

    feat_patterns = generator.feature_shuffle_patterns_["none"]
    for p in feat_patterns:
      self.assertLen(p, 2)

  def test_variable_crosses_shapes(self):
    X_enc = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # "split" allocation with sqrt feature crosses: sqrt(5) -> 2, so
    # even-indexed members get no crosses and odd-indexed members get 2.
    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        n_feature_crosses="sqrt",
        random_state=42,
    )
    generator.fit(X_enc, y)

    self.assertEqual(generator.k_crosses_list_, [0, 2, 0, 2, 0])

    data = generator.transform(X_enc)
    X_out, _ = data["none"]
    # max features = 5 + 2 = 7
    self.assertEqual(X_out.shape[2], 7)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class OOFPredictionTest(absltest.TestCase):

  def test_classifier_predict_oof_proba(self):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    classifier = TabFMClassifier(
        model=model, n_estimators=2, batch_size=2
    )

    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)

    classifier.fit(X, y)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 2))

    with mock.patch.object(
        classifier, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      oof_preds = classifier.predict_oof_proba(cv=2)

      self.assertEqual(oof_preds.shape, (2, 10, 2))
      self.assertTrue(mock_batch_forward.called)

  def test_regressor_feature_crosses(self):
    regressor = TabFMRegressor(
        n_feature_crosses="sqrt",
        n_estimators=2,
        model=mock.Mock(),
    )

    X = np.random.rand(10, 4)
    y = np.random.rand(10)

    regressor.fit(X, y)

    # Split allocation with sqrt(4)=2 crosses on the odd member: pooled data
    # has N + 2 = 4 + 2 = 6 columns.
    self.assertEqual(regressor.ensemble_generator_.X_.shape[1], 6)

    X_test = np.random.rand(5, 4)
    data = regressor.ensemble_generator_.transform(X_test)
    for _, (Xs, _) in data.items():
      # Members are padded to N + K = 4 + 2 = 6 columns.
      self.assertEqual(Xs.shape[2], 6)

  def test_regressor_svd_features(self):
    regressor = TabFMRegressor(
        n_svd_features="sqrt",
        n_estimators=2,
        model=mock.Mock(),
    )

    X = pd.DataFrame({
        "num1": np.random.rand(10),
        "num2": np.random.rand(10),
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    })
    y = np.random.rand(10)

    regressor.fit(X, y)

    # 3 original (2 num + 1 cat) + 1 SVD (sqrt(3)=1) = 4 columns.
    self.assertEqual(regressor.ensemble_generator_.X_.shape[1], 4)

    X_test = pd.DataFrame({
        "num1": np.random.rand(5),
        "num2": np.random.rand(5),
        "cat1": ["a", "b", "a", "b", "a"],
    })
    X_test_encoded = regressor.X_encoder_.transform(X_test)
    data = regressor.ensemble_generator_.transform(X_test_encoded)
    for _, (Xs, _) in data.items():
      self.assertEqual(Xs.shape[2], 4)

  def test_regressor_nnls_blending(self):
    model = tabfm_model.TabFM(
        loss="rmse",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    regressor = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        enable_nnls=True,
        nnls_beta=0.5,
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    def mock_forward(Xs_batch, ys_batch, _cat_mask_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 1))

    with mock.patch.object(
        regressor, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      regressor.fit(X, y)

      self.assertTrue(mock_batch_forward.called)
      self.assertTrue(hasattr(regressor, "ensemble_weights_"))
      self.assertEqual(regressor.ensemble_weights_.shape, (2,))
      self.assertAlmostEqual(np.sum(regressor.ensemble_weights_), 1.0)

      test_preds = regressor.predict(X[:3])
      self.assertEqual(test_preds.shape, (3,))

  def test_regressor_sqrt_schedule(self):
    regressor = TabFMRegressor(
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        max_num_features=4,
        n_estimators=2,
        model=mock.Mock(),
    )

    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    regressor.fit(X, y)

    # Split allocation: sqrt(min(5, 4))=2. 5 original + 2 crosses pool +
    # 2 SVD pool = 9 columns.
    self.assertEqual(regressor.ensemble_generator_.X_.shape[1], 9)

    X_test = np.random.rand(5, 5)
    X_test_encoded = regressor.X_encoder_.transform(X_test)
    data = regressor.ensemble_generator_.transform(X_test_encoded)
    for _, (Xs, _) in data.items():
      # Members padded to max_num_features (4) + K_crosses (2) + K_svd (2) = 8.
      self.assertEqual(Xs.shape[2], 8)

  def test_regressor_predict_oof(self):
    model = tabfm_model.TabFM(
        loss="rmse",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    regressor = TabFMRegressor(
        model=model, n_estimators=2, batch_size=2
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    regressor.fit(X, y)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 1))

    with mock.patch.object(
        regressor, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      oof_preds = regressor.predict_oof(cv=2)

      self.assertEqual(oof_preds.shape, (2, 10))
      self.assertTrue(mock_batch_forward.called)

  def test_classifier_predict_oof_proba_with_subsampling(self):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    classifier = TabFMClassifier(
        model=model,
        n_estimators=2,
        batch_size=2,
        max_num_rows=8,
    )

    X = np.random.rand(10, 3)
    y = np.array([0, 1] * 5)

    classifier.fit(X, y)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 2))

    with mock.patch.object(
        classifier, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      oof_preds = classifier.predict_oof_proba(cv=2)

      self.assertEqual(oof_preds.shape, (2, 10, 2))
      self.assertTrue(mock_batch_forward.called)

  def test_regressor_variable_crosses_predict_flow(self):
    regressor = TabFMRegressor(
        n_estimators=2,
        batch_size=2,
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        model=mock.Mock(),
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    X_test = np.random.rand(5, 3)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 1))

    with mock.patch.object(
        regressor, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      regressor.fit(X, y)
      # Split allocation with sqrt(3)=1 for both crosses and SVD.
      self.assertEqual(regressor.ensemble_generator_.k_crosses_list_, [0, 1])
      self.assertEqual(regressor.ensemble_generator_.k_svd_list_, [0, 1])

      preds = regressor.predict(X_test)
      self.assertEqual(preds.shape, (5,))

      self.assertTrue(mock_batch_forward.called)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class BatchForwardTest(absltest.TestCase):

  def test_classifier_batch_forward(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )

    classifier = TabFMClassifier(
        model=model, n_estimators=4, batch_size=2
    )
    classifier.n_classes_ = 3
    classifier.classes_ = np.array([0, 1, 2])

    # Generate dummy input arrays.
    # Xs shape: (n_estimators, max_seq_len, num_features)
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.randint(0, 3, size=(4, 6))

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = classifier._batch_forward(Xs, ys)

    # After concatenation, output shape should be
    # (n_estimators, test_size, num_classes).
    # Since Xs length is 10 and ys length (train_size) is 6, test_size is 4.
    self.assertEqual(outputs.shape, (4, 4, 3))

  def test_regressor_batch_forward_rmse(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="rmse",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )

    regressor = TabFMRegressor(model=model, n_estimators=4)

    # Generate dummy input arrays.
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.rand(4, 6)

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = regressor._batch_forward(Xs, ys)

    # Output shape should be (n_estimators, test_size, out_dim).
    # Since Xs length is 10 and ys length (train_size) is 6, test_size is 4.
    # TabFM with rmse outputs 1 value per prediction.
    self.assertEqual(outputs.shape, (4, 4, 1))

  def test_regressor_batch_forward_cross_entropy(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )

    regressor = TabFMRegressor(model=model, n_estimators=4)

    # Generate dummy input arrays.
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.rand(4, 6)

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = regressor._batch_forward(Xs, ys)

    # Output shape should be (n_estimators, test_size, out_dim).
    # TabFM with cross_entropy outputs max_classes bins.
    self.assertEqual(outputs.shape, (4, 4, 10))


@unittest.skipUnless(HAS_JAX, "JAX is required")
class ModelTypeMismatchTest(absltest.TestCase):
  """Using the wrong checkpoint type fails fast with an actionable error."""

  def _tiny_model(self, loss):
    return tabfm_model.TabFM(
        loss=loss,
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )

  def test_regressor_predict_raises_on_classification_model(self):
    regressor = TabFMRegressor(
        model=self._tiny_model("cross_entropy"), n_estimators=2
    )
    X = np.random.rand(10, 3)
    regressor.fit(X, np.random.rand(10))

    with self.assertRaisesRegex(ValueError, "model_type='regression'"):
      regressor.predict(np.random.rand(4, 3))

  def test_regressor_predict_oof_raises_on_classification_model(self):
    regressor = TabFMRegressor(
        model=self._tiny_model("cross_entropy"), n_estimators=2
    )
    X = np.random.rand(10, 3)
    regressor.fit(X, np.random.rand(10))

    with self.assertRaisesRegex(ValueError, "model_type='regression'"):
      regressor.predict_oof(cv=2)

  def test_classifier_predict_proba_raises_on_regression_model(self):
    classifier = TabFMClassifier(model=self._tiny_model("rmse"), n_estimators=2)
    X = np.random.rand(10, 3)
    classifier.fit(X, np.array([0, 1] * 5))

    with self.assertRaisesRegex(ValueError, "fit on 2 classes"):
      classifier.predict_proba(np.random.rand(4, 3))

  def test_classifier_predict_oof_proba_raises_on_regression_model(self):
    classifier = TabFMClassifier(model=self._tiny_model("rmse"), n_estimators=2)
    X = np.random.rand(10, 3)
    classifier.fit(X, np.array([0, 1] * 5))

    with self.assertRaisesRegex(ValueError, "fit on 2 classes"):
      classifier.predict_oof_proba(cv=2)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class CalibrationTest(absltest.TestCase):

  def setUp(self):
    self.model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )

  def test_calibration_binary(self):
    y = np.concatenate(
        [np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)]
    )
    X = np.random.rand(100, 3)
    mock_oof = np.random.dirichlet([1, 1], size=(2, 100))

    for method in ["platt"]:
      classifier = TabFMClassifier(
          model=self.model,
          n_estimators=2,
          binary_calibration_method=method,
      )

      with mock.patch.object(
          classifier, "predict_oof_proba", return_value=mock_oof
      ):
        classifier.fit(X, y)

      self.assertTrue(hasattr(classifier, "calibration_params_"))

      mock_out = np.random.dirichlet([1, 1], size=(2, 100))

      with mock.patch.object(
          classifier, "_batch_forward", return_value=mock_out
      ):
        probs = classifier.predict_proba(X)
        self.assertEqual(probs.shape, (100, 2))

  def test_calibration_multiclass(self):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    y = np.concatenate([
        np.zeros(50, dtype=np.int64),
        np.ones(50, dtype=np.int64),
        np.full(50, 2, dtype=np.int64),
    ])
    X = np.random.rand(150, 3)
    mock_oof = np.random.dirichlet([1, 1, 1], size=(2, 150))

    for method in ["vector"]:
      classifier = TabFMClassifier(
          model=model,
          n_estimators=2,
          multiclass_calibration_method=method,
      )

      with mock.patch.object(
          classifier, "predict_oof_proba", return_value=mock_oof
      ):
        classifier.fit(X, y)

      self.assertTrue(hasattr(classifier, "calibration_params_"))

      mock_out = np.random.dirichlet([1, 1, 1], size=(2, 150))

      with mock.patch.object(
          classifier, "_batch_forward", return_value=mock_out
      ):
        probs = classifier.predict_proba(X)
        self.assertEqual(probs.shape, (150, 3))


@unittest.skipUnless(HAS_JAX, "JAX is required")
class StackingTest(absltest.TestCase):

  def setUp(self):
    self.model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )

  def test_classifier_crosses_and_svd(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        max_num_features=4,
    )

    X = pd.DataFrame({
        "num1": np.random.rand(10),
        "num2": np.random.rand(10),
        "num3": np.random.rand(10),
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "cat2": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
    })
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    classifier.fit(X, y)

    self.assertEqual(classifier.ensemble_generator_.X_.shape[1], 9)

    probs = classifier.predict_proba(X)
    self.assertEqual(probs.shape, (10, 2))

  def test_classifier_nnls_blending(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        enable_nnls=True,
        nnls_beta=0.5,
        average_logits=False,
    )

    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    mock_oof = np.random.dirichlet([1, 1], size=(2, 10))

    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      classifier.fit(X, y)

    self.assertTrue(hasattr(classifier, "ensemble_weights_"))
    self.assertEqual(classifier.ensemble_weights_.shape, (2,))
    self.assertAlmostEqual(np.sum(classifier.ensemble_weights_), 1.0)

    mock_logits = np.random.rand(2, 5, 2)
    with mock.patch.object(
        classifier, "_predict_proba_internal", return_value=mock_logits
    ):
      probs = classifier.predict_proba(X[:5])
      self.assertEqual(probs.shape, (5, 2))

  def test_classifier_nnls_and_calibration(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        enable_nnls=True,
        nnls_beta=0.5,
        binary_calibration_method="platt",
        average_logits=False,
    )

    X = np.random.rand(50, 3)
    y = np.concatenate(
        [np.zeros(25, dtype=np.int64), np.ones(25, dtype=np.int64)]
    )

    mock_oof = np.random.dirichlet([1, 1], size=(2, 50))

    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      with mock.patch.object(classifier, "_fit_calibration") as mock_fit_cal:
        classifier.fit(X, y)

        self.assertTrue(mock_fit_cal.called)
        P_arg = mock_fit_cal.call_args[0][0]
        expected_P = np.tensordot(
            classifier.ensemble_weights_, mock_oof, axes=(0, 0)
        )
        np.testing.assert_allclose(P_arg, expected_P)

  def test_classifier_nnls_and_calibration_small_dataset(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        enable_nnls=True,
        nnls_beta=0.5,
        binary_calibration_method="platt",
        calibration_lambda=0.01,
        average_logits=False,
    )

    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mock_oof = np.random.dirichlet([1, 1], size=(2, 10))

    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      classifier.fit(X, y)

    self.assertEqual(classifier.active_calibration_method_, "platt")
    self.assertTrue(hasattr(classifier, "calibration_params_"))
    self.assertIn("A", classifier.calibration_params_)

  def test_average_logits_and_enable_nnls_raises(self):
    with self.assertRaises(ValueError):
      TabFMClassifier(
          model=self.model,
          average_logits=True,
          enable_nnls=True,
      )

  def test_max_num_rows(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        max_num_rows=6,
    )
    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    mock_oof = np.random.dirichlet([1, 1], size=(2, 10))
    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      classifier.fit(X, y)

    self.assertEqual(classifier.ensemble_generator_.X_.shape[0], 10)
    patterns_none = classifier.ensemble_generator_.row_subsample_patterns_[
        "none"
    ]
    patterns_power = classifier.ensemble_generator_.row_subsample_patterns_[
        "power"
    ]
    self.assertLen(patterns_none[0], 6)
    self.assertLen(patterns_power[0], 6)

  def test_min_rows_for_single_val_split_classifier(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=1,
        enable_nnls=True,
        average_logits=False,
        min_rows_for_single_val_split=2,
        num_folds_for_cv=5,
    )
    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    with mock.patch.object(
        classifier, "_batch_forward", return_value=np.zeros((1, 2, 2))
    ) as mock_forward:
      classifier.fit(X, y)

    # With 10 rows and 5 folds, val fold size is 2.
    # It should therefore only run 1 fold instead of 5.
    self.assertEqual(mock_forward.call_count, 1)

  def test_min_rows_for_single_val_split_regressor(self):
    regressor = TabFMRegressor(
        model=self.model,
        n_estimators=1,
        enable_nnls=True,
        min_rows_for_single_val_split=2,
        num_folds_for_cv=5,
    )
    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    with mock.patch.object(
        regressor, "_batch_forward", return_value=np.zeros((1, 2, 1))
    ) as mock_forward:
      regressor.fit(X, y)

    self.assertEqual(mock_forward.call_count, 1)


class EnsemblePresetTest(absltest.TestCase):

  def test_classifier_ensemble_preset(self):
    clf = TabFMClassifier.ensemble(model=mock.Mock())
    self.assertEqual(clf.n_estimators, 32)
    self.assertFalse(clf.average_logits)
    self.assertEqual(clf.n_feature_crosses, "sqrt")
    self.assertEqual(clf.n_svd_features, "sqrt")
    self.assertTrue(clf.enable_nnls)
    self.assertEqual(clf.binary_calibration_method, "platt")
    self.assertEqual(clf.multiclass_calibration_method, "vector")
    # Default-mode knobs are unchanged by the preset.
    self.assertEqual(clf.max_num_features, 500)

  def test_regressor_ensemble_preset(self):
    reg = TabFMRegressor.ensemble(model=mock.Mock())
    self.assertEqual(reg.n_estimators, 32)
    self.assertEqual(reg.n_feature_crosses, "sqrt")
    self.assertEqual(reg.n_svd_features, "sqrt")
    self.assertTrue(reg.enable_nnls)
    self.assertEqual(reg.max_num_features, 500)

  def test_ensemble_overrides_take_precedence(self):
    clf = TabFMClassifier.ensemble(
        model=mock.Mock(), n_estimators=8, enable_nnls=False
    )
    self.assertEqual(clf.n_estimators, 8)
    self.assertFalse(clf.enable_nnls)
    # Non-overridden preset values are preserved.
    self.assertEqual(clf.n_feature_crosses, "sqrt")

  def test_default_mode_keeps_features_off(self):
    clf = TabFMClassifier(model=mock.Mock())
    self.assertEqual(clf.n_estimators, 32)
    self.assertEqual(clf.max_num_features, 500)
    self.assertTrue(clf.average_logits)
    self.assertEqual(clf.n_feature_crosses, 0)
    self.assertEqual(clf.n_svd_features, 0)
    self.assertFalse(clf.enable_nnls)
    self.assertIsNone(clf.binary_calibration_method)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class LabelEncodingTest(absltest.TestCase):

  def test_classes_are_alphabetical(self):
    # classes_ must follow sklearn's sorted (alphabetical) convention,
    # independent of label order of appearance. Regression guard for the
    # y-encoder being constructed with mode="alphabetical".
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    clf = TabFMClassifier(model=model, n_estimators=2)
    x = np.random.rand(6, 3)
    # Appearance order is ("z", "a"); alphabetical order is ("a", "z").
    y = np.array(["z", "a", "z", "a", "z", "a"])
    clf.fit(x, y)
    np.testing.assert_array_equal(clf.classes_, np.array(["a", "z"]))
    self.assertEqual(clf.y_encoder_.mode, "alphabetical")


class NotFittedTest(absltest.TestCase):

  def test_predict_before_fit_raises_not_fitted(self):
    X = np.zeros((3, 2))
    with self.assertRaises(NotFittedError):
      TabFMRegressor(model=mock.Mock()).predict(X)
    with self.assertRaises(NotFittedError):
      TabFMClassifier(model=mock.Mock()).predict(X)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class EstimatorPickleTest(absltest.TestCase):
  """Fitted estimators must survive pickle after a predict.

  AutoGluon / TabArena save the fitted estimator with stdlib pickle. The
  first predict memoizes nnx.jit-compiled step functions on the estimator
  (see _batch_forward); those closures cannot be pickled, so they are
  dropped from the pickled state and rebuilt lazily after restore.
  """

  def _tiny_model(self, loss):
    return tabfm_model.TabFM(
        loss=loss,
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )

  def test_classifier_pickle_round_trip_after_predict(self):
    classifier = TabFMClassifier(
        model=self._tiny_model("cross_entropy"), n_estimators=2
    )
    X = np.random.rand(12, 3)
    X_test = np.random.rand(4, 3)
    classifier.fit(X, np.array([0, 1] * 6))
    proba = classifier.predict_proba(X_test)

    restored = pickle.loads(pickle.dumps(classifier))

    np.testing.assert_allclose(restored.predict_proba(X_test), proba)

  def test_regressor_pickle_round_trip_after_predict(self):
    regressor = TabFMRegressor(model=self._tiny_model("rmse"), n_estimators=2)
    X = np.random.rand(12, 3)
    X_test = np.random.rand(4, 3)
    regressor.fit(X, np.random.rand(12))
    preds = regressor.predict(X_test)

    restored = pickle.loads(pickle.dumps(regressor))

    np.testing.assert_allclose(restored.predict(X_test), preds)


class DatetimeDetectionTest(absltest.TestCase):

  def test_detects_datetime_in_object_and_string_dtypes(self):
    # Regression guard: pandas>=3 defaults text columns to the StringDtype
    # (not object). _looks_like_datetime must accept both, otherwise
    # date-as-text columns are missed and silently treated as categorical.
    dates = ["2020-01-01", "2021-05-02", "2019-12-31", "2018-07-15"] * 4
    self.assertTrue(_looks_like_datetime(pd.Series(dates, dtype=object)))
    self.assertTrue(_looks_like_datetime(pd.Series(dates, dtype="string")))

  def test_non_datetime_text_not_detected(self):
    words = ["red", "green", "blue", "red"] * 4
    self.assertFalse(_looks_like_datetime(pd.Series(words, dtype=object)))
    self.assertFalse(_looks_like_datetime(pd.Series(words, dtype="string")))

  def test_partially_date_column_is_detected(self):
    # Intentional leniency: a column that is partly dates (here ~50%) is still
    # treated as datetime even with some non-parseable values mixed in. The
    # coerce + threshold parse is deliberate, so date columns with stray junk
    # are not lost to the categorical fallback.
    vals = ["2020-01-01", "2021-05-02", "red", "blue"] * 5
    self.assertTrue(_looks_like_datetime(pd.Series(vals, dtype=object)))
    self.assertTrue(_looks_like_datetime(pd.Series(vals, dtype="string")))

  def test_mostly_non_date_column_not_detected(self):
    # Below the parse threshold (~10% dates) -> not datetime -> categorical.
    vals = ["red", "green", "blue", "yellow", "purple",
            "a", "b", "c", "d", "e",
            "f", "g", "h", "i", "j",
            "k", "l", "m", "2020-01-01", "2021-05-02"]  # 2/20 = 10% dates
    self.assertFalse(_looks_like_datetime(pd.Series(vals, dtype=object)))
    self.assertFalse(_looks_like_datetime(pd.Series(vals, dtype="string")))


class ColumnNameRobustnessTest(absltest.TestCase):

  def test_duplicate_column_names_raise_a_clear_error(self):
    # Joins/concats commonly produce duplicate column names. sklearn's
    # ColumnTransformer cannot process them, so fail fast with an actionable
    # message instead of the cryptic "'DataFrame' object has no attribute
    # 'dtype'" raised previously.
    X = pd.DataFrame(
        [[1.0, "x", 2.0], [3.0, "y", 4.0], [5.0, "x", 6.0]],
        columns=["a", "b", "a"],
    )

    with self.assertRaisesRegex(ValueError, r"duplicate column names.*'a'"):
      TransformToNumerical(min_cat_frequency=1).fit(X)

  def test_classifier_fit_duplicate_column_names_raises_clear_error(self):
    classifier = TabFMClassifier(
        model=mock.Mock(max_classes=10), n_estimators=2
    )
    X = pd.DataFrame(np.random.rand(10, 2), columns=["a", "a"])
    y = np.array([0, 1] * 5)

    with self.assertRaisesRegex(ValueError, "duplicate column names"):
      classifier.fit(X, y)

  def test_datetime_column_with_non_string_name(self):
    # DataFrames built from arrays get integer column labels; the datetime
    # expansion used to crash on `0 + "."` when building derived names.
    X = pd.DataFrame({0: pd.date_range("2020-01-01", periods=4, freq="D")})

    out = TransformToNumerical().fit_transform(X)

    self.assertEqual(out.shape, (4, 5))  # unix-ns + 4 derived features


class GlobalPandasOutputTest(absltest.TestCase):
  """Fit must ignore a user's global transform_output="pandas" config."""

  def test_regressor_fit_with_pandas_output_config(self):
    # A user who calls sklearn.set_config(transform_output="pandas") used to
    # crash the regressor: the internal target StandardScaler returned a
    # DataFrame, and `.flatten()` on it raised AttributeError. See issue #58.
    regressor = TabFMRegressor(n_estimators=2, model=mock.Mock())

    X = np.random.rand(10, 4)
    y = np.random.rand(10)

    with config_context(transform_output="pandas"):
      regressor.fit(X, y)
      # The target scaler stays pinned to ndarray output under the config.
      scaled = regressor.y_scaler_.transform(y.reshape(-1, 1))

    self.assertIsInstance(scaled, np.ndarray)


if __name__ == "__main__":
  absltest.main()

```

## File: tabfm/src/hugging_face/convert_and_upload.py

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

"""Utility to convert JAX TabFM checkpoints to PyTorch and upload to Hugging Face."""

import os
import logging
from typing import Any, Optional, Tuple, Literal
from absl import app
from absl import flags
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Float, Int

from tabfm.src.jax import tabfm_v1_0_0
from tabfm.src.pytorch import model as MT
from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
from tabfm.src.hugging_face.torch_convert import jax_params, convert

# Architecture config of the v1.0.0 checkpoint.
V1_0_0_CONFIG = dict(
    embed_dim=256,
    max_classes=10,
    col_num_blocks=3,
    col_nhead=4,
    col_num_inds=256,
    row_num_blocks=3,
    row_nhead=8,
    row_num_cls=8,
    icl_num_blocks=24,
    icl_nhead=8,
    ff_factor=4,
    feature_group_size=3,
)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "model_type",
    "all",
    ["classification", "regression", "all"],
    "Type of model to convert.",
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Local directory to save the converted PyTorch checkpoint.",
    required=True,
)
flags.DEFINE_string(
    "repo_id",
    None,
    "Hugging Face repository ID (e.g. 'google/tabfm-1.0.0-pytorch'). "
    "If not provided, the conversion is run locally and not uploaded.",
)
flags.DEFINE_string(
    "token",
    None,
    "Hugging Face write token. Required if repo_id is provided.",
)
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Local path to JAX checkpoint directory. If None, JAX weights will be "
    "downloaded from Hugging Face.",
)


def convert_model(
    model_type: Literal["classification", "regression"],
    checkpoint_path: Optional[str] = None,
) -> Tuple[TabFM_HF, float]:
  """Converts JAX checkpoint of model_type to PyTorch TabFM and runs parity verification."""
  logging.info("Loading JAX %s model...", model_type)
  is_classifier = (model_type == "classification")
  
  # Load JAX model
  jm = tabfm_v1_0_0.load(
      model_type=model_type,
      checkpoint_path=checkpoint_path,
      col_attention_impl="jax", row_attention_impl="jax", icl_attention_impl="jax", dtype=jnp.float32,
  )
  jp = jax_params(jm)
  
  # Retrieve the decoder hidden dimension from the loaded JAX model weights
  decoder_hidden = jp["icl_predictor.decoder.layers.0.kernel"].shape[1]
  
  logging.info("Instantiating PyTorch model...")
  torch_model = TabFM_HF(
      decoder_hidden=decoder_hidden,
      is_classifier=is_classifier,
      **V1_0_0_CONFIG,
  )
  
  logging.info("Converting parameters...")
  state_dict, missing = convert(jp, torch_model)
  if missing:
    raise ValueError(f"Missing parameters during conversion: {missing}")
    
  torch_model.load_state_dict(state_dict, strict=True)
  torch_model.eval()
  
  # Run a parity check on dummy inputs to ensure the conversion is correct.
  logging.info("Verifying parity on dummy inputs...")
  max_diff = verify_parity(jm, torch_model, is_classifier)
  logging.info("Parity check max difference: %e", max_diff)
  if max_diff >= 1e-4:
    raise ValueError(f"Parity check failed with max difference: {max_diff}")
    
  return torch_model, max_diff


def verify_parity(
    jax_model: Any,
    torch_model: MT.TabFM,
    is_classifier: bool,
) -> float:
  """Runs JAX and PyTorch models on identical random inputs and returns max absolute difference."""
  B, T, H = 1, 12, 5
  rng = np.random.default_rng(0)
  x: Float[np.ndarray, "B T H"] = rng.random((B, T, H)).astype(np.float32)
  
  if is_classifier:
    y: Int[np.ndarray, "B T"] = rng.integers(0, 3, (B, T)).astype(np.int32)
  else:
    y: Float[np.ndarray, "B T"] = rng.standard_normal((B, T)).astype(np.float32)
    
  ts: Int[np.ndarray, "B"] = np.array([7], dtype=np.int32)
  
  # Run JAX
  jout = np.asarray(jax_model(jnp.asarray(x), jnp.asarray(y), train_size=jnp.asarray(ts)))
  
  # Run PyTorch
  with torch.no_grad():
    tout = torch_model(torch.tensor(x), torch.tensor(y), torch.tensor(ts)).numpy()
    
  max_diff = float(np.max(np.abs(jout - tout)))
  return max_diff


def save_checkpoint(
    model: TabFM_HF,
    output_dir: str,
    model_type: str,
) -> str:
  """Saves model weights and config to output_dir using save_pretrained."""
  model_dir = os.path.join(output_dir, model_type)
  os.makedirs(model_dir, exist_ok=True)
  logging.info("Saving model to %s...", model_dir)
  model.save_pretrained(model_dir)
  logging.info("Saved model to %s", model_dir)
  return model_dir


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
    
  # Set JAX to run on CPU to avoid device initialization errors on machines
  # without proper JAX CUDA driver setups.
  os.environ["JAX_PLATFORMS"] = "cpu"
  
  model_types = (
      ["classification", "regression"]
      if FLAGS.model_type == "all"
      else [FLAGS.model_type]
  )
  
  local_dirs = {}
  for mtype in model_types:
    logging.info("=== Starting conversion for %s ===", mtype)
    torch_model, max_diff = convert_model(mtype, FLAGS.checkpoint_path)
    logging.info("Conversion successful! Parity max diff = %e", max_diff)
    
    saved_dir = save_checkpoint(torch_model, FLAGS.output_dir, mtype)
    local_dirs[mtype] = saved_dir
    
  if FLAGS.repo_id:
    if not FLAGS.token:
      raise ValueError("Hugging Face token is required when repo_id is provided.")

    from huggingface_hub import HfApi  # pylint: disable=g-import-not-at-top
    api = HfApi(token=FLAGS.token)

    for mtype, sdir in local_dirs.items():
      logging.info("Uploading %s folder to %s...", mtype, FLAGS.repo_id)
      api.upload_folder(
          folder_path=sdir,
          repo_id=FLAGS.repo_id,
          path_in_repo=mtype,
          repo_type="model",
      )
    logging.info("Successfully uploaded PyTorch checkpoints to Hugging Face!")


if __name__ == "__main__":
  app.run(main)

```

## File: tabfm/src/hugging_face/torch_convert.py

```
"""End-to-end JAX->PyTorch converter + parity test on a small TabFM.

Validates the whole port + the weight converter (unstacking vmap'd blocks,
transposing Linear kernels, name remap) by comparing a small random-init JAX
TabFM against the torch port on identical input, in float32.
"""
import re
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import torch
from flax import nnx

from tabfm.src.jax.model import TabFM as JaxTabFM, YEmbeddingScheme
from tabfm.src.pytorch import model as MT

CFG = dict(embed_dim=8, max_classes=3, col_num_blocks=2, col_nhead=2,
           col_num_inds=4, row_num_blocks=2, row_nhead=2, row_num_cls=2,
           icl_num_blocks=2, icl_nhead=2, ff_factor=2, feature_group_size=3)


def jax_params(m):
  flat = jtu.tree_flatten_with_path(nnx.state(m))[0]
  return {".".join(str(getattr(k, "key", k)) for k in kp).replace("..value", ""): np.asarray(v)
          for kp, v in flat}


def to_jax_name(tkey):
  """Map a torch state_dict key -> (jax_name, block_idx_or_None, kind)."""
  block_idx = None
  jk = tkey
  m = re.search(r"\.blocks\.(\d+)\.", tkey)
  if m:
    block_idx = int(m.group(1))
    jk = jk.replace(f".blocks.{block_idx}.", ".blocks.")
  # any MLP (decoder, y_encoder, regression y_embedder_lookup): torch layers.k
  # (consecutive Linears) -> jax layers.2k (activation module sits between).
  m2 = re.search(r"\.layers\.(\d+)\.", jk)
  if m2:
    k = int(m2.group(1))
    jk = re.sub(r"\.layers\.\d+\.", f".layers.{2*k}.", jk, count=1)
  if jk.endswith(".per_dim_scale"):
    return jk + ".per_dim_scale", block_idx, "direct"
  if jk.endswith(".weight"):
    base = jk[:-len(".weight")]
    # classification y-embedding is an nnx.Embed (param named ".embedding");
    # the regression variant is an MLP and falls through to kernel resolution.
    if jk.endswith("y_embedder_lookup.weight"):
      return base + ".embedding", block_idx, "direct"
    return base, block_idx, "weight"  # resolve scale/kernel by shape later
  if jk.endswith(".bias"):
    return jk, block_idx, "direct"
  return jk, block_idx, "direct"  # buffers, cls_tokens, ind_vectors


def convert(jax_p, torch_model):
  sd = {}
  missing = []
  for tkey, tparam in {**dict(torch_model.named_parameters()),
                       **dict(torch_model.named_buffers())}.items():
    jname, bi, kind = to_jax_name(tkey)
    # resolve ".weight" ambiguity (RMSNorm scale vs Linear kernel) by trying both
    candidates = [jname] if kind != "weight" else [jname + ".scale", jname + ".kernel"]
    arr = None
    for c in candidates:
      if c in jax_p:
        arr = jax_p[c]; chosen = c; break
    if arr is None:
      # use_bias=False in the source => no bias param; equivalent to bias 0.
      if tkey.endswith(".bias"):
        sd[tkey] = torch.zeros_like(tparam)
        continue
      missing.append((tkey, candidates)); continue
    if bi is not None:
      arr = arr[bi]
    if chosen.endswith(".kernel") and arr.ndim == 2:
      arr = arr.T
    t = torch.tensor(np.array(arr, dtype=np.float32))
    if tuple(t.shape) != tuple(tparam.shape):
      missing.append((tkey, f"shape {tuple(t.shape)} != {tuple(tparam.shape)} from {chosen}")); continue
    sd[tkey] = t
  return sd, missing


def run_parity(loss, is_classifier):
  tag = "classification" if is_classifier else "regression"
  jm = JaxTabFM(loss=loss, use_fourier_features=True, feature_group=True,
                activation="swiglu",
                y_embedding_scheme=YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
                rngs=nnx.Rngs(0), dtype=jnp.float32, **CFG)
  jp = jax_params(jm)
  tm = MT.TabFM(is_classifier=is_classifier, **{k: v for k, v in CFG.items()}).eval().float()

  sd, missing = convert(jp, tm)
  print(f"[{tag}] converted {len(sd)} / "
        f"{len(list(tm.named_parameters()))+len(list(tm.named_buffers()))} tensors; "
        f"missing {len(missing)}")
  for mk in missing[:12]:
    print("  MISSING:", mk)
  if missing:
    return False
  tm.load_state_dict(sd, strict=True)

  B, T, H = 1, 12, 5
  rng = np.random.default_rng(0)
  x = rng.random((B, T, H)).astype(np.float32)
  if is_classifier:
    y = rng.integers(0, 3, (B, T)).astype(np.int32)
  else:
    y = rng.standard_normal((B, T)).astype(np.float32)
  ts = np.array([7], dtype=np.int32)
  jout = np.asarray(jm(jnp.asarray(x), jnp.asarray(y), train_size=jnp.asarray(ts)))
  with torch.no_grad():
    tout = tm(torch.tensor(x), torch.tensor(y), torch.tensor(ts)).numpy()
  diff = float(np.max(np.abs(jout - tout)))
  ok = diff < 1e-4
  print(f"[{tag}] shapes {jout.shape} {tout.shape}  max abs diff = {diff:.3e}"
        f"  -> {'PARITY' if ok else 'DIFF'}")
  return ok


def main():
  ok = [run_parity("cross_entropy", True), run_parity("rmse", False)]
  print("ALL OK" if all(ok) else "FAILED")


if __name__ == "__main__":
  main()

```

## File: tabfm/src/jax/BUILD

```
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tabfm:__subpackages__"])

py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    deps = [
        ":memory_efficient_attention",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//einops",
        "@pip//flax",
        "@pip//jax",
        "@pip//jaxtyping",
    ],
)

py_library(
    name = "checkpointing",
    srcs = ["checkpointing.py"],
    deps = [
        ":model",
        "@pip//flax",
        "@pip//importlib_resources",
        "@pip//orbax_checkpoint",
    ],
)

py_library(
    name = "memory_efficient_attention",
    srcs = ["memory_efficient_attention.py"],
    deps = [
        "@pip//jax",
        "@pip//jaxtyping",
    ],
)

py_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    shard_count = 50,
    deps = [
        ":model",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//flax",
        "@pip//jax",
        "@pip//numpy",
        "@pip//parameterized",
    ],
)

py_test(
    name = "memory_efficient_attention_test",
    srcs = ["memory_efficient_attention_test.py"],
    deps = [
        ":memory_efficient_attention",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//numpy",
    ],
)

py_test(
    name = "checkpointing_test",
    srcs = ["checkpointing_test.py"],
    deps = [
        ":checkpointing",
        ":model",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//flax",
        "@pip//importlib_resources",
        "@pip//jax",
        "@pip//optax",
    ],
)

py_library(
    name = "tabfm_v1_0_0",
    srcs = ["tabfm_v1_0_0.py"],
    deps = [
        ":checkpointing",
        ":model",
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//orbax_checkpoint",
    ],
)

```

## File: tabfm/src/jax/__init__.py

```
# tabfm src jax package

```

## File: tabfm/src/jax/checkpointing.py

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

import logging
import os
from typing import Any
from flax import nnx
import orbax.checkpoint as ocp


def create_checkpoint_manager(
    checkpoint_dir: str, max_checkpoints: int | None = None, read_only: bool = False
) -> ocp.CheckpointManager:
  if read_only:
    options = ocp.CheckpointManagerOptions(read_only=True)
    return ocp.CheckpointManager(
        os.path.join(checkpoint_dir, "orbax"), options=options
    )
  else:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints,
        create=True,
        async_options=ocp.options.AsyncOptions(timeout_secs=1800),
    )
    return ocp.CheckpointManager(
        os.path.join(checkpoint_dir, "orbax"),
        options=options,
    )


def save_checkpoint_state(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    state: nnx.ModelAndOptimizer,
    dataset_iter: Any | None,
):
  """Saves the training state using Orbax CheckpointManager."""
  logging.info("Saving checkpoint at step %d...", step)
  # The manager saves the step and handles the directory structure.
  _, params = nnx.split(state)
  save_args = {"params": ocp.args.StandardSave(params)}
  if dataset_iter is not None:
    if hasattr(dataset_iter, "get_state"):
      save_args["dataset_iter_state"] = ocp.args.JsonSave(
          dataset_iter.get_state()
      )
    else:
      logging.warning(
          "dataset_iter provided but has no get_state method. Iterator state"
          " will not be saved."
      )
  checkpoint_manager.save(step, args=ocp.args.Composite(**save_args))
  # It's good practice to wait for the save to complete.
  checkpoint_manager.wait_until_finished()
  logging.info("Checkpoint saved.")


def restore_checkpoint_state(
    checkpoint_manager: ocp.CheckpointManager,
    restore_step: int,
    abstract_params: Any,
    dataset_iter: Any | None = None,
):
  """Restores the training state from an Orbax checkpoint."""
  logging.info("Restoring checkpoint from step %d...", restore_step)

  restore_args = {
      "params": ocp.args.StandardRestore(abstract_params, strict=False),  # pytype: disable=wrong-keyword-args
  }
  if dataset_iter is not None:
    if hasattr(dataset_iter, "set_state"):
      restore_args["dataset_iter_state"] = ocp.args.JsonRestore()
    else:
      logging.warning(
          "dataset_iter provided but has no set_state method. Iterator state"
          " cannot be restored natively."
      )

  restored = checkpoint_manager.restore(
      restore_step, args=ocp.args.Composite(**restore_args)
  )

  if dataset_iter is not None:
    if hasattr(dataset_iter, "set_state") and "dataset_iter_state" in restored:
      dataset_iter.set_state(restored["dataset_iter_state"])
      restored_dataset_iter = dataset_iter
      logging.info("Restored state and iterator from step %d.", restore_step)
    else:
      restored_dataset_iter = dataset_iter
      logging.info("Restored state from step %d.", restore_step)
    return restored["params"], restored_dataset_iter
  else:
    logging.info("Restored state from step %d.", restore_step)
    return restored["params"], None


def load_model(checkpoint_dir: str, step: int | None = None) -> Any:
  """Loads a TabFM model from an Orbax checkpoint directory.

  This function expects:
  1. A 'config.json' file in `checkpoint_dir` containing the model architecture
     parameters (e.g., embed_dim, col_num_blocks, etc.).
  2. An Orbax checkpoint under `checkpoint_dir/orbax`.

  Args:
    checkpoint_dir: Path to the directory containing config.json and the orbax/
      folder.
    step: The checkpoint step to restore. If None, auto-detects the latest step.

  Returns:
    An instantiated TabFM model with restored parameters.
  """
  import json  # pylint: disable=g-import-not-at-top
  from tabfm.src.model import TabFM  # pylint: disable=g-import-not-at-top

  # 1. Load architecture config
  config_path = os.path.join(checkpoint_dir, "config.json")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Model configuration not found at {config_path}")

  with open(config_path, "r") as f:
    model_config = json.load(f)

  # 2. Instantiate TabFM with dummy RNGs (overwritten by checkpoint)
  rngs = nnx.Rngs(0)
  model = TabFM(rngs=rngs, **model_config)

  # 3. Create CheckpointManager
  checkpoint_manager = create_checkpoint_manager(checkpoint_dir, read_only=True)

  # 4. Determine step to restore
  if step is None:
    step = checkpoint_manager.latest_step()
    if step is None:
      raise ValueError(f"No checkpoints found in {checkpoint_dir}/orbax")

  # 5. Split state to get abstract params for restoration
  state = nnx.state(model)

  # 6. Restore parameters
  restored = checkpoint_manager.restore(
      step,
      args=ocp.args.Composite(
          params=ocp.args.StandardRestore(state, strict=False)
      ),
  )

  # 7. Update model with restored state
  nnx.update(model, restored["params"])

  return model

```

## File: tabfm/src/jax/checkpointing_test.py

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

import unittest

import chex
from flax import nnx
import jax.numpy as jnp
import optax
from tabfm.src.jax import checkpointing
from tabfm.src.jax import model as tabfm_model

from absl.testing import absltest
from absl import flags
import sys
if not flags.FLAGS.is_parsed():
  flags.FLAGS(sys.argv[:1])


class DummyDatasetIter:

  def __init__(self, data=0):
    self.data = data

  def __next__(self):
    val = self.data
    self.data += 1
    return {"data": jnp.array(val)}

  def get_state(self):
    return {"data": self.data}

  def set_state(self, state):
    self.data = state["data"]


class CheckpointingTest(unittest.IsolatedAsyncioTestCase, absltest.TestCase):

  def _create_dummy_state(self, rngs):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
        rngs=rngs,
    )
    tx = optax.adam(1e-3)
    return nnx.ModelAndOptimizer(model, tx)

  async def test_checkpointing(self):
    rngs = nnx.Rngs(0)
    state = self._create_dummy_state(rngs)

    # Create dummy dataset iterator with get_state / set_state
    dataset_iter = DummyDatasetIter()
    self.assertEqual(next(dataset_iter)["data"], 0)  # consume 0

    # Save checkpoint
    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = checkpointing.create_checkpoint_manager(
        checkpoint_dir, max_checkpoints=1
    )

    checkpointing.save_checkpoint_state(checkpoint_manager, 1, state, dataset_iter)
    self.assertTrue((checkpoint_manager.directory / '1').exists())

    # Restore checkpoint
    rngs_new = nnx.Rngs(1) # use different rngs to ensure state is overwritten
    state_to_restore = self._create_dummy_state(rngs_new)
    dataset_iter_to_restore = DummyDatasetIter()  # new iterator starting from 0

    restored_params, restored_iter = checkpointing.restore_checkpoint_state(
        checkpoint_manager,
        1,
        nnx.eval_shape(lambda: nnx.state(state_to_restore)),
        dataset_iter_to_restore,
    )
    nnx.update(state_to_restore, restored_params)

    # Check that restored state params are same as original
    chex.assert_trees_all_equal(nnx.state(state), nnx.state(state_to_restore))

    # Check that iterator is restored to correct position
    self.assertEqual(next(restored_iter)["data"], 1)


if __name__ == "__main__":
  absltest.main()

```

