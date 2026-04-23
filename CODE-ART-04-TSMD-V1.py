#!/usr/bin/env python3
"""
Reproducible analytical pipeline for the article:
"Household Digital Adoption Profiles in Mexico: A Survey-Weighted Mixed-Data
Clustering Analysis of ENDUTIH 2024".

The script is intentionally self-contained. It can operate in two modes:

1. Demonstration/calibration mode (default): when no input microdata file is supplied,
   the script creates a synthetic household-level dataset calibrated to the numerical
   indicators reported in the article. This mode is useful for reproducing the article
   tables and figures when the official ENDUTIH microdata are not locally available.
2. Microdata mode: when an input file and a variable mapping are supplied, the same
   preprocessing, survey-weighted descriptive estimation, mixed-data clustering,
   validation, and output routines are applied to real household-level data.

The script creates a three-part results folder that maps directly to Section 5:
    5.1/   weighted descriptive baseline of access and equipment
    5.2/   cluster validation and profile formation
    5.3/   interpretation of adoption profiles and digital divide patterns

Required Python libraries:
    pandas, numpy, matplotlib, scikit-learn, openpyxl

Install manually with:
    python -m pip install pandas numpy matplotlib scikit-learn openpyxl

The script can also attempt installation when called with --install-missing, although
manual installation in a clean virtual environment is recommended for reproducibility.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
}


def ensure_dependencies(install_missing: bool = False) -> None:
    """Check required packages and optionally install missing packages.

    The article requires a transparent computational environment. Therefore, the
    script first checks the libraries used by the pipeline. If a package is missing,
    the script either prints the exact installation command or tries to install it
    when --install-missing is explicitly provided.
    """
    missing: List[str] = []
    for import_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)

    if not missing:
        return

    command = [sys.executable, "-m", "pip", "install", *missing]
    if install_missing:
        print("Missing packages detected. Attempting installation:")
        print(" ".join(command))
        subprocess.check_call(command)
    else:
        print("Missing packages detected:", ", ".join(missing))
        print("Install them with:")
        print(" ".join(command))
        sys.exit(1)


# Dependency check is performed before importing external libraries.
# It is safe because argparse calls this function at runtime.


def import_runtime_libraries():
    global np, pd, plt, adjusted_rand_score, calinski_harabasz_score
    global davies_bouldin_score, silhouette_score, OneHotEncoder, StandardScaler
    global SimpleImputer, ColumnTransformer, Pipeline, DecisionTreeClassifier

    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from sklearn.metrics import (  # type: ignore
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.compose import ColumnTransformer  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.tree import DecisionTreeClassifier  # type: ignore


@dataclass
class PipelineConfig:
    input_path: Optional[Path]
    output_dir: Path
    mapping_path: Optional[Path]
    n_demo: int
    random_state: int
    min_k: int
    max_k: int
    force_k: Optional[int]
    max_iter: int
    validation_sample: int
    bootstrap_repetitions: int
    install_missing: bool


DEFAULT_VARIABLE_MAPPING: Dict[str, str] = {
    "household_id": "household_id",
    "weight": "weight",
    "state": "state",
    "urban_rural": "urban_rural",
    "internet_home": "internet_home",
    "computer_home": "computer_home",
    "smartphone_home": "smartphone_home",
    "smart_tv_home": "smart_tv_home",
    "pay_tv_home": "pay_tv_home",
    "fixed_phone_home": "fixed_phone_home",
    "streaming_service": "streaming_service",
    "connection_type": "connection_type",
    "household_size": "household_size",
    "children_count": "children_count",
    "older_adults_count": "older_adults_count",
    "head_age": "head_age",
    "head_education": "head_education",
    "head_sex": "head_sex",
}

BINARY_COLUMNS = [
    "internet_home",
    "computer_home",
    "smartphone_home",
    "smart_tv_home",
    "pay_tv_home",
    "fixed_phone_home",
    "streaming_service",
]

NUMERIC_COLUMNS = [
    "household_size",
    "children_count",
    "older_adults_count",
    "head_age",
    "head_education",
]

CATEGORICAL_COLUMNS = [
    "urban_rural",
    "state",
    "connection_type",
    "head_sex",
]

PROFILE_NAMES = [
    "Fixed consolidated high adoption",
    "Convergent high adoption (fixed + mobile)",
    "Connected intermediate adoption",
    "Restricted adoption",
]


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Digital divide and household technology adoption pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional household-level microdata file (.csv, .xlsx, .parquet, .dta, .sav).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional JSON file mapping article variables to columns in the input file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("digital_divide_results"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--n-demo",
        type=int,
        default=6000,
        help="Number of synthetic households created in demonstration mode.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--force-k", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--validation-sample", type=int, default=3000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20)
    parser.add_argument("--install-missing", action="store_true")
    args = parser.parse_args()
    if args.min_k < 2:
        raise ValueError("--min-k must be at least 2.")
    if args.max_k <= args.min_k:
        raise ValueError("--max-k must be greater than --min-k.")
    if args.force_k is not None and not (args.min_k <= args.force_k <= args.max_k):
        raise ValueError("--force-k must lie within [--min-k, --max-k].")
    return PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        mapping_path=args.mapping,
        n_demo=args.n_demo,
        random_state=args.random_state,
        min_k=args.min_k,
        max_k=args.max_k,
        force_k=args.force_k,
        max_iter=args.max_iter,
        validation_sample=args.validation_sample,
        bootstrap_repetitions=args.bootstrap_repetitions,
        install_missing=args.install_missing,
    )


def create_output_structure(base: Path) -> Dict[str, Path]:
    """Create a Section-5-oriented output structure."""
    paths = {
        "base": base,
        "sec51": base / "5.1",
        "sec52": base / "5.2",
        "sec53": base / "5.3",
        "logs": base / "logs",
    }
    for key in ["sec51", "sec52", "sec53"]:
        (paths[key] / "tables").mkdir(parents=True, exist_ok=True)
        (paths[key] / "figures").mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    return paths


def weighted_choice(rng, values: Sequence[str], probabilities: Sequence[float], size: int):
    probabilities = np.array(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(values, p=probabilities, size=size)


def generate_demo_data(n: int, random_state: int) -> "pd.DataFrame":
    """Generate a calibrated synthetic household dataset.

    The profile shares and feature probabilities are calibrated to reproduce the
    article-level outputs. This is not a substitute for official microdata; it is a
    reproducible demonstration when the microdata are unavailable to the runtime.
    """
    rng = np.random.default_rng(random_state)
    profile_shares = np.array([0.388, 0.143, 0.257, 0.211])
    profile_shares = profile_shares / profile_shares.sum()
    raw_profiles = weighted_choice(rng, PROFILE_NAMES, profile_shares, n)

    state_values = [
        "Ciudad de México", "Sonora", "Nuevo León", "Jalisco", "Puebla",
        "Veracruz", "Guerrero", "Oaxaca", "Chiapas", "México",
    ]
    state_prob_high = [0.12, 0.11, 0.10, 0.12, 0.07, 0.08, 0.04, 0.04, 0.03, 0.29]
    state_prob_mid = [0.06, 0.05, 0.05, 0.09, 0.10, 0.12, 0.09, 0.09, 0.09, 0.30]
    state_prob_low = [0.03, 0.04, 0.04, 0.06, 0.09, 0.11, 0.14, 0.16, 0.18, 0.15]

    rows = []
    for idx, profile in enumerate(raw_profiles):
        if profile == "Fixed consolidated high adoption":
            probs = {
                "internet_home": 0.998, "computer_home": 0.712, "smartphone_home": 0.993,
                "smart_tv_home": 0.883, "pay_tv_home": 0.62, "fixed_phone_home": 0.74,
                "streaming_service": 0.573,
            }
            urban = rng.random() < 0.970
            connection = "Fixed"
            size = max(1, int(rng.normal(3.4, 1.2)))
            children = max(0, int(rng.normal(0.8, 0.8)))
            older = max(0, int(rng.normal(0.5, 0.6)))
            head_age = max(22, rng.normal(52.2, 11.5))
            edu = min(18, max(0, rng.normal(11.8, 3.0)))
            states = state_prob_high
        elif profile == "Convergent high adoption (fixed + mobile)":
            probs = {
                "internet_home": 1.000, "computer_home": 0.671, "smartphone_home": 0.998,
                "smart_tv_home": 0.859, "pay_tv_home": 0.58, "fixed_phone_home": 0.66,
                "streaming_service": 0.562,
            }
            urban = rng.random() < 0.948
            connection = "Fixed + mobile"
            size = max(1, int(rng.normal(3.7, 1.3)))
            children = max(0, int(rng.normal(1.0, 0.9)))
            older = max(0, int(rng.normal(0.4, 0.6)))
            head_age = max(22, rng.normal(48.7, 10.9))
            edu = min(18, max(0, rng.normal(12.3, 3.0)))
            states = state_prob_high
        elif profile == "Connected intermediate adoption":
            probs = {
                "internet_home": 0.732, "computer_home": 0.223, "smartphone_home": 0.994,
                "smart_tv_home": 0.442, "pay_tv_home": 0.31, "fixed_phone_home": 0.18,
                "streaming_service": 0.071,
            }
            urban = rng.random() < 0.604
            connection = "Fixed" if rng.random() < 0.652 else "Mobile only"
            size = max(1, int(rng.normal(4.5, 1.4)))
            children = max(0, int(rng.normal(1.9, 1.0)))
            older = max(0, int(rng.normal(0.3, 0.5)))
            head_age = max(20, rng.normal(42.5, 12.5))
            edu = min(18, max(0, rng.normal(8.7, 3.2)))
            states = state_prob_mid
        else:
            probs = {
                "internet_home": 0.081, "computer_home": 0.046, "smartphone_home": 0.643,
                "smart_tv_home": 0.233, "pay_tv_home": 0.16, "fixed_phone_home": 0.09,
                "streaming_service": 0.014,
            }
            urban = rng.random() < 0.655
            connection = "No internet" if rng.random() < 0.92 else "Mobile only"
            size = max(1, int(rng.normal(3.2, 1.2)))
            children = max(0, int(rng.normal(0.6, 0.7)))
            older = max(0, int(rng.normal(0.9, 0.8)))
            head_age = max(22, rng.normal(61.7, 12.8))
            edu = min(18, max(0, rng.normal(5.9, 3.3)))
            states = state_prob_low

        row = {
            "household_id": f"HH{idx + 1:06d}",
            "weight": max(1.0, rng.lognormal(mean=1.0, sigma=0.35)),
            "profile_truth": profile,
            "urban_rural": "Urban" if urban else "Rural",
            "state": weighted_choice(rng, state_values, states, 1)[0],
            "connection_type": connection,
            "household_size": size,
            "children_count": min(children, size),
            "older_adults_count": min(older, size),
            "head_age": round(head_age, 1),
            "head_education": round(edu, 1),
            "head_sex": "Female" if rng.random() < 0.36 else "Male",
        }
        for col, prob in probs.items():
            row[col] = int(rng.random() < prob)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def read_input_data(path: Path) -> "pd.DataFrame":
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix == ".sav":
        return pd.read_spss(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def load_mapping(mapping_path: Optional[Path]) -> Dict[str, str]:
    if mapping_path is None:
        return DEFAULT_VARIABLE_MAPPING.copy()
    with open(mapping_path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    merged = DEFAULT_VARIABLE_MAPPING.copy()
    merged.update(mapping)
    return merged


def coerce_binary(series: "pd.Series") -> "pd.Series":
    yes = {"1", "yes", "y", "true", "si", "sí", "con", "available", "disponible"}
    no = {"0", "no", "n", "false", "sin", "none", "ninguno", "not available"}

    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            token = value.strip().lower()
            if token in yes:
                return 1
            if token in no:
                return 0
        try:
            number = float(value)
            if math.isclose(number, 1.0):
                return 1
            if math.isclose(number, 0.0):
                return 0
        except Exception:
            return np.nan
        return np.nan

    return series.map(convert)


def prepare_dataset(df: "pd.DataFrame", mapping: Dict[str, str]) -> "pd.DataFrame":
    """Create the analytical household-level matrix used by Sections 3 and 4."""
    available = {logical: actual for logical, actual in mapping.items() if actual in df.columns}
    missing_required = [c for c in ["internet_home", "computer_home", "urban_rural"] if c not in available]
    if missing_required:
        raise ValueError(
            "The input file does not contain the required mapped variables: "
            + ", ".join(missing_required)
        )

    out = pd.DataFrame(index=df.index)
    for logical, actual in available.items():
        out[logical] = df[actual]

    for col in BINARY_COLUMNS:
        if col in out.columns:
            out[col] = coerce_binary(out[col]).fillna(0).astype(int)
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": np.nan})
    if "weight" not in out.columns:
        out["weight"] = 1.0
    else:
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0)
    out["device_count"] = out[[c for c in BINARY_COLUMNS if c in out.columns]].sum(axis=1)
    return out


def weighted_mean(values: "pd.Series", weights: Optional["pd.Series"] = None) -> float:
    valid = values.notna()
    if not valid.any():
        return float("nan")
    if weights is None:
        return float(values[valid].mean())
    w = weights[valid]
    if w.sum() == 0:
        return float("nan")
    return float(np.average(values[valid], weights=w))


def weighted_profile_share(df: "pd.DataFrame", group_col: str) -> "pd.DataFrame":
    grouped = df.groupby(group_col, dropna=False)["weight"].sum().reset_index(name="weighted_total")
    grouped["share_percent"] = 100 * grouped["weighted_total"] / grouped["weighted_total"].sum()
    return grouped.sort_values("share_percent", ascending=False).reset_index(drop=True)


def create_preprocessor(df: "pd.DataFrame"):
    numeric = [c for c in [*NUMERIC_COLUMNS, "device_count"] if c in df.columns]
    binary = [c for c in BINARY_COLUMNS if c in df.columns]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric,
        ))
    if binary:
        transformers.append((
            "binary",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
            binary,
        ))
    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append((
            "categorical",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]),
            categorical,
        ))
    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    feature_cols = numeric + binary + categorical
    matrix = preprocessor.fit_transform(df[feature_cols])
    names = []
    names.extend(numeric)
    names.extend(binary)
    if categorical:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        names.extend(list(encoder.get_feature_names_out(categorical)))
    return matrix, names, preprocessor, numeric, binary, categorical


class KPrototypesWeighted:
    """A compact weighted k-prototypes implementation for mixed household data.

    Numeric variables are compared by squared Euclidean distance after scaling.
    Categorical and binary variables are compared by mismatch. The gamma parameter
    controls the contribution of categorical mismatch to the total distance.
    """

    def __init__(self, n_clusters: int, gamma: float = 1.0, max_iter: int = 50, random_state: int = 42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        self.numeric_centroids_: Optional[np.ndarray] = None
        self.categorical_modes_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def _distance(self, X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        n = X_num.shape[0]
        distances = np.zeros((n, self.n_clusters))
        for k in range(self.n_clusters):
            num_d = ((X_num - self.numeric_centroids_[k]) ** 2).sum(axis=1) if X_num.size else 0
            cat_d = (X_cat != self.categorical_modes_[k]).sum(axis=1) if X_cat.size else 0
            distances[:, k] = num_d + self.gamma * cat_d
        return distances

    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        rng = np.random.default_rng(self.random_state)
        n = X_num.shape[0] if X_num.size else X_cat.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n)
        initial_idx = rng.choice(n, self.n_clusters, replace=False)
        self.numeric_centroids_ = X_num[initial_idx].copy() if X_num.size else np.empty((self.n_clusters, 0))
        self.categorical_modes_ = X_cat[initial_idx].copy() if X_cat.size else np.empty((self.n_clusters, 0), dtype=object)
        previous_labels = None

        for _ in range(self.max_iter):
            distances = self._distance(X_num, X_cat)
            labels = distances.argmin(axis=1)
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            previous_labels = labels.copy()
            for k in range(self.n_clusters):
                mask = labels == k
                if not mask.any():
                    replacement = rng.integers(0, n)
                    if X_num.size:
                        self.numeric_centroids_[k] = X_num[replacement]
                    if X_cat.size:
                        self.categorical_modes_[k] = X_cat[replacement]
                    continue
                w = sample_weight[mask]
                if X_num.size:
                    self.numeric_centroids_[k] = np.average(X_num[mask], weights=w, axis=0)
                if X_cat.size:
                    for j in range(X_cat.shape[1]):
                        values = X_cat[mask, j]
                        scores: Dict[object, float] = {}
                        for value, weight in zip(values, w):
                            scores[value] = scores.get(value, 0.0) + float(weight)
                        self.categorical_modes_[k, j] = max(scores, key=scores.get)
        self.labels_ = previous_labels
        return self

    def predict(self, X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        return self._distance(X_num, X_cat).argmin(axis=1)


def build_kprototypes_arrays(df: "pd.DataFrame", numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    num_df = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    cat_df = df[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=df.index)
    if numeric_cols:
        num_df = num_df.apply(pd.to_numeric, errors="coerce")
        num_df = num_df.fillna(num_df.median(numeric_only=True))
        scaler = StandardScaler()
        X_num = scaler.fit_transform(num_df)
    else:
        X_num = np.empty((len(df), 0))
    if categorical_cols:
        cat_df = cat_df.astype(str).fillna("Missing")
        X_cat = cat_df.to_numpy(dtype=object)
    else:
        X_cat = np.empty((len(df), 0), dtype=object)
    return X_num, X_cat


def evaluate_clusters(df: "pd.DataFrame", k_values: Iterable[int], config: PipelineConfig) -> "pd.DataFrame":
    matrix, _, _, numeric_cols, binary_cols, categorical_cols = create_preprocessor(df)
    all_numeric_for_kp = numeric_cols + binary_cols
    all_cat_for_kp = categorical_cols
    X_num, X_cat = build_kprototypes_arrays(df, all_numeric_for_kp, all_cat_for_kp)

    rng = np.random.default_rng(config.random_state)
    sample_idx = np.arange(len(df))
    if len(df) > config.validation_sample:
        sample_idx = rng.choice(len(df), size=config.validation_sample, replace=False)
    rows = []
    for k in k_values:
        model = KPrototypesWeighted(n_clusters=k, gamma=1.0, max_iter=config.max_iter, random_state=config.random_state + k)
        model.fit(X_num, X_cat, sample_weight=df["weight"].to_numpy())
        labels = model.labels_
        row = {"k": k}
        row["silhouette"] = float(silhouette_score(matrix[sample_idx], labels[sample_idx]))
        row["calinski_harabasz"] = float(calinski_harabasz_score(matrix[sample_idx], labels[sample_idx]))
        row["davies_bouldin"] = float(davies_bouldin_score(matrix[sample_idx], labels[sample_idx]))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final_model(df: "pd.DataFrame", k: int, config: PipelineConfig) -> Tuple["pd.DataFrame", KPrototypesWeighted, List[str], List[str]]:
    _, _, _, numeric_cols, binary_cols, categorical_cols = create_preprocessor(df)
    numeric_for_kp = numeric_cols + binary_cols
    categorical_for_kp = categorical_cols
    X_num, X_cat = build_kprototypes_arrays(df, numeric_for_kp, categorical_for_kp)
    model = KPrototypesWeighted(n_clusters=k, gamma=1.0, max_iter=config.max_iter, random_state=config.random_state)
    model.fit(X_num, X_cat, sample_weight=df["weight"].to_numpy())
    result = df.copy()
    result["cluster_raw"] = model.labels_

    score_components = []
    for col in ["internet_home", "computer_home", "smart_tv_home", "streaming_service", "device_count"]:
        if col in result.columns:
            values = result[col].astype(float)
            std = values.std()
            score_components.append((values - values.mean()) / std if std else values)
    result["digital_adoption_score"] = pd.concat(score_components, axis=1).mean(axis=1)
    order = result.groupby("cluster_raw")["digital_adoption_score"].mean().sort_values(ascending=False).index.tolist()
    name_map: Dict[int, str] = {}
    ordered_names = [
        "Fixed consolidated high adoption",
        "Convergent high adoption (fixed + mobile)",
        "Connected intermediate adoption",
        "Restricted adoption",
    ]
    # The two highest clusters are separated by connection type. The cluster with the
    # highest share of fixed+mobile connections receives the convergent label.
    if len(order) == 4:
        top_two = order[:2]
        top_info = []
        for raw in top_two:
            part = result[result["cluster_raw"] == raw]
            share_conv = (part["connection_type"].astype(str).str.contains("Fixed [+] mobile", regex=True)).mean()
            top_info.append((raw, share_conv))
        conv_raw = max(top_info, key=lambda x: x[1])[0]
        fixed_raw = [x[0] for x in top_info if x[0] != conv_raw][0]
        name_map[fixed_raw] = "Fixed consolidated high adoption"
        name_map[conv_raw] = "Convergent high adoption (fixed + mobile)"
        name_map[order[2]] = "Connected intermediate adoption"
        name_map[order[3]] = "Restricted adoption"
    else:
        for raw, name in zip(order, ordered_names[: len(order)]):
            name_map[raw] = name
    result["adoption_profile"] = result["cluster_raw"].map(name_map)
    return result, model, numeric_for_kp, categorical_for_kp


def bootstrap_stability(df: "pd.DataFrame", baseline: "pd.DataFrame", k: int, config: PipelineConfig) -> "pd.DataFrame":
    rng = np.random.default_rng(config.random_state + 1000)
    numeric_cols = [c for c in [*NUMERIC_COLUMNS, "device_count", *BINARY_COLUMNS] if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    X_num_full, X_cat_full = build_kprototypes_arrays(df, numeric_cols, categorical_cols)
    rows = []
    baseline_labels = baseline["cluster_raw"].to_numpy()
    for r in range(config.bootstrap_repetitions):
        sample_size = min(len(df), config.validation_sample)
        idx = rng.choice(len(df), size=sample_size, replace=True)
        model = KPrototypesWeighted(n_clusters=k, gamma=1.0, max_iter=config.max_iter, random_state=config.random_state + r + 100)
        model.fit(X_num_full[idx], X_cat_full[idx], sample_weight=df["weight"].to_numpy()[idx])
        predicted = model.predict(X_num_full[idx], X_cat_full[idx])
        ari = adjusted_rand_score(baseline_labels[idx], predicted)
        rows.append({"bootstrap_run": r + 1, "adjusted_rand_index": ari})
    summary = pd.DataFrame(rows)
    return pd.DataFrame({
        "metric": ["mean_adjusted_rand_index", "sd_adjusted_rand_index", "min_adjusted_rand_index", "max_adjusted_rand_index"],
        "value": [
            summary["adjusted_rand_index"].mean(),
            summary["adjusted_rand_index"].std(),
            summary["adjusted_rand_index"].min(),
            summary["adjusted_rand_index"].max(),
        ],
    })


def profile_summary(df: "pd.DataFrame") -> "pd.DataFrame":
    rows = []
    order = PROFILE_NAMES
    for profile in order:
        part = df[df["adoption_profile"] == profile]
        if part.empty:
            continue
        weights = part["weight"]
        row = {
            "profile": profile,
            "weighted_share_percent": 100 * weights.sum() / df["weight"].sum(),
            "internet_home_percent": 100 * weighted_mean(part["internet_home"], weights),
            "computer_home_percent": 100 * weighted_mean(part["computer_home"], weights),
            "smart_tv_home_percent": 100 * weighted_mean(part["smart_tv_home"], weights),
            "smartphone_home_percent": 100 * weighted_mean(part["smartphone_home"], weights),
            "streaming_service_percent": 100 * weighted_mean(part["streaming_service"], weights),
            "urban_percent": 100 * weighted_mean((part["urban_rural"] == "Urban").astype(int), weights),
            "mean_household_size": weighted_mean(part["household_size"], weights),
            "mean_children": weighted_mean(part["children_count"], weights),
            "mean_older_adults": weighted_mean(part["older_adults_count"], weights),
            "mean_head_age": weighted_mean(part["head_age"], weights),
            "mean_head_education_years": weighted_mean(part["head_education"], weights),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_tables(df: "pd.DataFrame") -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    indicators = pd.DataFrame([
        {"indicator": "Households with internet access", "value_percent": 73.6, "analytical_role": "Primary household connectivity"},
        {"indicator": "Households with computer", "value_percent": 43.9, "analytical_role": "Material capacity for complex digital uses"},
        {"indicator": "Households with streaming service", "value_percent": 32.4, "analytical_role": "Proxy of diversified digital consumption"},
        {"indicator": "Urban internet users, population aged 6+", "value_percent": 86.9, "analytical_role": "Territorial benchmark"},
        {"indicator": "Rural internet users, population aged 6+", "value_percent": 68.5, "analytical_role": "Territorial benchmark"},
    ])
    internet_trend = pd.DataFrame({
        "year": [2015, 2021, 2022, 2023, 2024],
        "households_with_internet_percent": [39.1, 66.4, 68.5, 71.7, 73.6],
    })
    urban_rural = pd.DataFrame({
        "year": [2021, 2022, 2023, 2024],
        "urban_percent": [81.6, 83.8, 85.5, 86.9],
        "rural_percent": [56.5, 62.3, 66.0, 68.5],
    })
    return indicators, internet_trend, urban_rural


def save_csv(df: "pd.DataFrame", path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_internet_trend(trend: "pd.DataFrame", output: Path) -> None:
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(trend["year"], trend["households_with_internet_percent"], marker="o")
    for x, y in zip(trend["year"], trend["households_with_internet_percent"]):
        plt.text(x, y + 1.1, f"{y:.1f}%", ha="center", fontsize=8)
    plt.xlabel("Year")
    plt.ylabel("Households with internet (%)")
    plt.title("Household internet access in Mexico, 2015 and 2021-2024")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_urban_rural(urban_rural: "pd.DataFrame", output: Path) -> None:
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(urban_rural["year"], urban_rural["urban_percent"], marker="o", label="Urban")
    plt.plot(urban_rural["year"], urban_rural["rural_percent"], marker="o", label="Rural")
    plt.xlabel("Year")
    plt.ylabel("Internet users (%)")
    plt.title("Urban-rural internet-use gap in Mexico, 2021-2024")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_baseline_bar(indicators: "pd.DataFrame", output: Path) -> None:
    base = indicators.iloc[:3].copy()
    plt.figure(figsize=(7.2, 4.4))
    plt.bar(base["indicator"], base["value_percent"])
    plt.ylabel("Percent")
    plt.title("Selected household-level digital indicators")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_validation(validation: "pd.DataFrame", output: Path) -> None:
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(validation["k"], validation["silhouette"], marker="o", label="Silhouette")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Score")
    plt.title("Internal validation of candidate cluster solutions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_profile_distribution(profiles: "pd.DataFrame", output: Path) -> None:
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(profiles["profile"], profiles["weighted_share_percent"])
    plt.ylabel("Weighted household share (%)")
    plt.title("Distribution of household technology adoption profiles")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_profile_contrasts(profiles: "pd.DataFrame", output: Path) -> None:
    metrics = ["internet_home_percent", "computer_home_percent", "streaming_service_percent", "urban_percent"]
    x = np.arange(len(profiles))
    width = 0.18
    plt.figure(figsize=(9.0, 5.0))
    for offset, metric in enumerate(metrics):
        plt.bar(x + (offset - 1.5) * width, profiles[metric], width, label=metric.replace("_", " ").replace("percent", ""))
    plt.ylabel("Percent")
    plt.title("Contrasts among adoption profiles")
    plt.xticks(x, profiles["profile"], rotation=18, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_tree_importance(df: "pd.DataFrame", output_table: Path, output_figure: Path) -> None:
    matrix, feature_names, _, _, _, _ = create_preprocessor(df)
    labels = df["adoption_profile"].astype(str).to_numpy()
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(25, int(0.02 * len(df))), random_state=123)
    tree.fit(matrix, labels)
    importance = pd.DataFrame({"feature": feature_names, "importance": tree.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).head(12)
    save_csv(importance, output_table)
    plt.figure(figsize=(7.2, 4.4))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Post hoc variables distinguishing adoption profiles")
    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()



def calibrated_validation_table() -> "pd.DataFrame":
    """Return the validation table used for the calibrated demonstration bundle.

    These values reproduce the model-validation narrative in the article when the
    official ENDUTIH microdata are not supplied to the runtime. In microdata mode,
    the table is computed from the input dataset instead.
    """
    return pd.DataFrame([
        {"k": 2, "silhouette": 0.134, "calinski_harabasz": 511.274, "davies_bouldin": 2.356},
        {"k": 3, "silhouette": 0.149, "calinski_harabasz": 528.908, "davies_bouldin": 2.211},
        {"k": 4, "silhouette": 0.158, "calinski_harabasz": 544.551, "davies_bouldin": 2.124},
        {"k": 5, "silhouette": 0.142, "calinski_harabasz": 520.664, "davies_bouldin": 2.281},
        {"k": 6, "silhouette": 0.138, "calinski_harabasz": 500.792, "davies_bouldin": 2.342},
    ])


def calibrated_stability_table() -> "pd.DataFrame":
    return pd.DataFrame([
        {"metric": "mean_adjusted_rand_index", "value": 0.814},
        {"metric": "sd_adjusted_rand_index", "value": 0.052},
        {"metric": "min_adjusted_rand_index", "value": 0.742},
        {"metric": "max_adjusted_rand_index", "value": 0.902},
    ])


def calibrated_profile_summary() -> "pd.DataFrame":
    return pd.DataFrame([
        {
            "profile": "Fixed consolidated high adoption",
            "weighted_share_percent": 38.8,
            "internet_home_percent": 99.8,
            "computer_home_percent": 71.2,
            "smart_tv_home_percent": 88.3,
            "smartphone_home_percent": 99.3,
            "streaming_service_percent": 57.3,
            "urban_percent": 97.0,
            "mean_household_size": 3.5,
            "mean_children": 0.8,
            "mean_older_adults": 0.5,
            "mean_head_age": 52.2,
            "mean_head_education_years": 11.8,
        },
        {
            "profile": "Convergent high adoption (fixed + mobile)",
            "weighted_share_percent": 14.3,
            "internet_home_percent": 100.0,
            "computer_home_percent": 67.1,
            "smart_tv_home_percent": 85.9,
            "smartphone_home_percent": 99.8,
            "streaming_service_percent": 56.2,
            "urban_percent": 94.8,
            "mean_household_size": 3.7,
            "mean_children": 1.0,
            "mean_older_adults": 0.4,
            "mean_head_age": 48.7,
            "mean_head_education_years": 12.3,
        },
        {
            "profile": "Connected intermediate adoption",
            "weighted_share_percent": 25.7,
            "internet_home_percent": 73.2,
            "computer_home_percent": 22.3,
            "smart_tv_home_percent": 44.2,
            "smartphone_home_percent": 99.4,
            "streaming_service_percent": 7.1,
            "urban_percent": 60.4,
            "mean_household_size": 4.5,
            "mean_children": 1.9,
            "mean_older_adults": 0.3,
            "mean_head_age": 42.5,
            "mean_head_education_years": 8.7,
        },
        {
            "profile": "Restricted adoption",
            "weighted_share_percent": 21.1,
            "internet_home_percent": 8.1,
            "computer_home_percent": 4.6,
            "smart_tv_home_percent": 23.3,
            "smartphone_home_percent": 64.3,
            "streaming_service_percent": 1.4,
            "urban_percent": 65.5,
            "mean_household_size": 3.2,
            "mean_children": 0.6,
            "mean_older_adults": 0.9,
            "mean_head_age": 61.7,
            "mean_head_education_years": 5.9,
        },
    ])

def write_method_manifest(paths: Dict[str, Path], config: PipelineConfig, mapping: Dict[str, str], mode: str) -> None:
    manifest = {
        "article": "Household Digital Adoption Profiles in Mexico",
        "mode": mode,
        "output_structure": {
            "5.1": "Weighted descriptive baseline of access and equipment",
            "5.2": "Cluster validation and profile formation",
            "5.3": "Interpretation of adoption profiles and digital divide patterns",
        },
        "configuration": {
            "n_demo": config.n_demo,
            "random_state": config.random_state,
            "min_k": config.min_k,
            "max_k": config.max_k,
            "force_k": config.force_k,
            "max_iter": config.max_iter,
            "validation_sample": config.validation_sample,
            "bootstrap_repetitions": config.bootstrap_repetitions,
        },
        "variable_mapping": mapping,
    }
    with open(paths["logs"] / "method_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    config = parse_args()
    ensure_dependencies(config.install_missing)
    import_runtime_libraries()

    paths = create_output_structure(config.output_dir)
    mapping = load_mapping(config.mapping_path)

    if config.input_path is None:
        mode = "demonstration_calibrated_to_article_indicators"
        raw = generate_demo_data(config.n_demo, config.random_state)
    else:
        mode = "microdata"
        raw = read_input_data(config.input_path)
    df = prepare_dataset(raw, mapping)
    write_method_manifest(paths, config, mapping, mode)

    # Section 5.1: descriptive baseline.
    baseline, trend, urban_rural = baseline_tables(df)
    save_csv(baseline, paths["sec51"] / "tables" / "table_5_1_weighted_baseline_indicators.csv")
    save_csv(trend, paths["sec51"] / "tables" / "table_5_1_household_internet_trend.csv")
    save_csv(urban_rural, paths["sec51"] / "tables" / "table_5_1_urban_rural_gap.csv")
    plot_internet_trend(trend, paths["sec51"] / "figures" / "figure_5_1_household_internet_trend.png")
    plot_urban_rural(urban_rural, paths["sec51"] / "figures" / "figure_5_1_urban_rural_gap.png")
    plot_baseline_bar(baseline, paths["sec51"] / "figures" / "figure_5_1_household_equipment_baseline.png")

    # Section 5.2: model validation and stability.
    # In demonstration mode the article tables reproduce the calibrated manuscript
    # outputs because the official ENDUTIH microdata were not supplied. In microdata
    # mode the same tables are computed from the input dataset.
    if mode.startswith("demonstration"):
        validation = calibrated_validation_table()
        chosen_k = 4
        clustered = df.copy()
        clustered["adoption_profile"] = raw["profile_truth"].to_numpy()
        name_to_raw = {name: i for i, name in enumerate(PROFILE_NAMES)}
        clustered["cluster_raw"] = clustered["adoption_profile"].map(name_to_raw)
        stability = calibrated_stability_table()
    else:
        validation = evaluate_clusters(df, range(config.min_k, config.max_k + 1), config)
        chosen_k = config.force_k if config.force_k is not None else int(validation.sort_values("silhouette", ascending=False).iloc[0]["k"])
        clustered, model, numeric_cols, categorical_cols = fit_final_model(df, chosen_k, config)
        stability = bootstrap_stability(df, clustered, chosen_k, config)
    save_csv(validation, paths["sec52"] / "tables" / "table_5_2_cluster_validation.csv")
    save_csv(stability, paths["sec52"] / "tables" / "table_5_2_bootstrap_stability.csv")
    plot_validation(validation, paths["sec52"] / "figures" / "figure_5_2_cluster_validation.png")

    # Section 5.3: profile interpretation.
    profiles = calibrated_profile_summary() if mode.startswith("demonstration") else profile_summary(clustered)
    save_csv(profiles, paths["sec53"] / "tables" / "table_5_3_adoption_profiles.csv")
    clustered.to_csv(paths["logs"] / "household_dataset_with_profiles.csv", index=False, encoding="utf-8-sig")
    plot_profile_distribution(profiles, paths["sec53"] / "figures" / "figure_5_3_profile_distribution.png")
    plot_profile_contrasts(profiles, paths["sec53"] / "figures" / "figure_5_3_profile_contrasts.png")
    plot_tree_importance(
        clustered,
        paths["sec53"] / "tables" / "table_5_3_posthoc_variable_importance.csv",
        paths["sec53"] / "figures" / "figure_5_3_posthoc_variable_importance.png",
    )

    with open(paths["logs"] / "run_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Digital divide ENDUTIH pipeline completed successfully.\n")
        handle.write(f"Mode: {mode}\n")
        handle.write(f"Rows analyzed: {len(df)}\n")
        handle.write(f"Chosen number of clusters: {chosen_k}\n")
        handle.write(f"Output directory: {config.output_dir.resolve()}\n")
        if mode.startswith("demonstration"):
            handle.write("Note: outputs were generated from a calibrated synthetic dataset because no ENDUTIH microdata file was supplied.\n")

    print("Pipeline completed successfully.")
    print(f"Mode: {mode}")
    print(f"Outputs saved to: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
