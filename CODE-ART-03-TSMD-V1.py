
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Early Customer Churn Detection in E-Commerce Platforms:
A Data Mining Framework for Temporal Risk Scoring and Retention Prioritization.

This script generates a privacy-preserving synthetic e-commerce churn benchmark,
trains four supervised classifiers under a chronological train-validation-test
protocol, optimizes an operational threshold on the validation split, evaluates
the models on the temporal test split, and exports article-ready tables and
figures into three result folders:

    results/5.1  Descriptive evidence from temporal splits and class profiles
    results/5.2  Comparative predictive performance under temporal validation
    results/5.3  Interpretability and retention prioritization

The code is intentionally self-contained. It checks the required libraries and
attempts to install missing dependencies before execution. All generated data,
tables, figures, and summary outputs are reproducible through the fixed random
seed.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
}


def ensure_required_libraries() -> None:
    """Verify required libraries and install missing packages when needed."""
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        print("Missing libraries detected:", ", ".join(missing))
        print("Attempting installation with pip. If this fails, run manually:")
        print(f"{sys.executable} -m pip install " + " ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_required_libraries()

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ExperimentConfig:
    """Global experiment configuration."""

    random_seed: int = 20260422
    n_customers: int = 15000
    risk_intercept: float = -4.0
    risk_scale: float = 4.0
    output_dir: Path = Path("ecommerce_churn_results")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transformation."""
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def generate_synthetic_ecommerce_churn_data(config: ExperimentConfig) -> pd.DataFrame:
    """
    Generate a synthetic customer-snapshot benchmark with transactional,
    navigational, marketing, payment, and service-quality signals.

    The latent risk mechanism deliberately links churn propensity to recency,
    service friction, discount dependence, browsing-to-purchase imbalance,
    weak engagement, and lower economic value. The output remains synthetic and
    does not contain personally identifiable information.
    """
    rng = np.random.default_rng(config.random_seed)
    n = config.n_customers

    snapshot_month = rng.integers(1, 19, n)

    customer_segment = rng.choice(
        ["Premium", "Loyal", "Standard", "Value-seeker", "New"],
        n,
        p=[0.12, 0.25, 0.36, 0.17, 0.10],
    )
    region = rng.choice(
        ["Central", "North", "South", "West", "Metropolitan"],
        n,
        p=[0.26, 0.20, 0.18, 0.18, 0.18],
    )
    device = rng.choice(["Desktop", "Mobile", "Tablet"], n, p=[0.36, 0.54, 0.10])
    acquisition_channel = rng.choice(
        ["Organic", "Paid search", "Marketplace", "Social", "Referral"],
        n,
        p=[0.30, 0.22, 0.20, 0.18, 0.10],
    )
    preferred_payment_method = rng.choice(
        ["Credit card", "Debit card", "Wallet", "Bank transfer", "Cash on delivery"],
        n,
        p=[0.38, 0.25, 0.20, 0.10, 0.07],
    )

    segment_effect = {
        "Premium": 0.60,
        "Loyal": 0.35,
        "Standard": 0.00,
        "Value-seeker": -0.25,
        "New": -0.45,
    }
    segment_score = np.vectorize(segment_effect.get)(customer_segment)

    tenure_months = np.clip(
        rng.gamma(shape=2.5, scale=10.0, size=n)
        + 15.0 * (customer_segment == "Premium")
        + 8.0 * (customer_segment == "Loyal")
        - 8.0 * (customer_segment == "New"),
        1,
        84,
    )

    purchase_frequency_90d = (
        rng.poisson(np.clip(2.2 + 1.4 * segment_score + 0.03 * tenure_months, 0.2, 8), n)
        + 1
    )

    discount_ratio = np.clip(
        rng.beta(1.5, 5.0, n)
        + 0.18 * (customer_segment == "Value-seeker")
        + 0.05 * (acquisition_channel == "Paid search"),
        0,
        0.95,
    )

    sessions_90d = np.clip(
        rng.poisson(np.clip(7.0 + 2.0 * purchase_frequency_90d + 1.5 * (customer_segment == "Premium"), 1, 30), n),
        1,
        None,
    )

    avg_session_depth = np.clip(
        rng.normal(
            5.0
            + 0.25 * purchase_frequency_90d
            + 0.40 * (customer_segment == "Premium")
            - 0.20 * (device == "Mobile"),
            1.2,
            n,
        ),
        1,
        15,
    )

    avg_order_value = np.clip(
        rng.lognormal(
            mean=3.1
            + 0.25 * (customer_segment == "Premium")
            + 0.10 * (customer_segment == "Loyal")
            - 0.15 * (customer_segment == "Value-seeker"),
            sigma=0.45,
            size=n,
        ),
        5,
        250,
    )

    monetary_value_90d = np.clip(
        avg_order_value * purchase_frequency_90d * (1 + rng.normal(0, 0.15, n)),
        0,
        None,
    )

    support_contacts_90d = np.clip(
        rng.poisson(
            np.clip(
                0.15
                + 0.30 * (acquisition_channel == "Marketplace")
                + 0.20 * (preferred_payment_method == "Cash on delivery")
                + 0.15 * (customer_segment == "New"),
                0.01,
                3,
            ),
            n,
        ),
        0,
        7,
    )

    complaints_180d = np.clip(
        rng.poisson(
            np.clip(
                0.08 + 0.30 * support_contacts_90d + 0.12 * (acquisition_channel == "Marketplace"),
                0.01,
                3,
            ),
            n,
        ),
        0,
        8,
    )

    delivery_delay_ratio = np.clip(
        rng.beta(1.5, 12.0, n)
        + 0.05 * (region == "South")
        + 0.05 * (acquisition_channel == "Marketplace"),
        0,
        0.8,
    )

    return_ratio = np.clip(
        rng.beta(1.0, 10.0, n) + 0.04 * complaints_180d + 0.02 * support_contacts_90d,
        0,
        0.8,
    )

    email_open_rate = np.clip(
        rng.beta(
            2.5
            + 0.50 * (customer_segment == "Loyal")
            + 0.80 * (customer_segment == "Premium"),
            4.0,
            n,
        )
        - 0.15 * (customer_segment == "New"),
        0,
        1,
    )

    push_click_rate = np.clip(
        rng.beta(1.5 + 0.40 * (customer_segment == "Loyal"), 8.0, n)
        + 0.05 * (device == "Mobile"),
        0,
        1,
    )

    catalog_diversity = np.clip(
        rng.poisson(
            np.clip(
                4.0
                + 0.50 * purchase_frequency_90d
                + 2.0 * (customer_segment == "Premium")
                + 1.0 * (customer_segment == "Loyal"),
                1,
                20,
            ),
            n,
        ),
        1,
        30,
    )

    nighttime_activity_ratio = np.clip(rng.beta(1.2, 8.0, n) + 0.05 * (device == "Mobile"), 0, 1)

    payment_incidents = np.clip(
        rng.poisson(
            0.05
            + 0.20 * (preferred_payment_method == "Cash on delivery")
            + 0.08 * (preferred_payment_method == "Bank transfer")
            + 0.05 * (customer_segment == "New"),
            n,
        ),
        0,
        4,
    )

    browsing_to_purchase_ratio = np.clip(
        (sessions_90d + 1) / (purchase_frequency_90d + 1)
        + rng.normal(0, 0.6, n)
        + 1.20 * (customer_segment == "New")
        + 0.50 * (customer_segment == "Value-seeker"),
        0.2,
        20,
    )

    last_basket_value = np.clip(avg_order_value * (1 + rng.normal(0, 0.30, n)), 1, 300)

    recency_days = np.clip(
        rng.gamma(shape=2.2, scale=12.0, size=n)
        + 7.0 * (customer_segment == "New")
        + 5.0 * (customer_segment == "Value-seeker")
        - 4.0 * (customer_segment == "Premium")
        - 2.0 * purchase_frequency_90d
        + 5.0 * support_contacts_90d
        + 18.0 * rng.binomial(1, 0.08, n),
        1,
        180,
    )

    raw_risk = (
        0.045 * recency_days
        - 0.280 * purchase_frequency_90d
        - 0.004 * monetary_value_90d
        + 0.480 * support_contacts_90d
        + 0.320 * complaints_180d
        + 1.600 * discount_ratio
        + 1.100 * delivery_delay_ratio
        + 0.900 * return_ratio
        + 0.250 * payment_incidents
        + 0.060 * browsing_to_purchase_ratio
        - 1.300 * email_open_rate
        - 0.700 * push_click_rate
        - 0.110 * catalog_diversity
        - 0.011 * tenure_months
        - 0.350 * (customer_segment == "Premium")
        - 0.180 * (customer_segment == "Loyal")
        + 0.350 * (customer_segment == "Value-seeker")
        + 0.180 * (customer_segment == "New")
        + 0.140 * (acquisition_channel == "Marketplace")
        + 0.080 * (device == "Mobile")
        + 0.025 * (snapshot_month - 9)
    )

    latent_churn_probability = sigmoid(config.risk_intercept + config.risk_scale * raw_risk)
    churn_label = rng.binomial(1, latent_churn_probability)

    split = np.where(snapshot_month <= 12, "train", np.where(snapshot_month <= 15, "validation", "test"))

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "snapshot_month": snapshot_month,
            "customer_segment": customer_segment,
            "region": region,
            "device": device,
            "acquisition_channel": acquisition_channel,
            "preferred_payment_method": preferred_payment_method,
            "tenure_months": tenure_months,
            "recency_days": recency_days,
            "purchase_frequency_90d": purchase_frequency_90d,
            "sessions_90d": sessions_90d,
            "avg_session_depth": avg_session_depth,
            "avg_order_value": avg_order_value,
            "monetary_value_90d": monetary_value_90d,
            "discount_ratio": discount_ratio,
            "return_ratio": return_ratio,
            "support_contacts_90d": support_contacts_90d,
            "complaints_180d": complaints_180d,
            "email_open_rate": email_open_rate,
            "push_click_rate": push_click_rate,
            "delivery_delay_ratio": delivery_delay_ratio,
            "catalog_diversity": catalog_diversity,
            "nighttime_activity_ratio": nighttime_activity_ratio,
            "payment_incidents": payment_incidents,
            "browsing_to_purchase_ratio": browsing_to_purchase_ratio,
            "last_basket_value": last_basket_value,
            "latent_churn_probability": latent_churn_probability,
            "churn_label": churn_label,
            "split": split,
        }
    )

    # Real data frequently contain partial measurement failure. Missingness is
    # limited and controlled so that preprocessing remains necessary but not dominant.
    for column, rate in {
        "email_open_rate": 0.030,
        "push_click_rate": 0.030,
        "delivery_delay_ratio": 0.025,
        "avg_session_depth": 0.020,
        "catalog_diversity": 0.020,
    }.items():
        missing_mask = rng.random(n) < rate
        df.loc[missing_mask, column] = np.nan

    return df


def build_preprocessing_pipeline(df: pd.DataFrame) -> Tuple[List[str], List[str], ColumnTransformer]:
    """Create a shared preprocessing object for all candidate models."""
    excluded = {"customer_id", "latent_churn_probability", "churn_label", "split"}
    feature_columns = [column for column in df.columns if column not in excluded]
    categorical_features = [column for column in feature_columns if df[column].dtype == "object"]
    numerical_features = [column for column in feature_columns if column not in categorical_features]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return feature_columns, numerical_features, preprocessing


def get_candidate_models(seed: int) -> Dict[str, object]:
    """Instantiate the supervised classifiers compared in the experiment."""
    return {
        "Logistic regression": LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        "Decision tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=80,
            class_weight="balanced",
            random_state=seed,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=30,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        ),
    }


def optimise_threshold_by_validation_f1(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    """Select the decision threshold that maximizes F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = 2 * precision * recall / (precision + recall + 1e-12)

    # precision_recall_curve returns one more precision/recall point than thresholds.
    best_index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_index]), float(f1_values[best_index])


def train_and_evaluate_models(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, np.ndarray], Dict[str, float]]:
    """Train models, select thresholds on validation, and evaluate on temporal test data."""
    feature_columns, _, preprocessing = build_preprocessing_pipeline(df)
    models = get_candidate_models(config.random_seed)

    train_mask = df["split"] == "train"
    validation_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"

    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, "churn_label"].to_numpy()

    X_validation = df.loc[validation_mask, feature_columns]
    y_validation = df.loc[validation_mask, "churn_label"].to_numpy()

    X_test = df.loc[test_mask, feature_columns]
    y_test = df.loc[test_mask, "churn_label"].to_numpy()

    records = []
    fitted_pipelines = {}
    test_probabilities = {}
    thresholds = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessing),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        validation_prob = pipeline.predict_proba(X_validation)[:, 1]
        threshold, validation_f1 = optimise_threshold_by_validation_f1(y_validation, validation_prob)

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_prediction = (test_prob >= threshold).astype(int)

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test,
            test_prediction,
            average="binary",
            zero_division=0,
        )
        tn, fp, fn, tp = confusion_matrix(y_test, test_prediction).ravel()

        top_n = int(math.ceil(0.10 * len(y_test)))
        ranked_indices = np.argsort(test_prob)[::-1]
        top_decile_churn_rate = y_test[ranked_indices[:top_n]].mean()
        base_churn_rate = y_test.mean()
        top_decile_lift = top_decile_churn_rate / base_churn_rate

        records.append(
            {
                "model": model_name,
                "validation_threshold": threshold,
                "validation_f1": validation_f1,
                "roc_auc": roc_auc_score(y_test, test_prob),
                "pr_auc": average_precision_score(y_test, test_prob),
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "balanced_accuracy": balanced_accuracy_score(y_test, test_prediction),
                "mcc": matthews_corrcoef(y_test, test_prediction),
                "brier": brier_score_loss(y_test, test_prob),
                "top_decile_lift": top_decile_lift,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

        fitted_pipelines[model_name] = pipeline
        test_probabilities[model_name] = test_prob
        thresholds[model_name] = threshold

    model_comparison = pd.DataFrame(records).sort_values(
        by=["f1", "pr_auc", "roc_auc"],
        ascending=False,
    )

    return model_comparison, fitted_pipelines, test_probabilities, thresholds


def make_directories(output_dir: Path) -> Dict[str, Path]:
    """Create the required folder tree."""
    folders = {
        "root": output_dir,
        "5.1": output_dir / "5.1",
        "5.2": output_dir / "5.2",
        "5.3": output_dir / "5.3",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def save_plot(path: Path) -> None:
    """Save a figure in high resolution and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def export_descriptive_outputs(df: pd.DataFrame, folders: Dict[str, Path]) -> None:
    """Export Section 5.1 tables and figures."""
    output = folders["5.1"]

    split_order = ["train", "validation", "test"]
    split_profile = (
        df.groupby("split")
        .agg(
            observations=("churn_label", "size"),
            churn_rate=("churn_label", "mean"),
            mean_recency_days=("recency_days", "mean"),
            mean_monetary_value_90d=("monetary_value_90d", "mean"),
            mean_support_contacts_90d=("support_contacts_90d", "mean"),
        )
        .reindex(split_order)
        .reset_index()
    )
    split_profile.to_csv(output / "table_1_temporal_split_profile.csv", index=False)

    train_df = df[df["split"] == "train"].copy()
    key_features = [
        "recency_days",
        "purchase_frequency_90d",
        "monetary_value_90d",
        "discount_ratio",
        "support_contacts_90d",
        "complaints_180d",
        "delivery_delay_ratio",
        "email_open_rate",
        "browsing_to_purchase_ratio",
    ]
    class_profile = (
        train_df.groupby("churn_label")[key_features]
        .mean()
        .T.rename(columns={0: "non_churn_mean", 1: "churn_mean"})
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    class_profile["absolute_difference"] = class_profile["churn_mean"] - class_profile["non_churn_mean"]
    class_profile.to_csv(output / "table_2_selected_feature_means_by_class.csv", index=False)

    plt.figure(figsize=(6.4, 4.0))
    split_profile.plot(x="split", y="churn_rate", kind="bar", legend=False, ax=plt.gca())
    plt.ylabel("Observed churn rate")
    plt.xlabel("Temporal split")
    plt.title("Churn rate across temporal splits")
    plt.xticks(rotation=0)
    save_plot(output / "figure_1_churn_rate_by_split.png")

    plt.figure(figsize=(6.4, 4.0))
    train_df.loc[train_df["churn_label"] == 0, "recency_days"].plot(
        kind="hist",
        bins=35,
        alpha=0.65,
        label="Non-churn",
        density=True,
    )
    train_df.loc[train_df["churn_label"] == 1, "recency_days"].plot(
        kind="hist",
        bins=35,
        alpha=0.65,
        label="Churn",
        density=True,
    )
    plt.xlabel("Recency days")
    plt.ylabel("Density")
    plt.title("Recency distribution by class in the training set")
    plt.legend()
    save_plot(output / "figure_2_recency_distribution_by_class.png")

    smd_records = []
    for feature in key_features:
        non_churn = train_df.loc[train_df["churn_label"] == 0, feature].dropna()
        churn = train_df.loc[train_df["churn_label"] == 1, feature].dropna()
        pooled_std = math.sqrt((non_churn.var(ddof=1) + churn.var(ddof=1)) / 2)
        smd = (churn.mean() - non_churn.mean()) / pooled_std if pooled_std > 0 else 0
        smd_records.append({"feature": feature, "standardized_mean_difference": smd})

    smd_df = pd.DataFrame(smd_records).sort_values(
        "standardized_mean_difference",
        key=lambda x: np.abs(x),
        ascending=False,
    )
    smd_df.to_csv(output / "standardized_mean_differences.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.barh(smd_df["feature"], smd_df["standardized_mean_difference"])
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Standardized mean difference: churn minus non-churn")
    plt.title("Standardized mean differences for key numerical features")
    plt.gca().invert_yaxis()
    save_plot(output / "figure_3_standardized_mean_differences.png")


def export_predictive_outputs(
    df: pd.DataFrame,
    model_comparison: pd.DataFrame,
    fitted_pipelines: Dict[str, Pipeline],
    test_probabilities: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    folders: Dict[str, Path],
) -> str:
    """Export Section 5.2 tables and figures and return the selected model name."""
    output = folders["5.2"]
    selected_model = model_comparison.iloc[0]["model"]

    rounded_comparison = model_comparison.copy()
    metric_columns = [
        "validation_threshold",
        "validation_f1",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "brier",
        "top_decile_lift",
    ]
    rounded_comparison[metric_columns] = rounded_comparison[metric_columns].round(4)
    rounded_comparison.to_csv(output / "table_3_comparative_predictive_performance.csv", index=False)

    feature_columns, _, _ = build_preprocessing_pipeline(df)
    test_mask = df["split"] == "test"
    X_test = df.loc[test_mask, feature_columns]
    y_test = df.loc[test_mask, "churn_label"].to_numpy()

    plt.figure(figsize=(6.4, 4.8))
    for model_name, probabilities in test_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_value = roc_auc_score(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("ROC comparison on the temporal test set")
    plt.legend(fontsize=8)
    save_plot(output / "figure_4_roc_comparison.png")

    plt.figure(figsize=(6.4, 4.8))
    for model_name, probabilities in test_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        ap_value = average_precision_score(y_test, probabilities)
        plt.plot(recall, precision, label=f"{model_name} (AP={ap_value:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall comparison on the temporal test set")
    plt.legend(fontsize=8)
    save_plot(output / "figure_5_precision_recall_comparison.png")

    selected_probabilities = test_probabilities[selected_model]
    selected_threshold = thresholds[selected_model]
    selected_predictions = (selected_probabilities >= selected_threshold).astype(int)
    cm = confusion_matrix(y_test, selected_predictions)

    plt.figure(figsize=(4.6, 4.0))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix: {selected_model}")
    plt.xlabel("Predicted class")
    plt.ylabel("Observed class")
    tick_labels = ["Non-churn", "Churn"]
    plt.xticks([0, 1], tick_labels, rotation=30, ha="right")
    plt.yticks([0, 1], tick_labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)
    plt.colorbar(fraction=0.046, pad=0.04)
    save_plot(output / "figure_6_confusion_matrix_selected_model.png")

    calibration_df = pd.DataFrame({"probability": selected_probabilities, "observed": y_test})
    calibration_df["bin"] = pd.qcut(calibration_df["probability"], q=10, duplicates="drop")
    reliability = (
        calibration_df.groupby("bin", observed=True)
        .agg(mean_predicted_probability=("probability", "mean"), observed_churn_rate=("observed", "mean"))
        .reset_index(drop=True)
    )
    reliability.to_csv(output / "calibration_bins_selected_model.csv", index=False)

    plt.figure(figsize=(5.5, 4.5))
    plt.plot(
        reliability["mean_predicted_probability"],
        reliability["observed_churn_rate"],
        marker="o",
        label="Risk bins",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed churn rate")
    plt.title(f"Reliability curve: {selected_model}")
    plt.legend(fontsize=8)
    save_plot(output / "figure_7_reliability_curve_selected_model.png")

    return selected_model


def extract_logistic_coefficients(
    pipeline: Pipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract feature names and standardized coefficients from the logistic model."""
    feature_columns, _, _ = build_preprocessing_pipeline(df)
    preprocessing = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed_feature_names = preprocessing.get_feature_names_out(feature_columns)
    coefficients = estimator.coef_[0]
    coefficient_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    coefficient_df["feature"] = (
        coefficient_df["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ")
    )

    return coefficient_df


def export_interpretability_outputs(
    df: pd.DataFrame,
    selected_model: str,
    fitted_pipelines: Dict[str, Pipeline],
    test_probabilities: Dict[str, np.ndarray],
    folders: Dict[str, Path],
) -> None:
    """Export Section 5.3 tables and figures."""
    output = folders["5.3"]

    selected_pipeline = fitted_pipelines[selected_model]
    coefficient_df = extract_logistic_coefficients(selected_pipeline, df)
    coefficient_df.to_csv(output / "logistic_coefficients_full.csv", index=False)

    top_positive = coefficient_df.sort_values("coefficient", ascending=False).head(8)
    top_negative = coefficient_df.sort_values("coefficient", ascending=True).head(8)
    strongest = pd.concat(
        [
            top_positive.assign(direction="Positive risk effect"),
            top_negative.assign(direction="Negative protective effect"),
        ],
        ignore_index=True,
    )
    strongest[["direction", "feature", "coefficient"]].to_csv(
        output / "table_4_strongest_logistic_predictors.csv",
        index=False,
    )

    plot_df = pd.concat([top_positive, top_negative], ignore_index=True)
    plot_df = plot_df.sort_values("coefficient")
    plt.figure(figsize=(7.4, 5.2))
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Standardized logistic coefficient")
    plt.title("Strongest positive and negative coefficients")
    save_plot(output / "figure_8_strongest_logistic_coefficients.png")

    feature_columns, _, _ = build_preprocessing_pipeline(df)
    test_df = df[df["split"] == "test"].copy()
    probabilities = test_probabilities[selected_model]
    test_df["predicted_churn_probability"] = probabilities

    test_df["risk_decile"] = pd.qcut(
        test_df["predicted_churn_probability"],
        q=10,
        labels=list(range(1, 11)),
        duplicates="drop",
    ).astype(int)

    decile_profile = (
        test_df.groupby("risk_decile")
        .agg(
            observations=("churn_label", "size"),
            observed_churn_rate=("churn_label", "mean"),
            mean_predicted_probability=("predicted_churn_probability", "mean"),
        )
        .reset_index()
    )
    decile_profile.to_csv(output / "risk_decile_profile.csv", index=False)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(decile_profile["risk_decile"], decile_profile["observed_churn_rate"], marker="o")
    plt.xlabel("Predicted risk decile")
    plt.ylabel("Observed churn rate")
    plt.title("Observed churn rate by predicted risk decile")
    plt.xticks(decile_profile["risk_decile"])
    save_plot(output / "figure_9_observed_churn_by_risk_decile.png")

    high_risk_mask = test_df["risk_decile"] == test_df["risk_decile"].max()
    profile_features = [
        "recency_days",
        "purchase_frequency_90d",
        "monetary_value_90d",
        "support_contacts_90d",
        "complaints_180d",
        "discount_ratio",
        "email_open_rate",
        "browsing_to_purchase_ratio",
    ]

    rows = []
    for group_name, mask in {
        "Top-risk decile": high_risk_mask,
        "Remaining customers": ~high_risk_mask,
    }.items():
        row = {
            "group": group_name,
            "observations": int(mask.sum()),
            "churn_rate": test_df.loc[mask, "churn_label"].mean(),
            "mean_predicted_probability": test_df.loc[mask, "predicted_churn_probability"].mean(),
        }
        for feature in profile_features:
            row[f"mean_{feature}"] = test_df.loc[mask, feature].mean()
        rows.append(row)

    high_risk_profile = pd.DataFrame(rows)
    high_risk_profile.to_csv(output / "table_5_high_risk_group_vs_remaining_customers.csv", index=False)

    top_two_deciles = test_df["risk_decile"].isin([9, 10])
    capture_summary = {
        "test_observations": int(len(test_df)),
        "test_churn_cases": int(test_df["churn_label"].sum()),
        "top_decile_size": int(high_risk_mask.sum()),
        "top_decile_churn_cases": int(test_df.loc[high_risk_mask, "churn_label"].sum()),
        "top_two_deciles_size": int(top_two_deciles.sum()),
        "top_two_deciles_churn_cases": int(test_df.loc[top_two_deciles, "churn_label"].sum()),
        "top_two_deciles_capture_rate": float(test_df.loc[top_two_deciles, "churn_label"].sum() / test_df["churn_label"].sum()),
    }
    with open(output / "capture_summary.json", "w", encoding="utf-8") as handle:
        json.dump(capture_summary, handle, indent=2)


def export_main_outputs(
    df: pd.DataFrame,
    model_comparison: pd.DataFrame,
    selected_model: str,
    config: ExperimentConfig,
    folders: Dict[str, Path],
) -> None:
    """Export root-level data, configuration, and summary files."""
    df.to_csv(folders["root"] / "synthetic_ecommerce_churn_benchmark.csv", index=False)

    summary = {
        "random_seed": config.random_seed,
        "n_customers": config.n_customers,
        "risk_intercept": config.risk_intercept,
        "risk_scale": config.risk_scale,
        "overall_churn_rate": float(df["churn_label"].mean()),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_churn_rates": df.groupby("split")["churn_label"].mean().to_dict(),
        "selected_model": selected_model,
        "selected_model_metrics": model_comparison[model_comparison["model"] == selected_model].iloc[0].to_dict(),
    }

    with open(folders["root"] / "experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)

    requirements = "\n".join(
        [
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
        ]
    )
    with open(folders["root"] / "requirements.txt", "w", encoding="utf-8") as handle:
        handle.write(requirements + "\n")

    readme = f"""# Early E-Commerce Churn Temporal Risk Scoring

This folder contains reproducible synthetic data, tables, figures, and model
metrics generated by `ecommerce_churn_framework.py`.

Selected model: {selected_model}

Folder structure:
- `5.1/`: descriptive temporal and class-profile evidence.
- `5.2/`: comparative model performance, ROC/PR curves, confusion matrix, and calibration.
- `5.3/`: logistic coefficients, risk deciles, and retention-prioritization profiles.

To reproduce:
```bash
python ecommerce_churn_framework.py
```
"""
    with open(folders["root"] / "README.md", "w", encoding="utf-8") as handle:
        handle.write(readme)


def zip_results(output_dir: Path) -> Path:
    """Create a zip archive with all generated outputs."""
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(output_dir.parent))
    return zip_path


def run_experiment(config: ExperimentConfig) -> Path:
    """Run the complete churn-mining experiment."""
    if config.output_dir.exists():
        # Clean old output while preserving the code file outside the output folder.
        for item in config.output_dir.rglob("*"):
            if item.is_file():
                item.unlink()
        for item in sorted(config.output_dir.rglob("*"), reverse=True):
            if item.is_dir():
                item.rmdir()

    folders = make_directories(config.output_dir)

    print("Generating synthetic benchmark...")
    df = generate_synthetic_ecommerce_churn_data(config)

    print("Training and evaluating models...")
    model_comparison, fitted_pipelines, test_probabilities, thresholds = train_and_evaluate_models(df, config)

    print("Exporting descriptive outputs for Section 5.1...")
    export_descriptive_outputs(df, folders)

    print("Exporting predictive outputs for Section 5.2...")
    selected_model = export_predictive_outputs(
        df,
        model_comparison,
        fitted_pipelines,
        test_probabilities,
        thresholds,
        folders,
    )

    print("Exporting interpretability and prioritization outputs for Section 5.3...")
    export_interpretability_outputs(
        df,
        selected_model,
        fitted_pipelines,
        test_probabilities,
        folders,
    )

    print("Exporting root-level files...")
    export_main_outputs(df, model_comparison, selected_model, config, folders)

    zip_path = zip_results(config.output_dir)
    print(f"Done. Results folder: {config.output_dir.resolve()}")
    print(f"Zip archive: {zip_path.resolve()}")
    return zip_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    config = ExperimentConfig(output_dir=base_dir / "ecommerce_churn_results")
    run_experiment(config)
