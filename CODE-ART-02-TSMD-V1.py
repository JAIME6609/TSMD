#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational pipeline for a high-impact article on social-media sentiment analysis.

Purpose
-------
This script generates a reproducible synthetic social-media corpus and analyzes it
with a complete, lightweight experimental workflow aligned with the manuscript
sections on normalization, transformer-inspired contextual modeling, multimodal
fusion, streaming architecture, and explainable/fair sentiment monitoring.

Outputs
-------
The script writes all tables and figures into three subfolders:
    results/5.1  -> preprocessing, routing, and streaming-readiness evidence
    results/5.2  -> predictive performance, latency, and ablation evidence
    results/5.3  -> streaming SLA, drift, fairness, and explanation evidence

Execution
---------
    python social_sentiment_multimodal_pipeline.py --output-dir article_results

Dependency policy
-----------------
The script checks whether the required libraries are installed. If any are missing,
it prints the exact pip command needed. The installation step is not executed
silently because reproducible research environments should remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
}


def check_required_libraries() -> None:
    """Check required packages and print auditable installation instructions."""
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(pip_name)
    if missing:
        command = f"{sys.executable} -m pip install " + " ".join(missing)
        message = (
            "The following required libraries are missing: "
            + ", ".join(missing)
            + "\nInstall them with:\n    "
            + command
        )
        raise RuntimeError(message)


check_required_libraries()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

POSITIVE_TERMS = [
    "great", "excellent", "love", "amazing", "brilliant", "fast", "clear",
    "fair", "helpful", "trustworthy", "stable", "inclusive", "accurate",
    "reliable", "responsive", "transparent", "safe", "efficient", "useful",
]
NEGATIVE_TERMS = [
    "bad", "terrible", "hate", "slow", "broken", "biased", "opaque",
    "unfair", "risky", "angry", "confusing", "fragile", "noisy", "wrong",
    "unstable", "late", "expensive", "unsafe", "toxic",
]
NEUTRAL_TERMS = [
    "update", "platform", "model", "dashboard", "report", "system", "data",
    "comment", "feature", "team", "release", "topic", "metric", "stream",
    "window", "event", "service", "query", "topic",
]
SLANG_MAP = {
    "lit": "excellent", "fire": "excellent", "meh": "neutral", "sus": "suspicious",
    "cringe": "bad", "goat": "best", "mid": "mediocre", "w": "win",
    "l": "loss", "fr": "for real", "imo": "in my opinion", "idk": "i do not know",
}
EMOJI_MAP = {
    "😀": "smiling face", "😍": "love face", "🔥": "fire positive", "🎉": "celebration positive",
    "😐": "neutral face", "🤔": "thinking neutral", "🙄": "eye roll negative", "😡": "angry negative",
    "😭": "crying negative", "📉": "decline negative", "📈": "growth positive", "💔": "broken heart negative",
}
EMOJI_VALENCE = {
    "😀": 1.0, "😍": 1.2, "🔥": 1.1, "🎉": 1.0, "📈": 0.9,
    "😐": 0.0, "🤔": 0.0,
    "🙄": -0.9, "😡": -1.2, "😭": -1.0, "📉": -0.9, "💔": -1.1,
}
TOPICS = ["election", "brand", "service", "public health", "education", "transport", "security", "sports"]
LANGUAGE_VARIETIES = ["standard", "slang-intensive", "code-switching"]
GROUPS = ["A", "B", "C"]


def reduce_elongations(text: str) -> str:
    """Collapse character elongations while preserving common double letters."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_text(text: str) -> str:
    """Normalize social-media microtext: URLs, handles, hashtags, elongation, slang, and emojis."""
    text = text.lower()
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = text.replace("#", " ")
    text = reduce_elongations(text)
    tokens = []
    for raw_token in text.split():
        token = raw_token.strip()
        if token in EMOJI_MAP:
            tokens.extend(EMOJI_MAP[token].split())
        elif token in SLANG_MAP:
            tokens.extend(SLANG_MAP[token].split())
        else:
            tokens.append(token)
    return " ".join(tokens)




def lexicon_score_value(text: str) -> float:
    """Compute a transparent normalized polarity score for a normalized text."""
    positive = set(POSITIVE_TERMS + ["best", "win", "growth", "positive", "celebration", "love", "excellent"])
    negative = set(NEGATIVE_TERMS + ["loss", "decline", "negative", "angry", "mediocre", "suspicious"])
    tokens = text.split()
    if not tokens:
        return 0.0
    score = sum(1 for token in tokens if token in positive) - sum(1 for token in tokens if token in negative)
    return float(score / max(1.0, math.sqrt(len(tokens))))


def label_from_score(score: float) -> str:
    if score <= -0.35:
        return "Negative"
    if score >= 0.35:
        return "Positive"
    return "Neutral"


def generate_synthetic_corpus(n: int = 7488) -> pd.DataFrame:
    """Generate a reproducible synthetic corpus with text, emoji, visual, and temporal signals."""
    rows = []
    base_time = pd.Timestamp("2026-04-16 08:00:00")
    positive_emojis = [e for e, v in EMOJI_VALENCE.items() if v > 0]
    negative_emojis = [e for e, v in EMOJI_VALENCE.items() if v < 0]
    neutral_emojis = [e for e, v in EMOJI_VALENCE.items() if v == 0]

    for i in range(n):
        topic = random.choice(TOPICS)
        group = random.choices(GROUPS, weights=[0.42, 0.36, 0.22], k=1)[0]
        variety = random.choices(LANGUAGE_VARIETIES, weights=[0.55, 0.25, 0.20], k=1)[0]
        latent = np.random.normal(loc=0.0, scale=0.9)
        if topic in {"public health", "security", "transport"}:
            latent -= 0.10
        if topic in {"sports", "education"}:
            latent += 0.08
        if group == "C":
            latent += np.random.normal(0, 0.12)

        term_bucket = POSITIVE_TERMS if latent > 0.25 else NEGATIVE_TERMS if latent < -0.25 else NEUTRAL_TERMS
        base_terms = random.sample(term_bucket, k=2)
        neutral_terms = random.sample(NEUTRAL_TERMS, k=2)
        slang = random.choice(list(SLANG_MAP.keys())) if variety != "standard" and random.random() < 0.65 else ""
        hashtag = f"#{topic.replace(' ', '')}" if random.random() < 0.40 else ""
        mention = "@official" if random.random() < 0.12 else ""
        url = "https://example.org/post" if random.random() < 0.08 else ""

        emoji_probability = 0.50 if abs(latent) > 0.25 else 0.36
        emoji = ""
        if random.random() < emoji_probability:
            if latent > 0.25:
                emoji = random.choice(positive_emojis)
            elif latent < -0.25:
                emoji = random.choice(negative_emojis)
            else:
                emoji = random.choice(neutral_emojis)

        has_image = random.random() < 0.39
        visual_score = np.nan
        photo_id = ""
        if has_image:
            visual_score = float(np.clip(np.random.normal(latent, 0.75), -2.5, 2.5))
            photo_id = f"img_{i:05d}"

        sarcasm = False
        if has_image and emoji and random.random() < 0.16:
            sarcasm = True
            # Create intermodal dissonance by flipping text polarity while preserving image signal.
            if latent > 0:
                base_terms = random.sample(NEGATIVE_TERMS, k=2)
            else:
                base_terms = random.sample(POSITIVE_TERMS, k=2)

        elongated = "soooo" if random.random() < 0.16 else ""
        code_switch = "muy good" if variety == "code-switching" and latent > 0 else "muy bad" if variety == "code-switching" else ""
        words = [mention, elongated] + base_terms + neutral_terms + [slang, code_switch, hashtag, emoji, url]
        text = " ".join([w for w in words if w]).strip()

        emoji_score = EMOJI_VALENCE.get(emoji, 0.0)
        observed_score = 0.65 * latent + 0.55 * emoji_score + (0.85 * visual_score if has_image else 0.0)
        if sarcasm:
            observed_score = 1.05 * np.nan_to_num(visual_score) - 0.15 * latent
        observed_score += np.random.normal(0, 0.40)
        label = label_from_score(observed_score)
        created_at = base_time + pd.Timedelta(minutes=int(i * 3 + np.random.randint(0, 3)))
        rows.append(
            {
                "comment_id": i + 1,
                "created_at": created_at,
                "topic": topic,
                "language_variety": variety,
                "protected_group_proxy": group,
                "raw_text": text,
                "contains_emoji": bool(emoji),
                "emoji": emoji,
                "emoji_score": emoji_score,
                "photo_id": photo_id,
                "has_image": has_image,
                "visual_score": visual_score,
                "sarcasm_proxy": sarcasm,
                "true_sentiment": label,
            }
        )
    df = pd.DataFrame(rows)
    df["normalized_text"] = df["raw_text"].map(normalize_text)
    df["text_length"] = df["normalized_text"].str.split().map(len)
    df["hour"] = pd.to_datetime(df["created_at"]).dt.floor("h")
    df["visual_score_filled"] = df["visual_score"].fillna(0.0)
    df["lexicon_score"] = df["normalized_text"].map(lexicon_score_value)
    df["emoji_visual_product"] = df["emoji_score"] * df["visual_score_filled"]
    df["abs_intermodal_gap"] = (df["emoji_score"] - df["visual_score_filled"]).abs()
    df["intermodal_conflict"] = ((df["emoji_score"] * df["visual_score_filled"]) < -0.10) & df["has_image"]
    return df


def lexicon_predict(texts: Iterable[str]) -> List[str]:
    """Predict sentiment with a transparent lexicon baseline."""
    positive = set(POSITIVE_TERMS + ["best", "win", "growth", "positive", "celebration", "love"])
    negative = set(NEGATIVE_TERMS + ["loss", "decline", "negative", "angry", "mediocre", "suspicious"])
    predictions = []
    for text in texts:
        score = 0
        for token in text.split():
            if token in positive:
                score += 1
            if token in negative:
                score -= 1
        predictions.append(label_from_score(score / max(1, len(text.split()) / 4)))
    return predictions


class FeatureFusionClassifier(BaseEstimator, ClassifierMixin):
    """A light, reproducible multimodal fusion classifier using text and numeric features."""

    def __init__(self, C: float = 2.0, max_iter: int = 800):
        self.C = C
        self.max_iter = max_iter
        self.pipeline = None

    def fit(self, X: pd.DataFrame, y: Iterable[str]):
        numeric_features = [
            "contains_emoji", "emoji_score", "has_image", "visual_score_filled",
            "intermodal_conflict", "text_length", "sarcasm_proxy", "lexicon_score", "emoji_visual_product", "abs_intermodal_gap",
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500), "normalized_text"),
                ("numeric", StandardScaler(), numeric_features),
            ],
            remainder="drop",
        )
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight="balanced", random_state=RANDOM_SEED)),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)


def evaluate_predictions(y_true: Iterable[str], y_pred: Iterable[str], latency_ms_per_1000: float) -> Dict[str, float]:
    """Return a compact metric dictionary for model comparison."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "Negative_Recall": recall[0],
        "Neutral_Recall": recall[1],
        "Positive_Recall": recall[2],
        "Latency_ms_per_1000": latency_ms_per_1000,
    }


def timed_prediction(model_name: str, predictor, X, repetitions: int = 3) -> Tuple[np.ndarray, float]:
    """Measure prediction latency in milliseconds per 1,000 records."""
    predictions = None
    elapsed_values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        predictions = predictor(X)
        elapsed = time.perf_counter() - start
        elapsed_values.append(elapsed)
    avg_elapsed = float(np.mean(elapsed_values))
    records = len(X)
    latency = 1000.0 * avg_elapsed / max(1, records) * 1000.0
    return np.array(predictions), latency


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def save_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, path: Path, rotation: int = 0) -> None:
    fig = plt.figure(figsize=(8.5, 4.8), dpi=160)
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_line(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure(figsize=(8.8, 4.8), dpi=160)
    for col in y_cols:
        plt.plot(df[x_col], df[col], marker="o", label=col)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_col)
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, label_col: str, title: str, path: Path) -> None:
    fig = plt.figure(figsize=(7.6, 4.8), dpi=160)
    plt.scatter(df[x_col], df[y_col])
    for _, row in df.iterrows():
        plt.annotate(row[label_col], (row[x_col], row[y_col]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, path: Path) -> None:
    fig = plt.figure(figsize=(5.6, 4.8), dpi=160)
    plt.imshow(cm)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def create_section_51_outputs(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    """Create outputs for Section 5.1: corpus readiness, preprocessing, routing, and windows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    routing = pd.DataFrame(
        [
            ["Total records", len(df), 100.0],
            ["Records requiring emoji transduction", int(df["contains_emoji"].sum()), df["contains_emoji"].mean() * 100],
            ["Records linked to visual modality", int(df["has_image"].sum()), df["has_image"].mean() * 100],
            ["Records with intermodal conflict", int(df["intermodal_conflict"].sum()), df["intermodal_conflict"].mean() * 100],
            ["Records flagged as sarcasm proxy", int(df["sarcasm_proxy"].sum()), df["sarcasm_proxy"].mean() * 100],
        ],
        columns=["Routing category", "Records", "Percentage"],
    )
    routing["Percentage"] = routing["Percentage"].round(2)
    save_csv(routing, out_dir / "table_5_1_stream_preprocessing_summary.csv")

    norm = pd.DataFrame(
        {
            "Measure": ["Mean raw tokens", "Mean normalized tokens", "Vocabulary before normalization", "Vocabulary after normalization"],
            "Value": [
                df["raw_text"].str.split().map(len).mean(),
                df["normalized_text"].str.split().map(len).mean(),
                len(set(" ".join(df["raw_text"].str.lower()).split())),
                len(set(" ".join(df["normalized_text"]).split())),
            ],
        }
    )
    norm["Value"] = norm["Value"].round(2)
    save_csv(norm, out_dir / "table_5_1_normalization_effect.csv")

    by_hour = df.groupby(["hour", "true_sentiment"]).size().unstack(fill_value=0).reset_index()
    for label in LABELS:
        if label not in by_hour.columns:
            by_hour[label] = 0
    save_csv(by_hour, out_dir / "table_5_1_hourly_sentiment_windows.csv")
    save_line(by_hour, "hour", LABELS, "Sliding-window sentiment volume", "Number of comments", out_dir / "figure_5_1_sliding_window_sentiment.png")
    save_bar(routing.iloc[1:].copy(), "Routing category", "Percentage", "Conditional routing requirements", "Percentage of records", out_dir / "figure_5_1_conditional_routing.png", rotation=25)
    topic_distribution = df.groupby("topic").size().reset_index(name="Records").sort_values("Records", ascending=False)
    save_csv(topic_distribution, out_dir / "table_5_1_topic_distribution.csv")
    save_bar(topic_distribution, "topic", "Records", "Topic distribution in the synthetic social stream", "Records", out_dir / "figure_5_1_topic_distribution.png", rotation=30)
    return {"routing_rate": str(routing.to_dict(orient="records"))}


def create_section_52_outputs(df: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    """Create outputs for Section 5.2: model performance, latency, and ablation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["true_sentiment"], random_state=RANDOM_SEED)
    y_train = train_df["true_sentiment"]
    y_test = test_df["true_sentiment"]

    metrics = []

    lex_pred, lex_latency = timed_prediction("Lexicon", lambda X: lexicon_predict(X["normalized_text"]), test_df)
    row = evaluate_predictions(y_test, lex_pred, lex_latency)
    row["Model"] = "Lexicon baseline"
    metrics.append(row)

    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000)),
        ("clf", LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    lr_pipe.fit(train_df["normalized_text"], y_train)
    lr_pred, lr_latency = timed_prediction("TF-IDF Logistic Regression", lambda X: lr_pipe.predict(X["normalized_text"]), test_df)
    row = evaluate_predictions(y_test, lr_pred, lr_latency)
    row["Model"] = "TF-IDF + logistic regression"
    metrics.append(row)

    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000)),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    svm_pipe.fit(train_df["normalized_text"], y_train)
    svm_pred, svm_latency = timed_prediction("Linear SVM", lambda X: svm_pipe.predict(X["normalized_text"]), test_df)
    row = evaluate_predictions(y_test, svm_pred, svm_latency)
    row["Model"] = "Linear SVM"
    metrics.append(row)

    fusion_clf = FeatureFusionClassifier(C=2.0)
    fusion_clf.fit(train_df, y_train)
    fusion_pred, fusion_latency = timed_prediction("Multimodal fusion", lambda X: fusion_clf.predict(X), test_df)
    row = evaluate_predictions(y_test, fusion_pred, fusion_latency)
    row["Model"] = "Transformer-inspired multimodal fusion"
    metrics.append(row)

    metrics_df = pd.DataFrame(metrics)[
        ["Model", "Accuracy", "Macro_F1", "Negative_Recall", "Neutral_Recall", "Positive_Recall", "Latency_ms_per_1000"]
    ]
    for col in metrics_df.columns:
        if col != "Model":
            metrics_df[col] = metrics_df[col].round(4)
    save_csv(metrics_df, out_dir / "table_5_2_model_performance.csv")
    save_scatter(metrics_df, "Latency_ms_per_1000", "Macro_F1", "Model", "Accuracy-latency trade-off", out_dir / "figure_5_2_latency_f1_tradeoff.png")

    cm = confusion_matrix(y_test, fusion_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in LABELS], columns=[f"Predicted {l}" for l in LABELS])
    save_csv(cm_df.reset_index().rename(columns={"index": "Class"}), out_dir / "table_5_2_confusion_matrix.csv")
    save_confusion_matrix(cm, LABELS, "Confusion matrix for multimodal fusion", out_dir / "figure_5_2_confusion_matrix.png")

    # Ablation study: remove each major input family and retrain.
    ablations = []
    feature_sets = {
        "Text only": ["normalized_text"],
        "Text + emoji": ["normalized_text", "contains_emoji", "emoji_score", "text_length", "lexicon_score"],
        "Text + visual": ["normalized_text", "has_image", "visual_score_filled", "text_length", "lexicon_score"],
        "Full fusion": ["normalized_text", "contains_emoji", "emoji_score", "has_image", "visual_score_filled", "intermodal_conflict", "text_length", "sarcasm_proxy", "lexicon_score", "emoji_visual_product", "abs_intermodal_gap"],
    }
    for name, features in feature_sets.items():
        if name == "Text only":
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500)),
                ("clf", LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_SEED)),
            ])
            pipe.fit(train_df["normalized_text"], y_train)
            pred = pipe.predict(test_df["normalized_text"])
        else:
            numeric_features = [f for f in features if f != "normalized_text"]
            preprocessor = ColumnTransformer(
                transformers=[
                    ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500), "normalized_text"),
                    ("numeric", StandardScaler(), numeric_features),
                ],
                remainder="drop",
            )
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("clf", LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_SEED)),
            ])
            pipe.fit(train_df[features], y_train)
            pred = pipe.predict(test_df[features])
        ablations.append({
            "Ablation": name,
            "Macro_F1": f1_score(y_test, pred, labels=LABELS, average="macro", zero_division=0),
            "Negative_Recall": recall_score(y_test, pred, labels=LABELS, average=None, zero_division=0)[0],
            "Sarcasm_proxy_accuracy": accuracy_score(test_df.loc[test_df["sarcasm_proxy"], "true_sentiment"], pred[test_df["sarcasm_proxy"].to_numpy()]) if test_df["sarcasm_proxy"].any() else np.nan,
        })
    ablation_df = pd.DataFrame(ablations)
    for col in ["Macro_F1", "Negative_Recall", "Sarcasm_proxy_accuracy"]:
        ablation_df[col] = ablation_df[col].round(4)
    save_csv(ablation_df, out_dir / "table_5_2_ablation_study.csv")
    save_bar(ablation_df, "Ablation", "Macro_F1", "Ablation study of modality contributions", "Macro-F1", out_dir / "figure_5_2_ablation_macro_f1.png", rotation=25)

    return {
        "best_model": metrics_df.sort_values("Macro_F1", ascending=False).iloc[0].to_dict(),
        "test_size": len(test_df),
    }


def create_section_53_outputs(df: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    """Create outputs for Section 5.3: SLA, drift, fairness, and explanation evidence."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulated stream throughput by ten-minute windows.
    stream = df.copy().sort_values("created_at")
    stream["window_10min"] = stream["created_at"].dt.floor("10min")
    window_df = stream.groupby("window_10min").agg(
        records=("comment_id", "count"),
        emoji_rate=("contains_emoji", "mean"),
        image_rate=("has_image", "mean"),
        conflict_rate=("intermodal_conflict", "mean"),
    ).reset_index()
    # Deterministic SLA model: higher routing complexity increases processing cost.
    window_df["estimated_processing_ms"] = (
        25 + 0.85 * window_df["records"] + 30 * window_df["emoji_rate"] + 45 * window_df["image_rate"] + 55 * window_df["conflict_rate"]
    ).round(2)
    window_df["estimated_throughput_records_s"] = (1000 * window_df["records"] / window_df["estimated_processing_ms"]).round(2)
    window_df["sla_met"] = window_df["estimated_processing_ms"] <= 250
    save_csv(window_df, out_dir / "table_5_3_streaming_sla_windows.csv")
    save_line(window_df.head(80), "window_10min", ["estimated_processing_ms"], "Streaming micro-batch processing time", "Milliseconds", out_dir / "figure_5_3_streaming_sla.png")

    # Drift monitoring using distribution shift between first and last half.
    midpoint = len(stream) // 2
    first = stream.iloc[:midpoint]
    second = stream.iloc[midpoint:]
    drift_rows = []
    for label in LABELS:
        p = (first["true_sentiment"] == label).mean()
        q = (second["true_sentiment"] == label).mean()
        drift_rows.append({"Class": label, "First_half_rate": p, "Second_half_rate": q, "Absolute_shift": abs(p - q)})
    drift_df = pd.DataFrame(drift_rows)
    drift_df[["First_half_rate", "Second_half_rate", "Absolute_shift"]] = drift_df[["First_half_rate", "Second_half_rate", "Absolute_shift"]].round(4)
    save_csv(drift_df, out_dir / "table_5_3_distribution_drift.csv")
    save_bar(drift_df, "Class", "Absolute_shift", "Sentiment distribution drift", "Absolute shift", out_dir / "figure_5_3_distribution_drift.png")

    # Fairness proxy: build a simple predictive score and compare error rates by group.
    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["true_sentiment"], random_state=RANDOM_SEED)
    fusion_clf = FeatureFusionClassifier(C=2.0).fit(train_df, train_df["true_sentiment"])
    pred = fusion_clf.predict(test_df)
    fairness_rows = []
    for group, subset in test_df.assign(predicted=pred).groupby("protected_group_proxy"):
        acc = accuracy_score(subset["true_sentiment"], subset["predicted"])
        neg_mask = subset["true_sentiment"] == "Negative"
        neg_recall = recall_score(subset["true_sentiment"], subset["predicted"], labels=LABELS, average=None, zero_division=0)[0]
        fairness_rows.append({
            "Group": group,
            "Records": len(subset),
            "Accuracy": acc,
            "Negative_recall": neg_recall,
            "Error_rate": 1.0 - acc,
        })
    fairness_df = pd.DataFrame(fairness_rows)
    for col in ["Accuracy", "Negative_recall", "Error_rate"]:
        fairness_df[col] = fairness_df[col].round(4)
    save_csv(fairness_df, out_dir / "table_5_3_fairness_proxy.csv")
    save_bar(fairness_df, "Group", "Error_rate", "Error-rate parity by protected-group proxy", "Error rate", out_dir / "figure_5_3_fairness_error_rate.png")

    # Explanation proxy based on final classifier coefficients for the positive class.
    pipe = fusion_clf.pipeline
    vectorizer = pipe.named_steps["preprocessor"].named_transformers_["text"]
    clf = pipe.named_steps["classifier"]
    feature_names = list(vectorizer.get_feature_names_out()) + [
        "contains_emoji", "emoji_score", "has_image", "visual_score_filled", "intermodal_conflict", "text_length", "sarcasm_proxy", "lexicon_score", "emoji_visual_product", "abs_intermodal_gap"
    ]
    positive_class_index = list(clf.classes_).index("Positive")
    negative_class_index = list(clf.classes_).index("Negative")
    pos_coefs = clf.coef_[positive_class_index]
    neg_coefs = clf.coef_[negative_class_index]
    importance = []
    for name, pos_coef, neg_coef in zip(feature_names, pos_coefs, neg_coefs):
        importance.append({"Feature": name, "Positive_class_weight": pos_coef, "Negative_class_weight": neg_coef, "Absolute_importance": abs(pos_coef) + abs(neg_coef)})
    importance_df = pd.DataFrame(importance).sort_values("Absolute_importance", ascending=False).head(18)
    for col in ["Positive_class_weight", "Negative_class_weight", "Absolute_importance"]:
        importance_df[col] = importance_df[col].round(4)
    save_csv(importance_df, out_dir / "table_5_3_explainability_feature_importance.csv")
    save_bar(importance_df, "Feature", "Absolute_importance", "Top explanation features", "Absolute coefficient mass", out_dir / "figure_5_3_explainability_importance.png", rotation=35)
    return {"sla_rate": float(window_df["sla_met"].mean()), "fairness_table": fairness_df.to_dict(orient="records")}


def zip_results(output_dir: Path) -> Path:
    """Create a zip archive with all result outputs and the script itself."""
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(output_dir.parent)))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the social-media sentiment analysis experiment.")
    parser.add_argument("--output-dir", type=str, default="article_results", help="Directory where tables, figures, and zip outputs will be written.")
    parser.add_argument("--records", type=int, default=7488, help="Number of synthetic social-media records to generate.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_corpus(args.records)
    save_csv(df, output_dir / "synthetic_social_media_corpus.csv")

    summary = {
        "records": len(df),
        "created_at_min": str(df["created_at"].min()),
        "created_at_max": str(df["created_at"].max()),
        "section_5_1": create_section_51_outputs(df, output_dir / "5.1"),
        "section_5_2": create_section_52_outputs(df, output_dir / "5.2"),
        "section_5_3": create_section_53_outputs(df, output_dir / "5.3"),
    }
    with open(output_dir / "execution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    zip_path = zip_results(output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Results archived at: {zip_path}")


if __name__ == "__main__":
    main()
