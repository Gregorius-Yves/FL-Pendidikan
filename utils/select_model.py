"""
[CHALLENGE 0 - WAJIB] Pemilihan Model Global Optimum
======================================================
Mata Kuliah : Sistem Komputasi Terdistribusi
Topik       : Federated Learning pada Dunia Pendidikan
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.pipeline        import Pipeline

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CANDIDATE_MODELS = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ]),
    "DecisionTree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=7, random_state=42)),
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)),
    ]),
    "NaiveBayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ]),
}

FEDAVG_COMPATIBILITY = {
    "LogisticRegression": "✅ Sangat cocok — coef_ & intercept_ dapat di-average",
    "DecisionTree":       "⚠️  Perlu modifikasi — struktur tree sulit di-average langsung",
    "RandomForest":       "⚠️  Perlu agregasi khusus — rata-rata banyak tree",
    "NaiveBayes":         "✅ Cocok — average theta_ & var_ antar client",
    "KNN":                "❌ Tidak cocok — tidak punya parameter numerik eksplisit",
}


def load_data(client_id, data_dir):
    path = os.path.join(data_dir, f"client_{client_id}", "student_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    X  = df.drop(columns=["pass"]).values.astype(float)
    y  = df["pass"].values.astype(int)
    print(f"Dataset dimuat: {len(df)} baris, {X.shape[1]} fitur")
    print(f"Distribusi label — pass=1: {(y==1).sum()}, pass=0: {(y==0).sum()}")
    return X, y


def evaluate_all_models(X_train, X_test, y_train, y_test):
    results = {}
    print(f"\n{'Model':<22} {'Accuracy':>10} {'F1-Macro':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for model_name, pipeline in CANDIDATE_MODELS.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
        results[model_name] = {
            "accuracy":  round(float(acc),  4),
            "f1_macro":  round(float(f1),   4),
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
        }
        print(f"{model_name:<22} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")
    print("-" * 65)
    return results


def cross_validate_best(model_name, X, y, k=5):
    pipeline   = CANDIDATE_MODELS[model_name]
    cv         = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    f1_scores  = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    return {
        "model":              model_name,
        "k":                  k,
        "cv_accuracy_mean":   round(float(acc_scores.mean()), 4),
        "cv_accuracy_std":    round(float(acc_scores.std()),  4),
        "cv_f1_mean":         round(float(f1_scores.mean()),  4),
        "cv_f1_std":          round(float(f1_scores.std()),   4),
        "cv_accuracy_scores": [round(float(s), 4) for s in acc_scores],
    }


def plot_comparison(results, client_id):
    model_names = list(results.keys())
    metrics     = ["accuracy", "f1_macro", "precision", "recall"]
    colors      = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    x           = np.arange(len(model_names))
    width       = 0.2
    fig, ax     = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[m].get(metric, 0) for m in model_names]
        bars   = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                        color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"Perbandingan Model Kandidat — Client {client_id}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Skor")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"client_{client_id}_model_selection.png")
    plt.savefig(out, dpi=150)
    print(f"Grafik disimpan: {out}")
    plt.close()


def print_fedavg_compatibility():
    print("\n" + "=" * 65)
    print("KOMPATIBILITAS MODEL DENGAN ALGORITMA FedAvg")
    print("=" * 65)
    for model, note in FEDAVG_COMPATIBILITY.items():
        print(f"  {model:<22} {note}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--data_dir",  type=str, default="./data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv_folds",  type=int, default=5)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f" CHALLENGE 0: PEMILIHAN MODEL GLOBAL OPTIMUM")
    print(f" Client ID  : {args.client_id}")
    print(f"{'='*65}\n")

    X, y = load_data(args.client_id, args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}\n")

    print_fedavg_compatibility()

    print("─" * 65)
    print("C0-A: Mengevaluasi semua model kandidat...")
    print("─" * 65)
    results = evaluate_all_models(X_train, X_test, y_train, y_test)

    best_model_name = max(results, key=lambda m: results[m]["f1_macro"])
    print(f"\n→ Model terbaik berdasarkan F1-macro: [{best_model_name}]")
    print(f"  Skor: {results[best_model_name]}")

    print(f"\n{'─'*65}")
    print(f"C0-B: Cross-validating [{best_model_name}] dengan {args.cv_folds}-fold CV...")
    print("─" * 65)
    cv_result = cross_validate_best(best_model_name, X, y, k=args.cv_folds)
    print(f"  CV Accuracy : {cv_result['cv_accuracy_mean']:.4f} ± {cv_result['cv_accuracy_std']:.4f}")
    print(f"  CV F1-Macro : {cv_result['cv_f1_mean']:.4f} ± {cv_result['cv_f1_std']:.4f}")
    print(f"  Skor per fold: {cv_result['cv_accuracy_scores']}")

    plot_comparison(results, args.client_id)

    output = {
        "client_id":            args.client_id,
        "all_models":           results,
        "best_model":           best_model_name,
        "cross_validation":     cv_result,
        "fedavg_compatibility": FEDAVG_COMPATIBILITY,
    }
    json_path = os.path.join(RESULTS_DIR, f"client_{args.client_id}_model_selection.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nHasil lengkap disimpan: {json_path}")

    print("\n" + "=" * 65)
    print("  HASIL KEPUTUSAN MODEL [C0-C]")
    print("=" * 65)
    if CHOSEN_MODEL_NAME:
        print(f"  ✓ Model terpilih  : {CHOSEN_MODEL_NAME}")
        print(f"  ✓ Kompatibilitas  : {FEDAVG_COMPATIBILITY.get(CHOSEN_MODEL_NAME, '-')}")
        print(f"  ✓ Alasan          : {CHOSEN_MODEL_REASON}")
    else:
        print("  ⚠  CHOSEN_MODEL_NAME belum diisi!")
    print("=" * 65)


# =============================================================================
# C0-C: Keputusan Model Terpilih
# =============================================================================
CHOSEN_MODEL_NAME: str   = "LogisticRegression"
CHOSEN_MODEL_REASON: str = (
    "Logistic Regression memiliki akurasi dan F1-score tertinggi di antara "
    "model yang kompatibel dengan FedAvg. Koefisien (coef_ dan intercept_) "
    "berupa array numerik sehingga dapat langsung di-average menggunakan "
    "algoritma FedAvg tanpa modifikasi tambahan. CV std kecil menunjukkan "
    "model stabil di berbagai fold data."
)

if __name__ == "__main__":
    main()
