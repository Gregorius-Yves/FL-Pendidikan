"""
[CHALLENGE 0 - WAJIB] Pemilihan Model Global Optimum
======================================================
Mata Kuliah : Sistem Komputasi Terdistribusi
Topik       : Federated Learning pada Dunia Pendidikan

DESKRIPSI TUGAS
---------------
Sebelum mengimplementasikan Federated Learning, setiap kelompok harus
menentukan SATU model terbaik yang akan digunakan sebagai global model
dalam skenario FL. Langkah ini disebut "Model Selection" — tahap penting
dalam pipeline machine learning.

Mengapa perlu model selection terlebih dahulu?
  - Tidak semua classifier cocok untuk FL (harus bisa di-average weightnya)
  - Baseline lokal menentukan apakah FL memberikan nilai tambah
  - Pilihan model mempengaruhi komunikasi, komputasi, dan akurasi global

TUGAS
-----
1. [WAJIB C0-A] Lengkapi fungsi evaluate_all_models() — jalankan 4 model
   kandidat di data lokal dan bandingkan hasilnya.

2. [WAJIB C0-B] Lengkapi fungsi cross_validate_best() — validasi model
   terpilih menggunakan k-fold cross validation.

3. [WAJIB C0-C] Isi variabel CHOSEN_MODEL_NAME di bagian bawah file
   berdasarkan hasil eksperimen, lalu tuliskan alasan pemilihan di
   CHOSEN_MODEL_REASON.

4. [WAJIB C0-D] Salin nama model yang dipilih ke dalam client.py
   (lihat TODO [CHALLENGE 3] di file tersebut).

Jalankan script ini di setiap VM sebelum menjalankan server/client:
    python utils/select_model.py --client_id 1 --data_dir ./data

Output:
    results/client_X_model_selection.json  — hasil perbandingan semua model
    results/client_X_model_selection.png   — grafik perbandingan
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay, confusion_matrix,
)
from sklearn.pipeline import Pipeline

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Daftar model kandidat ─────────────────────────────────────────────────────
# Semua model ini HANYA berbasis CPU — tidak perlu GPU.
# Catatan penting untuk FL: model yang bisa di-FedAvg harus memiliki
# representasi "weights" berupa array numerik (coef_, intercept_, dll.)
CANDIDATE_MODELS = {
    # ✅ Cocok untuk FedAvg — punya coef_ & intercept_
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ]),

    # ⚠️  Bisa di-FedAvg dengan pendekatan alternatif (rata-rata threshold tiap node)
    "DecisionTree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=7, random_state=42)),
    ]),

    # ⚠️  Ensemble — susah di-FedAvg langsung, perlu agregasi khusus
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)),
    ]),

    # ✅ Cocok untuk FedAvg via Naive Bayes parameter averaging (mean, var)
    "NaiveBayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),

    # ❓ Tidak punya weights eksplisit — tidak langsung cocok untuk FedAvg
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ]),
}

# Kompatibilitas FedAvg (untuk referensi mahasiswa)
FEDAVG_COMPATIBILITY = {
    "LogisticRegression": "✅ Sangat cocok — coef_ & intercept_ dapat di-average",
    "DecisionTree":       "⚠️  Perlu modifikasi — struktur tree sulit di-average langsung",
    "RandomForest":       "⚠️  Perlu agregasi khusus — rata-rata banyak tree",
    "NaiveBayes":         "✅ Cocok — average theta_ & var_ antar client",
    "KNN":                "❌ Tidak cocok — tidak punya parameter numerik eksplisit",
}


def load_data(client_id: int, data_dir: str):
    path = os.path.join(data_dir, f"client_{client_id}", "student_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            f"Jalankan dulu: python utils/prepare_dataset.py"
        )
    df = pd.read_csv(path)
    X = df.drop(columns=["pass"]).values.astype(float)
    y = df["pass"].values.astype(int)
    print(f"Dataset dimuat: {len(df)} baris, {X.shape[1]} fitur")
    print(f"Distribusi label — pass=1: {(y==1).sum()}, pass=0: {(y==0).sum()}")
    return X, y


def evaluate_all_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Latih dan evaluasi semua model kandidat pada data lokal.
    Kembalikan dict berisi metrik masing-masing model.
    """
    # =========================================================================
    # TODO [CHALLENGE C0-A - WAJIB] : Evaluasi Semua Model Kandidat
    # =========================================================================
    # Iterasi semua model di CANDIDATE_MODELS, latih dengan X_train/y_train,
    # lalu evaluasi di X_test/y_test.
    #
    # Untuk setiap model, hitung:
    #   - accuracy  : accuracy_score(y_test, y_pred)
    #   - f1_macro  : f1_score(y_test, y_pred, average="macro", zero_division=0)
    #   - precision : precision_score(y_test, y_pred, average="macro", zero_division=0)
    #   - recall    : recall_score(y_test, y_pred, average="macro", zero_division=0)
    #
    # Simpan ke dalam dict results dengan struktur:
    #   results[model_name] = {
    #       "accuracy": ..., "f1_macro": ..., "precision": ..., "recall": ...
    #   }
    #
    # Cetak ringkasan hasil ke terminal.
    #
    # Hint: Pipeline sklearn sudah include StandardScaler, cukup panggil
    #       pipeline.fit(X_train, y_train) dan pipeline.predict(X_test)
    # =========================================================================
    raise NotImplementedError(
        "[CHALLENGE C0-A] Implementasikan evaluate_all_models()!\n"
        "Iterasi CANDIDATE_MODELS, latih tiap model, hitung 4 metrik evaluasi."
    )


def cross_validate_best(model_name: str, X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
    """
    Lakukan k-fold cross validation pada model terbaik untuk mendapatkan
    estimasi performa yang lebih robust (tidak bergantung pada satu split).
    """
    # =========================================================================
    # TODO [CHALLENGE C0-B - WAJIB] : K-Fold Cross Validation
    # =========================================================================
    # Langkah:
    #   1. Ambil pipeline dari CANDIDATE_MODELS[model_name]
    #   2. Gunakan StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    #      (Stratified menjaga proporsi kelas di setiap fold)
    #   3. Gunakan cross_val_score() dengan scoring="accuracy" dan "f1_macro"
    #      Contoh:
    #        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    #        acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    #   4. Kembalikan dict:
    #        {
    #          "model": model_name,
    #          "k": k,
    #          "cv_accuracy_mean": float,
    #          "cv_accuracy_std": float,
    #          "cv_f1_mean": float,
    #          "cv_f1_std": float,
    #          "cv_accuracy_scores": list,
    #        }
    #
    # Mengapa CV lebih baik dari single train/test split?
    # Karena mengurangi variance estimasi dan memberikan confidence interval.
    # =========================================================================
    raise NotImplementedError(
        "[CHALLENGE C0-B] Implementasikan cross_validate_best()!\n"
        f"Lakukan {k}-fold CV untuk model '{model_name}'."
    )


def plot_comparison(results: dict, client_id: int):
    """Buat grafik batang perbandingan semua model kandidat."""
    model_names = list(results.keys())
    metrics = ["accuracy", "f1_macro", "precision", "recall"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(f"Perbandingan Model Kandidat — Client {client_id}\n"
                 f"(Dataset: Student Performance | Task: Prediksi Kelulusan)")
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
    plt.show()


def print_fedavg_compatibility():
    """Tampilkan tabel kompatibilitas model dengan FedAvg."""
    print("\n" + "=" * 65)
    print("KOMPATIBILITAS MODEL DENGAN ALGORITMA FedAvg")
    print("=" * 65)
    for model, note in FEDAVG_COMPATIBILITY.items():
        print(f"  {model:<22} {note}")
    print("=" * 65)
    print("Pilih model yang ✅ agar dapat langsung digunakan dalam FL!\n")


def main():
    parser = argparse.ArgumentParser(description="Model Selection untuk Federated Learning")
    parser.add_argument("--client_id", type=int, required=True, help="ID client (1-4)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Direktori data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporsi data test")
    parser.add_argument("--cv_folds", type=int, default=5, help="Jumlah fold cross validation")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f" CHALLENGE 0: PEMILIHAN MODEL GLOBAL OPTIMUM")
    print(f" Client ID  : {args.client_id}")
    print(f"{'='*65}\n")

    # 1. Load data
    X, y = load_data(args.client_id, args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}\n")

    # 2. Tampilkan kompatibilitas FedAvg
    print_fedavg_compatibility()

    # 3. Evaluasi semua model [C0-A]
    print("─" * 65)
    print("Mengevaluasi semua model kandidat...")
    print("─" * 65)
    results = evaluate_all_models(X_train, X_test, y_train, y_test)

    # 4. Tentukan model terbaik berdasarkan F1-macro (lebih adil untuk imbalanced)
    best_model_name = max(results, key=lambda m: results[m]["f1_macro"])
    print(f"\n→ Model terbaik berdasarkan F1-macro: [{best_model_name}]")
    print(f"  Skor: {results[best_model_name]}")

    # 5. Cross-validate model terbaik [C0-B]
    print(f"\n─" * 65 + "")
    print(f"Cross-validating [{best_model_name}] dengan {args.cv_folds}-fold CV...")
    print("─" * 65)
    cv_result = cross_validate_best(best_model_name, X, y, k=args.cv_folds)
    print(f"  CV Accuracy : {cv_result['cv_accuracy_mean']:.4f} ± {cv_result['cv_accuracy_std']:.4f}")
    print(f"  CV F1-Macro : {cv_result['cv_f1_mean']:.4f} ± {cv_result['cv_f1_std']:.4f}")
    print(f"  Skor per fold: {[round(s, 4) for s in cv_result['cv_accuracy_scores']]}")

    # 6. Grafik perbandingan
    plot_comparison(results, args.client_id)

    # 7. Simpan hasil ke JSON
    output = {
        "client_id": args.client_id,
        "all_models": results,
        "best_model": best_model_name,
        "cross_validation": cv_result,
        "fedavg_compatibility": FEDAVG_COMPATIBILITY,
    }
    json_path = os.path.join(RESULTS_DIR, f"client_{args.client_id}_model_selection.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nHasil lengkap disimpan: {json_path}")

    # 8. Prompt mahasiswa untuk mengisi keputusan [C0-C & C0-D]
    print("\n" + "=" * 65)
    print("  TODO [CHALLENGE C0-C] : ISI KEPUTUSAN ANDA")
    print("=" * 65)
    print("  Setelah melihat hasil di atas, isi dua variabel berikut")
    print("  di bagian bawah file ini (utils/select_model.py):\n")
    print("    CHOSEN_MODEL_NAME   = \"...\"  # nama model yang dipilih")
    print("    CHOSEN_MODEL_REASON = \"...\"  # alasan pemilihan (1-2 kalimat)")
    print()
    print("  TODO [CHALLENGE C0-D] : SALIN KE client.py")
    print("  Setelah memilih model, buka client/client.py dan")
    print("  ganti model default di TODO [CHALLENGE 3] sesuai")
    print("  pilihan di sini.")
    print("=" * 65)

    # Cek apakah mahasiswa sudah mengisi keputusan
    if CHOSEN_MODEL_NAME:
        print(f"\n✓ Pilihan model tercatat: [{CHOSEN_MODEL_NAME}]")
        if CHOSEN_MODEL_NAME in FEDAVG_COMPATIBILITY:
            print(f"  Kompatibilitas FL: {FEDAVG_COMPATIBILITY[CHOSEN_MODEL_NAME]}")
        if CHOSEN_MODEL_REASON:
            print(f"  Alasan: {CHOSEN_MODEL_REASON}")
    else:
        print("\n⚠  CHOSEN_MODEL_NAME belum diisi. Lengkapi Challenge C0-C!")


# =============================================================================
# TODO [CHALLENGE C0-C - WAJIB] : Isi Keputusan Model Terbaik
# =============================================================================
# Setelah menjalankan script ini dan melihat hasil perbandingan,
# isi dua variabel di bawah dengan keputusan kelompok Anda:
#
#   CHOSEN_MODEL_NAME  : nama model yang dipilih (harus ada di CANDIDATE_MODELS)
#                        Pilihan: "LogisticRegression", "DecisionTree",
#                                 "RandomForest", "NaiveBayes", "KNN"
#
#   CHOSEN_MODEL_REASON: alasan singkat pemilihan (1-2 kalimat),
#                        pertimbangkan: akurasi, F1-score, CV stability,
#                        DAN kompatibilitas dengan FedAvg!
#
# Contoh pengisian:
#   CHOSEN_MODEL_NAME   = "LogisticRegression"
#   CHOSEN_MODEL_REASON = "Akurasi 0.82 dan F1 0.79, tertinggi di antara model
#                          yang kompatibel dengan FedAvg. CV std kecil (0.02)
#                          menunjukkan model stabil."
# =============================================================================
CHOSEN_MODEL_NAME: str = ""      # TODO: isi nama model pilihan Anda
CHOSEN_MODEL_REASON: str = ""    # TODO: isi alasan pemilihan


if __name__ == "__main__":
    main()
