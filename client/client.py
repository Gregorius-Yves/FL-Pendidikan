"""
FL Client - Local Model Trainer (VM-1 s/d VM-4)
================================================
Mata Kuliah : Sistem Komputasi Terdistribusi
Topik       : Federated Learning pada Dunia Pendidikan
Role        : Melatih model lokal dengan data milik kelompok,
              mengirim weights ke server, menerima global weights.

Jalankan di VM-1 hingga VM-4:
    python client.py --client_id 1 --server_host <IP_SERVER>
    python client.py --client_id 2 --server_host <IP_SERVER>
    python client.py --client_id 3 --server_host <IP_SERVER>
    python client.py --client_id 4 --server_host <IP_SERVER>

Argumen:
    --client_id   : ID kelompok (1-4)
    --server_host : IP address VM server
    --server_port : Port server (default 9999)
    --data_dir    : Direktori dataset (default ./data)
    --rounds      : Jumlah ronde (harus sama dengan server)
"""

import argparse
import socket
import pickle
import struct
import numpy as np
import pandas as pd
import os
import logging
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_score, recall_score, confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CLIENT-%(client_id)s] %(message)s",
)


# ── Helper framing ────────────────────────────────────────────────────────────
def send_data(conn: socket.socket, data: bytes):
    length = struct.pack(">I", len(data))
    conn.sendall(length + data)


def recv_data(conn: socket.socket) -> bytes:
    raw_len = _recvall(conn, 4)
    if not raw_len:
        return b""
    length = struct.unpack(">I", raw_len)[0]
    return _recvall(conn, length)


def _recvall(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


# ── Model Wrapper ─────────────────────────────────────────────────────────────
class FLModelWrapper:
    """
    Wrapper untuk model scikit-learn agar bisa digunakan dalam
    skenario Federated Learning berbasis weight-averaging.

    Karena sklearn tidak memiliki 'weights' seperti neural network,
    kita gunakan LogisticRegression dengan representasi koefisien
    sebagai 'weights' yang dapat di-average.
    """

    def __init__(self, num_features: int, num_classes: int):
        self.num_features = num_features
        self.num_classes = num_classes

        # =====================================================================
        # TODO [CHALLENGE 3 - WAJIB] : Terapkan Model Optimum dari Challenge 0
        # =====================================================================
        # Anda sudah menentukan model terbaik di utils/select_model.py
        # (Challenge C0-C). Sekarang terapkan pilihan tersebut di sini.
        #
        # Ganti model default (LogisticRegression) dengan model pilihan Anda:
        #
        #   Jika memilih LogisticRegression (direkomendasikan untuk FL):
        #     → Tidak perlu diubah
        #
        #   Jika memilih DecisionTree:
        #     self.model = DecisionTreeClassifier(max_depth=7, random_state=42)
        #     → get_weights() dan set_weights() HARUS dimodifikasi!
        #       DecisionTree tidak punya coef_/intercept_. Gunakan pendekatan
        #       alternatif, misalnya: serialisasi struktur pohon sebagai array.
        #
        #   Jika memilih NaiveBayes:
        #     self.model = GaussianNB()
        #     → get_weights() ambil: theta_ (mean) dan var_ (variance)
        #     → set_weights() assign: theta_ dan var_
        #       Ini tetap kompatibel dengan FedAvg (average mean & variance)!
        #
        #   Jika memilih KNN atau RandomForest:
        #     ⚠️  Diskusikan dulu dengan instruktur —
        #         model ini TIDAK langsung kompatibel dengan FedAvg.
        #         Anda perlu merancang mekanisme agregasi alternatif.
        #
        # DOKUMENTASIKAN alasan pilihan Anda di laporan!
        # =====================================================================
        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto",
            C=1.0,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._initialized = False

    def get_weights(self) -> dict:
        """Ambil koefisien model sebagai 'weights' untuk dikirim ke server."""
        if not self._initialized:
            # Kembalikan bobot nol sebagai inisialisasi awal
            return {
                "coef": np.zeros((self.num_classes if self.num_classes > 2 else 1,
                                  self.num_features)),
                "intercept": np.zeros(self.num_classes if self.num_classes > 2 else 1),
            }
        # =====================================================================
        # TODO [CHALLENGE 4 - WAJIB] : Ambil bobot (koefisien) dari model
        # =====================================================================
        # LogisticRegression menyimpan koefisiennya di:
        #   self.model.coef_      — matriks (n_classes, n_features)
        #   self.model.intercept_ — vektor (n_classes,)
        #
        # Kembalikan dict dengan key "coef" dan "intercept".
        # Gunakan .copy() agar tidak terjadi aliasing (bug referensi).
        # Contoh struktur return:
        #   return {"coef": ???, "intercept": ???}
        # =====================================================================
        raise NotImplementedError(
            "[CHALLENGE 4] Implementasikan get_weights() di client.py!\n"
            "Ambil self.model.coef_ dan self.model.intercept_"
        )

    def set_weights(self, weights: dict):
        """Terapkan global weights dari server ke model lokal."""
        if not self._initialized:
            # Inisialisasi model dummy agar atribut (coef_, intercept_) tersedia
            dummy_X = np.random.randn(10, self.num_features)
            dummy_y = np.random.randint(0, self.num_classes, 10)
            self.model.fit(dummy_X, dummy_y)
            self._initialized = True
        # =====================================================================
        # TODO [CHALLENGE 5 - WAJIB] : Terapkan global weights ke model lokal
        # =====================================================================
        # Setelah menerima global_weights dari server, model lokal harus
        # di-update agar dimulai dari titik agregat, bukan dari nol.
        #
        # Tugas:
        #   1. Set self.model.coef_      = nilai "coef" dari dict weights
        #   2. Set self.model.intercept_ = nilai "intercept" dari dict weights
        # Gunakan .copy() untuk menghindari aliasing.
        # =====================================================================
        raise NotImplementedError(
            "[CHALLENGE 5] Implementasikan set_weights() di client.py!\n"
            "Assign weights['coef'] dan weights['intercept'] ke model."
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._initialized = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "report": classification_report(y, y_pred, zero_division=0),
        }
        # =====================================================================
        # TODO [CHALLENGE 6 - OPSIONAL] : Tambahkan Metrik Evaluasi Lebih Lengkap
        # =====================================================================
        # Tambahkan metrik berikut ke dalam dict metrics:
        #   - "precision" : precision_score(y, y_pred, average="macro", zero_division=0)
        #   - "recall"    : recall_score(y, y_pred, average="macro", zero_division=0)
        #   - "confusion_matrix" : confusion_matrix(y, y_pred).tolist()
        #                          (gunakan .tolist() agar bisa di-serialize ke JSON)
        #
        # Pertanyaan: Mengapa F1-score lebih informatif dari accuracy
        # pada dataset yang tidak seimbang (imbalanced class)?
        # =====================================================================
        return metrics


# ── Data Loader ───────────────────────────────────────────────────────────────
def load_data(client_id: int, data_dir: str):
    """
    Load dataset milik client ini.
    File: data/client_{client_id}/student_data.csv
    """
    path = os.path.join(data_dir, f"client_{client_id}", "student_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            f"Jalankan dulu: python utils/prepare_dataset.py"
        )

    df = pd.read_csv(path)
    logging.info(f"Dataset dimuat: {path} — {len(df)} baris")

    # Kolom target
    target_col = "pass"  # 0 = gagal, 1 = lulus

    # Pisahkan fitur dan label
    X = df.drop(columns=[target_col]).values.astype(float)
    y = df[target_col].values.astype(int)

    return X, y, df.drop(columns=[target_col]).columns.tolist()


# ── Client utama ──────────────────────────────────────────────────────────────
class FederatedClient:
    def __init__(self, args):
        self.client_id = args.client_id
        self.server_host = args.server_host
        self.server_port = args.server_port
        self.data_dir = args.data_dir
        self.num_rounds = args.rounds
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Setup logger dengan client_id
        self.logger = logging.LoggerAdapter(
            logging.getLogger(), {"client_id": self.client_id}
        )

        # Load data
        self.X, self.y, self.feature_names = load_data(self.client_id, self.data_dir)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        num_classes = len(np.unique(self.y))
        self.model = FLModelWrapper(
            num_features=self.X_train.shape[1],
            num_classes=num_classes,
        )

        self.logger.info(
            f"Data siap — train: {len(self.X_train)}, test: {len(self.X_test)}, "
            f"features: {self.X_train.shape[1]}, classes: {num_classes}"
        )

    def local_train(self, global_weights: dict | None):
        """Latih model lokal, inisialisasi dari global weights jika ada."""
        if global_weights is not None:
            self.model.set_weights(global_weights)

        # Fine-tune / train ulang dengan data lokal
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> dict:
        return self.model.score(self.X_test, self.y_test)

    def run(self):
        global_weights = None
        all_results = []

        # =====================================================================
        # TODO [CHALLENGE 7 - OPSIONAL] : Implementasikan Early Stopping
        # =====================================================================
        # Early stopping menghentikan training lebih awal jika akurasi tidak
        # meningkat selama N ronde berturut-turut (mencegah overfitting & hemat waktu).
        #
        # Langkah:
        #   1. Tambahkan variabel: best_accuracy = 0.0  dan  no_improve_count = 0
        #   2. Setiap selesai satu ronde, bandingkan accuracy dengan best_accuracy
        #   3. Jika accuracy > best_accuracy: update best_accuracy, reset no_improve_count = 0
        #   4. Jika tidak: no_improve_count += 1
        #   5. Jika no_improve_count >= PATIENCE (misal 3), break dari loop ronde
        #
        # Tambahkan argparse --patience (default 3) agar bisa diatur dari CLI.
        # =====================================================================
        PATIENCE = 3  # TODO [CHALLENGE 7]: gunakan args.patience jika sudah ditambahkan
        best_accuracy = 0.0
        no_improve_count = 0

        for rnd in range(1, self.num_rounds + 1):
            self.logger.info(f"Ronde {rnd}/{self.num_rounds} — local training...")

            # 1) Latih lokal
            self.local_train(global_weights)

            # 2) Evaluasi
            metrics = self.evaluate()
            self.logger.info(
                f"Ronde {rnd} | Akurasi lokal: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1_macro']:.4f}"
            )
            print(metrics["report"])

            # 3) Kirim ke server
            payload = pickle.dumps(
                {
                    "client_id": self.client_id,
                    "weights": self.model.get_weights(),
                    "num_samples": len(self.X_train),
                    "metrics": {
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                    },
                    "round": rnd,
                }
            )

            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect((self.server_host, self.server_port))
                self.logger.info(f"Terhubung ke server {self.server_host}:{self.server_port}")

                send_data(conn, payload)
                self.logger.info(f"Weights terkirim ke server")

                # 4) Terima global weights
                raw = recv_data(conn)
                response = pickle.loads(raw)
                global_weights = response["global_weights"]
                self.logger.info(f"Global weights diterima dari server (ronde {response['round']})")
                conn.close()

            except Exception as exc:
                self.logger.error(f"Gagal terhubung ke server: {exc}")
                break

            all_results.append(
                {"round": rnd, "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}
            )

            # =================================================================
            # TODO [CHALLENGE 7 - OPSIONAL]: Lengkapi logika early stopping di sini
            # =================================================================
            # Setelah all_results.append(...), tambahkan:
            #   if metrics["accuracy"] > best_accuracy:
            #       best_accuracy = metrics["accuracy"]
            #       no_improve_count = 0
            #   else:
            #       no_improve_count += 1
            #   if no_improve_count >= PATIENCE:
            #       self.logger.info(f"Early stopping pada ronde {rnd}")
            #       break
            # =================================================================

        # Simpan hasil
        result_path = os.path.join(
            self.results_dir, f"client_{self.client_id}_results.json"
        )
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2)
        self.logger.info(f"Hasil disimpan di {result_path}")

        # Evaluasi final dengan global model
        if global_weights is not None:
            self.model.set_weights(global_weights)
            final_metrics = self.evaluate()
            self.logger.info(
                f"\n{'='*50}\nEVALUASI FINAL (Global Model)\n"
                f"Akurasi: {final_metrics['accuracy']:.4f}\n"
                f"F1 Macro: {final_metrics['f1_macro']:.4f}\n"
                f"{final_metrics['report']}\n{'='*50}"
            )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client_id", type=int, required=True, help="ID client (1-4)")
    parser.add_argument("--server_host", type=str, default="127.0.0.1", help="IP server")
    parser.add_argument("--server_port", type=int, default=9999, help="Port server")
    parser.add_argument("--data_dir", type=str, default="./data", help="Direktori data")
    parser.add_argument("--rounds", type=int, default=10, help="Jumlah ronde federasi")
    args = parser.parse_args()

    client = FederatedClient(args)
    client.run()
