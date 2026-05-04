"""
FL Client - VM-1 / Kelompok 1
"""

import os
import sys
import json
import socket
import struct
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

DEFAULT_SERVER_HOST  = "127.0.0.1"
DEFAULT_SERVER_PORT  = 9999
DEFAULT_ROUNDS       = 10
RANDOM_SEED          = 42
EARLY_STOP_PATIENCE  = 3
EARLY_STOP_MIN_DELTA = 0.001


class FLModelWrapper:
    def __init__(self):
        self.scaler      = StandardScaler()
        self.model       = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, C=1.0, solver="lbfgs")
        self._is_fitted  = False
        self._n_features = None

    def get_weights(self):
        if not self._is_fitted:
            raise RuntimeError("Model belum dilatih.")
        return {
            "coef":      self.model.coef_.flatten().astype(np.float64),
            "intercept": self.model.intercept_.flatten().astype(np.float64),
        }

    def set_weights(self, weights_dict):
        if self._n_features is None:
            raise RuntimeError("Latih model sekali sebelum set_weights().")
        coef      = np.array(weights_dict["coef"]).flatten()
        intercept = np.array(weights_dict["intercept"]).flatten()
        self.model.coef_      = coef.reshape(1, -1)
        self.model.intercept_ = intercept.reshape(1,)

    def train(self, X_train, y_train):
        if not self._is_fitted:
            X_scaled = self.scaler.fit_transform(X_train)
        else:
            X_scaled = self.scaler.transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._is_fitted  = True
        self._n_features = X_train.shape[1]
        y_pred = self.model.predict(X_scaled)
        return self._metrics(y_train, y_pred, "train")

    def evaluate(self, X_test, y_test):
        if not self._is_fitted:
            raise RuntimeError("Model belum dilatih.")
        X_scaled = self.scaler.transform(X_test)
        y_pred   = self.model.predict(X_scaled)
        return self._metrics(y_test, y_pred, "test")

    def _metrics(self, y_true, y_pred, split):
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_true, y_pred).tolist()
        return {
            f"{split}_accuracy":         round(float(acc),  4),
            f"{split}_f1":               round(float(f1),   4),
            f"{split}_precision":        round(float(prec), 4),
            f"{split}_recall":           round(float(rec),  4),
            f"{split}_confusion_matrix": cm,
        }


class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_score  = -np.inf
        self.counter     = 0
        self.should_stop = False

    def step(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  [EarlyStopping] Dihentikan.")
        return self.should_stop


def send_data(sock, raw):
    length = struct.pack(">I", len(raw))
    sock.sendall(length + raw)


def recv_data(sock):
    raw_len = _recvall(sock, 4)
    if not raw_len:
        raise ConnectionError("Server menutup koneksi.")
    length = struct.unpack(">I", raw_len)[0]
    return _recvall(sock, length)


def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def load_data(client_id, data_dir):
    path = os.path.join(data_dir, f"client_{client_id}", "student_data.csv")
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        sys.exit(1)
    df           = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "pass"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["pass"].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[DATA] Client {client_id}: {len(df)} baris | "
          f"train={len(X_train)}, test={len(X_test)} | "
          f"pass_rate={y.mean():.2%}")
    return X_train, X_test, y_train, y_test, len(df)


def save_results(client_id, history, results_dir="./results"):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"client_{client_id}_results.json")
    with open(path, "w") as f:
        json.dump({"client_id": client_id, "rounds": history}, f, indent=2)
    print(f"[INFO] Hasil disimpan: {path}")


def run_client(client_id, server_host, server_port, rounds, data_dir):
    print("=" * 60)
    print(f"  FL CLIENT -- Kelompok 1 | Client ID: {client_id}")
    print(f"  Server: {server_host}:{server_port} | Rounds: {rounds}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, n_samples = load_data(client_id, data_dir)

    model         = FLModelWrapper()
    early_stop    = EarlyStopping()
    history       = []
    best_accuracy = -np.inf
    best_weights  = None

    for round_num in range(1, rounds + 1):
        print(f"\n{'--'*25}")
        print(f"  RONDE {round_num}/{rounds}")
        print(f"{'--'*25}")

        # 1. Latih model lokal
        train_metrics = model.train(X_train, y_train)
        print(f"  [TRAIN] acc={train_metrics['train_accuracy']:.4f} | "
              f"f1={train_metrics['train_f1']:.4f}")

        # 2. Evaluasi lokal
        local_metrics = model.evaluate(X_test, y_test)
        print(f"  [EVAL-LOCAL]  acc={local_metrics['test_accuracy']:.4f} | "
              f"f1={local_metrics['test_f1']:.4f} | "
              f"prec={local_metrics['test_precision']:.4f} | "
              f"rec={local_metrics['test_recall']:.4f}")
        print(f"  [C6] Confusion Matrix: {local_metrics['test_confusion_matrix']}")

        # 3. Reconnect ke server setiap ronde
        connected = False
        try:
            print(f"  [NET] Menghubungi server {server_host}:{server_port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(120)
            sock.connect((server_host, server_port))
            print("  [NET] Terhubung ke server.")
            connected = True

            # 4. Kirim weights
            weights_dict = model.get_weights()
            payload = {
                "client_id":   client_id,
                "num_samples": n_samples,
                "weights":     weights_dict,
                "metrics": {
                    "accuracy":  local_metrics["test_accuracy"],
                    "f1":        local_metrics["test_f1"],
                    "precision": local_metrics["test_precision"],
                    "recall":    local_metrics["test_recall"],
                },
            }
            send_data(sock, pickle.dumps(payload))
            print(f"  [NET] Weights dikirim. (coef shape={weights_dict['coef'].shape})")

            # 5. Terima global weights
            raw_response        = recv_data(sock)
            server_response     = pickle.loads(raw_response)
            global_weights_dict = server_response.get("global_weights")

            if global_weights_dict is None:
                print("  [WARN] Tidak ada global weights dari server.")
                sock.close()
                continue

            # 6. Terapkan global weights
            model.set_weights(global_weights_dict)
            print("  [FL]  Global weights diterapkan ke model lokal.")
            sock.close()

        except ConnectionRefusedError:
            print(f"  [ERROR] Koneksi ditolak — pastikan server sudah berjalan.")
            break
        except ConnectionError as e:
            print(f"  [ERROR] Koneksi terputus: {e}")
            break
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            break

        if not connected:
            break

        # 7. Evaluasi setelah global weights
        global_metrics = model.evaluate(X_test, y_test)
        print(f"  [EVAL-GLOBAL] acc={global_metrics['test_accuracy']:.4f} | "
              f"f1={global_metrics['test_f1']:.4f} | "
              f"prec={global_metrics['test_precision']:.4f} | "
              f"rec={global_metrics['test_recall']:.4f}")

        history.append({
            "round":          round_num,
            "local_metrics":  local_metrics,
            "global_metrics": global_metrics,
            "n_samples":      n_samples,
        })

        if global_metrics["test_accuracy"] > best_accuracy:
            best_accuracy = global_metrics["test_accuracy"]
            best_weights  = {k: v.copy() for k, v in model.get_weights().items()}
            print(f"  [*] Best accuracy diperbarui: {best_accuracy:.4f}")

        if early_stop.step(global_metrics["test_accuracy"]):
            if best_weights is not None:
                model.set_weights(best_weights)
            break

    # Evaluasi akhir
    print("\n" + "=" * 60)
    print("  EVALUASI AKHIR")
    print("=" * 60)
    final = model.evaluate(X_test, y_test)
    print(f"  Accuracy         : {final['test_accuracy']:.4f}")
    print(f"  F1               : {final['test_f1']:.4f}")
    print(f"  Precision        : {final['test_precision']:.4f}")
    print(f"  Recall           : {final['test_recall']:.4f}")
    print(f"  Confusion Matrix : {final['test_confusion_matrix']}")
    print(f"  Best accuracy    : {best_accuracy:.4f}")
    print(f"  Total ronde      : {len(history)}")

    save_results(client_id, history)


def main():
    parser = argparse.ArgumentParser(description="FL Client -- Kelompok 1")
    parser.add_argument("--client_id",   type=int, required=True)
    parser.add_argument("--server_host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server_port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--rounds",      type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--data_dir",    default="./data")
    args = parser.parse_args()

    run_client(
        client_id   = args.client_id,
        server_host = args.server_host,
        server_port = args.server_port,
        rounds      = args.rounds,
        data_dir    = args.data_dir,
    )


if __name__ == "__main__":
    main()
