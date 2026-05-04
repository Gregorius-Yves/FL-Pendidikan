"""
FL Server - Global Model Aggregator (VM-0 / Server)
=====================================================
Mata Kuliah : Sistem Komputasi Terdistribusi
Topik       : Federated Learning pada Dunia Pendidikan
Role        : Menerima model weights dari setiap client,
              melakukan FedAvg aggregation, lalu mengirim
              global model kembali ke semua client.

Jalankan di VM-0 (Server):
    python server.py
"""

import socket
import threading
import pickle
import struct
import numpy as np
import json
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log"),
    ],
)

# ── Konfigurasi ──────────────────────────────────────────────────────────────
HOST = "0.0.0.0"          # Dengarkan semua interface
PORT = 9999
NUM_CLIENTS = 4            # Jumlah kelompok / client VM
NUM_ROUNDS = 10            # Jumlah ronde federasi

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helper: send / recv dengan length-prefix framing ─────────────────────────
def send_data(conn: socket.socket, data: bytes):
    """Kirim data dengan prefix panjang 4-byte big-endian."""
    length = struct.pack(">I", len(data))
    conn.sendall(length + data)


def recv_data(conn: socket.socket) -> bytes:
    """Terima data dengan prefix panjang 4-byte big-endian."""
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


# ── FedAvg ────────────────────────────────────────────────────────────────────
def federated_averaging(client_updates: list[dict]) -> dict:
    """
    Weighted FedAvg: rata-rata bobot model tertimbang oleh jumlah sampel
    masing-masing client.

    client_updates: list of dict dengan key:
        'weights'     : dict {param_name: np.ndarray}  — koefisien model tiap client
        'num_samples' : int                             — jumlah data latih tiap client
        'metrics'     : dict (accuracy, f1, dll.)

    Rumus:
        w_global = Σ (n_k / n_total) * w_k
        di mana n_k = jumlah sampel client k, n_total = total semua sampel
    """
    # =========================================================================
    # TODO [CHALLENGE 1 - WAJIB] : Implementasikan Algoritma FedAvg
    # =========================================================================
    # Langkah-langkah:
    #   1. Hitung total_samples dari semua client_updates
    #   2. Ambil daftar nama parameter dari client pertama
    #         (gunakan: client_updates[0]["weights"].keys())
    #   3. Untuk setiap nama parameter, hitung rata-rata tertimbang:
    #         weighted_sum = Σ  update["weights"][name]  *  (update["num_samples"] / total_samples)
    #   4. Simpan hasil ke dict global_weights
    #   5. Kembalikan global_weights
    #
    # Hint: np.ndarray mendukung operasi aritmatika langsung (* dan +)
    # =========================================================================
    raise NotImplementedError(
        "[CHALLENGE 1] Implementasikan fungsi federated_averaging() di server.py!\n"
        "Baca docstring dan komentar di atas untuk panduan."
    )


# ── Server utama ──────────────────────────────────────────────────────────────
class FederatedServer:
    def __init__(self):
        self.global_weights: dict | None = None
        self.round_results: list = []
        # TODO [CHALLENGE 2]: tambahkan self.best_acc = 0.0 di sini jika mengerjakan Challenge 2

    def handle_client(
        self,
        conn: socket.socket,
        addr,
        updates: list,
        lock: threading.Lock,
        barrier: threading.Barrier,
        current_round: int,
    ):
        client_id = None
        try:
            # 1) Terima update dari client
            raw = recv_data(conn)
            update = pickle.loads(raw)
            client_id = update.get("client_id", str(addr))
            logging.info(
                f"Round {current_round} | Client {client_id} | "
                f"samples={update['num_samples']} | "
                f"acc={update['metrics'].get('accuracy', 0):.4f}"
            )

            with lock:
                updates.append(update)

            # 2) Tunggu semua client selesai mengirim
            barrier.wait()

            # 3) Kirim global weights kembali
            payload = pickle.dumps(
                {"global_weights": self.global_weights, "round": current_round}
            )
            send_data(conn, payload)
            logging.info(f"Round {current_round} | Global weights dikirim ke Client {client_id}")

        except Exception as exc:
            logging.error(f"Error pada client {client_id}: {exc}")
        finally:
            conn.close()

    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(NUM_CLIENTS)
        logging.info(f"Server aktif di {HOST}:{PORT} — menunggu {NUM_CLIENTS} client...")

        for rnd in range(1, NUM_ROUNDS + 1):
            logging.info(f"\n{'='*50}")
            logging.info(f"RONDE {rnd}/{NUM_ROUNDS} dimulai")
            logging.info(f"{'='*50}")

            updates: list = []
            lock = threading.Lock()
            barrier = threading.Barrier(NUM_CLIENTS)
            threads = []

            # Terima koneksi dari semua client
            for _ in range(NUM_CLIENTS):
                conn, addr = server_sock.accept()
                logging.info(f"Koneksi diterima dari {addr}")
                t = threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr, updates, lock, barrier, rnd),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # Aggregasi FedAvg
            self.global_weights = federated_averaging(updates)
            logging.info(f"Round {rnd} | FedAvg selesai")

            # =================================================================
            # TODO [CHALLENGE 2 - OPSIONAL] : Simpan Best Global Model
            # =================================================================
            # Setelah setiap ronde, bandingkan rata-rata akurasi client dengan
            # akurasi terbaik sebelumnya. Jika lebih baik, simpan global_weights
            # ke file (misal: results/best_global_weights.pkl).
            #
            # Langkah:
            #   1. Hitung avg_acc = rata-rata accuracy dari semua updates
            #   2. Bandingkan dengan self.best_acc (tambahkan atribut di __init__)
            #   3. Jika avg_acc > self.best_acc, update self.best_acc dan
            #      simpan self.global_weights ke file pickle
            # Hint: import pickle, lalu gunakan pickle.dump()
            # =================================================================

            # Simpan metrik
            round_metrics = {
                "round": rnd,
                "clients": [
                    {
                        "client_id": u["client_id"],
                        "num_samples": u["num_samples"],
                        "metrics": u["metrics"],
                    }
                    for u in updates
                ],
            }
            self.round_results.append(round_metrics)
            avg_acc = np.mean([u["metrics"].get("accuracy", 0) for u in updates])
            logging.info(f"Round {rnd} | Rata-rata akurasi client: {avg_acc:.4f}")

        # Simpan hasil akhir
        results_path = os.path.join(RESULTS_DIR, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(self.round_results, f, indent=2)
        logging.info(f"\nTraining selesai! Hasil disimpan di {results_path}")
        server_sock.close()


if __name__ == "__main__":
    server = FederatedServer()
    server.run()
