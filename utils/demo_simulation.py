"""
Demo Mode - Simulasi FL di satu mesin (tanpa jaringan)
=======================================================
Berguna untuk menguji logika FL sebelum deploy ke VM terpisah.

Jalankan:
    python utils/demo_simulation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split

from client.model import FLModelWrapper
from utils.prepare_dataset import download_dataset, preprocess, split_non_iid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DEMO] %(message)s")

NUM_CLIENTS = 4
NUM_ROUNDS  = 10
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def federated_averaging(client_updates):
    total_samples = sum(u["num_samples"] for u in client_updates)
    param_names   = list(client_updates[0]["weights"].keys())
    global_weights = {}
    for name in param_names:
        global_weights[name] = sum(
            u["weights"][name] * (u["num_samples"] / total_samples)
            for u in client_updates
        )
    return global_weights


def main():
    logging.info("=== DEMO SIMULASI FEDERATED LEARNING ===")
    logging.info("Dataset: UCI Student Performance")
    logging.info(f"Jumlah client: {NUM_CLIENTS}, Jumlah ronde: {NUM_ROUNDS}")

    # 1. Persiapkan data
    raw_df = download_dataset()
    df     = preprocess(raw_df)
    splits = split_non_iid(df, NUM_CLIENTS)

    # 2. Siapkan data lokal setiap client
    client_data = []
    for i, split_df in enumerate(splits):
        X = split_df.drop(columns=["pass"]).values.astype(float)
        y = split_df["pass"].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = FLModelWrapper()
        client_data.append({
            "id":      i + 1,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "model":   model,
            "results": [],
        })
        logging.info(f"Client {i+1}: train={len(X_train)}, test={len(X_test)}")

    global_weights    = None
    all_round_metrics = []

    # 3. Ronde federasi
    for rnd in range(1, NUM_ROUNDS + 1):
        logging.info(f"\n{'='*50}")
        logging.info(f"RONDE {rnd}/{NUM_ROUNDS}")
        logging.info(f"{'='*50}")

        client_updates = []

        for c in client_data:
            # 1. Terapkan global weights sebelum latih (kecuali ronde pertama)
            if global_weights is not None:
                c["model"].set_weights(global_weights)

            # 2. Latih lokal
            c["model"].fit(c["X_train"], c["y_train"])

            # 3. Evaluasi
            metrics = c["model"].score(c["X_test"], c["y_test"])
            logging.info(
                f"Client {c['id']} | acc={metrics['accuracy']:.4f} | f1={metrics['f1_macro']:.4f}"
            )

            c["results"].append({
                "round":    rnd,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            })

            # 4. Kumpulkan weights untuk FedAvg
            client_updates.append({
                "client_id":   c["id"],
                "weights":     c["model"].get_weights(),
                "num_samples": len(c["X_train"]),
                "metrics": {
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                },
            })

        # 5. FedAvg — hitung global model
        global_weights = federated_averaging(client_updates)
        avg_acc = np.mean([u["metrics"]["accuracy"] for u in client_updates])
        logging.info(f"FedAvg selesai | Rata-rata akurasi: {avg_acc:.4f}")
        all_round_metrics.append({
            "round":        rnd,
            "avg_accuracy": avg_acc,
            "clients":      client_updates,
        })

    # 4. Evaluasi final dengan global model
    logging.info(f"\n{'='*50}")
    logging.info("EVALUASI FINAL — Global Model")
    logging.info(f"{'='*50}")
    for c in client_data:
        c["model"].set_weights(global_weights)
        final = c["model"].score(c["X_test"], c["y_test"])
        logging.info(
            f"Client {c['id']} | Akurasi Final: {final['accuracy']:.4f} | F1: {final['f1_macro']:.4f}"
        )

    # 5. Simpan hasil
    for c in client_data:
        path = os.path.join(RESULTS_DIR, f"client_{c['id']}_results.json")
        with open(path, "w") as f:
            json.dump({"client_id": c["id"], "rounds": c["results"]}, f, indent=2)

    server_path = os.path.join(RESULTS_DIR, "training_results.json")
    with open(server_path, "w") as f:
        serializable = []
        for r in all_round_metrics:
            serializable.append({
                "round":        r["round"],
                "avg_accuracy": r["avg_accuracy"],
                "clients": [
                    {
                        "client_id":   u["client_id"],
                        "num_samples": u["num_samples"],
                        "metrics":     u["metrics"],
                    }
                    for u in r["clients"]
                ],
            })
        json.dump(serializable, f, indent=2)

    logging.info(f"\nSemua hasil disimpan di: {RESULTS_DIR}")
    logging.info("Jalankan: python utils/visualize_results.py untuk melihat grafik")


if __name__ == "__main__":
    main()
