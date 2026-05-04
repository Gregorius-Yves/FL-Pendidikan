"""
Visualisasi Hasil Federated Learning
=====================================
Membuat grafik perbandingan akurasi per ronde untuk semua client
dan evaluasi konvergensi global model.

Jalankan setelah training selesai:
    python utils/visualize_results.py
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_client_results() -> dict:
    results = {}
    pattern = os.path.join(RESULTS_DIR, "client_*_results.json")
    for path in sorted(glob.glob(pattern)):
        client_id = os.path.basename(path).replace("client_", "").replace("_results.json", "")
        with open(path) as f:
            results[f"Client {client_id}"] = json.load(f)
    return results


def load_server_results() -> list:
    path = os.path.join(RESULTS_DIR, "training_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def plot_accuracy_per_round(client_results: dict):
    plt.figure(figsize=(10, 6))
    for client_name, rounds in client_results.items():
        rounds_num = [r["round"] for r in rounds]
        accuracies = [r["accuracy"] for r in rounds]
        plt.plot(rounds_num, accuracies, marker="o", label=client_name)

    plt.title("Akurasi Lokal per Ronde — Federated Learning\n(Dataset: Student Performance)")
    plt.xlabel("Ronde Federasi")
    plt.ylabel("Akurasi")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "accuracy_per_round.png")
    plt.savefig(out, dpi=150)
    print(f"Grafik disimpan: {out}")
    plt.show()


def plot_f1_per_round(client_results: dict):
    plt.figure(figsize=(10, 6))
    for client_name, rounds in client_results.items():
        rounds_num = [r["round"] for r in rounds]
        f1s = [r["f1_macro"] for r in rounds]
        plt.plot(rounds_num, f1s, marker="s", linestyle="--", label=client_name)

    plt.title("F1-Score (Macro) per Ronde — Federated Learning")
    plt.xlabel("Ronde Federasi")
    plt.ylabel("F1-Score Macro")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "f1_per_round.png")
    plt.savefig(out, dpi=150)
    print(f"Grafik disimpan: {out}")
    plt.show()


def plot_avg_accuracy(server_results: list):
    if not server_results:
        print("Data server tidak ditemukan, skip grafik server.")
        return
    rounds = [r["round"] for r in server_results]
    avg_accs = [
        np.mean([c["metrics"]["accuracy"] for c in r["clients"]])
        for r in server_results
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, avg_accs, marker="D", color="red", linewidth=2, label="Rata-rata global")
    plt.title("Rata-rata Akurasi Global Model per Ronde")
    plt.xlabel("Ronde Federasi")
    plt.ylabel("Rata-rata Akurasi")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "global_avg_accuracy.png")
    plt.savefig(out, dpi=150)
    print(f"Grafik disimpan: {out}")
    plt.show()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client_results = load_client_results()
    server_results = load_server_results()

    if not client_results:
        print("Belum ada hasil client. Jalankan training dulu.")
        return

    print(f"Memuat hasil dari {len(client_results)} client...")
    plot_accuracy_per_round(client_results)
    plot_f1_per_round(client_results)
    plot_avg_accuracy(server_results)

    # Ringkasan tabel
    print("\n=== RINGKASAN HASIL ===")
    print(f"{'Client':<12} {'Ronde Terbaik':<16} {'Akurasi Terbaik':<18} {'F1 Terbaik'}")
    print("-" * 60)
    for cname, rounds in client_results.items():
        best = max(rounds, key=lambda r: r["accuracy"])
        print(f"{cname:<12} {best['round']:<16} {best['accuracy']:.4f}{'':<12} {best['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
