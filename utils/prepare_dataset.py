"""
Persiapan Dataset - Student Performance
=========================================
Script ini mengunduh/menyiapkan dataset UCI Student Performance,
lalu membaginya menjadi 4 bagian untuk masing-masing client VM.

Dataset: UCI Student Performance Dataset
URL    : https://archive.ics.uci.edu/ml/datasets/student+performance
Fitur  : Demografi, kebiasaan belajar, nilai akademik
Target : pass (1 = G3 >= 10, 0 = G3 < 10)

Jalankan sekali sebelum menjalankan server/client:
    python utils/prepare_dataset.py

Output:
    data/client_1/student_data.csv
    data/client_2/student_data.csv
    data/client_3/student_data.csv
    data/client_4/student_data.csv
    data/full_dataset.csv
"""

import os
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NUM_CLIENTS = 4

# URL dataset UCI Student Performance
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00320/student.zip"
)


def download_dataset() -> pd.DataFrame:
    """Unduh dan parse dataset Student Performance (student-mat.csv)."""
    print("Mengunduh dataset dari UCI Repository...")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
            zip_content = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            # Ada dua file: student-mat.csv (matematika) & student-por.csv (portugis)
            # Kita gunakan keduanya dan gabungkan untuk mendapatkan data lebih banyak
            dfs = []
            for fname in ["student-mat.csv", "student-por.csv"]:
                if fname in z.namelist():
                    with z.open(fname) as f:
                        df = pd.read_csv(f, sep=";")
                        df["subject"] = fname.replace("student-", "").replace(".csv", "")
                        dfs.append(df)
            combined = pd.concat(dfs, ignore_index=True)
        print(f"Dataset berhasil diunduh: {len(combined)} baris")
        return combined
    except Exception as e:
        print(f"Gagal mengunduh: {e}")
        print("Membuat dataset sintetis sebagai fallback...")
        return generate_synthetic_dataset()


def generate_synthetic_dataset() -> pd.DataFrame:
    """
    Buat dataset sintetis berbasis distribusi Student Performance UCI
    jika download gagal. Dataset ini merepresentasikan data siswa SMA.
    """
    np.random.seed(42)
    n = 1200  # Cukup besar untuk dibagi 4 kelompok

    schools = np.random.choice(["GP", "MS"], n)
    sex = np.random.choice(["F", "M"], n)
    age = np.random.randint(15, 23, n)
    address = np.random.choice(["U", "R"], n, p=[0.7, 0.3])
    famsize = np.random.choice(["LE3", "GT3"], n, p=[0.3, 0.7])
    Pstatus = np.random.choice(["T", "A"], n, p=[0.8, 0.2])
    Medu = np.random.randint(0, 5, n)
    Fedu = np.random.randint(0, 5, n)
    Mjob = np.random.choice(["teacher", "health", "services", "at_home", "other"], n)
    Fjob = np.random.choice(["teacher", "health", "services", "at_home", "other"], n)
    reason = np.random.choice(["home", "reputation", "course", "other"], n)
    guardian = np.random.choice(["mother", "father", "other"], n)
    traveltime = np.random.randint(1, 5, n)
    studytime = np.random.randint(1, 5, n)
    failures = np.random.choice([0, 1, 2, 3], n, p=[0.67, 0.2, 0.1, 0.03])
    schoolsup = np.random.choice(["yes", "no"], n, p=[0.3, 0.7])
    famsup = np.random.choice(["yes", "no"], n, p=[0.6, 0.4])
    paid = np.random.choice(["yes", "no"], n, p=[0.4, 0.6])
    activities = np.random.choice(["yes", "no"], n)
    nursery = np.random.choice(["yes", "no"], n, p=[0.8, 0.2])
    higher = np.random.choice(["yes", "no"], n, p=[0.8, 0.2])
    internet = np.random.choice(["yes", "no"], n, p=[0.75, 0.25])
    romantic = np.random.choice(["yes", "no"], n, p=[0.35, 0.65])
    famrel = np.random.randint(1, 6, n)
    freetime = np.random.randint(1, 6, n)
    goout = np.random.randint(1, 6, n)
    Dalc = np.random.randint(1, 6, n)
    Walc = np.random.randint(1, 6, n)
    health = np.random.randint(1, 6, n)
    absences = np.random.poisson(4, n)

    # G1, G2 (nilai semester 1 & 2)
    G1 = np.clip(np.random.normal(11, 3, n).astype(int), 0, 20)
    G2 = np.clip(G1 + np.random.normal(0, 2, n).astype(int), 0, 20)
    G3 = np.clip(G2 + np.random.normal(0, 1.5, n).astype(int), 0, 20)
    # Pengaruh studytime terhadap G3
    G3 = np.clip(G3 + studytime - 2, 0, 20)

    df = pd.DataFrame({
        "school": schools, "sex": sex, "age": age,
        "address": address, "famsize": famsize, "Pstatus": Pstatus,
        "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
        "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime,
        "failures": failures, "schoolsup": schoolsup,
        "famsup": famsup, "paid": paid, "activities": activities,
        "nursery": nursery, "higher": higher, "internet": internet,
        "romantic": romantic, "famrel": famrel, "freetime": freetime,
        "goout": goout, "Dalc": Dalc, "Walc": Walc, "health": health,
        "absences": absences, "G1": G1, "G2": G2, "G3": G3,
    })
    print(f"Dataset sintetis dibuat: {len(df)} baris")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing:
    - Encode kolom kategorikal
    - Buat kolom target 'pass' (G3 >= 10)
    - Drop G3 (untuk menghindari data leakage, gunakan G1 & G2 saja)
    """
    df = df.copy()

    # Target
    df["pass"] = (df["G3"] >= 10).astype(int)
    df = df.drop(columns=["G3"])  # Hapus nilai akhir, prediksi dari G1 & G2

    # Encode kolom kategorik
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # =========================================================================
    # TODO [CHALLENGE 9 - OPSIONAL] : Feature Engineering
    # =========================================================================
    # Tambahkan fitur baru yang mungkin meningkatkan akurasi prediksi:
    #
    #   1. "avg_grade"   : rata-rata G1 dan G2  →  (df["G1"] + df["G2"]) / 2
    #   2. "grade_trend" : tren nilai            →  df["G2"] - df["G1"]
    #                      (positif = membaik, negatif = menurun)
    #   3. "study_fail"  : interaksi studytime*failures  →  df["studytime"] * df["failures"]
    #
    # Setelah menambahkan, jalankan ulang prepare_dataset.py dan amati:
    # - Apakah jumlah fitur bertambah? (cek df.shape)
    # - Apakah akurasi model meningkat?
    # =========================================================================

    return df


def split_non_iid(df: pd.DataFrame, num_clients: int) -> list[pd.DataFrame]:
    """
    Bagi dataset secara Non-IID (heterogen) agar simulasi FL lebih realistis.
    Setiap client mendapat distribusi data yang berbeda.

    Non-IID artinya data antar client TIDAK terdistribusi identik:
    - Beberapa client mungkin punya lebih banyak siswa lulus
    - Beberapa client mungkin punya rentang usia yang berbeda
    Ini mencerminkan kondisi nyata: tiap sekolah/lembaga punya karakteristik data sendiri.
    """
    # =========================================================================
    # TODO [CHALLENGE 8 - WAJIB] : Implementasikan Non-IID Split
    # =========================================================================
    # Tugas: bagi df menjadi num_clients bagian secara Non-IID.
    #
    # Cara 1 (Sederhana) — Sorted Chunk:
    #   1. Urutkan df berdasarkan kolom ["school", "age", "pass"]
    #   2. Hitung chunk_size = len(df) // num_clients
    #   3. Bagi menjadi num_clients potongan berurutan
    #   4. Kembalikan list of DataFrame
    #
    # Cara 2 (Lebih Baik) — Stratified Non-IID:
    #   1. Pisahkan baris pass=1 dan pass=0
    #   2. Distribusikan secara tidak merata:
    #      Client 1: 80% pass=1, 20% pass=0
    #      Client 2: 60% pass=1, 40% pass=0
    #      Client 3: 40% pass=1, 60% pass=0
    #      Client 4: 20% pass=1, 80% pass=0
    #   (simulasi perbedaan kualitas pendidikan antar institusi)
    #
    # Diskusikan: bagaimana Non-IID mempengaruhi konvergensi FedAvg?
    # =========================================================================
    raise NotImplementedError(
        "[CHALLENGE 8] Implementasikan split_non_iid() di utils/prepare_dataset.py!\n"
        "Pilih Cara 1 (sederhana) atau Cara 2 (stratified non-IID)."
    )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download / load dataset
    raw_df = download_dataset()

    # 2. Preprocessing
    df = preprocess(raw_df)
    print(f"\nDistribusi label (pass):\n{df['pass'].value_counts()}")

    # 3. Simpan dataset lengkap
    full_path = os.path.join(DATA_DIR, "full_dataset.csv")
    df.to_csv(full_path, index=False)
    print(f"\nDataset penuh disimpan: {full_path} ({len(df)} baris, {df.shape[1]} kolom)")

    # 4. Split Non-IID ke setiap client
    splits = split_non_iid(df, NUM_CLIENTS)

    print("\nPembagian dataset ke client:")
    print(f"{'Client':<10} {'Baris':<10} {'Pass=1':<10} {'Pass=0':<10}")
    print("-" * 40)
    for i, split_df in enumerate(splits, start=1):
        client_dir = os.path.join(DATA_DIR, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
        out_path = os.path.join(client_dir, "student_data.csv")
        split_df.to_csv(out_path, index=False)
        pass_1 = (split_df["pass"] == 1).sum()
        pass_0 = (split_df["pass"] == 0).sum()
        print(f"Client {i:<5} {len(split_df):<10} {pass_1:<10} {pass_0:<10}")
        print(f"  → Disimpan: {out_path}")

    print("\n✓ Persiapan dataset selesai!")
    print("  Selanjutnya: jalankan server.py di VM-0, lalu client.py di VM-1 s/d VM-4")


if __name__ == "__main__":
    main()
