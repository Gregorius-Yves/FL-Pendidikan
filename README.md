# Federated Learning pada Dunia Pendidikan
## Mata Kuliah: Sistem Komputasi Terdistribusi

---

## Deskripsi Project

Project ini mengimplementasikan **Federated Learning (FL)** untuk memprediksi kelulusan siswa menggunakan dataset Student Performance (UCI Machine Learning Repository). Setiap kelompok (VM-1 s/d VM-4) melatih model lokal menggunakan data milik masing-masing, kemudian server (VM-0) mengagregasi model menggunakan algoritma **FedAvg** (Federated Averaging).

### Konsep Utama
- **Data Privacy**: Data siswa tidak pernah meninggalkan VM masing-masing kelompok
- **Federated Averaging (FedAvg)**: Aggregasi bobot model dari semua client, tertimbang jumlah sampel
- **Non-IID Data**: Setiap kelompok mendapat distribusi data yang berbeda (lebih realistis)
- **CPU-Based ML**: Menggunakan Logistic Regression (sklearn) — efisien tanpa GPU

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    VM-0 (Server / Aggregator)                │
│                       IP: <SERVER_IP>:9999                   │
│                                                              │
│  1. Terima weights dari semua client                        │
│  2. Hitung FedAvg                                           │
│  3. Kirim global model kembali ke semua client              │
└───────────────────────┬─────────────────────────────────────┘
                        │  TCP Socket
          ┌─────────────┼─────────────────┐
          │             │                 │
    ┌─────▼──┐    ┌─────▼──┐    ┌─────▼──┐    ┌────────┐
    │ VM-1   │    │ VM-2   │    │ VM-3   │    │ VM-4   │
    │Klp. 1  │    │Klp. 2  │    │Klp. 3  │    │Klp. 4  │
    │        │    │        │    │        │    │        │
    │Data:   │    │Data:   │    │Data:   │    │Data:   │
    │~300 brs│    │~300 brs│    │~300 brs│    │~300 brs│
    └────────┘    └────────┘    └────────┘    └────────┘
```

---

## Dataset

| Keterangan | Detail |
|------------|--------|
| **Nama** | UCI Student Performance |
| **Sumber** | https://archive.ics.uci.edu/ml/datasets/student+performance |
| **Total data** | ~1.200 baris (gabungan Math + Portuguese) |
| **Fitur** | 32 fitur (demografi, kebiasaan belajar, nilai G1, G2) |
| **Target** | `pass` (1 = G3 ≥ 10, 0 = G3 < 10) |
| **Pembagian** | Non-IID: setiap VM mendapat ~300 baris dengan distribusi berbeda |

**Fitur Utama:**
- `age`, `sex`, `school`, `address` — demografi
- `studytime`, `failures`, `absences` — kebiasaan belajar
- `Medu`, `Fedu` — pendidikan orang tua
- `G1`, `G2` — nilai semester 1 & 2 (fitur prediktif kuat)
- `higher` — niat melanjutkan pendidikan tinggi

---

## Struktur Direktori

```
Case-2/
├── server/
│   └── server.py           # VM-0: Global aggregator (FedAvg)
├── client/
│   └── client.py           # VM-1 s/d VM-4: Local trainer
├── utils/
│   ├── prepare_dataset.py  # Download & split dataset
│   ├── demo_simulation.py  # Simulasi FL di 1 mesin (untuk demo)
│   └── visualize_results.py # Grafik hasil training
├── data/                   # Dibuat otomatis oleh prepare_dataset.py
│   ├── full_dataset.csv
│   ├── client_1/student_data.csv
│   ├── client_2/student_data.csv
│   ├── client_3/student_data.csv
│   └── client_4/student_data.csv
├── results/                # Dibuat otomatis saat training
│   ├── training_results.json
│   ├── client_1_results.json ... client_4_results.json
│   ├── accuracy_per_round.png
│   └── global_avg_accuracy.png
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Instalasi Dependensi (di semua VM)

```bash
pip install -r requirements.txt
```

### 2. Persiapan Dataset (di VM-0 atau semua VM)

```bash
python utils/prepare_dataset.py
```

Lalu **salin** folder `data/client_X/` ke VM yang sesuai:
```bash
# Dari VM-0, kirim ke masing-masing VM
scp -r data/client_1/ user@IP_VM1:~/Case-2/data/client_1/
scp -r data/client_2/ user@IP_VM2:~/Case-2/data/client_2/
scp -r data/client_3/ user@IP_VM3:~/Case-2/data/client_3/
scp -r data/client_4/ user@IP_VM4:~/Case-2/data/client_4/
```

### 3. ⭐ Pemilihan Model Optimum — Challenge 0 (di setiap VM client)

**Langkah ini wajib dikerjakan sebelum menjalankan FL!**

```bash
python utils/select_model.py --client_id <ID>  --data_dir ./data
```

Script ini akan:
- Membandingkan 5 model kandidat (Logistic Regression, Decision Tree, Random Forest, Naive Bayes, KNN)
- Menampilkan akurasi, F1, precision, recall setiap model
- Menampilkan kompatibilitas masing-masing model dengan algoritma FedAvg
- Melakukan k-fold cross validation pada model terbaik

Setelah selesai, isi `CHOSEN_MODEL_NAME` dan `CHOSEN_MODEL_REASON` di bagian bawah `select_model.py`, lalu terapkan model tersebut di `client/client.py` (lihat TODO [CHALLENGE 3]).

### 4. Jalankan Server (VM-0)

```bash
python server/server.py
```

### 5. Jalankan Client (VM-1 s/d VM-4)

Di VM-1:
```bash
python client/client.py --client_id 1 --server_host <IP_VM0>
```
Di VM-2:
```bash
python client/client.py --client_id 2 --server_host <IP_VM0>
```
Di VM-3:
```bash
python client/client.py --client_id 3 --server_host <IP_VM0>
```
Di VM-4:
```bash
python client/client.py --client_id 4 --server_host <IP_VM0>
```

> **Catatan**: Server harus sudah jalan sebelum client dijalankan. Semua 4 client harus connect dalam ronde yang sama.

### 6. Demo di 1 Mesin (tanpa jaringan)

Untuk testing sebelum deploy ke VM:
```bash
python utils/demo_simulation.py
```

### 7. Visualisasi Hasil

```bash
python utils/visualize_results.py
```

---

## Parameter yang Dapat Diubah

### Server (`server.py`)
| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `PORT` | 9999 | Port TCP server |
| `NUM_CLIENTS` | 4 | Jumlah client yang ditunggu per ronde |
| `NUM_ROUNDS` | 10 | Total ronde federasi |

### Client (`client.py`)
| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--client_id` | *wajib* | ID client (1-4) |
| `--server_host` | 127.0.0.1 | IP VM server |
| `--server_port` | 9999 | Port server |
| `--rounds` | 10 | Harus sama dengan server |
| `--data_dir` | ./data | Lokasi dataset |

---

## Algoritma Federated Averaging (FedAvg)

$$w_{global} = \sum_{k=1}^{K} \frac{n_k}{n} \cdot w_k$$

Dimana:
- $K$ = jumlah client (4)
- $n_k$ = jumlah sampel client ke-$k$
- $n$ = total sampel semua client
- $w_k$ = bobot model lokal client ke-$k$
- $w_{global}$ = bobot model global hasil aggregasi

---

## Daftar Challenge

| No | File | Tipe | Deskripsi |
|----|------|------|-----------|
| **C0-A** | `utils/select_model.py` | 🔴 **WAJIB** | Evaluasi 5 model kandidat — bandingkan accuracy, F1, precision, recall |
| **C0-B** | `utils/select_model.py` | 🔴 **WAJIB** | K-Fold Cross Validation pada model terbaik |
| **C0-C** | `utils/select_model.py` | 🔴 **WAJIB** | Isi `CHOSEN_MODEL_NAME` dan `CHOSEN_MODEL_REASON` — keputusan model optimum |
| **C0-D** | `client/client.py` | 🔴 **WAJIB** | Terapkan model pilihan ke `FLModelWrapper` (lihat TODO CHALLENGE 3) |
| **C1** | `server/server.py` | 🔴 **WAJIB** | Implementasi algoritma **FedAvg** — weighted average bobot dari semua client |
| **C2** | `server/server.py` | 🟡 Opsional | Simpan **best global model** per ronde |
| **C3** | `client/client.py` | 🔴 **WAJIB** | Terapkan model hasil Challenge 0, sesuaikan `get_weights`/`set_weights` |
| **C4** | `client/client.py` | 🔴 **WAJIB** | Implementasi `get_weights()` — ambil koefisien model |
| **C5** | `client/client.py` | 🔴 **WAJIB** | Implementasi `set_weights()` — terapkan global weights ke model lokal |
| **C6** | `client/client.py` | 🟡 Opsional | Tambahkan precision, recall, confusion matrix ke metrik evaluasi |
| **C7** | `client/client.py` | 🟡 Opsional | Implementasi **early stopping** |
| **C8** | `utils/prepare_dataset.py` | 🔴 **WAJIB** | Implementasi `split_non_iid()` — bagi dataset secara heterogen |
| **C9** | `utils/prepare_dataset.py` | 🟡 Opsional | **Feature engineering** — `avg_grade`, `grade_trend`, `study_fail` |

**Urutan pengerjaan:** C8 → C0-A → C0-B → C0-C → C0-D/C3 → C4 → C5 → C1 → *(opsional: C2, C6, C7, C9)*

### Kelompok 1 (VM-1)
- Setup dan konfigurasi VM
- Jalankan `client.py --client_id 1`
- Analisis data lokal di `data/client_1/`
- Catat akurasi per ronde

### Kelompok 2 (VM-2)
- Setup dan konfigurasi VM
- Jalankan `client.py --client_id 2`
- Analisis perbedaan distribusi data dibanding kelompok lain
- Catat akurasi per ronde

### Kelompok 3 (VM-3)
- Setup dan konfigurasi VM
- Jalankan `client.py --client_id 3`
- Eksperimen: ubah jumlah ronde, amati konvergensi
- Catat akurasi per ronde

### Kelompok 4 (VM-4)
- Setup dan konfigurasi VM
- Jalankan `client.py --client_id 4`
- Eksperimen: bandingkan model lokal vs global
- Catat akurasi per ronde

### Semua Kelompok (Laporan)
1. Jelaskan konsep Federated Learning dan kaitannya dengan privasi data
2. Bandingkan: apakah global model lebih baik dari model lokal masing-masing?
3. Apa keuntungan dan keterbatasan FL pada kasus pendidikan?
4. Diskusikan implikasi Non-IID data terhadap performa model

---

## Pertanyaan Diskusi

1. Mengapa data tidak boleh dikumpulkan di satu tempat dalam konteks pendidikan?
2. Bagaimana FedAvg menjamin privasi? Apakah ada kelemahannya?
3. Apa perbedaan IID dan Non-IID dalam FL? Mengapa Non-IID lebih realistis?
4. Jika salah satu VM mati di tengah training, apa yang terjadi?
5. Bagaimana meningkatkan keamanan komunikasi antar VM?

---

## Referensi

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data* — Paper asli FedAvg
- Cortez & Silva (2008). *Using Data Mining to Predict Secondary School Student Performance* — Paper dataset
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/student+performance
