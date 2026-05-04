import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def load_data(client_id, data_dir, seed=42):
    path = os.path.join(data_dir, f"client_{client_id}", "student_data.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File path not found: {path}")

    df = pd.read_csv(path)

    if "pass" not in df.columns:
        raise ValueError("Column 'pass' not found in  dataset.")

    feature_cols = df.columns.drop("pass")

    X = df[feature_cols].values.astype(np.float32)
    y = df["pass"].values.astype(int)

    stratify_arg = y if len(set(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_arg
    )

    print(f"[DATA] Client {client_id}: {len(df)} rows | "
          f"train={len(X_train)}, test={len(X_test)} | "
          f"pass_rate={y.mean():.2%}")

    return X_train, X_test, y_train, y_test, len(df)

def save_results(client_id, history, results_dir="../results"):
    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(results_dir, f"client_{client_id}_results.json")

    def _json_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"client_id": client_id, "rounds": history},
            f,
            indent=2,
            default=_json_serializer
        )

    print(f"[INFO] Path Saved: {path}")
