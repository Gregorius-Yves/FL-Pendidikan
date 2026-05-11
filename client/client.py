from . import data
from . import network
from . import early_stopping
import numpy as np
from . import model as model_lib

def run_client(client_context: dict, server_host, server_port, rounds, data_dir):
    """
    do not even attempt to run this without reading what client_context is in main.py
    """

    print("=" * 60)
    print("FL CLIENT -- {0} | Client ID: {1}".format(client_context.name, client_context.client_id))
    print("Server: {0}:{1} | Rounds: {2}".format(server_host, server_port, rounds))
    print("=" * 60)

    X_train, X_test, y_train, y_test, n_samples = data.load_data(client_context.client_id, data_dir)

    model = model_lib.FLModelWrapper()
    early_stop = early_stopping.EarlyStopping()
    history = []
    best_accuracy = -np.inf
    best_weights = None

    for round in range(1, rounds + 1):
        print(f"\n{'--'*25}")
        print(f"  ROUND {round}/{rounds}")
        print(f"{'--'*25}")

        train_metrics = model.train(X_train, y_train)
        print(f"  [TRAIN] acc={train_metrics['train_accuracy']:.4f} | "
              f"f1={train_metrics['train_f1']:.4f}")

        local_metrics = model.evaluate(X_test, y_test)
        print(f"  [EVAL-LOCAL]  acc={local_metrics['test_accuracy']:.4f} | "
              f"f1={local_metrics['test_f1']:.4f} | "
              f"prec={local_metrics['test_precision']:.4f} | "
              f"rec={local_metrics['test_recall']:.4f}")
        print(f"  [C6] Confusion Matrix: {local_metrics['test_confusion_matrix']}")

        weights_dict = model.get_weights()
        payload = {
            "client_id": client_context.client_id,
            "num_samples": n_samples,
            "weights": weights_dict,
            "metrics": {
                "accuracy": local_metrics["test_accuracy"],
                "f1": local_metrics["test_f1"],
                "precision": local_metrics["test_precision"],
                "recall": local_metrics["test_recall"],
            },
        }

        rsp = network.server_handler_interface(server_host, server_port, payload)

        global_weights = rsp.get("global_weights")

        if global_weights is None:
            print("  [WARN] No Global Weights from server.")
            continue

        model.set_weights(global_weights)

        global_metrics = model.evaluate(X_test, y_test)
        print(f"  [EVAL-GLOBAL] acc={global_metrics['test_accuracy']:.4f} | "
              f"f1={global_metrics['test_f1']:.4f} | "
              f"prec={global_metrics['test_precision']:.4f} | "
              f"rec={global_metrics['test_recall']:.4f}")

        history.append({
            "round": round,
            "local_metrics": local_metrics,
            "global_metrics": global_metrics,
            "n_samples": n_samples,
        })

        if global_metrics["test_accuracy"] > best_accuracy:
            best_accuracy = global_metrics["test_accuracy"]
            best_weights = {k: v.copy() for k, v in model.get_weights().items()}
            print(f"  [*] Best accuracy diperbarui: {best_accuracy:.4f}")

        if early_stop.step(global_metrics["test_accuracy"]):
            if best_weights is not None:
                model.set_weights(best_weights)
            break

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

    data.save_results(client_context.client_id, history)
