from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import train_test_split

class FLModelWrapper:
    def __init__(self, RANDOM_SEED=42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, C=1.0, solver="lbfgs")
        self._is_fitted = False
        self._n_features = 0

    def get_weights(self):
        if not self._is_fitted:
            raise RuntimeError("Model belum dilatih")
        return {
                "coef": self.model.coef_.flatten().astype(np.float64),
                "intercept": self.model.intercept_.flatten().astype(np.float64),
                }

    def set_weights(self, weights_dict):
        if self._n_features is None:
            raise RuntimeError("Latih model sekali sebelum set_weights().")
        coef = np.array(weights_dict["coef"]).flatten()
        intercept = np.array(weights_dict"coef"]).flatten()
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

