import sys

import numpy as np


def _print_epoch_progress(epoch, total, loss):
    bar_len = 28
    filled = int(bar_len * (epoch + 1) / total)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    sys.stdout.write(f"\r    Epoch {epoch + 1:>4}/{total} [{bar}] loss={loss:.4f}")
    sys.stdout.flush()


class LogisticRegressionScratch:
    def __init__(self, lr=0.05, epochs=1200, l2_lambda=1e-3, pos_weight=1.0, neg_weight=1.0):
        self.lr = lr
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.w = None
        self.b = 0.0
        self.history = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, verbose=False, early_stopping_patience=None, min_delta=1e-5):
        """Fit with optional early stopping on training loss plateau.

        Early stopping (proposal §5 Risk: Overfitting) halts training when the loss
        fails to improve by ``min_delta`` for ``early_stopping_patience`` epochs.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            logits = X @ self.w + self.b
            y_hat = self._sigmoid(logits)

            sample_weight = np.where(y == 1, self.pos_weight, self.neg_weight)
            error = (y_hat - y) * sample_weight
            dw = (X.T @ error) / n_samples + self.l2_lambda * self.w
            db = np.sum(error) / n_samples

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = (
                -np.mean(
                    sample_weight
                    * (y * np.log(y_hat + 1e-12) + (1 - y) * np.log(1 - y_hat + 1e-12))
                )
                + 0.5 * self.l2_lambda * np.sum(self.w ** 2)
            )
            self.history.append(float(loss))
            if verbose:
                _print_epoch_progress(epoch, self.epochs, loss)

            if early_stopping_patience is not None:
                if best_loss - loss > min_delta:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        sys.stdout.write(f"  (early stop at epoch {epoch + 1})\n")
                        sys.stdout.flush()
                    return
        if verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def predict_proba(self, X):
        logits = X @ self.w + self.b
        return self._sigmoid(logits)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

