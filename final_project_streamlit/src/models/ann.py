import sys

import numpy as np


def _print_epoch_progress(epoch, total, loss):
    bar_len = 28
    filled = int(bar_len * (epoch + 1) / total)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    sys.stdout.write(f"\r    Epoch {epoch + 1:>4}/{total} [{bar}] loss={loss:.4f}")
    sys.stdout.flush()


class ANNClassifierScratch:
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        lr=0.01,
        epochs=800,
        batch_size=128,
        l2_lambda=1e-3,
        dropout_p=0.3,
        pos_weight=1.0,
        neg_weight=1.0,
        seed=42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.dropout_p = dropout_p
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.W1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.history = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_grad(z):
        return (z > 0.0).astype(float)

    def _forward(self, X, training=True):
        z1 = X @ self.W1 + self.b1
        h1 = self._relu(z1)

        dropout_mask = None
        if training and self.dropout_p > 0:
            keep_p = 1.0 - self.dropout_p
            dropout_mask = (self.rng.random(h1.shape) < keep_p).astype(float)
            h1 = (h1 * dropout_mask) / keep_p

        z2 = h1 @ self.W2 + self.b2
        y_hat = self._sigmoid(z2).reshape(-1)
        return z1, h1, z2, y_hat, dropout_mask

    def fit(self, X, y, verbose=False, early_stopping_patience=None, min_delta=1e-5):
        """Fit with mini-batch SGD + optional early stopping on training loss plateau.

        Early stopping (proposal §5 Risk: Overfitting) halts training when the loss
        fails to improve by ``min_delta`` for ``early_stopping_patience`` epochs.
        """
        n = X.shape[0]
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.epochs):
            indices = self.rng.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                m = xb.shape[0]

                z1, h1, _, y_hat, mask = self._forward(xb, training=True)
                sample_weight = np.where(yb == 1, self.pos_weight, self.neg_weight).reshape(-1, 1)

                dlogits = ((y_hat - yb).reshape(-1, 1) * sample_weight) / m
                dW2 = h1.T @ dlogits + self.l2_lambda * self.W2
                db2 = np.sum(dlogits, axis=0)

                dh1 = dlogits @ self.W2.T
                if mask is not None:
                    dh1 *= mask / (1.0 - self.dropout_p)
                dz1 = dh1 * self._relu_grad(z1)
                dW1 = xb.T @ dz1 + self.l2_lambda * self.W1
                db1 = np.sum(dz1, axis=0)

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            probs = self.predict_proba(X)
            full_weights = np.where(y == 1, self.pos_weight, self.neg_weight)
            loss = (
                -np.mean(
                    full_weights * (y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12))
                )
                + 0.5 * self.l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
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
        _, _, _, y_hat, _ = self._forward(X, training=False)
        return y_hat

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

